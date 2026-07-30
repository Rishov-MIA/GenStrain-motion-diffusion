import copy
import inspect
import json
import math
import os
import random
import re
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.cuda.amp import autocast, GradScaler
from torch.optim import Adam
from torch.utils import data

from video_diffusion_pytorch.utils import generate_displacement_quiver_gifs_with_gt
from video_diffusion_pytorch.video_diffusion_cross_attention_with_motion_after_only_contour import (
    EMA,
    Trainer,
    cycle,
    exists,
    noop,
    normalize_cond_img,
    random_pick_condition_videos,
)
from video_diffusion_pytorch.frame_validity import load_valid_frames_map, resolve_valid_frames

try:
    import wandb
except ImportError:  # wandb is optional; training works without it
    wandb = None


def normalize_allowed_groups(allowed_groups):
    """Normalize config disease-group entries to ``(name, exclude_terms)`` pairs.

    Each entry may be either a plain string (the group name) or a dict
    ``{"name": ..., "exclude": [...]}``. ``exclude`` is an optional list of terms
    that VETO the match: the group matches a disease string only if its name is a
    substring AND none of its exclude terms are. This lets e.g. "Healthy" absorb
    "Healthy Cardiotoxicity" while dropping "Healthy Pediatric":
        {"name": "Healthy", "exclude": ["Pediatric"]}
    Returns a list of (name: str, exclude_terms: tuple[str, ...]).
    """
    normalized = []
    for entry in allowed_groups:
        if isinstance(entry, str):
            normalized.append((entry, ()))
        elif isinstance(entry, dict):
            name = entry["name"]
            excludes = tuple(entry.get("exclude", ()))
            normalized.append((name, excludes))
        else:
            raise TypeError(
                f"allowed_disease_groups entries must be str or {{name, exclude}} "
                f"dicts, got {type(entry).__name__}: {entry!r}"
            )
    return normalized


def disease_to_group(disease, allowed_groups):
    """Map a raw metadata disease string to a group name or None.

    ``allowed_groups`` is the list of entries from the config's
    dro.allowed_disease_groups (see configs/README.md), each a plain group-name
    string or a ``{"name", "exclude"}`` dict. Matching is case-insensitive
    substring matching in list order (first match wins), so ordering encodes
    precedence for multi-label strings — a more specific name (e.g.
    "Healthy Pediatric") must precede a broader one (e.g. "Healthy"). A group's
    optional ``exclude`` terms veto the match when present in the disease string.
    """
    if not isinstance(disease, str):
        return None

    disease_lower = disease.lower()
    for group_name, exclude_terms in normalize_allowed_groups(allowed_groups):
        if group_name.lower() not in disease_lower:
            continue
        if any(term.lower() in disease_lower for term in exclude_terms):
            continue  # vetoed by an exclude term (e.g. "Pediatric" for Healthy)
        return group_name
    return None


class GroupedContourDataset(data.Dataset):
    def __init__(
        self,
        input_video_folder,
        image_size,
        contour_condition_video_dir,
        metadata_json_path,
        channels=2,
        num_frames=16,
        exts=("npy",),
        allowed_groups=None,
        valid_frames_map=None,       # {filename: valid_frame_count}; empty => all frames valid
        valid_frames_missing="full",  # behavior when a file is absent from the map
    ):
        super().__init__()
        if not allowed_groups:
            raise ValueError(
                "GroupedContourDataset requires a non-empty allowed_groups "
                "(set dro.allowed_disease_groups in the config)"
            )
        self.input_video_folder = Path(input_video_folder)
        self.contour_condition_video_dir = Path(contour_condition_video_dir)
        self.metadata_json_path = Path(metadata_json_path)
        self.image_size = image_size
        self.channels = channels
        self.num_frames = num_frames
        self.valid_frames_map = valid_frames_map or {}
        self.valid_frames_missing = valid_frames_missing
        # Raw entries (str or {name, exclude}) drive matching; the derived names,
        # in list order, drive membership checks and the group ordering.
        self.allowed_groups = tuple(allowed_groups)
        self.allowed_group_names = tuple(
            name for name, _ in normalize_allowed_groups(allowed_groups)
        )

        if not self.metadata_json_path.exists():
            raise FileNotFoundError(f"metadata JSON not found: {self.metadata_json_path}")

        with self.metadata_json_path.open("r") as f:
            metadata = json.load(f)

        input_paths = [p for ext in exts for p in self.input_video_folder.glob(f"**/*.{ext}")]
        contour_paths = [p for ext in exts for p in self.contour_condition_video_dir.glob(f"**/*.{ext}")]

        input_by_name = {p.name: p for p in input_paths}
        contour_by_name = {p.name: p for p in contour_paths}
        input_names = set(input_by_name)
        contour_names = set(contour_by_name)
        missing_contour = sorted(input_names - contour_names)
        missing_input = sorted(contour_names - input_names)
        assert not missing_contour and not missing_input, (
            "Target and contour folders must contain the same filenames. "
            f"Missing contour files for {len(missing_contour)} targets; "
            f"missing target files for {len(missing_input)} contours."
        )
        paired_names = sorted(input_names)

        skipped = Counter()
        raw_samples = []

        for filename in paired_names:
            info = self._lookup_metadata(metadata, filename)
            if info is None:
                skipped["missing_metadata"] += 1
                continue

            group_name = disease_to_group(info.get("disease"), self.allowed_groups)
            if group_name is None or group_name not in self.allowed_group_names:
                skipped["unsupported_disease"] += 1
                continue

            raw_samples.append(
                {
                    "input_path": input_by_name[filename],
                    "contour_path": contour_by_name[filename],
                    "filename": filename,
                    "group_name": group_name,
                }
            )

        self.group_names = [group for group in self.allowed_group_names if any(s["group_name"] == group for s in raw_samples)]
        self.group_to_idx = {group_name: idx for idx, group_name in enumerate(self.group_names)}
        self.samples = []
        self.group_to_indices = defaultdict(list)

        for sample in raw_samples:
            group_idx = self.group_to_idx[sample["group_name"]]
            sample["group_idx"] = group_idx
            self.group_to_indices[group_idx].append(len(self.samples))
            self.samples.append(sample)

        self.group_counts = [len(self.group_to_indices[idx]) for idx in range(len(self.group_names))]
        self.skipped = dict(skipped)

    @staticmethod
    def _lookup_metadata(metadata, filename):
        path = Path(filename)
        # Candidate stems, most specific first. Augmented copies are named
        # "<original>_scale_<f>" (e.g. "10AF05640_mid_20160818_NS_scale_0p77",
        # where 0p77 == a 0.77 scale) and share the source sample's disease, so
        # strip a trailing "_scale_<f>" to fall back to the original file's
        # metadata key. Likewise strip a trailing "_cycleN" as before.
        stems = []
        for stem in (path.stem, re.sub(r"_scale_\d+(?:p\d+)?$", "", path.stem)):
            for candidate in (stem, re.sub(r"_cycle[A-Za-z0-9]+$", "", stem)):
                if candidate not in stems:
                    stems.append(candidate)

        keys = [filename]
        for stem in stems:
            keys.extend((f"{stem}.mat", stem))

        for key in keys:
            if key in metadata:
                return metadata[key]
        return None

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        # No in-process dict cache: this dataset is meant to be consumed through
        # DataLoaders with num_workers > 0, where each forked worker would keep
        # its own copy of such a cache (unshared, unbounded memory growth). The
        # DataLoader's prefetching gives us the overlap we care about instead.
        input_video_tensor = torch.from_numpy(np.load(sample["input_path"])).float()
        contour_condition_video_tensor = torch.from_numpy(np.load(sample["contour_path"])).float()

        valid_frames = resolve_valid_frames(
            self.valid_frames_map, sample["filename"], self.num_frames, self.valid_frames_missing
        )

        return (
            input_video_tensor.squeeze(0),
            contour_condition_video_tensor,
            sample["group_idx"],
            sample["filename"],
            valid_frames,
        )

    def indices_for_group(self, group_idx):
        """Dataset indices belonging to a single group (for a per-group sampler)."""
        return list(self.group_to_indices[group_idx])


class GroupDROTrainer(Trainer):
    def __init__(
        self,
        diffusion_model,
        input_video_folder,
        *,
        contour_condition_video_dir=None,
        sampling_contour_condition_video_dir=None,
        metadata_json_path=None,
        # Disease groups to keep, in string-match PRIORITY order (first match
        # wins). REQUIRED for training (no hardcoded fallback) — comes from the
        # config's dro.allowed_disease_groups. See configs/README.md / disease_to_group.
        allowed_disease_groups=None,
        # DRO group-weight learning rate (eta_q in the algorithm). The per-step
        # update multiplies a group's raw weight by exp(eta_q * S_g), with
        # S_g ~= L_g (a MEAN denoising loss, empirically ~0.4-1.1 here). So the
        # right eta_q keeps exp(eta_q * L_g) close to 1 — i.e. eta_q on the order
        # of 0.01-0.05, NOT 1.0. With eta_q=1.0 each sample of a group ~doubles
        # its raw weight (exp(0.6)~=1.8), so whichever group is sampled most in
        # the first few steps runs away and q collapses onto it regardless of
        # which group is actually hardest (observed: q -> ~1.0 on Healthy by
        # step ~150). Tuning target: watch the printed per-group q — it should
        # stay spread and drift toward the group with PERSISTENTLY higher loss,
        # never pin near 1.0. If q is dead-flat at 1/K, raise eta_q; if any group
        # exceeds ~0.7, lower it.
        dro_eta_q=0.02,
        dro_adjustment_c=0.0,  # generalization-adjustment constant C (also on the mean-loss scale)
        # Freeze q at its uniform init (1/K) and backprop the raw, UNWEIGHTED loss.
        # This disables the DRO reweighting entirely. NOTE: it does NOT reproduce
        # base GenStrain, because group selection is still uniform-over-groups
        # (group-balanced sampling), not i.i.d. from the natural data distribution.
        dro_freeze_q=False,
        weight_decay=0.0,
        ema_decay=0.995,
        num_frames=16,
        train_batch_size=32,
        train_lr=1e-4,
        train_num_steps=100000,
        gradient_accumulate_every=1,
        amp=False,
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every=1000,
        max_grad_norm=None,
        experiment_name="test-exp",
        num_workers=4,
        inference_only=False,
        use_wandb=False,
        wandb_project="genstrain-motion-diffusion",
        wandb_config=None,
        valid_frames_csv=None,
        valid_frames_filename_col="dense_filename",
        valid_frames_count_col="dense_valid_frames",
        valid_frames_missing="full",
    ):
        object.__init__(self)
        if gradient_accumulate_every != 1:
            raise ValueError("GroupDROTrainer currently supports gradient_accumulate_every=1")
        if metadata_json_path is None and not inference_only:
            raise ValueError("metadata_json_path is required for GroupDROTrainer")

        self.model = diffusion_model
        self.device = next(diffusion_model.parameters()).device
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.sampling_contour_condition_video_dir = sampling_contour_condition_video_dir

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        self.dro_eta_q = float(dro_eta_q)
        self.dro_adjustment_c = float(dro_adjustment_c)
        self.dro_freeze_q = bool(dro_freeze_q)
        self.weight_decay = float(weight_decay)
        self.num_workers = int(num_workers)
        # Disease groups come strictly from the config (dro.allowed_disease_groups);
        # there is no hardcoded fallback. Required whenever we build the dataset.
        if not allowed_disease_groups and not inference_only:
            raise ValueError(
                "allowed_disease_groups is required (set dro.allowed_disease_groups "
                "in the config)"
            )
        self.allowed_disease_groups = (
            tuple(allowed_disease_groups) if allowed_disease_groups else None
        )

        image_size = diffusion_model.image_size
        channels = diffusion_model.channels
        num_frames = diffusion_model.num_frames

        # Optional per-sample valid-frame masking. Empty map (no CSV) => masking
        # off, i.e. the loss is byte-for-byte the original all-frames loss.
        self.valid_frames_map = load_valid_frames_map(
            valid_frames_csv, valid_frames_filename_col, valid_frames_count_col
        )
        self.use_frame_validity = bool(self.valid_frames_map)

        if not inference_only:
            self.ds = GroupedContourDataset(
                input_video_folder,
                image_size,
                contour_condition_video_dir,
                metadata_json_path,
                channels=channels,
                num_frames=num_frames,
                allowed_groups=self.allowed_disease_groups,
                valid_frames_map=self.valid_frames_map,
                valid_frames_missing=valid_frames_missing,
            )

            if self.use_frame_validity:
                print(f"valid-frame loss masking ON ({len(self.valid_frames_map)} CSV entries)")

            print(f"found {len(self.ds)} supported grouped videos as .npy files at {input_video_folder}")
            print(f"group counts: {dict(zip(self.ds.group_names, self.ds.group_counts))}")
            if self.ds.skipped:
                print(f"skipped files: {self.ds.skipped}")

            assert len(self.ds) > 0, "need at least 1 supported grouped video to start training"
            assert len(self.ds.group_names) > 0, "need at least 1 active group to start training"

            self.group_names = list(self.ds.group_names)
            self.group_counts = torch.tensor(self.ds.group_counts, dtype=torch.float32, device=self.device)
            # True cumulative mean-since-start per group: running sum + count of the
            # losses seen for each group, so cumulative_mean = sum / count. Each
            # group is only sampled on some steps; the cumulative mean is carried
            # forward on the steps it isn't sampled.
            self._group_loss_sum = {name: 0.0 for name in self.group_names}
            self._group_loss_count = {name: 0 for name in self.group_names}
            self.dro_log_q = torch.full(
                (len(self.group_names),),
                -math.log(len(self.group_names)),
                dtype=torch.float32,
                device=self.device,
            )

            # One infinite DataLoader per group (algorithm line 4: sample B examples
            # from a single group per step). We sample WITH REPLACEMENT via a
            # RandomSampler(replacement=True), matching the pseudocode's i.i.d. draw
            # and — crucially — always yielding exactly B samples even when a group
            # has fewer than B members (drop_last=True is then a no-op since the
            # sampler never runs out). This keeps L_g / S_g / the q-update on a
            # consistent B for every group, and hands loading to background workers
            # (pin_memory + prefetch) instead of blocking the train loop on disk.
            # One loader per group stays alive for the whole run, so cap workers per
            # group by its size to avoid spawning idle processes for tiny groups.
            self.group_loaders = []
            for group_idx in range(len(self.group_names)):
                group_indices = self.ds.indices_for_group(group_idx)
                subset = data.Subset(self.ds, group_indices)
                workers = min(self.num_workers, len(group_indices))
                # num_samples is just an epoch length before the sampler reshuffles
                # its RNG draw; cycle() restarts it forever. Make it large so worker
                # epochs are long and restart overhead is negligible.
                sampler = data.RandomSampler(
                    subset,
                    replacement=True,
                    num_samples=max(len(group_indices), self.batch_size) * 1000,
                )
                loader = data.DataLoader(
                    subset,
                    batch_size=self.batch_size,
                    sampler=sampler,
                    pin_memory=True,
                    num_workers=workers,
                    drop_last=True,
                    persistent_workers=workers > 0,
                )
                self.group_loaders.append(cycle(loader))

            self.opt = Adam(diffusion_model.parameters(), lr=train_lr, weight_decay=self.weight_decay)
        else:
            self.group_names = []
            self.group_counts = torch.empty(0, dtype=torch.float32, device=self.device)
            self.dro_log_q = torch.empty(0, dtype=torch.float32, device=self.device)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)
        self.max_grad_norm = max_grad_norm

        self.experiment_name = experiment_name

        # Optional Weights & Biases logging. Off by default so existing runs are
        # unaffected; enable via the config (see configs/README.md).
        self.use_wandb = use_wandb and not inference_only
        if self.use_wandb:
            assert wandb is not None, "use_wandb=True but wandb is not installed (pip install wandb)"
            wandb.init(
                project=wandb_project,
                name=experiment_name,
                config=wandb_config,
                resume="allow",
            )

        checkpoints_folder = f"./{self.experiment_name}/checkpoints"
        self.checkpoints_folder = Path(checkpoints_folder)
        self.checkpoints_folder.mkdir(exist_ok=True, parents=True)

        self.reset_parameters()

    @property
    def dro_q(self):
        return self.dro_log_q.exp()

    def save(self, milestone):
        ckpt = {
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
            "scaler": self.scaler.state_dict(),
            "dro_log_q": self.dro_log_q.detach().cpu(),
            "dro_group_names": list(self.group_names),
            "dro_group_counts": self.group_counts.detach().cpu(),
            "dro_eta_q": self.dro_eta_q,
            "dro_adjustment_c": self.dro_adjustment_c,
            "dro_weight_decay": self.weight_decay,
        }
        torch.save(ckpt, str(self.checkpoints_folder / f"model-{milestone}.pt"))

    def load(self, milestone, **kwargs):
        if milestone == -1:
            all_milestones = [int(p.stem.split("-")[-1]) for p in Path(self.checkpoints_folder).glob("**/*.pt")]
            assert len(all_milestones) > 0, "need at least one milestone to load from (milestone == -1)"
            milestone = max(all_milestones)

        ckpt_path = str(self.checkpoints_folder / f"model-{milestone}.pt")
        if "weights_only" in inspect.signature(torch.load).parameters:
            ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        else:
            ckpt = torch.load(ckpt_path, map_location=self.device)

        print(f"loaded checkpoint: model-{milestone}.pt\n")

        self.step = ckpt["step"]
        self.model.load_state_dict(ckpt["model"], **kwargs)
        self.ema_model.load_state_dict(ckpt["ema"], **kwargs)
        self.scaler.load_state_dict(ckpt["scaler"])

        if "dro_log_q" not in ckpt:
            print("checkpoint has no DRO state; keeping uniform group weights")
            return

        checkpoint_group_names = list(ckpt.get("dro_group_names", []))
        if checkpoint_group_names != self.group_names:
            raise ValueError(
                "DRO checkpoint group names do not match the current dataset: "
                f"checkpoint={checkpoint_group_names}, current={self.group_names}"
            )

        # Group names match but counts may have drifted (e.g. data added between
        # runs). We keep the learned q from the checkpoint, but the adjustment
        # term C/sqrt(n_g) will use the CURRENT n_g — warn so this is not silent.
        ckpt_counts = ckpt.get("dro_group_counts")
        if ckpt_counts is not None:
            ckpt_counts = torch.as_tensor(ckpt_counts, dtype=torch.float32)
            if not torch.equal(ckpt_counts.cpu(), self.group_counts.detach().cpu()):
                print(
                    "WARNING: DRO group counts changed since the checkpoint was saved "
                    f"(checkpoint={ckpt_counts.tolist()}, current={self.group_counts.tolist()}). "
                    "Keeping learned q but using current counts for the C/sqrt(n_g) adjustment."
                )

        self.dro_log_q = ckpt["dro_log_q"].to(self.device, dtype=torch.float32)

    def sample_group_idx(self):
        return random.randrange(len(self.group_names))

    def update_dro_weight(self, group_idx, group_loss):
        with torch.no_grad():
            adjustment = self.dro_adjustment_c / self.group_counts[group_idx].sqrt()
            score = group_loss.detach().float() + adjustment
            if not self.dro_freeze_q:
                self.dro_log_q[group_idx] = self.dro_log_q[group_idx] + self.dro_eta_q * score
                self.dro_log_q = self.dro_log_q - torch.logsumexp(self.dro_log_q, dim=0)
            q_g = self.dro_q[group_idx].detach()
        return score.detach(), q_g

    def train(self, prob_focus_present=0.0, focus_present_mask=None, log_fn=noop):
        assert callable(log_fn)

        while self.step < self.train_num_steps:
            group_idx = self.sample_group_idx()
            group_name = self.group_names[group_idx]

            # Pull one B-sized batch from the chosen group's cycled loader.
            input_video, contour_cond_video, _, _, valid_frames = next(self.group_loaders[group_idx])
            input_video = input_video.to(self.device, non_blocking=True)
            contour_cond_video = contour_cond_video.to(self.device, non_blocking=True)
            valid_frames = valid_frames.to(self.device, non_blocking=True) if self.use_frame_validity else None

            with autocast(enabled=self.amp):
                loss, x0, x_start, disp_recon_mse = self.model(
                    input_video,
                    cond=[contour_cond_video],
                    valid_frames=valid_frames,
                    prob_focus_present=prob_focus_present,
                    focus_present_mask=focus_present_mask,
                )

            score, q_g = self.update_dro_weight(group_idx, loss)
            # When q is frozen, backprop the raw loss so the gradient is not scaled
            # by the constant 1/K (which would just shrink the effective LR). This
            # makes each step's optimizer update match base GenStrain; only the
            # (still group-balanced) sampling distribution differs.
            weighted_loss = loss if self.dro_freeze_q else q_g * loss
            self.scaler.scale(weighted_loss).backward()

            log = {
                "loss": loss.item(),
                "weighted_loss": weighted_loss.item(),
                "disp_recon_mse": disp_recon_mse.item(),
                "dro_score": score.item(),
                "dro_q": q_g.item(),
                "group": group_name,
            }

            q_parts = ", ".join(f"{name}={q:.4f}" for name, q in zip(self.group_names, self.dro_q.detach().cpu()))
            print(
                f"{self.step}: group={group_name} "
                f"loss={loss.item():.6f} weighted={weighted_loss.item():.6f} "
                f"disp_recon_mse={disp_recon_mse.item():.6f} q={q_g.item():.6f} | {q_parts}"
            )

            if exists(self.max_grad_norm):
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            # if (self.step % 300 == 0) and (self.step != 0):
            #     predicted_start_training_dir = f"./{self.experiment_name}/predicted_start_training_gifs"
            #     os.makedirs(predicted_start_training_dir, exist_ok=True)
            #     x0_np = x0.detach().cpu().numpy()
            #     x_start_np = x_start.detach().cpu().numpy()

            #     generate_displacement_quiver_gifs_with_gt(
            #         milestone="train",
            #         filenames=[f"train_visual_{self.step}_{group_name}"],
            #         pred_disp=np.expand_dims(x0_np[0:1], axis=0),
            #         gt_disp=np.expand_dims(x_start_np[0:1], axis=0),
            #         output_dir=predicted_start_training_dir,
            #     )

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                self.save(milestone)

                sample_contour_cond, sample_cond_filenames = random_pick_condition_videos(
                    num_samples=1,
                    contour_cond_video_dir=self.sampling_contour_condition_video_dir,
                )
                sample_contour_cond = normalize_cond_img(sample_contour_cond)
                sample_contour_cond = sample_contour_cond.to(self.device)

                self.sample_and_save(
                    milestone,
                    contour_cond_video=sample_contour_cond,
                    cond_filenames=sample_cond_filenames,
                    save_folder=f"./{self.experiment_name}/sampled_videos_infos",
                    sampler=self.preview_sampler,
                    ddim_steps=self.preview_ddim_steps,
                    ddim_eta=self.preview_ddim_eta,
                )

            # Update the running (EMA) loss for the group sampled this step. Done
            # unconditionally so the cumulative mean is correct even if wandb is off.
            step_loss = loss.item()
            self._group_loss_sum[group_name] += step_loss
            self._group_loss_count[group_name] += 1

            if self.use_wandb:
                # Log the scalar fields plus per-group q_<name> curves; skip the
                # string "group" field (wandb can't plot it).
                wandb_log = {k: v for k, v in log.items() if k != "group"}
                wandb_log["step"] = self.step
                for name, q in zip(self.group_names, self.dro_q.detach().cpu().tolist()):
                    wandb_log[f"q_{name}"] = q
                # Raw loss for the group sampled THIS step (sparse per group), plus
                # the true cumulative mean since start for EVERY group (dense,
                # carried forward every step).
                wandb_log[f"loss_{group_name}"] = step_loss
                for name in self.group_names:
                    count = self._group_loss_count[name]
                    if count > 0:
                        wandb_log[f"loss_{name}_cummean"] = self._group_loss_sum[name] / count
                wandb.log(wandb_log, step=self.step)

            log_fn(log)
            self.step += 1

        if self.use_wandb:
            wandb.finish()

        print("training completed")
