"""Single-GPU inverse-frequency reweighting for the contour-conditioned model.

The group-robustness baseline that sits beside Group DRO: instead of learning
group weights online, each sample's loss is scaled by a STATIC weight inversely
proportional to its disease group's size, so every group contributes equally.

With K groups, n_g samples in group g and N = sum_g n_g, the weight is the
importance ratio that maps the natural group distribution onto the uniform one
(natural p_g = n_g/N, target u_g = 1/K):

    w_g = u_g / p_g = (1/K) / (n_g/N) = N / (K * n_g)

which makes the reweighting exact:

    E_natural[w_g * l] = sum_g (n_g/N) * N/(K*n_g) * L_g = (1/K) * sum_g L_g

i.e. the unweighted average of the per-group mean losses. Batches are drawn from
the NATURAL training distribution (a plain shuffled loader), and the weights are
normalized within each batch:

    loss = sum_i w_g(i) * l_i / sum_i w_g(i)

The batch normalization pins the effective learning rate exactly; it also means
any constant factor in w cancels, so this is identical to the "INS" convention
w_g = (1/n_g)/sum_j(1/n_j)*K that differs from the above by a constant.

Contrast with Group DRO (trainer_group_dro.py), which re-estimates q every step
from observed losses and draws a single-group batch per step. Note also that the
equivalent RESAMPLING form of this technique (Sagawa et al. 2020 implement it
that way, with a WeightedRandomSampler) is deliberately NOT used here: it would
revisit each minority video ~K*n_maj/n_min times as often as a majority one,
which on a few-hundred-clip dataset is direct memorization pressure on exactly
the group we are trying to improve. Loss weighting visits every sample at its
natural rate and only scales the gradient.
"""

import copy
import inspect
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils import data

# AMP: prefer the new torch.amp API (torch >= 2.3), fall back to torch.cuda.amp.
try:
    from torch.amp import autocast as _autocast, GradScaler as _GradScaler

    def autocast(enabled=True):
        return _autocast("cuda", enabled=enabled)

    def GradScaler(enabled=True):
        return _GradScaler("cuda", enabled=enabled)
except ImportError:  # torch < 2.3
    from torch.cuda.amp import autocast, GradScaler

from video_diffusion_pytorch.trainer_group_dro import GroupedContourDataset
from video_diffusion_pytorch.video_diffusion_cross_attention_with_motion_after_only_contour import (
    EMA,
    Trainer,
    cycle,
    exists,
    noop,
    normalize_cond_img,
    random_pick_condition_videos,
)
from video_diffusion_pytorch.frame_validity import load_valid_frames_map

try:
    import wandb
except ImportError:  # wandb is optional; training works without it
    wandb = None


def inverse_frequency_weights(group_counts):
    """Static per-group weights w_g = N / (K * n_g).

    ``group_counts`` is a sequence of per-group sample counts. Returns a float32
    tensor of the same length whose sample-weighted mean is exactly 1 (so the
    weighted loss keeps the same scale as an unweighted one) and for which every
    group's TOTAL weight n_g * w_g is equal (so no group dominates).
    """
    counts = torch.as_tensor(list(group_counts), dtype=torch.float32)
    if (counts <= 0).any():
        raise ValueError(f"every active group needs at least one sample, got {counts.tolist()}")
    return counts.sum() / (len(counts) * counts)


class InverseFrequencyTrainer(Trainer):
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
        # config's reweight.allowed_disease_groups. Identical semantics to the
        # Group-DRO config field; see configs/README.md / disease_to_group.
        allowed_disease_groups=None,
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
            raise ValueError("InverseFrequencyTrainer currently supports gradient_accumulate_every=1")
        if metadata_json_path is None and not inference_only:
            raise ValueError("metadata_json_path is required for InverseFrequencyTrainer")

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

        self.weight_decay = float(weight_decay)
        self.num_workers = int(num_workers)
        # Disease groups come strictly from the config
        # (reweight.allowed_disease_groups); there is no hardcoded fallback.
        if not allowed_disease_groups and not inference_only:
            raise ValueError(
                "allowed_disease_groups is required (set reweight.allowed_disease_groups "
                "in the config)"
            )
        self.allowed_disease_groups = (
            tuple(allowed_disease_groups) if allowed_disease_groups else None
        )

        image_size = diffusion_model.image_size
        channels = diffusion_model.channels
        num_frames = diffusion_model.num_frames

        # Optional per-sample valid-frame masking. Empty map (no CSV) => masking off.
        self.valid_frames_map = load_valid_frames_map(
            valid_frames_csv, valid_frames_filename_col, valid_frames_count_col
        )
        self.use_frame_validity = bool(self.valid_frames_map)

        if not inference_only:
            # Same dataset as Group-DRO: its 5-tuple __getitem__ already hands back
            # a per-sample group_idx, which is exactly what indexes the weights.
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
            self.group_weights = inverse_frequency_weights(self.ds.group_counts).to(self.device)
            print(
                "inverse-frequency weights w_g = N/(K*n_g): "
                + ", ".join(
                    f"{name}={w:.4f}"
                    for name, w in zip(self.group_names, self.group_weights.cpu().tolist())
                )
            )

            # True cumulative mean-since-start per group: running sum + count of
            # the per-sample losses seen for each group, so cumulative_mean =
            # sum / count. Exact here (not a per-step approximation) because the
            # loss is computed per sample.
            self._group_loss_sum = {name: 0.0 for name in self.group_names}
            self._group_loss_count = {name: 0 for name in self.group_names}

            # ONE loader over the NATURAL training distribution — no sampler. The
            # reweighting lives entirely in the loss, so batches must keep their
            # natural group mixture; a balanced sampler here would double-correct.
            loader = data.DataLoader(
                self.ds,
                batch_size=self.batch_size,
                shuffle=True,
                pin_memory=True,
                num_workers=self.num_workers,
                drop_last=True,
                persistent_workers=self.num_workers > 0,
            )
            self.dl = cycle(loader)

            self.opt = Adam(diffusion_model.parameters(), lr=train_lr, weight_decay=self.weight_decay)
        else:
            self.group_names = []
            self.group_counts = torch.empty(0, dtype=torch.float32, device=self.device)
            self.group_weights = torch.empty(0, dtype=torch.float32, device=self.device)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)
        self.max_grad_norm = max_grad_norm

        self.experiment_name = experiment_name

        # Optional Weights & Biases logging. Off by default; enable via the config.
        self.use_wandb = use_wandb and not inference_only
        if self.use_wandb:
            assert wandb is not None, "use_wandb=True but wandb is not installed (pip install wandb)"
            wandb.init(
                project=wandb_project,
                name=experiment_name,
                config=wandb_config,
                resume="allow",
            )
            # The weights are static, so log them once rather than every step.
            wandb.log(
                {f"w_{name}": w for name, w in zip(self.group_names, self.group_weights.cpu().tolist())},
                step=0,
            )

        checkpoints_folder = f"./{self.experiment_name}/checkpoints"
        self.checkpoints_folder = Path(checkpoints_folder)
        self.checkpoints_folder.mkdir(exist_ok=True, parents=True)

        self.reset_parameters()

    def save(self, milestone):
        ckpt = {
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
            "scaler": self.scaler.state_dict(),
            "reweight_group_names": list(self.group_names),
            "reweight_group_counts": self.group_counts.detach().cpu(),
            "reweight_group_weights": self.group_weights.detach().cpu(),
            "reweight_weight_decay": self.weight_decay,
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

        # There is no learned reweighting state to restore — the weights are a pure
        # function of the current dataset's group counts, recomputed in __init__.
        # But that means a dataset change silently changes the objective, so check.
        checkpoint_group_names = list(ckpt.get("reweight_group_names", []))
        if checkpoint_group_names and checkpoint_group_names != self.group_names:
            raise ValueError(
                "checkpoint group names do not match the current dataset: "
                f"checkpoint={checkpoint_group_names}, current={self.group_names}"
            )

        ckpt_counts = ckpt.get("reweight_group_counts")
        if ckpt_counts is not None:
            ckpt_counts = torch.as_tensor(ckpt_counts, dtype=torch.float32)
            if not torch.equal(ckpt_counts.cpu(), self.group_counts.detach().cpu()):
                print(
                    "WARNING: group counts changed since the checkpoint was saved "
                    f"(checkpoint={ckpt_counts.tolist()}, current={self.group_counts.tolist()}). "
                    "The inverse-frequency weights have been recomputed from the CURRENT "
                    "counts, so this run optimizes a different objective than the one that "
                    "produced the checkpoint."
                )

    def weighted_loss(self, per_sample_loss, group_idx):
        """Batch-normalized inverse-frequency weighted mean of per-sample losses.

        Returns ``(loss, weights)``. Dividing by the batch's own weight sum keeps
        the result on the same scale as an unweighted mean regardless of how the
        batch's group mixture came out, which pins the effective learning rate.
        """
        weights = self.group_weights[group_idx]
        loss = (weights * per_sample_loss).sum() / weights.sum().clamp(min=1e-12)
        return loss, weights

    def accumulate_group_losses(self, per_sample_loss, group_idx):
        """Fold this batch's per-sample losses into the per-group running means."""
        detached = per_sample_loss.detach().float()
        per_group_now = {}
        for gi, name in enumerate(self.group_names):
            mask = group_idx == gi
            count = int(mask.sum())
            if count == 0:
                continue
            total = detached[mask].sum().item()
            self._group_loss_sum[name] += total
            self._group_loss_count[name] += count
            per_group_now[name] = total / count
        return per_group_now

    def train(self, prob_focus_present=0.0, focus_present_mask=None, log_fn=noop):
        assert callable(log_fn)

        while self.step < self.train_num_steps:
            # Natural-distribution batch: a mixture of groups, in their true ratios.
            input_video, contour_cond_video, group_idx, _, valid_frames = next(self.dl)
            input_video = input_video.to(self.device, non_blocking=True)
            contour_cond_video = contour_cond_video.to(self.device, non_blocking=True)
            group_idx = group_idx.to(self.device, non_blocking=True)
            valid_frames = valid_frames.to(self.device, non_blocking=True) if self.use_frame_validity else None

            with autocast(enabled=self.amp):
                per_sample_loss, x0, x_start, per_sample_mse = self.model(
                    input_video,
                    cond=[contour_cond_video],
                    valid_frames=valid_frames,
                    per_sample=True,
                    prob_focus_present=prob_focus_present,
                    focus_present_mask=focus_present_mask,
                )

            loss, _ = self.weighted_loss(per_sample_loss, group_idx)
            self.scaler.scale(loss).backward()

            # Unweighted batch mean, for an apples-to-apples curve against runs
            # that do not reweight. disp_recon_mse stays unweighted too — it is a
            # diagnostic of reconstruction quality on the batch as drawn, not part
            # of the objective.
            unweighted_loss = per_sample_loss.detach().float().mean()
            disp_recon_mse = per_sample_mse.detach().float().mean()
            per_group_now = self.accumulate_group_losses(per_sample_loss, group_idx)

            log = {
                "loss": loss.item(),
                "unweighted_loss": unweighted_loss.item(),
                "disp_recon_mse": disp_recon_mse.item(),
            }

            group_parts = ", ".join(f"{name}={value:.4f}" for name, value in per_group_now.items())
            print(
                f"{self.step}: loss={loss.item():.6f} unweighted={unweighted_loss.item():.6f} "
                f"disp_recon_mse={disp_recon_mse.item():.6f} | {group_parts}"
            )

            if exists(self.max_grad_norm):
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

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

            if self.use_wandb:
                wandb_log = dict(log)
                wandb_log["step"] = self.step
                # Raw loss for each group PRESENT in this batch (sparse per group),
                # plus the true cumulative mean since start for every group seen so
                # far (dense, carried forward). Same key names as the Group-DRO
                # trainer so the two runs overlay directly.
                for name, value in per_group_now.items():
                    wandb_log[f"loss_{name}"] = value
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
