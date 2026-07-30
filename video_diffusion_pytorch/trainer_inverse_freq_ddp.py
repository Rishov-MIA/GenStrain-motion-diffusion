"""Multi-GPU (DDP) inverse-frequency reweighting for the contour-conditioned model.

Same objective as trainer_inverse_freq.InverseFrequencyTrainer — see that module
for the derivation of w_g = N / (K * n_g) and why loss weighting is used rather
than resampling. This file only adds the distributed plumbing.

Launch with torchrun, one process per GPU:

    torchrun --standalone --nproc_per_node=4 train_inverse_freq_ddp.py

Two DDP-specific points:

1. Unlike the Group-DRO DDP trainer (which draws a single-group batch per step
   and so needs per-group loaders), this trainer samples the NATURAL data
   distribution, so it uses a plain DistributedSampler — each rank sees a
   disjoint shard of each epoch.

2. The batch-normalized weighted loss must be normalized over the GLOBAL batch,
   not per rank. Self-normalizing on each rank independently would let a rank
   whose batch happened to draw many minority samples (large sum of weights)
   have its gradient scaled differently from its peers, which is not the
   objective we want. So the denominator is all-reduced, and the per-rank loss
   is multiplied by world_size to cancel DDP's 1/W gradient averaging:

       loss_r = W * sum_{i in rank r} w_i*l_i / sum_{all ranks} w_i

   DDP then forms (1/W) * sum_r grad(loss_r) = grad[ sum_all w*l / sum_all w ],
   exactly the globally self-normalized weighted mean.
"""

import copy
import inspect
from pathlib import Path

import torch
from torch import nn
from torch.optim import Adam
from torch.utils import data
from torch.nn.parallel import DistributedDataParallel as DDP

# AMP: prefer the new torch.amp API (torch >= 2.3), fall back to torch.cuda.amp.
try:
    from torch.amp import autocast as _autocast, GradScaler as _GradScaler

    def autocast(enabled=True):
        return _autocast("cuda", enabled=enabled)

    def GradScaler(enabled=True):
        return _GradScaler("cuda", enabled=enabled)
except ImportError:  # torch < 2.3
    from torch.cuda.amp import autocast, GradScaler

from video_diffusion_pytorch.video_diffusion_cross_attention_with_motion_after_only_contour import (
    EMA,
    cycle,
    exists,
    noop,
    normalize_cond_img,
    random_pick_condition_videos,
)
from video_diffusion_pytorch.trainer_group_dro import GroupedContourDataset
from video_diffusion_pytorch.trainer_inverse_freq import (
    InverseFrequencyTrainer,
    inverse_frequency_weights,
)
from video_diffusion_pytorch.trainer_ddp import (
    cleanup_distributed,
    get_rank,
    get_world_size,
    is_dist,
    is_main_process,
    setup_distributed,
)
from video_diffusion_pytorch.frame_validity import load_valid_frames_map

try:
    import wandb
except ImportError:  # wandb is optional; training works without it
    wandb = None

__all__ = [
    "InverseFrequencyDDPTrainer",
    "setup_distributed",
    "cleanup_distributed",
    "is_main_process",
    "get_rank",
    "get_world_size",
    "is_dist",
]


class InverseFrequencyDDPTrainer(InverseFrequencyTrainer):
    def __init__(self, diffusion_model, input_video_folder, *, local_rank=0, **kwargs):
        # We deliberately do NOT call super().__init__ (it builds a non-distributed
        # loader and never wraps the model). We replicate the relevant setup here
        # in a DDP-aware way, mirroring trainer_group_dro_ddp.GroupDRODDPTrainer.
        object.__init__(self)

        gradient_accumulate_every = kwargs.get("gradient_accumulate_every", 1)
        if gradient_accumulate_every != 1:
            raise ValueError("InverseFrequencyDDPTrainer currently supports gradient_accumulate_every=1")

        inference_only = kwargs.get("inference_only", False)
        metadata_json_path = kwargs.get("metadata_json_path")
        if metadata_json_path is None and not inference_only:
            raise ValueError("metadata_json_path is required for InverseFrequencyDDPTrainer")

        self.local_rank = local_rank
        self.device = torch.device("cuda", local_rank)
        self.world_size = get_world_size()

        ema_decay = kwargs.get("ema_decay", 0.995)
        self.model = diffusion_model  # already on the correct device by caller
        self.ema = EMA(ema_decay)
        # EMA is maintained from the unwrapped model. Every rank keeps its own copy
        # (identical, since post-allreduce weights match); only rank 0 saves/samples it.
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = kwargs.get("update_ema_every", 10)
        self.sampling_contour_condition_video_dir = kwargs.get("sampling_contour_condition_video_dir")

        self.step_start_ema = kwargs.get("step_start_ema", 2000)
        self.save_and_sample_every = kwargs.get("save_and_sample_every", 1000)

        # PER-GPU batch size. Effective global batch = batch_size * world_size.
        self.batch_size = kwargs.get("train_batch_size", 32)
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = kwargs.get("train_num_steps", 100000)

        self.weight_decay = float(kwargs.get("weight_decay", 0.0))
        self.num_workers = int(kwargs.get("num_workers", 4))
        self.max_grad_norm = kwargs.get("max_grad_norm")
        # Disease groups come strictly from the config
        # (reweight.allowed_disease_groups); there is no hardcoded fallback.
        allowed_disease_groups = kwargs.get("allowed_disease_groups")
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

        contour_condition_video_dir = kwargs.get("contour_condition_video_dir")
        train_lr = kwargs.get("train_lr", 1e-4)

        # Optional per-sample valid-frame masking. Empty map (no CSV) => masking off.
        self.valid_frames_map = load_valid_frames_map(
            kwargs.get("valid_frames_csv"),
            kwargs.get("valid_frames_filename_col", "dense_filename"),
            kwargs.get("valid_frames_count_col", "dense_valid_frames"),
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
                valid_frames_missing=kwargs.get("valid_frames_missing", "full"),
            )

            if is_main_process() and self.use_frame_validity:
                print(f"valid-frame loss masking ON ({len(self.valid_frames_map)} CSV entries)")

            if is_main_process():
                print(f"found {len(self.ds)} supported grouped videos as .npy files at {input_video_folder}")
                print(f"group counts: {dict(zip(self.ds.group_names, self.ds.group_counts))}")
                if self.ds.skipped:
                    print(f"skipped files: {self.ds.skipped}")

            assert len(self.ds) > 0, "need at least 1 supported grouped video to start training"
            assert len(self.ds.group_names) > 0, "need at least 1 active group to start training"

            self.group_names = list(self.ds.group_names)
            self.group_counts = torch.tensor(self.ds.group_counts, dtype=torch.float32, device=self.device)
            # Static weights, identical on every rank (pure function of the counts).
            self.group_weights = inverse_frequency_weights(self.ds.group_counts).to(self.device)
            if is_main_process():
                print(
                    "inverse-frequency weights w_g = N/(K*n_g): "
                    + ", ".join(
                        f"{name}={w:.4f}"
                        for name, w in zip(self.group_names, self.group_weights.cpu().tolist())
                    )
                )

            # Cumulative per-group loss means, fed the GLOBAL (all-reduced) sums and
            # counts so every rank holds the same value; only rank 0 logs it.
            self._group_loss_sum = {name: 0.0 for name in self.group_names}
            self._group_loss_count = {name: 0 for name in self.group_names}

            # Natural-distribution sampling, sharded across ranks. Unlike the
            # Group-DRO DDP trainer there is no per-group loader: the reweighting
            # lives in the loss, so batches must keep their natural group mixture.
            self.sampler = data.DistributedSampler(
                self.ds,
                num_replicas=self.world_size,
                rank=get_rank(),
                shuffle=True,
                drop_last=True,
            )
            loader = data.DataLoader(
                self.ds,
                batch_size=self.batch_size,
                sampler=self.sampler,
                pin_memory=True,
                num_workers=self.num_workers,
                drop_last=True,
                persistent_workers=self.num_workers > 0,
            )
            self.dl = cycle(loader)

            # Wrap first, then build the optimizer on the wrapped params.
            self.ddp_model = DDP(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )
            self.opt = Adam(self.ddp_model.parameters(), lr=train_lr, weight_decay=self.weight_decay)
        else:
            self.group_names = []
            self.group_counts = torch.empty(0, dtype=torch.float32, device=self.device)
            self.group_weights = torch.empty(0, dtype=torch.float32, device=self.device)
            self.ddp_model = self.model

        self.step = 0

        self.amp = kwargs.get("amp", False)
        self.scaler = GradScaler(enabled=self.amp)

        self.experiment_name = kwargs.get("experiment_name", "test-exp")

        # Optional Weights & Biases logging, on rank 0 only (off by default).
        self.use_wandb = (
            kwargs.get("use_wandb", False) and not inference_only and is_main_process()
        )
        if self.use_wandb:
            assert wandb is not None, "use_wandb=True but wandb is not installed (pip install wandb)"
            wandb.init(
                project=kwargs.get("wandb_project", "genstrain-motion-diffusion"),
                name=self.experiment_name,
                config=kwargs.get("wandb_config"),
                resume="allow",
            )
            # The weights are static, so log them once rather than every step.
            wandb.log(
                {f"w_{name}": w for name, w in zip(self.group_names, self.group_weights.cpu().tolist())},
                step=0,
            )

        checkpoints_folder = f"./{self.experiment_name}/checkpoints"
        self.checkpoints_folder = Path(checkpoints_folder)
        if is_main_process():
            self.checkpoints_folder.mkdir(exist_ok=True, parents=True)

        self.reset_parameters()

    # ---- unwrap DDP for EMA / checkpointing ----

    @property
    def raw_model(self):
        return self.ddp_model.module if isinstance(self.ddp_model, DDP) else self.ddp_model

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.raw_model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.raw_model)

    # ---- checkpointing (rank 0 only writes) ----

    def save(self, milestone):
        if not is_main_process():
            return
        ckpt = {
            "step": self.step,
            "model": self.raw_model.state_dict(),
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
        map_location = {"cuda:0": f"cuda:{self.local_rank}"}
        if "weights_only" in inspect.signature(torch.load).parameters:
            ckpt = torch.load(ckpt_path, map_location=map_location, weights_only=False)
        else:
            ckpt = torch.load(ckpt_path, map_location=map_location)

        if is_main_process():
            print(f"loaded checkpoint: model-{milestone}.pt\n")

        self.step = ckpt["step"]
        self.raw_model.load_state_dict(ckpt["model"], **kwargs)
        self.ema_model.load_state_dict(ckpt["ema"], **kwargs)
        self.scaler.load_state_dict(ckpt["scaler"])

        # No learned reweighting state to restore — the weights are recomputed in
        # __init__ from the current counts. Guard against a silent objective change.
        checkpoint_group_names = list(ckpt.get("reweight_group_names", []))
        if checkpoint_group_names and checkpoint_group_names != self.group_names:
            raise ValueError(
                "checkpoint group names do not match the current dataset: "
                f"checkpoint={checkpoint_group_names}, current={self.group_names}"
            )

        ckpt_counts = ckpt.get("reweight_group_counts")
        if ckpt_counts is not None:
            ckpt_counts = torch.as_tensor(ckpt_counts, dtype=torch.float32)
            if not torch.equal(ckpt_counts.cpu(), self.group_counts.detach().cpu()) and is_main_process():
                print(
                    "WARNING: group counts changed since the checkpoint was saved "
                    f"(checkpoint={ckpt_counts.tolist()}, current={self.group_counts.tolist()}). "
                    "The inverse-frequency weights have been recomputed from the CURRENT "
                    "counts, so this run optimizes a different objective than the one that "
                    "produced the checkpoint."
                )

    # ---- distributed reductions ----

    def _all_reduce_sum(self, tensor):
        if is_dist():
            torch.distributed.all_reduce(tensor, op=torch.distributed.ReduceOp.SUM)
        return tensor

    def weighted_loss(self, per_sample_loss, group_idx):
        """Globally batch-normalized weighted mean (see the module docstring).

        The denominator spans ALL ranks, and the ``* world_size`` factor cancels
        DDP's 1/W gradient averaging, so the gradient is exactly that of
        ``sum_all(w*l) / sum_all(w)``.
        """
        weights = self.group_weights[group_idx]
        global_den = self._all_reduce_sum(weights.detach().float().sum().clone())
        loss = (weights * per_sample_loss).sum() / global_den.clamp(min=1e-12)
        if is_dist():
            loss = loss * self.world_size
        return loss, weights

    def accumulate_group_losses(self, per_sample_loss, group_idx):
        """Per-group loss sums/counts, all-reduced so every rank sees the world."""
        detached = per_sample_loss.detach().float()
        k = len(self.group_names)
        sums = torch.zeros(k, dtype=torch.float32, device=self.device)
        counts = torch.zeros(k, dtype=torch.float32, device=self.device)
        sums.index_add_(0, group_idx, detached)
        counts.index_add_(0, group_idx, torch.ones_like(detached))

        self._all_reduce_sum(sums)
        self._all_reduce_sum(counts)

        per_group_now = {}
        for gi, name in enumerate(self.group_names):
            count = counts[gi].item()
            if count == 0:
                continue
            total = sums[gi].item()
            self._group_loss_sum[name] += total
            self._group_loss_count[name] += int(count)
            per_group_now[name] = total / count
        return per_group_now

    def _global_mean(self, local_value):
        """Mean of a per-rank scalar across the world (for logging only)."""
        if not is_dist():
            return local_value.detach().float()
        tensor = local_value.detach().float().clone()
        self._all_reduce_sum(tensor)
        return tensor / self.world_size

    def train(self, prob_focus_present=0.0, focus_present_mask=None, log_fn=noop):
        assert callable(log_fn)

        while self.step < self.train_num_steps:
            # Reshuffle per step so each rank's shard changes; cycle() restarts the
            # loader forever, and set_epoch keeps the two in sync across ranks.
            self.sampler.set_epoch(self.step)

            input_video, contour_cond_video, group_idx, _, valid_frames = next(self.dl)
            input_video = input_video.to(self.device, non_blocking=True)
            contour_cond_video = contour_cond_video.to(self.device, non_blocking=True)
            group_idx = group_idx.to(self.device, non_blocking=True)
            valid_frames = valid_frames.to(self.device, non_blocking=True) if self.use_frame_validity else None

            with autocast(enabled=self.amp):
                per_sample_loss, x0, x_start, per_sample_mse = self.ddp_model(
                    input_video,
                    cond=[contour_cond_video],
                    valid_frames=valid_frames,
                    per_sample=True,
                    prob_focus_present=prob_focus_present,
                    focus_present_mask=focus_present_mask,
                )

            loss, _ = self.weighted_loss(per_sample_loss, group_idx)
            self.scaler.scale(loss).backward()

            # Logged values are global means so the curves match a single-GPU run.
            # _global_mean undoes the * world_size factor in `loss`, so the logged
            # value is exactly the globally self-normalized weighted mean.
            global_loss = self._global_mean(loss)
            global_unweighted = self._global_mean(per_sample_loss.mean())
            global_disp_recon_mse = self._global_mean(per_sample_mse.mean())
            per_group_now = self.accumulate_group_losses(per_sample_loss, group_idx)

            log = {
                "loss": global_loss.item(),
                "unweighted_loss": global_unweighted.item(),
                "disp_recon_mse": global_disp_recon_mse.item(),
            }

            if is_main_process():
                group_parts = ", ".join(f"{name}={value:.4f}" for name, value in per_group_now.items())
                print(
                    f"{self.step}: loss={log['loss']:.6f} unweighted={log['unweighted_loss']:.6f} "
                    f"disp_recon_mse={log['disp_recon_mse']:.6f} | {group_parts}"
                )

            if exists(self.max_grad_norm):
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.ddp_model.parameters(), self.max_grad_norm)

            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if is_main_process():
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
                        ddim_spacing=self.preview_ddim_spacing,
                        ddim_clip_x_start=self.preview_ddim_clip_x_start,
                        ddim_clip_percentile=self.preview_ddim_clip_percentile,
                    )

            if self.use_wandb:
                wandb_log = dict(log)
                wandb_log["step"] = self.step
                for name, value in per_group_now.items():
                    wandb_log[f"loss_{name}"] = value
                for name in self.group_names:
                    count = self._group_loss_count[name]
                    if count > 0:
                        wandb_log[f"loss_{name}_cummean"] = self._group_loss_sum[name] / count
                wandb.log(wandb_log, step=self.step)

            log_fn(log)
            self.step += 1

            # Keep rank-0's sampling/checkpointing from desyncing the next reduction.
            if is_dist():
                torch.distributed.barrier()

        if self.use_wandb:
            wandb.finish()

        if is_main_process():
            print("training completed")
