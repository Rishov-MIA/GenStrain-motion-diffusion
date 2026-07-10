"""
DistributedDataParallel (DDP) variant of the Group-DRO trainer for multi-GPU,
single-node training.

This subclasses ``GroupDROTrainer`` and overrides only the parts that must be
DDP-aware. The Group-DRO algorithm has two coupling points that make naive DDP
incorrect, and this file handles both:

  1. THE SAME GROUP PER STEP ON ALL RANKS.
     DDP averages each parameter's gradient across ranks. If different ranks
     trained different groups in the same step, that average would mix gradients
     from different groups — meaningless. So rank 0 samples ``group_idx`` once and
     broadcasts it; every rank trains that group. The effective (global) batch for
     the step is ``train_batch_size * world_size``, all drawn from the one group.

  2. THE q UPDATE USES THE GLOBAL GROUP LOSS AND STAYS IDENTICAL ON EVERY RANK.
     Each rank computes its local L_g on its own shard of the group batch. We
     all-reduce (mean) those into the true global L_g, then every rank runs the
     SAME deterministic exponentiated-gradient update on its own copy of
     ``dro_log_q``. Because the inputs (global L_g, group_idx, counts) are
     identical across ranks, ``dro_log_q`` and hence q_g stay bit-identical, so
     the ``q_g * loss`` model step is consistent everywhere.

Per-group loaders: each rank keeps its own set of per-group loaders, seeded
differently per rank, so the ranks draw DIFFERENT samples of the SAME group in a
step (a genuine larger effective batch), still with replacement.

Checkpoint save/load, EMA, sampling and GIF/console logging happen on rank 0 only,
using the unwrapped model.

Launch with torchrun, e.g. (4 GPUs on one node):

    torchrun --standalone --nproc_per_node=4 train_group_dro_ddp.py

IMPORTANT: ``train_batch_size`` is PER-GPU. The effective per-step group batch is
``train_batch_size * world_size``. Scale ``train_lr`` accordingly (linear scaling
is a reasonable start). ``dro_eta_q`` / ``dro_adjustment_c`` operate on the global
mean loss and do not need rescaling with world size.
"""

import copy
import inspect
import math
import os
import random
from pathlib import Path

import numpy as np
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

from video_diffusion_pytorch.utils import generate_displacement_quiver_gifs_with_gt
from video_diffusion_pytorch.video_diffusion_cross_attention_with_motion_after_only_contour import (
    EMA,
    cycle,
    exists,
    noop,
    normalize_cond_img,
    random_pick_condition_videos,
)
from video_diffusion_pytorch.trainer_group_dro import (
    GroupDROTrainer,
    GroupedContourDataset,
)

try:
    import wandb
except ImportError:  # wandb is optional; training works without it
    wandb = None


def is_dist():
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_rank():
    return torch.distributed.get_rank() if is_dist() else 0


def get_world_size():
    return torch.distributed.get_world_size() if is_dist() else 1


def is_main_process():
    return get_rank() == 0


def setup_distributed():
    """Initialize the process group from torchrun-provided env vars.

    Returns (local_rank, global_rank, world_size, device).
    """
    local_rank = int(os.environ.get("LOCAL_RANK", 0))
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
    if not (torch.distributed.is_available() and torch.distributed.is_initialized()):
        torch.distributed.init_process_group(backend="nccl", device_id=device)
    return local_rank, get_rank(), get_world_size(), device


def cleanup_distributed():
    if is_dist():
        torch.distributed.destroy_process_group()


class GroupDRODDPTrainer(GroupDROTrainer):
    def __init__(self, diffusion_model, input_video_folder, *, local_rank=0, **kwargs):
        # We deliberately do NOT call super().__init__ (it builds non-distributed
        # loaders and never wraps the model). We replicate the relevant setup here
        # in a DDP-aware way, mirroring trainer_ddp.DDPTrainer.
        object.__init__(self)

        gradient_accumulate_every = kwargs.get("gradient_accumulate_every", 1)
        if gradient_accumulate_every != 1:
            raise ValueError("GroupDRODDPTrainer currently supports gradient_accumulate_every=1")

        inference_only = kwargs.get("inference_only", False)
        metadata_json_path = kwargs.get("metadata_json_path")
        if metadata_json_path is None and not inference_only:
            raise ValueError("metadata_json_path is required for GroupDRODDPTrainer")

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

        # PER-GPU batch size. Effective per-step group batch = batch_size * world_size.
        self.batch_size = kwargs.get("train_batch_size", 32)
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = kwargs.get("train_num_steps", 100000)

        # See GroupDROTrainer for eta_q tuning notes: it must keep exp(eta_q*L_g)
        # near 1 (eta_q ~0.01-0.05 for these mean losses), else q collapses onto
        # one group. Acts on the global mean loss; not rescaled by world size.
        self.dro_eta_q = float(kwargs.get("dro_eta_q", 0.02))
        self.dro_adjustment_c = float(kwargs.get("dro_adjustment_c", 0.0))
        self.dro_freeze_q = bool(kwargs.get("dro_freeze_q", False))
        self.weight_decay = float(kwargs.get("weight_decay", 0.0))
        self.num_workers = int(kwargs.get("num_workers", 4))

        image_size = diffusion_model.image_size
        channels = diffusion_model.channels
        num_frames = diffusion_model.num_frames

        contour_condition_video_dir = kwargs.get("contour_condition_video_dir")
        train_lr = kwargs.get("train_lr", 1e-4)

        if not inference_only:
            self.ds = GroupedContourDataset(
                input_video_folder,
                image_size,
                contour_condition_video_dir,
                metadata_json_path,
                channels=channels,
                num_frames=num_frames,
            )

            if is_main_process():
                print(f"found {len(self.ds)} supported grouped videos as .npy files at {input_video_folder}")
                print(f"group counts: {dict(zip(self.ds.group_names, self.ds.group_counts))}")
                if self.ds.skipped:
                    print(f"skipped files: {self.ds.skipped}")

            assert len(self.ds) > 0, "need at least 1 supported grouped video to start training"
            assert len(self.ds.group_names) > 0, "need at least 1 active group to start training"

            self.group_names = list(self.ds.group_names)
            self.group_counts = torch.tensor(self.ds.group_counts, dtype=torch.float32, device=self.device)
            # True cumulative mean-since-start per group: running sum + count, fed
            # the GLOBAL group loss (identical across ranks); only rank 0 logs it.
            self._group_loss_sum = {name: 0.0 for name in self.group_names}
            self._group_loss_count = {name: 0 for name in self.group_names}
            self.dro_log_q = torch.full(
                (len(self.group_names),),
                -math.log(len(self.group_names)),
                dtype=torch.float32,
                device=self.device,
            )

            # Per-group, per-rank loaders (with replacement). Seeding the sampler
            # generator by rank makes each rank draw DIFFERENT samples of the same
            # group in a step, so the world's effective batch is a genuine larger
            # i.i.d. draw from that group (not the same batch duplicated per GPU).
            self.group_loaders = []
            for group_idx in range(len(self.group_names)):
                group_indices = self.ds.indices_for_group(group_idx)
                subset = data.Subset(self.ds, group_indices)
                workers = min(self.num_workers, len(group_indices))
                generator = torch.Generator()
                generator.manual_seed(1234 + 97 * get_rank() + group_idx)
                sampler = data.RandomSampler(
                    subset,
                    replacement=True,
                    num_samples=max(len(group_indices), self.batch_size) * 1000,
                    generator=generator,
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
            self.dro_log_q = torch.empty(0, dtype=torch.float32, device=self.device)
            self.ddp_model = self.model

        self.step = 0

        self.amp = kwargs.get("amp", False)
        self.scaler = GradScaler(enabled=self.amp)
        self.max_grad_norm = kwargs.get("max_grad_norm")

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

    # ---- checkpointing: rank 0 only, unwrapped model ----

    def save(self, milestone):
        if not is_main_process():
            return
        ckpt = {
            "step": self.step,
            "model": self.raw_model.state_dict(),
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

        map_location = {"cuda:0": f"cuda:{self.local_rank}"}
        ckpt_path = str(self.checkpoints_folder / f"model-{milestone}.pt")
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

        if "dro_log_q" not in ckpt:
            if is_main_process():
                print("checkpoint has no DRO state; keeping uniform group weights")
            return

        checkpoint_group_names = list(ckpt.get("dro_group_names", []))
        if checkpoint_group_names != self.group_names:
            raise ValueError(
                "DRO checkpoint group names do not match the current dataset: "
                f"checkpoint={checkpoint_group_names}, current={self.group_names}"
            )

        ckpt_counts = ckpt.get("dro_group_counts")
        if ckpt_counts is not None and is_main_process():
            ckpt_counts_t = torch.as_tensor(ckpt_counts, dtype=torch.float32)
            if not torch.equal(ckpt_counts_t.cpu(), self.group_counts.detach().cpu()):
                print(
                    "WARNING: DRO group counts changed since the checkpoint was saved "
                    f"(checkpoint={ckpt_counts_t.tolist()}, current={self.group_counts.tolist()}). "
                    "Keeping learned q but using current counts for the C/sqrt(n_g) adjustment."
                )

        # Every rank loads the same q so ranks stay in lockstep.
        self.dro_log_q = ckpt["dro_log_q"].to(self.device, dtype=torch.float32)

    # ---- group selection: rank 0 chooses, all ranks agree ----

    def sample_group_idx(self):
        """Sample a group on rank 0 and broadcast so every rank trains it."""
        if is_dist():
            idx_tensor = torch.empty(1, dtype=torch.long, device=self.device)
            if is_main_process():
                idx_tensor[0] = random.randrange(len(self.group_names))
            torch.distributed.broadcast(idx_tensor, src=0)
            return int(idx_tensor.item())
        return random.randrange(len(self.group_names))

    def _global_group_loss(self, local_loss):
        """All-reduce the per-rank group loss into the global mean L_g."""
        if not is_dist():
            return local_loss.detach().float()
        loss_tensor = local_loss.detach().float().clone()
        torch.distributed.all_reduce(loss_tensor, op=torch.distributed.ReduceOp.SUM)
        loss_tensor /= self.world_size
        return loss_tensor

    # ---- training loop ----

    def train(self, prob_focus_present=0.0, focus_present_mask=None, log_fn=noop):
        assert callable(log_fn)

        while self.step < self.train_num_steps:
            group_idx = self.sample_group_idx()
            group_name = self.group_names[group_idx]

            # Each rank pulls its own B-sized batch of THIS group (different samples
            # per rank via per-rank seeding), for a global batch of B * world_size.
            input_video, contour_cond_video, _, _ = next(self.group_loaders[group_idx])
            input_video = input_video.to(self.device, non_blocking=True)
            contour_cond_video = contour_cond_video.to(self.device, non_blocking=True)

            with autocast(enabled=self.amp):
                loss, x0, x_start = self.ddp_model(
                    input_video,
                    cond=[contour_cond_video],
                    prob_focus_present=prob_focus_present,
                    focus_present_mask=focus_present_mask,
                )

            # Global L_g = mean of per-rank losses. Update q identically on every
            # rank so dro_log_q / q_g stay in lockstep and the model step matches.
            global_loss = self._global_group_loss(loss)
            score, q_g = self.update_dro_weight(group_idx, global_loss)

            # Backward on q_g * local_loss. DDP averages grads across ranks, giving
            # q_g * mean_ranks(local grads) == q_g * grad(global L_g). With q frozen,
            # backprop the raw loss (see GroupDROTrainer.train for the rationale).
            weighted_loss = loss if self.dro_freeze_q else q_g * loss
            self.scaler.scale(weighted_loss).backward()

            log = {
                "loss": global_loss.item(),
                "weighted_loss": weighted_loss.item(),
                "dro_score": score.item(),
                "dro_q": q_g.item(),
                "group": group_name,
            }

            if is_main_process():
                q_parts = ", ".join(f"{name}={q:.4f}" for name, q in zip(self.group_names, self.dro_q.detach().cpu()))
                print(
                    f"{self.step}: group={group_name} "
                    f"loss={global_loss.item():.6f} weighted={weighted_loss.item():.6f} "
                    f"q={q_g.item():.6f} | {q_parts}"
                )

            if exists(self.max_grad_norm):
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.ddp_model.parameters(), self.max_grad_norm)

            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            # rank-0-only visualization + checkpoint/sampling
            if is_main_process():
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
                    )

            if self.use_wandb:  # rank 0 only
                # Accumulate the true cumulative mean for the group sampled this
                # step, using the GLOBAL group loss.
                step_loss = global_loss.item()
                self._group_loss_sum[group_name] += step_loss
                self._group_loss_count[group_name] += 1

                # Log the scalar fields plus per-group q_<name> curves; skip the
                # string "group" field (wandb can't plot it).
                wandb_log = {k: v for k, v in log.items() if k != "group"}
                wandb_log["step"] = self.step
                for name, q in zip(self.group_names, self.dro_q.detach().cpu().tolist()):
                    wandb_log[f"q_{name}"] = q
                # Raw GLOBAL loss for the group sampled THIS step (sparse per group),
                # plus the true cumulative mean since start for EVERY group (dense,
                # carried forward every step).
                wandb_log[f"loss_{group_name}"] = step_loss
                for name in self.group_names:
                    count = self._group_loss_count[name]
                    if count > 0:
                        wandb_log[f"loss_{name}_cummean"] = self._group_loss_sum[name] / count
                wandb.log(wandb_log, step=self.step)

            log_fn(log)
            self.step += 1

            # Keep all ranks aligned before the next step so rank-0 sampling (which
            # can be slow) doesn't desync the next gradient all-reduce.
            if is_dist():
                torch.distributed.barrier()

        if self.use_wandb:
            wandb.finish()

        if is_main_process():
            print("training completed")
