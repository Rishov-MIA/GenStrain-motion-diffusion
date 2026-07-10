"""
DistributedDataParallel (DDP) variant of the Trainer for multi-GPU, single-node
training.

This subclasses the existing single-GPU Trainer in
`video_diffusion_cross_attention_with_motion_after_only_contour` and overrides
only the parts that must be DDP-aware:

  * the model is wrapped in DistributedDataParallel
  * the DataLoader uses a DistributedSampler so each rank sees a disjoint shard
  * checkpoint save/load, EMA, sampling, GIF/console logging happen on rank 0 only
  * the underlying (unwrapped) model is used for EMA + checkpointing

Launch with torchrun, e.g. (4 GPUs on one node):

    torchrun --standalone --nproc_per_node=4 train_full_region_ddp.py

IMPORTANT: with DDP the `train_batch_size` you pass is PER-GPU. The effective
(global) batch is `train_batch_size * world_size`. Scale the learning rate
accordingly (a common starting point is linear scaling).
"""

import os
import copy
import inspect

from tqdm import tqdm

import torch
from torch import nn
from torch.optim import Adam
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP

# AMP: prefer the new torch.amp API (torch >= 2.3). Fall back to the deprecated
# torch.cuda.amp on older torch so this file stays backward compatible.
try:
    from torch.amp import autocast as _autocast, GradScaler as _GradScaler

    def autocast(enabled=True):
        return _autocast("cuda", enabled=enabled)

    def GradScaler(enabled=True):
        return _GradScaler("cuda", enabled=enabled)
except ImportError:  # torch < 2.3
    from torch.cuda.amp import autocast, GradScaler

from video_diffusion_pytorch.video_diffusion_cross_attention_with_motion_after_only_contour import (
    Trainer,
    EMA,
    Dataset,
    cycle,
    exists,
    noop,
    normalize_cond_img,
    random_pick_condition_videos,
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
        # Pass device_id so collectives (e.g. barrier) know the device and don't
        # warn about inferring it from the current device.
        torch.distributed.init_process_group(backend="nccl", device_id=device)
    return local_rank, get_rank(), get_world_size(), device


def cleanup_distributed():
    if is_dist():
        torch.distributed.destroy_process_group()


class DDPTrainer(Trainer):
    def __init__(self, diffusion_model, input_video_folder, *, local_rank=0, **kwargs):
        # We need to customize DataLoader / model wrapping, so we deliberately do
        # NOT call super().__init__ (it builds a non-distributed DataLoader and
        # never wraps the model). Instead we replicate the relevant setup here.
        self.local_rank = local_rank
        self.device = torch.device("cuda", local_rank)
        self.world_size = get_world_size()

        ema_decay = kwargs.get("ema_decay", 0.995)
        self.model = diffusion_model  # already on the correct device by caller
        self.ema = EMA(ema_decay)
        # EMA is maintained from the *unwrapped* model on rank 0 only.
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = kwargs.get("update_ema_every", 10)
        self.sampling_contour_condition_video_dir = kwargs.get("sampling_contour_condition_video_dir")

        self.step_start_ema = kwargs.get("step_start_ema", 2000)
        self.save_and_sample_every = kwargs.get("save_and_sample_every", 1000)

        # NOTE: per-GPU batch size.
        self.batch_size = kwargs.get("train_batch_size", 32)
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = kwargs.get("gradient_accumulate_every", 2)
        self.train_num_steps = kwargs.get("train_num_steps", 100000)

        image_size = diffusion_model.image_size
        channels = diffusion_model.channels
        num_frames = diffusion_model.num_frames

        inference_only = kwargs.get("inference_only", False)
        contour_condition_video_dir = kwargs.get("contour_condition_video_dir")
        train_lr = kwargs.get("train_lr", 1e-4)

        if not inference_only:
            self.ds = Dataset(
                input_video_folder,
                image_size,
                contour_condition_video_dir,
                channels=channels,
                num_frames=num_frames,
            )

            if is_main_process():
                print(f"found {len(self.ds)} videos as .npy files at {input_video_folder}")
            assert len(self.ds) > 0, "need to have at least 1 video to start training"

            self.sampler = DistributedSampler(
                self.ds,
                num_replicas=self.world_size,
                rank=get_rank(),
                shuffle=True,
                drop_last=True,
            )
            self.dl = cycle(
                data.DataLoader(
                    self.ds,
                    batch_size=self.batch_size,
                    sampler=self.sampler,
                    pin_memory=True,
                    num_workers=kwargs.get("num_workers", 4),
                    drop_last=True,
                )
            )
            # Optimizer is built on the DDP-wrapped module's params (same params,
            # just wrapped). We wrap first, then build the optimizer.
            self.ddp_model = DDP(
                self.model,
                device_ids=[local_rank],
                output_device=local_rank,
                find_unused_parameters=False,
            )
            self.opt = Adam(self.ddp_model.parameters(), lr=train_lr)
        else:
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
            # Plot the epoch metrics against epoch number (not step): each gets its
            # own panel with `epoch` as the x-axis. Step-level metrics (loss,
            # loss_cummean, disp_recon_mse) keep the default global-step axis.
            wandb.define_metric("epoch")
            wandb.define_metric("epoch_loss", step_metric="epoch")
            wandb.define_metric("epoch_disp_recon_mse", step_metric="epoch")

        # True cumulative mean-since-start of the training loss (rank 0 uses the
        # local loss, matching the console print). Not checkpointed.
        self._loss_sum = 0.0
        self._loss_count = 0

        # ---- epoch loss bookkeeping ----
        # One epoch = the model has seen the whole dataset once. With
        # DistributedSampler(drop_last=True) each rank sees len(ds)//world_size
        # samples, and every optimizer step consumes
        # per_gpu_batch_size * world_size * gradient_accumulate_every samples, so:
        if not inference_only:
            global_batch = self.batch_size * self.world_size * self.gradient_accumulate_every
            self.steps_per_epoch = max(1, len(self.ds) // global_batch)
        else:
            self.steps_per_epoch = None
        # Running sums of the (globally-averaged) per-step loss and disp_recon_mse
        # within the current epoch, and how many steps have contributed. Reset at
        # each epoch boundary. (Both share _epoch_step_count as the denominator.)
        self._epoch_loss_sum = 0.0
        self._epoch_disp_recon_mse_sum = 0.0
        self._epoch_step_count = 0

        from pathlib import Path

        checkpoints_folder = f"./{self.experiment_name}/checkpoints"
        self.checkpoints_folder = Path(checkpoints_folder)
        if is_main_process():
            self.checkpoints_folder.mkdir(exist_ok=True, parents=True)

        self.reset_parameters()

    # ---- checkpointing: unwrap DDP, only rank 0 writes ----

    @property
    def raw_model(self):
        """The underlying nn.Module regardless of DDP wrapping."""
        return self.ddp_model.module if isinstance(self.ddp_model, DDP) else self.ddp_model

    def save(self, milestone):
        if not is_main_process():
            return
        ckpt = {
            "step": self.step,
            "model": self.raw_model.state_dict(),
            "ema": self.ema_model.state_dict(),
            "scaler": self.scaler.state_dict(),
        }
        torch.save(ckpt, str(self.checkpoints_folder / f"model-{milestone}.pt"))

    def load(self, milestone, **kwargs):
        from pathlib import Path

        if milestone == -1:
            all_milestones = [int(p.stem.split("-")[-1]) for p in Path(self.checkpoints_folder).glob("**/*.pt")]
            assert len(all_milestones) > 0, "need at least one milestone to load from (milestone == -1)"
            milestone = max(all_milestones)

        # Load onto this rank's device.
        # torch >= 2.6 flipped torch.load's default to weights_only=True, which
        # rejects our checkpoint dicts; pass weights_only=False there. Older torch
        # (e.g. 2.0) doesn't accept that kwarg, so only pass it when supported.
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

    # ---- EMA from the unwrapped model ----

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.raw_model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.raw_model)

    # ---- training loop ----

    def train(self, prob_focus_present=0.0, focus_present_mask=None, log_fn=noop):
        assert callable(log_fn)

        # Rank-0-only horizontal progress bar for the current epoch. It fills as
        # steps within the epoch complete and resets at each epoch boundary. The
        # bar's fill and the epoch-loss average share one counter
        # (self._epoch_step_count), so they always reset together. On a resume both
        # start at 0 and the bar restarts the current epoch from the beginning (the
        # loss of earlier steps in that epoch is unrecoverable anyway); the epoch
        # NUMBER still reflects the restored step so labels stay correct.
        epoch_bar = None
        current_epoch = self.step // self.steps_per_epoch
        self._epoch_loss_sum = 0.0
        self._epoch_disp_recon_mse_sum = 0.0
        self._epoch_step_count = 0
        if is_main_process():
            epoch_bar = tqdm(
                total=self.steps_per_epoch,
                desc=f"epoch {current_epoch}",
                unit="step",
                leave=True,
                dynamic_ncols=True,
            )

        while self.step < self.train_num_steps:
            # Reshuffle the shard each "epoch-ish" so ranks don't repeat order.
            if hasattr(self, "sampler"):
                self.sampler.set_epoch(self.step)

            for i in range(self.gradient_accumulate_every):
                input_video, contour_cond_video = next(self.dl)
                input_video = input_video.to(self.device, non_blocking=True)
                contour_cond_video = contour_cond_video.to(self.device, non_blocking=True)

                # Only sync gradients on the last accumulation micro-step.
                is_last_micro = i == (self.gradient_accumulate_every - 1)
                sync_ctx = (
                    self.ddp_model.no_sync()
                    if (isinstance(self.ddp_model, DDP) and not is_last_micro)
                    else _nullcontext()
                )

                with sync_ctx:
                    with autocast(enabled=self.amp):
                        loss, _, _, disp_recon_mse = self.ddp_model(
                            input_video,
                            cond=[contour_cond_video],
                            prob_focus_present=prob_focus_present,
                            focus_present_mask=focus_present_mask,
                        )
                        self.scaler.scale(loss / self.gradient_accumulate_every).backward()

            # Globally-averaged loss across all ranks (true global-batch loss),
            # used for the epoch loss. all_reduce over a detached copy so it never
            # touches autograd. Falls back to the local loss when not distributed.
            # The displacement reconstruction MSE is averaged the same way.
            if is_dist():
                global_loss_t = loss.detach().clone()
                torch.distributed.all_reduce(global_loss_t, op=torch.distributed.ReduceOp.SUM)
                global_loss = global_loss_t.item() / self.world_size

                disp_mse_t = disp_recon_mse.detach().clone()
                torch.distributed.all_reduce(disp_mse_t, op=torch.distributed.ReduceOp.SUM)
                global_disp_recon_mse = disp_mse_t.item() / self.world_size
            else:
                global_loss = loss.item()
                global_disp_recon_mse = disp_recon_mse.item()

            # Per-step loss line. Route through the bar's writer so it scrolls
            # above the pinned epoch progress bar instead of fighting it.
            if is_main_process():
                epoch_bar.write(
                    f"{self.step}: {loss.item()} | disp_recon_mse: {global_disp_recon_mse}"
                )

            log = {"loss": loss.item(), "disp_recon_mse": global_disp_recon_mse, "step": self.step}
            # Cumulative mean of the loss since start (rank 0 only; it's the sole
            # consumer of `log` via wandb, and only rank 0 logs).
            if is_main_process():
                self._loss_sum += loss.item()
                self._loss_count += 1
                log["loss_cummean"] = self._loss_sum / self._loss_count

            # Accumulate the epoch loss / disp_recon_mse on every step (all ranks
            # stay in sync on the global values; only rank 0 logs them) and advance
            # the epoch bar. When the current step completes a full pass over the
            # dataset, emit the means, then reset the bar and counters for the next
            # epoch.
            self._epoch_loss_sum += global_loss
            self._epoch_disp_recon_mse_sum += global_disp_recon_mse
            self._epoch_step_count += 1
            if is_main_process():
                epoch_bar.update(1)
                epoch_bar.set_postfix(loss=global_loss)
            if self._epoch_step_count >= self.steps_per_epoch:
                epoch_loss = self._epoch_loss_sum / self._epoch_step_count
                epoch_disp_recon_mse = self._epoch_disp_recon_mse_sum / self._epoch_step_count
                epoch = (self.step + 1) // self.steps_per_epoch
                if is_main_process():
                    epoch_bar.write(
                        f"  epoch {epoch} | epoch_loss {epoch_loss} | "
                        f"epoch_disp_recon_mse {epoch_disp_recon_mse}"
                    )
                    log["epoch"] = epoch
                    log["epoch_loss"] = epoch_loss
                    log["epoch_disp_recon_mse"] = epoch_disp_recon_mse
                    epoch_bar.reset()
                    epoch_bar.set_description(f"epoch {epoch + 1}")
                self._epoch_loss_sum = 0.0
                self._epoch_disp_recon_mse_sum = 0.0
                self._epoch_step_count = 0

            if exists(self.max_grad_norm):
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.ddp_model.parameters(), self.max_grad_norm)

            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            # rank-0-only sampling / checkpointing
            if is_main_process():
                if self.step != 0 and self.step % self.save_and_sample_every == 0:
                    milestone = self.step // self.save_and_sample_every
                    self.save(milestone)

                    sample_contour_cond, sample_cond_filenames = random_pick_condition_videos(
                        num_samples=1,
                        contour_cond_video_dir=self.sampling_contour_condition_video_dir,
                    )
                    sample_contour_cond = normalize_cond_img(sample_contour_cond)
                    device = next(self.ema_model.parameters()).device
                    sample_contour_cond = sample_contour_cond.to(device)

                    self.sample_and_save(
                        milestone,
                        contour_cond_video=sample_contour_cond,
                        cond_filenames=sample_cond_filenames,
                        save_folder=f"./{self.experiment_name}/sampled_videos_infos",
                    )

            if self.use_wandb:
                wandb.log(log, step=self.step)

            log_fn(log)
            self.step += 1

            # Keep all ranks aligned before the next step (so rank-0 sampling,
            # which can take a while, doesn't desync gradient all-reduces).
            if is_dist():
                torch.distributed.barrier()

        if is_main_process() and epoch_bar is not None:
            epoch_bar.close()

        if self.use_wandb:
            wandb.finish()

        if is_main_process():
            print("training completed")


class _nullcontext:
    def __enter__(self):
        return None

    def __exit__(self, *exc):
        return False
