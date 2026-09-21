"""DDP (multi-GPU, single-node) trainer for the label-conditioned model.

Same DDP mechanics as trainer_ddp.DDPTrainer (DistributedSampler shard per
rank, DDP-wrapped model, rank-0 EMA / checkpoints / previews, globally averaged
loss and metrics) with three label-specific changes:

  * the dataset is LabelDataset, so every batch carries [B, 3] label indices
  * the labels are passed into the model, which applies classifier-free dropout
  * checkpoints carry the label vocabulary stamp and refuse a mismatch on load

Launch with torchrun, e.g.:

    torchrun --standalone --nproc_per_node=4 train_label_cond_ddp.py
"""

import inspect
from pathlib import Path

import torch
from torch import nn
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils import data
from torch.utils.data.distributed import DistributedSampler
from tqdm import tqdm

from video_diffusion_pytorch.checkpoint_compat import record_disp_scale, verify_disp_scale
from video_diffusion_pytorch.label_cond import (
    LabelCondUnet3D,
    LabelDataset,
    LabelSamplingMixin,
    record_label_cond,
    verify_label_cond,
)
from video_diffusion_pytorch.label_metadata import LABEL_MODES
from video_diffusion_pytorch.trainer_ddp import (
    DDPTrainer,
    _nullcontext,
    autocast,
    get_rank,
    global_metric_items,
    is_dist,
    is_main_process,
)
from video_diffusion_pytorch.video_diffusion_cross_attention_with_motion_after_only_contour import (
    cycle,
    exists,
    noop,
    normalize_cond_img,
    random_pick_condition_videos,
)

try:
    import wandb
except ImportError:
    wandb = None


class LabelCondDDPTrainer(LabelSamplingMixin, DDPTrainer):
    def __init__(self, diffusion_model, input_video_folder, *, label_encoder, preview_label_mode="full",
                 local_rank=0, **kwargs):
        if not isinstance(diffusion_model.denoise_fn, LabelCondUnet3D):
            raise TypeError("LabelCondDDPTrainer needs a LabelCondGaussianDiffusion wrapping a LabelCondUnet3D")
        if preview_label_mode not in LABEL_MODES:
            raise ValueError(f"preview_label_mode must be one of {LABEL_MODES}, got {preview_label_mode!r}")
        self.label_encoder = label_encoder
        self.preview_label_mode = preview_label_mode

        super().__init__(diffusion_model, input_video_folder, local_rank=local_rank, **kwargs)

        # Swap the plain Dataset the parent built for the labelled one; the DDP
        # wrapper and optimizer the parent created are unaffected.
        if not kwargs.get("inference_only", False):
            self.ds = LabelDataset(
                input_video_folder,
                diffusion_model.image_size,
                kwargs.get("contour_condition_video_dir"),
                channels=diffusion_model.channels,
                num_frames=diffusion_model.num_frames,
                valid_frames_map=self.valid_frames_map,
                valid_frames_missing=kwargs.get("valid_frames_missing", "full"),
                label_encoder=label_encoder,
            )
            self.sampler = DistributedSampler(
                self.ds, num_replicas=self.world_size, rank=get_rank(), shuffle=True, drop_last=True,
            )
            self.dl = cycle(
                data.DataLoader(
                    self.ds, batch_size=self.batch_size, sampler=self.sampler, pin_memory=True,
                    num_workers=kwargs.get("num_workers", 4), drop_last=True,
                )
            )

    def _disp_metric_names(self):
        """Declare the rotation diagnostics' epoch panels up front too."""
        names = set(super()._disp_metric_names())
        if getattr(self.raw_model, "disp_metrics", True):
            names.update(("rotation_mae_deg", "gt_rotation_abs_deg"))
            if getattr(self.raw_model, "rotation_loss_weight", 0.0) > 0:
                names.add("rotation_loss")
        return sorted(names)

    # ---- checkpoints -------------------------------------------------------
    def save(self, milestone):
        if not is_main_process():
            return
        ckpt = {
            "step": self.step,
            "model": self.raw_model.state_dict(),
            "ema": self.ema_model.state_dict(),
            "scaler": self.scaler.state_dict(),
        }
        record_disp_scale(ckpt, self.raw_model)
        record_label_cond(ckpt, self.raw_model, self.label_encoder)
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
        verify_disp_scale(ckpt, self.raw_model, milestone)
        verify_label_cond(ckpt, self.raw_model, self.label_encoder, milestone)
        self.step = ckpt["step"]
        self.raw_model.load_state_dict(ckpt["model"], **kwargs)
        self.ema_model.load_state_dict(ckpt["ema"], **kwargs)
        self.scaler.load_state_dict(ckpt["scaler"])

    # ---- training loop ----------------------------------------------------
    def train(self, prob_focus_present=0.0, focus_present_mask=None, log_fn=noop):
        assert callable(log_fn)
        epoch_bar = None
        current_epoch = self.step // self.steps_per_epoch
        self._epoch_loss_sum = 0.0
        self._epoch_disp_recon_mse_sum = 0.0
        self._epoch_step_count = 0
        self._epoch_disp_sums = {}
        self._epoch_disp_counts = {}
        if is_main_process():
            epoch_bar = tqdm(total=self.steps_per_epoch, desc=f"epoch {current_epoch}", unit="step", leave=True, dynamic_ncols=True)

        while self.step < self.train_num_steps:
            if hasattr(self, "sampler"):
                self.sampler.set_epoch(self.step)

            for i in range(self.gradient_accumulate_every):
                input_video, contour_cond_video, valid_frames, labels = next(self.dl)
                input_video = input_video.to(self.device, non_blocking=True)
                contour_cond_video = contour_cond_video.to(self.device, non_blocking=True)
                labels = labels.to(self.device, non_blocking=True)
                valid_frames = valid_frames.to(self.device, non_blocking=True) if self.use_frame_validity else None

                is_last_micro = i == (self.gradient_accumulate_every - 1)
                sync_ctx = (
                    self.ddp_model.no_sync()
                    if (isinstance(self.ddp_model, DDP) and not is_last_micro)
                    else _nullcontext()
                )
                with sync_ctx:
                    with autocast(enabled=self.amp):
                        loss, _, _, disp_recon_mse, disp_metrics = self.ddp_model(
                            input_video,
                            cond=[contour_cond_video],
                            valid_frames=valid_frames,
                            labels=labels,
                            prob_focus_present=prob_focus_present,
                            focus_present_mask=focus_present_mask,
                        )
                        self.scaler.scale(loss / self.gradient_accumulate_every).backward()

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
            disp_items = global_metric_items(disp_metrics, self.world_size)

            if is_main_process():
                disp_parts = "".join(f" | {k}: {v}" for k, v in sorted(disp_items.items()))
                epoch_bar.write(f"{self.step}: {global_loss} | disp_recon_mse: {global_disp_recon_mse}{disp_parts}")

            log = {"loss": global_loss, "disp_recon_mse": global_disp_recon_mse, "step": self.step, **disp_items}
            if is_main_process():
                self._loss_sum += global_loss
                self._loss_count += 1
                log["loss_cummean"] = self._loss_sum / self._loss_count

            self._epoch_loss_sum += global_loss
            self._epoch_disp_recon_mse_sum += global_disp_recon_mse
            self._epoch_step_count += 1
            for name, value in disp_items.items():
                self._epoch_disp_sums[name] = self._epoch_disp_sums.get(name, 0.0) + value
                self._epoch_disp_counts[name] = self._epoch_disp_counts.get(name, 0) + 1
            if is_main_process():
                epoch_bar.update(1)
                epoch_bar.set_postfix(loss=global_loss)
            if self._epoch_step_count >= self.steps_per_epoch:
                epoch_loss = self._epoch_loss_sum / self._epoch_step_count
                epoch_disp_recon_mse = self._epoch_disp_recon_mse_sum / self._epoch_step_count
                epoch_disp = {f"epoch_{n}": tot / self._epoch_disp_counts[n] for n, tot in self._epoch_disp_sums.items()}
                epoch = (self.step + 1) // self.steps_per_epoch
                if is_main_process():
                    epoch_parts = "".join(f" | {k} {v}" for k, v in sorted(epoch_disp.items()))
                    epoch_bar.write(f"  epoch {epoch} | epoch_loss {epoch_loss} | epoch_disp_recon_mse {epoch_disp_recon_mse}{epoch_parts}")
                    log["epoch"] = epoch
                    log["epoch_loss"] = epoch_loss
                    log["epoch_disp_recon_mse"] = epoch_disp_recon_mse
                    log.update(epoch_disp)
                    epoch_bar.reset()
                    epoch_bar.set_description(f"epoch {epoch + 1}")
                self._epoch_loss_sum = 0.0
                self._epoch_disp_recon_mse_sum = 0.0
                self._epoch_step_count = 0
                self._epoch_disp_sums = {}
                self._epoch_disp_counts = {}

            if exists(self.max_grad_norm):
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.ddp_model.parameters(), self.max_grad_norm)

            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if is_main_process() and self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                self.save(milestone)
                sample_contour_cond, sample_cond_filenames = random_pick_condition_videos(
                    num_samples=1, contour_cond_video_dir=self.sampling_contour_condition_video_dir,
                )
                sample_contour_cond = normalize_cond_img(sample_contour_cond)
                device = next(self.ema_model.parameters()).device
                sample_contour_cond = sample_contour_cond.to(device)
                sample_labels = self.labels_for(sample_cond_filenames).to(device)
                self.sample_and_save(
                    milestone,
                    contour_cond_video=sample_contour_cond,
                    cond_filenames=sample_cond_filenames,
                    labels=sample_labels,
                    save_folder=f"./{self.experiment_name}/sampled_videos_infos",
                    sampler=self.preview_sampler,
                    ddim_steps=self.preview_ddim_steps,
                    ddim_eta=self.preview_ddim_eta,
                    ddim_spacing=self.preview_ddim_spacing,
                    ddim_clip_x_start=self.preview_ddim_clip_x_start,
                    ddim_clip_percentile=self.preview_ddim_clip_percentile,
                )

            if self.use_wandb:
                wandb.log(log, step=self.step)
            log_fn(log)
            self.step += 1

            if is_dist():
                torch.distributed.barrier()

        if is_main_process() and epoch_bar is not None:
            epoch_bar.close()
        if self.use_wandb:
            wandb.finish()
        if is_main_process():
            print("training completed")
