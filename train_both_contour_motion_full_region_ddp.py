"""
Multi-GPU (single-node) DDP version of train_both_contour_motion_full_region.py.

Full-region-noise training conditioned on BOTH the contour mask and the motion
field. All hyperparameters live in a JSON config (default:
configs/train_both_contour_motion_full_region_ddp.json); see configs/README.md
for what each field means. Launch with torchrun. Example for 4 GPUs on one node:

    torchrun --standalone --nproc_per_node=4 train_both_contour_motion_full_region_ddp.py

For 2 GPUs, with a custom config:

    torchrun --standalone --nproc_per_node=2 train_both_contour_motion_full_region_ddp.py --config configs/my_experiment_ddp.json

NOTE: `training.per_gpu_batch_size` is PER-GPU. With N GPUs the effective global
batch size is `per_gpu_batch_size * N`. The single-GPU script used a global batch
of 16. To keep the same global batch on 4 GPUs, set per_gpu_batch_size = 4; on
2 GPUs set per_gpu_batch_size = 8. Adjust `training.lr` if you intentionally grow
the global batch (linear LR scaling is a reasonable starting point).

The contour and motion conditions are paired by SORTED FILENAME ORDER, both in
the training Dataset and when drawing a preview sample, so the two condition
directories must contain the same number of .npy files under matching names.
"""

import argparse
import json
from pathlib import Path

from video_diffusion_pytorch.video_diffusion_cross_attention_with_both_contour_motion import (
    Unet3D,
    GaussianDiffusion,
)
from video_diffusion_pytorch.config_snapshot import save_config_snapshot
from video_diffusion_pytorch.trainer_ddp_both_contour_motion import (
    BothCondDDPTrainer,
    setup_distributed,
    cleanup_distributed,
    is_main_process,
)

DEFAULT_CONFIG = (
    Path(__file__).resolve().parent / "configs" / "train_both_contour_motion_full_region_ddp.json"
)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="path to the JSON hyperparameter config (see configs/)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    with args.config.open() as f:
        cfg = json.load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    diffusion_cfg = cfg["diffusion"]
    train_cfg = cfg["training"]
    log_cfg = cfg.get("logging", {})
    # Optional per-sample valid-frame loss masking (see configs/README.md).
    # Absent/empty block => masking off (original all-frames loss).
    fv_cfg = cfg.get("frame_validity") or {}

    local_rank, global_rank, world_size, device = setup_distributed()

    if is_main_process():
        save_config_snapshot(cfg, cfg["experiment_name"])

    model = Unet3D(
        dim=model_cfg["dim"],
        cond_dim=model_cfg["cond_dim"],  # video encoder output dim
        channels=model_cfg["channels"],
        dim_mults=tuple(model_cfg["dim_mults"]),
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=diffusion_cfg["image_size"],
        channels=diffusion_cfg["channels"],
        num_frames=diffusion_cfg["num_frames"],
        timesteps=diffusion_cfg["timesteps"],
        loss_type=diffusion_cfg["loss_type"],
        contour_noise_only=diffusion_cfg["contour_noise_only"],
        # Optional: predates most configs, so .get() with the default.
        disp_rel_metric=diffusion_cfg.get("disp_rel_metric", "mse"),
        disp_metrics=diffusion_cfg.get("disp_metrics", True),
        disp_scale=diffusion_cfg.get("disp_scale", 5.0),
    ).to(device)

    base_path = Path(data_cfg["base_path"])

    # Sampling conditions can come from a separate dataset root via the optional
    # sampling_base_path config field. Falls back to base_path when absent.
    sampling_base_path = Path(data_cfg.get("sampling_base_path", str(base_path)))

    trainer = BothCondDDPTrainer(
        diffusion_model=diffusion,
        input_video_folder=str(base_path / data_cfg["input_video_subdir"]),
        local_rank=local_rank,
        contour_condition_video_dir=str(base_path / data_cfg["contour_condition_subdir"]),
        motion_condition_video_dir=str(base_path / data_cfg["motion_condition_subdir"]),
        sampling_contour_condition_video_dir=str(
            sampling_base_path / data_cfg["sampling_contour_condition_subdir"]
        ),
        sampling_motion_condition_video_dir=str(
            sampling_base_path / data_cfg["sampling_motion_condition_subdir"]
        ),
        train_batch_size=train_cfg["per_gpu_batch_size"],  # PER-GPU
        train_lr=train_cfg["lr"],
        save_and_sample_every=train_cfg["save_and_sample_every"],
        train_num_steps=train_cfg["num_steps"],
        gradient_accumulate_every=train_cfg["gradient_accumulate_every"],
        ema_decay=train_cfg["ema_decay"],
        amp=train_cfg["amp"],
        experiment_name=cfg["experiment_name"],
        use_wandb=log_cfg.get("use_wandb", False),
        wandb_project=log_cfg.get("wandb_project", "genstrain-motion-diffusion"),
        wandb_config=cfg,
        valid_frames_csv=fv_cfg.get("csv_path"),
        valid_frames_filename_col=fv_cfg.get("filename_col", "dense_filename"),
        valid_frames_count_col=fv_cfg.get("valid_frames_col", "dense_valid_frames"),
        valid_frames_missing=fv_cfg.get("missing", "full"),
        num_workers=data_cfg["num_workers"],
    )

    # Load the latest checkpoint (milestone = -1) to resume from latest.
    # Wrapped in try/except so a fresh run (no checkpoints yet) doesn't crash.
    if train_cfg["resume"]:
        try:
            trainer.load(milestone=-1)
        except AssertionError:
            if is_main_process():
                print("no checkpoint found, starting from scratch")

    try:
        # Sampler for the periodic preview samples taken during training.
        # Defaults to ddpm, i.e. unchanged behaviour (see configs/README.md).
        trainer.configure_preview_sampler(
            sampler=train_cfg.get("preview_sampler", "ddpm"),
            ddim_steps=train_cfg.get("preview_ddim_steps", 50),
            ddim_eta=train_cfg.get("preview_ddim_eta", 0.0),
            spacing=train_cfg.get("preview_ddim_spacing", "uniform"),
            clip_x_start=train_cfg.get("preview_ddim_clip_x_start", None),
            clip_percentile=train_cfg.get("preview_ddim_clip_percentile", 0.995),
        )
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
