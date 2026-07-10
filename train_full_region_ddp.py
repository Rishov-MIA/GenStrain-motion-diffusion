"""
Multi-GPU (single-node) DDP version of train_full_region.py.

All hyperparameters live in a JSON config (default: configs/train_full_region_ddp.json);
see configs/README.md for what each field means. Launch with torchrun. Example
for 4 GPUs on one node:

    torchrun --standalone --nproc_per_node=4 train_full_region_ddp.py

For 2 GPUs, with a custom config:

    torchrun --standalone --nproc_per_node=2 train_full_region_ddp.py --config configs/my_experiment_ddp.json

NOTE: `training.per_gpu_batch_size` is PER-GPU. With N GPUs the effective global
batch size is `per_gpu_batch_size * N`. The single-GPU script used a global batch
of 20. To keep the same global batch on 4 GPUs, set per_gpu_batch_size = 5; on
2 GPUs set per_gpu_batch_size = 10. Adjust `training.lr` if you intentionally
grow the global batch (linear LR scaling is a reasonable starting point).
"""

import argparse
import json
from pathlib import Path

from video_diffusion_pytorch.video_diffusion_cross_attention_with_motion_after_only_contour import (
    Unet3D,
    GaussianDiffusion,
)
from video_diffusion_pytorch.config_snapshot import save_config_snapshot
from video_diffusion_pytorch.trainer_ddp import (
    DDPTrainer,
    setup_distributed,
    cleanup_distributed,
    is_main_process,
)

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "train_full_region_ddp.json"


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
    ).to(device)

    base_path = Path(data_cfg["base_path"])

    trainer = DDPTrainer(
        diffusion_model=diffusion,
        input_video_folder=str(base_path / data_cfg["input_video_subdir"]),
        local_rank=local_rank,
        contour_condition_video_dir=str(base_path / data_cfg["contour_condition_subdir"]),
        sampling_contour_condition_video_dir=str(base_path / data_cfg["sampling_contour_condition_subdir"]),
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
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
