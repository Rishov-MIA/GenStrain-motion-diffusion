"""
Multi-GPU (single-node) DDP version of train_group_dro.py.

Group-DRO training for the contour-conditioned displacement diffusion model.
Each step samples ONE disease group (uniformly, on rank 0 and broadcast to all
ranks), draws a per-rank batch from it, updates that group's weight q_g on the
GLOBAL (all-reduced) group loss, then takes a q_g-weighted model step with weight
decay. See video_diffusion_pytorch/trainer_group_dro_ddp.py for the DDP details.

Groups are the `disease` field from the metadata JSON, restricted to:
    Acute MI, Healthy, Healthy Pediatric, DCM, Myocarditis, LBBB
(substring match in priority order, so "LBBB & Recovered DCM" -> LBBB). Augmented
samples are ignored: the real per-group counts n_g are just the files present.

All hyperparameters live in a JSON config (default: configs/train_group_dro_ddp.json);
see configs/README.md for what each field means. Launch with torchrun. Example
for 4 GPUs on one node:

    torchrun --standalone --nproc_per_node=4 train_group_dro_ddp.py

For 2 GPUs, with a custom config:

    torchrun --standalone --nproc_per_node=2 train_group_dro_ddp.py --config configs/my_experiment_ddp.json

NOTE: `training.per_gpu_batch_size` is PER-GPU. With N GPUs the effective
per-step group batch is `per_gpu_batch_size * N`, all from the one sampled
group. Scale `training.lr` if you grow the global batch (linear scaling is a
reasonable start). The Group-DRO hyperparameters (dro.eta_q, dro.adjustment_c)
act on the global mean loss and do NOT need rescaling with the number of GPUs.
"""

import argparse
import json
from pathlib import Path

from video_diffusion_pytorch.video_diffusion_cross_attention_with_motion_after_only_contour import (
    Unet3D,
    GaussianDiffusion,
)
from video_diffusion_pytorch.config_snapshot import save_config_snapshot
from video_diffusion_pytorch.trainer_group_dro_ddp import (
    GroupDRODDPTrainer,
    setup_distributed,
    cleanup_distributed,
    is_main_process,
)

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "train_group_dro_ddp.json"


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
    dro_cfg = cfg["dro"]
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

    trainer = GroupDRODDPTrainer(
        diffusion_model=diffusion,
        input_video_folder=str(base_path / data_cfg["input_video_subdir"]),
        local_rank=local_rank,
        contour_condition_video_dir=str(base_path / data_cfg["contour_condition_subdir"]),
        sampling_contour_condition_video_dir=str(base_path / data_cfg["sampling_contour_condition_subdir"]),
        metadata_json_path=data_cfg["metadata_json_path"],
        train_batch_size=train_cfg["per_gpu_batch_size"],  # PER-GPU
        train_lr=train_cfg["lr"],
        save_and_sample_every=train_cfg["save_and_sample_every"],
        train_num_steps=train_cfg["num_steps"],
        gradient_accumulate_every=train_cfg["gradient_accumulate_every"],  # must be 1
        ema_decay=train_cfg["ema_decay"],
        amp=train_cfg["amp"],
        experiment_name=cfg["experiment_name"],
        use_wandb=log_cfg.get("use_wandb", False),
        wandb_project=log_cfg.get("wandb_project", "genstrain-motion-diffusion"),
        wandb_config=cfg,
        num_workers=data_cfg["num_workers"],
        # ---- Group-DRO hyperparameters ----
        dro_eta_q=dro_cfg["eta_q"],
        dro_adjustment_c=dro_cfg["adjustment_c"],
        dro_freeze_q=dro_cfg["freeze_q"],
        weight_decay=train_cfg["weight_decay"],
    )

    # Load the latest checkpoint (milestone = -1) to resume; skip if none exists.
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
