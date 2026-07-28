"""
Multi-GPU (single-node) DDP version of train_inverse_freq.py.

Inverse-frequency reweighting for the contour-conditioned displacement diffusion
model — the group-robustness baseline that sits beside Group DRO. Each step draws
a batch from the NATURAL training distribution (a plain DistributedSampler shard,
so the batch keeps its natural group mixture) and scales each sample's loss by a
STATIC weight inversely proportional to its disease group's size:

    w_g = N / (K * n_g)

so every group contributes equally to the objective. The weights never change
during training — that is the whole contrast with Group DRO, which re-estimates
its weights every step. See video_diffusion_pytorch/trainer_inverse_freq_ddp.py
for the DDP details, in particular why the weighted mean is normalized over the
GLOBAL batch rather than per rank.

Groups are the `disease` field from the metadata JSON, restricted to
`reweight.allowed_disease_groups` (substring match in priority order, so
"LBBB & Recovered DCM" -> LBBB).

All hyperparameters live in a JSON config (default: configs/train_inverse_freq_ddp.json);
see configs/README.md for what each field means. Launch with torchrun. Example
for 4 GPUs on one node:

    torchrun --standalone --nproc_per_node=4 train_inverse_freq_ddp.py

For 2 GPUs, with a custom config:

    torchrun --standalone --nproc_per_node=2 train_inverse_freq_ddp.py --config configs/my_experiment_ddp.json

NOTE: `training.per_gpu_batch_size` is PER-GPU. With N GPUs the effective global
batch is `per_gpu_batch_size * N`. Scale `training.lr` if you grow the global
batch (linear scaling is a reasonable start). The reweighting itself has no
hyperparameters to rescale — the weights follow from the dataset's group counts.
"""

import argparse
import json
from pathlib import Path

from video_diffusion_pytorch.video_diffusion_cross_attention_with_motion_after_only_contour import (
    Unet3D,
    GaussianDiffusion,
)
from video_diffusion_pytorch.config_snapshot import save_config_snapshot
from video_diffusion_pytorch.trainer_inverse_freq_ddp import (
    InverseFrequencyDDPTrainer,
    setup_distributed,
    cleanup_distributed,
    is_main_process,
)

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "train_inverse_freq_ddp.json"


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
    reweight_cfg = cfg["reweight"]
    log_cfg = cfg.get("logging", {})
    # Optional per-sample valid-frame loss masking (see configs/README.md).
    # Absent/empty block => masking off (original all-frames loss).
    fv_cfg = cfg.get("frame_validity") or {}

    # Fill any {bs} placeholder in the experiment name with the per-GPU batch size,
    # so each sweep point writes to its own checkpoint folder / wandb run. Nothing
    # on disk is mutated.
    cfg["experiment_name"] = cfg["experiment_name"].format(bs=train_cfg["per_gpu_batch_size"])

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

    # Sampling conditions can come from a separate dataset root via the optional
    # sampling_base_path config field. Falls back to base_path when absent.
    sampling_base_path = Path(data_cfg.get("sampling_base_path", str(base_path)))

    trainer = InverseFrequencyDDPTrainer(
        diffusion_model=diffusion,
        input_video_folder=str(base_path / data_cfg["input_video_subdir"]),
        local_rank=local_rank,
        contour_condition_video_dir=str(base_path / data_cfg["contour_condition_subdir"]),
        sampling_contour_condition_video_dir=str(sampling_base_path / data_cfg["sampling_contour_condition_subdir"]),
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
        valid_frames_csv=fv_cfg.get("csv_path"),
        valid_frames_filename_col=fv_cfg.get("filename_col", "dense_filename"),
        valid_frames_count_col=fv_cfg.get("valid_frames_col", "dense_valid_frames"),
        valid_frames_missing=fv_cfg.get("missing", "full"),
        num_workers=data_cfg["num_workers"],
        # ---- inverse-frequency reweighting ----
        allowed_disease_groups=reweight_cfg.get("allowed_disease_groups"),
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
