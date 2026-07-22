"""Full-region-noise training on augmented data, conditioned on the contour mask only.

Same as train_full_region.py, but the dataset lives under an augmentation-count
folder (aug_1x, aug_2x, ... aug_10x). Pass the count with --aug; it is joined onto
base_path via the config's `aug_subdir` template and substituted for any "{aug}"
placeholders in the config (e.g. in experiment_name), so nothing on disk is mutated:

    python train_full_region_augmented.py --aug 5
    python train_full_region_augmented.py --aug 5 --config configs/my_experiment.json

With --aug 5 the data root becomes
    {base_path}/aug_5x/dense/train/displacement_dense   (etc.)

See configs/README.md for what each field means.
"""

import argparse
import json
from pathlib import Path

from video_diffusion_pytorch.config_snapshot import save_config_snapshot
from video_diffusion_pytorch.video_diffusion_cross_attention_with_motion_after_only_contour import (
    Unet3D,
    GaussianDiffusion,
    Trainer,
)

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "train_full_region_augmented.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--aug",
        type=int,
        required=True,
        help="augmentation count N, selecting the aug_Nx dataset folder (e.g. 5 -> aug_5x)",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="path to the JSON hyperparameter config (see configs/)",
    )
    return parser.parse_args()


def substitute_aug(value, aug):
    """Recursively replace the "{aug}" placeholder with `aug` in all string values."""
    if isinstance(value, str):
        return value.format(aug=aug)
    if isinstance(value, dict):
        return {k: substitute_aug(v, aug) for k, v in value.items()}
    if isinstance(value, list):
        return [substitute_aug(v, aug) for v in value]
    return value


def main():
    args = parse_args()
    with args.config.open() as f:
        cfg = json.load(f)

    # Bake the augmentation count into every "{aug}" placeholder in the config
    # (data paths, experiment_name, ...). Nothing on disk is changed.
    cfg = substitute_aug(cfg, args.aug)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    diffusion_cfg = cfg["diffusion"]
    train_cfg = cfg["training"]
    log_cfg = cfg.get("logging", {})
    # Optional per-sample valid-frame loss masking (see configs/README.md).
    # Absent/empty block => masking off (original all-frames loss).
    fv_cfg = cfg.get("frame_validity") or {}

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
    ).cuda()

    # aug_subdir (e.g. "aug_5x") sits between the dataset root and the split subdirs.
    data_root = Path(data_cfg["base_path"]) / data_cfg["aug_subdir"]

    # Sampling conditions can come from a separate (non-augmented) dataset root.
    # Falls back to the augmented data_root when sampling_base_path is absent.
    sampling_root = Path(data_cfg.get("sampling_base_path", str(data_root)))

    trainer = Trainer(
        diffusion_model=diffusion,
        input_video_folder=str(data_root / data_cfg["input_video_subdir"]),
        contour_condition_video_dir=str(data_root / data_cfg["contour_condition_subdir"]),
        sampling_contour_condition_video_dir=str(sampling_root / data_cfg["sampling_contour_condition_subdir"]),
        train_batch_size=train_cfg["batch_size"],
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
    )

    if train_cfg["resume"]:
        try:
            trainer.load(milestone=-1)
        except AssertionError:
            print("no checkpoint found, starting from scratch")

    trainer.train()


if __name__ == "__main__":
    main()
