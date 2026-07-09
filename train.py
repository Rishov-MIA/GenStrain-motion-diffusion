"""Contour-region-noise training, conditioned on the contour mask only.

All hyperparameters live in a JSON config (default: configs/train.json).
Run a different experiment by pointing at another config:

    python train.py --config configs/my_experiment.json

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

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "train.json"


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

    base_path = Path(data_cfg["base_path"])

    trainer = Trainer(
        diffusion_model=diffusion,
        input_video_folder=str(base_path / data_cfg["input_video_subdir"]),
        contour_condition_video_dir=str(base_path / data_cfg["contour_condition_subdir"]),
        sampling_contour_condition_video_dir=str(base_path / data_cfg["sampling_contour_condition_subdir"]),
        train_batch_size=train_cfg["batch_size"],
        train_lr=train_cfg["lr"],
        save_and_sample_every=train_cfg["save_and_sample_every"],
        train_num_steps=train_cfg["num_steps"],
        gradient_accumulate_every=train_cfg["gradient_accumulate_every"],
        ema_decay=train_cfg["ema_decay"],
        amp=train_cfg["amp"],
        experiment_name=cfg["experiment_name"],
    )

    if train_cfg["resume"]:
        try:
            trainer.load(milestone=-1)
        except AssertionError:
            print("no checkpoint found, starting from scratch")

    trainer.train()


if __name__ == "__main__":
    main()
