"""Single-GPU inverse-frequency reweighting for the contour-conditioned model.

The group-robustness baseline that sits beside Group DRO. Batches are drawn from
the natural training distribution, and each sample's loss is scaled by a STATIC
weight inversely proportional to its disease group's size:

    w_g = N / (K * n_g)

so every group contributes equally to the objective regardless of how many
samples it has. Unlike Group DRO the weights never change during training. See
video_diffusion_pytorch/trainer_inverse_freq.py for the derivation.

All hyperparameters live in a JSON config (default: configs/train_inverse_freq.json).
Run a different experiment by pointing at another config:

    python train_inverse_freq.py --config configs/my_experiment.json

See configs/README.md for what each field means.
"""

import argparse
import json
from pathlib import Path

import torch

from video_diffusion_pytorch.config_snapshot import save_config_snapshot
from video_diffusion_pytorch.trainer_inverse_freq import InverseFrequencyTrainer
from video_diffusion_pytorch.video_diffusion_cross_attention_with_motion_after_only_contour import (
    GaussianDiffusion,
    Unet3D,
)

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "train_inverse_freq.json"


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

    # Fill any {bs} placeholder in the experiment name with the actual batch size,
    # so each sweep point writes to its own checkpoint folder / wandb run. Nothing
    # on disk is mutated. There are no reweighting hyperparameters to sweep — the
    # weights are fully determined by the dataset's group counts.
    cfg["experiment_name"] = cfg["experiment_name"].format(bs=train_cfg["batch_size"])

    save_config_snapshot(cfg, cfg["experiment_name"])

    if not torch.cuda.is_available():
        raise RuntimeError("train_inverse_freq.py requires a CUDA GPU")

    model = Unet3D(
        dim=model_cfg["dim"],
        cond_dim=model_cfg["cond_dim"],
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

    # Sampling conditions can come from a separate dataset root via the optional
    # sampling_base_path config field. Falls back to base_path when absent.
    sampling_base_path = Path(data_cfg.get("sampling_base_path", str(base_path)))

    trainer = InverseFrequencyTrainer(
        diffusion_model=diffusion,
        input_video_folder=str(base_path / data_cfg["input_video_subdir"]),
        contour_condition_video_dir=str(base_path / data_cfg["contour_condition_subdir"]),
        sampling_contour_condition_video_dir=str(sampling_base_path / data_cfg["sampling_contour_condition_subdir"]),
        metadata_json_path=data_cfg["metadata_json_path"],
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
        allowed_disease_groups=reweight_cfg.get("allowed_disease_groups"),
        weight_decay=train_cfg["weight_decay"],
        num_workers=data_cfg["num_workers"],
    )

    if train_cfg["resume"]:
        try:
            trainer.load(milestone=-1)
        except AssertionError:
            print("no checkpoint found, starting from scratch")

    # Sampler for the periodic preview samples taken during training.

    # Defaults to ddpm, i.e. unchanged behaviour (see configs/README.md).

    trainer.configure_preview_sampler(

        sampler=train_cfg.get("preview_sampler", "ddpm"),

        ddim_steps=train_cfg.get("preview_ddim_steps", 50),

        ddim_eta=train_cfg.get("preview_ddim_eta", 0.0),

    )

    trainer.train()


if __name__ == "__main__":
    main()
