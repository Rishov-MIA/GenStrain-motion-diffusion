"""Single-GPU Group-DRO training for the contour-conditioned displacement model.

All hyperparameters live in a JSON config (default: configs/train_group_dro.json).
Run a different experiment by pointing at another config:

    python train_group_dro.py --config configs/my_experiment.json

See configs/README.md for what each field means, including the dro.eta_q
tuning guidance.
"""

import argparse
import json
from pathlib import Path

import torch

from video_diffusion_pytorch.config_snapshot import save_config_snapshot
from video_diffusion_pytorch.trainer_group_dro import GroupDROTrainer
from video_diffusion_pytorch.video_diffusion_cross_attention_with_motion_after_only_contour import (
    GaussianDiffusion,
    Unet3D,
)

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "train_group_dro.json"


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

    # Fill any {eta_q} / {adjustment_c} / {bs} placeholders in the experiment name
    # with the actual hyperparameters, so each (eta_q, adjustment_c, batch_size)
    # sweep writes to its own checkpoint folder / wandb run. Nothing on disk is
    # mutated. {bs} is the (single-GPU) training batch size.
    cfg["experiment_name"] = cfg["experiment_name"].format(
        eta_q=dro_cfg["eta_q"],
        adjustment_c=dro_cfg["adjustment_c"],
        bs=train_cfg["batch_size"],
    )

    save_config_snapshot(cfg, cfg["experiment_name"])

    if not torch.cuda.is_available():
        raise RuntimeError("train_group_dro.py requires a CUDA GPU")

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

    trainer = GroupDROTrainer(
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
        dro_eta_q=dro_cfg["eta_q"],
        dro_adjustment_c=dro_cfg["adjustment_c"],
        dro_freeze_q=dro_cfg["freeze_q"],
        allowed_disease_groups=dro_cfg.get("allowed_disease_groups"),
        weight_decay=train_cfg["weight_decay"],
        num_workers=data_cfg["num_workers"],
    )

    if train_cfg["resume"]:
        try:
            trainer.load(milestone=-1)
        except AssertionError:
            print("no checkpoint found, starting from scratch")

    trainer.train()


if __name__ == "__main__":
    main()
