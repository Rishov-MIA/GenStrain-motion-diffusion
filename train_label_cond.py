"""Label-conditioned training (mask video + disease/level/site embeddings).

Same full-region-noise, contour-conditioned model as train_full_region.py,
plus three categorical conditions from processed_excel.json fed through the
UNet's cond_dim path with classifier-free dropout (see
video_diffusion_pytorch/label_cond.py). All hyperparameters live in a JSON
config (default: configs/train_label_cond.json):

    python train_label_cond.py --config configs/train_label_cond.json

The experiment name may carry placeholders that are filled from the config
(see resolve_experiment_name), e.g. "proposal-26frames-label-cond-rot{rot_w}"
becomes ...-rot0.0 / ...-rot0.01, so each ablation setting gets its own folder.

Config additions over train_full_region.json:

    model.cond_dim          embedding width shared by the three labels (e.g. 64)
    labels.metadata_json    processed_excel.json with disease/site per slice
    labels.null_cond_prob   whole-sample label dropout: trains the unconditional path (0.2)
    labels.null_label_prob  per-label independent dropout: trains partial modes such as
                            level_only, the deployment default (0.15)
    labels.use_site         also embed the acquisition site (true)
    labels.preview_mode     labels used for training-time previews: full | level_only | none
    diffusion.rotation_loss_weight    weight of the rotation-curve auxiliary loss; 0 = off (default)
    diffusion.rotation_loss_snr_clip  min-SNR clip applied to that loss per timestep (5.0)

Single GPU. Checkpoints are stamped with the label vocabulary and refuse to
load under a different one.
"""

import argparse
import json
from pathlib import Path

from video_diffusion_pytorch.config_snapshot import save_config_snapshot
from video_diffusion_pytorch.label_cond import (
    LabelCondGaussianDiffusion,
    LabelCondTrainer,
    LabelCondUnet3D,
)
from video_diffusion_pytorch.label_metadata import LabelEncoder

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "train_label_cond.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="path to the JSON config (see configs/)")
    return parser.parse_args()


def resolve_experiment_name(cfg, batch_size=None):
    """Fill the placeholders in cfg["experiment_name"] from the config values, so
    every setting of the ablation writes to its own checkpoint folder / wandb run
    (same convention as train_group_dro.py). Available placeholders:

        {rot_w}     diffusion.rotation_loss_weight     {snr_clip}  diffusion.rotation_loss_snr_clip
        {cno}       diffusion.contour_noise_only (0/1) {cond_dim}  model.cond_dim
        {null_p}    labels.null_cond_prob              {label_p}   labels.null_label_prob
        {site}      labels.use_site (0/1)              {bs}        training batch size (per GPU under DDP)

    A name without placeholders is returned unchanged. The inference script
    resolves the same template from its own config, so the two configs must
    carry the same values for the placeholders they use.
    """
    diffusion_cfg, label_cfg, train_cfg = cfg["diffusion"], cfg["labels"], cfg["training"]
    if batch_size is None:
        batch_size = train_cfg.get("batch_size", train_cfg.get("per_gpu_batch_size"))
    name = cfg["experiment_name"].format(
        rot_w=diffusion_cfg.get("rotation_loss_weight", 0.0),
        snr_clip=diffusion_cfg.get("rotation_loss_snr_clip", 5.0),
        cno=int(bool(diffusion_cfg.get("contour_noise_only", False))),
        cond_dim=cfg["model"]["cond_dim"],
        null_p=label_cfg.get("null_cond_prob", 0.2),
        label_p=label_cfg.get("null_label_prob", 0.15),
        site=int(bool(label_cfg.get("use_site", True))),
        bs=batch_size,
    )
    cfg["experiment_name"] = name
    return name


def build_encoder(label_cfg):
    return LabelEncoder(metadata_json_path=label_cfg["metadata_json"], use_site=label_cfg.get("use_site", True))


def build_model(model_cfg, diffusion_cfg, label_cfg, encoder):
    unet = LabelCondUnet3D(
        dim=model_cfg["dim"],
        cond_dim=model_cfg["cond_dim"],
        channels=model_cfg["channels"],
        dim_mults=tuple(model_cfg["dim_mults"]),
        label_vocab_sizes=encoder.vocab_sizes,
    )
    diffusion = LabelCondGaussianDiffusion(
        unet,
        image_size=diffusion_cfg["image_size"],
        channels=diffusion_cfg["channels"],
        num_frames=diffusion_cfg["num_frames"],
        timesteps=diffusion_cfg["timesteps"],
        loss_type=diffusion_cfg["loss_type"],
        contour_noise_only=diffusion_cfg["contour_noise_only"],
        disp_rel_metric=diffusion_cfg.get("disp_rel_metric", "mse"),
        disp_metrics=diffusion_cfg.get("disp_metrics", True),
        disp_scale=diffusion_cfg.get("disp_scale", 5.0),
        null_cond_prob=label_cfg.get("null_cond_prob", 0.2),
        null_label_prob=label_cfg.get("null_label_prob", 0.15),
        # Rotation auxiliary loss; 0 (the default) leaves the objective as the
        # plain eps-loss. See LabelCondGaussianDiffusion.
        rotation_loss_weight=diffusion_cfg.get("rotation_loss_weight", 0.0),
        rotation_loss_snr_clip=diffusion_cfg.get("rotation_loss_snr_clip", 5.0),
    )
    return diffusion


def main():
    args = parse_args()
    with args.config.open() as f:
        cfg = json.load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    diffusion_cfg = cfg["diffusion"]
    train_cfg = cfg["training"]
    label_cfg = cfg["labels"]
    log_cfg = cfg.get("logging", {})
    fv_cfg = cfg.get("frame_validity") or {}

    resolve_experiment_name(cfg, batch_size=train_cfg["batch_size"])
    print(f"experiment: {cfg['experiment_name']}")
    save_config_snapshot(cfg, cfg["experiment_name"])

    encoder = build_encoder(label_cfg)
    diffusion = build_model(model_cfg, diffusion_cfg, label_cfg, encoder).cuda()

    base_path = Path(data_cfg["base_path"])
    trainer = LabelCondTrainer(
        diffusion_model=diffusion,
        input_video_folder=str(base_path / data_cfg["input_video_subdir"]),
        contour_condition_video_dir=str(base_path / data_cfg["contour_condition_subdir"]),
        sampling_contour_condition_video_dir=str(base_path / data_cfg["sampling_contour_condition_subdir"]),
        label_encoder=encoder,
        preview_label_mode=label_cfg.get("preview_mode", "full"),
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

    if train_cfg.get("resume", True):
        try:
            trainer.load(milestone=-1)
        except AssertionError:
            print("no checkpoint found, starting from scratch")

    trainer.configure_preview_sampler(
        sampler=train_cfg.get("preview_sampler", "ddpm"),
        ddim_steps=train_cfg.get("preview_ddim_steps", 50),
        ddim_eta=train_cfg.get("preview_ddim_eta", 0.0),
        spacing=train_cfg.get("preview_ddim_spacing", "uniform"),
        clip_x_start=train_cfg.get("preview_ddim_clip_x_start", None),
        clip_percentile=train_cfg.get("preview_ddim_clip_percentile", 0.995),
    )

    trainer.train()


if __name__ == "__main__":
    main()
