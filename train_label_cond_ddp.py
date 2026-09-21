"""Multi-GPU (single-node) DDP version of train_label_cond.py.

    torchrun --standalone --nproc_per_node=4 train_label_cond_ddp.py
    torchrun --standalone --nproc_per_node=2 train_label_cond_ddp.py --config configs/my_label_cond_ddp.json

`training.per_gpu_batch_size` is PER-GPU; the effective global batch is
per_gpu_batch_size * N GPUs. See train_label_cond.py for the `labels` config
block and configs/README.md for every other field. Checkpoints are
interchangeable with the single-GPU script and with inference_label_cond.py.
"""

import argparse
import json
from pathlib import Path

from train_label_cond import build_encoder, build_model, resolve_experiment_name
from video_diffusion_pytorch.config_snapshot import save_config_snapshot
from video_diffusion_pytorch.trainer_ddp import cleanup_distributed, is_main_process, setup_distributed
from video_diffusion_pytorch.trainer_label_cond_ddp import LabelCondDDPTrainer

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "train_label_cond_ddp.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG, help="path to the JSON config (see configs/)")
    return parser.parse_args()


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

    # {bs} = per-GPU batch here, matching the Group-DRO DDP convention.
    resolve_experiment_name(cfg, batch_size=train_cfg["per_gpu_batch_size"])
    local_rank, global_rank, world_size, device = setup_distributed()
    if is_main_process():
        print(f"experiment: {cfg['experiment_name']}")
        save_config_snapshot(cfg, cfg["experiment_name"])

    encoder = build_encoder(label_cfg)
    diffusion = build_model(model_cfg, diffusion_cfg, label_cfg, encoder).to(device)

    base_path = Path(data_cfg["base_path"])
    sampling_base_path = Path(data_cfg.get("sampling_base_path", str(base_path)))

    trainer = LabelCondDDPTrainer(
        diffusion_model=diffusion,
        input_video_folder=str(base_path / data_cfg["input_video_subdir"]),
        local_rank=local_rank,
        label_encoder=encoder,
        preview_label_mode=label_cfg.get("preview_mode", "full"),
        contour_condition_video_dir=str(base_path / data_cfg["contour_condition_subdir"]),
        sampling_contour_condition_video_dir=str(sampling_base_path / data_cfg["sampling_contour_condition_subdir"]),
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
        num_workers=data_cfg.get("num_workers", 4),
    )

    if train_cfg.get("resume", True):
        try:
            trainer.load(milestone=-1)
        except AssertionError:
            if is_main_process():
                print("no checkpoint found, starting from scratch")

    try:
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
