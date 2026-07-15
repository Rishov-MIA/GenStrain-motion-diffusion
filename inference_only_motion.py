"""Inference for the motion-only-conditioned displacement model.

All settings live in a JSON config (default: configs/inference_only_motion.json).
Run a different setup by pointing at another config:

    python inference_only_motion.py --config configs/my_inference.json

The per-run runtime knobs (video_type, start, end, milestone) default to the
config's `inference` block but can be overridden on the CLI:

    python inference_only_motion.py --video_type dense --start 0 --end 50

See configs/README.md for what each field means.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from video_diffusion_pytorch.config_snapshot import save_config_snapshot
from video_diffusion_pytorch.video_diffusion_cross_attention_with_motion_only import (
    GaussianDiffusion,
    Trainer,
    Unet3D,
)

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "inference_only_motion.json"


def count_condition_videos(cond_video_dir, start, end):
    """Number of files the sampling loop will process — mirrors the generator's
    sorted-glob + [start:end] slice — so the progress bar knows its total."""
    n = len(sorted(Path(cond_video_dir).glob("*.npy")))
    if end is None:
        end = n
    return len(range(n)[start:end])


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=DEFAULT_CONFIG,
        help="path to the JSON inference config (see configs/)",
    )
    # Runtime knobs. Default is None so we can tell "not passed" from an
    # explicit value and fall back to the config's inference block.
    parser.add_argument('--video_type', type=str, default=None, help='video type: cine and dense')
    parser.add_argument('--start', type=int, default=None, help='start index')
    parser.add_argument('--end', type=int, default=None, help='end index')
    parser.add_argument('--milestone', type=int, default=None, help='checkpoint milestone (-1 = latest)')
    parser.add_argument('--no_config_snapshot', action='store_true',
                        help='skip writing the config snapshot (set by the multi-GPU launcher on all but one worker to avoid a concurrent-write race)')
    return parser.parse_args()


def normalize_divide_by_5(x: torch.Tensor) -> torch.Tensor:
    return x / 5.0


def pick_condition_videos_one_video_at_once(motion_cond_video_dir, start, end):
    motion_cond_video_dir = Path(motion_cond_video_dir)
    motion_cond_video_paths = sorted(list(motion_cond_video_dir.glob("*.npy")))

    if end is None:
        end = len(motion_cond_video_paths)

    indices = range(len(motion_cond_video_paths))

    for idx in indices[start:end]:
        motion_cond_video_path = motion_cond_video_paths[idx]
        cond_filename = f"{motion_cond_video_path.stem}.npy"
        print(cond_filename)

        motion_cond = np.load(motion_cond_video_path)
        motion_cond_tensor = torch.from_numpy(motion_cond).float()  # shape [2, F, H, W] or [1, 2, F, H, W]

        # Ensure shape is [1, 2, F, H, W] (batch=1, 2 channels, F frames, H, W)
        if motion_cond_tensor.dim() == 4:
            motion_cond_tensor = motion_cond_tensor.unsqueeze(0)

        yield motion_cond_tensor, [cond_filename]


def main():
    args = parse_args()
    with args.config.open() as f:
        cfg = json.load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    diffusion_cfg = cfg["diffusion"]
    train_cfg = cfg["training"]
    infer_cfg = cfg.get("inference", {})

    # CLI overrides the config's inference block; the config supplies defaults.
    video_type = args.video_type if args.video_type is not None else infer_cfg.get("video_type", "cine")
    start = args.start if args.start is not None else infer_cfg.get("start", 0)
    end = args.end if args.end is not None else infer_cfg.get("end", None)
    milestone = args.milestone if args.milestone is not None else infer_cfg.get("milestone", -1)

    if not args.no_config_snapshot:
        save_config_snapshot(cfg, cfg["experiment_name"])

    if not torch.cuda.is_available():
        raise RuntimeError("inference_only_motion.py requires a CUDA GPU")

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
    ).cuda()

    base_path = Path(data_cfg["base_path"])

    trainer = Trainer(
        diffusion_model=diffusion,
        input_video_folder=str(base_path / data_cfg["input_video_subdir"]),
        motion_condition_video_dir=str(base_path / data_cfg["motion_condition_subdir"]),
        sampling_motion_condition_video_dir=str(base_path / data_cfg["sampling_motion_condition_subdir"]),
        train_batch_size=train_cfg["batch_size"],
        train_lr=train_cfg["lr"],
        save_and_sample_every=train_cfg["save_and_sample_every"],
        train_num_steps=train_cfg["num_steps"],
        gradient_accumulate_every=train_cfg["gradient_accumulate_every"],
        ema_decay=train_cfg["ema_decay"],
        amp=train_cfg["amp"],
        experiment_name=cfg["experiment_name"],
        inference_only=True,
    )

    # Load the requested checkpoint (milestone = -1 → latest)
    trainer.load(milestone=milestone)

    motion_test_subdir = data_cfg["motion_test_subdir_template"].format(video_type=video_type)
    motion_cond_video_dir = str(base_path / motion_test_subdir)
    video_generator = pick_condition_videos_one_video_at_once(motion_cond_video_dir, start, end)
    total_videos = count_condition_videos(motion_cond_video_dir, start, end)

    device = next(trainer.ema_model.parameters()).device
    custom_save_folder = f"./{cfg['experiment_name']}/sampling_time_sampled_{video_type}_part_videos_infos/"
    os.makedirs(custom_save_folder, exist_ok=True)

    for motion_cond_video, cond_filename in tqdm(video_generator, total=total_videos, desc="videos", unit="video", position=0):
        motion_cond_video = normalize_divide_by_5(motion_cond_video).to(device)

        # Generate samples
        if video_type == "cine":
            reconstructed_disp = None
            trainer.sample_and_save_one_video_at_a_time("final", motion_cond_video, cond_filename, reconstructed_disp, None, save_folder=custom_save_folder)
        elif video_type == "dense":
            gt_disp_dir = str(base_path / data_cfg["gt_disp_dense_subdir"])
            gt_disp = np.load(os.path.join(gt_disp_dir, cond_filename[0]))
            reconstructed_disp = None
            trainer.sample_and_save_one_video_at_a_time("final", motion_cond_video, cond_filename, reconstructed_disp, gt_disp, save_folder=custom_save_folder)


if __name__ == "__main__":
    main()
