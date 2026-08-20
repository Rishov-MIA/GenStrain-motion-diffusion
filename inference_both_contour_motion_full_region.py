"""Inference for the contour+motion-conditioned, full-region-noise model.

All settings live in a JSON config
(default: configs/inference_both_contour_motion_full_region.json). Run a
different setup by pointing at another config:

    python inference_both_contour_motion_full_region.py --config configs/my_inference.json

The per-run runtime knobs (video_type, start, end, milestone, sampler, ddim_steps, ddim_eta)
default to the config's `inference` block but can be overridden on the CLI:

    python inference_both_contour_motion_full_region.py --video_type dense --start 0 --end 50

--skip_existing resumes an interrupted run: any video whose prediction is
already in the experiment's inference_disps/ folder is skipped.

    python inference_both_contour_motion_full_region.py --skip_existing

See configs/README.md for what each field means.
"""

import argparse
import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from video_diffusion_pytorch.completed_outputs import (
    completed_output_names,
    inference_output_dir,
    select_condition_paths,
)
from video_diffusion_pytorch.config_snapshot import save_config_snapshot
from video_diffusion_pytorch.video_diffusion_cross_attention_with_both_contour_motion import (
    GaussianDiffusion,
    normalize_disp,
    Trainer,
    Unet3D,
)

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "inference_both_contour_motion_full_region.json"


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
    parser.add_argument('--sampler', type=str, default=None, choices=['ddpm', 'ddim'],
                        help="sampler: 'ddpm' = full-chain ancestral sampling (default), "
                             "'ddim' = strided, uses ddim_steps denoise calls instead of diffusion.timesteps")
    parser.add_argument('--ddim_steps', type=int, default=None,
                        help='number of DDIM steps; ignored unless --sampler ddim')
    parser.add_argument('--ddim_eta', type=float, default=None,
                        help='DDIM stochasticity, 0.0 = deterministic; ignored unless --sampler ddim')
    parser.add_argument('--ddim_spacing', type=str, default=None, choices=['uniform', 'logsnr'],
                        help="how DDIM picks its timestep subsequence: 'uniform' in t (default) or "
                             "'logsnr', which spends steps where the noise level actually moves")
    parser.add_argument('--ddim_clip_x_start', type=str, default=None,
                        help="bound the predicted x0 each DDIM step: 'none' (default), 'dynamic' "
                             "(per-sample percentile clip), or a float for a fixed [-v, v] clamp")
    parser.add_argument('--ddim_clip_percentile', type=float, default=None,
                        help='percentile for --ddim_clip_x_start dynamic (default 0.995)')
    parser.add_argument('--ddim_start_t', type=int, default=None,
                        help='start DDIM at this timestep instead of timesteps-1. t=999 has a ~64,000x '
                             'x0 amplification vs ~642x at t=998, so 998 is a good value')
    parser.add_argument('--seed', type=int, default=None,
                        help='seed the initial noise x_T so runs are comparable; omit for random')
    parser.add_argument('--skip_existing', action='store_true',
                        help='resume: skip any video whose prediction is already in the experiment\'s '
                             'inference_disps/ folder (truncated files are re-sampled)')
    parser.add_argument('--no_config_snapshot', action='store_true',
                        help='skip writing the config snapshot (set by the multi-GPU launcher on all but one worker to avoid a concurrent-write race)')
    return parser.parse_args()


def normalize_cond_img(t):
    return t / 255.0


def pick_condition_videos_one_video_at_once(contour_cond_video_paths, motion_cond_video_dir):
    motion_cond_video_dir = Path(motion_cond_video_dir)

    for contour_cond_video_path in contour_cond_video_paths:
        cond_filename = f"{contour_cond_video_path.stem}.npy"
        print(cond_filename)

        contour_cond = np.load(contour_cond_video_path)
        contour_cond_tensor = torch.from_numpy(contour_cond).float()  # shape [1, F, H, W]

        motion_cond_video_path = motion_cond_video_dir / cond_filename
        motion_cond = np.load(motion_cond_video_path)
        motion_cond_tensor = torch.from_numpy(motion_cond).float()  # shape [1, 2, F, H, W] — batch dim already included

        yield contour_cond_tensor.unsqueeze(0), motion_cond_tensor, [cond_filename]


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
    sampler = args.sampler if args.sampler is not None else infer_cfg.get("sampler", "ddpm")
    ddim_steps = args.ddim_steps if args.ddim_steps is not None else infer_cfg.get("ddim_steps", 50)
    ddim_eta = args.ddim_eta if args.ddim_eta is not None else infer_cfg.get("ddim_eta", 0.0)
    ddim_spacing = args.ddim_spacing if args.ddim_spacing is not None else infer_cfg.get("ddim_spacing", "uniform")
    ddim_clip_percentile = (args.ddim_clip_percentile if args.ddim_clip_percentile is not None
                            else infer_cfg.get("ddim_clip_percentile", 0.995))
    seed = args.seed if args.seed is not None else infer_cfg.get("seed", None)
    skip_existing = args.skip_existing or infer_cfg.get("skip_existing", False)
    ddim_start_t = args.ddim_start_t if args.ddim_start_t is not None else infer_cfg.get("ddim_start_t", None)
    # "none"/null -> no clipping; "dynamic" -> percentile clip; anything else -> a float bound.
    ddim_clip_x_start = (args.ddim_clip_x_start if args.ddim_clip_x_start is not None
                         else infer_cfg.get("ddim_clip_x_start", None))
    if isinstance(ddim_clip_x_start, str) and ddim_clip_x_start.lower() in ("none", "null", ""):
        ddim_clip_x_start = None
    elif isinstance(ddim_clip_x_start, str) and ddim_clip_x_start != "dynamic":
        ddim_clip_x_start = float(ddim_clip_x_start)

    if not args.no_config_snapshot:
        save_config_snapshot(cfg, cfg["experiment_name"])

    if not torch.cuda.is_available():
        raise RuntimeError("inference_both_contour_motion_full_region.py requires a CUDA GPU")

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
        disp_scale=diffusion_cfg.get("disp_scale", 5.0),
    ).cuda()

    base_path = Path(data_cfg["base_path"])

    trainer = Trainer(
        diffusion_model=diffusion,
        input_video_folder=str(base_path / data_cfg["input_video_subdir"]),
        contour_condition_video_dir=str(base_path / data_cfg["contour_condition_subdir"]),
        motion_condition_video_dir=str(base_path / data_cfg["motion_condition_subdir"]),
        sampling_contour_condition_video_dir=str(base_path / data_cfg["train_sampling_contour_condition_subdir"]),
        sampling_motion_condition_video_dir=str(base_path / data_cfg["train_sampling_motion_condition_subdir"]),
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

    contour_test_subdir = data_cfg["sampling_contour_condition_subdir"].format(video_type=video_type)
    motion_test_subdir = data_cfg["sampling_motion_condition_subdir"].format(video_type=video_type)
    contour_cond_video_dir = str(base_path / contour_test_subdir)
    motion_cond_video_dir = str(base_path / motion_test_subdir)

    # --skip_existing: a prediction already sitting in the output folder means an
    # earlier run finished that video, so drop it from this run's work list.
    output_dir = inference_output_dir(cfg["experiment_name"], video_type)
    done_names = completed_output_names(output_dir) if skip_existing else None
    cond_video_paths = select_condition_paths(contour_cond_video_dir, start, end, done_names)
    if skip_existing:
        print(f"skip_existing: {len(done_names)} prediction(s) already in {output_dir}; "
              f"{len(cond_video_paths)} video(s) left to sample in [{start}, {end})")
        if not cond_video_paths:
            return

    video_generator = pick_condition_videos_one_video_at_once(cond_video_paths, motion_cond_video_dir)
    total_videos = len(cond_video_paths)

    device = next(trainer.ema_model.parameters()).device
    custom_save_folder = f"./{cfg['experiment_name']}/sampling_time_sampled_{video_type}_part_videos_infos/"
    os.makedirs(custom_save_folder, exist_ok=True)

    for contour_cond_video, motion_cond_video, cond_filename in tqdm(video_generator, total=total_videos, desc="videos", unit="video", position=0):
        contour_cond_video = normalize_cond_img(contour_cond_video).to(device)
        motion_cond_video = normalize_disp(motion_cond_video, diffusion.disp_scale).to(device)

        # Generate samples
        if video_type == "cine":
            reconstructed_disp = None
            trainer.sample_and_save_one_video_at_a_time("final", contour_cond_video, motion_cond_video, cond_filename, reconstructed_disp, None, save_folder=custom_save_folder, sampler=sampler, ddim_steps=ddim_steps, ddim_eta=ddim_eta, ddim_spacing=ddim_spacing, ddim_clip_x_start=ddim_clip_x_start, ddim_clip_percentile=ddim_clip_percentile, ddim_start_t=ddim_start_t, seed=seed)
        elif video_type == "dense":
            gt_disp_dir = str(base_path / data_cfg["gt_disp_dense_subdir"])
            gt_disp = np.load(os.path.join(gt_disp_dir, cond_filename[0]))
            reconstructed_disp = None
            trainer.sample_and_save_one_video_at_a_time("final", contour_cond_video, motion_cond_video, cond_filename, reconstructed_disp, gt_disp, save_folder=custom_save_folder, sampler=sampler, ddim_steps=ddim_steps, ddim_eta=ddim_eta, ddim_spacing=ddim_spacing, ddim_clip_x_start=ddim_clip_x_start, ddim_clip_percentile=ddim_clip_percentile, ddim_start_t=ddim_start_t, seed=seed)


if __name__ == "__main__":
    main()
