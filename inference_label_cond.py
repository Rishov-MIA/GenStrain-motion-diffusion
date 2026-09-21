"""Inference for the label-conditioned displacement model (train_label_cond.py).

    python inference_label_cond.py --config configs/inference_label_cond.json
    python inference_label_cond.py --video_type dense --label_mode level_only --cond_scale 1.5

The label mode decides which conditions the sampler is given (the checkpoint
serves all three, thanks to classifier-free dropout during training):

    full        disease + slice level + site from processed_excel.json
    level_only  slice level only (parsed from the filename); disease/site null.
                The deployment setting: diagnosis unknown, level always known.
    none        all null -> the unconditional path, the mask-only behaviour

cond_scale > 1 amplifies the label's effect by classifier-free guidance
(forward_with_cond_scale). With label_mode none it has no effect.

Two free post-processing fixes are on by default, both pure error under
StrainAnalysis' epe_disp_rel scoring: the frame-0 field is zeroed (Lagrangian
displacement is zero there by definition) and, when a frame-count CSV is
configured for the video type, frames past the valid count are zeroed.

--skip_existing resumes an interrupted run. Multi-GPU: run_inference_multi_gpu.py
can launch this script the same way as inference_full_region.py.
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
from video_diffusion_pytorch.frame_validity import load_valid_frames_map
from video_diffusion_pytorch.label_cond import LabelCondTrainer
from video_diffusion_pytorch.label_metadata import LABEL_MODES
from train_label_cond import build_encoder, build_model, resolve_experiment_name

DEFAULT_CONFIG = Path(__file__).resolve().parent / "configs" / "inference_label_cond.json"


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--video_type", type=str, default=None, help="cine or dense")
    parser.add_argument("--start", type=int, default=None)
    parser.add_argument("--end", type=int, default=None)
    parser.add_argument("--milestone", type=int, default=None, help="checkpoint milestone (-1 = latest)")
    parser.add_argument("--label_mode", type=str, default=None, choices=LABEL_MODES)
    parser.add_argument("--cond_scale", type=float, default=None, help="classifier-free guidance scale (1.0 = off)")
    parser.add_argument("--sampler", type=str, default=None, choices=["ddpm", "ddim"])
    parser.add_argument("--ddim_steps", type=int, default=None)
    parser.add_argument("--ddim_eta", type=float, default=None)
    parser.add_argument("--ddim_spacing", type=str, default=None, choices=["uniform", "logsnr"])
    parser.add_argument("--ddim_clip_x_start", type=str, default=None)
    parser.add_argument("--ddim_clip_percentile", type=float, default=None)
    parser.add_argument("--ddim_start_t", type=int, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--n_samples", type=int, default=None,
                        help="samples per condition video; >1 writes <name>_s<k>.npy for k>=1 plus <name>.npy for k=0")
    parser.add_argument("--no_zero_frame0", action="store_true")
    parser.add_argument("--no_zero_invalid_frames", action="store_true")
    parser.add_argument("--no_gif", action="store_true")
    parser.add_argument("--skip_existing", action="store_true")
    parser.add_argument("--no_config_snapshot", action="store_true")
    return parser.parse_args()


def normalize_cond_img(t):
    return t / 255.0


def main():
    args = parse_args()
    with args.config.open() as f:
        cfg = json.load(f)

    data_cfg = cfg["data"]
    model_cfg = cfg["model"]
    diffusion_cfg = cfg["diffusion"]
    train_cfg = cfg["training"]
    label_cfg = cfg["labels"]
    infer_cfg = cfg.get("inference", {})

    pick = lambda name, default_: getattr(args, name) if getattr(args, name) is not None else infer_cfg.get(name, default_)
    video_type = pick("video_type", "dense")
    start = pick("start", 0)
    end = pick("end", None)
    milestone = pick("milestone", -1)
    label_mode = pick("label_mode", "level_only")
    cond_scale = float(pick("cond_scale", 1.0))
    sampler = pick("sampler", "ddpm")
    ddim_steps = pick("ddim_steps", 50)
    ddim_eta = pick("ddim_eta", 0.0)
    ddim_spacing = pick("ddim_spacing", "uniform")
    ddim_clip_percentile = pick("ddim_clip_percentile", 0.995)
    ddim_start_t = pick("ddim_start_t", None)
    seed = pick("seed", None)
    n_samples = int(pick("n_samples", 1))
    ddim_clip_x_start = pick("ddim_clip_x_start", None)
    if isinstance(ddim_clip_x_start, str) and ddim_clip_x_start.lower() in ("none", "null", ""):
        ddim_clip_x_start = None
    elif isinstance(ddim_clip_x_start, str) and ddim_clip_x_start != "dynamic":
        ddim_clip_x_start = float(ddim_clip_x_start)
    zero_frame0 = not args.no_zero_frame0 and infer_cfg.get("zero_frame0", True)
    zero_invalid = not args.no_zero_invalid_frames and infer_cfg.get("zero_invalid_frames", True)
    skip_existing = args.skip_existing or infer_cfg.get("skip_existing", False)

    # Same placeholder resolution as training, so the config can keep the
    # template; {bs} is training.batch_size or per_gpu_batch_size, whichever
    # the checkpoint's training config used.
    resolve_experiment_name(cfg)
    print(f"experiment: {cfg['experiment_name']}")
    if not args.no_config_snapshot:
        save_config_snapshot(cfg, cfg["experiment_name"])
    if not torch.cuda.is_available():
        raise RuntimeError("inference_label_cond.py requires a CUDA GPU")

    encoder = build_encoder(label_cfg)
    diffusion = build_model(model_cfg, diffusion_cfg, label_cfg, encoder).cuda()
    base_path = Path(data_cfg["base_path"])

    trainer = LabelCondTrainer(
        diffusion_model=diffusion,
        input_video_folder=str(base_path / data_cfg["input_video_subdir"]),
        contour_condition_video_dir=str(base_path / data_cfg["contour_condition_subdir"]),
        sampling_contour_condition_video_dir=str(base_path / data_cfg["train_sampling_contour_condition_subdir"]),
        label_encoder=encoder,
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
    trainer.load(milestone=milestone)

    # Valid-frame counts for this video type, if a CSV is configured for it.
    valid_frames_map = {}
    fv_cfg = (cfg.get("frame_validity") or {}).get(video_type) or {}
    if zero_invalid and fv_cfg.get("csv_path"):
        valid_frames_map = load_valid_frames_map(
            fv_cfg["csv_path"], fv_cfg.get("filename_col", f"{video_type}_filename"),
            fv_cfg.get("valid_frames_col", f"{video_type}_valid_frames"),
        )
        print(f"zeroing frames past the valid count for {len(valid_frames_map)} files")
    elif zero_invalid:
        print(f"no frame_validity.{video_type} CSV configured; invalid-frame zeroing off")

    contour_cond_video_dir = str(base_path / data_cfg["sampling_contour_condition_subdir"].format(video_type=video_type))
    output_dir = inference_output_dir(cfg["experiment_name"], video_type)
    done_names = completed_output_names(output_dir) if skip_existing else None
    cond_video_paths = select_condition_paths(contour_cond_video_dir, start, end, done_names)
    if skip_existing:
        print(f"skip_existing: {len(done_names)} done in {output_dir}; {len(cond_video_paths)} left in [{start}, {end})")
        if not cond_video_paths:
            return

    gt_disp_dir = str(base_path / data_cfg["gt_disp_dense_subdir"]) if video_type == "dense" else None
    device = next(trainer.ema_model.parameters()).device
    save_folder = f"./{cfg['experiment_name']}/sampling_time_sampled_{video_type}_part_videos_infos/"
    os.makedirs(save_folder, exist_ok=True)
    print(f"label_mode={label_mode} cond_scale={cond_scale} sampler={sampler} n_samples={n_samples} "
          f"zero_frame0={zero_frame0} zero_invalid_frames={bool(valid_frames_map)}")

    for cond_path in tqdm(cond_video_paths, desc="videos", unit="video"):
        cond_filename = f"{cond_path.stem}.npy"
        contour = normalize_cond_img(torch.from_numpy(np.load(cond_path)).float().unsqueeze(0)).to(device)
        labels = trainer.labels_for([cond_filename], mode=label_mode).to(device)
        gt_disp = np.load(os.path.join(gt_disp_dir, cond_filename)) if gt_disp_dir and os.path.exists(os.path.join(gt_disp_dir, cond_filename)) else None
        n_valid = valid_frames_map.get(cond_filename)

        for k in range(n_samples):
            out_name = cond_filename if k == 0 else f"{cond_path.stem}_s{k}.npy"
            trainer.sample_and_save_one_video_at_a_time(
                "final", contour, [out_name], None, gt_disp, cond_scale=cond_scale, save_folder=save_folder,
                sampler=sampler, ddim_steps=ddim_steps, ddim_eta=ddim_eta, ddim_spacing=ddim_spacing,
                ddim_clip_x_start=ddim_clip_x_start, ddim_clip_percentile=ddim_clip_percentile,
                ddim_start_t=ddim_start_t, seed=None if seed is None else int(seed) + k,
                labels=labels, zero_frame0=zero_frame0, valid_frames=[n_valid],
                make_gif=(not args.no_gif and k == 0),
            )


if __name__ == "__main__":
    main()
