"""Shard inference across multiple GPUs — the easy near-linear speedup.

Each test video is sampled independently (a 1000-step DDPM chain that can't be
parallelized internally), but the videos are independent of each other. So we
just split the sorted list of condition files into one contiguous slice per GPU
and launch one inference process per GPU, each pinned to its GPU via
CUDA_VISIBLE_DEVICES and handed its slice via --start/--end. All processes write
into the same experiment folder with per-video filenames, so results merge
automatically.

No model/config changes and no torchrun: this reuses the existing
inference_*.py scripts exactly as they run single-GPU.

    # auto-detect GPUs, run the default config across all of them
    python run_inference_multi_gpu.py --script inference_both_contour_motion_full_region.py --config configs/inference_both_contour_motion_full_region.json
    python run_inference_multi_gpu.py --script inference_full_region.py
    python run_inference_multi_gpu.py --script inference_full_region.py --config configs/inference_augmented.json
    python run_inference_multi_gpu.py --script inference_full_region.py --config configs/inference_group_dro.json
    
    # pick a config + which GPUs to use, override video_type
    python run_inference_multi_gpu.py \
        --script inference_full_region.py \
        --config configs/inference_group_dro.json \
        --gpus 0,1,2,3 --video_type dense

    # dry run: print the per-GPU commands without launching
    python run_inference_multi_gpu.py --script inference_full_region.py --dry_run

Each worker script must accept --config/--video_type/--start/--end and slice the
sorted *.npy condition list by [start:end] — which all inference_*.py do.
"""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

# Per-script rule for the directory whose *.npy files are sharded, mirroring how
# each inference script resolves it from the config + resolved video_type. These
# must stay in sync with the scripts: if they disagree, the shard totals would be
# computed from a different folder than the workers actually read.
#   "contour_template" -> {base_path}/{data.sampling_contour_condition_subdir}
#   "motion_template"  -> {base_path}/{data.sampling_motion_condition_subdir}
COND_DIR_RULE = {
    "inference.py": "contour_template",
    "inference_full_region.py": "contour_template",
    "inference_both_contour_motion.py": "contour_template",
    "inference_both_contour_motion_full_region.py": "contour_template",
    "inference_only_motion.py": "motion_template",
}

REPO_ROOT = Path(__file__).resolve().parent


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--script", required=True, help="inference_*.py to shard (e.g. inference_full_region.py)")
    parser.add_argument("--config", type=Path, default=None,
                        help="config passed to each worker; defaults to the script's own default config")
    parser.add_argument("--gpus", type=str, default=None,
                        help="comma-separated GPU ids (e.g. 0,1,2,3). Default: all visible GPUs")
    parser.add_argument("--video_type", type=str, default=None,
                        help="override video_type for every worker (else the config's value)")
    parser.add_argument("--milestone", type=int, default=None, help="override checkpoint milestone for every worker")
    parser.add_argument("--sampler", type=str, default=None, choices=["ddpm", "ddim"],
                        help="override the sampler for every worker (else the config's value, default ddpm)")
    parser.add_argument("--ddim_steps", type=int, default=None,
                        help="override ddim_steps for every worker; ignored unless the sampler is ddim")
    parser.add_argument("--ddim_eta", type=float, default=None,
                        help="override ddim_eta for every worker; ignored unless the sampler is ddim")
    parser.add_argument("--ddim_spacing", type=str, default=None, choices=["uniform", "logsnr"],
                        help="override the DDIM timestep spacing for every worker")
    parser.add_argument("--ddim_clip_x_start", type=str, default=None,
                        help="override x0 clipping for every worker: 'none', 'dynamic', or a float")
    parser.add_argument("--ddim_clip_percentile", type=float, default=None,
                        help="override the dynamic-clip percentile for every worker")
    parser.add_argument("--ddim_start_t", type=int, default=None,
                        help="override the DDIM start timestep for every worker")
    parser.add_argument("--seed", type=int, default=None,
                        help="seed x_T identically on every worker (each still gets its own videos)")
    parser.add_argument("--start", type=int, default=None,
                        help="only shard the [start:end) window of the file list (default: whole list)")
    parser.add_argument("--end", type=int, default=None,
                        help="only shard the [start:end) window of the file list (default: through the last file)")
    parser.add_argument("--python", type=str, default=sys.executable, help="python interpreter for the workers")
    parser.add_argument("--dry_run", action="store_true", help="print the per-GPU commands without launching them")
    return parser.parse_args()


def default_config_for(script_name):
    """Read DEFAULT_CONFIG the same way the worker would: configs/<script>.json."""
    return REPO_ROOT / "configs" / (Path(script_name).stem + ".json")


def detect_gpus():
    """Return a list of GPU-id strings. Honor CUDA_VISIBLE_DEVICES if the parent
    already restricts it; otherwise query torch for the physical device count."""
    env = os.environ.get("CUDA_VISIBLE_DEVICES")
    if env:
        return [g.strip() for g in env.split(",") if g.strip() != ""]
    try:
        import torch
        n = torch.cuda.device_count()
    except Exception as e:
        raise SystemExit(f"could not detect GPUs via torch ({e}); pass --gpus explicitly")
    if n == 0:
        raise SystemExit("no CUDA GPUs detected; pass --gpus explicitly if this is wrong")
    return [str(i) for i in range(n)]


def resolve_cond_dir(script_name, cfg, video_type):
    """Reproduce the exact directory the worker globs, so we count the same files."""
    base_path = Path(cfg["data"]["base_path"])
    rule = COND_DIR_RULE[script_name]
    if rule == "contour_template":
        return base_path / cfg["data"]["sampling_contour_condition_subdir"].format(video_type=video_type)
    if rule == "motion_template":
        return base_path / cfg["data"]["sampling_motion_condition_subdir"].format(video_type=video_type)
    raise ValueError(f"unknown cond-dir rule for {script_name}")


def even_shards(start, end, n_shards):
    """Split [start, end) into n_shards contiguous [lo, hi) windows, sizes differing
    by at most 1. Empty windows (more GPUs than videos) are dropped."""
    total = end - start
    base, extra = divmod(total, n_shards)
    shards = []
    lo = start
    for i in range(n_shards):
        size = base + (1 if i < extra else 0)
        if size == 0:
            continue
        shards.append((lo, lo + size))
        lo += size
    return shards


def main():
    args = parse_args()

    script_name = Path(args.script).name
    if script_name not in COND_DIR_RULE:
        raise SystemExit(f"unknown script {script_name!r}; expected one of: {', '.join(sorted(COND_DIR_RULE))}")
    script_path = REPO_ROOT / script_name
    if not script_path.exists():
        raise SystemExit(f"script not found: {script_path}")

    config_path = args.config if args.config is not None else default_config_for(script_name)
    if not config_path.exists():
        raise SystemExit(f"config not found: {config_path}")
    with config_path.open() as f:
        cfg = json.load(f)

    infer_cfg = cfg.get("inference", {})
    # Resolve video_type exactly like the worker: CLI > config > "cine".
    video_type = args.video_type if args.video_type is not None else infer_cfg.get("video_type", "cine")

    cond_dir = resolve_cond_dir(script_name, cfg, video_type)
    if not cond_dir.is_dir():
        raise SystemExit(
            f"condition dir does not exist: {cond_dir}\n"
            f"(resolved from config {config_path} + video_type={video_type}). "
            f"Check data.base_path / the *_subdir template and that this runs where the data lives."
        )
    n_files = len(sorted(cond_dir.glob("*.npy")))
    if n_files == 0:
        raise SystemExit(f"no *.npy condition files in {cond_dir}")

    # Window to shard: CLI start/end override the config's inference block; end=None → whole list.
    win_start = args.start if args.start is not None else infer_cfg.get("start", 0) or 0
    win_end = args.end if args.end is not None else infer_cfg.get("end", None)
    if win_end is None:
        win_end = n_files
    win_start = max(0, min(win_start, n_files))
    win_end = max(win_start, min(win_end, n_files))

    gpus = [g.strip() for g in args.gpus.split(",") if g.strip()] if args.gpus else detect_gpus()
    if not gpus:
        raise SystemExit("no GPUs to run on")

    shards = even_shards(win_start, win_end, len(gpus))
    if not shards:
        raise SystemExit(f"nothing to do: window [{win_start}, {win_end}) is empty")

    print(f"script      : {script_name}")
    print(f"config      : {config_path}")
    print(f"experiment  : {cfg.get('experiment_name')}")
    print(f"video_type  : {video_type}")
    print(f"cond dir    : {cond_dir}  ({n_files} files)")
    print(f"window      : [{win_start}, {win_end})  ->  {win_end - win_start} videos")
    print(f"gpus        : {', '.join(gpus)}  ({len(shards)} worker(s))")
    print()

    procs = []
    for idx, (gpu, (lo, hi)) in enumerate(zip(gpus, shards)):
        cmd = [
            args.python, str(script_path),
            "--config", str(config_path),
            "--video_type", video_type,
            "--start", str(lo),
            "--end", str(hi),
        ]
        if args.milestone is not None:
            cmd += ["--milestone", str(args.milestone)]
        if args.sampler is not None:
            cmd += ["--sampler", args.sampler]
        if args.ddim_steps is not None:
            cmd += ["--ddim_steps", str(args.ddim_steps)]
        if args.ddim_eta is not None:
            cmd += ["--ddim_eta", str(args.ddim_eta)]
        if args.ddim_spacing is not None:
            cmd += ["--ddim_spacing", args.ddim_spacing]
        if args.ddim_clip_x_start is not None:
            cmd += ["--ddim_clip_x_start", args.ddim_clip_x_start]
        if args.ddim_clip_percentile is not None:
            cmd += ["--ddim_clip_percentile", str(args.ddim_clip_percentile)]
        if args.ddim_start_t is not None:
            cmd += ["--ddim_start_t", str(args.ddim_start_t)]
        if args.seed is not None:
            cmd += ["--seed", str(args.seed)]
        # Every worker calls save_config_snapshot on the SAME ./{experiment}/config.json.
        # Concurrently that races (one renames it to a backup, the rest hit
        # FileNotFoundError). Let only the first worker write it; the rest skip.
        if idx != 0:
            cmd += ["--no_config_snapshot"]
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=gpu)
        print(f"GPU {gpu}: videos [{lo}, {hi})  ({hi - lo})   CUDA_VISIBLE_DEVICES={gpu} {' '.join(cmd)}")
        if not args.dry_run:
            procs.append((gpu, lo, hi, subprocess.Popen(cmd, env=env, cwd=str(REPO_ROOT))))

    if args.dry_run:
        print("\n[dry run] nothing launched.")
        return

    print(f"\nlaunched {len(procs)} worker(s); waiting…")
    failed = []
    for gpu, lo, hi, p in procs:
        rc = p.wait()
        status = "ok" if rc == 0 else f"FAILED (exit {rc})"
        print(f"GPU {gpu} [{lo}, {hi}): {status}")
        if rc != 0:
            failed.append(gpu)

    if failed:
        raise SystemExit(f"\n{len(failed)} worker(s) failed on GPU(s): {', '.join(failed)}")
    print("\nall workers finished successfully.")


if __name__ == "__main__":
    main()
