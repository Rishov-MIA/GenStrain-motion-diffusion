"""Which predictions already exist, so an interrupted inference run can resume.

Every inference_*.py writes one prediction per condition video to
`./{experiment_name}/sampling_time_sampled_{video_type}_part_videos_infos/inference_disps/<cond>.npy`,
named after the condition file itself. That makes the output folder a resume
log: a condition file whose prediction is already there does not need to be
sampled again — which is what `--skip_existing` acts on.

Both sides resolve it through here: the workers (to drop finished videos from
their own slice) and run_inference_multi_gpu.py (to balance only the remaining
work across GPUs), so the two can never disagree about what "already done" means.
"""

from pathlib import Path

import numpy as np


def inference_output_dir(experiment_name, video_type, root="."):
    """The folder the workers save predictions into. Mirrors the `custom_save_folder`
    each inference_*.py builds, plus the `inference_disps` subfolder the trainer's
    sample_and_save_one_video_at_a_time writes into."""
    return (Path(root) / str(experiment_name)
            / f"sampling_time_sampled_{video_type}_part_videos_infos" / "inference_disps")


def output_name_for(cond_path):
    """The prediction filename a worker writes for a given condition file."""
    return f"{Path(cond_path).stem}.npy"


def completed_output_names(output_dir):
    """Names of the predictions in `output_dir` that are complete and readable.

    A run killed mid-save leaves a truncated .npy behind — exactly the file a
    resume is most likely to trip over. Opening each one with mmap reads the
    header and checks the file is as long as the header claims, without loading
    any data, so partial files are reported as not-done and get sampled again.
    """
    output_dir = Path(output_dir)
    if not output_dir.is_dir():
        return set()

    done = set()
    for path in sorted(output_dir.glob("*.npy")):
        try:
            np.load(path, mmap_mode="r")
        except Exception:
            print(f"ignoring unreadable/partial prediction (will re-sample): {path}")
            continue
        done.add(path.name)
    return done


def select_condition_paths(cond_video_dir, start, end, done_names=None):
    """The condition files a sampling loop will process: the sorted *.npy glob
    sliced by [start:end), minus any whose prediction is already in `done_names`
    (pass None to keep every file in the slice)."""
    paths = sorted(Path(cond_video_dir).glob("*.npy"))[start:end]
    if done_names is None:
        return paths
    return [p for p in paths if output_name_for(p) not in done_names]
