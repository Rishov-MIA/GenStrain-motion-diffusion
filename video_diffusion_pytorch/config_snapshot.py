"""Snapshot the training config into the experiment folder for tracking."""

import json
import time
from pathlib import Path


def save_config_snapshot(cfg, experiment_name):
    """Write `cfg` to ./{experiment_name}/config.json (the trainers' results
    folder) so the hyperparameters used for a run can be inspected later.

    config.json always holds the current run's config. If a snapshot from an
    earlier run exists with different contents (e.g. a hyperparameter was
    changed before resuming), it is preserved as config_<timestamp>.json
    instead of being overwritten. In DDP, call this from rank 0 only.
    """
    exp_dir = Path(f"./{experiment_name}")
    exp_dir.mkdir(parents=True, exist_ok=True)
    snapshot_path = exp_dir / "config.json"
    new_text = json.dumps(cfg, indent=4) + "\n"

    if snapshot_path.exists():
        old_text = snapshot_path.read_text()
        if old_text == new_text:
            return snapshot_path
        backup_path = exp_dir / f"config_{time.strftime('%Y%m%d_%H%M%S')}.json"
        snapshot_path.rename(backup_path)
        print(f"config changed since last run; previous snapshot kept at {backup_path}")

    snapshot_path.write_text(new_text)
    print(f"saved config snapshot to {snapshot_path}")
    return snapshot_path
