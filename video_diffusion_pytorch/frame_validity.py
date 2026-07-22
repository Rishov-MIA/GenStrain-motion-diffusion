"""Per-sample valid-frame masking for the displacement diffusion loss.

Clips are resampled to a fixed ``num_frames`` (e.g. 26), but a clip may only
have the first ``N`` frames real, the rest padding. A CSV lists, per video, how
many leading frames are valid; this module turns that into a frame mask so the
training loss ignores the padding frames.

Convention (confirmed with the data owner): ``valid_frames = N`` means frames
``[0:N]`` are valid and ``[N:num_frames]`` are padding.

Everything here is OPTIONAL and OFF by default: with no CSV, callers pass
``valid_frames=None`` and the loss path is byte-for-byte the original one.

Split into a torch-free part (CSV load, count resolution — unit-testable without
a GPU) and ``build_frame_mask`` (needs torch).
"""

import csv
from pathlib import Path


def load_valid_frames_map(csv_path, filename_col="dense_filename", count_col="dense_valid_frames"):
    """Read ``csv_path`` into a ``{filename: valid_frame_count}`` dict.

    Keys are stored in several normalized forms (as written, and with/without a
    ``.npy`` suffix) so lookups match regardless of whether the CSV lists bare
    stems or full filenames. Returns an empty dict for a missing/empty path so
    callers can treat "no CSV" as "masking off".
    """
    if not csv_path:
        return {}
    path = Path(csv_path)
    if not path.exists():
        raise FileNotFoundError(f"frame-validity CSV not found: {path}")

    mapping = {}
    with path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None or filename_col not in reader.fieldnames or count_col not in reader.fieldnames:
            raise ValueError(
                f"CSV {path} must have columns {filename_col!r} and {count_col!r}; "
                f"found {reader.fieldnames}"
            )
        for row in reader:
            raw_name = (row[filename_col] or "").strip()
            raw_count = (row[count_col] or "").strip()
            if not raw_name or raw_count == "":
                continue
            count = int(float(raw_count))  # tolerate "26" or "26.0"
            for key in _name_variants(raw_name):
                mapping[key] = count
    return mapping


def _name_variants(name):
    """All key spellings we accept for one filename: as-is, stem, and stem+.npy."""
    name = name.strip()
    stem = name[:-4] if name.lower().endswith(".npy") else name
    return {name, stem, f"{stem}.npy"}


def resolve_valid_frames(mapping, filename, num_frames, missing="full"):
    """Valid-frame count for one file, clamped to ``[1, num_frames]``.

    ``missing`` controls what happens when the file is absent from the CSV:
      - ``"full"`` (default): treat all ``num_frames`` as valid (no masking for
        that sample) — keeps training robust to gaps in the CSV.
      - ``"error"``: raise, if you want to guarantee every sample is listed.
    """
    if not mapping:
        return num_frames
    for key in _name_variants(str(filename)):
        if key in mapping:
            return max(1, min(int(mapping[key]), num_frames))
    if missing == "error":
        raise KeyError(f"{filename!r} not found in frame-validity CSV")
    return num_frames


def build_frame_mask(valid_frames, num_frames, device=None, dtype=None):
    """Build a ``[B, 1, F, 1, 1]`` float mask that is 1 on valid frames, 0 on
    padding, for broadcasting against a ``[B, C, F, H, W]`` loss tensor.

    ``valid_frames`` is a 1-D integer tensor of length B (per-sample counts).
    A frame index ``f`` is valid iff ``f < valid_frames[b]`` (leading-N rule).
    """
    import torch

    valid_frames = torch.as_tensor(valid_frames, device=device).reshape(-1)
    b = valid_frames.shape[0]
    if device is None:
        device = valid_frames.device
    frame_idx = torch.arange(num_frames, device=device).reshape(1, num_frames)  # [1, F]
    mask = (frame_idx < valid_frames.reshape(b, 1).to(device)).to(dtype or torch.float32)  # [B, F]
    return mask.reshape(b, 1, num_frames, 1, 1)
