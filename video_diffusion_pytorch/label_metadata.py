"""Per-slice labels (disease group, slice level, site) from processed_excel.json.

One place for the label vocabulary so the Dataset, the trainer's preview
sampling, the inference script and the checkpoint compatibility check all
agree on it. Torch-free on purpose (numpy scripts import it too).

Every label table has one extra "null" row at the END, used both for
classifier-free dropout during training and for "label unknown" at inference.
It is a separate learned vector per label type, never index 0 of the table, so
"unknown" and the first real class cannot share a slot.
"""

import json
import re
from pathlib import Path

# Disease-string -> group. Order = priority: first rule that matches wins.
# Groups with < ~10 patients in processed_excel.json (amyloid, HCM, aortic
# stenosis, ...) deliberately fall into "Other": an embedding row trained on
# six patients would only overfit.
DISEASE_RULES = [
    ("LBBB-scar", lambda d: "lbbb" in d and "scar" in d),
    ("LBBB", lambda d: "lbbb" in d),
    ("Healthy", lambda d: "healthy" in d or "normal" in d or "volunteer" in d),
    ("DCM", lambda d: "dcm" in d or "dilated" in d or "low ef" in d or "cardiomyopathy" in d),
    ("MI", lambda d: "acute mi" in d or "chronic mi" in d or "ischemic" in d),
    ("HF", lambda d: d.startswith("hf") or "hfpef" in d),
    ("Myocarditis", lambda d: "myocarditis" in d),
]
DISEASE_GROUPS = [name for name, _ in DISEASE_RULES] + ["Other"]
SLICE_LEVELS = ["Apex", "Mid", "Base"]
SITES = ["UVA", "Brompton", "Glassgow", "St_Francis", "Lyon", "Kentucky", "Emory", "Stanford"]

LABEL_NAMES = ("disease", "level", "site")
LABEL_MODES = ("full", "level_only", "none")


def disease_group(disease):
    d = (disease or "").lower()
    for name, rule in DISEASE_RULES:
        if rule(d):
            return name
    return "Other"


def slice_level(filename):
    low = filename.lower()
    for lvl in SLICE_LEVELS:
        if lvl.lower() in low:
            return lvl
    return None


def lookup_metadata(metadata, filename):
    """Metadata entry for a mask/displacement filename, or None.

    Same key fallbacks as trainer_group_dro.GroupedContourDataset: strip a
    trailing "_scale_<f>" (augmented copies) and "_cycleN", try "<stem>.mat"
    and the bare stem.
    """
    stem = Path(filename).stem
    stems = []
    for s in (stem, re.sub(r"_scale_\d+(?:p\d+)?$", "", stem)):
        for c in (s, re.sub(r"_cycle[A-Za-z0-9]+$", "", s)):
            if c not in stems:
                stems.append(c)
    keys = [filename]
    for s in stems:
        keys.extend((f"{s}.mat", s))
    for k in keys:
        if k in metadata:
            return metadata[k]
    return None


class LabelEncoder:
    """filename -> [disease_idx, level_idx, site_idx] with per-table null rows."""

    def __init__(self, metadata_json_path=None, use_site=True):
        self.metadata_json_path = str(metadata_json_path) if metadata_json_path else None
        self.metadata = {}
        if metadata_json_path:
            with open(metadata_json_path) as f:
                self.metadata = json.load(f)
        self.use_site = bool(use_site)
        self.vocab = {"disease": list(DISEASE_GROUPS), "level": list(SLICE_LEVELS), "site": list(SITES)}

    # ---- vocabulary -------------------------------------------------------
    @property
    def vocab_sizes(self):
        """Real-class counts per label, excluding the null row."""
        return tuple(len(self.vocab[n]) for n in LABEL_NAMES)

    @property
    def null_indices(self):
        return tuple(len(self.vocab[n]) for n in LABEL_NAMES)

    def spec(self):
        """What gets stamped into checkpoints; see LabelCondTrainer.save."""
        return {"vocab": {n: list(v) for n, v in self.vocab.items()}, "use_site": self.use_site}

    # ---- encoding ---------------------------------------------------------
    def encode(self, filename):
        """Indices for one file. Missing metadata -> disease/site null; the slice
        level comes from the filename so it is available even without metadata."""
        info = lookup_metadata(self.metadata, filename) or {}
        disease_idx = self.vocab["disease"].index(disease_group(info.get("disease"))) if info else self.null_indices[0]
        lvl = slice_level(filename)
        level_idx = self.vocab["level"].index(lvl) if lvl else self.null_indices[1]
        site = info.get("site")
        site_idx = self.vocab["site"].index(site) if (self.use_site and site in self.vocab["site"]) else self.null_indices[2]
        return [disease_idx, level_idx, site_idx]

    def apply_mode(self, indices, mode):
        """Blank labels according to the inference label mode.

        full       -> use everything known
        level_only -> keep slice level, null disease and site (deployment default)
        none       -> all null: the unconditional path, i.e. today's mask-only model
        """
        if mode not in LABEL_MODES:
            raise ValueError(f"unknown label mode {mode!r}, expected one of {LABEL_MODES}")
        d, l, s = indices
        nd, nl, ns = self.null_indices
        if mode == "none":
            return [nd, nl, ns]
        if mode == "level_only":
            return [nd, l, ns]
        return [d, l, s]

    def describe(self, indices):
        names = []
        for n, idx in zip(LABEL_NAMES, indices):
            v = self.vocab[n]
            names.append(v[idx] if idx < len(v) else "null")
        return "/".join(names)

    def coverage(self, filenames):
        """(#with metadata, #without) for a list of filenames; for startup logging."""
        hit = sum(1 for f in filenames if lookup_metadata(self.metadata, f) is not None)
        return hit, len(filenames) - hit
