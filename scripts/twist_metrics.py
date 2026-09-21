#!/usr/bin/env python3
"""Twist / rotation metrics for sampled displacement fields, per disease group.

Strain (CC/RR/RC) is computed from the Green-Lagrange tensor, which is invariant
to rigid rotation, so no strain number can see whether a slice rotates. This
script scores the ROTATION of predicted DENSE displacement fields against the
ground truth, broken down by disease group, slice level and patient.

Rotation curve theta(t): for every frame-0 myocardial pixel with radius vector
r = p - c (c = frame-0 mask centroid) and Lagrangian displacement u, the local
rotation is (r_x u_y - r_y u_x) / |r|^2; theta(t) is its |r|-weighted mean over
the ROI, in degrees. Sign convention follows the array axes (channel 0 = x /
column, channel 1 = y / row), so it is consistent between GT and prediction
but not necessarily "clockwise from apex".

Usage (numpy only, no torch, runs on a login node):

    python scripts/twist_metrics.py <experiment_dir_or_inference_disps_dir> [...]
        [--data <dataset root with dense/test>] [--metadata processed_excel.json]
        [--csv out.csv] [--min-group 5]

With an experiment dir, the newest config*.json in it supplies the dataset
root (data.sampling_base_path) and inference_disps is taken from
sampling_time_sampled_dense_part_videos_infos/. Several experiments can be
given at once for a side-by-side table.
"""
import argparse
import csv
import glob
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

DEFAULT_METADATA = (
    "/scratch/vst2hb/Dataset_Processing/Data Bank/proposal-genstrain-new/processed_excel.json"
)
DEFAULT_SAMPLE_SUBDIR = "sampling_time_sampled_dense_part_videos_infos/inference_disps"

# Disease-string -> group. Order matters: first match wins. Groups with < ~10
# patients in processed_excel.json (amyloid, HCM, aortic stenosis, ...) fall
# into "Other" on purpose.
GROUP_RULES = [
    ("LBBB-scar", lambda d: "lbbb" in d and "scar" in d),
    ("LBBB", lambda d: "lbbb" in d),
    ("Healthy", lambda d: "healthy" in d or "normal" in d or "volunteer" in d),
    ("DCM", lambda d: "dcm" in d or "dilated" in d or "low ef" in d or "cardiomyopathy" in d),
    ("MI", lambda d: "acute mi" in d or "chronic mi" in d or "ischemic" in d),
    ("HF", lambda d: d.startswith("hf") or "hfpef" in d),
    ("Myocarditis", lambda d: "myocarditis" in d),
]
GROUP_ORDER = [name for name, _ in GROUP_RULES] + ["Other"]
LEVELS = ("Apex", "Mid", "Base")

_Y, _X = np.mgrid[0:64, 0:64]


# ----------------------------------------------------------------------------
# geometry
# ----------------------------------------------------------------------------
def rotation_curve(u, roi):
    """u: [2, F, H, W] displacement, roi: [H, W] bool frame-0 mask -> theta[F] (deg)."""
    H, W = roi.shape
    Y, X = (_Y, _X) if (H, W) == (64, 64) else np.mgrid[0:H, 0:W]
    cy, cx = Y[roi].mean(), X[roi].mean()
    rx, ry = (X - cx)[roi], (Y - cy)[roi]
    r2 = rx ** 2 + ry ** 2
    r = np.sqrt(r2)
    ux, uy = u[0][:, roi], u[1][:, roi]
    local = (rx * uy - ry * ux) / r2  # [F, N]
    return np.degrees((local * r).sum(1) / r.sum())


def tangential_fraction(u, roi):
    H, W = roi.shape
    Y, X = (_Y, _X) if (H, W) == (64, 64) else np.mgrid[0:H, 0:W]
    cy, cx = Y[roi].mean(), X[roi].mean()
    rx, ry = (X - cx)[roi], (Y - cy)[roi]
    r = np.sqrt(rx ** 2 + ry ** 2)
    ux, uy = u[0][:, roi], u[1][:, roi]
    tan = (rx * uy - ry * ux) / r
    tot = np.linalg.norm(np.stack([ux, uy]))
    return float(np.linalg.norm(tan) / tot) if tot > 0 else float("nan")


def signed_peak(theta):
    return float(theta[np.argmax(np.abs(theta))])


def late_systolic_mean(theta):
    F = len(theta)
    return float(theta[int(0.4 * F): max(int(0.7 * F), int(0.4 * F) + 1)].mean())


# ----------------------------------------------------------------------------
# metadata
# ----------------------------------------------------------------------------
def disease_group(disease):
    d = (disease or "").lower()
    for name, rule in GROUP_RULES:
        if rule(d):
            return name
    return "Other"


def slice_level(filename):
    for lvl in LEVELS:
        if lvl.lower() in filename.lower():
            return lvl
    return "?"


def patient_id(filename):
    """Text before the first _apex/_mid/_base token; '_scale_x' and '_cycleX' never precede it."""
    return re.split(r"_(apex|mid|base)", filename, flags=re.I)[0]


def lookup_metadata(metadata, filename):
    """Same key fallbacks as trainer_group_dro.GroupedContourDataset._lookup_metadata."""
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


def load_frame_counts(path):
    counts = {}
    if not path or not os.path.exists(path):
        return counts
    with open(path) as f:
        for row in csv.DictReader(f):
            name = row.get("dense_filename") or row.get("filename")
            try:
                counts[name] = int(row.get("dense_valid_frames") or row.get("valid_frames"))
            except (TypeError, ValueError):
                pass
    return counts


# ----------------------------------------------------------------------------
# experiment resolution
# ----------------------------------------------------------------------------
def resolve_experiment(arg, data_override):
    """-> (name, inference_disps_dir, dataset_root)."""
    p = Path(arg).resolve()
    if p.name == "inference_disps":
        pred_dir = p
        exp_dir = p.parent.parent
    else:
        exp_dir = p
        pred_dir = exp_dir / DEFAULT_SAMPLE_SUBDIR
        if not pred_dir.is_dir():
            cands = sorted(exp_dir.glob("*/inference_disps"))
            if not cands:
                sys.exit(f"{exp_dir}: no inference_disps found")
            pred_dir = cands[-1]
            print(f"[{exp_dir.name}] using {pred_dir.relative_to(exp_dir)}", file=sys.stderr)
    data_root = data_override
    if data_root is None:
        cfgs = sorted(exp_dir.glob("config*.json"), key=os.path.getmtime)
        for cfg in reversed(cfgs):
            try:
                d = json.load(open(cfg)).get("data", {})
            except Exception:
                continue
            data_root = d.get("sampling_base_path") or d.get("base_path")
            if data_root:
                break
    if data_root is None:
        sys.exit(f"{exp_dir}: cannot infer dataset root, pass --data")
    return exp_dir.name, pred_dir, Path(data_root)


# ----------------------------------------------------------------------------
# scoring
# ----------------------------------------------------------------------------
def score_experiment(name, pred_dir, data_root, metadata, split="test"):
    gt_disp_dir = data_root / "dense" / split / "displacement_dense"
    gt_mask_dir = data_root / "dense" / split / "dense_mask"
    frame_counts = load_frame_counts(data_root / f"frame_counts_{split}.csv")
    if not gt_disp_dir.is_dir():
        sys.exit(f"{gt_disp_dir} not found")
    rows = []
    skipped = 0
    for pred_path in sorted(glob.glob(str(pred_dir / "*.npy"))):
        fname = os.path.basename(pred_path)
        gt_path, mask_path = gt_disp_dir / fname, gt_mask_dir / fname
        if not (gt_path.exists() and mask_path.exists()):
            skipped += 1
            continue
        pred = np.load(pred_path)
        gt = np.load(gt_path)
        mask = np.load(mask_path)
        pred = pred[0] if pred.ndim == 5 else pred
        gt = gt[0] if gt.ndim == 5 else gt
        mask = mask[0] if mask.ndim == 4 else mask
        if pred.shape != gt.shape:
            skipped += 1
            continue
        F = min(frame_counts.get(fname, gt.shape[1]), gt.shape[1])
        roi = mask[0] > 0.5 * mask.max() if mask.max() > 0 else mask[0] > 0
        if roi.sum() < 10:
            skipped += 1
            continue
        pred, gt = pred[:, :F], gt[:, :F]
        th_p, th_g = rotation_curve(pred, roi), rotation_curve(gt, roi)
        gt_mag = np.sqrt((gt ** 2).sum(0))[:, roi].mean()
        pred_mag = np.sqrt((pred ** 2).sum(0))[:, roi].mean()
        epe = np.sqrt(((pred - gt) ** 2).sum(0))[:, roi].mean()
        info = lookup_metadata(metadata, fname) or {}
        rows.append(dict(
            experiment=name, file=fname, patient=patient_id(fname), level=slice_level(fname),
            disease=info.get("disease", "missing"), group=disease_group(info.get("disease")),
            site=info.get("site", "missing"), valid_frames=F,
            peak_gt=signed_peak(th_g), peak_pred=signed_peak(th_p),
            late_gt=late_systolic_mean(th_g), late_pred=late_systolic_mean(th_p),
            ttp_gt=int(np.argmax(np.abs(th_g))), ttp_pred=int(np.argmax(np.abs(th_p))),
            curve_corr=float(np.corrcoef(th_p, th_g)[0, 1]) if th_g.std() > 0 and th_p.std() > 0 else float("nan"),
            tanfrac_gt=tangential_fraction(gt, roi), tanfrac_pred=tangential_fraction(pred, roi),
            amp_ratio=float(pred_mag / gt_mag) if gt_mag > 0 else float("nan"),
            epe_rel=float(epe / gt_mag) if gt_mag > 0 else float("nan"),
        ))
    if skipped:
        print(f"[{name}] skipped {skipped} prediction files without matching GT/mask or shape", file=sys.stderr)
    return rows


def _corr(a, b):
    a, b = np.asarray(a, float), np.asarray(b, float)
    return float(np.corrcoef(a, b)[0, 1]) if len(a) > 4 and a.std() > 0 and b.std() > 0 else float("nan")


def summarize(rows, min_group):
    """Per-group summary dict for one experiment's rows."""
    by = defaultdict(list)
    for r in rows:
        by[r["group"]].append(r)
    out = {}
    for g in GROUP_ORDER + sorted(set(by) - set(GROUP_ORDER)):
        R = by.get(g, [])
        if len(R) < min_group:
            continue
        pp = np.array([r["peak_pred"] for r in R])
        pg = np.array([r["peak_gt"] for r in R])
        am = np.array([r["level"] in ("Apex", "Mid") for r in R])
        # patient-level late-systolic direction on apex+mid slices
        pat_g, pat_p = defaultdict(list), defaultdict(list)
        for r in R:
            if r["level"] in ("Apex", "Mid"):
                pat_g[r["patient"]].append(r["late_gt"])
                pat_p[r["patient"]].append(r["late_pred"])
        mg = np.array([np.mean(v) for v in pat_g.values()])
        mp = np.array([np.mean(pat_p[k]) for k in pat_g])
        out[g] = dict(
            n=len(R), patients=len({r["patient"] for r in R}),
            peak_corr=_corr(pp, pg),
            sign_agree=float(np.mean(np.sign(pp) == np.sign(pg))),
            sign_agree_am=float(np.mean(np.sign(pp[am]) == np.sign(pg[am]))) if am.any() else float("nan"),
            pat_sign_agree_am=float(np.mean(np.sign(mg) == np.sign(mp))) if len(mg) else float("nan"),
            gt_reversed_frac=float(np.mean(mg > 1)) if len(mg) else float("nan"),
            peak_ratio=float(np.median(np.abs(pp)) / np.median(np.abs(pg))),
            amp_ratio=float(np.nanmedian([r["amp_ratio"] for r in R])),
            epe_rel=float(np.nanmean([r["epe_rel"] for r in R])),
            curve_corr=float(np.nanmean([r["curve_corr"] for r in R])),
            ttp_mae=float(np.mean([abs(r["ttp_pred"] - r["ttp_gt"]) for r in R])),
            tanfrac=f"{np.nanmedian([r['tanfrac_pred'] for r in R]):.2f}/{np.nanmedian([r['tanfrac_gt'] for r in R]):.2f}",
        )
    return out


COLUMNS = [
    ("n", "n", "{:d}"), ("patients", "pts", "{:d}"),
    ("peak_corr", "peak-corr", "{:.2f}"), ("sign_agree", "sign", "{:.2f}"),
    ("sign_agree_am", "sign(A+M)", "{:.2f}"), ("pat_sign_agree_am", "pt-sign(A+M)", "{:.2f}"),
    ("gt_reversed_frac", "GT-rev", "{:.2f}"), ("peak_ratio", "|pk|p/g", "{:.2f}"),
    ("amp_ratio", "|u|p/g", "{:.2f}"), ("epe_rel", "epe_rel", "{:.2f}"),
    ("curve_corr", "curve-corr", "{:.2f}"), ("ttp_mae", "ttp-MAE", "{:.1f}"), ("tanfrac", "tanfrac p/g", "{}"),
]


def print_table(name, summary):
    print(f"\n=== {name} ===")
    head = f"{'group':12s}" + "".join(f"{h:>13s}" for _, h, _ in COLUMNS)
    print(head)
    for g, s in summary.items():
        print(f"{g:12s}" + "".join(f"{fmt.format(s[k]):>13s}" for k, _, fmt in COLUMNS))
    print("  sign = fraction of slices whose predicted peak-rotation sign matches GT; (A+M) = apex+mid slices only;"
          "\n  pt-sign = patient-level late-systolic direction agreement; GT-rev = fraction of GT patients rotating the reversed way;"
          "\n  |pk|p/g = median |peak rotation| pred/GT; |u|p/g = median displacement magnitude pred/GT; epe_rel: 1.0 == predicting zero motion.")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("experiments", nargs="+", help="experiment dirs or inference_disps dirs")
    ap.add_argument("--data", help="dataset root containing dense/<split>/ (default: from the experiment config)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--metadata", default=DEFAULT_METADATA, help="processed_excel.json")
    ap.add_argument("--csv", help="write per-slice metrics to this CSV")
    ap.add_argument("--min-group", type=int, default=5, help="hide groups with fewer slices than this")
    args = ap.parse_args()

    metadata = json.load(open(args.metadata)) if os.path.exists(args.metadata) else {}
    if not metadata:
        print(f"warning: metadata {args.metadata} not found; all slices land in group 'Other'", file=sys.stderr)

    all_rows = []
    for exp in args.experiments:
        name, pred_dir, data_root = resolve_experiment(exp, args.data)
        rows = score_experiment(name, pred_dir, data_root, metadata, split=args.split)
        if not rows:
            print(f"[{name}] no scoreable files in {pred_dir}", file=sys.stderr)
            continue
        print_table(f"{name}  ({pred_dir.relative_to(pred_dir.parents[1])}, GT: {data_root.name}/{args.split})",
                    summarize(rows, args.min_group))
        all_rows.extend(rows)

    if args.csv and all_rows:
        with open(args.csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(all_rows[0].keys()))
            w.writeheader()
            w.writerows(all_rows)
        print(f"\nper-slice metrics written to {args.csv}")


if __name__ == "__main__":
    main()
