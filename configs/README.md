# Configs

Each JSON file here holds every hyperparameter for one training or inference
script, named `configs/<script_name>.json`. The scripts load their default
config automatically; point them at another file to run a different experiment
without touching code:

```bash
python train_group_dro.py --config configs/my_experiment.json
torchrun --standalone --nproc_per_node=4 train_group_dro_ddp.py --config configs/my_experiment_ddp.json
```

## Training configs

| Config | Script | Variant |
| --- | --- | --- |
| `train.json` | `train.py` | contour-region noise, contour condition |
| `train_full_region.json` | `train_full_region.py` | full-region noise, contour condition |
| `train_full_region_augmented.json` | `train_full_region_augmented.py` | full-region noise, contour condition, augmented data (`--aug N` selects `aug_Nx`) |
| `train_full_region_ddp.json` | `train_full_region_ddp.py` | full-region noise, contour condition, multi-GPU |
| `train_full_region_augmented_ddp.json` | `train_full_region_augmented_ddp.py` | full-region noise, contour condition, augmented data (`--aug N`), multi-GPU |
| `train_both_contour_motion.json` | `train_both_contour_motion.py` | contour-region noise, contour + motion conditions |
| `train_both_contour_motion_full_region.json` | `train_both_contour_motion_full_region.py` | full-region noise, contour + motion conditions |
| `train_only_motion.json` | `train_only_motion.py` | motion condition only |
| `train_group_dro.json` | `train_group_dro.py` | Group-DRO, contour condition |
| `train_group_dro_ddp.json` | `train_group_dro_ddp.py` | Group-DRO, contour condition, multi-GPU |

Multi-GPU scripts are launched with torchrun (see each script's docstring).

## Inference configs

Each `inference_*.py` loads the matching config from the table below and
generates samples from the checkpoint of the experiment named in the config.
Fields the config shares with training (`data`, `model`, `diffusion`,
`training`) mean the same thing, and **must match the run that produced the
checkpoint** — the model/diffusion shape has to match the saved weights, and
`experiment_name` selects which checkpoint folder to load from. The extra
`inference` block holds the per-run runtime knobs; each can also be passed on
the CLI to override the config for a single run:

```bash
python inference.py --config configs/inference.json
python inference.py --video_type dense --start 0 --end 50   # override the config's inference block
```

| Config | Script | Variant |
| --- | --- | --- |
| `inference.json` | `inference.py` | contour-region noise, contour condition |
| `inference_full_region.json` | `inference_full_region.py` | full-region noise, contour condition |
| `inference_both_contour_motion.json` | `inference_both_contour_motion.py` | contour-region noise, contour + motion conditions |
| `inference_both_contour_motion_full_region.json` | `inference_both_contour_motion_full_region.py` | full-region noise, contour + motion conditions |
| `inference_only_motion.json` | `inference_only_motion.py` | motion condition only |

### Augmented and Group-DRO runs

Augmentation (`train_full_region_augmented*.py`) and Group-DRO
(`train_group_dro*.py`) only change the **training loss / data pipeline**, not
the model — both train the same `motion_after_only_contour` `Unet3D` /
`GaussianDiffusion` that the contour-condition inference scripts already load.
So there is **no separate inference script** for them: sample with
`inference_full_region.py` (their checkpoints use full-region noise) pointed at
a config that matches the run. Two ready-made ones:

| Config | Script | For |
| --- | --- | --- |
| `inference_augmented.json` | `inference_full_region.py` | any `train_full_region_augmented*` checkpoint |
| `inference_group_dro.json` | `inference_full_region.py` | any `train_group_dro*` checkpoint |

```bash
python inference_full_region.py --config configs/inference_group_dro.json --video_type dense
```

Two things must be set to match the run you want to sample:

1. **`experiment_name` = the RESOLVED training folder name.** The training
   configs contain `{aug}` / `{eta_q}` / `{adjustment_c}` / `{bs}` placeholders
   that the training scripts fill at launch; the inference config needs the
   final substituted string (e.g. `...-aug_5x`,
   `...-group-dro-etaq0.04-c1.0-bs16`) so `trainer.load()` reads from the right
   `./{experiment_name}` folder. The shipped defaults are examples — edit them.
2. **The `model` / `diffusion` shape must match the checkpoint** — these
   proposal runs use `image_size: 64`, `num_frames: 26`,
   `contour_noise_only: false` (26-frame data), which is why they need their own
   configs rather than the 20-frame `inference_full_region.json`.

For augmented runs, `data.base_path` should point at the **non-augmented
sampling dataset** (the training config's `sampling_base_path`), since inference
only samples from the test split — augmentation applies to training data only.

### Faster inference: shard videos across GPUs

Each video is sampled by a 1000-step DDPM chain that can't be parallelized
internally, but the videos are independent, so the easy near-linear speedup is
to split the test set across GPUs — one process per GPU, each handling a
contiguous slice of the sorted condition files. `run_inference_multi_gpu.py`
does this for you: it reads the same config, reproduces the exact folder the
worker globs, counts the files, splits them evenly, and launches one worker per
GPU (each pinned via `CUDA_VISIBLE_DEVICES` and handed its slice via
`--start`/`--end`). All workers write per-video filenames into the same
experiment folder, so results merge automatically. No model changes, no
torchrun.

```bash
# auto-detect all GPUs, default config for the script
python run_inference_multi_gpu.py --script inference_full_region.py

# pick config + GPUs, override video_type
python run_inference_multi_gpu.py \
    --script inference_full_region.py \
    --config configs/inference_group_dro.json \
    --gpus 0,1,2,3 --video_type dense

# print the per-GPU commands without launching
python run_inference_multi_gpu.py --script inference_full_region.py --dry_run
```

Equivalent to launching the slices by hand (what the tool automates):

```bash
CUDA_VISIBLE_DEVICES=0 python inference_full_region.py --start 0  --end 33  &
CUDA_VISIBLE_DEVICES=1 python inference_full_region.py --start 33 --end 66  &
CUDA_VISIBLE_DEVICES=2 python inference_full_region.py --start 66 --end 99  &
CUDA_VISIBLE_DEVICES=3 python inference_full_region.py --start 99 --end 132 &
wait
```

## Config tracking

On startup every script saves the config it was given to
`./{experiment_name}/config.json` (next to the checkpoints), so you can always
see which hyperparameters a run used. If a snapshot from an earlier run exists
with different contents (e.g. you changed a value before resuming), the old
snapshot is kept as `config_<timestamp>.json` before the new one is written.
In DDP only rank 0 writes.

## Fields

### top level
- `experiment_name` — names the results/checkpoint folder. Group-DRO configs may
  include `{eta_q}` / `{adjustment_c}` / `{bs}` placeholders, which the group-DRO
  scripts fill at launch — `{eta_q}` / `{adjustment_c}` from the `dro` section, and
  `{bs}` from the training batch size (`batch_size` for the single-GPU script,
  `per_gpu_batch_size` for the DDP script). Each sweep point thus writes to its own
  folder / wandb run.

### `data`
- `base_path` — dataset root. The `*_subdir` entries are joined onto it to
  form the input video, contour-condition, motion-condition, and sampling
  condition folders. Scripts only read the subdir keys they use (e.g. only
  the `both_contour_motion` and `only_motion` configs have `motion_*` keys).
- `aug_subdir` — augmented config only. Sits between `base_path` and the split
  subdirs. It is a template containing `{aug}` (e.g. `"aug_{aug}x"`); the
  required `--aug N` flag on `train_full_region_augmented.py` fills it in
  (`--aug 5` → `aug_5x`) and also substitutes `{aug}` anywhere else in the
  config (e.g. `experiment_name`), so each augmentation count writes to its own
  checkpoint folder. Nothing on disk is mutated. Example:

  ```bash
  python train_full_region_augmented.py --aug 5
  ```
- `sampling_base_path` — augmented config only, optional. Dataset root for the
  sampling condition folder(s), used instead of `base_path`/`aug_subdir`. Set it
  when sampling should draw from a separate (e.g. non-augmented) dataset — the
  `sampling_*_subdir` entries are joined onto this path. When absent, sampling
  falls back to the augmented `data_root` (`base_path`/`aug_subdir`).
- `metadata_json_path` — Group-DRO only: the processed Excel metadata JSON
  (disease groups).
- `num_workers` — DataLoader workers (DDP and Group-DRO trainers only; the
  plain single-GPU `Trainer` does not take it).

### `model` (Unet3D)
- `dim`, `cond_dim`, `channels`, `dim_mults` — passed straight through.

### `diffusion` (GaussianDiffusion)
- `image_size` — H == W.
- `num_frames` — frames per clip.
- `timesteps` — number of diffusion steps T.
- `loss_type` — `"l1"` or `"l2"`.
- `contour_noise_only` — true = noise only inside the mask contour region,
  false = full image. Not present for `train_only_motion.json` (that model
  has no contour input).

### `training`
- `batch_size` (single GPU) / `per_gpu_batch_size` (DDP). In DDP the
  effective global batch is `per_gpu_batch_size * num_gpus` (for Group-DRO,
  all drawn from the one sampled group) — consider scaling `lr` if you grow
  the global batch (linear scaling is a reasonable start).
- `lr` — eta, the model learning rate.
- `num_steps`, `gradient_accumulate_every` (the Group-DRO trainers require 1),
  `ema_decay`, `amp`, `save_and_sample_every`.
- `weight_decay` — Group-DRO only: lambda, L2 weight decay in the model update.
- `resume` — true = load the latest checkpoint (milestone -1) before
  training, falling back to a fresh start if none exists; false = always
  start from scratch.

### `dro` (Group-DRO only)
- `eta_q` — group-weight learning rate. The q update multiplies a group's raw
  weight by `exp(eta_q * S_g)`, with `S_g ~= L_g`, the MEAN denoising loss
  (empirically ~0.4-1.1 here). Pick `eta_q` so `exp(eta_q * L_g)` stays near
  1 → ~0.01-0.05, NOT 1.0. With `eta_q = 1.0` each sample of a group roughly
  doubles its raw weight, so q collapses onto whichever group is sampled first
  (observed: q → ~1.0 on Healthy by step ~150). Tuning target: watch the
  printed per-group q — it should stay spread and drift toward the
  persistently-hardest group, never pin near 1.0. Dead-flat at 1/K → raise it;
  any group above ~0.7 → lower it. In DDP these act on the GLOBAL mean loss
  and do NOT need rescaling with the number of GPUs.
- `adjustment_c` — generalization-adjustment constant C, on the same mean-loss
  scale (0 = plain group-DRO).
- `freeze_q` — freeze q at uniform (1/K) and backprop the raw, unweighted
  loss, disabling DRO reweighting. Sampling is still group-balanced
  (uniform-over-groups), so this is NOT identical to base GenStrain's
  natural-distribution training. Set false for real group-DRO.
- `allowed_disease_groups` — REQUIRED list of disease groups to keep, in
  string-match PRIORITY order (first match wins); there is no hardcoded fallback
  (training errors out if it is missing/empty). A metadata disease string is
  matched by case-insensitive substring, so ordering resolves multi-label
  strings: `["LBBB", "Healthy"]` maps `"LBBB & Recovered DCM"` → `LBBB`, and a
  more specific name must precede a broader one. Each entry is either a plain
  group-name string, or a `{"name": ..., "exclude": [terms]}` object whose
  `exclude` terms VETO the match. Example — keep `Healthy` (absorbing e.g.
  `"Healthy Cardiotoxicity"`) but drop `"Healthy Pediatric"`:

  ```json
  "allowed_disease_groups": [
      "LBBB",
      { "name": "Healthy", "exclude": ["Pediatric"] }
  ]
  ```

  A group is kept only if its name is a substring AND none of its `exclude`
  terms are. `exclude` is case-insensitive and position-independent.

### `logging` (optional)
Weights & Biases loss monitoring. The whole block is optional — omit it and
logging stays off (the code defaults to `use_wandb=false`). Console per-step
loss printing is unaffected either way.
- `use_wandb` — true = stream training loss to wandb. Requires
  `pip install wandb` and a one-time `wandb login`. In the DDP trainers only
  rank 0 logs. Defaults to false. Every trainer logs, per step, the raw `loss`
  and `loss_cummean` (the true cumulative mean of the loss since the start of
  the run — `sum / count` — a smooth trend line). For Group-DRO it additionally
  logs `weighted_loss`, `dro_score`, `dro_q`, a per-group weight curve
  `q_<group>`, and per-group loss curves — `loss_<group>` (the raw loss on the
  step that group was sampled, so each such series is sparse) and
  `loss_<group>_cummean` (the group's cumulative mean loss, dense/carried
  forward). To compare how hard each group is / whether the hard ones are
  improving, watch the `loss_<group>_cummean` curves. NOTE: the cumulative-mean
  accumulators are not checkpointed, so they restart from zero when you resume a
  run. (In DDP the per-group loss is the GLOBAL group loss all-reduced across
  ranks.)
- `wandb_project` — wandb project name the run is created under. The run is
  named after `experiment_name`, and `resume="allow"` continues the same-named
  run when you restart from a checkpoint. Defaults to
  `genstrain-motion-diffusion`. The full config is uploaded as the run config.

### `inference` (inference configs only)
Per-run runtime knobs for the `inference_*.py` scripts. Each field also has a
matching CLI flag (`--video_type`, `--start`, `--end`, `--milestone`); when the
flag is passed it wins, otherwise the config value is used. On startup each
inference script also saves a config snapshot to
`./{experiment_name}/config.json` (see [Config tracking](#config-tracking)).
- `milestone` — checkpoint milestone to load; `-1` = the latest checkpoint.
- `video_type` — which test split to sample conditions from: `cine` or `dense`.
  `dense` also loads ground-truth displacement (from `gt_disp_dense_subdir`) so
  it can be saved alongside the sample; `cine` has no ground truth.
- `start`, `end` — index range into the sorted list of condition `.npy` files
  to sample; `end: null` means "through the last file".

### inference `data` subdir keys
The inference configs reuse the training `data` keys and add a few for the
test-split condition folders and (for `dense` runs) the ground-truth
displacement. `*_subdir_template` values contain `{video_type}`, filled in from
the resolved `video_type` at runtime so one config covers every split.
- `contour_test_subdir_template` — contour-mask test folder, e.g.
  `"{video_type}/test/{video_type}_mask"` (contour configs).
- `motion_test_subdir_template` — motion-condition test folder, e.g.
  `"tlrn_{video_type}_mask_motion/test"` (motion configs).
- `gt_disp_dense_subdir` — ground-truth displacement folder used when
  `video_type == "dense"`.
