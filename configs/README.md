# Configs

Each JSON file here holds every hyperparameter for one training or inference
run. The scripts load their default config automatically; point them at another
file to run a different experiment without touching code:

```bash
python train.py --config configs/my_experiment.json
python inference.py --config configs/my_inference.json
```

## Training configs

| Config | Script | Variant |
| --- | --- | --- |
| `train.json` | `train.py` | contour condition |
| `train_both_contour_motion_full_region.json` | `train_both_contour_motion_full_region.py` | full-region noise, contour + motion conditions |
| `train_both_contour_motion.json` | `train_both_contour_motion_full_region.py` | contour-region noise, contour + motion conditions |

The last two run the **same script** — the only difference between them is
`diffusion.contour_noise_only`, so the second is selected with `--config`.

## Inference configs

Each inference script loads the matching config below and generates samples from
the checkpoint of the experiment named in it. Fields the config shares with
training (`data`, `model`, `diffusion`, `training`) mean the same thing and
**must match the run that produced the checkpoint** — the model/diffusion shape
has to match the saved weights, and `experiment_name` selects which checkpoint
folder to load from. The extra `inference` block holds the per-run runtime
knobs, each of which can also be passed on the CLI to override the config for a
single run:

```bash
python inference.py --config configs/inference.json
python inference.py --video_type dense --start 0 --end 50   # override the config's inference block
```

| Config | Script | Variant |
| --- | --- | --- |
| `inference.json` | `inference.py` | contour condition |
| `inference_both_contour_motion_full_region.json` | `inference_both_contour_motion_full_region.py` | full-region noise, contour + motion conditions |

## Config tracking

On startup every script saves the config it was given to
`./{experiment_name}/config.json`, next to the checkpoints, so you can always
see which hyperparameters a run used. If a snapshot from an earlier run exists
with different contents (e.g. you changed a value before resuming), the old
snapshot is kept as `config_<timestamp>.json` before the new one is written. The
inference scripts accept `--no_config_snapshot` to skip the write; the training
scripts always write it.

## Fields

### top level

- `experiment_name` — names the results/checkpoint folder.

### `data`

- `base_path` — dataset root. The `*_subdir` entries are joined onto it to form
  the input video, contour-condition, motion-condition, and sampling condition
  folders. Scripts only read the subdir keys they use, so only the
  `both_contour_motion` configs carry the `motion_*` keys.
- `motion_condition_subdir` / `sampling_motion_condition_subdir` —
  contour + motion configs only: the motion-condition (displacement field)
  folders for training and for preview sampling. The contour and motion folders
  are paired by **sorted filename order**, so each must hold the same number of
  `.npy` files under matching names — the Dataset asserts on the count, but a
  name mismatch would silently mis-pair.
- In the inference configs the `sampling_*_subdir` entries are templates
  containing `{video_type}`, filled in from the resolved `video_type` at launch.
  `gt_disp_dense_subdir` supplies the ground truth loaded for comparison under
  `video_type: dense`.

### `model` (Unet3D)

- `dim`, `cond_dim`, `channels`, `dim_mults` — passed straight through.

### `diffusion` (GaussianDiffusion)

- `image_size` — H == W.
- `num_frames` — frames per clip.
- `timesteps` — number of diffusion steps T.
- `loss_type` — `"l1"` or `"l2"`.
- `contour_noise_only` — true = noise only inside the mask contour region,
  false = full image.

`image_size` and `num_frames` must match the actual shape of your `.npy` files —
nothing resizes or resamples, and a mismatch asserts on the first step.

The contour + motion configs add three more:

- `disp_scale` (default **5.0**) — the divisor applied to displacement fields
  before diffusion, and the multiplier applied to every output afterwards.

  Unlike the other fields here it describes the *units the weights are in*, not
  a hyperparameter you can retune on a resume. Changing it means training from
  scratch: the trainer stamps the scale into each checkpoint and **refuses to
  load one saved under a different value**, naming both numbers. A checkpoint
  with no recorded scale is assumed to be 5.0 — it was hardcoded to that before
  it became configurable — so existing checkpoints keep loading at the default
  and fail loudly under anything else. Keep the training config and the matching
  inference config in agreement; a mismatch would rescale every saved
  displacement field.

  It also changes the loss scale: the eps-prediction floor moves with `Var[x0]`,
  i.e. by `(5/disp_scale)²`, so **loss curves are not comparable across
  scales**. The dimensionless diagnostics below cancel it, and are the right
  thing to compare between runs.
- `disp_metrics` (default **true**) — master switch for the ROI displacement
  metrics block. Set it to `false` to reproduce a run from before these metrics
  existed: nothing extra is computed and the logs carry only `loss` and
  `disp_recon_mse`. They are diagnostics computed under `no_grad`, so this only
  changes what is logged (and a little compute) — **the gradients and the
  trained weights are identical either way**.
- `disp_rel_metric` — which **motion-normalized displacement ratio** to log
  alongside `disp_recon_mse`, scored over the contour ROI only:
  - `"mse"` (default) — `mse_disp_rel` = `mean‖Δu‖² / mean‖u_GT‖²` (normalized
    MSE, i.e. 1−R²).
  - `"epe"` — `epe_disp_rel` = `mean‖Δu‖₂ / mean‖u_GT‖₂`.
  - `null` — no ratio.

  Both ratios are dimensionless: 0 is perfect, **1.0 is exactly as bad as
  predicting zero motion**, >1 is worse than that. Being ratios they cancel the
  `disp_scale` normalization and any per-sample displacement unit scale, neither
  of which `disp_recon_mse` is immune to. They are **not** each other's square —
  never compare a run logged under one form against a run logged under the
  other.

  The ROI comes from the contour condition alone — the motion condition never
  enters it — so these numbers are directly comparable against a contour-only
  run on the same data. Note they are single-step estimates of `x0` at a
  randomly drawn `t`, so the per-step value is noisy and much larger than the
  same metric computed on fully sampled outputs. Read the epoch/cumulative
  curves.

The contour + motion entry script also accepts `loss_disp_normalize` (default
**false**) with `loss_disp_norm_bounds` and `loss_disp_norm_ema_decay`, which
divide each sample's loss by its own ground-truth motion so a hypokinetic case
is not treated as easy merely because its heart barely moves. Unlike
`disp_metrics` this **is the objective**: gradients and weights differ, so a run
with it on is a different model and its `loss` is a different quantity from a
baseline run's. None of the shipped configs enable it.

### `training`

- `batch_size`, `lr`, `num_steps`, `gradient_accumulate_every`, `ema_decay`,
  `amp`, `save_and_sample_every`.
- `resume` — true = load the latest checkpoint (milestone -1) before training,
  falling back to a fresh start if none exists; false = always start from
  scratch.

Every `save_and_sample_every` steps the trainer saves a checkpoint and samples
preview videos alongside it, so you can watch quality progress without stopping
the run. Previews use DDPM sampling and never affect the training loss.

### `logging`

- `use_wandb` (default false), `wandb_project`. The `wandb` import is guarded,
  so training runs fine without the package installed.

### `frame_validity` (optional)

Clips resampled to a fixed `num_frames` may only have their first N frames real,
the rest padding. Point this block at a CSV listing the real count per video and
the loss will ignore the padding:

- `csv_path` — the CSV.
- `filename_col` / `valid_frames_col` — which columns to read.
- `missing` — what to do with files absent from the CSV: `"full"` treats them as
  fully valid, `"error"` raises so you can guarantee full coverage.

`valid_frames = N` means frames `[0:N]` are real and `[N:num_frames]` are
padding. Omit the block entirely and the loss is byte-for-byte the original
all-frames loss.

### `inference`

Per-run knobs, each overridable by the CLI flag of the same name:

- `milestone` — checkpoint to load; `-1` is the latest.
- `video_type` — `cine` or `dense`. `dense` also loads ground-truth
  displacements for comparison.
- `start` / `end` — index range into the sorted condition folder.
- `seed` — seed the initial noise so runs are comparable.
- `sampler` — `ddpm`, which walks all `timesteps` (1000) denoising steps. The
  remaining `ddim_*` keys are inert under it.
- `skip_existing` (contour + motion script only, default false) — see below.

### Resuming a partial run: `skip_existing`

Sampling writes one file per video to
`./{experiment_name}/sampling_time_sampled_{video_type}_part_videos_infos/inference_disps/`,
named after the condition file. That folder is therefore a record of what is
already done, and `--skip_existing` uses it: any video whose prediction is
already there is not sampled again. So a run killed by a job timeout, a crash,
or Ctrl-C is resumed by re-running the same command with the flag added.

```bash
python inference_both_contour_motion_full_region.py --skip_existing
```

Details worth knowing:

- **A prediction truncated by a kill is re-sampled.** A file only counts as done
  if its `.npy` header reads back and the file is as long as the header claims,
  so the half-written file a crash leaves behind does not silently stay half
  written.
- **It matches on filename, not on how the sample was produced.** Existing
  predictions are kept as they are, so do not use it to "fill in" a split whose
  finished files came from a different checkpoint, sampler, or seed — for a
  re-sample under new settings, point at a fresh `experiment_name` or delete the
  stale files first.
- **Only the `inference_disps/*.npy` prediction is checked**, not the
  `quiver_plots/*.gif` preview: a skipped video keeps whatever gif the earlier
  run wrote and gets no new one.
- `skip_existing: true` in a config's `inference` block turns it on by default
  for that config; the flag can only turn it on, never off.
