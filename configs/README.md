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
| `train_both_contour_motion_full_region_ddp.json` | `train_both_contour_motion_full_region_ddp.py` | full-region noise, contour + motion conditions, multi-GPU |
| `train_both_contour_motion_full_region_disp_norm.json` | `train_both_contour_motion_full_region.py` | full-region noise, contour + motion conditions, **motion-normalized loss** (`loss_disp_normalize`) |
| `train_both_contour_motion_full_region_disp_norm_ddp.json` | `train_both_contour_motion_full_region_ddp.py` | full-region noise, contour + motion conditions, motion-normalized loss, multi-GPU |
| `train_only_motion.json` | `train_only_motion.py` | motion condition only |
| `train_group_dro.json` | `train_group_dro.py` | Group-DRO, contour condition |
| `train_group_dro_ddp.json` | `train_group_dro_ddp.py` | Group-DRO, contour condition, multi-GPU |
| `train_inverse_freq.json` | `train_inverse_freq.py` | inverse-frequency reweighting, contour condition |
| `train_inverse_freq_ddp.json` | `train_inverse_freq_ddp.py` | inverse-frequency reweighting, contour condition, multi-GPU |

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
python inference.py --sampler ddim --ddim_steps 50          # few-step sampling, same weights
```

| Config | Script | Variant |
| --- | --- | --- |
| `inference.json` | `inference.py` | contour-region noise, contour condition |
| `inference_full_region.json` | `inference_full_region.py` | full-region noise, contour condition |
| `inference_both_contour_motion.json` | `inference_both_contour_motion.py` | contour-region noise, contour + motion conditions |
| `inference_both_contour_motion_full_region.json` | `inference_both_contour_motion_full_region.py` | full-region noise, contour + motion conditions |
| `inference_both_contour_motion_full_region_disp_norm.json` | `inference_both_contour_motion_full_region.py` | samples the motion-normalized-loss checkpoint (same weights shape, different `experiment_name`) |
| `inference_only_motion.json` | `inference_only_motion.py` | motion condition only |

### Augmented, Group-DRO and inverse-frequency runs

Augmentation (`train_full_region_augmented*.py`), Group-DRO
(`train_group_dro*.py`) and inverse-frequency reweighting
(`train_inverse_freq*.py`) only change the **training loss / data pipeline**, not
the model — all train the same `motion_after_only_contour` `Unet3D` /
`GaussianDiffusion` that the contour-condition inference scripts already load.
So there is **no separate inference script** for them: sample with
`inference_full_region.py` (their checkpoints use full-region noise) pointed at
a config that matches the run. Three ready-made ones:

| Config | Script | For |
| --- | --- | --- |
| `inference_augmented.json` | `inference_full_region.py` | any `train_full_region_augmented*` checkpoint |
| `inference_group_dro.json` | `inference_full_region.py` | any `train_group_dro*` checkpoint |
| `inference_inverse_freq.json` | `inference_full_region.py` | any `train_inverse_freq*` checkpoint |

```bash
python inference_full_region.py --config configs/inference_group_dro.json --video_type dense
```

Two things must be set to match the run you want to sample:

1. **`experiment_name` = the RESOLVED training folder name.** The training
   configs contain `{aug}` / `{eta_q}` / `{adjustment_c}` / `{bs}` placeholders
   that the training scripts fill at launch; the inference config needs the
   final substituted string (e.g. `...-aug_5x`,
   `...-group-dro-etaq0.04-c1.0-bs16`, `...-inverse-freq-bs16`) so
   `trainer.load()` reads from the right
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
  folder / wandb run. The inverse-frequency configs support `{bs}` only — the
  reweighting has no hyperparameters to sweep.

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
- `sampling_base_path` — optional; read by the DDP, augmented, Group-DRO and
  inverse-frequency scripts (**not** by the plain single-GPU `train.py`,
  `train_full_region.py`, `train_both_contour_motion*.py` or
  `train_only_motion.py`, which always join the `sampling_*_subdir` entries onto
  `base_path`). Dataset root for the sampling condition folder(s), used instead
  of `base_path` (or, in the augmented configs, `base_path`/`aug_subdir`). Set it
  when sampling should draw from a separate (e.g. non-augmented) dataset — the
  `sampling_*_subdir` entries are joined onto this path. When absent, sampling
  falls back to the training data root.
- `motion_condition_subdir` / `sampling_motion_condition_subdir` — the
  `both_contour_motion` and `only_motion` configs only: the motion-condition
  (displacement field) folders for training and for preview sampling. The
  contour and motion folders are paired by **sorted filename order**, so each
  must hold the same number of `.npy` files under matching names — the Dataset
  asserts on the count, but a name mismatch would silently mis-pair.
- `metadata_json_path` — Group-DRO and inverse-frequency only: the processed
  Excel metadata JSON (disease groups).
- `num_workers` — DataLoader workers (DDP, Group-DRO and inverse-frequency
  trainers only; the plain single-GPU `Trainer` does not take it).

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
- `disp_scale` (default **5.0**) — the divisor applied to displacement fields
  before diffusion, and the multiplier applied to every output afterwards. It was
  hardcoded to `5.0` until it became configurable, so **every checkpoint trained
  before then assumes 5.0** — which is why that is the default.

  Unlike the other fields here it describes the *units the weights are in*, not a
  hyperparameter you can retune on a resume. Changing it means training from
  scratch: the trainers stamp the scale into each checkpoint and **refuse to load
  one saved under a different value**, naming both numbers. A checkpoint with no
  recorded scale is assumed to be 5.0, so existing checkpoints keep loading at the
  default and fail loudly under anything else. Keep the training config and the
  matching `inference_*.json` in agreement — inference reads this field too, and
  a mismatch would rescale every saved displacement field.

  It changes the loss scale: the eps-prediction floor moves with `Var[x0]`, i.e.
  by `(5/disp_scale)²`, so **loss curves are not comparable across scales**. The
  dimensionless diagnostics are: `mse_disp_rel`, `epe_disp_rel` and Group-DRO's
  `r_g` all cancel it, and are the right things to compare between runs.
- `disp_metrics` (default **true**) — master switch for the ROI displacement
  metrics block. Not present for `train_only_motion.json`: that model has no
  contour input, so it has no ROI to score over and computes none of these
  metrics (neither this field nor `disp_rel_metric` below applies to it). Set it
  to `false` to **reproduce a run from before these metrics
  existed**: nothing extra is computed and the logs carry only `loss` and
  `disp_recon_mse` — not even `gt_disp_msq`, which `disp_rel_metric: null` still
  emits. The metrics are diagnostics computed under `no_grad`, so this only
  changes what is logged (and a little compute) — **the gradients and the trained
  weights are identical either way**, whichever setting you use.

  One interaction, and it fails loudly rather than silently: Group-DRO's
  `score_normalize` reads `gt_disp_msq` from this block, so setting
  `disp_metrics: false` alongside `dro.score_normalize: true` raises at startup.
  To reproduce a pre-normalization DRO run, turn both off.
- `disp_rel_metric` — which **motion-normalized displacement ratio** to log
  alongside `disp_recon_mse`, scored over the contour ROI only:
  - `"mse"` (default) — `mse_disp_rel` = `mean‖Δu‖² / mean‖u_GT‖²` (normalized
    MSE, i.e. 1−R²).
  - `"epe"` — `epe_disp_rel` = `mean‖Δu‖₂ / mean‖u_GT‖₂`. Same definition and
    name as the `epe_disp_rel` column StrainAnalysis writes, so the training
    curve and the paper table are the same quantity.
  - `null` — no ratio.

  Both ratios are dimensionless: 0 is perfect, **1.0 is exactly as bad as
  predicting zero motion**, >1 is worse than that. Being ratios they cancel the
  `/5` normalization and any per-sample displacement unit scale, neither of
  which `disp_recon_mse` is immune to. They are **not** each other's square —
  never compare a run logged under one form against a run logged under the other.

  `gt_disp_msq` (mean `‖u_GT‖²` over the ROI) is logged regardless of this
  setting, because Group-DRO consumes it as the score normalizer below. Use
  `disp_metrics: false` above to suppress that too.

  In the contour **+ motion** configs the ROI is still taken from the contour
  condition alone — the motion condition never enters it — so these numbers are
  directly comparable against a contour-only run on the same data.

  Note these are single-step estimates of `x0` at a randomly drawn `t`, so the
  per-step value is noisy and much larger than the same metric computed on fully
  sampled outputs. Read the epoch/cumulative curves, and keep the authoritative
  numbers in the StrainAnalysis pipeline.
- `loss_disp_normalize` (default **false**) — divide each sample's loss by its
  own ground-truth motion, so a hypokinetic case is not treated as easy merely
  because its heart barely moves. **Contour + motion configs only** — the other
  entry scripts do not pass the field.

  Why the eps-loss needs it: `eps − eps_hat = −sqrt(a/(1−a)) · (x0 − x0_hat)`, so
  the loss a sample can reach carries the displacement amplitude **squared**. Two
  patients at identical *relative* accuracy therefore log very different losses,
  and the quieter one contributes proportionally less gradient. The divisor is
  `mean‖u_GT‖²` over the contour ROI — the same `gt_disp_msq` the metrics above
  report, from one shared definition — which is the dimensionally right divisor
  for a squared error.

  What actually divides the loss is the **ratio** `w_i = clamp(msq_i / ref, lo,
  hi)`, not a raw `1/msq`, where `ref` is an EMA of the batch-mean msq:
  - `mean(w) ≈ 1`, so the loss keeps its usual magnitude — **`lr`,
    `max_grad_norm` and the y-axis all carry over** from an unnormalized run. A
    raw `1/msq` multiplies the loss by ~25 on `disp_scale: 5.0` data, i.e. scales
    the effective learning rate by the same factor.
  - The clamp keeps a motionless sample from dividing by ~0 and capturing the
    batch gradient; at the default bounds nothing moves more than 5x in either
    direction.
  - The EMA reference means only the *spread within* a batch matters, not whether
    that batch happened to draw vigorous hearts.

  Unlike `disp_metrics`, **this is the objective**: gradients and weights differ,
  so a run with it on is a different model, and its `loss` is a different
  quantity from a baseline run's. Three extra fields are logged to bridge that:
  `loss_unnormalized` (what `loss` would be at `w == 1`, comparable to a baseline
  curve), `disp_norm_weight` (batch-mean `w`) and `disp_norm_ref` (the EMA
  reference). Compare runs on `mse_disp_rel`, which is dimensionless either way.

  Two smaller consequences worth knowing:
  - The loss is reduced **per sample and then averaged**, where the plain
    branches pool every element of the batch into one mean. Clips now count once
    each rather than in proportion to ROI size and valid-frame count — that is
    the per-patient weighting the normalization is after, but it shifts the loss
    slightly even at `w == 1`.
  - It needs the contour condition to build the ROI, and **raises** if one is
    missing rather than silently training unnormalized.

  Under DDP the batch statistic is all-reduced before it enters the EMA, so every
  rank divides by an identical reference. The reference is carried in the
  checkpoint dict (not `state_dict`), so checkpoints stay loadable by every other
  variant of this model in both directions; a checkpoint saved before this
  existed simply re-warms the EMA from the first batch after the resume.
- `loss_disp_norm_bounds` (default **`[0.2, 5.0]`**) — `[lo, hi]` clamp on `w`.
  Only read when `loss_disp_normalize` is true. Tighten toward `[1, 1]` for a
  gentler effect; widen `lo` toward 0 to let near-static samples dominate.
- `loss_disp_norm_ema_decay` (default **0.99**) — smoothing on the reference,
  i.e. ~100 steps of memory. Only read when `loss_disp_normalize` is true. `0.0`
  makes the reference this batch's mean, which adds the batch-to-batch noise of
  the denominator straight into the effective learning rate.

### `training`
- `batch_size` (single GPU) / `per_gpu_batch_size` (DDP). In DDP the
  effective global batch is `per_gpu_batch_size * num_gpus` (for Group-DRO,
  all drawn from the one sampled group) — consider scaling `lr` if you grow
  the global batch (linear scaling is a reasonable start).
- `lr` — eta, the model learning rate.
- `num_steps`, `gradient_accumulate_every` (the Group-DRO and inverse-frequency
  trainers require 1), `ema_decay`, `amp`, `save_and_sample_every`.
- `weight_decay` — Group-DRO and inverse-frequency only: lambda, L2 weight decay
  in the model update. It is the **main knob for the inverse-frequency
  baseline**, which has no hyperparameters of its own: Sagawa et al.'s central
  finding is that reweighting needs strong regularization to beat plain ERM on
  worst-group error in overparameterized models.
- `resume` — true = load the latest checkpoint (milestone -1) before
  training, falling back to a fresh start if none exists; false = always
  start from scratch.
- `preview_sampler`, `preview_ddim_steps`, `preview_ddim_eta`,
  `preview_ddim_spacing`, `preview_ddim_clip_x_start`,
  `preview_ddim_clip_percentile` — sampler for the
  **preview sample taken every `save_and_sample_every` steps**, independent of
  the `inference` block. Same meaning as the inference fields, and `ddpm` is the
  default here too. See [Preview sampling during training](#preview-sampling-during-training).

### Preview sampling during training
Every `save_and_sample_every` steps the trainer saves a checkpoint and samples
one preview video. Under `preview_sampler: "ddpm"` that preview walks all
`diffusion.timesteps` (1000) steps, and each step costs **two** network calls,
not one: `sample_and_save*` defaults to `cond_scale=2.0`, and
`forward_with_cond_scale` runs a conditional and a null pass whenever
`cond_scale != 1` (classifier-free guidance).

So the preview is **not free** — on a 700k-step run with
`save_and_sample_every: 300` that is ~2,333 previews x 1000 steps x 2 passes =
~4.7M extra forward passes, on the order of 20% of training wall-clock. Under
DDP it is worse than it looks: only rank 0 samples, so every other rank sits at
the next collective until it finishes.

Setting `preview_sampler: "ddim"` with `preview_ddim_steps: 50` cuts that to
~233k forward passes, recovering most of the overhead:

```json
"training": {
    "save_and_sample_every": 300,
    "preview_sampler": "ddim",
    "preview_ddim_steps": 50,
    "preview_ddim_eta": 0.0
}
```

The trade-off is that previews then no longer show exactly what full DDPM
inference would produce, so treat them as a progress signal rather than a
fidelity check. `ddpm` remains the default so training behaviour is unchanged
unless you opt in. Bad values are rejected at startup, not at the first preview
thousands of steps in.

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
- `score_normalize` (default **true**) — divide `L_g` by the group's relative
  motion scale before it becomes the DRO score.

  **Why.** The achievable ε-MSE is `(ᾱ/(1-ᾱ)) · Var[x0 | x_t]`, and scaling a
  group's displacement by `k` scales `Var[x0]` — hence the loss — by `k²`. So a
  hypokinetic (diseased) group has a genuinely *lower loss floor*. DRO reads the
  raw loss level rather than the excess over that floor, concludes the group is
  already solved, and steers q away from it. Normalizing leaves DRO comparing
  how well each group is served *relative to how much its hearts actually move*.
  The divisor is the mean **square** displacement (`gt_disp_msq`), not the
  magnitude, because that is what scales with the loss.

  **What it does and does not touch.** Only the scalar fed to the q update. The
  backward stays `q_g * loss` — normalizing per-sample gradients would be an
  unbounded multiplier exactly on the near-static samples. The divisor is `r_g =
  msq_g / mean_g(msq)`, a *relative* scale centered on 1.0 and clamped to
  `[0.25, 4]`, so the score keeps its usual units and **`eta_q` /
  `adjustment_c` need no retuning**. `r_g` is logged per step as
  `dro_motion_scale`, and `gt_disp_msq_<group>_cummean` shows each group's scale.

  Set `false` to reproduce runs from before this existed. Note the correction is
  weakest early in training: at init the model predicts ≈0 and the ε-loss is ≈1
  per dimension for *every* group regardless of motion, so the division briefly
  over-corrects. Watch the `q_<group>` curves over the first few thousand steps.
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

### `reweight` (inverse-frequency only)

The second group-robustness baseline, beside Group-DRO. Batches are drawn from
the **natural** training distribution (so a batch is a natural mixture of
groups), and each sample's loss is multiplied by a **static** weight inversely
proportional to its group's size. With `K` groups, `n_g` samples in group *g*
and `N = sum_g n_g`:

```
w_g = (1/K) / (n_g/N) = N / (K * n_g)
```

which is the importance ratio mapping the natural group distribution onto the
uniform one, so the objective becomes the plain average of per-group mean
losses, `(1/K) * sum_g L_g` — a small group counts as much as a large one. The
weights are computed once from the group counts and never change; that is the
whole contrast with Group-DRO, which re-estimates `q` every step.

The weights are normalized **within each batch** (`sum_i w_i*l_i / sum_i w_i`;
in DDP over the global batch across all ranks), which keeps the weighted loss on
the same scale as an unweighted one and so pins the effective learning rate. A
side effect is that any constant factor in `w` cancels — the "INS" convention
`(1/n_g) / sum_j(1/n_j) * K` differs from the above by exactly such a constant
and trains identically.

There are **no tuning knobs by design**; the only field is:

- `allowed_disease_groups` — REQUIRED, identical semantics and ordering rules to
  `dro.allowed_disease_groups` above (first-match-wins case-insensitive substring
  matching, with optional `{"name", "exclude"}` vetoes). It feeds the same
  dataset class.

**Relationship to `dro.freeze_q`.** Both optimize the same objective, so do not
mistake the two runs for duplicates — they differ in mechanism. `freeze_q` draws
a **pure single-group batch** per step (an artifact of Group-DRO needing a
per-group loss estimate) and leaves the loss unweighted; this baseline draws each
sample at its natural rate and weights the loss instead. The published
reweighting baseline is also sometimes implemented as *resampling* (a
`WeightedRandomSampler` with weight `1/n_g` and an unweighted loss); that form is
deliberately **not** used here, because it revisits each minority video roughly
`K*n_maj/n_min` times as often as a majority one, which on a small dataset is
direct memorization pressure on exactly the group the method is meant to help.

On startup the trainer prints the resolved weight table next to `group counts:`,
e.g. `inverse-frequency weights w_g = N/(K*n_g): LBBB=5.5000, Healthy=0.5500`.

### `frame_validity` (optional) — valid-frame loss masking
Clips are resampled to `diffusion.num_frames` (e.g. 26), but a clip may only
have the first `N` frames real and the rest padding. This block makes the
training loss ignore the padding frames, per sample, using a CSV of valid-frame
counts. It is **optional and off by default**: omit the block, or set
`csv_path` to `null`, and the loss is byte-for-byte the original all-frames
loss. Supported by every trainer (single-GPU, DDP, Group-DRO, augmented) and all
three model variants (contour, contour+motion, motion-only).

Convention: `valid_frames = N` means frames `[0:N]` are valid and
`[N:num_frames]` are padding (leading-N). The mask is applied to the training
`loss` **and** the `disp_recon_mse` diagnostic; the diffusion sampling process is
unchanged.

- `csv_path` — path to a CSV with one row per training video. `null`/absent =
  masking off. The shipped configs whose dataset is
  `all_data_resampled_26_frames` point at that dataset's `frame_counts_train.csv`
  and are ON; the augmented / Group-DRO configs (different datasets) ship with
  `csv_path: null` — set it to that dataset's own frame-counts CSV to enable.
- `filename_col` — CSV column with the video filename (default `dense_filename`).
  Matched against the training target's `.npy` filename; the `.npy` suffix is
  optional and matching is exact otherwise.
- `valid_frames_col` — CSV column with the leading valid-frame count (default
  `dense_valid_frames`). Values are clamped to `[1, num_frames]`.
- `missing` — what to do when a training file isn't in the CSV: `"full"`
  (default) treats all frames as valid (no masking for that sample, robust to
  gaps); `"error"` raises, if every sample must be listed.

```json
"frame_validity": {
    "csv_path": "/…/all_data_resampled_26_frames/frame_counts_train.csv",
    "filename_col": "dense_filename",
    "valid_frames_col": "dense_valid_frames",
    "missing": "full"
}
```

On startup a trainer prints `valid-frame loss masking ON (<N> CSV entries)` when
it is enabled (rank 0 only in DDP), so you can confirm it took effect.

**One difference for inverse-frequency runs.** Those trainers ask the loss for
per-sample values (they need one number per sample to weight), so the frame mask
normalizes **per sample** rather than jointly over the batch: every clip is
weighted equally, instead of in proportion to its valid-frame count as the
batch-joint reduction does. Arguably the more correct behavior, but it is a real
difference from the other trainers — worth noting if you compare a
frame-masked inverse-frequency run against a frame-masked Group-DRO one. With
`csv_path: null` (the shipped default) the two are identical.

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

  The inverse-frequency trainers log the same `loss_<group>` /
  `loss_<group>_cummean` keys — deliberately, so the two baselines overlay
  directly on one chart — plus `unweighted_loss` (the plain batch mean, for an
  apples-to-apples curve against runs that do not reweight) and the static
  per-group weights `w_<group>`, logged once at step 0. Here `loss` is the
  weighted objective actually being minimized. Because the loss is computed per
  sample, the per-group values are exact means over every sample of that group
  seen so far, not a per-step approximation; `loss_<group>` is still sparse
  (only groups present in that batch).
- `wandb_project` — wandb project name the run is created under. The run is
  named after `experiment_name`, and `resume="allow"` continues the same-named
  run when you restart from a checkpoint. Defaults to
  `genstrain-motion-diffusion`. The full config is uploaded as the run config.

### `inference` (inference configs only)
Per-run runtime knobs for the `inference_*.py` scripts. Each field also has a
matching CLI flag (`--video_type`, `--start`, `--end`, `--milestone`,
`--sampler`, `--ddim_steps`, `--ddim_eta`); when the flag is passed it wins,
otherwise the config value is used. On startup each inference script also saves
a config snapshot to `./{experiment_name}/config.json`
(see [Config tracking](#config-tracking)).
- `milestone` — checkpoint milestone to load; `-1` = the latest checkpoint.
- `video_type` — which test split to sample conditions from: `cine` or `dense`.
  `dense` also loads ground-truth displacement (from `gt_disp_dense_subdir`) so
  it can be saved alongside the sample; `cine` has no ground truth.
- `start`, `end` — index range into the sorted list of condition `.npy` files to
  sample. Every config ships with `start: 0` / `end: null`, i.e. **all videos in
  the folder**; `end: null` means "through the last file". Narrow the range only
  to sample a subset — and note `run_inference_multi_gpu.py` sets it per worker,
  so leave it open there.
- `sampler` — `ddpm` (default) or `ddim`. See
  [Choosing a sampler](#choosing-a-sampler) below.
- `ddim_steps` — number of denoising steps when `sampler` is `ddim`. Ignored
  under `ddpm`, which always walks all `diffusion.timesteps` steps.
- `ddim_eta` — DDIM stochasticity when `sampler` is `ddim`. `0.0` (default) is
  the deterministic path; `1.0` reproduces the DDPM posterior noise level.
  Ignored under `ddpm`.
- `ddim_spacing` — `uniform` (default, evenly spaced in t) or `logsnr` (evenly
  spaced in log-SNR). See [Tuning DDIM](#tuning-ddim-when-a-low-step-count-looks-bad).
- `ddim_clip_x_start` — `null` (default, no clipping), `"dynamic"`, or a float
  bound. Bounds the predicted x0 each DDIM step.
- `ddim_clip_percentile` — percentile for `"dynamic"` (default `0.995`).
- `seed` — seeds the initial noise `x_T`; `null` (default) draws randomly. Needed
  to compare step counts fairly.

### Choosing a sampler
Both samplers run the **same trained weights** — DDIM needs no retraining and no
checkpoint changes. The difference is only how many times the network is called
to denoise one video:

| `sampler` | denoise calls | notes |
| --- | --- | --- |
| `ddpm` (default) | `diffusion.timesteps` (1000) | the original ancestral sampler; unchanged, still the reference output |
| `ddim` | `ddim_steps` | strided subsequence of the same chain; ~`1000 / ddim_steps` faster |

`ddpm` stays the default so existing runs and saved results are untouched. Reach
for `ddim` when sampling a whole test split is the bottleneck:

```bash
# 50 steps instead of 1000, deterministic
python inference_full_region.py --sampler ddim --ddim_steps 50

# shard the same DDIM run across GPUs
python run_inference_multi_gpu.py --script inference_full_region.py \
    --sampler ddim --ddim_steps 50
```

`ddim_steps` trades speed for fidelity, so **sweep it against your DDPM baseline
on a few videos before trusting it for a full split** — compare the saved
`inference_disps/*.npy` against the `ddpm` output for the same conditions and
pick the smallest step count whose displacement error is acceptable. 50 is a
common starting point, not a validated default for this data.

### Tuning DDIM when a low step count looks bad
If `ddim_steps: 50` degrades but a large count (e.g. 400) is fine, the problem is
usually **not** that the data needs many steps. It is that uniform-in-t striding
is coarsest exactly where this schedule curves hardest.

`cosine_beta_schedule` clips betas at `0.9999`, so `alphas_cumprod` drops ~10,000x
between t=998 and t=999 — a log-SNR cliff at the very top of the chain. And
`predict_start_from_noise` scales by `1/sqrt(alphas_cumprod[t])`, about **6.5e4 at
t=999**, so a small eps error there becomes a huge x0 error. DDPM's posterior
coefficients damp that; DDIM at `eta=0` does not, because the first step's x0 sets
the entire trajectory. Measured worst single log-SNR jump:

| spacing | `ddim_steps` | max jump | steps at t>=900 |
| --- | --- | --- | --- |
| `uniform` | 50 | 15.20 | 5 |
| `uniform` | 400 | 11.41 | 40 |
| `logsnr` | 50 | **9.21** | 15 |

`logsnr` at 50 steps is better conditioned than `uniform` at 400 — which is the
whole point of the knob. Two fields address this:

- `ddim_spacing: "logsnr"` — spend steps where the noise level actually moves
  instead of uniformly in t. Try this **first**; it costs nothing.
- `ddim_clip_x_start: "dynamic"` — bound the predicted x0 each step, killing the
  6.5e4x amplification. Per-sample percentile clip (`ddim_clip_percentile`,
  default `0.995`) with **no** rescale, since the Imagen-style `clamp(-s,s)/s`
  assumes [-1,1] data and these fields are normalized by /5. A float instead of
  `"dynamic"` clamps to a fixed `[-v, v]`.

Also sweep `cond_scale`. Guidance overshoot is self-correcting under stochastic
sampling and accumulates under deterministic few-step sampling, so a value tuned
at 1000 DDPM steps is often too strong for 50-step DDIM.

Set `seed` to compare fairly: without it every run draws a different `x_T`, so a
50-vs-400 difference is confounded by the noise draw. With a seed, DDIM at
`ddim_eta: 0.0` is fully reproducible. (For `ddpm` or `ddim_eta > 0` the seed
fixes `x_T` only — the per-step noise stays stochastic.)

```bash
# recommended first attempt at a low step count
python inference_full_region.py --sampler ddim --ddim_steps 50 \
    --ddim_spacing logsnr --ddim_clip_x_start dynamic --seed 0
```

### inference `data` subdir keys
The inference configs reuse the training `data` keys and add a few for the
condition folders sampled at inference and (for `dense` runs) the ground-truth
displacement.

There are two similarly-named pairs; the `train_`-prefixed ones are inert at
inference. **The un-prefixed `sampling_*` keys are the ones that decide which
videos get inferred and how many.** They may contain `{video_type}`, filled in
from the resolved `video_type` at runtime so one config covers every split (same
style as training's `"aug_subdir": "aug_{aug}x"`).

- `sampling_contour_condition_subdir` — contour-mask folder to sample, e.g.
  `"{video_type}/test/{video_type}_mask"` (contour configs). **This selects the
  inferred videos.** Change it to sample a different split, e.g.
  `"{video_type}/test_OB/{video_type}_mask"`.
- `sampling_motion_condition_subdir` — motion-condition folder to sample, e.g.
  `"tlrn_{video_type}_mask_motion/test"` (motion configs). Selects the inferred
  videos for `inference_only_motion.py`.
- `train_sampling_contour_condition_subdir` / `train_sampling_motion_condition_subdir`
  — passed through to the `Trainer` constructor, which only reads them inside
  `train()` (for its periodic sample-during-training). **Unused at inference**;
  they are kept so an inference config still fully describes the run. Training
  always samples from a fixed `dense/test/...`, which is why these have no
  `{video_type}`.
- `gt_disp_dense_subdir` — ground-truth displacement folder used when
  `video_type == "dense"`. If you point the sampling key at a non-`test` split,
  move this to the matching split too (e.g. `dense/test_OB/displacement_dense`)
  — the ground truth is loaded per sample and would otherwise silently come from
  the wrong split.
