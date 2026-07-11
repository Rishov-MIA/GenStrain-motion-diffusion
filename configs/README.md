# Training configs

Each JSON file here holds every hyperparameter for one training script, named
`configs/<script_name>.json`. The scripts load their default config
automatically; point them at another file to run a different experiment
without touching code:

```bash
python train_group_dro.py --config configs/my_experiment.json
torchrun --standalone --nproc_per_node=4 train_group_dro_ddp.py --config configs/my_experiment_ddp.json
```

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
  include `{eta_q}` / `{adjustment_c}` placeholders, which the group-DRO scripts
  fill with the actual `dro` hyperparameters at launch (see the `dro` section), so
  each sweep point writes to its own folder / wandb run.

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
- `allowed_disease_groups` — optional list of disease groups to keep, in
  string-match PRIORITY order (first match wins). A metadata disease string is
  matched by case-insensitive substring, so ordering resolves multi-label
  strings: e.g. `["LBBB", "Healthy"]` maps `"LBBB & Recovered DCM"` → `LBBB`,
  and a more specific `"Healthy Pediatric"` must precede `"Healthy"` or it would
  collapse into `Healthy`. Omit to use the code default
  (`ALLOWED_DISEASE_GROUPS` in `trainer_group_dro.py`).

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
