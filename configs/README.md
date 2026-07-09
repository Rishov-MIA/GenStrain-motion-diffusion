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
- `experiment_name` — names the results/checkpoint folder.

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
