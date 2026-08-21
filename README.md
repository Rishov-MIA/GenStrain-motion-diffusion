# GenStrain video diffusion

This repo trains and runs inference for a 3D video diffusion model that generates displacement-field videos from conditioning videos. Two conditioning setups are available:

| Variant          | Conditioned on                                                  | Train                                        | Inference                                        |
| ---------------- | --------------------------------------------------------------- | -------------------------------------------- | ------------------------------------------------ |
| Contour only     | a contour/mask video                                            | `train.py`                                 | `inference.py`                                 |
| Contour + motion | a contour/mask video**and** a motion (displacement) video | `train_both_contour_motion_full_region.py` | `inference_both_contour_motion_full_region.py` |

Both are config-driven in the same way, and everything below applies to both unless a section says otherwise.

## Environment setup

```bash
conda create --name video-diffusion-env python=3.10 -y
conda activate video-diffusion-env

mamba install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

Optional MKL downgrade (only if you hit the PyTorch MKL issue):

```bash
mamba install mkl=2024.0.0 mkl-devel=2024.0.0 mkl-include=2024.0.0 -c conda-forge
```

## Data layout

Training and inference expect `.npy` videos. The trainer loads pairs of videos by matching filenames in two folders.

Expected layout (adapt as needed):

```
<base_path>/
  dense/
    train/
      displacement_dense/    # target videos (.npy)
      dense_mask/            # contour/mask videos (.npy)
    test/
      dense_mask/            # contour/mask videos for sampling (.npy)
      displacement_dense/    # optional GT for evaluation (.npy)
  cine/
    test/
      cine_mask/             # contour/mask videos for sampling (.npy)
  tlrn_dense_mask_motion/    # contour + motion variant only
    train/                   # motion-condition displacement fields (.npy)
    test/                    # motion-condition fields for sampling (.npy)
  tlrn_cine_mask_motion/     # contour + motion variant only
    test/                    # motion-condition fields for sampling (.npy)
```

Notes:

- The dataset uses `.npy` files only and matches them one-to-one by sorted filename order. **All files of** `b` **,** `cine_mask` **and** `displacement_dense` **should have the same filenames.**
- `dense_mask` or `cine_mask` directory `.npy `files are expected to be shaped `[1, F, H, W]`, where `F` and `H`/`W` must equal the `num_frames` and `image_size` set in your config. The shipped configs use `F=26`, `H=64`, `W=64`.
- **Mask videos are binary with values 0 and 255.0**; traininng and inference normalizes them to 0–1 by dividing by 255.0.
- `displacement_dense` directory  `.npy` files should contain displacement fields shaped `[1, 2, F, H, W]`.
- Motion-condition `.npy` files (contour + motion variant only) are displacement fields shaped `[1, 2, F, H, W]` too. They pair with the contour masks by **sorted filename order**, so both folders must hold the same number of files under matching names — the count is asserted at startup, but a name mismatch would silently mis-pair.
- Inference picks the motion folder from `--video_type`, since `sampling_motion_condition_subdir` is the template `tlrn_{video_type}_mask_motion/test`: `dense` reads `tlrn_dense_mask_motion/test` and `cine` reads `tlrn_cine_mask_motion/test`, matching how `{video_type}/test/{video_type}_mask` picks the contour folder. Training is not templated — the shipped configs read the `dense` folders.

## Training

Hyperparameters are **not** edited in `train.py` — they live in a JSON config, `configs/train.json` by default.

1) Point the config at your data and name the experiment:

```json
{
  "experiment_name": "my-experiment",
  "data": {
    "base_path": "/path/to/data",
    "input_video_subdir": "dense/train/displacement_dense",
    "contour_condition_subdir": "dense/train/dense_mask",
    "sampling_contour_condition_subdir": "dense/test/dense_mask"
  }
}
```

2) Run:

```bash
python train.py                                  # uses configs/train.json
python train.py --config configs/my_run.json     # any other config
```

Checkpoints are saved to `./<experiment_name>/checkpoints/`. Every run also writes a timestamped copy of the config it ran with, so a finished experiment records its own settings.

### Config blocks

| Block              | Purpose                                                                                            |
| ------------------ | -------------------------------------------------------------------------------------------------- |
| `data`           | `base_path` plus the four subdirectories under it                                                |
| `model`          | `Unet3D`: `dim`, `cond_dim`, `channels`, `dim_mults`                                     |
| `diffusion`      | `image_size`, `num_frames`, `channels`, `timesteps`, `loss_type`, `contour_noise_only` |
| `training`       | batch size, lr, step count, EMA, AMP, checkpoint interval,`resume`                               |
| `logging`        | optional Weights & Biases settings                                                                 |
| `frame_validity` | optional valid-frame loss masking                                                                  |

`image_size` and `num_frames` must match the actual shape of your `.npy` files — nothing resizes or resamples, and a mismatch asserts on the first step.

`contour_noise_only: true` applies the diffusion noise only inside the contour region rather than over the whole frame.

### Contour + motion conditioning

`train_both_contour_motion_full_region.py` trains the variant that conditions on a motion (displacement) video alongside the contour mask. It is driven by a config exactly like `train.py`:

```bash
python train_both_contour_motion_full_region.py                                                  # configs/train_both_contour_motion_full_region.json
python train_both_contour_motion_full_region.py --config configs/train_both_contour_motion.json  # contour-region noise
```

The two shipped configs run the same script and differ only in where the noise is applied:

| Config                                                           | `contour_noise_only`                           |
| ---------------------------------------------------------------- | ------------------------------------------------ |
| `configs/train_both_contour_motion_full_region.json` (default) | `false` — noise over the whole frame          |
| `configs/train_both_contour_motion.json`                       | `true` — noise inside the contour region only |

Their `data` block adds `motion_condition_subdir` and `sampling_motion_condition_subdir` alongside the contour folders, and the `diffusion` block adds three keys:

| Key                 | Meaning                                                                                                                        |
| ------------------- | ------------------------------------------------------------------------------------------------------------------------------ |
| `disp_scale`      | divisor applied to displacement fields before diffusion, and the multiplier applied to every output afterwards. Default`5.0` |
| `disp_metrics`    | log the ROI displacement diagnostics. Default`true`; diagnostics only, so the trained weights are identical either way       |
| `disp_rel_metric` | which motion-normalized ratio to log:`"mse"` (default), `"epe"`, or `null` for none                                      |

`disp_scale` describes the **units the weights were trained in**, not a hyperparameter you can retune on a resume. It is stamped into every checkpoint, and loading one under a different value fails with both numbers named rather than silently rescaling every output. A checkpoint saved before the field existed is assumed to be `5.0`. Keep the training config and the matching inference config in agreement.

The displacement ratios are dimensionless: 0 is perfect and **1.0 is exactly as bad as predicting zero motion**. They are single-step estimates at a randomly drawn `t`, so read the trend rather than any one step's value. The ROI is taken from the contour condition alone — the motion condition never enters it — so these numbers compare directly against a contour-only run on the same data.

### Resume training

Set `"resume": true` in the `training` block. The latest checkpoint loads automatically on startup, and training starts from scratch if none exists — no code edit needed.

### Preview sampling during training

Every `save_and_sample_every` steps the trainer samples a few preview videos alongside the checkpoint, so you can watch quality progress without stopping the run. Previews use DDPM sampling and never affect the training loss.

### Valid-frame loss masking (optional)

Clips resampled to a fixed `num_frames` may only have their first N frames real, the rest padding. Point `frame_validity` at a CSV listing the real count per video and the loss will ignore the padding:

```json
"frame_validity": {
  "csv_path": "/path/to/frame_counts_train.csv",
  "filename_col": "dense_filename",
  "valid_frames_col": "dense_valid_frames",
  "missing": "error"
}
```

`valid_frames = N` means frames `[0:N]` are real and `[N:num_frames]` are padding. `missing` controls files absent from the CSV: `"full"` treats them as fully valid, `"error"` raises so you can guarantee full coverage. Omit the block entirely and the loss is byte-for-byte the original all-frames loss.

### Weights & Biases (optional)

```json
"logging": { "use_wandb": true, "wandb_project": "genstrain-motion-diffusion" }
```

Off by default, and the `wandb` import is guarded — training runs fine without the package installed.

## Inference

Settings live in `configs/inference.json` by default. Requires a CUDA GPU.

```bash
python inference.py                                    # uses configs/inference.json
python inference.py --config configs/my_inference.json
python inference.py --video_type dense --start 0 --end 50
```

The config's `inference` block supplies defaults; any flag passed on the CLI overrides it for that run.

| Flag / config key      | Meaning                                                                               |
| ---------------------- | ------------------------------------------------------------------------------------- |
| `--video_type`       | `cine` or `dense`. `dense` also loads ground-truth displacements for comparison |
| `--start`, `--end` | index range into the sorted contour/mask folder                                       |
| `--milestone`        | checkpoint to load;`-1` is the latest                                               |
| `--seed`             | seed the initial noise so runs are comparable                                         |

Sampling uses DDPM, walking all `timesteps` (1000) denoising steps.

The `model` and `diffusion` blocks must match the checkpoint you are loading — in particular `image_size` and `num_frames`, or the weights will not load.

Outputs are saved to `./<experiment_name>/sampling_time_sampled_<video_type>_part_videos_infos/`, created automatically.

### Contour + motion inference

`inference_both_contour_motion_full_region.py` samples the contour + motion model, defaulting to `configs/inference_both_contour_motion_full_region.json`:

```bash
python inference_both_contour_motion_full_region.py                                                                  # uses the default config
python inference_both_contour_motion_full_region.py --config configs/inference_both_contour_motion_full_region.json  # same thing, spelled out
python inference_both_contour_motion_full_region.py --video_type dense --start 0 --end 50
```

Same flags as above, plus one:

| Flag / config key   | Meaning                                                                                   |
| ------------------- | ----------------------------------------------------------------------------------------- |
| `--skip_existing` | skip any video whose prediction is already in the experiment's`inference_disps/` folder |

Its `data` block names both sampling folders, and `sampling_contour_condition_subdir` / `sampling_motion_condition_subdir` are templates containing `{video_type}`, filled in from `--video_type` at launch. Under `--video_type dense` the ground truth for comparison comes from `gt_disp_dense_subdir`.

`--skip_existing` resumes a run killed by a job timeout or a crash — re-run the same command with the flag added and only the missing videos are sampled. A prediction truncated mid-write does not count as done and is sampled again. It matches on filename only, not on how the sample was produced, so don't use it to fill in a split whose finished files came from a different checkpoint or seed — point at a fresh `experiment_name` or delete the stale files first. `"skip_existing": true` in a config's `inference` block turns it on by default; the flag can only turn it on, never off.
