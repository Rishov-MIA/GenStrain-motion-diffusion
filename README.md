# GenStrain video diffusion

This repo trains and runs inference for a 3D video diffusion model conditioned on contour/mask videos.

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
```

Notes:

- The dataset uses `.npy` files only and matches them one-to-one by sorted filename order. **All files of** `b` **,** `cine_mask` **and** `displacement_dense` **should have the same filenames.**
- `dense_mask` or `cine_mask` directory `.npy `files are expected to be shaped `[1, F, H, W]`, where `F` and `H`/`W` must equal the `num_frames` and `image_size` set in your config. The shipped configs use `F=26`, `H=64`, `W=64`.
- **Mask videos are binary with values 0 and 255.0**; traininng and inference normalizes them to 0–1 by dividing by 255.0.
- `displacement_dense` directory  `.npy` files should contain displacement fields shaped `[1, 2, F, H, W]`.

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

| Block | Purpose |
| --- | --- |
| `data` | `base_path` plus the four subdirectories under it |
| `model` | `Unet3D`: `dim`, `cond_dim`, `channels`, `dim_mults` |
| `diffusion` | `image_size`, `num_frames`, `channels`, `timesteps`, `loss_type`, `contour_noise_only` |
| `training` | batch size, lr, step count, EMA, AMP, checkpoint interval, `resume` |
| `logging` | optional Weights & Biases settings |
| `frame_validity` | optional valid-frame loss masking |

`image_size` and `num_frames` must match the actual shape of your `.npy` files — nothing resizes or resamples, and a mismatch asserts on the first step.

`contour_noise_only: true` applies the diffusion noise only inside the contour region rather than over the whole frame.

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

| Flag / config key | Meaning |
| --- | --- |
| `--video_type` | `cine` or `dense`. `dense` also loads ground-truth displacements for comparison |
| `--start`, `--end` | index range into the sorted contour/mask folder |
| `--milestone` | checkpoint to load; `-1` is the latest |
| `--seed` | seed the initial noise so runs are comparable |

Sampling uses DDPM, walking all `timesteps` (1000) denoising steps.

The `model` and `diffusion` blocks must match the checkpoint you are loading — in particular `image_size` and `num_frames`, or the weights will not load.

Outputs are saved to `./<experiment_name>/sampling_time_sampled_<video_type>_part_videos_infos/`, created automatically.
