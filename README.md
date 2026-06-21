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
- `dense_mask` or `cine_mask` directory `.npy `files are expected to be shaped `[1, F, H, W]` where `F=20`, `H=48`, `W=48`
- **Mask videos are binary with values 0 and 255.0**; traininng and inference normalizes them to 0–1 by dividing by 255.0.
- `displacement_dense` directory  `.npy` files should contain displacement fields shaped `[1, 2, F, H, W]`.

## Training

1) Edit the dataset base path and experiment name in `train.py`:

```python
base_path = "/path/to/data"
exp_name = "my-experiment"
```

2) (Optional) Adjust model/training hyperparameters in `train.py`.
3) Run training:

```bash
python train.py
```

Checkpoints are saved to `./<exp_name>/checkpoints/`.

### Resume training

Uncomment this line in `train.py` to resume from the latest checkpoint:

```python
trainer.load(milestone=-1)
```

### Multi-GPU training (DDP)

For faster training on a single machine with multiple GPUs, use `train_full_region_ddp.py`,
which trains the full-region model with PyTorch `DistributedDataParallel` (DDP). The single-GPU
scripts are untouched; this is a separate entry point.

1) Edit the dataset base path, experiment name, and data dimensions at the top of
   `train_full_region_ddp.py`:

```python
base_path = "/path/to/data"
IMAGE_SIZE = 64   # H == W of your .npy videos
NUM_FRAMES = 32   # number of frames in your .npy videos
PER_GPU_BATCH = 5 # see note below
```

2) Launch with `torchrun`, setting `--nproc_per_node` to your GPU count (single node):

```bash
# 4 GPUs
torchrun --standalone --nproc_per_node=4 train_full_region_ddp.py

# 2 GPUs
torchrun --standalone --nproc_per_node=2 train_full_region_ddp.py
```

Key points:

- **`PER_GPU_BATCH` is per-GPU.** The effective (global) batch is `PER_GPU_BATCH * num_gpus`.
  To reproduce a single-GPU global batch of 20 on 4 GPUs, set `PER_GPU_BATCH = 5`. If you grow
  the global batch, scale `train_lr` accordingly (linear scaling is a reasonable starting point).
- Larger data costs more memory: 64×64 with 32 frames is roughly 4–5× the per-sample memory of
  48×48 with 20 frames. Lower `PER_GPU_BATCH` (start small, e.g. 2) until it fits, and use
  `gradient_accumulate_every` to recover effective batch size if needed.
- `IMAGE_SIZE` / `NUM_FRAMES` must match the actual shape of your `.npy` files — the dataset does
  not resize or resample, and a shape mismatch will assert at the first step.
- **Resume is automatic.** On startup the script loads the latest checkpoint (`milestone=-1`) if
  one exists, and otherwise starts from scratch. Checkpoints are written by rank 0 to
  `./<exp_name>/checkpoints/` and are compatible with the single-GPU trainer and inference scripts.
- Checkpointing, sampling, and logging run on rank 0 only, so you get a single set of outputs.
- This path is single-node. Multi-node would only require a different `torchrun` rendezvous launch;
  the training code already supports it.

## Inference

1) Edit `base_path` and `exp_name` in `inference.py` to match your data and trained experiment.
2) Make sure checkpoints exist in `./<exp_name>/checkpoints/`.
3) Run inference:

```bash
python inference.py --video_type cine
```

Arguments:

- `--video_type`: `cine`, `dense`, `paired_cine`, `paired_dense`
- `--start`, `--end`: index range for videos in the contour/mask folder (optional)

Outputs are saved to `./<exp_name>/sampling_time_sampled_<video_type>_part_videos_infos/` and the folder is created automatically. Update `custom_save_folder` in `inference.py` if you want a different location.
