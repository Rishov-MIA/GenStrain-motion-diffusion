# GenStrain Video Diffusion

This repo trains and runs inference for 3D video diffusion models over displacement-field videos.

Current model paths:

- **Unconditional base model**: learns the displacement video distribution without any condition.
- **Motion ControlNet model**: freezes the base UNet and trains a ControlNet branch that injects a 2-channel motion condition through zero-initialized residual adapters.
- **Legacy motion-conditioned model**: `inference.py` still uses `video_diffusion_cross_attention_only_reg_motion.py`, where motion is mixed directly through cross-attention inside the main UNet.

## Environment Setup

```bash
conda create --name video-diffusion-env python=3.10 -y
conda activate video-diffusion-env

mamba install pytorch==2.0.0 torchvision==0.15.0 torchaudio==2.0.0 pytorch-cuda=11.8 -c pytorch -c nvidia -y
pip install -r requirements.txt
```

Optional MKL downgrade, only if you hit the PyTorch MKL issue:

```bash
mamba install mkl=2024.0.0 mkl-devel=2024.0.0 mkl-include=2024.0.0 -c conda-forge
```

## Data Layout

Training and inference expect `.npy` videos.

Expected layout, adapt as needed:

```text
<base_path>/
  dense/
    train/
      displacement_dense/    # target displacement videos (.npy)
    test/
      displacement_dense/    # motion conditions and optional GT (.npy)
  paired_dense/
    test/
      displacement_dense/    # optional paired GT / conditions
```

Notes:

- Displacement videos should be shaped `[1, 2, F, H, W]`.
- The default scripts use `F=20`, `H=48`, `W=48`.
- Displacement and motion-condition values are normalized by dividing by `5.0`.
- If your motion condition lives in a different folder from the target displacement videos, update `motion_condition_video_dir`, `sampling_motion_condition_video_dir`, or pass `--motion_condition_dir` during inference.
- When loading paired target/condition data, files are matched by sorted filename order, so keep filenames aligned.

## Stage 1: Train Base Model

Edit `base_path` and `exp_name` in `train_base_unconditioned.py`:

```python
base_path = "/path/to/data"
exp_name = "base-unconditioned-experiment"
```

Run:

```bash
python train_base_unconditioned.py
```

Base checkpoints are saved to:

```text
./base-unconditioned-experiment/checkpoints/
```

To resume, uncomment:

```python
# trainer.load(milestone=-1)
```

## Stage 2: Train Motion ControlNet

First train or provide an unconditional base checkpoint. Then edit `train_controlnet.py`:

```python
base_checkpoint_path = "./base-unconditioned-experiment/checkpoints/model-<milestone>.pt"
exp_name = "controlnet-motion-experiment"
```

If `base_checkpoint_path` is left as `None`, ControlNet training starts with a randomly initialized base UNet, which is not the intended ControlNet workflow.

Run:

```bash
python train_controlnet.py
```

ControlNet checkpoints are saved separately:

```text
./controlnet-motion-experiment/checkpoints/
```

These do not overwrite the base checkpoints unless both scripts use the same `exp_name`.

## ControlNet Inference

Run with the latest ControlNet checkpoint:

```bash
python inference_controlnet.py --video_type dense --exp_name controlnet-motion-experiment --milestone -1
```

Useful arguments:

- `--motion_condition_dir`: folder of motion-condition `.npy` files.
- `--gt_dir`: optional folder of ground-truth displacement `.npy` files.
- `--start`, `--end`: index range over sorted condition files.
- `--cond_scale`: ControlNet strength. `1.0` is default; lower softens control, higher strengthens it.
- `--save_folder`: optional output folder.

Example with explicit folders:

```bash
python inference_controlnet.py \
  --motion_condition_dir /path/to/conditions \
  --gt_dir /path/to/ground_truth \
  --exp_name controlnet-motion-experiment \
  --milestone -1
```

Outputs are saved by default to:

```text
./controlnet-motion-experiment/controlnet_sampled_<video_type>_part_videos_infos/
```

## Legacy Motion-Conditioned Inference

`inference.py` uses the older directly motion-conditioned model:

```bash
python inference.py --video_type dense
```

For new ControlNet experiments, use `inference_controlnet.py`.
