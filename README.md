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

- The dataset uses `.npy` files only and matches them one-to-one by sorted filename order.
- `dense_mask` or `cine_mask` directory `.npy `files are expected to be shaped `[1, F, H, W]` where `F=20 `, `H=48 `, `W=48`
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
