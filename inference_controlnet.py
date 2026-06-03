import argparse
import os
from pathlib import Path

import numpy as np
import torch

from video_diffusion_pytorch.video_diffusion_controlnet_motion import (
    BaseUnet3D,
    ControlNet3D,
    ControlledUnet3D,
    GaussianDiffusion,
    Trainer,
)


# DATA BASE PATH
base_path = "/scratch/vst2hb/video-diffusion-pytorch/new-data-sona-latest/all_data_resampled_20_frames-rv"


def normalize_divide_by_5(x: torch.Tensor) -> torch.Tensor:
    return x / 5.0


def pick_condition_videos_one_video_at_once(motion_condition_dir, start, end):
    motion_condition_dir = Path(motion_condition_dir)
    motion_condition_paths = sorted(list(motion_condition_dir.glob("*.npy")))

    for motion_condition_path in motion_condition_paths[start:end]:
        motion_condition = np.load(motion_condition_path)
        motion_condition_tensor = torch.from_numpy(motion_condition).float()

        if motion_condition_tensor.dim() == 5 and motion_condition_tensor.shape[0] == 1:
            motion_condition_tensor = motion_condition_tensor.squeeze(0)

        cond_filename = f"{motion_condition_path.stem}.npy"
        print(cond_filename)

        yield motion_condition_tensor.unsqueeze(0), [cond_filename]


parser = argparse.ArgumentParser(description="ControlNet motion video diffusion inference")
parser.add_argument("--video_type", type=str, default="dense", help="dataset split/type under base_path")
parser.add_argument("--start", type=int, default=0, help="start index")
parser.add_argument("--end", type=int, default=132, help="end index")
parser.add_argument("--exp_name", type=str, default="controlnet-motion-experiment", help="checkpoint experiment folder")
parser.add_argument("--milestone", type=int, default=-1, help="checkpoint milestone, -1 loads latest")
parser.add_argument("--cond_scale", type=float, default=1.0, help="ControlNet conditioning strength")
parser.add_argument("--motion_condition_dir", type=str, default=None, help="folder of motion condition .npy files")
parser.add_argument("--gt_dir", type=str, default=None, help="optional folder of ground-truth displacement .npy files")
parser.add_argument("--save_folder", type=str, default=None, help="optional output folder")
args = parser.parse_args()


motion_condition_dir = args.motion_condition_dir
if motion_condition_dir is None:
    motion_condition_dir = f"{base_path}/{args.video_type}/test/displacement_dense"

gt_dir = args.gt_dir
if gt_dir is None and args.video_type in ["dense", "paired_dense", "paired_cine"]:
    gt_dir = f"{base_path}/{args.video_type}/test/displacement_dense"

base_unet = BaseUnet3D(
    dim=48,
    cond_dim=None,
    channels=2,
    dim_mults=(1, 2, 4, 8),
)

controlnet = ControlNet3D(
    dim=48,
    cond_dim=None,
    channels=2,
    control_channels=2,
    dim_mults=(1, 2, 4, 8),
)

model = ControlledUnet3D(
    base_unet=base_unet,
    controlnet=controlnet,
    freeze_base=True,
)

diffusion = GaussianDiffusion(
    model,
    image_size=48,
    channels=2,
    num_frames=20,
    timesteps=1000,
    loss_type="l2",
).cuda()

trainer = Trainer(
    diffusion_model=diffusion,
    input_video_folder=f"{base_path}/dense/train/displacement_dense",
    motion_condition_video_dir=f"{base_path}/dense/train/displacement_dense",
    sampling_motion_condition_video_dir=motion_condition_dir,
    train_batch_size=20,
    train_lr=1e-5,
    save_and_sample_every=500,
    train_num_steps=700000,
    gradient_accumulate_every=1,
    ema_decay=0.995,
    amp=True,
    experiment_name=args.exp_name,
    inference_only=True,
)

trainer.load(milestone=args.milestone)

device = next(trainer.ema_model.parameters()).device
save_folder = args.save_folder
if save_folder is None:
    save_folder = f"./{args.exp_name}/controlnet_sampled_{args.video_type}_part_videos_infos"
os.makedirs(save_folder, exist_ok=True)

video_generator = pick_condition_videos_one_video_at_once(motion_condition_dir, args.start, args.end)

for motion_cond_video, cond_filename in video_generator:
    motion_cond_video = normalize_divide_by_5(motion_cond_video).to(device)

    gt_disp = None
    if gt_dir is not None:
        gt_path = os.path.join(gt_dir, cond_filename[0])
        if os.path.exists(gt_path):
            gt_disp = np.load(gt_path)

    trainer.sample_and_save_one_video_at_a_time(
        "final",
        motion_cond_video=motion_cond_video,
        cond_filenames=cond_filename,
        reconstructed_disp=None,
        gt_disp=gt_disp,
        cond_scale=args.cond_scale,
        save_folder=save_folder,
    )
