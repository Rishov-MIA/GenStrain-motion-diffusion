import argparse
import torch
from video_diffusion_pytorch.video_diffusion_cross_attention_with_motion_only import Unet3D, GaussianDiffusion, Trainer
import random
from pathlib import Path
import numpy as np
import os


# DATA BASE PATH
base_path = "/scratch/vst2hb/video-diffusion-pytorch/new-data-sona-latest/all_data_resampled_20_frames-most-latest"

# Define argument parser
parser = argparse.ArgumentParser(description="Video Diffusion Pytorch Script (Motion Only)")
parser.add_argument('--video_type', type=str, default="cine", help='video type: cine and dense')
parser.add_argument('--start', type=int, default=0, help='start index')
parser.add_argument('--end', type=int, default=132, help='end index')

# Parse the arguments
args = parser.parse_args()

# Use the parameters
video_type = args.video_type
start = args.start
end = args.end


def normalize_divide_by_5(x: torch.Tensor) -> torch.Tensor:
    return x / 5.0


def pick_condition_videos_one_video_at_once(motion_cond_video_dir, start, end):
    motion_cond_video_dir = Path(motion_cond_video_dir)
    motion_cond_video_paths = sorted(list(motion_cond_video_dir.glob("*.npy")))
    indices = range(len(motion_cond_video_paths))

    for idx in indices[start:end]:
        motion_cond_video_path = motion_cond_video_paths[idx]
        cond_filename = f"{motion_cond_video_path.stem}.npy"
        print(cond_filename)

        motion_cond = np.load(motion_cond_video_path)
        motion_cond_tensor = torch.from_numpy(motion_cond).float()  # shape [2, F, H, W] or [1, 2, F, H, W]

        # Ensure shape is [1, 2, F, H, W] (batch=1, 2 channels, F frames, H, W)
        if motion_cond_tensor.dim() == 4:
            motion_cond_tensor = motion_cond_tensor.unsqueeze(0)

        yield motion_cond_tensor, [cond_filename]


model = Unet3D(
    dim = 48,
    cond_dim=None,
    channels=2,
    dim_mults = (1, 2, 4, 8),
)

diffusion = GaussianDiffusion(
    model,
    image_size = 48,
    channels = 2,
    num_frames = 20,
    timesteps = 1000,
    loss_type = 'l2',
).cuda()

exp_name = "experiment-name-motion-only"

trainer = Trainer(
    diffusion_model=diffusion,
    input_video_folder=f'{base_path}/dense/train/displacement_dense',
    motion_condition_video_dir=f'{base_path}/tlrn_dense_mask_motion/train',
    sampling_motion_condition_video_dir=f'{base_path}/tlrn_dense_mask_motion/test',
    train_batch_size = 20,
    train_lr = 1e-5,
    save_and_sample_every = 500,
    train_num_steps = 700000,
    gradient_accumulate_every = 1,
    ema_decay = 0.995,
    amp = True,
    experiment_name=exp_name
)

# Load the latest checkpoint (milestone = -1)
trainer.load(milestone=-1)

motion_cond_video_dir = f"{base_path}/tlrn_{video_type}_mask_motion/test"
video_generator = pick_condition_videos_one_video_at_once(motion_cond_video_dir, start, end)

# Move to GPU if necessary
device = next(trainer.ema_model.parameters()).device
custom_save_folder = f"./{exp_name}/sampling_time_sampled_{video_type}_part_videos_infos/"
os.makedirs(custom_save_folder, exist_ok=True)


for motion_cond_video, cond_filename in video_generator:
    motion_cond_video = normalize_divide_by_5(motion_cond_video).to(device)

    # Generate samples
    if video_type == "cine":
        reconstructed_disp = None
        trainer.sample_and_save_one_video_at_a_time("final", motion_cond_video, cond_filename, reconstructed_disp, None, save_folder=custom_save_folder)
    elif video_type == "dense":
        gt_disp_dir = f"{base_path}/dense/test/displacement_dense"
        gt_disp = np.load(os.path.join(gt_disp_dir, cond_filename[0]))
        reconstructed_disp = None
        trainer.sample_and_save_one_video_at_a_time("final", motion_cond_video, cond_filename, reconstructed_disp, gt_disp, save_folder=custom_save_folder)
    elif video_type in ["paired_cine", "paired_dense"]:
        gt_disp_dir = f"{base_path}/paired_dense/test/displacement_dense"
        gt_disp = np.load(os.path.join(gt_disp_dir, cond_filename[0]))
        reconstructed_disp = None
        trainer.sample_and_save_one_video_at_a_time("final", motion_cond_video, cond_filename, reconstructed_disp, gt_disp, save_folder=custom_save_folder)
