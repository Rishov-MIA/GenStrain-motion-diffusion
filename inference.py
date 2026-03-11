import argparse
import torch
from video_diffusion_pytorch.video_diffusion_cross_attention_with_motion_after_only_contour import Unet3D, GaussianDiffusion, Trainer
import random
from pathlib import Path
import numpy as np
import os


# DATA BASE PATH
base_path = "/scratch/vst2hb/video-diffusion-pytorch/new-data-sona-latest/all_data_resampled_20_frames-most-latest"

# Define argument parser
parser = argparse.ArgumentParser(description="Video Diffusion Pytorch Script")
parser.add_argument('--video_type', type=str, default="cine", help='video type: cine and dense')
parser.add_argument('--motion_place', type=str, default="after", help='before, after')
parser.add_argument('--start', type=int, default=0, help='start index')
parser.add_argument('--end', type=int, default=132, help='end index')

# Parse the arguments
args = parser.parse_args()

# Use the parameters
video_type = args.video_type
motion_place = args.motion_place
start = args.start
end = args.end


def normalize_cond_img(t):
    return t / 255.0

def normalize_divide_by_5(x: torch.Tensor) -> torch.Tensor:
    return x / 5.0


def pick_condition_videos_one_video_at_once(contour_cond_video_dir, start, end):
    # Convert string to a Path object
    contour_cond_video_dir = Path(contour_cond_video_dir)
    contour_cond_video_paths = sorted(list(contour_cond_video_dir.glob("*.npy")))

    # Select random indices
    num_samples = len(contour_cond_video_paths)
    indices = range(len(contour_cond_video_paths))

    for idx in indices[start: end]:
        contour_cond_video_path = contour_cond_video_paths[idx]
        contour_cond = np.load(contour_cond_video_path)
        contour_cond_tensor = torch.from_numpy(contour_cond).float()  # shape [1, F, H, W]
        cond_filename = f"{contour_cond_video_path.stem}.npy"
        print(cond_filename)

        yield contour_cond_tensor.unsqueeze(0), [cond_filename]


model = Unet3D(
    dim = 48,
    cond_dim=None,           # video encoder output dim
    channels=2,
    dim_mults = (1, 2, 4, 8),
)

diffusion = GaussianDiffusion(
    model,
    image_size = 48,
    channels = 2,
    num_frames = 20,
    timesteps = 1000,   # number of steps
    loss_type = 'l2',   # L1 or L2
    contour_noise_only = False,  # True = noise only in mask contour region, False = noise on full image
).cuda()

exp_name = "experiment-name"

trainer = Trainer(
    diffusion_model=diffusion,
    input_video_folder=f'{base_path}/dense/train/displacement_dense',
    contour_condition_video_dir=f'{base_path}/dense/train/dense_mask',
    sampling_contour_condition_video_dir=f'{base_path}/dense/test/dense_mask',
    train_batch_size = 20,
    train_lr = 1e-5,
    save_and_sample_every = 500,
    train_num_steps = 700000,          # total training steps
    gradient_accumulate_every = 1,     # gradient accumulation steps
    ema_decay = 0.995,                 # exponential moving average decay
    amp = True,                        # turn on mixed precision
    experiment_name=exp_name
)

# Load the latest checkpoint (milestone = -1)
trainer.load(milestone=-1)

contour_cond_video_dir = f"{base_path}/{video_type}/test/{video_type}_mask"
video_generator = pick_condition_videos_one_video_at_once(contour_cond_video_dir, start, end)


# Move to GPU if necessary
device = next(trainer.ema_model.parameters()).device
custom_save_folder = f"./{exp_name}/sampling_time_sampled_{video_type}_part_videos_infos/"
os.makedirs(custom_save_folder, exist_ok=True)


for contour_cond_video, cond_filename in video_generator:
    contour_cond_video = normalize_cond_img(contour_cond_video)
    contour_cond_video = contour_cond_video.to(device)

    # Generate samples
    if video_type == "cine":
        reconstructed_disp = None
        trainer.sample_and_save_one_video_at_a_time("final", contour_cond_video, cond_filename, reconstructed_disp, None, save_folder=custom_save_folder)
    elif video_type == "dense":
        gt_disp_dir = f"{base_path}/dense/test/displacement_dense"
        gt_disp = np.load(os.path.join(gt_disp_dir, cond_filename[0]))
        reconstructed_disp = None
        trainer.sample_and_save_one_video_at_a_time("final", contour_cond_video, cond_filename, reconstructed_disp, gt_disp, save_folder=custom_save_folder)
    elif video_type in ["paired_cine", "paired_dense"]:
        gt_disp_dir = f"{base_path}/paired_dense/test/displacement_dense"
        gt_disp = np.load(os.path.join(gt_disp_dir, cond_filename[0]))
        reconstructed_disp = None
        trainer.sample_and_save_one_video_at_a_time("final", contour_cond_video, cond_filename, reconstructed_disp, gt_disp, save_folder=custom_save_folder)
