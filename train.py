import torch
from video_diffusion_pytorch.video_diffusion_cross_attention_with_motion_after_only_contour import Unet3D, GaussianDiffusion, Trainer


# DATA BASE PATH
base_path = "/scratch/vst2hb/video-diffusion-pytorch/new-data-sona-latest/all_data_resampled_20_frames-rv"

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
    save_and_sample_every = 5,
    train_num_steps = 700000,         # total training steps
    gradient_accumulate_every = 1,    # gradient accumulation steps
    ema_decay = 0.995,                # exponential moving average decay
    amp = True,                        # turn on mixed precision
    experiment_name=exp_name
)

# Load the latest checkpoint (milestone = -1) if you want to retrain from latest checkpoint
# trainer.load(milestone=-1)

trainer.train()