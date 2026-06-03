from video_diffusion_pytorch.video_diffusion_controlnet_motion import (
    BaseUnet3D,
    ControlNet3D,
    ControlledUnet3D,
    GaussianDiffusion,
    Trainer,
    load_base_unet_checkpoint,
)


# DATA BASE PATH
base_path = "/scratch/vst2hb/video-diffusion-pytorch/new-data-sona-latest/all_data_resampled_20_frames-rv"

# Set this to a trained unconditional base diffusion checkpoint.
# Expected format can be either:
# - Trainer checkpoint with "model" / "ema" containing denoise_fn.* keys
# - raw BaseUnet3D state_dict
base_checkpoint_path = None

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

if base_checkpoint_path is not None:
    load_base_unet_checkpoint(model, base_checkpoint_path, use_ema=True, strict=True)

diffusion = GaussianDiffusion(
    model,
    image_size=48,
    channels=2,
    num_frames=20,
    timesteps=1000,
    loss_type="l2",
).cuda()

exp_name = "controlnet-motion-experiment"

trainer = Trainer(
    diffusion_model=diffusion,
    input_video_folder=f"{base_path}/dense/train/displacement_dense",
    motion_condition_video_dir=f"{base_path}/dense/train/displacement_dense",
    sampling_motion_condition_video_dir=f"{base_path}/dense/test/displacement_dense",
    train_batch_size=20,
    train_lr=1e-5,
    save_and_sample_every=5,
    train_num_steps=700000,
    gradient_accumulate_every=1,
    ema_decay=0.995,
    amp=True,
    experiment_name=exp_name,
)

# trainer.load(milestone=-1)
trainer.train()
