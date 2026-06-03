from video_diffusion_pytorch.video_diffusion_base_unconditioned import (
    GaussianDiffusion,
    Trainer,
    Unet3D,
)


# DATA BASE PATH
base_path = "/scratch/vst2hb/video-diffusion-pytorch/new-data-sona-latest/all_data_resampled_20_frames-rv"

model = Unet3D(
    dim=48,
    cond_dim=None,
    channels=2,
    dim_mults=(1, 2, 4, 8),
)

diffusion = GaussianDiffusion(
    model,
    image_size=48,
    channels=2,
    num_frames=20,
    timesteps=1000,
    loss_type="l2",
).cuda()

exp_name = "base-unconditioned-experiment"

trainer = Trainer(
    diffusion_model=diffusion,
    input_video_folder=f"{base_path}/dense/train/displacement_dense",
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
