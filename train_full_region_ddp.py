"""
Multi-GPU (single-node) DDP version of train_full_region.py.

Launch with torchrun. Example for 4 GPUs on one node:

    torchrun --standalone --nproc_per_node=4 train_full_region_ddp.py

For 2 GPUs:

    torchrun --standalone --nproc_per_node=2 train_full_region_ddp.py

NOTE: `train_batch_size` below is PER-GPU. With N GPUs the effective global
batch size is `train_batch_size * N`. The single-GPU script used a global batch
of 20. To keep the same global batch on 4 GPUs, set train_batch_size = 5; on
2 GPUs set train_batch_size = 10. Adjust `train_lr` if you intentionally grow
the global batch (linear LR scaling is a reasonable starting point).
"""

from video_diffusion_pytorch.video_diffusion_cross_attention_with_motion_after_only_contour import (
    Unet3D,
    GaussianDiffusion,
)
from video_diffusion_pytorch.trainer_ddp import (
    DDPTrainer,
    setup_distributed,
    cleanup_distributed,
    is_main_process,
)


# DATA BASE PATH
base_path = "/scratch/vst2hb/Dataset_Processing/Data Bank/proposal-genstrain-new/all_data_resampled_32_frames"

# New data dimensions.
IMAGE_SIZE = 64   # H == W
NUM_FRAMES = 32

# Per-GPU batch size. Effective global batch = PER_GPU_BATCH * num_gpus.
PER_GPU_BATCH = 5


def main():
    local_rank, global_rank, world_size, device = setup_distributed()

    model = Unet3D(
        dim=48,
        cond_dim=None,  # video encoder output dim
        channels=2,
        dim_mults=(1, 2, 4, 8),
    )

    diffusion = GaussianDiffusion(
        model,
        image_size=IMAGE_SIZE,  # 64x64
        channels=2,
        num_frames=NUM_FRAMES,  # 32 frames
        timesteps=1000,  # number of steps
        loss_type="l2",  # L1 or L2
        contour_noise_only=False,  # True = noise only in mask contour region, False = full image
    ).to(device)

    exp_name = "new-data-noise-full-region-cond-mask-only"

    trainer = DDPTrainer(
        diffusion_model=diffusion,
        input_video_folder=f"{base_path}/dense/train/displacement_dense",
        local_rank=local_rank,
        contour_condition_video_dir=f"{base_path}/dense/train/dense_mask",
        sampling_contour_condition_video_dir=f"{base_path}/dense/test/dense_mask",
        train_batch_size=PER_GPU_BATCH,  # PER-GPU
        train_lr=1e-5,
        save_and_sample_every=300,
        train_num_steps=700000,
        gradient_accumulate_every=1,
        ema_decay=0.995,
        amp=True,
        experiment_name=exp_name,
        num_workers=4,
    )

    # Load the latest checkpoint (milestone = -1) to resume from latest.
    # Wrap in try/except so a fresh run (no checkpoints yet) doesn't crash.
    try:
        trainer.load(milestone=-1)
    except AssertionError:
        if is_main_process():
            print("no checkpoint found, starting from scratch")

    try:
        trainer.train()
    finally:
        cleanup_distributed()


if __name__ == "__main__":
    main()
