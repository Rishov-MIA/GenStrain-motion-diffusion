"""
Multi-GPU (single-node) DDP version of train_group_dro.py.

Group-DRO training for the contour-conditioned displacement diffusion model.
Each step samples ONE disease group (uniformly, on rank 0 and broadcast to all
ranks), draws a per-rank batch from it, updates that group's weight q_g on the
GLOBAL (all-reduced) group loss, then takes a q_g-weighted model step with weight
decay. See video_diffusion_pytorch/trainer_group_dro_ddp.py for the DDP details.

Groups are the `disease` field from the metadata JSON, restricted to:
    Acute MI, Healthy, Healthy Pediatric, DCM, Myocarditis, LBBB
(substring match in priority order, so "LBBB & Recovered DCM" -> LBBB). Augmented
samples are ignored: the real per-group counts n_g are just the files present.

Launch with torchrun. Example for 4 GPUs on one node:

    torchrun --standalone --nproc_per_node=4 train_group_dro_ddp.py

For 2 GPUs:

    torchrun --standalone --nproc_per_node=2 train_group_dro_ddp.py

NOTE: `PER_GPU_BATCH` is PER-GPU. With N GPUs the effective per-step group batch
is `PER_GPU_BATCH * N`, all from the one sampled group. Scale `train_lr` if you
grow the global batch (linear scaling is a reasonable start). The Group-DRO
hyperparameters (dro_eta_q, dro_adjustment_c) act on the global mean loss and do
NOT need rescaling with the number of GPUs.
"""

from video_diffusion_pytorch.video_diffusion_cross_attention_with_motion_after_only_contour import (
    Unet3D,
    GaussianDiffusion,
)
from video_diffusion_pytorch.trainer_group_dro_ddp import (
    GroupDRODDPTrainer,
    setup_distributed,
    cleanup_distributed,
    is_main_process,
)


# DATA BASE PATH
base_path = "/scratch/vst2hb/Dataset_Processing/Data Bank/proposal-genstrain-new/all_data_resampled_26_frames"
metadata_json_path = "/scratch/vst2hb/Dataset_Processing/Data Bank/proposal-genstrain-new/processed_excel.json"

# Data dimensions.
IMAGE_SIZE = 64   # H == W
NUM_FRAMES = 26

# Per-GPU batch size. Effective per-step group batch = PER_GPU_BATCH * num_gpus.
PER_GPU_BATCH = 10

# Group-DRO hyperparameters.
# q update multiplies a group's raw weight by exp(DRO_ETA_Q * S_g), S_g ~= L_g, the
# MEAN denoising loss (~0.4-1.1 here). Pick DRO_ETA_Q so exp(DRO_ETA_Q * L_g) stays
# near 1 -> ~0.01-0.05. DRO_ETA_Q=1.0 makes q collapse onto whichever group is
# sampled first. Watch the printed per-group q: it should stay spread and drift to
# the persistently-hardest group, never pin near 1.0. These act on the GLOBAL mean
# loss and are NOT rescaled by world size.
DRO_ETA_Q = 0.02
DRO_ADJUSTMENT_C = 0.0     # C: generalization-adjustment constant (0 = plain group-DRO)
WEIGHT_DECAY = 1e-4        # lambda: L2 weight decay in the model update
NUM_WORKERS = 4


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
        image_size=IMAGE_SIZE,
        channels=2,
        num_frames=NUM_FRAMES,
        timesteps=1000,   # number of diffusion steps T
        loss_type="l2",   # L1 or L2
        contour_noise_only=False,  # True = noise only in mask contour region, False = full image
    ).to(device)

    exp_name = "proposal-genstrain-new-26-frames-group-dro"

    trainer = GroupDRODDPTrainer(
        diffusion_model=diffusion,
        input_video_folder=f"{base_path}/dense/train/displacement_dense",
        local_rank=local_rank,
        contour_condition_video_dir=f"{base_path}/dense/train/dense_mask",
        sampling_contour_condition_video_dir=f"{base_path}/dense/test/dense_mask",
        metadata_json_path=metadata_json_path,
        train_batch_size=PER_GPU_BATCH,  # PER-GPU
        train_lr=1e-5,                   # eta: model learning rate
        save_and_sample_every=300,
        train_num_steps=700000,
        gradient_accumulate_every=1,     # GroupDRODDPTrainer requires this to be 1
        ema_decay=0.995,
        amp=True,
        experiment_name=exp_name,
        num_workers=NUM_WORKERS,
        # ---- Group-DRO hyperparameters ----
        dro_eta_q=DRO_ETA_Q,             # eta_q: group-weight learning rate
        dro_adjustment_c=DRO_ADJUSTMENT_C,  # C
        weight_decay=WEIGHT_DECAY,       # lambda
    )

    # Load the latest checkpoint (milestone = -1) to resume; skip if none exists.
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
