import torch

from video_diffusion_pytorch.trainer_group_dro import GroupDROTrainer
from video_diffusion_pytorch.video_diffusion_cross_attention_with_motion_after_only_contour import (
    GaussianDiffusion,
    Unet3D,
)


# DATA BASE PATH
base_path = "/scratch/vst2hb/Dataset_Processing/Data Bank/proposal-genstrain-new/all_data_resampled_26_frames"
metadata_json_path = "/scratch/vst2hb/Dataset_Processing/Data Bank/proposal-genstrain-new/processed_excel.json"

# Proposal dataset dimensions.
IMAGE_SIZE = 64
NUM_FRAMES = 26
BATCH_SIZE = 10

# Group-DRO hyperparameters.
# The q update multiplies a group's raw weight by exp(DRO_ETA_Q * S_g), with
# S_g ~= L_g, the MEAN denoising loss (empirically ~0.4-1.1 here). Pick DRO_ETA_Q
# so exp(DRO_ETA_Q * L_g) stays near 1 -> ~0.01-0.05. DRO_ETA_Q=1.0 nearly doubles
# a group's weight per sample and makes q collapse onto whichever group is sampled
# first (observed: q -> ~1.0 on Healthy by step ~150). Watch the per-group q each
# step: it should stay spread and drift toward the persistently-hardest group,
# never pin near 1.0. Flat at 1/K -> raise it; any group > ~0.7 -> lower it.
DRO_ETA_Q = 0.02
DRO_ADJUSTMENT_C = 0.0     # C: generalization-adjustment constant (0 = plain group-DRO)
# Freeze q at uniform (1/K) and backprop the raw loss -> disables DRO reweighting.
# Sampling is still group-balanced (uniform-over-groups), so this is NOT identical
# to base GenStrain's natural-distribution training. Set False for real group-DRO.
DRO_FREEZE_Q = True
WEIGHT_DECAY = 1e-4        # lambda: L2 weight decay in the model update
NUM_WORKERS = 4


if not torch.cuda.is_available():
    raise RuntimeError("train_group_dro.py requires a CUDA GPU")


model = Unet3D(
    dim=48,
    cond_dim=None,
    channels=2,
    dim_mults=(1, 2, 4, 8),
)

diffusion = GaussianDiffusion(
    model,
    image_size=IMAGE_SIZE,
    channels=2,
    num_frames=NUM_FRAMES,
    timesteps=1000,
    loss_type="l2",
    contour_noise_only=False,
).cuda()

exp_name = "proposal-genstrain-new-26-frames-group-dro"

trainer = GroupDROTrainer(
    diffusion_model=diffusion,
    input_video_folder=f"{base_path}/dense/train/displacement_dense",
    contour_condition_video_dir=f"{base_path}/dense/train/dense_mask",
    sampling_contour_condition_video_dir=f"{base_path}/dense/test/dense_mask",
    metadata_json_path=metadata_json_path,
    train_batch_size=BATCH_SIZE,
    train_lr=1e-5,
    save_and_sample_every=300,
    train_num_steps=700000,
    gradient_accumulate_every=1,
    ema_decay=0.995,
    amp=True,
    experiment_name=exp_name,
    dro_eta_q=DRO_ETA_Q,
    dro_adjustment_c=DRO_ADJUSTMENT_C,
    dro_freeze_q=DRO_FREEZE_Q,
    weight_decay=WEIGHT_DECAY,
    num_workers=NUM_WORKERS,
)

try:
    trainer.load(milestone=-1)
except AssertionError:
    print("no checkpoint found, starting from scratch")

trainer.train()
