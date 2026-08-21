"""Checkpoint/config compatibility checks shared by every Trainer variant.

Some config fields describe the UNITS the weights were trained in rather than a
free-running hyperparameter. Resuming such a checkpoint under a different value
does not fail — it trains on, and samples, in silently rescaled units. These
helpers stamp those fields into the checkpoint and refuse the mismatch on load.

Kept in its own module (importing nothing from the package) so all nine
save/load implementations share one definition and cannot drift apart.
"""

# Key under which the displacement normalization divisor is stored.
DISP_SCALE_KEY = "disp_scale"

# What a checkpoint without the key was necessarily trained at: the divisor was
# hardcoded to 5.0 before it became configurable. Assuming this (rather than
# refusing to load) is what keeps every pre-existing checkpoint loadable.
LEGACY_DISP_SCALE = 5.0


def record_disp_scale(ckpt, model):
    """Stamp the displacement scale this run is using into the checkpoint dict.

    `model` is the unwrapped GaussianDiffusion (raw_model under DDP). Falls back
    to the legacy value for any model predating the attribute, so this never
    raises while saving.
    """
    ckpt[DISP_SCALE_KEY] = float(getattr(model, DISP_SCALE_KEY, LEGACY_DISP_SCALE))
    return ckpt


def verify_disp_scale(ckpt, model, milestone=None):
    """Raise if `ckpt` was trained under a different displacement scale.

    The weights encode x0 in units of the scale they were trained with. Loading
    them under a different one rescales every prediction by exactly the ratio,
    with nothing in the loss or the metrics to reveal it: the dimensionless
    ratios (mse_disp_rel, DRO's r_g) are invariant, and the eps-loss just settles
    at a different floor that looks like an ordinary change in difficulty. Only
    the saved displacement fields come out wrong. Hence a hard failure naming
    both values rather than a warning.

    A checkpoint with no recorded scale is treated as LEGACY_DISP_SCALE, so old
    checkpoints load unchanged under the default and fail loudly under any other.
    """
    saved = float(ckpt.get(DISP_SCALE_KEY, LEGACY_DISP_SCALE))
    current = float(getattr(model, DISP_SCALE_KEY, LEGACY_DISP_SCALE))
    if saved != current:
        where = f"model-{milestone}.pt" if milestone is not None else "checkpoint"
        raise ValueError(
            f"{where} was trained with diffusion.disp_scale={saved}, but this run is "
            f"configured for {current}. The weights predict displacement in the units "
            f"of the scale they were trained with, so resuming or sampling under a "
            f"different one silently rescales every output. Set diffusion.disp_scale "
            f"to {saved} to use this checkpoint, or train a new one at {current}."
            + (
                f" (No scale was recorded in {where}, so it is assumed to be the "
                f"pre-configurable default of {LEGACY_DISP_SCALE}.)"
                if DISP_SCALE_KEY not in ckpt
                else ""
            )
        )
    return saved
