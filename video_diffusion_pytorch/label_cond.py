"""Label-conditioned variant of the contour-only displacement diffusion model.

Adds three categorical conditions on top of the binary mask video -- disease
group, slice level (apex/mid/base) and acquisition site -- fed through the
cond_dim path that Unet3D already reserves (its ResnetBlocks take
time_dim + cond_dim; `forward_with_cond_scale` implements classifier-free
guidance). Each label has its own embedding table with a learned null row.
During training every sample's labels are replaced by the null rows with
probability `null_cond_prob`, so the same checkpoint serves three inference
modes (see label_metadata.LabelEncoder.apply_mode):

    full        disease + level + site known (DENSE evaluation)
    level_only  slice level only, which is always known (deployment default)
    none        all null: the unconditional path, i.e. the mask-only model

Why: per-disease scoring (scripts/twist_metrics.py) showed the mask-only model
hedging its amplitude on healthy/MI slices because it cannot tell them from
sign-ambiguous LBBB cases with the same contour, and never committing to a
rotation direction the contour does not reveal. Labels let it stop blending
groups it can now tell apart. They do NOT resolve the direction split inside
unscarred LBBB -- that needs a conduction/activation cue the metadata lacks.

Everything not about labels is inherited unchanged from
`video_diffusion_cross_attention_with_motion_after_only_contour`, so the two
model families stay byte-identical elsewhere. The single-GPU trainer lives
here; the DDP one is trainer_label_cond_ddp.LabelCondDDPTrainer, sharing the
sampling code through LabelSamplingMixin. Checkpoints are interchangeable.
"""

import copy
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils import data
from tqdm import tqdm

from video_diffusion_pytorch.checkpoint_compat import record_disp_scale, verify_disp_scale
from video_diffusion_pytorch.frame_validity import build_frame_mask
from video_diffusion_pytorch.label_metadata import LABEL_MODES, LABEL_NAMES, LabelEncoder
from video_diffusion_pytorch.utils import (
    generate_displacement_quiver_gif_comparison,
    generate_displacement_quiver_gif_predicted_reconstructed_gt,
    generate_mask_disp_side_by_side_gif,
)
from video_diffusion_pytorch.video_diffusion_cross_attention_with_motion_after_only_contour import (
    Dataset,
    GaussianDiffusion,
    Trainer,
    Unet3D,
    autocast,
    cycle,
    default,
    exists,
    extract_noise_mask_from_first_frame,
    finite_metric_items,
    normalize_cond_img,
    prob_mask_like,
    random_pick_condition_videos,
    unnormalize_disp,
)

try:
    import wandb
except ImportError:
    wandb = None

# Checkpoint key holding the label vocabulary the weights were trained with.
LABEL_COND_KEY = "label_cond"


# ----------------------------------------------------------------------------
# UNet with label embeddings
# ----------------------------------------------------------------------------
def _split_dim(total, parts):
    base = total // parts
    dims = [base] * parts
    dims[0] += total - base * parts
    return dims


class LabelCondUnet3D(Unet3D):
    """Unet3D whose has_cond branch embeds (disease, level, site) indices.

    `label_vocab_sizes` are the REAL class counts; each table gets one extra
    null row at index == size. `cond_dim` is split across the three tables.
    """

    def __init__(self, *args, cond_dim=64, label_vocab_sizes=(8, 3, 8), **kwargs):
        if cond_dim is None or int(cond_dim) < len(label_vocab_sizes):
            raise ValueError(f"label conditioning needs cond_dim >= {len(label_vocab_sizes)}, got {cond_dim!r}")
        super().__init__(*args, cond_dim=int(cond_dim), **kwargs)
        # The parent allocates a single null_cond_emb for its (never finished)
        # text-conditioning path. Ours has one null row PER label table, so that
        # parameter would never receive a gradient -- and DDP is built with
        # find_unused_parameters=False, which raises on exactly that. Drop it.
        del self.null_cond_emb
        self.null_cond_emb = None
        self.label_vocab_sizes = tuple(int(n) for n in label_vocab_sizes)
        emb_dims = _split_dim(int(cond_dim), len(self.label_vocab_sizes))
        self.label_embs = nn.ModuleList(
            [nn.Embedding(n + 1, d) for n, d in zip(self.label_vocab_sizes, emb_dims)]
        )
        self.register_buffer(
            "label_null_indices", torch.tensor(self.label_vocab_sizes, dtype=torch.long), persistent=False
        )

    def embed_labels(self, labels, batch, device, null_cond_prob=0.0, null_label_prob=0.0):
        """[B, n_labels] long (or None == all null) -> [B, cond_dim].

        Two dropout mechanisms, both classifier-free style:
          * null_cond_prob  -- a whole sample's labels are swapped for the null
            rows together. Trains the unconditional path ("none" mode).
          * null_label_prob -- each remaining label is nulled independently.
            Trains every PARTIAL combination, in particular "level_only"
            (disease null, slice level kept), which is the deployment mode and
            would otherwise never be seen in training.
        Indices already equal to the null row pass through unchanged, which is
        how partial labels are supplied at inference.
        """
        null = self.label_null_indices.to(device)
        if labels is None:
            labels = null.unsqueeze(0).expand(batch, -1)
        labels = labels.to(device=device, dtype=torch.long)
        if labels.shape != (batch, len(self.label_embs)):
            raise ValueError(f"labels must be [{batch}, {len(self.label_embs)}], got {tuple(labels.shape)}")
        null_rows = null.unsqueeze(0).expand_as(labels)
        if null_cond_prob > 0:
            drop = prob_mask_like((batch,), null_cond_prob, device=device)
            labels = torch.where(drop.unsqueeze(1), null_rows, labels)
        if null_label_prob > 0:
            drop = prob_mask_like(labels.shape, null_label_prob, device=device)
            labels = torch.where(drop, null_rows, labels)
        return torch.cat([emb(labels[:, i]) for i, emb in enumerate(self.label_embs)], dim=-1)

    def forward(
        self,
        x,
        time,
        cond=None,  # cond = [contour_cond_video]
        null_cond_prob=0.0,
        focus_present_mask=None,
        prob_focus_present=0.0,
        labels=None,
        null_label_prob=0.0,
    ):
        assert exists(cond), "cond (the contour video) must be passed"
        batch, device = x.shape[0], x.device

        focus_present_mask = default(
            focus_present_mask, lambda: prob_mask_like((batch,), prob_focus_present, device=device)
        )
        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device=x.device)

        x = self.init_conv(x)
        contour_cond = self.init_conv_contour_cond(cond[0])
        x = self.init_temporal_attn([x, contour_cond], x, pos_bias=time_rel_pos_bias)
        r = x.clone()

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        # The label path: the only functional difference from Unet3D.forward.
        label_emb = self.embed_labels(
            labels, batch, device, null_cond_prob=null_cond_prob, null_label_prob=null_label_prob
        )
        t = torch.cat((t, label_emb.to(t.dtype)), dim=-1)

        h = []
        for block1, block2, cond_conv_1, cond_conv_2, spatial_attn, temporal_attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)
            contour_cond = cond_conv_1(contour_cond)
            contour_cond = cond_conv_2(contour_cond)
            x = spatial_attn([x, contour_cond], x)
            x = temporal_attn([x, contour_cond], x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
            h.append(x)
            x = downsample(x)
            contour_cond = downsample(contour_cond)

        x = self.mid_block1(x, t)
        x = self.mid_spatial_attn([x, contour_cond], x)
        x = self.mid_temporal_attn([x, contour_cond], x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
        x = self.mid_block2(x, t)

        for block1, block2, cond_conv_1, cond_conv_2, spatial_attn, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t)
            x = block2(x, t)
            contour_cond = cond_conv_1(contour_cond)
            contour_cond = cond_conv_2(contour_cond)
            x = spatial_attn([x, contour_cond], x)
            x = temporal_attn([x, contour_cond], x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)
            x = upsample(x)
            contour_cond = upsample(contour_cond)

        x = torch.cat((x, r), dim=1)
        return self.final_conv(x)


# ----------------------------------------------------------------------------
# Diffusion wrapper: thread `labels` through training and both samplers
# ----------------------------------------------------------------------------
def rotation_curves(u, roi, eps=1e-6):
    """Per-frame bulk rotation of a displacement field about the ROI centroid.

    u:   [B, 2, F, H, W] displacement in pixels (channel 0 = x / column,
         channel 1 = y / row). Only the two channels' relative convention
         matters: a global sign flip cancels between prediction and truth.
    roi: [B, H, W] frame-0 myocardium, 1 inside.
    Returns theta [B, F] in degrees: the |r|-weighted mean over ROI pixels of
    the local rotation (r x u) / |r|^2, r being the pixel's offset from the
    frame-0 centroid. Same quantity scripts/twist_metrics.py scores offline,
    so the training curve and the evaluation table read the same number.
    Differentiable in u (it is linear in u), float32 throughout.
    """
    u = u.float()
    roi = roi.float()
    B, _, F, H, W = u.shape
    ys = torch.arange(H, device=u.device, dtype=torch.float32).view(1, H, 1)
    xs = torch.arange(W, device=u.device, dtype=torch.float32).view(1, 1, W)
    n = roi.sum(dim=(1, 2)).clamp(min=1.0).view(B, 1, 1)
    cx = (roi * xs).sum(dim=(1, 2)).view(B, 1, 1) / n
    cy = (roi * ys).sum(dim=(1, 2)).view(B, 1, 1) / n
    rx = (xs - cx) * roi                       # [B, H, W]
    ry = (ys - cy) * roi
    r = torch.sqrt(rx ** 2 + ry ** 2).clamp(min=eps)
    # local rotation * |r| = (rx*uy - ry*ux) / |r|, summed over the ROI
    cross = (rx.unsqueeze(1) * u[:, 1] - ry.unsqueeze(1) * u[:, 0]) / r.unsqueeze(1)  # [B, F, H, W]
    theta = (cross * roi.unsqueeze(1)).sum(dim=(2, 3)) / (r * roi).sum(dim=(1, 2)).clamp(min=eps).unsqueeze(1)
    return torch.rad2deg(theta)


class LabelCondGaussianDiffusion(GaussianDiffusion):
    """GaussianDiffusion + labels + an optional rotation auxiliary loss.

    rotation_loss_weight > 0 adds, to the eps-loss, an L1 penalty between the
    per-frame rotation curve of the single-step x0 estimate and that of the
    ground truth, over the frame-0 myocardium (and the valid frames). The
    curve collapses the whole tangential content of a slice into F numbers,
    so the objective sees twist directly instead of through an 8%-of-frame
    ROI. x0 at high noise is unreliable (predict_start_from_noise amplifies
    eps error by ~1/sqrt(alphas_cumprod)), so each sample's term is weighted
    by min(SNR_t, rotation_loss_snr_clip): the term vanishes at high noise
    and saturates at the clip for low noise (min-SNR-gamma applied to an
    x0-space loss). The curve is in degrees, so with clip 5 the weighted term
    is O(5 x degrees of error); a rotation_loss_weight around 0.01 puts it on
    the scale of the eps-loss. rotation_mae_deg / gt_rotation_abs_deg are
    logged as diagnostics whether or not the loss is on.
    """

    def __init__(self, denoise_fn, *, null_cond_prob=0.2, null_label_prob=0.15,
                 rotation_loss_weight=0.0, rotation_loss_snr_clip=5.0, **kwargs):
        super().__init__(denoise_fn, **kwargs)
        for name, p in (("null_cond_prob", null_cond_prob), ("null_label_prob", null_label_prob)):
            if not 0.0 <= float(p) < 1.0:
                raise ValueError(f"{name} must be in [0, 1), got {p!r}")
        self.null_cond_prob = float(null_cond_prob)
        self.null_label_prob = float(null_label_prob)
        if float(rotation_loss_weight) < 0 or float(rotation_loss_snr_clip) <= 0:
            raise ValueError("rotation_loss_weight must be >= 0 and rotation_loss_snr_clip > 0")
        self.rotation_loss_weight = float(rotation_loss_weight)
        self.rotation_loss_snr_clip = float(rotation_loss_snr_clip)

    def p_losses(self, x_start, t, cond=None, noise=None, valid_frames=None, per_sample=False, **kwargs):
        loss, x0_disp, gt_disp, disp_recon_mse, disp_metrics = super().p_losses(
            x_start, t, cond=cond, noise=noise, valid_frames=valid_frames, per_sample=per_sample, **kwargs
        )
        want_loss = self.rotation_loss_weight > 0
        want_metric = self.disp_metrics
        if not (want_loss or want_metric) or cond is None or not isinstance(cond[0], torch.Tensor):
            return loss, x0_disp, gt_disp, disp_recon_mse, disp_metrics

        # x0_disp / gt_disp come back un-normalized (pixels) and x0 keeps its
        # graph: predict_start_from_noise -> unnormalize_disp are both linear.
        roi = extract_noise_mask_from_first_frame(cond[0])[:, 0, 0]  # [B, H, W]
        f = x0_disp.shape[2]
        theta_pred = rotation_curves(x0_disp, roi)
        theta_gt = rotation_curves(gt_disp, roi)
        err = (theta_pred - theta_gt).abs()  # [B, F] degrees
        if valid_frames is not None:
            frame_mask = build_frame_mask(valid_frames, f, device=err.device, dtype=err.dtype)[:, 0, :, 0, 0]  # [B, F]
            per_sample_mae = (err * frame_mask).sum(1) / frame_mask.sum(1).clamp(min=1.0)
            gt_abs = (theta_gt.abs() * frame_mask).sum(1) / frame_mask.sum(1).clamp(min=1.0)
        else:
            per_sample_mae = err.mean(1)
            gt_abs = theta_gt.abs().mean(1)

        if want_metric:
            disp_metrics = dict(disp_metrics)
            disp_metrics["rotation_mae_deg"] = per_sample_mae.detach().mean()
            disp_metrics["gt_rotation_abs_deg"] = gt_abs.detach().mean()

        if want_loss:
            alphas = self.alphas_cumprod[t].float()
            snr_w = (alphas / (1.0 - alphas).clamp(min=1e-8)).clamp(max=self.rotation_loss_snr_clip)
            rot = snr_w * per_sample_mae  # [B]
            rot = rot if per_sample else rot.mean()
            loss = loss + self.rotation_loss_weight * rot.to(loss.dtype)
            if want_metric:
                disp_metrics["rotation_loss"] = (self.rotation_loss_weight * rot).detach().mean()
        return loss, x0_disp, gt_disp, disp_recon_mse, disp_metrics

    # p_losses forwards **kwargs into denoise_fn unchanged, so training only
    # needs the labels and the dropout probabilities injected here. Sampling
    # never passes them, so the denoiser default of 0 applies there.
    def forward(self, x, cond=None, *args, labels=None, **kwargs):
        return super().forward(
            x, cond, *args, labels=labels,
            null_cond_prob=self.null_cond_prob, null_label_prob=self.null_label_prob, **kwargs,
        )

    def p_mean_variance(self, x, t, clip_denoised: bool, cond=None, cond_scale=1.0, labels=None):
        x_recon = self.predict_start_from_noise(
            x, t=t,
            noise=self.denoise_fn.forward_with_cond_scale(x, t, cond=cond, cond_scale=cond_scale, labels=labels),
        )
        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()
    def p_sample(self, x, t, cond=None, cond_scale=1.0, clip_denoised=True, noise_mask=None, labels=None):
        b = x.shape[0]
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, cond=cond, cond_scale=cond_scale, labels=labels
        )
        noise = torch.randn_like(x)
        if noise_mask is not None:
            noise = noise * noise_mask
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        result = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise
        if noise_mask is not None:
            result = result * noise_mask
        return result

    @torch.inference_mode()
    def p_sample_loop(self, shape, cond=None, cond_scale=1.0, seed=None, labels=None):
        device = self.betas.device
        b = shape[0]
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(int(seed))
        img = torch.randn(shape, device=device, generator=generator)

        noise_mask = None
        if self.contour_noise_only and cond is not None and isinstance(cond[0], torch.Tensor):
            noise_mask = extract_noise_mask_from_first_frame(cond[0])
            img = img * noise_mask

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc="sampling loop time step", total=self.num_timesteps):
            img = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long), cond=[cond[0]],
                cond_scale=cond_scale, noise_mask=noise_mask, labels=labels,
            )
        return unnormalize_disp(img, self.disp_scale)

    @torch.inference_mode()
    def ddim_sample_loop(self, shape, cond=None, cond_scale=1.0, ddim_steps=50, ddim_eta=0.0,
                         spacing="uniform", clip_x_start=None, clip_percentile=0.995,
                         start_t=None, seed=None, labels=None):
        device = self.betas.device
        b = shape[0]
        generator = None
        if seed is not None:
            generator = torch.Generator(device=device).manual_seed(int(seed))
        img = torch.randn(shape, device=device, generator=generator)

        noise_mask = None
        if self.contour_noise_only and cond is not None and isinstance(cond[0], torch.Tensor):
            noise_mask = extract_noise_mask_from_first_frame(cond[0])
            img = img * noise_mask

        time_pairs = self.ddim_timestep_pairs(ddim_steps, spacing=spacing, start_t=start_t)
        for time, time_prev in tqdm(time_pairs, desc="ddim sampling loop time step", total=len(time_pairs)):
            t = torch.full((b,), time, device=device, dtype=torch.long)
            pred_noise = self.denoise_fn.forward_with_cond_scale(img, t, cond=[cond[0]], cond_scale=cond_scale, labels=labels)
            x_start = self.predict_start_from_noise(img, t=t, noise=pred_noise)
            x_start = self.clip_x_start(x_start, mode=clip_x_start, percentile=clip_percentile)

            alpha = self.alphas_cumprod[time]
            alpha_prev = self.alphas_cumprod[time_prev] if time_prev >= 0 else torch.ones_like(alpha)
            sigma = ddim_eta * ((1 - alpha_prev) / (1 - alpha)).sqrt() * (1 - alpha / alpha_prev).sqrt()
            dir_coef = (1 - alpha_prev - sigma**2).clamp(min=0.0).sqrt()
            img = alpha_prev.sqrt() * x_start + dir_coef * pred_noise

            if ddim_eta > 0 and time_prev >= 0:
                noise = torch.randn(img.shape, device=img.device, dtype=img.dtype, generator=generator)
                if noise_mask is not None:
                    noise = noise * noise_mask
                img = img + sigma * noise
            if noise_mask is not None:
                img = img * noise_mask
        return unnormalize_disp(img, self.disp_scale)

    @torch.inference_mode()
    def sample(self, cond=None, cond_scale=1.0, batch_size=16, sampler="ddpm", ddim_steps=50,
               ddim_eta=0.0, ddim_spacing="uniform", ddim_clip_x_start=None,
               ddim_clip_percentile=0.995, ddim_start_t=None, seed=None, labels=None):
        device = next(self.denoise_fn.parameters()).device
        contour_cond_video = cond[0].to(device)
        shape = (contour_cond_video.shape[0], self.channels, self.num_frames, self.image_size, self.image_size)
        cond = [contour_cond_video]
        if labels is not None:
            labels = labels.to(device)

        if sampler == "ddpm":
            return self.p_sample_loop(shape, cond=cond, cond_scale=cond_scale, seed=seed, labels=labels)
        if sampler == "ddim":
            return self.ddim_sample_loop(
                shape, cond=cond, cond_scale=cond_scale, ddim_steps=ddim_steps, ddim_eta=ddim_eta,
                spacing=ddim_spacing, clip_x_start=ddim_clip_x_start, clip_percentile=ddim_clip_percentile,
                start_t=ddim_start_t, seed=seed, labels=labels,
            )
        raise ValueError(f"unknown sampler {sampler!r}, expected 'ddpm' or 'ddim'")


# ----------------------------------------------------------------------------
# Dataset returning labels
# ----------------------------------------------------------------------------
class LabelDataset(Dataset):
    """Dataset + a [3] long tensor of (disease, level, site) indices per item."""

    def __init__(self, *args, label_encoder, **kwargs):
        super().__init__(*args, **kwargs)
        self.label_encoder = label_encoder
        # Encode once: the lookup is a few dict probes per file but __getitem__
        # runs every step.
        self.labels = [
            torch.tensor(label_encoder.encode(Path(input_path).name), dtype=torch.long)
            for input_path, _ in self.video_pairs
        ]
        hit, miss = label_encoder.coverage([Path(p).name for p, _ in self.video_pairs])
        print(f"label metadata: {hit} files matched, {miss} without an entry (disease/site -> null)")
        counts = {}
        for lab in self.labels:
            name = label_encoder.describe(lab.tolist())
            counts[name] = counts.get(name, 0) + 1
        for name, n in sorted(counts.items(), key=lambda kv: -kv[1])[:20]:
            print(f"  {n:5d}  {name}")

    def __getitem__(self, index):
        video, cond, valid_frames = super().__getitem__(index)
        return video, cond, valid_frames, self.labels[index]


# ----------------------------------------------------------------------------
# Trainer
# ----------------------------------------------------------------------------
def record_label_cond(ckpt, model, encoder):
    """Stamp cond_dim + vocabulary into the checkpoint (see verify_label_cond)."""
    ckpt[LABEL_COND_KEY] = {
        "cond_dim": int(sum(e.embedding_dim for e in model.denoise_fn.label_embs)),
        "vocab_sizes": list(model.denoise_fn.label_vocab_sizes),
        "null_cond_prob": float(model.null_cond_prob),
        "null_label_prob": float(getattr(model, "null_label_prob", 0.0)),
        **encoder.spec(),
    }
    return ckpt


def verify_label_cond(ckpt, model, encoder, milestone=None):
    """Refuse a checkpoint whose label vocabulary differs from this run's.

    Embedding rows are addressed by index, so a reordered or resized vocabulary
    would silently hand every slice a different class. A checkpoint with no
    stamp at all was trained without labels: it cannot be loaded here.
    """
    where = f"model-{milestone}.pt" if milestone is not None else "checkpoint"
    saved = ckpt.get(LABEL_COND_KEY)
    if saved is None:
        raise ValueError(
            f"{where} carries no label-conditioning stamp, so it was trained by a different "
            f"model family (mask-only). Sample it with inference_full_region.py instead."
        )
    current = record_label_cond({}, model, encoder)[LABEL_COND_KEY]
    # use_site is part of the contract too: a model trained with site nulled
    # has never trained the real site rows, so feeding them at inference
    # would inject untrained embeddings.
    for key in ("cond_dim", "vocab_sizes", "vocab", "use_site"):
        if saved.get(key) != current[key]:
            raise ValueError(
                f"{where} was trained with label_cond.{key}={saved.get(key)!r}, but this run has "
                f"{current[key]!r}. The embedding rows are addressed by index, so the vocabulary "
                f"must match exactly; use the config the checkpoint was trained with."
            )
    return saved


class LabelSamplingMixin:
    """Label-aware preview/inference sampling shared by the single-GPU and DDP
    trainers. Expects self.ema_model (a LabelCondGaussianDiffusion) and
    self.label_encoder / self.preview_label_mode."""

    # ---- labels for sampling ----------------------------------------------
    def labels_for(self, filenames, mode=None):
        """[N, 3] long tensor for a list of condition filenames under a label mode."""
        mode = self.preview_label_mode if mode is None else mode
        rows = [self.label_encoder.apply_mode(self.label_encoder.encode(f), mode) for f in filenames]
        return torch.tensor(rows, dtype=torch.long)

    # ---- sampling ---------------------------------------------------------
    def _sample(self, contour_cond_video, labels, cond_scale, sampler, ddim_steps, ddim_eta, ddim_spacing,
                ddim_clip_x_start, ddim_clip_percentile, ddim_start_t, seed, zero_frame0, valid_frames):
        sampled = self.ema_model.sample(
            cond=[contour_cond_video], cond_scale=cond_scale, batch_size=contour_cond_video.shape[0],
            sampler=sampler, ddim_steps=ddim_steps, ddim_eta=ddim_eta, ddim_spacing=ddim_spacing,
            ddim_clip_x_start=ddim_clip_x_start, ddim_clip_percentile=ddim_clip_percentile,
            ddim_start_t=ddim_start_t, seed=seed, labels=labels,
        )
        sampled = sampled.cpu().numpy()  # [N, 2, F, H, W]
        # Lagrangian displacement is zero at frame 0 by definition, and zero on
        # padded frames past the valid count. Anything the sampler puts there is
        # pure error under StrainAnalysis' epe scoring, so clear it.
        if zero_frame0:
            sampled[:, :, 0] = 0.0
        if valid_frames is not None:
            for i, n_valid in enumerate(np.broadcast_to(np.asarray(valid_frames), (sampled.shape[0],))):
                if n_valid is not None and 0 < int(n_valid) < sampled.shape[2]:
                    sampled[i, :, int(n_valid):] = 0.0
        return np.expand_dims(sampled, axis=1)  # [N, 1, 2, F, H, W]

    @torch.inference_mode()
    def sample_and_save(self, milestone, contour_cond_video=None, cond_filenames=None, cond_scale=1.0,
                        save_folder="./sampled_videos_infos", sampler="ddpm", ddim_steps=50, ddim_eta=0.0,
                        ddim_spacing="uniform", ddim_clip_x_start=None, ddim_clip_percentile=0.995,
                        ddim_start_t=None, seed=None, labels=None, zero_frame0=True, valid_frames=None):
        sampled_videos = self._sample(contour_cond_video, labels, cond_scale, sampler, ddim_steps, ddim_eta,
                                      ddim_spacing, ddim_clip_x_start, ddim_clip_percentile, ddim_start_t, seed,
                                      zero_frame0, valid_frames)
        contour_np = contour_cond_video.cpu().numpy() * 255.0
        save_folder = Path(save_folder)
        sampled_folder = save_folder / "inference_disps"
        sampled_folder.mkdir(exist_ok=True, parents=True)
        for i in range(sampled_videos.shape[0]):
            base_name = cond_filenames[i] if cond_filenames is not None and i < len(cond_filenames) else f"sample_{i:03d}"
            np.save(sampled_folder / base_name, sampled_videos[i])
        output_gif_dir = save_folder / "quiver_plots"
        output_gif_dir.mkdir(parents=True, exist_ok=True)
        tag = cond_filenames[0].split(".npy")[0] if cond_filenames else "sample_000"
        if labels is not None:
            tag += "_" + self.label_encoder.describe(labels[0].tolist()).replace("/", "-")
        generate_mask_disp_side_by_side_gif(contour_np[0], sampled_videos[0], output_path=f"{output_gif_dir}/milestone_{milestone}_{tag}.gif")

    @torch.inference_mode()
    def sample_and_save_one_video_at_a_time(self, milestone, contour_cond_video=None, cond_filenames=None,
                                            reconstructed_disp=None, gt_disp=None, cond_scale=1.0,
                                            save_folder="./sampled_videos_infos", sampler="ddpm", ddim_steps=50,
                                            ddim_eta=0.0, ddim_spacing="uniform", ddim_clip_x_start=None,
                                            ddim_clip_percentile=0.995, ddim_start_t=None, seed=None,
                                            labels=None, zero_frame0=True, valid_frames=None, make_gif=True):
        sampled_videos = self._sample(contour_cond_video, labels, cond_scale, sampler, ddim_steps, ddim_eta,
                                      ddim_spacing, ddim_clip_x_start, ddim_clip_percentile, ddim_start_t, seed,
                                      zero_frame0, valid_frames)
        contour_np = contour_cond_video.cpu().numpy() * 255.0
        save_folder = Path(save_folder)
        sampled_folder = save_folder / "inference_disps"
        sampled_folder.mkdir(exist_ok=True, parents=True)
        output_gif_dir = save_folder / "quiver_plots"
        output_gif_dir.mkdir(parents=True, exist_ok=True)

        for i in range(sampled_videos.shape[0]):
            base_name = cond_filenames[i] if cond_filenames is not None and i < len(cond_filenames) else f"sample_{i:03d}"
            single = sampled_videos[i]  # [1, 2, F, H, W]
            np.save(sampled_folder / base_name, single)
            if not make_gif:
                continue
            gif = f"{output_gif_dir}/{base_name.split('.npy')[0]}.gif"
            if reconstructed_disp is None and gt_disp is None:
                generate_mask_disp_side_by_side_gif(contour_np[i], single, output_path=gif)
            elif reconstructed_disp is None:
                generate_displacement_quiver_gif_comparison(single, gt_disp, output_path=gif, titles=["Predicted", "Ground Truth"])
            elif gt_disp is None:
                generate_displacement_quiver_gif_comparison(single, reconstructed_disp, output_path=gif, titles=["Predicted", "Reconstructed"])
            else:
                generate_displacement_quiver_gif_predicted_reconstructed_gt(single, reconstructed_disp, gt_disp, output_path=gif)


class LabelCondTrainer(LabelSamplingMixin, Trainer):
    """Trainer whose batches carry labels, with label-aware preview sampling
    and checkpoint stamping. Single GPU."""

    def __init__(self, diffusion_model, input_video_folder, *, label_encoder, preview_label_mode="full",
                 contour_condition_video_dir=None, valid_frames_missing="full", inference_only=False, **kwargs):
        super().__init__(
            diffusion_model, input_video_folder,
            contour_condition_video_dir=contour_condition_video_dir,
            valid_frames_missing=valid_frames_missing, inference_only=inference_only, **kwargs,
        )
        self.label_encoder = label_encoder
        if preview_label_mode not in LABEL_MODES:
            raise ValueError(f"preview_label_mode must be one of {LABEL_MODES}, got {preview_label_mode!r}")
        self.preview_label_mode = preview_label_mode
        if not isinstance(diffusion_model.denoise_fn, LabelCondUnet3D):
            raise TypeError("LabelCondTrainer needs a LabelCondGaussianDiffusion wrapping a LabelCondUnet3D")

        if not inference_only:
            # Replace the plain Dataset the parent built with the labelled one.
            self.ds = LabelDataset(
                input_video_folder,
                self.image_size,
                contour_condition_video_dir,
                channels=diffusion_model.channels,
                num_frames=diffusion_model.num_frames,
                valid_frames_map=self.valid_frames_map,
                valid_frames_missing=valid_frames_missing,
                label_encoder=label_encoder,
            )
            self.dl = cycle(data.DataLoader(self.ds, batch_size=self.batch_size, shuffle=True, pin_memory=True))

    # ---- checkpoints -------------------------------------------------------
    def save(self, milestone):
        ckpt = {
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
            "scaler": self.scaler.state_dict(),
        }
        record_disp_scale(ckpt, self.model)
        record_label_cond(ckpt, self.model, self.label_encoder)
        torch.save(ckpt, str(self.checkpoints_folder / f"model-{milestone}.pt"))

    def load(self, milestone, **kwargs):
        if milestone == -1:
            all_milestones = [int(p.stem.split("-")[-1]) for p in Path(self.checkpoints_folder).glob("**/*.pt")]
            assert len(all_milestones) > 0, "need to have at least one milestone to load from latest checkpoint (milestone == -1)"
            milestone = max(all_milestones)
        ckpt_path = str(self.checkpoints_folder / f"model-{milestone}.pt")
        try:
            ckpt = torch.load(ckpt_path, weights_only=False)
        except TypeError:
            ckpt = torch.load(ckpt_path)
        print(f"loaded checkpoint: model-{milestone}.pt\n")
        verify_disp_scale(ckpt, self.model, milestone)
        verify_label_cond(ckpt, self.model, self.label_encoder, milestone)
        self.step = ckpt["step"]
        self.model.load_state_dict(ckpt["model"], **kwargs)
        self.ema_model.load_state_dict(ckpt["ema"], **kwargs)
        self.scaler.load_state_dict(ckpt["scaler"])

    # ---- training loop ----------------------------------------------------
    def train(self, prob_focus_present=0.0, focus_present_mask=None, log_fn=lambda *a, **k: None):
        current_epoch = self.step // self.steps_per_epoch
        self._epoch_loss_sum = 0.0
        self._epoch_disp_recon_mse_sum = 0.0
        self._epoch_step_count = 0
        epoch_bar = tqdm(total=self.steps_per_epoch, desc=f"epoch {current_epoch}", unit="step", leave=True, dynamic_ncols=True)

        while self.step < self.train_num_steps:
            for _ in range(self.gradient_accumulate_every):
                input_video, contour_cond_video, valid_frames, labels = next(self.dl)
                input_video = input_video.cuda()
                contour_cond_video = contour_cond_video.cuda()
                labels = labels.cuda()
                valid_frames = valid_frames.cuda() if self.use_frame_validity else None

                with autocast(enabled=self.amp):
                    loss, x0, x_start, disp_recon_mse, disp_metrics = self.model(
                        input_video,
                        cond=[contour_cond_video],
                        valid_frames=valid_frames,
                        labels=labels,
                        prob_focus_present=prob_focus_present,
                        focus_present_mask=focus_present_mask,
                    )
                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()

                disp_items = finite_metric_items(disp_metrics)
                disp_parts = "".join(f" | {k}: {v}" for k, v in sorted(disp_items.items()))
                epoch_bar.write(f"{self.step}: {loss.item()} | disp_recon_mse: {disp_recon_mse.item()}{disp_parts}")

            self._loss_sum += loss.item()
            self._loss_count += 1
            log = {
                "loss": loss.item(),
                "loss_cummean": self._loss_sum / self._loss_count,
                "disp_recon_mse": disp_recon_mse.item(),
                "step": self.step,
                **disp_items,
            }

            self._epoch_loss_sum += loss.item()
            self._epoch_disp_recon_mse_sum += disp_recon_mse.item()
            self._epoch_step_count += 1
            epoch_bar.update(1)
            epoch_bar.set_postfix(loss=loss.item())
            if self._epoch_step_count >= self.steps_per_epoch:
                epoch_loss = self._epoch_loss_sum / self._epoch_step_count
                epoch_disp_recon_mse = self._epoch_disp_recon_mse_sum / self._epoch_step_count
                epoch = (self.step + 1) // self.steps_per_epoch
                epoch_bar.write(f"  epoch {epoch} | epoch_loss {epoch_loss} | epoch_disp_recon_mse {epoch_disp_recon_mse}")
                log["epoch"] = epoch
                log["epoch_loss"] = epoch_loss
                log["epoch_disp_recon_mse"] = epoch_disp_recon_mse
                epoch_bar.reset()
                epoch_bar.set_description(f"epoch {epoch + 1}")
                self._epoch_loss_sum = 0.0
                self._epoch_disp_recon_mse_sum = 0.0
                self._epoch_step_count = 0

            if exists(self.max_grad_norm):
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                self.save(milestone)

                sample_contour_cond, sample_cond_filenames = random_pick_condition_videos(
                    num_samples=1, contour_cond_video_dir=self.sampling_contour_condition_video_dir,
                )
                sample_contour_cond = normalize_cond_img(sample_contour_cond)
                device = next(self.ema_model.parameters()).device
                sample_contour_cond = sample_contour_cond.to(device)
                sample_labels = self.labels_for(sample_cond_filenames).to(device)

                self.sample_and_save(
                    milestone,
                    contour_cond_video=sample_contour_cond,
                    cond_filenames=sample_cond_filenames,
                    labels=sample_labels,
                    save_folder=f"./{self.experiment_name}/sampled_videos_infos",
                    sampler=self.preview_sampler,
                    ddim_steps=self.preview_ddim_steps,
                    ddim_eta=self.preview_ddim_eta,
                    ddim_spacing=self.preview_ddim_spacing,
                    ddim_clip_x_start=self.preview_ddim_clip_x_start,
                    ddim_clip_percentile=self.preview_ddim_clip_percentile,
                )

            if self.use_wandb:
                wandb.log(log, step=self.step)
            log_fn(log)
            self.step += 1

        epoch_bar.close()
        if self.use_wandb:
            wandb.finish()
        print("training completed")
