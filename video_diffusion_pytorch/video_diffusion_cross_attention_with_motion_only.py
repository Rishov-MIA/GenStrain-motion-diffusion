import cv2
import os
import numpy as np
import math
import random
import copy
import torch
from torch import nn, einsum
import torch.nn.functional as F
from functools import partial

from torch.utils import data
from pathlib import Path
from torch.optim import Adam
from torchvision import transforms as T, utils
from torch.cuda.amp import autocast, GradScaler
from PIL import Image

from tqdm import tqdm
from einops import rearrange
from einops_exts import check_shape, rearrange_many

from rotary_embedding_torch import RotaryEmbedding

from video_diffusion_pytorch.utils import (
    generate_displacement_quiver_gifs,
    generate_mask_disp_side_by_side_gif,
    generate_displacement_quiver_gifs_with_gt,
    generate_displacement_quiver_gif_predicted_reconstructed_gt,
    generate_displacement_quiver_gif_comparison
)

# helpers functions


def exists(x):
    return x is not None


def noop(*args, **kwargs):
    pass


def is_odd(n):
    return (n % 2) == 1


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def cycle(dl):
    while True:
        for data in dl:
            yield data


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


def is_list_str(x):
    if not isinstance(x, (list, tuple)):
        return False
    return all([type(el) == str for el in x])


# relative positional bias


class RelativePositionBias(nn.Module):
    def __init__(self, heads=8, num_buckets=32, max_distance=128):
        super().__init__()
        self.num_buckets = num_buckets
        self.max_distance = max_distance
        self.relative_attention_bias = nn.Embedding(num_buckets, heads)

    @staticmethod
    def _relative_position_bucket(relative_position, num_buckets=32, max_distance=128):
        ret = 0
        n = -relative_position

        num_buckets //= 2
        ret += (n < 0).long() * num_buckets
        n = torch.abs(n)

        max_exact = num_buckets // 2
        is_small = n < max_exact

        val_if_large = (
            max_exact
            + (torch.log(n.float() / max_exact) / math.log(max_distance / max_exact) * (num_buckets - max_exact)).long()
        )
        val_if_large = torch.min(val_if_large, torch.full_like(val_if_large, num_buckets - 1))

        ret += torch.where(is_small, n, val_if_large)
        return ret

    def forward(self, n, device):
        q_pos = torch.arange(n, dtype=torch.long, device=device)
        k_pos = torch.arange(n, dtype=torch.long, device=device)
        rel_pos = rearrange(k_pos, "j -> 1 j") - rearrange(q_pos, "i -> i 1")
        rp_bucket = self._relative_position_bucket(
            rel_pos, num_buckets=self.num_buckets, max_distance=self.max_distance
        )
        values = self.relative_attention_bias(rp_bucket)
        return rearrange(values, "i j h -> h i j")


# small helper modules


class EMA:
    def __init__(self, beta):
        super().__init__()
        self.beta = beta

    def update_model_average(self, ma_model, current_model):
        for current_params, ma_params in zip(current_model.parameters(), ma_model.parameters()):
            old_weight, up_weight = ma_params.data, current_params.data
            ma_params.data = self.update_average(old_weight, up_weight)

    def update_average(self, old, new):
        if old is None:
            return new
        return old * self.beta + (1 - self.beta) * new


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, prev_x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + prev_x


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


def Upsample(dim):
    return nn.ConvTranspose3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


def Downsample(dim):
    return nn.Conv3d(dim, dim, (1, 4, 4), (1, 2, 2), (0, 1, 1))


class LayerNorm(nn.Module):
    def __init__(self, dim, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(1, dim, 1, 1, 1))

    def forward(self, x):
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) / (var + self.eps).sqrt() * self.gamma


class RMSNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.scale = dim**0.5
        self.gamma = nn.Parameter(torch.ones(dim, 1, 1, 1))

    def forward(self, x):
        return F.normalize(x, dim=1) * self.scale * self.gamma


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x, **kwargs):
        if type(x) == list:
            x[0] = self.norm(x[0])
            x[1] = self.norm(x[1])
        else:
            x = self.norm(x)
        return self.fn(x, **kwargs)


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out):
        super().__init__()
        self.proj = nn.Conv3d(dim, dim_out, (1, 3, 3), padding=(0, 1, 1))
        self.norm = RMSNorm(dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        return self.act(x)


class ResnetBlock(nn.Module):
    def __init__(self, dim, dim_out, *, time_emb_dim=None):
        super().__init__()
        self.mlp = nn.Sequential(nn.SiLU(), nn.Linear(time_emb_dim, dim_out * 2)) if exists(time_emb_dim) else None

        self.block1 = Block(dim, dim_out)
        self.block2 = Block(dim_out, dim_out)
        self.res_conv = nn.Conv3d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):

        scale_shift = None
        if exists(self.mlp):
            assert exists(time_emb), "time emb must be passed in"
            time_emb = self.mlp(time_emb)
            time_emb = rearrange(time_emb, "b c -> b c 1 1 1")
            scale_shift = time_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)
        return h + self.res_conv(x)


class SpatialLinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_q = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.to_kv = nn.Conv2d(dim, hidden_dim * 2, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, f, h, w = x[0].shape
        x[0] = rearrange(x[0], "b c f h w -> (b f) c h w")
        x[1] = rearrange(x[1], "b c f h w -> (b f) c h w")

        q = self.to_q(x[0])
        kv = self.to_kv(x[1]).chunk(2, dim=1)
        qkv = (q, kv[0], kv[1])
        q, k, v = rearrange_many(qkv, "b (h c) x y -> b h c (x y)", h=self.heads)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        out = self.to_out(out)
        return rearrange(out, "(b f) c h w -> b c f h w", b=b)


# attention along space and time


class EinopsToAndFrom(nn.Module):
    def __init__(self, from_einops, to_einops, fn):
        super().__init__()
        self.from_einops = from_einops
        self.to_einops = to_einops
        self.fn = fn

    def forward(self, x, **kwargs):
        shape = x[0].shape
        reconstitute_kwargs = dict(tuple(zip(self.from_einops.split(" "), shape)))
        x[0] = rearrange(x[0], f"{self.from_einops} -> {self.to_einops}")
        x[1] = rearrange(x[1], f"{self.from_einops} -> {self.to_einops}")
        x = self.fn(x, **kwargs)
        x = rearrange(x, f"{self.to_einops} -> {self.from_einops}", **reconstitute_kwargs)
        return x


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32, rotary_emb=None):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.rotary_emb = rotary_emb
        self.to_q = nn.Linear(dim, hidden_dim, bias=False)
        self.to_kv = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x, pos_bias=None, focus_present_mask=None):
        n, device = x[0].shape[-2], x[0].device
        q = self.to_q(x[0])
        kv = self.to_kv(x[1]).chunk(2, dim=-1)
        qkv = (q, kv[0], kv[1])

        if exists(focus_present_mask) and focus_present_mask.all():
            values = qkv[-1]
            return self.to_out(values)

        q, k, v = rearrange_many(qkv, "... n (h d) -> ... h n d", h=self.heads)

        q = q * self.scale

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        sim = einsum("... h i d, ... h j d -> ... h i j", q, k)

        if exists(pos_bias):
            sim = sim + pos_bias

        if exists(focus_present_mask) and not (~focus_present_mask).all():
            attend_all_mask = torch.ones((n, n), device=device, dtype=torch.bool)
            attend_self_mask = torch.eye(n, device=device, dtype=torch.bool)

            mask = torch.where(
                rearrange(focus_present_mask, "b -> b 1 1 1 1"),
                rearrange(attend_self_mask, "i j -> 1 1 1 i j"),
                rearrange(attend_all_mask, "i j -> 1 1 1 i j"),
            )

            sim = sim.masked_fill(~mask, -torch.finfo(sim.dtype).max)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = einsum("... h i j, ... h j d -> ... h i d", attn, v)
        out = rearrange(out, "... h n d -> ... n (h d)")
        return self.to_out(out)


# model


class Conv1x1(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.conv = nn.Conv3d(dim_in, dim_out, kernel_size=1, stride=1, padding=0)

    def forward(self, x):
        return self.conv(x)


class Unet3D(nn.Module):
    def __init__(
        self,
        dim,
        cond_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        attn_heads=8,
        attn_dim_head=32,
        use_bert_text_cond=False,
        init_dim=None,
        init_kernel_size=7,
        use_sparse_linear_attn=True,
        block_type="resnet",
    ):
        super().__init__()
        self.channels = channels

        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))

        temporal_attn = lambda dim: EinopsToAndFrom(
            "b c f h w", "b (h w) f c", Attention(dim, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb)
        )

        self.time_rel_pos_bias = RelativePositionBias(heads=attn_heads, max_distance=32)

        # initial conv

        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(
            channels, init_dim, (1, init_kernel_size, init_kernel_size), padding=(0, init_padding, init_padding)
        )

        # motion condition has 2 channels (x and y displacement)
        motion_cond_channels = 2
        self.init_conv_motion_cond = nn.Conv3d(
            motion_cond_channels,
            init_dim,
            (1, init_kernel_size, init_kernel_size),
            padding=(0, init_padding, init_padding),
        )

        self.init_temporal_attn = Residual(PreNorm(init_dim, temporal_attn(init_dim)))

        # dimensions

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        # time conditioning

        time_dim = dim * 4
        self.time_mlp = nn.Sequential(
            SinusoidalPosEmb(dim), nn.Linear(dim, time_dim), nn.GELU(), nn.Linear(time_dim, time_dim)
        )

        # condition dimension (for text or external embeddings)
        self.has_cond = exists(cond_dim)
        self.null_cond_emb = nn.Parameter(torch.randn(1, cond_dim)) if self.has_cond else None
        cond_dim_total = time_dim + (cond_dim if exists(cond_dim) else 0)

        # UNet encoder/decoder
        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = ResnetBlock

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = idx >= len(in_out) - 1
            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=cond_dim_total),
                        block_klass(dim_out, dim_out, time_emb_dim=cond_dim_total),
                        Conv1x1(dim_in, dim_out),
                        Conv1x1(dim_out, dim_out),
                        (
                            Residual(PreNorm(dim_out, SpatialLinearAttention(dim_out, heads=attn_heads)))
                            if use_sparse_linear_attn
                            else nn.Identity()
                        ),
                        Residual(PreNorm(dim_out, temporal_attn(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=cond_dim_total)

        spatial_attn = EinopsToAndFrom("b c f h w", "b f (h w) c", Attention(mid_dim, heads=attn_heads))

        self.mid_spatial_attn = Residual(PreNorm(mid_dim, spatial_attn))
        self.mid_temporal_attn = Residual(PreNorm(mid_dim, temporal_attn(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=cond_dim_total)

        for idx, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = idx >= len(in_out) - 1
            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=cond_dim_total),
                        block_klass(dim_in, dim_in, time_emb_dim=cond_dim_total),
                        Conv1x1(dim_out, dim_in),
                        Conv1x1(dim_in, dim_in),
                        (
                            Residual(PreNorm(dim_in, SpatialLinearAttention(dim_in, heads=attn_heads)))
                            if use_sparse_linear_attn
                            else nn.Identity()
                        ),
                        Residual(PreNorm(dim_in, temporal_attn(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = default(out_dim, channels)
        self.final_conv = nn.Sequential(block_klass(dim * 2, dim), nn.Conv3d(dim, out_dim, 1))

    def forward_with_cond_scale(self, *args, cond_scale=2.0, **kwargs):
        logits = self.forward(*args, null_cond_prob=0.0, **kwargs)
        if cond_scale == 1 or not self.has_cond:
            return logits

        null_logits = self.forward(*args, null_cond_prob=1.0, **kwargs)
        return null_logits + (logits - null_logits) * cond_scale

    def forward(
        self,
        x,
        time,
        cond=None,  # cond = [motion_cond_video]  shape [B, 2, F, H, W]
        null_cond_prob=0.0,
        focus_present_mask=None,
        prob_focus_present=0.0,
    ):
        assert not (self.has_cond and not exists(cond)), "cond must be passed in if cond_dim specified"
        batch, device = x.shape[0], x.device

        focus_present_mask = default(
            focus_present_mask, lambda: prob_mask_like((batch,), prob_focus_present, device=device)
        )

        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device=x.device)

        x = self.init_conv(x)

        motion_cond = cond[0]  # [B, 2, F, H, W]
        motion_cond = self.init_conv_motion_cond(motion_cond)

        x = self.init_temporal_attn([x, motion_cond], x, pos_bias=time_rel_pos_bias)

        r = x.clone()

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        if self.has_cond:
            print("dont come here")

        h = []
        for block1, block2, cond_conv_1, cond_conv_2, spatial_attn, temporal_attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)

            motion_cond = cond_conv_1(motion_cond)
            motion_cond = cond_conv_2(motion_cond)

            x = spatial_attn([x, motion_cond], x)
            x = temporal_attn([x, motion_cond], x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)

            h.append(x)
            x = downsample(x)

            motion_cond = downsample(motion_cond)

        x = self.mid_block1(x, t)

        x = self.mid_spatial_attn([x, motion_cond], x)

        x = self.mid_temporal_attn(
            [x, motion_cond], x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask
        )

        x = self.mid_block2(x, t)

        for block1, block2, cond_conv_1, cond_conv_2, spatial_attn, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)

            x = block1(x, t)
            x = block2(x, t)

            motion_cond = cond_conv_1(motion_cond)
            motion_cond = cond_conv_2(motion_cond)

            x = spatial_attn([x, motion_cond], x)
            x = temporal_attn([x, motion_cond], x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)

            x = upsample(x)

            motion_cond = upsample(motion_cond)

        x = torch.cat((x, r), dim=1)
        return self.final_conv(x)


# gaussian diffusion trainer class


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def cosine_beta_schedule(timesteps, s=0.008):
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(((x / timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.9999)


class GaussianDiffusion(nn.Module):
    def __init__(
        self,
        denoise_fn,
        *,
        image_size,
        num_frames,
        text_use_bert_cls=False,
        channels=3,
        timesteps=1000,
        loss_type="l1",
        use_dynamic_thres=False,
        dynamic_thres_percentile=0.9,
        global_means=None,
        global_stds=None,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn

        betas = cosine_beta_schedule(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        register_buffer("posterior_variance", posterior_variance)
        register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        register_buffer(
            "posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        if global_means is None:
            global_means = torch.zeros(channels, dtype=torch.float32)
        else:
            global_means = torch.as_tensor(global_means, dtype=torch.float32)

        if global_stds is None:
            global_stds = torch.ones(channels, dtype=torch.float32)
        else:
            global_stds = torch.as_tensor(global_stds, dtype=torch.float32)

        assert global_means.shape[0] == channels
        assert global_stds.shape[0] == channels

        register_buffer("global_means", global_means)
        register_buffer("global_stds", global_stds)

        self.text_use_bert_cls = text_use_bert_cls
        self.use_dynamic_thres = use_dynamic_thres
        self.dynamic_thres_percentile = dynamic_thres_percentile

    def q_mean_variance(self, x_start, t):
        mean = extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
        variance = extract(1.0 - self.alphas_cumprod, t, x_start.shape)
        log_variance = extract(self.log_one_minus_alphas_cumprod, t, x_start.shape)
        return mean, variance, log_variance

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
            - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start
            + extract(self.posterior_mean_coef2, t, x_t.shape) * x_t
        )
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def p_mean_variance(self, x, t, clip_denoised: bool, cond=None, cond_scale=1.0):
        x_recon = self.predict_start_from_noise(
            x, t=t, noise=self.denoise_fn.forward_with_cond_scale(x, t, cond=cond, cond_scale=cond_scale)
        )

        if clip_denoised:
            s = 1.0
            if self.use_dynamic_thres:
                s = torch.quantile(rearrange(x_recon, "b ... -> b (...)").abs(), self.dynamic_thres_percentile, dim=-1)
                s.clamp_(min=1.0)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()
    def p_sample(self, x, t, cond=None, cond_scale=1.0, clip_denoised=True):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, cond=cond, cond_scale=cond_scale
        )
        noise = torch.randn_like(x)
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        return model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

    @torch.inference_mode()
    def p_sample_loop(self, shape, cond=None, cond_scale=1.0):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc="sampling loop time step", total=self.num_timesteps):
            img = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long), cond=[cond[0]], cond_scale=cond_scale
            )

        return unnormalize_divide_by_5(img)

    @torch.inference_mode()
    def sample(self, cond=None, cond_scale=1.0, batch_size=16):
        device = next(self.denoise_fn.parameters()).device

        if is_list_str(cond):
            cond = bert_embed(tokenize(cond)).to(device)
        elif len(cond) == 1 and isinstance(cond[0], torch.Tensor):
            motion_cond_video = cond[0].to(device)

        batch_size = motion_cond_video.shape[0]
        image_size = self.image_size
        channels = self.channels
        num_frames = self.num_frames
        return self.p_sample_loop(
            (batch_size, channels, num_frames, image_size, image_size), cond=[motion_cond_video], cond_scale=cond_scale
        )

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))
        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, cond=None, noise=None, **kwargs):
        # cond: [motion_cond_video]  shape [B, 2, F, H, W]
        b, c, f, h, w, device = *x_start.shape, x_start.device
        noise = default(noise, lambda: torch.randn_like(x_start))

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        if is_list_str(cond):
            cond = bert_embed(tokenize(cond), return_cls_repr=self.text_use_bert_cls)
            cond = cond.to(device)
        elif len(cond) == 1 and isinstance(cond[0], torch.Tensor):
            motion_cond_video = cond[0].to(device)

        x_recon = self.denoise_fn(x_noisy, t, cond=[motion_cond_video], **kwargs)

        x0 = self.predict_start_from_noise(x_noisy, t=t, noise=x_recon)

        mse_disp = torch.mean((x_start - x0) ** 2)
        print(f"MSE disp in training: {mse_disp}")

        if self.loss_type == "l1":
            loss = F.l1_loss(noise, x_recon)
        elif self.loss_type == "l2":
            loss = F.mse_loss(noise, x_recon)
        else:
            raise NotImplementedError()

        return loss, unnormalize_divide_by_5(x0), unnormalize_divide_by_5(x_start)

    def forward(self, x, cond=None, *args, **kwargs):
        # cond type: [motion_cond_video]  shape [B, 2, F, H, W]
        (
            b,
            device,
            img_size,
        ) = (
            x.shape[0],
            x.device,
            self.image_size,
        )
        check_shape(x, "b c f h w", c=self.channels, f=self.num_frames, h=img_size, w=img_size)
        t = torch.randint(0, self.num_timesteps, (b,), device=device).long()

        x = normalize_divide_by_5(x)
        motion_cond_video = normalize_divide_by_5(cond[0])
        return self.p_losses(x, t, cond=[motion_cond_video], **kwargs)


# trainer class

def identity(t, *args, **kwargs):
    return t


def normalize_divide_by_5(x: torch.Tensor) -> torch.Tensor:
    return x / 5.0


def unnormalize_divide_by_5(x: torch.Tensor) -> torch.Tensor:
    return x * 5.0


def normalize_cond_img(t):
    return t / 255.0


def unnormalize_img(t):
    return x * 5.0


def cast_num_frames(t, *, frames):
    f = t.shape[1]

    if f == frames:
        return t

    if f > frames:
        return t[:, :frames]

    return F.pad(t, (0, 0, 0, 0, 0, frames - f))


class Dataset(data.Dataset):
    def __init__(
        self,
        input_video_folder,
        image_size,
        motion_condition_video_dir,
        channels=2,
        num_frames=16,
        horizontal_flip=False,
        force_num_frames=True,
        exts=["npy"],
    ):
        super().__init__()
        self.input_video_folder = input_video_folder
        self.motion_condition_video_dir = motion_condition_video_dir
        self.image_size = image_size
        self.channels = channels

        self.video_pairs = []

        input_video_paths = [p for ext in exts for p in Path(f"{input_video_folder}").glob(f"**/*.{ext}")]
        input_video_paths = sorted(input_video_paths)

        motion_condition_video_paths = [
            p for ext in exts for p in Path(f"{motion_condition_video_dir}").glob(f"**/*.{ext}")
        ]
        motion_condition_video_paths = sorted(motion_condition_video_paths)

        assert len(input_video_paths) == len(
            motion_condition_video_paths
        ), "Number of target and motion conditioning videos must match"

        for input_path, motion_condition_path in zip(input_video_paths, motion_condition_video_paths):
            self.video_pairs.append((input_path, motion_condition_path))

        self.cache = {}

    def __len__(self):
        return len(self.video_pairs)

    def __getitem__(self, index):
        input_video_path, motion_condition_video_path = self.video_pairs[index]

        if index not in self.cache:
            input_video_tensor = torch.from_numpy(np.load(input_video_path)).float()
            motion_condition_video_tensor = torch.from_numpy(np.load(motion_condition_video_path)).float()
            self.cache[index] = (input_video_tensor, motion_condition_video_tensor)
        else:
            input_video_tensor, motion_condition_video_tensor = self.cache[index]

        return (
            input_video_tensor.squeeze(0),
            motion_condition_video_tensor,
        )


def random_pick_condition_videos(num_samples, motion_cond_video_dir):
    motion_cond_video_dir = Path(motion_cond_video_dir)
    motion_cond_video_paths = sorted(list(motion_cond_video_dir.glob("*.npy")))

    num_samples = min(num_samples, len(motion_cond_video_paths))
    indices = random.sample(range(len(motion_cond_video_paths)), num_samples)

    motion_cond_tensors = []
    filenames = []

    for idx in indices:
        motion_cond_video_path = motion_cond_video_paths[idx]
        motion_cond = np.load(motion_cond_video_path)
        motion_cond_tensor = torch.from_numpy(motion_cond).float()  # shape [2, F, H, W]
        motion_cond_tensors.append(motion_cond_tensor.unsqueeze(0))  # [1, 2, F, H, W]
        filenames.append(f"{motion_cond_video_path.stem}.npy")

    motion_cond_tensors = torch.cat(motion_cond_tensors, dim=0)

    return motion_cond_tensors, filenames


class Trainer(object):
    def __init__(
        self,
        diffusion_model,
        input_video_folder,
        *,
        motion_condition_video_dir=None,
        sampling_motion_condition_video_dir=None,
        ema_decay=0.995,
        num_frames=16,
        train_batch_size=32,
        train_lr=1e-4,
        train_num_steps=100000,
        gradient_accumulate_every=2,
        amp=False,
        step_start_ema=2000,
        update_ema_every=10,
        save_and_sample_every=1000,
        max_grad_norm=None,
        experiment_name="test-exp",
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.sampling_motion_condition_video_dir = sampling_motion_condition_video_dir

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        image_size = diffusion_model.image_size
        channels = diffusion_model.channels
        num_frames = diffusion_model.num_frames

        self.ds = Dataset(
            input_video_folder,
            image_size,
            motion_condition_video_dir,
            channels=channels,
            num_frames=num_frames,
        )

        print(f"found {len(self.ds)} videos as .npy files at {input_video_folder}")
        assert len(self.ds) > 0, "need to have at least 1 video to start training"

        self.dl = cycle(data.DataLoader(self.ds, batch_size=train_batch_size, shuffle=True, pin_memory=True))
        self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)
        self.max_grad_norm = max_grad_norm

        self.experiment_name = experiment_name

        checkpoints_folder = f"./{self.experiment_name}/checkpoints"
        self.checkpoints_folder = Path(checkpoints_folder)
        self.checkpoints_folder.mkdir(exist_ok=True, parents=True)

        self.reset_parameters()

    def reset_parameters(self):
        self.ema_model.load_state_dict(self.model.state_dict())

    def step_ema(self):
        if self.step < self.step_start_ema:
            self.reset_parameters()
            return
        self.ema.update_model_average(self.ema_model, self.model)

    def save(self, milestone):
        data = {
            "step": self.step,
            "model": self.model.state_dict(),
            "ema": self.ema_model.state_dict(),
            "scaler": self.scaler.state_dict(),
        }
        torch.save(data, str(self.checkpoints_folder / f"model-{milestone}.pt"))

    def load(self, milestone, **kwargs):
        if milestone == -1:
            all_milestones = [int(p.stem.split("-")[-1]) for p in Path(self.checkpoints_folder).glob("**/*.pt")]
            assert (
                len(all_milestones) > 0
            ), "need to have at least one milestone to load from latest checkpoint (milestone == -1)"
            milestone = max(all_milestones)

        data = torch.load(str(self.checkpoints_folder / f"model-{milestone}.pt"))

        print(f"loaded checkpoint: model-{milestone}.pt\n")

        self.step = data["step"]
        self.model.load_state_dict(data["model"], **kwargs)
        self.ema_model.load_state_dict(data["ema"], **kwargs)
        self.scaler.load_state_dict(data["scaler"])

    def train(self, prob_focus_present=0.0, focus_present_mask=None, log_fn=noop):
        assert callable(log_fn)

        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                input_video, motion_cond_video = next(self.dl)
                input_video = input_video.cuda()
                motion_cond_video = motion_cond_video.cuda()

                with autocast(enabled=self.amp):
                    loss, x0, x_start = self.model(
                        input_video,
                        cond=[motion_cond_video],
                        prob_focus_present=prob_focus_present,
                        focus_present_mask=focus_present_mask,
                    )

                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()

                print(f"{self.step}: {loss.item()}")

            log = {"loss": loss.item()}

            if exists(self.max_grad_norm):
                self.scaler.unscale_(self.opt)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)

            self.scaler.step(self.opt)
            self.scaler.update()
            self.opt.zero_grad()

            if self.step % self.update_ema_every == 0:
                self.step_ema()

            if (self.step % 300 == 0) and (self.step != 0):
                predicted_start_training_dir = f"./{self.experiment_name}/predicted_start_training_gifs"
                os.makedirs(predicted_start_training_dir, exist_ok=True)
                x0 = x0.detach().cpu().numpy()
                x_start = x_start.detach().cpu().numpy()

                generate_displacement_quiver_gifs_with_gt(
                    milestone="train",
                    filenames=[f"train_visual_{self.step}"],
                    pred_disp=np.expand_dims(x0[0:1], axis=0),
                    gt_disp=np.expand_dims(x_start[0:1], axis=0),
                    output_dir=predicted_start_training_dir,
                )

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                self.save(milestone)

                sample_motion_cond, sample_cond_filenames = random_pick_condition_videos(
                    num_samples=1,
                    motion_cond_video_dir=self.sampling_motion_condition_video_dir,
                )
                sample_motion_cond = normalize_divide_by_5(sample_motion_cond)

                device = next(self.ema_model.parameters()).device
                sample_motion_cond = sample_motion_cond.to(device)

                self.sample_and_save(
                    milestone,
                    motion_cond_video=sample_motion_cond,
                    cond_filenames=sample_cond_filenames,
                    save_folder=f"./{self.experiment_name}/sampled_videos_infos",
                )

            log_fn(log)
            self.step += 1

        print("training completed")

    @torch.inference_mode()
    def sample_and_save(
        self,
        milestone,
        motion_cond_video=None,
        cond_filenames=None,
        cond_scale=2.0,
        save_folder="./sampled_videos_infos",
    ):
        num_samples = motion_cond_video.shape[0]

        sampled_videos = self.ema_model.sample(cond=[motion_cond_video], cond_scale=cond_scale, batch_size=num_samples)
        sampled_videos = sampled_videos.cpu().numpy()
        sampled_videos = np.expand_dims(sampled_videos, axis=1)

        save_folder = Path(save_folder)
        save_folder.mkdir(exist_ok=True, parents=True)

        sampled_folder = save_folder / "inference_disps"
        sampled_folder.mkdir(exist_ok=True, parents=True)

        for i in range(num_samples):
            if cond_filenames is not None and i < len(cond_filenames):
                base_name = cond_filenames[i]
            else:
                base_name = f"sample_{i:03d}"

            sampled_disp_single = sampled_videos[i]
            sampled_disp_path = sampled_folder / f"{base_name}"
            np.save(sampled_disp_path, sampled_disp_single)

        output_gif_dir = f"{save_folder}/quiver_plots"
        output_dir = Path(output_gif_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        generate_displacement_quiver_gifs(
            milestone,
            sampled_videos,
            cond_filenames,
            output_dir=output_gif_dir,
        )

    @torch.inference_mode()
    def sample_and_save_one_video_at_a_time(
        self,
        milestone,
        motion_cond_video=None,
        cond_filenames=None,
        reconstructed_disp=None,
        gt_disp=None,
        cond_scale=2.0,
        save_folder="./sampled_videos_infos",
    ):
        # motion_cond_video shape: [B, 2, F, H, W]
        num_samples = motion_cond_video.shape[0]

        sampled_videos = self.ema_model.sample(cond=[motion_cond_video], cond_scale=cond_scale, batch_size=num_samples)
        sampled_videos = sampled_videos.cpu().numpy()
        sampled_videos = np.expand_dims(sampled_videos, axis=1)

        save_folder = Path(save_folder)
        save_folder.mkdir(exist_ok=True, parents=True)

        sampled_folder = save_folder / "inference_disps"
        sampled_folder.mkdir(exist_ok=True, parents=True)

        output_gif_dir = f"{save_folder}/quiver_plots"
        output_dir = Path(output_gif_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        for i in range(num_samples):
            if cond_filenames is not None and i < len(cond_filenames):
                base_name = cond_filenames[i]
            else:
                base_name = f"sample_{i:03d}"

            sampled_disp_single = sampled_videos[i]

            sampled_disp_path = sampled_folder / f"{base_name}"
            np.save(sampled_disp_path, sampled_disp_single)

            if reconstructed_disp is None and gt_disp is None:
                generate_displacement_quiver_gifs(milestone, np.expand_dims(sampled_disp_single, axis=0), [base_name], output_dir=output_gif_dir)
            elif reconstructed_disp is None and gt_disp is not None:
                generate_displacement_quiver_gif_comparison(sampled_disp_single, gt_disp, output_path=f"{output_gif_dir}/{cond_filenames[0].split('.npy')[0]}.gif", titles=["Predicted", "Ground Truth"])
            elif reconstructed_disp is not None and gt_disp is None:
                generate_displacement_quiver_gif_comparison(sampled_disp_single, reconstructed_disp, output_path=f"{output_gif_dir}/{cond_filenames[0].split('.npy')[0]}.gif", titles=["Predicted", "Reconstructed"])
            else:
                generate_displacement_quiver_gif_predicted_reconstructed_gt(sampled_disp_single, reconstructed_disp, gt_disp, output_path=f"{output_gif_dir}/{cond_filenames[0].split('.npy')[0]}.gif")
