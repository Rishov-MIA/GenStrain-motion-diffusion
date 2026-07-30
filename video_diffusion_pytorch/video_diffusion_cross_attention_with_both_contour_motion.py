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
# AMP: prefer the new torch.amp API (torch >= 2.3). Fall back to the deprecated
# torch.cuda.amp on older torch so this file stays backward compatible.
try:
    from torch.amp import autocast as _autocast, GradScaler as _GradScaler

    def autocast(enabled=True):
        return _autocast("cuda", enabled=enabled)

    def GradScaler(enabled=True):
        return _GradScaler("cuda", enabled=enabled)
except ImportError:  # torch < 2.3
    from torch.cuda.amp import autocast, GradScaler
from PIL import Image

from tqdm import tqdm
from einops import rearrange
from einops_exts import check_shape, rearrange_many

from rotary_embedding_torch import RotaryEmbedding

try:
    import wandb
except ImportError:  # wandb is optional; training works without it
    wandb = None

from video_diffusion_pytorch.utils import (
    generate_displacement_gridplot_gifs,
    generate_displacement_quiver_gifs,
    save_grayscale_videos,
    generate_mask_disp_side_by_side_gif,
    generate_displacement_quiver_gifs_with_gt,
    generate_displacement_quiver_gif_predicted_reconstructed_gt,
    generate_displacement_quiver_gif_comparison
)
from video_diffusion_pytorch.frame_validity import (
    build_frame_mask,
    load_valid_frames_map,
    resolve_valid_frames,
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
        # self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
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
        # qkv = self.to_qkv(x).chunk(3, dim=1)
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
        # self.to_q = nn.Linear(dim, hidden_dim * 3, bias=False)
        self.to_q = nn.Linear(dim, hidden_dim, bias=False)
        self.to_kv = nn.Linear(dim, hidden_dim * 2, bias=False)
        self.to_out = nn.Linear(hidden_dim, dim, bias=False)

    def forward(self, x, pos_bias=None, focus_present_mask=None):
        n, device = x[0].shape[-2], x[0].device
        q = self.to_q(x[0])
        kv = self.to_kv(x[1]).chunk(2, dim=-1)
        qkv = (q, kv[0], kv[1])

        if exists(focus_present_mask) and focus_present_mask.all():
            # if all batch samples are focusing on present
            # it would be equivalent to passing that token's values through to the output
            values = qkv[-1]
            return self.to_out(values)

        # split out heads
        q, k, v = rearrange_many(qkv, "... n (h d) -> ... h n d", h=self.heads)

        # q, k, v = rearrange(q, '... n (h d) -> ... h n d', h = self.heads), rearrange(k, '... n (h d) -> ... h n d', h = self.heads), rearrange(v, '... n (h d) -> ... h n d', h = self.heads)

        # scale

        q = q * self.scale

        # rotate positions into queries and keys for time attention

        if exists(self.rotary_emb):
            q = self.rotary_emb.rotate_queries_or_keys(q)
            k = self.rotary_emb.rotate_queries_or_keys(k)

        # similarity

        sim = einsum("... h i d, ... h j d -> ... h i j", q, k)

        # relative positional bias

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

        # numerical stability

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        # aggregate values

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

        # temporal attention and its relative positional encoding

        rotary_emb = RotaryEmbedding(min(32, attn_dim_head))

        temporal_attn = lambda dim: EinopsToAndFrom(
            "b c f h w", "b (h w) f c", Attention(dim, heads=attn_heads, dim_head=attn_dim_head, rotary_emb=rotary_emb)
        )

        self.time_rel_pos_bias = RelativePositionBias(
            heads=attn_heads, max_distance=32
        )  # realistically will not be able to generate that many frames of video... yet

        # initial conv

        init_dim = default(init_dim, dim)
        assert is_odd(init_kernel_size)

        init_padding = init_kernel_size // 2
        self.init_conv = nn.Conv3d(
            channels, init_dim, (1, init_kernel_size, init_kernel_size), padding=(0, init_padding, init_padding)
        )

        contour_cond_channels = 1
        self.init_conv_contour_cond = nn.Conv3d(
            contour_cond_channels,
            init_dim,
            (1, init_kernel_size, init_kernel_size),
            padding=(0, init_padding, init_padding),
        )

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

        # choose the block class
        block_klass = ResnetBlock

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])

        # modules for all layers

        for idx, (dim_in, dim_out) in enumerate(in_out):
            is_last = idx >= len(in_out) - 1
            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=cond_dim_total),
                        block_klass(dim_out, dim_out, time_emb_dim=cond_dim_total),
                        Conv1x1(dim_in, dim_out),
                        Conv1x1(dim_out, dim_out),
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
        cond=None,  # cond = [contour_cond_video, motion_cond_video]
        null_cond_prob=0.0,
        focus_present_mask=None,
        prob_focus_present=0.0,  # probability at which a given batch sample will focus on the present (0. is all off, 1. is completely arrested attention across time)
    ):
        assert not (self.has_cond and not exists(cond)), "cond must be passed in if cond_dim specified"
        batch, device = x.shape[0], x.device

        focus_present_mask = default(
            focus_present_mask, lambda: prob_mask_like((batch,), prob_focus_present, device=device)
        )

        time_rel_pos_bias = self.time_rel_pos_bias(x.shape[2], device=x.device)

        x = self.init_conv(x)
        # needed for if self.has_cond:
        # cond_original = cond.clone()

        contour_cond = cond[0]
        motion_cond = cond[1]

        contour_cond = self.init_conv_contour_cond(contour_cond)
        motion_cond = self.init_conv_motion_cond(motion_cond)

        x = self.init_temporal_attn([x, contour_cond], x, pos_bias=time_rel_pos_bias)
        x = self.init_temporal_attn([x, motion_cond], x, pos_bias=time_rel_pos_bias)

        r = x.clone()

        t = self.time_mlp(time) if exists(self.time_mlp) else None

        # classifier free guidance

        if self.has_cond:
            print("dont come here")
            # encoder = ConditionVideoEncoder(in_channels=1) # condition video has 1 channel
            # encoded_cond = encoder(cond_original)
            # batch, device = x.shape[0], x.device
            # mask = prob_mask_like((batch,), null_cond_prob, device = device)
            # encoded_cond = torch.where(rearrange(mask, 'b -> b 1'), self.null_cond_emb, encoded_cond)
            # # time + cond embedding concatenate
            # t = torch.cat((t, encoded_cond), dim = -1)

        h = []
        # count = 0
        for block1, block2, contour_cond_conv_1, contour_cond_conv_2, motion_cond_conv_1, motion_cond_conv_2, spatial_attn, temporal_attn, downsample in self.downs:
            x = block1(x, t)
            x = block2(x, t)

            contour_cond = contour_cond_conv_1(contour_cond)
            contour_cond = contour_cond_conv_2(contour_cond)

            motion_cond = motion_cond_conv_1(motion_cond)
            motion_cond = motion_cond_conv_2(motion_cond)

            x = spatial_attn([x, contour_cond], x)
            x = temporal_attn([x, contour_cond], x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)

            x = spatial_attn([x, motion_cond], x)
            x = temporal_attn([x, motion_cond], x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)

            h.append(x)
            x = downsample(x)

            contour_cond = downsample(contour_cond)
            motion_cond = downsample(motion_cond)

        x = self.mid_block1(x, t)

        x = self.mid_spatial_attn([x, contour_cond], x)
        x = self.mid_spatial_attn([x, motion_cond], x)

        x = self.mid_temporal_attn(
            [x, contour_cond], x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask
        )
        x = self.mid_temporal_attn(
            [x, motion_cond], x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask
        )

        x = self.mid_block2(x, t)

        for block1, block2, contour_cond_conv_1, contour_cond_conv_2, motion_cond_conv_1, motion_cond_conv_2, spatial_attn, temporal_attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)

            x = block1(x, t)
            x = block2(x, t)

            contour_cond = contour_cond_conv_1(contour_cond)
            contour_cond = contour_cond_conv_2(contour_cond)

            motion_cond = motion_cond_conv_1(motion_cond)
            motion_cond = motion_cond_conv_2(motion_cond)

            x = spatial_attn([x, contour_cond], x)
            x = temporal_attn([x, contour_cond], x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)

            x = spatial_attn([x, motion_cond], x)
            x = temporal_attn([x, motion_cond], x, pos_bias=time_rel_pos_bias, focus_present_mask=focus_present_mask)

            x = upsample(x)

            contour_cond = upsample(contour_cond)
            motion_cond = upsample(motion_cond)

        x = torch.cat((x, r), dim=1)
        return self.final_conv(x)


# gaussian diffusion trainer class


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def extract_noise_mask_from_first_frame(mask):
    """
    Extract a spatial noise mask from the FIRST FRAME of the binary mask.

    Since displacement is always relative to the first frame, the noise region
    is determined by frame 0 and broadcast across all frames.

    Handles both 0-255 and 0-1 valued masks by binarizing with threshold 0.5.

    Args:
        mask: tensor of shape [B, 1, F, H, W] with values in [0, 1] or [0, 255]

    Returns:
        noise mask of shape [B, 1, F, H, W] (first frame repeated across all frames)
    """
    f = mask.shape[2]
    first_frame = mask[:, :, 0, :, :]  # [B, 1, H, W]
    # Binarize: handles both 0-255 (threshold > 0.5) and 0-1 ranges
    first_frame = (first_frame > 0.5).float()
    # Broadcast across all frames: [B, 1, F, H, W]
    return first_frame.unsqueeze(2).expand(-1, -1, f, -1, -1)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
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
        use_dynamic_thres=False,  # from the Imagen paper
        dynamic_thres_percentile=0.9,
        global_means=None,
        global_stds=None,
        contour_noise_only=False,
    ):
        super().__init__()
        self.channels = channels
        self.image_size = image_size
        self.num_frames = num_frames
        self.denoise_fn = denoise_fn
        self.contour_noise_only = contour_noise_only

        betas = cosine_beta_schedule(timesteps)

        alphas = 1.0 - betas
        alphas_cumprod = torch.cumprod(alphas, axis=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.0)

        (timesteps,) = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # register buffer helper function that casts float64 to float32

        register_buffer = lambda name, val: self.register_buffer(name, val.to(torch.float32))

        register_buffer("betas", betas)
        register_buffer("alphas_cumprod", alphas_cumprod)
        register_buffer("alphas_cumprod_prev", alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer("sqrt_alphas_cumprod", torch.sqrt(alphas_cumprod))
        register_buffer("sqrt_one_minus_alphas_cumprod", torch.sqrt(1.0 - alphas_cumprod))
        register_buffer("log_one_minus_alphas_cumprod", torch.log(1.0 - alphas_cumprod))
        register_buffer("sqrt_recip_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod))
        register_buffer("sqrt_recipm1_alphas_cumprod", torch.sqrt(1.0 / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1.0 - alphas_cumprod_prev) / (1.0 - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer("posterior_variance", posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer("posterior_log_variance_clipped", torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer("posterior_mean_coef1", betas * torch.sqrt(alphas_cumprod_prev) / (1.0 - alphas_cumprod))
        register_buffer(
            "posterior_mean_coef2", (1.0 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1.0 - alphas_cumprod)
        )

        # ------------ Store global_means, global_stds as buffers ------------
        if global_means is None:
            # By default, zero means (one per channel)
            global_means = torch.zeros(channels, dtype=torch.float32)
        else:
            # Ensure it's a tensor
            global_means = torch.as_tensor(global_means, dtype=torch.float32)

        if global_stds is None:
            # By default, ones for std (one per channel)
            global_stds = torch.ones(channels, dtype=torch.float32)
        else:
            global_stds = torch.as_tensor(global_stds, dtype=torch.float32)

        # Optional: check shape
        assert global_means.shape[0] == channels, f"global_means must have shape [channels], got {global_means.shape}"
        assert global_stds.shape[0] == channels, f"global_stds must have shape [channels], got {global_stds.shape}"

        # Register them
        register_buffer("global_means", global_means)
        register_buffer("global_stds", global_stds)

        # text conditioning parameters

        self.text_use_bert_cls = text_use_bert_cls

        # dynamic thresholding when sampling

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
        # print(f"max: {x_recon.max()}    min: {x_recon.min()}")

        if clip_denoised:
            s = 1.0
            if self.use_dynamic_thres:
                s = torch.quantile(rearrange(x_recon, "b ... -> b (...)").abs(), self.dynamic_thres_percentile, dim=-1)

                s.clamp_(min=1.0)
                s = s.view(-1, *((1,) * (x_recon.ndim - 1)))

            # clip by threshold, depending on whether static or dynamic
            # x_recon = x_recon.clamp(-s, s) / s

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(x_start=x_recon, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance

    @torch.inference_mode()
    def p_sample(self, x, t, cond=None, cond_scale=1.0, clip_denoised=True, noise_mask=None):
        b, *_, device = *x.shape, x.device
        model_mean, _, model_log_variance = self.p_mean_variance(
            x=x, t=t, clip_denoised=clip_denoised, cond=cond, cond_scale=cond_scale
        )
        noise = torch.randn_like(x)

        # Mask stochastic noise to contour region only
        if noise_mask is not None:
            noise = noise * noise_mask

        # no noise when t == 0
        nonzero_mask = (1 - (t == 0).float()).reshape(b, *((1,) * (len(x.shape) - 1)))
        result = model_mean + nonzero_mask * (0.5 * model_log_variance).exp() * noise

        # Zero out non-contour pixels to prevent unconstrained predictions from accumulating
        if noise_mask is not None:
            result = result * noise_mask

        return result

    @torch.inference_mode()
    def p_sample_loop(self, shape, cond=None, cond_scale=1.0):
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        # When contour_noise_only, compute noise mask and apply to initial noise
        noise_mask = None
        if self.contour_noise_only and cond is not None and isinstance(cond[0], torch.Tensor):
            noise_mask = extract_noise_mask_from_first_frame(cond[0])  # [B, 1, F, H, W]
            img = img * noise_mask  # start with noise only in contour region

        for i in tqdm(reversed(range(0, self.num_timesteps)), desc="sampling loop time step", total=self.num_timesteps):
            img = self.p_sample(
                img, torch.full((b,), i, device=device, dtype=torch.long), cond=cond, cond_scale=cond_scale, noise_mask=noise_mask
            )

        return unnormalize_divide_by_5(img)

    def ddim_timestep_pairs(self, ddim_steps):
        """(t, t_prev) pairs for a uniformly spaced DDIM subsequence of the full
        training chain, walked from noisiest to cleanest. The last pair ends at
        -1, which stands for alphas_cumprod_prev = 1 (i.e. fully denoised)."""
        # Guard the silent failure: ddim_steps <= 0 yields an empty pair list, so
        # the sample loop would return the initial noise as if it had denoised it.
        if not 1 <= ddim_steps <= self.num_timesteps:
            raise ValueError(f"ddim_steps must be in [1, {self.num_timesteps}], got {ddim_steps}")

        times = torch.linspace(-1, self.num_timesteps - 1, steps=ddim_steps + 1)
        times = list(reversed(times.long().tolist()))
        return list(zip(times[:-1], times[1:]))

    @torch.inference_mode()
    def ddim_sample_loop(self, shape, cond=None, cond_scale=1.0, ddim_steps=50, ddim_eta=0.0):
        """DDIM sampler over a subsequence of the training chain. Uses the same
        trained eps-prediction network as `p_sample_loop` with no retraining,
        for `num_timesteps / ddim_steps` times fewer denoise_fn calls.
        `ddim_eta=0` is the deterministic ODE path; `ddim_eta=1` recovers
        DDPM-like posterior noise."""
        device = self.betas.device

        b = shape[0]
        img = torch.randn(shape, device=device)

        # When contour_noise_only, compute noise mask and apply to initial noise
        noise_mask = None
        if self.contour_noise_only and cond is not None and isinstance(cond[0], torch.Tensor):
            noise_mask = extract_noise_mask_from_first_frame(cond[0])  # [B, 1, F, H, W]
            img = img * noise_mask  # start with noise only in contour region

        time_pairs = self.ddim_timestep_pairs(ddim_steps)

        for time, time_prev in tqdm(time_pairs, desc="ddim sampling loop time step", total=len(time_pairs)):
            t = torch.full((b,), time, device=device, dtype=torch.long)
            pred_noise = self.denoise_fn.forward_with_cond_scale(img, t, cond=cond, cond_scale=cond_scale)

            # No thresholding here, to stay comparable with p_mean_variance,
            # whose clamp is commented out.
            x_start = self.predict_start_from_noise(img, t=t, noise=pred_noise)

            alpha = self.alphas_cumprod[time]
            alpha_prev = self.alphas_cumprod[time_prev] if time_prev >= 0 else torch.ones_like(alpha)

            sigma = ddim_eta * ((1 - alpha_prev) / (1 - alpha)).sqrt() * (1 - alpha / alpha_prev).sqrt()
            dir_coef = (1 - alpha_prev - sigma**2).clamp(min=0.0).sqrt()

            img = alpha_prev.sqrt() * x_start + dir_coef * pred_noise

            # no noise on the final step (time_prev == -1), matching p_sample
            if ddim_eta > 0 and time_prev >= 0:
                noise = torch.randn_like(img)
                # Mask stochastic noise to contour region only
                if noise_mask is not None:
                    noise = noise * noise_mask
                img = img + sigma * noise

            # Zero out non-contour pixels to prevent unconstrained predictions from accumulating
            if noise_mask is not None:
                img = img * noise_mask

        return unnormalize_divide_by_5(img)

    @torch.inference_mode()
    def sample(self, cond=None, cond_scale=1.0, batch_size=16, sampler="ddpm", ddim_steps=50, ddim_eta=0.0):
        device = next(self.denoise_fn.parameters()).device

        if is_list_str(cond):
            cond = bert_embed(tokenize(cond)).to(device)
        elif len(cond) == 2 and isinstance(cond[0], torch.Tensor) and isinstance(cond[1], torch.Tensor):  # video
            contour_cond_video = cond[0].to(device)
            motion_cond_video = cond[1].to(device)

        batch_size = contour_cond_video.shape[0]
        image_size = self.image_size
        channels = self.channels
        num_frames = self.num_frames
        shape = (batch_size, channels, num_frames, image_size, image_size)
        cond = [contour_cond_video, motion_cond_video]

        if sampler == "ddpm":
            return self.p_sample_loop(shape, cond=cond, cond_scale=cond_scale)
        if sampler == "ddim":
            return self.ddim_sample_loop(
                shape, cond=cond, cond_scale=cond_scale, ddim_steps=ddim_steps, ddim_eta=ddim_eta
            )
        raise ValueError(f"unknown sampler {sampler!r}, expected 'ddpm' or 'ddim'")

    @torch.inference_mode()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)), desc="interpolation sample time step", total=t):
            img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None, noise_mask=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        if noise_mask is not None:
            noise = noise * noise_mask

        return (
            extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start
            + extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) * noise
        )

    def p_losses(self, x_start, t, cond=None, noise=None, valid_frames=None, **kwargs):
        # cond: [contour_cond_video, motion_cond_video]
        b, c, f, h, w, device = *x_start.shape, x_start.device
        noise = default(noise, lambda: torch.randn_like(x_start))

        # Optional per-sample valid-frame mask [B, 1, F, 1, 1]: 1 on the leading
        # valid frames, 0 on padding. None (no CSV) => original, unmasked loss.
        frame_mask = None
        if valid_frames is not None:
            frame_mask = build_frame_mask(valid_frames, f, device=device, dtype=x_start.dtype)

        # When contour_noise_only is enabled, mask noise using first frame of contour mask
        noise_mask = None
        if self.contour_noise_only and cond is not None and isinstance(cond[0], torch.Tensor):
            noise_mask = extract_noise_mask_from_first_frame(cond[0])  # [B, 1, F, H, W]
            noise = noise * noise_mask  # broadcasts to [B, 2, F, H, W]

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)

        # Zero out non-contour pixels in x_noisy so the model always sees zeros
        # outside the contour — matching what it sees during inference
        if noise_mask is not None:
            x_noisy = x_noisy * noise_mask

        # add encoder here
        if is_list_str(cond):
            cond = bert_embed(tokenize(cond), return_cls_repr=self.text_use_bert_cls)  # (3, 768) => (B, embedding_size)
            cond = cond.to(device)
        elif len(cond) == 2 and isinstance(cond[0], torch.Tensor) and isinstance(cond[1], torch.Tensor):  # video
            contour_cond_video = cond[0].to(device)
            motion_cond_video = cond[1].to(device)

        x_recon = self.denoise_fn(x_noisy, t, cond=[contour_cond_video, motion_cond_video], **kwargs)

        # epsilon given. extract x0
        # forward x0 print from predicted noise x_recon
        x0 = self.predict_start_from_noise(x_noisy, t=t, noise=x_recon)

        # Reconstruction MSE between the ground-truth displacement field (x_start)
        # and the model's predicted x0, in normalized space. This measures the
        # actual displacement-prediction quality, distinct from `loss` (the
        # noise-prediction objective the model is trained on).
        if frame_mask is None:
            disp_recon_mse = torch.mean((x_start - x0) ** 2)
        else:
            disp_recon_mse = ((x_start - x0) ** 2 * frame_mask).sum() / frame_mask.expand_as(x_start).sum().clamp(min=1.0)

        if self.contour_noise_only and noise_mask is not None:
            # Compute loss only in the contour region
            if frame_mask is None:
                masked_noise = noise * noise_mask
                masked_recon = x_recon * noise_mask
                num_contour_pixels = noise_mask.sum().clamp(min=1.0)
                if self.loss_type == "l1":
                    loss = (masked_noise - masked_recon).abs().sum() / num_contour_pixels
                elif self.loss_type == "l2":
                    loss = ((masked_noise - masked_recon) ** 2).sum() / num_contour_pixels
                else:
                    raise NotImplementedError()
            else:
                # ...and, when given, only inside the valid frames
                region = noise_mask * frame_mask
                num_contour_pixels = region.sum().clamp(min=1.0)
                diff = noise - x_recon
                if self.loss_type == "l1":
                    loss = (diff.abs() * region).sum() / num_contour_pixels
                elif self.loss_type == "l2":
                    loss = ((diff ** 2) * region).sum() / num_contour_pixels
                else:
                    raise NotImplementedError()
        else:
            if frame_mask is None:
                if self.loss_type == "l1":
                    loss = F.l1_loss(noise, x_recon)
                elif self.loss_type == "l2":
                    loss = F.mse_loss(noise, x_recon)
                else:
                    raise NotImplementedError()
            else:
                # Full-region loss, averaged over valid frames only
                diff = noise - x_recon
                if self.loss_type == "l1":
                    per_element = diff.abs()
                elif self.loss_type == "l2":
                    per_element = diff ** 2
                else:
                    raise NotImplementedError()
                denom = frame_mask.expand_as(per_element).sum().clamp(min=1.0)
                loss = (per_element * frame_mask).sum() / denom

        return loss, unnormalize_divide_by_5(x0), unnormalize_divide_by_5(x_start), disp_recon_mse
        
    def forward(self, x, cond=None, *args, valid_frames=None, **kwargs):
        # cond type: [contour_cond_video, motion_cond_video]
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
        contour_cond_video = normalize_cond_img(cond[0])
        motion_cond_video = normalize_divide_by_5(cond[1])
        return self.p_losses(x, t, cond=[contour_cond_video, motion_cond_video], valid_frames=valid_frames, **kwargs)
    
    
    @torch.inference_mode()
    def forward_diffuse_single(
        self,
        x_start: torch.Tensor,
        t: int | torch.Tensor,
        *,
        noise: torch.Tensor | None = None,
        normalize_input: bool = True,
        return_noise: bool = True
    ):
        """
        Run the forward diffusion q(x_t | x_0) for a single sample at timestep t.

        Args:
            x_start: Tensor of shape (C, F, H, W) or (1, C, F, H, W) in your *unnormalized* scale
                    (i.e., pre normalize_divide_by_5). If normalize_input=False, it is assumed
                    already normalized like in `forward()`.
            t:       Integer timestep in [0, self.num_timesteps-1] or a 0/1-D tensor.
            noise:   Optional noise tensor of same shape as x_start (after adding batch dim).
                    If None, uses standard Gaussian.
            normalize_input: If True, applies `normalize_divide_by_5` to x_start before diffusion.
            return_noise: If True, also returns the noise actually used.

        Returns:
            x_t:  Tensor of shape (1, C, F, H, W)
            noise (optional): The exact noise used, same shape as x_t
        """
        # ensure batch dimension
        if x_start.dim() == 4:
            x_start = x_start.unsqueeze(0)  # (1, C, F, H, W)
        elif x_start.dim() != 5 or x_start.shape[0] != 1:
            raise ValueError("x_start must be (C, F, H, W) or (1, C, F, H, W) for a single sample")

        device = self.betas.device
        x_start = x_start.to(device)

        # match normalization used in training
        if normalize_input:
            x_start = normalize_divide_by_5(x_start)

        # timestep as a (1,) long tensor on correct device
        if isinstance(t, int):
            t = torch.tensor([t], device=device, dtype=torch.long)
        else:
            t = t.to(device).long().reshape(1)

        # noise
        if noise is None:
            noise = torch.randn_like(x_start)
        else:
            noise = noise.to(device)
            if noise.shape != x_start.shape:
                raise ValueError(f"noise must have shape {x_start.shape}, got {noise.shape}")

        # q(x_t | x_0)
        x_t = self.q_sample(x_start=x_start, t=t, noise=noise)

        # return in normalized space (like q_sample outputs).
        # If you prefer to get it back in the original scale, uncomment:
        # x_t = unnormalize_divide_by_5(x_t)

        return (x_t, noise) if return_noise else x_t



# trainer class

def identity(t, *args, **kwargs):
    return t


def normalize_divide_by_5(x: torch.Tensor) -> torch.Tensor:
    return x / 5.0


def unnormalize_divide_by_5(x: torch.Tensor) -> torch.Tensor:
    return x * 5.0


def normalize_channelwise(x: torch.Tensor) -> torch.Tensor:
    """
    Normalize a 5D tensor (b, c, f, h, w) channel-wise into [-1, 1].
    """
    x = x.float()
    mins = x.amin(dim=(0, 2, 3, 4), keepdim=True)
    maxs = x.amax(dim=(0, 2, 3, 4), keepdim=True)
    eps = 1e-8
    denom = (maxs - mins).clamp(min=eps)
    x_normalized = 2.0 * (x - mins) / denom - 1.0
    return x_normalized


def normalize_cond_img(t):
    return t / 255.0


def unnormalize_img(t):
    return t * 5.0


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
        contour_condition_video_dir,
        motion_condition_video_dir,
        channels=2,  # input video has 2 channels
        num_frames=16,
        horizontal_flip=False,
        force_num_frames=True,
        exts=["npy"],  # both video as .npy format
        valid_frames_map=None,       # {filename: valid_frame_count}; empty => all frames valid
        valid_frames_missing="full",  # behavior when a file is absent from the map
    ):
        super().__init__()
        self.input_video_folder = input_video_folder
        self.contour_condition_video_dir = contour_condition_video_dir
        self.motion_condition_video_dir = motion_condition_video_dir
        self.image_size = image_size
        self.channels = channels
        self.num_frames = num_frames
        self.valid_frames_map = valid_frames_map or {}
        self.valid_frames_missing = valid_frames_missing

        self.video_pairs = []

        input_video_paths = [p for ext in exts for p in Path(f"{input_video_folder}").glob(f"**/*.{ext}")]
        input_video_paths = sorted(input_video_paths)

        contour_condition_video_paths = [
            p for ext in exts for p in Path(f"{contour_condition_video_dir}").glob(f"**/*.{ext}")
        ]
        contour_condition_video_paths = sorted(contour_condition_video_paths)

        motion_condition_video_paths = [
            p for ext in exts for p in Path(f"{motion_condition_video_dir}").glob(f"**/*.{ext}")
        ]
        motion_condition_video_paths = sorted(motion_condition_video_paths)

        assert len(input_video_paths) == len(
            contour_condition_video_paths
        ), "Number of target and contour conditioning videos must match"
        assert len(input_video_paths) == len(
            motion_condition_video_paths
        ), "Number of target and motion conditioning videos must match"

        for input_path, contour_condition_path, motion_condition_path in zip(
            input_video_paths, contour_condition_video_paths, motion_condition_video_paths
        ):
            self.video_pairs.append((input_path, contour_condition_path, motion_condition_path))
        
        # Cache initialization
        self.cache = {}


    def __len__(self):
        return len(self.video_pairs)

    def __getitem__(self, index):
        input_video_path, contour_condition_video_path, motion_condition_video_path = self.video_pairs[index]

        # Lazy loading and caching: Load the video data only once, then cache it for future use
        if index not in self.cache:
            # Load input video data
            input_video_tensor = torch.from_numpy(np.load(input_video_path)).float()
            contour_condition_video_tensor = torch.from_numpy(np.load(contour_condition_video_path)).float()
            motion_condition_video_tensor = torch.from_numpy(np.load(motion_condition_video_path)).float()

            # Cache the data
            self.cache[index] = (input_video_tensor, contour_condition_video_tensor, motion_condition_video_tensor)
        else:
            # Retrieve from cache
            input_video_tensor, contour_condition_video_tensor, motion_condition_video_tensor = self.cache[index]

        # Per-sample valid-frame count (leading-N valid); num_frames when the
        # file isn't in the map or no CSV was given => an all-valid mask downstream.
        valid_frames = resolve_valid_frames(
            self.valid_frames_map, Path(input_video_path).name, self.num_frames, self.valid_frames_missing
        )

        # Optionally, you can apply any transformations here (like resizing, normalization, etc.)
        # For now, return the tensors directly
        return (
            input_video_tensor.squeeze(0),  # Remove extra dimension
            contour_condition_video_tensor,
            motion_condition_video_tensor.squeeze(0),  # Remove extra dimension
            valid_frames,
        )

def random_pick_condition_videos(num_samples, contour_cond_video_dir, motion_cond_video_dir):
    # Convert string to a Path object
    contour_cond_video_dir = Path(contour_cond_video_dir)
    motion_cond_video_dir = Path(motion_cond_video_dir)

    contour_cond_video_paths = sorted(list(contour_cond_video_dir.glob("*.npy")))
    motion_cond_video_paths = sorted(list(motion_cond_video_dir.glob("*.npy")))

    # Select random indices
    num_samples = min(num_samples, len(contour_cond_video_paths))
    indices = random.sample(range(len(contour_cond_video_paths)), num_samples)

    contour_cond_tensors = []
    motion_cond_tensors = []
    filenames = []

    for idx in indices:
        contour_cond_video_path = contour_cond_video_paths[idx]
        motion_cond_video_path = motion_cond_video_paths[idx]

        contour_cond = np.load(contour_cond_video_path)
        motion_cond = np.load(motion_cond_video_path)

        contour_cond_tensor = torch.from_numpy(contour_cond).float()  # shape [1, F, H, W]
        motion_cond_tensor = torch.from_numpy(motion_cond).float()  # shape [1, 2, F, H, W]

        contour_cond_tensors.append(contour_cond_tensor.unsqueeze(0))  # add batch dim => [batch, 1, F, H, W]
        motion_cond_tensors.append(motion_cond_tensor)  # add batch dim => [batch, 2, F, H, W]

        filenames.append(f"{contour_cond_video_path.stem}.npy")  # store the base filename, e.g. "my_video"

    contour_cond_tensors = torch.cat(contour_cond_tensors, dim=0)
    motion_cond_tensors = torch.cat(motion_cond_tensors, dim=0)

    return contour_cond_tensors, motion_cond_tensors, filenames


class Trainer(object):
    # Sampler for the periodic preview samples taken every `save_and_sample_every`
    # training steps. These are CLASS attributes rather than set in __init__ so
    # that every Trainer subclass has them — including the DDP trainers, which
    # deliberately skip super().__init__(). Named preview_* because `self.sampler`
    # is already the DDP DistributedSampler.
    preview_sampler = "ddpm"
    preview_ddim_steps = 50
    preview_ddim_eta = 0.0

    def configure_preview_sampler(self, sampler="ddpm", ddim_steps=50, ddim_eta=0.0):
        """Choose the sampler used for training-time preview samples. Validates at
        startup rather than letting a bad value fail at the first sampling step,
        thousands of steps into a run."""
        if sampler not in ("ddpm", "ddim"):
            raise ValueError(f"unknown preview sampler {sampler!r}, expected 'ddpm' or 'ddim'")
        num_timesteps = getattr(self.model, "num_timesteps", None)
        if sampler == "ddim" and num_timesteps is not None and not 1 <= ddim_steps <= num_timesteps:
            raise ValueError(f"preview ddim_steps must be in [1, {num_timesteps}], got {ddim_steps}")
        self.preview_sampler = sampler
        self.preview_ddim_steps = ddim_steps
        self.preview_ddim_eta = ddim_eta

    def __init__(
        self,
        diffusion_model,
        input_video_folder,
        *,
        contour_condition_video_dir=None,  # folder with condition dense videos for training
        motion_condition_video_dir=None,  # folder with motion condition dense videos for training
        sampling_contour_condition_video_dir=None,  # folder with condition cine videos for sampling
        sampling_motion_condition_video_dir=None,  # folder with motion condition cine videos for sampling
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
        inference_only=False,
        use_wandb=False,
        wandb_project="genstrain-motion-diffusion",
        wandb_config=None,
        valid_frames_csv=None,
        valid_frames_filename_col="dense_filename",
        valid_frames_count_col="dense_valid_frames",
        valid_frames_missing="full",
    ):
        super().__init__()
        self.model = diffusion_model
        self.ema = EMA(ema_decay)
        self.ema_model = copy.deepcopy(self.model)
        self.update_ema_every = update_ema_every
        self.sampling_contour_condition_video_dir = sampling_contour_condition_video_dir
        self.sampling_motion_condition_video_dir = sampling_motion_condition_video_dir
        # self.sampling_condition_video_paths = [p for p in Path(f"{sampling_contour_condition_video_dir}").glob(f"**/*.npy")]

        self.step_start_ema = step_start_ema
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.image_size = diffusion_model.image_size
        self.gradient_accumulate_every = gradient_accumulate_every
        self.train_num_steps = train_num_steps

        image_size = diffusion_model.image_size
        channels = diffusion_model.channels
        num_frames = diffusion_model.num_frames

        # Optional per-sample valid-frame masking. Empty map (no CSV) => masking
        # off, i.e. the loss is byte-for-byte the original all-frames loss.
        self.valid_frames_map = load_valid_frames_map(
            valid_frames_csv, valid_frames_filename_col, valid_frames_count_col
        )
        self.use_frame_validity = bool(self.valid_frames_map)

        if not inference_only:
            self.ds = Dataset(
                input_video_folder,
                image_size,
                contour_condition_video_dir,
                motion_condition_video_dir,
                channels=channels,
                num_frames=num_frames,
                valid_frames_map=self.valid_frames_map,
                valid_frames_missing=valid_frames_missing,
            )

            if self.use_frame_validity:
                print(f"valid-frame loss masking ON ({len(self.valid_frames_map)} CSV entries)")

            print(f"found {len(self.ds)} videos as .npy files at {input_video_folder}")
            assert len(self.ds) > 0, "need to have at least 1 video to start training (although 1 is not great, try 100k)"

            self.dl = cycle(data.DataLoader(self.ds, batch_size=train_batch_size, shuffle=True, pin_memory=True))
            self.opt = Adam(diffusion_model.parameters(), lr=train_lr)

        self.step = 0

        self.amp = amp
        self.scaler = GradScaler(enabled=amp)
        self.max_grad_norm = max_grad_norm

        self.experiment_name = experiment_name

        # Optional Weights & Biases logging. Off by default so existing runs are
        # unaffected; enable via the config (see configs/README.md).
        self.use_wandb = use_wandb and not inference_only
        if self.use_wandb:
            assert wandb is not None, "use_wandb=True but wandb is not installed (pip install wandb)"
            wandb.init(
                project=wandb_project,
                name=experiment_name,
                config=wandb_config,
                resume="allow",
            )
            # Plot the epoch metrics against epoch number (not step): each gets its
            # own panel with `epoch` as the x-axis. Step-level metrics (loss,
            # loss_cummean, disp_recon_mse) keep the default global-step axis.
            wandb.define_metric("epoch")
            wandb.define_metric("epoch_loss", step_metric="epoch")
            wandb.define_metric("epoch_disp_recon_mse", step_metric="epoch")

        # True cumulative mean-since-start of the training loss: running sum +
        # count, so cumulative_mean = sum / count. Not checkpointed (restarts from
        # zero on resume).
        self._loss_sum = 0.0
        self._loss_count = 0

        # ---- epoch loss bookkeeping ----
        # One epoch = the model has seen the whole dataset once. Every optimizer
        # step consumes batch_size * gradient_accumulate_every samples, so:
        if not inference_only:
            samples_per_step = self.batch_size * self.gradient_accumulate_every
            self.steps_per_epoch = max(1, len(self.ds) // samples_per_step)
        else:
            self.steps_per_epoch = None
        # Running sums of the per-step loss and displacement-reconstruction MSE
        # within the current epoch, plus how many steps have contributed. Reset at
        # each epoch boundary. (Both share _epoch_step_count as the denominator.)
        self._epoch_loss_sum = 0.0
        self._epoch_disp_recon_mse_sum = 0.0
        self._epoch_step_count = 0

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

        # torch >= 2.6 flipped torch.load's default to weights_only=True, which
        # rejects our checkpoint dicts; pass weights_only=False there. Older torch
        # (< 2.0) doesn't accept that kwarg, so fall back to the plain call.
        ckpt_path = str(self.checkpoints_folder / f"model-{milestone}.pt")
        try:
            data = torch.load(ckpt_path, weights_only=False)
        except TypeError:
            data = torch.load(ckpt_path)

        print(f"loaded checkpoint: model-{milestone}.pt\n")

        self.step = data["step"]
        self.model.load_state_dict(data["model"], **kwargs)
        self.ema_model.load_state_dict(data["ema"], **kwargs)
        self.scaler.load_state_dict(data["scaler"])

    def train(self, prob_focus_present=0.0, focus_present_mask=None, log_fn=noop):
        assert callable(log_fn)

        # Horizontal progress bar for the current epoch. It fills as steps within
        # the epoch complete and resets at each epoch boundary. The bar's fill and
        # the epoch-loss average share one counter (self._epoch_step_count), so they
        # always reset together. On a resume both start at 0 and the bar restarts
        # the current epoch from the beginning (the loss of earlier steps in that
        # epoch is unrecoverable anyway); the epoch NUMBER still reflects the
        # restored step so labels stay correct.
        current_epoch = self.step // self.steps_per_epoch
        self._epoch_loss_sum = 0.0
        self._epoch_disp_recon_mse_sum = 0.0
        self._epoch_step_count = 0
        epoch_bar = tqdm(
            total=self.steps_per_epoch,
            desc=f"epoch {current_epoch}",
            unit="step",
            leave=True,
            dynamic_ncols=True,
        )

        while self.step < self.train_num_steps:
            for i in range(self.gradient_accumulate_every):
                input_video, contour_cond_video, motion_cond_video, valid_frames = next(self.dl)
                input_video = input_video.cuda()
                contour_cond_video = contour_cond_video.cuda()
                motion_cond_video = motion_cond_video.cuda()
                valid_frames = valid_frames.cuda() if self.use_frame_validity else None

                with autocast(enabled=self.amp):
                    loss, x0, x_start, disp_recon_mse = self.model(
                        input_video,
                        cond=[contour_cond_video, motion_cond_video],
                        valid_frames=valid_frames,
                        prob_focus_present=prob_focus_present,
                        focus_present_mask=focus_present_mask,
                    )

                    self.scaler.scale(loss / self.gradient_accumulate_every).backward()

                # Route through the bar's writer so per-step loss lines scroll above
                # the pinned epoch progress bar instead of fighting it.
                epoch_bar.write(
                    f"{self.step}: {loss.item()} | disp_recon_mse: {disp_recon_mse.item()}"
                )

            # Accumulate the true cumulative mean of the loss since the start.
            self._loss_sum += loss.item()
            self._loss_count += 1

            log = {
                "loss": loss.item(),
                "loss_cummean": self._loss_sum / self._loss_count,
                "disp_recon_mse": disp_recon_mse.item(),
                "step": self.step,
            }

            # Accumulate the epoch loss / disp_recon_mse and advance the epoch bar.
            # When the current step completes a full pass over the dataset, emit the
            # means, then reset the bar and counters for the next epoch.
            self._epoch_loss_sum += loss.item()
            self._epoch_disp_recon_mse_sum += disp_recon_mse.item()
            self._epoch_step_count += 1
            epoch_bar.update(1)
            epoch_bar.set_postfix(loss=loss.item())
            if self._epoch_step_count >= self.steps_per_epoch:
                epoch_loss = self._epoch_loss_sum / self._epoch_step_count
                epoch_disp_recon_mse = self._epoch_disp_recon_mse_sum / self._epoch_step_count
                epoch = (self.step + 1) // self.steps_per_epoch
                epoch_bar.write(
                    f"  epoch {epoch} | epoch_loss {epoch_loss} | "
                    f"epoch_disp_recon_mse {epoch_disp_recon_mse}"
                )
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

            # Save recontructed x0 every 300 steps (customize as needed)
            # if (self.step % 300 == 0) and (self.step != 0):
            #     predicted_start_training_dir = f"./{self.experiment_name}/predicted_start_training_gifs"
            #     os.makedirs(predicted_start_training_dir, exist_ok=True)
            #     x0 = x0.detach().cpu().numpy()
            #     x_start = x_start.detach().cpu().numpy()

            #     generate_displacement_quiver_gifs_with_gt(
            #         milestone="train",
            #         filenames=[f"train_visual_{self.step}"],
            #         pred_disp=np.expand_dims(x0[0:1], axis=0),
            #         gt_disp=np.expand_dims(x_start[0:1], axis=0),
            #         output_dir=predicted_start_training_dir,
            #     )

            if self.step != 0 and self.step % self.save_and_sample_every == 0:
                milestone = self.step // self.save_and_sample_every
                self.save(milestone)

                sample_contour_cond, sample_motion_cond, sample_cond_filenames = random_pick_condition_videos(
                    num_samples=1,
                    contour_cond_video_dir=self.sampling_contour_condition_video_dir,
                    motion_cond_video_dir=self.sampling_motion_condition_video_dir,
                )
                # normalize cine contour condition image here
                sample_contour_cond = normalize_cond_img(sample_contour_cond)
                sample_motion_cond = normalize_divide_by_5(sample_motion_cond)

                # If your training set is on CPU, move to GPU:
                device = next(self.ema_model.parameters()).device
                sample_contour_cond = sample_contour_cond.to(device)
                sample_motion_cond = sample_motion_cond.to(device)

                # sample a grid
                self.sample_and_save(
                    milestone,
                    contour_cond_video=sample_contour_cond,
                    motion_cond_video=sample_motion_cond,
                    cond_filenames=sample_cond_filenames,
                    save_folder=f"./{self.experiment_name}/sampled_videos_infos",
                    sampler=self.preview_sampler,
                    ddim_steps=self.preview_ddim_steps,
                    ddim_eta=self.preview_ddim_eta,
                )

            if self.use_wandb:
                wandb.log(log, step=self.step)

            log_fn(log)
            self.step += 1

        epoch_bar.close()

        if self.use_wandb:
            wandb.finish()

        print("training completed")

    @torch.inference_mode()
    def sample_and_save(
        self,
        milestone,
        contour_cond_video=None,
        motion_cond_video=None,
        cond_filenames=None,
        cond_scale=2.0,
        save_folder="./sampled_videos_infos",
        sampler="ddpm",
        ddim_steps=50,
        ddim_eta=0.0,
    ):
        # Number of samples to generate
        # cond video shape [num_samples, 1, F, H, W]
        num_samples = contour_cond_video.shape[0]

        sampled_videos = self.ema_model.sample(
            cond=[contour_cond_video, motion_cond_video],
            cond_scale=cond_scale,
            batch_size=num_samples,
            sampler=sampler,
            ddim_steps=ddim_steps,
            ddim_eta=ddim_eta,
        )
        sampled_videos = sampled_videos.cpu().numpy() # [num_samples, 2, F, H, W]
        sampled_videos = np.expand_dims(sampled_videos, axis=1) # [num_samples, 1, 2, F, H, W]
        # to unnormalize it multiply with 255.0
        contour_cond_video = contour_cond_video.cpu().numpy() * 255.0

        save_folder = Path(save_folder)
        save_folder.mkdir(exist_ok=True, parents=True)

        # Create subdirectories for ground truths and sampled videos
        sampled_folder = save_folder / "inference_disps"
        sampled_folder.mkdir(exist_ok=True, parents=True)


        # Instead of saving one big file, loop over each sample
        for i in range(num_samples):
            # If cond_filenames is None or too short, just name it generically
            if cond_filenames is not None and i < len(cond_filenames):
                base_name = cond_filenames[i]
            else:
                base_name = f"sample_{i:03d}"

            sampled_disp_single = sampled_videos[i]  # shape [1, 2, F, H, W]

            # Build the output filenames
            sampled_disp_path = sampled_folder / f"{base_name}"
            np.save(sampled_disp_path, sampled_disp_single)

        output_gif_dir = f"{save_folder}/quiver_plots"
        # Create output directory if needed
        output_dir = Path(output_gif_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        generate_mask_disp_side_by_side_gif(
            contour_cond_video[0],
            sampled_videos[0],
            output_path=f"{output_gif_dir}/milestone_{milestone}_{cond_filenames[0].split('.npy')[0]}.gif",
        )
        
    @torch.inference_mode()
    def sample_and_save_full_video(
        self,
        milestone,
        contour_cond_video=None,
        cond_filename=None,
        cond_scale=2.0,
        save_folder="./sampled_videos_infos",
        sampler="ddpm",
        ddim_steps=50,
        ddim_eta=0.0,
    ):
        save_folder = Path(save_folder)
        save_folder.mkdir(exist_ok=True, parents=True)

        # Create subdirectories for ground truths and sampled videos
        gt_folder = save_folder / "gt_npys"
        sampled_folder = save_folder / "sampled_npys"

        gt_folder.mkdir(exist_ok=True, parents=True)
        sampled_folder.mkdir(exist_ok=True, parents=True)

        # Number of samples to generate
        # cond video shape [num_samples, 1, F, H, W] frames greater than 20
        # num_samples = contour_cond_video.shape[0]

        # for i in range(num_samples):
        single_full_video = contour_cond_video  # [1, F, H, W]
        device = single_full_video.device
        frames_count = single_full_video.shape[1]

        # Resize each frame: move to CPU → NumPy → resize → back to GPU tensor
        resized_frames = [
            torch.from_numpy(resize_to_64x64(single_full_video[0, j].detach().cpu().numpy()))  # (64, 64)
            .to(torch.float32)
            .to(device)  # still (64, 64)
            for j in range(frames_count)
        ]

        # Stack into [F, 64, 64], then add batch dimension → [1, F, 64, 64]
        resized_video = torch.stack(resized_frames, dim=0)  # [F, 64, 64]
        single_full_video = resized_video.unsqueeze(0)  # [1, F, 64, 64]

        # Get indices for splitting the video into parts
        parts_indices = get_dense_video_part_indices(frames_count)

        # Extract parts from the full video
        video_parts = []
        for part_indices in parts_indices:
            video_part = single_full_video[:, part_indices, :, :]  # [1, part_len, 64, 64]
            video_parts.append(video_part)

        # Stack parts → [num_parts, 1, part_len, 64, 64]
        concatenated_video_parts = torch.stack(video_parts, dim=0)

        # Sample from the model
        sampled_videos = self.ema_model.sample(
            cond=concatenated_video_parts,
            cond_scale=cond_scale,
            batch_size=concatenated_video_parts.shape[0],
            sampler=sampler,
            ddim_steps=ddim_steps,
            ddim_eta=ddim_eta,
        )

        # Merge parts back into full-length videos
        merged_sampled_video, merged_cond_video = reconstruct_full_videos(
            parts_indices=parts_indices,
            sampled_videos=sampled_videos,
            concatenated_video_parts=concatenated_video_parts,
        )

        # Move to CPU for saving or further processing
        merged_cond_video = merged_cond_video * 255.0  # Unnormalize

        # save file
        base_name = cond_filename[0]

        # Build the output filenames
        gt_outpath = gt_folder / f"{base_name}.npy"
        sampled_disp_path = sampled_folder / f"{base_name}.npy"

        np.save(gt_outpath, merged_cond_video)
        np.save(sampled_disp_path, merged_sampled_video)

        # generate_displacement_gridplot_gifs(milestone, sampled_videos, output_dir=f"{save_folder}/grid_plots")
        generate_displacement_quiver_gifs(
            milestone,
            np.expand_dims(merged_sampled_video, axis=0),
            cond_filename,
            output_dir=f"{save_folder}/quiver_plots",
        )
        save_grayscale_videos(
            milestone,
            np.expand_dims(merged_cond_video, axis=0),
            cond_filename,
            output_dir=f"{save_folder}/gt_cine_videos",
        )


    @torch.inference_mode()
    def sample_and_save_one_video_at_a_time(
        self,
        milestone,
        contour_cond_video=None,
        motion_cond_video=None,
        cond_filenames=None,
        reconstructed_disp=None,
        gt_disp=None,
        cond_scale=2.0,
        save_folder="./sampled_videos_infos",
        sampler="ddpm",
        ddim_steps=50,
        ddim_eta=0.0,
    ):
        # Number of samples to generate
        # cond video shape [num_samples, 1, F, H, W]
        num_samples = contour_cond_video.shape[0]

        sampled_videos = self.ema_model.sample(
            cond=[contour_cond_video, motion_cond_video],
            cond_scale=cond_scale,
            batch_size=num_samples,
            sampler=sampler,
            ddim_steps=ddim_steps,
            ddim_eta=ddim_eta,
        )
        sampled_videos = sampled_videos.cpu().numpy() # [num_samples, 2, F, H, W]
        sampled_videos = np.expand_dims(sampled_videos, axis=1) # [num_samples, 1, 2, F, H, W]
        # to unnormalize it multiply with 255.0
        contour_cond_video = contour_cond_video.cpu().numpy() * 255.0

        save_folder = Path(save_folder)
        save_folder.mkdir(exist_ok=True, parents=True)

        # Create subdirectories for ground truths and sampled videos
        sampled_folder = save_folder / "inference_disps"
        sampled_folder.mkdir(exist_ok=True, parents=True)

        output_gif_dir = f"{save_folder}/quiver_plots"
        # Create output directory if needed
        output_dir = Path(output_gif_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Instead of saving one big file, loop over each sample
        for i in range(num_samples):
            # If cond_filenames is None or too short, just name it generically
            if cond_filenames is not None and i < len(cond_filenames):
                base_name = cond_filenames[i]
            else:
                base_name = f"sample_{i:03d}"

            sampled_disp_single = sampled_videos[i]  # shape [1, 2, F, H, W]

            # Build the output filenames
            sampled_disp_path = sampled_folder / f"{base_name}"
            np.save(sampled_disp_path, sampled_disp_single)

            if reconstructed_disp is None and gt_disp is None:
                generate_mask_disp_side_by_side_gif(contour_cond_video[i], sampled_disp_single, output_path=f"{output_gif_dir}/{cond_filenames[0].split('.npy')[0]}.gif")
            elif reconstructed_disp is None and gt_disp is not None:
                generate_displacement_quiver_gif_comparison(sampled_disp_single, gt_disp, output_path=f"{output_gif_dir}/{cond_filenames[0].split('.npy')[0]}.gif", titles=["Predicted", "Ground Truth"])
            elif reconstructed_disp is not None and gt_disp is None:
                generate_displacement_quiver_gif_comparison(sampled_disp_single, reconstructed_disp, output_path=f"{output_gif_dir}/{cond_filenames[0].split('.npy')[0]}.gif", titles=["Predicted", "Reconstructed"])
            else:
                generate_displacement_quiver_gif_predicted_reconstructed_gt(sampled_disp_single, reconstructed_disp, gt_disp, output_path=f"{output_gif_dir}/{cond_filenames[0].split('.npy')[0]}.gif")
