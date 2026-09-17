"""Evaluate the cached decoder's chunk semantics in parallel across an utterance.

The streaming attention cache contains only past keys/values, so all queries
can share full-sequence K/V with an equivalent visibility mask. Causal convs
already operate in parallel. Transposed convolution needs an explicit boundary
correction: previously emitted chunk tails never see the next chunk's input.
No model parameters or deployment/inference cache behavior are changed.
"""
from functools import lru_cache

import torch
import torch.nn.functional as F

from lits.models.components.utils import Downsample1D, Upsample1D, mask_to_bias
from meanflow_distill.kv_cache_distill import compute_chunk_starts


@lru_cache(maxsize=8)
def attention_bias(ends, static_chunk_size, left_frames, dtype, device):
    if left_frames < 0:
        raise ValueError('Parallel streaming requires a finite frame-level left context')
    # Cache ordinary tensors, also when first called inside inference_mode.
    with torch.inference_mode(False):
        position = torch.arange(ends[-1], device=device)
        limits = torch.tensor(ends, device=device)
        chunk_end = limits[torch.bucketize(position, limits, right=True)]
        if static_chunk_size > 0:
            chunk_end = torch.minimum(chunk_end,
                (position.div(static_chunk_size, rounding_mode='floor') + 1) * static_chunk_size)
        visible = ((position[None, :] < chunk_end[:, None]) &
                   (position[None, :] >= position[:, None] - left_frames))
        return mask_to_bias(visible, dtype)[None, None]


def transformer_blocks(base, x, blocks, ends):
    hidden = x.transpose(1, 2)
    bias = attention_bias(tuple(ends), base.static_chunk_size,
                          base.decoder_left_frames, hidden.dtype, hidden.device)
    for block in blocks:
        hidden, _ = block.forward_streaming(hidden, attention_mask=bias)
    return hidden.transpose(1, 2)


def upsample_without_future(module, x, ends):
    """Match old emitted tail: at 2*j-1 omit next-chunk input j, kernel tap 0."""
    if not module.use_conv_transpose:
        raise ValueError('Parallel streaming requires the cached transposed-convolution upsampler')
    if (module.conv.kernel_size, module.conv.stride, module.conv.padding,
            module.conv.output_padding, module.conv.dilation, module.conv.groups) != (
            (4,), (2,), (1,), (0,), (1,), 1):
        raise ValueError('Unsupported streaming transposed-convolution geometry')
    output = module(x)
    if len(ends) > 1:
        indices = torch.tensor(ends[:-1], device=x.device)
        previous = x.index_select(-1, indices - 1).transpose(1, 2)
        boundary = F.linear(previous, module.conv.weight[:, :, 2].transpose(0, 1),
                            module.conv.bias).transpose(1, 2)
        output = output.index_copy(-1, indices * 2 - 1, boundary)
    return output


def velocity(estimator, x, mask, mu, t, spks=None, cond=None, *, r=None, chunk_size=100):
    base = getattr(estimator, 'base', estimator)
    stride = 2 ** (len(base.down_blocks) - 1)
    if chunk_size <= 0 or chunk_size % stride:
        raise ValueError('Chunk size must align with every decoder downsampling level')
    starts = compute_chunk_starts(x.shape[-1], chunk_size)
    ends = starts[1:] + [x.shape[-1]]
    if hasattr(estimator, 'interval_projector'):
        embedding = estimator._interval_time_emb(t if r is None else r, t)
    else:
        embedding = base.time_mlp(base.time_embeddings(t))
    values = [x, mu]
    if spks is not None:
        values.append(spks.unsqueeze(-1).expand(-1, -1, x.shape[-1]))
    if cond is not None:
        values.append(cond)
    x = torch.cat(values, dim=1)
    skips = []
    current_mask = mask
    for resnet, blocks, downsample in base.down_blocks:
        x = resnet(x, current_mask, embedding)
        x = transformer_blocks(base, x, blocks, ends)
        skips.append((x, current_mask, ends))
        x = downsample(x * current_mask)
        if isinstance(downsample, Downsample1D):
            current_mask = current_mask[..., ::2]
            ends = [(end + 1) // 2 for end in ends]
    for resnet, blocks in base.mid_blocks:
        x = resnet(x, current_mask, embedding)
        x = transformer_blocks(base, x, blocks, ends)
    for resnet, blocks, upsample in base.up_blocks:
        skip, current_mask, ends = skips.pop()
        x = torch.cat([x[..., :skip.shape[-1]], skip], dim=1)
        x = resnet(x, current_mask, embedding)
        x = transformer_blocks(base, x, blocks, ends)
        x = (upsample_without_future(upsample, x * current_mask, ends)
             if isinstance(upsample, Upsample1D) else upsample(x * current_mask))
    x = base.final_block(x, current_mask)
    return base.final_proj(x * current_mask) * mask


def euler_step(decoder, x, mu, mask, spks, r_val, t_val, *, chunk_size, **_):
    # The deployment solver passes scalar times, including under BF16 autocast.
    r = torch.as_tensor(r_val, device=x.device, dtype=x.dtype)
    t = torch.full_like(r, t_val)
    model = getattr(decoder.estimator, 'module', decoder.estimator)
    when = t if hasattr(model, 'interval_projector') else r
    return x + (t_val - r_val) * velocity(model, x, mask, mu, when, spks,
                                         r=r, chunk_size=chunk_size)


def trajectory(model, mu, mask, spks, z, grid, *, chunk_size, **kwargs):
    states, velocities = [z], []
    for r, t in zip(grid, grid[1:]):
        next_x = euler_step(model.decoder, states[-1], mu, mask, spks, r, t,
                            chunk_size=chunk_size)
        velocities.append((next_x - states[-1]) / (t - r))
        states.append(next_x)
    return states, velocities


def teacher_trajectory(teacher, mu, mask, spks, z, teacher_steps, **kwargs):
    states, velocities = trajectory(teacher, mu, mask, spks, z,
        [i / teacher_steps for i in range(teacher_steps + 1)], **kwargs)
    return torch.stack(states), torch.stack(velocities)


def student_trajectory(student, mu, mask, spks, z, student_t_grid, **kwargs):
    return trajectory(student, mu, mask, spks, z, student_t_grid, **kwargs)
