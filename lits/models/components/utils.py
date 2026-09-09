import logging
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union
from lits.models.components.transformer import RelPositionMultiHeadedAttention

log = logging.getLogger(__name__)

def subsequent_chunk_mask(
        size: int,
        chunk_size: int,
        num_left_chunks: int = -1,
        device: torch.device = torch.device("cpu"),
        max_left_frames: int = -1,
) -> torch.Tensor:
    """Create mask for subsequent steps (size, size) with chunk size,
       this is for streaming encoder

    Args:
        size (int): size of mask
        chunk_size (int): size of chunk
        num_left_chunks (int): number of left chunks
            <0: use full chunk
            >=0: use num_left_chunks
        device (torch.device): "cpu" or "cuda" or torch.Tensor.device

    Returns:
        torch.Tensor: mask

    Examples:
        >>> subsequent_chunk_mask(4, 2)
        [[1, 1, 0, 0],
         [1, 1, 0, 0],
         [1, 1, 1, 1],
         [1, 1, 1, 1]]
    """
    pos_idx = torch.arange(size, device=device)
    block_value = (torch.div(pos_idx, chunk_size, rounding_mode='trunc') + 1) * chunk_size
    ret = pos_idx.unsqueeze(0) < block_value.unsqueeze(1)
    if max_left_frames >= 0:
        left_boundary = torch.clamp(pos_idx - max_left_frames, min=0)
        ret = ret & (pos_idx.unsqueeze(0) >= left_boundary.unsqueeze(1))
    elif num_left_chunks >= 0:
        chunk_indices = torch.div(pos_idx, chunk_size, rounding_mode='trunc')
        left_boundary = torch.clamp(
            (chunk_indices - num_left_chunks) * chunk_size, min=0,
        )
        ret = ret & (pos_idx.unsqueeze(0) >= left_boundary.unsqueeze(1))
    return ret

def add_optional_chunk_mask(xs: torch.Tensor,
                            masks: torch.Tensor,
                            use_dynamic_chunk: bool,
                            use_dynamic_left_chunk: bool,
                            decoding_chunk_size: int,
                            static_chunk_size: int,
                            num_decoding_left_chunks: int,
                            enable_full_context: bool = True,
                            max_left_frames: int = -1):
    """ Apply optional mask for encoder.

    Args:
        xs (torch.Tensor): padded input, (B, L, D), L for max length
        mask (torch.Tensor): mask for xs, (B, 1, L)
        use_dynamic_chunk (bool): whether to use dynamic chunk or not
        use_dynamic_left_chunk (bool): whether to use dynamic left chunk for
            training.
        decoding_chunk_size (int): decoding chunk size for dynamic chunk, it's
            0: default for training, use random dynamic chunk.
            <0: for decoding, use full chunk.
            >0: for decoding, use fixed chunk size as set.
        static_chunk_size (int): chunk size for static chunk training/decoding
            if it's greater than 0, if use_dynamic_chunk is true,
            this parameter will be ignored
        num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
            >=0: use num_decoding_left_chunks
            <0: use all left chunks
        enable_full_context (bool):
            True: chunk size is either [1, 25] or full context(max_len)
            False: chunk size ~ U[1, 25]

    Returns:
        torch.Tensor: chunk mask of the input xs.
    """
    # Whether to use chunk mask or not
    if use_dynamic_chunk:
        max_len = xs.size(1)
        if decoding_chunk_size < 0:
            chunk_size = max_len
            num_left_chunks = -1
        elif decoding_chunk_size > 0:
            chunk_size = decoding_chunk_size
            num_left_chunks = num_decoding_left_chunks
        else:
            # chunk size is either [1, 25] or full context(max_len).
            # Since we use 4 times subsampling and allow up to 1s(100 frames)
            # delay, the maximum frame is 100 / 4 = 25.
            chunk_size = torch.randint(1, max_len, (1, )).item()
            num_left_chunks = -1
            if chunk_size > max_len // 2 and enable_full_context:
                chunk_size = max_len
            else:
                chunk_size = chunk_size % 25 + 1
                if use_dynamic_left_chunk:
                    max_left_chunks = (max_len - 1) // chunk_size
                    num_left_chunks = torch.randint(0, max_left_chunks,
                                                    (1, )).item()
        chunk_masks = subsequent_chunk_mask(xs.size(1), chunk_size,
                                            num_left_chunks,
                                            xs.device)  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)
    elif static_chunk_size > 0:
        num_left_chunks = num_decoding_left_chunks
        chunk_masks = subsequent_chunk_mask(xs.size(1), static_chunk_size,
                                            num_left_chunks,
                                            xs.device,
                                            max_left_frames=max_left_frames)  # (L, L)
        chunk_masks = chunk_masks.unsqueeze(0)  # (1, L, L)
        chunk_masks = masks & chunk_masks  # (B, L, L)
    else:
        chunk_masks = masks
    assert chunk_masks.dtype == torch.bool
    if not torch.compiler.is_compiling():
        zero_rows = chunk_masks.sum(dim=-1) == 0
        if zero_rows.sum().item() != 0:
            valid_queries = masks.squeeze(1).bool()
            valid_zero_rows = zero_rows & valid_queries if zero_rows.shape == valid_queries.shape else zero_rows
            if valid_zero_rows.sum().item() != 0:
                log.warning(
                    "chunk mask is all false for %s valid timestep(s); forcing those rows to true to avoid NaNs",
                    valid_zero_rows.sum().item(),
                )
            chunk_masks[zero_rows] = True
    return chunk_masks


def concat_channels(*tensors: torch.Tensor) -> torch.Tensor:
    """Concatenate (B, C, T) tensors along channel dim. ONNX-friendly Cat op."""
    return torch.cat(tensors, dim=1)


def channels_to_time(x: torch.Tensor) -> torch.Tensor:
    """(B, C, T) -> (B, T, C) via Transpose; avoids einops rearrange + contiguous."""
    return x.transpose(1, 2)


def time_to_channels(x: torch.Tensor) -> torch.Tensor:
    """(B, T, C) -> (B, C, T) via Transpose."""
    return x.transpose(1, 2)


def expand_spk_emb(spks: torch.Tensor, length: int) -> torch.Tensor:
    """(B, C) -> (B, C, T) for speaker conditioning without einops repeat."""
    return spks.unsqueeze(-1).expand(-1, -1, length)


def build_decoder_attn_mask(
    x: torch.Tensor,
    mask: torch.Tensor,
    *,
    streaming: bool,
    static_chunk_size: int,
    num_decoding_left_chunks: int = -1,
    max_left_frames: int = -1,
) -> torch.Tensor:
    """Build transformer attention bias for decoder U-Net blocks.

    Args:
        x: (B, T, C) hidden states in time-major layout.
        mask: (B, 1, T) mel padding mask.
        streaming: use chunk-wise causal mask when True.
        static_chunk_size: chunk size for streaming mask (ignored when not streaming).
        num_decoding_left_chunks: left-chunk attention limit for streaming decode.
        max_left_frames: frame-level left context limit; overrides chunk limit.
    """
    bool_mask = mask.bool()
    if streaming:
        chunk_mask = add_optional_chunk_mask(
            x,
            bool_mask,
            False,
            False,
            0,
            static_chunk_size,
            num_decoding_left_chunks,
            max_left_frames=max_left_frames,
        )
    else:
        chunk_mask = add_optional_chunk_mask(
            x, bool_mask, False, False, 0, 0, -1
        ).repeat(1, x.size(1), 1)
    return mask_to_bias(chunk_mask, x.dtype)


def decoder_kv_cache_limit(
    static_chunk_size: int,
    num_decoding_left_chunks: int,
    max_left_frames: int = -1,
) -> int:
    """Max decoder KV-cache length; -1 means unlimited (matches encoder convention)."""
    if max_left_frames >= 0:
        return max_left_frames
    if num_decoding_left_chunks < 0 or static_chunk_size <= 0:
        return -1
    return static_chunk_size * num_decoding_left_chunks


def trim_decoder_kv_cache(
    cache: torch.Tensor,
    max_size: int,
    cache_offset: int = 0,
) -> Tuple[torch.Tensor, int]:
    """Drop leftmost KV frames from cache and advance the global offset."""
    if max_size < 0 or cache.numel() == 0:
        return cache, cache_offset
    cache_len = cache.size(2)
    if cache_len <= max_size:
        return cache, cache_offset
    drop = cache_len - max_size
    return cache[:, :, drop:, :].contiguous(), cache_offset + drop


def build_streaming_decoder_attn_mask(
    new_len: int,
    cached_len: int,
    static_chunk_size: int,
    num_decoding_left_chunks: int,
    dtype: torch.dtype,
    device: torch.device,
    cache_offset: int = 0,
    max_left_frames: int = -1,
) -> torch.Tensor:
    """Build attention bias for streaming decoder with KV-cache.

    Constructs a mask where new-frame queries attend to all (cached + new)
    keys under the same chunk-causal pattern used at training time.

    Args:
        cache_offset: global mel-frame index of cache[:, :, 0, :].

    Returns:
        (1, 1, new_len, cached_len + new_len) float bias.
    """
    query_start = cache_offset + cached_len
    key_start = cache_offset
    key_end = cache_offset + cached_len + new_len
    query_idx = torch.arange(
        query_start, query_start + new_len, device=device,
    )
    key_idx = torch.arange(key_start, key_end, device=device)
    if static_chunk_size > 0:
        block_value = (
            torch.div(query_idx, static_chunk_size, rounding_mode='trunc') + 1
        ) * static_chunk_size
        mask = key_idx.unsqueeze(0) < block_value.unsqueeze(1)
        if max_left_frames >= 0:
            left_boundary = torch.clamp(query_idx - max_left_frames, min=0)
            mask = mask & (key_idx.unsqueeze(0) >= left_boundary.unsqueeze(1))
        elif num_decoding_left_chunks >= 0:
            chunk_indices = torch.div(
                query_idx, static_chunk_size, rounding_mode='trunc',
            )
            left_boundary = torch.clamp(
                (chunk_indices - num_decoding_left_chunks) * static_chunk_size,
                min=0,
            )
            mask = mask & (key_idx.unsqueeze(0) >= left_boundary.unsqueeze(1))
    elif max_left_frames >= 0:
        left_boundary = torch.clamp(query_idx - max_left_frames, min=0)
        mask = key_idx.unsqueeze(0) >= left_boundary.unsqueeze(1)
    else:
        mask = torch.ones(
            new_len, cached_len + new_len, dtype=torch.bool, device=device,
        )
    return mask_to_bias(mask.unsqueeze(0), dtype).unsqueeze(1)


class SinusoidalPosEmb(torch.nn.Module):
    """Sinusoidal positional embedding for time steps."""
    def __init__(self, dim):
        super().__init__()
        self.dim = dim
        assert self.dim % 2 == 0, "SinusoidalPosEmb requires dim to be even"

    def forward(self, x, scale=1000):
        if x.ndim < 1:
            x = x.unsqueeze(0)
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device).float() * -emb)
        emb = scale * x.unsqueeze(1) * emb.unsqueeze(0)
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb
    
class TimestepEmbedder(nn.Module):
    def __init__(self, dim, nfreq=256):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(nfreq, dim), nn.SiLU(), nn.Linear(dim, dim))
        self.nfreq = nfreq

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        half_dim = dim // 2
        freqs = torch.exp(
            -math.log(max_period)
            * torch.arange(start=0, end=half_dim, dtype=torch.float32)
            / half_dim
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat(
                [embedding, torch.zeros_like(embedding[:, :1])], dim=-1
            )
        return embedding

    def forward(self, t):
        t = t*1000
        t_freq = self.timestep_embedding(t, self.nfreq)
        t_emb = self.mlp(t_freq)
        return t_emb

class Block1D(torch.nn.Module):
    """1D Conv + GroupNorm + Mish block."""
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.block = torch.nn.Sequential(
            torch.nn.Conv1d(dim, dim_out, 3, padding=1),
            torch.nn.GroupNorm(groups, dim_out),
            nn.Mish(),
        )
    def forward(self, x, mask):
        output = self.block(x * mask)
        return output * mask

class ResnetBlock1D(torch.nn.Module):
    """1D ResNet block with time embedding."""
    def __init__(self, dim, dim_out, time_emb_dim, groups=8):
        super().__init__()
        self.mlp = torch.nn.Sequential(nn.Mish(), torch.nn.Linear(time_emb_dim, dim_out))
        self.block1 = Block1D(dim, dim_out, groups=groups)
        self.block2 = Block1D(dim_out, dim_out, groups=groups)
        self.res_conv = torch.nn.Conv1d(dim, dim_out, 1)
    def forward(self, x, mask, time_emb):
        h = self.block1(x, mask)
        h += self.mlp(time_emb).unsqueeze(-1)
        h = self.block2(h, mask)
        output = h + self.res_conv(x * mask)
        return output

class Downsample1D(nn.Module):
    """1D downsampling via strided conv."""
    def __init__(self, dim):
        super().__init__()
        self.conv = torch.nn.Conv1d(dim, dim, 3, 2, 1)
    def forward(self, x):
        return self.conv(x)

    def forward_streaming(
        self, x: torch.Tensor, buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Streaming forward that buffers 2 tail frames for boundary correctness.

        The stride-2 conv at position j reads input[2j-1 : 2j+2].  When a new
        chunk starts at an even index, the first new output depends on the last
        frame of the previous chunk.  Buffering 2 frames guarantees correct
        stride alignment: the first sub-output position is the old boundary
        (discarded) and subsequent positions are the true new outputs.
        """
        new_buffer = x[:, :, -2:].clone()
        if buffer is not None and buffer.size(-1) > 0:
            x = torch.cat([buffer, x], dim=-1)
            out = self.conv(x)
            out = out[:, :, 1:]
        else:
            out = self.conv(x)
        return out, new_buffer

class TimestepEmbedding(nn.Module):
    """Time step embedding with optional condition projection."""
    def __init__(
        self,
        in_channels: int,
        time_embed_dim: int,
        act_fn: str = "silu",
        out_dim: int = None,
        post_act_fn: Optional[str] = None,
        cond_proj_dim=None,
    ):
        super().__init__()
        self.linear_1 = nn.Linear(in_channels, time_embed_dim)
        if cond_proj_dim is not None:
            self.cond_proj = nn.Linear(cond_proj_dim, in_channels, bias=False)
        else:
            self.cond_proj = None
        from diffusers.models.activations import get_activation
        self.act = get_activation(act_fn)
        if out_dim is not None:
            time_embed_dim_out = out_dim
        else:
            time_embed_dim_out = time_embed_dim
        self.linear_2 = nn.Linear(time_embed_dim, time_embed_dim_out)
        if post_act_fn is None:
            self.post_act = None
        else:
            self.post_act = get_activation(post_act_fn)
    def forward(self, sample, condition=None):
        if condition is not None:
            sample = sample + self.cond_proj(condition)
        sample = self.linear_1(sample)
        if self.act is not None:
            sample = self.act(sample)
        sample = self.linear_2(sample)
        if self.post_act is not None:
            sample = self.post_act(sample)
        return sample

class Upsample1D(nn.Module):
    """A 1D upsampling layer with an optional convolution."""
    def __init__(self, channels, use_conv=False, use_conv_transpose=True, out_channels=None, name="conv"):
        super().__init__()
        self.channels = channels
        self.out_channels = out_channels or channels
        self.use_conv = use_conv
        self.use_conv_transpose = use_conv_transpose
        self.name = name
        self.conv = None
        if use_conv_transpose:
            self.conv = nn.ConvTranspose1d(channels, self.out_channels, 4, 2, 1)
        elif use_conv:
            self.conv = nn.Conv1d(self.channels, self.out_channels, 3, padding=1)
    def forward(self, inputs):
        assert inputs.shape[1] == self.channels
        if self.use_conv_transpose:
            return self.conv(inputs)
        outputs = F.interpolate(inputs, scale_factor=2.0, mode="nearest")
        if self.use_conv:
            outputs = self.conv(outputs)
        return outputs

    def forward_streaming(
        self, x: torch.Tensor, buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Streaming ConvTranspose1d upsampling with 1-frame half-res buffer.

        The transposed conv output at position 2P-1 depends on input[P], so we
        buffer the last half-res frame to reproduce the correct boundary when
        the next chunk arrives.  The first 2 full-res output positions from the
        overlap are discarded.
        """
        new_buffer = x[:, :, -1:].clone()
        if buffer is not None and buffer.size(-1) > 0:
            x = torch.cat([buffer, x], dim=-1)
            out = self.conv(x)
            out = out[:, :, 2:]
        else:
            out = self.conv(x)
        return out, new_buffer

def mask_to_bias(mask: torch.Tensor, dtype: torch.dtype) -> torch.Tensor:
    """Convert boolean mask to attention bias."""
    assert mask.dtype == torch.bool
    assert dtype in [torch.float32, torch.bfloat16, torch.float16]
    mask = mask.to(dtype)
    mask = (1.0 - mask) * -1.0e+10
    return mask

class Transpose(torch.nn.Module):
    """Module wrapper for torch.transpose."""
    def __init__(self, dim0: int, dim1: int):
        super().__init__()
        self.dim0 = dim0
        self.dim1 = dim1
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.transpose(x, self.dim0, self.dim1)
        return x

class CausalConv1d(torch.nn.Conv1d):
    """Causal 1D convolution."""
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        device=None,
        dtype=None
    ) -> None:
        super(CausalConv1d, self).__init__(in_channels, out_channels,
                                           kernel_size, stride,
                                           padding=0, dilation=dilation,
                                           groups=groups, bias=bias,
                                           padding_mode=padding_mode,
                                           device=device, dtype=dtype)
        assert stride == 1
        self.causal_padding = kernel_size - 1
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.pad(x, (self.causal_padding, 0), value=0.0)
        x = super(CausalConv1d, self).forward(x)
        return x

    def forward_streaming(
        self, x: torch.Tensor, buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Streaming forward: replace zero-padding with buffered tail frames.

        Args:
            x: (B, C_in, T_new)
            buffer: (B, C_in, causal_padding) or None (first call).

        Returns:
            (output, new_buffer)
        """
        if buffer is not None and buffer.size(-1) > 0:
            x_padded = torch.cat([buffer, x], dim=-1)
        else:
            x_padded = F.pad(x, (self.causal_padding, 0), value=0.0)
        new_buffer = x_padded[:, :, -self.causal_padding:].clone()
        out = super(CausalConv1d, self).forward(x_padded)
        return out, new_buffer

class CausalBlock1D(Block1D):
    def __init__(self, dim: int, dim_out: int):
        super(CausalBlock1D, self).__init__(dim, dim_out)
        self.block = torch.nn.Sequential(
            CausalConv1d(dim, dim_out, 3),
            Transpose(1, 2),
            nn.LayerNorm(dim_out),
            Transpose(1, 2),
            nn.Mish(),
        )

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        output = self.block(x * mask)
        return output * mask

    def forward_streaming(
        self, x: torch.Tensor, mask: torch.Tensor,
        conv_buffer: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Streaming forward: use tail buffer for the internal CausalConv1d."""
        x = x * mask
        conv: CausalConv1d = self.block[0]
        x, new_buffer = conv.forward_streaming(x, conv_buffer)
        for layer in self.block[1:]:
            x = layer(x)
        return x * mask, new_buffer

class CausalResnetBlock1D(ResnetBlock1D):
    def __init__(self, dim: int, dim_out: int, time_emb_dim: int, groups: int = 8):
        super(CausalResnetBlock1D, self).__init__(dim, dim_out, time_emb_dim, groups)
        self.block1 = CausalBlock1D(dim, dim_out)
        self.block2 = CausalBlock1D(dim_out, dim_out)

    def forward_streaming(
        self, x: torch.Tensor, mask: torch.Tensor, time_emb: torch.Tensor,
        conv_buffers: Optional[list] = None,
    ) -> Tuple[torch.Tensor, list]:
        """Streaming forward: thread tail buffers through both CausalBlock1Ds."""
        buf1 = conv_buffers[0] if conv_buffers else None
        buf2 = conv_buffers[1] if conv_buffers else None
        h, new_buf1 = self.block1.forward_streaming(x, mask, buf1)
        h = h + self.mlp(time_emb).unsqueeze(-1)
        h, new_buf2 = self.block2.forward_streaming(h, mask, buf2)
        output = h + self.res_conv(x * mask)
        return output, [new_buf1, new_buf2]

class BaseSubsampling(torch.nn.Module):

    def __init__(self):
        super().__init__()
        self.right_context = 0
        self.subsampling_rate = 1

    def position_encoding(self, offset: Union[int, torch.Tensor],
                          size: int) -> torch.Tensor:
        return self.pos_enc.position_encoding(offset, size)

class LinearNoSubsampling(BaseSubsampling):
    """Linear transform the input without subsampling

    Args:
        idim (int): Input dimension.
        odim (int): Output dimension.
        dropout_rate (float): Dropout rate.

    """

    def __init__(self, idim: int, odim: int, dropout_rate: float,
                 pos_enc_class: torch.nn.Module):
        """Construct an linear object."""
        super().__init__()
        self.out = torch.nn.Sequential(
            torch.nn.Linear(idim, odim),
            torch.nn.LayerNorm(odim, eps=1e-5),
            torch.nn.Dropout(dropout_rate),
        )
        self.pos_enc = pos_enc_class
        self.right_context = 0
        self.subsampling_rate = 1

    def forward(
        self,
        x: torch.Tensor,
        x_mask: torch.Tensor,
        offset: Union[int, torch.Tensor] = 0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Input x.

        Args:
            x (torch.Tensor): Input tensor (#batch, time, idim).
            x_mask (torch.Tensor): Input mask (#batch, 1, time).

        Returns:
            torch.Tensor: linear input tensor (#batch, time', odim),
                where time' = time .
            torch.Tensor: linear input mask (#batch, 1, time'),
                where time' = time .

        """
        x = self.out(x)
        x, pos_emb = self.pos_enc(x, offset)
        return x, pos_emb, x_mask
    
class EspnetRelPositionalEncoding(torch.nn.Module):
    """Relative positional encoding module (new implementation).

    Details can be found in https://github.com/espnet/espnet/pull/2816.

    See : Appendix B in https://arxiv.org/abs/1901.02860

    Args:
        d_model (int): Embedding dimension.
        dropout_rate (float): Dropout rate.
        max_len (int): Maximum input length.

    """

    def __init__(self, d_model: int, dropout_rate: float, max_len: int = 5000):
        """Construct an PositionalEncoding object."""
        super(EspnetRelPositionalEncoding, self).__init__()
        self.d_model = d_model
        self.xscale = math.sqrt(self.d_model)
        self.dropout = torch.nn.Dropout(p=dropout_rate)
        self.pe = None
        self.extend_pe(torch.tensor(0.0).expand(1, max_len))

    def extend_pe(self, x: torch.Tensor):
        """Reset the positional encodings."""
        if self.pe is not None:
            # self.pe contains both positive and negative parts
            # the length of self.pe is 2 * input_len - 1
            if self.pe.size(1) >= x.size(1) * 2 - 1:
                if self.pe.dtype != x.dtype or self.pe.device != x.device:
                    self.pe = self.pe.to(dtype=x.dtype, device=x.device)
                return
        # Suppose `i` means to the position of query vecotr and `j` means the
        # position of key vector. We use position relative positions when keys
        # are to the left (i>j) and negative relative positions otherwise (i<j).
        pe_positive = torch.zeros(x.size(1), self.d_model)
        pe_negative = torch.zeros(x.size(1), self.d_model)
        position = torch.arange(0, x.size(1), dtype=torch.float32).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, self.d_model, 2, dtype=torch.float32)
            * -(math.log(10000.0) / self.d_model)
        )
        pe_positive[:, 0::2] = torch.sin(position * div_term)
        pe_positive[:, 1::2] = torch.cos(position * div_term)
        pe_negative[:, 0::2] = torch.sin(-1 * position * div_term)
        pe_negative[:, 1::2] = torch.cos(-1 * position * div_term)

        # Reserve the order of positive indices and concat both positive and
        # negative indices. This is used to support the shifting trick
        # as in https://arxiv.org/abs/1901.02860
        pe_positive = torch.flip(pe_positive, [0]).unsqueeze(0)
        pe_negative = pe_negative[1:].unsqueeze(0)
        pe = torch.cat([pe_positive, pe_negative], dim=1)
        self.pe = pe.to(device=x.device, dtype=x.dtype)

    def forward(self, x: torch.Tensor, offset: Union[int, torch.Tensor] = 0) \
            -> Tuple[torch.Tensor, torch.Tensor]:
        """Add positional encoding.

        Args:
            x (torch.Tensor): Input tensor (batch, time, `*`).

        Returns:
            torch.Tensor: Encoded tensor (batch, time, `*`).

        """
        self.extend_pe(x)
        x = x * self.xscale
        pos_emb = self.position_encoding(size=x.size(1), offset=offset)
        return self.dropout(x), self.dropout(pos_emb)

    def position_encoding(self,
                          offset: Union[int, torch.Tensor],
                          size: int) -> torch.Tensor:
        """ For getting encoding in a streaming fashion

        Attention!!!!!
        we apply dropout only once at the whole utterance level in a none
        streaming way, but will call this function several times with
        increasing input size in a streaming scenario, so the dropout will
        be applied several times.

        Args:
            offset (int or torch.tensor): start offset
            size (int): required size of position encoding

        Returns:
            torch.Tensor: Corresponding encoding
        """
        # How to subscript a Union type:
        #   https://github.com/pytorch/pytorch/issues/69434
        if isinstance(offset, int):
            pos_emb = self.pe[
                :,
                self.pe.size(1) // 2 - size - offset + 1: self.pe.size(1) // 2 + size + offset,
            ]
        elif isinstance(offset, torch.Tensor):
            pos_emb = self.pe[
                :,
                self.pe.size(1) // 2 - size - offset + 1: self.pe.size(1) // 2 + size + offset,
            ]
        return pos_emb
    
class Swish(torch.nn.Module):
    """Construct an Swish object."""

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Return Swish activation function."""
        return x * torch.sigmoid(x)
    
class PreLookaheadLayer(nn.Module):
    def __init__(self, channels: int, pre_lookahead_len: int = 1):
        super().__init__()
        self.channels = channels
        self.pre_lookahead_len = pre_lookahead_len
        self.conv1 = nn.Conv1d(
            channels, channels,
            kernel_size=pre_lookahead_len + 1,
            stride=1, padding=0,
        )
        self.conv2 = nn.Conv1d(
            channels, channels,
            kernel_size=3, stride=1, padding=0,
        )

    def forward(self, inputs: torch.Tensor, context: torch.Tensor = torch.zeros(0, 0, 0)) -> torch.Tensor:
        """
        inputs: (batch_size, seq_len, channels)
        """
        outputs = inputs.transpose(1, 2).contiguous()
        context = context.transpose(1, 2).contiguous()
        # look ahead
        if context.size(2) == 0:
            outputs = F.pad(outputs, (0, self.pre_lookahead_len), mode='constant', value=0.0)
        else:
            assert self.training is False, 'you have passed context, make sure that you are running inference mode'
            assert context.size(2) == self.pre_lookahead_len
            outputs = F.pad(torch.concat([outputs, context], dim=2), (0, self.pre_lookahead_len - context.size(2)), mode='constant', value=0.0)
        outputs = F.leaky_relu(self.conv1(outputs))
        # outputs
        outputs = F.pad(outputs, (self.conv2.kernel_size[0] - 1, 0), mode='constant', value=0.0)
        outputs = self.conv2(outputs)
        outputs = outputs.transpose(1, 2).contiguous()

        # residual connection
        outputs = outputs + inputs
        return outputs
    
class ConformerEncoderLayer(nn.Module):
    """Encoder layer module.
    Args:
        size (int): Input dimension.
        self_attn (torch.nn.Module): Self-attention module instance.
            `MultiHeadedAttention` or `RelPositionMultiHeadedAttention`
            instance can be used as the argument.
        feed_forward (torch.nn.Module): Feed-forward module instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        feed_forward_macaron (torch.nn.Module): Additional feed-forward module
             instance.
            `PositionwiseFeedForward` instance can be used as the argument.
        conv_module (torch.nn.Module): Convolution module instance.
            `ConvlutionModule` instance can be used as the argument.
        dropout_rate (float): Dropout rate.
        normalize_before (bool):
            True: use layer_norm before each sub-block.
            False: use layer_norm after each sub-block.
    """

    def __init__(
        self,
        size: int,
        self_attn: torch.nn.Module,
        feed_forward: Optional[nn.Module] = None,
        feed_forward_macaron: Optional[nn.Module] = None,
        conv_module: Optional[nn.Module] = None,
        dropout_rate: float = 0.1,
        normalize_before: bool = True,
    ):
        """Construct an EncoderLayer object."""
        super().__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.feed_forward_macaron = feed_forward_macaron
        self.conv_module = conv_module
        self.norm_ff = nn.LayerNorm(size, eps=1e-12)  # for the FNN module
        self.norm_mha = nn.LayerNorm(size, eps=1e-12)  # for the MHA module
        if feed_forward_macaron is not None:
            self.norm_ff_macaron = nn.LayerNorm(size, eps=1e-12)
            self.ff_scale = 0.5
        else:
            self.ff_scale = 1.0
        if self.conv_module is not None:
            self.norm_conv = nn.LayerNorm(size, eps=1e-12)  # for the CNN module
            self.norm_final = nn.LayerNorm(
                size, eps=1e-12)  # for the final output of the block
        self.dropout = nn.Dropout(dropout_rate)
        self.size = size
        self.normalize_before = normalize_before

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        pos_emb: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        att_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
        cnn_cache: torch.Tensor = torch.zeros((0, 0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute encoded features.

        Args:
            x (torch.Tensor): (#batch, time, size)
            mask (torch.Tensor): Mask tensor for the input (#batch, time，time),
                (0, 0, 0) means fake mask.
            pos_emb (torch.Tensor): positional encoding, must not be None
                for ConformerEncoderLayer.
            mask_pad (torch.Tensor): batch padding mask used for conv module.
                (#batch, 1，time), (0, 0, 0) means fake mask.
            att_cache (torch.Tensor): Cache tensor of the KEY & VALUE
                (#batch=1, head, cache_t1, d_k * 2), head * d_k == size.
            cnn_cache (torch.Tensor): Convolution cache in conformer layer
                (#batch=1, size, cache_t2)
        Returns:
            torch.Tensor: Output tensor (#batch, time, size).
            torch.Tensor: Mask tensor (#batch, time, time).
            torch.Tensor: att_cache tensor,
                (#batch=1, head, cache_t1 + time, d_k * 2).
            torch.Tensor: cnn_cahce tensor (#batch, size, cache_t2).
        """

        # whether to use macaron style
        if self.feed_forward_macaron is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_ff_macaron(x)
            x = residual + self.ff_scale * self.dropout(
                self.feed_forward_macaron(x))
            if not self.normalize_before:
                x = self.norm_ff_macaron(x)

        # multi-headed self-attention module
        residual = x
        if self.normalize_before:
            x = self.norm_mha(x)
        x_att, new_att_cache = self.self_attn(x, x, x, mask, pos_emb,
                                              att_cache)
        x = residual + self.dropout(x_att)
        if not self.normalize_before:
            x = self.norm_mha(x)

        # convolution module
        # Fake new cnn cache here, and then change it in conv_module
        new_cnn_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)
        if self.conv_module is not None:
            residual = x
            if self.normalize_before:
                x = self.norm_conv(x)
            x, new_cnn_cache = self.conv_module(x, mask_pad, cnn_cache)
            x = residual + self.dropout(x)

            if not self.normalize_before:
                x = self.norm_conv(x)

        # feed forward module
        residual = x
        if self.normalize_before:
            x = self.norm_ff(x)

        x = residual + self.ff_scale * self.dropout(self.feed_forward(x))
        if not self.normalize_before:
            x = self.norm_ff(x)

        if self.conv_module is not None:
            x = self.norm_final(x)

        return x, mask, new_att_cache, new_cnn_cache
    
class PositionwiseFeedForward(torch.nn.Module):
    """Positionwise feed forward layer.

    FeedForward are appied on each position of the sequence.
    The output dim is same with the input dim.

    Args:
        idim (int): Input dimenstion.
        hidden_units (int): The number of hidden units.
        dropout_rate (float): Dropout rate.
        activation (torch.nn.Module): Activation function
    """

    def __init__(
            self,
            idim: int,
            hidden_units: int,
            dropout_rate: float,
            activation: torch.nn.Module = torch.nn.ReLU(),
    ):
        """Construct a PositionwiseFeedForward object."""
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = torch.nn.Linear(idim, hidden_units)
        self.activation = activation
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.w_2 = torch.nn.Linear(hidden_units, idim)

    def forward(self, xs: torch.Tensor) -> torch.Tensor:
        """Forward function.

        Args:
            xs: input tensor (B, L, D)
        Returns:
            output tensor, (B, L, D)
        """
        return self.w_2(self.dropout(self.activation(self.w_1(xs))))
    
class ConvolutionModule(nn.Module):
    """ConvolutionModule in Conformer model."""

    def __init__(self,
                 channels: int,
                 kernel_size: int = 15,
                 activation: nn.Module = nn.ReLU(),
                 norm: str = "batch_norm",
                 causal: bool = False,
                 bias: bool = True):
        """Construct an ConvolutionModule object.
        Args:
            channels (int): The number of channels of conv layers.
            kernel_size (int): Kernel size of conv layers.
            causal (int): Whether use causal convolution or not
        """
        super().__init__()

        self.pointwise_conv1 = nn.Conv1d(
            channels,
            2 * channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        # self.lorder is used to distinguish if it's a causal convolution,
        # if self.lorder > 0: it's a causal convolution, the input will be
        #    padded with self.lorder frames on the left in forward.
        # else: it's a symmetrical convolution
        if causal:
            padding = 0
            self.lorder = kernel_size - 1
        else:
            # kernel_size should be an odd number for none causal convolution
            assert (kernel_size - 1) % 2 == 0
            padding = (kernel_size - 1) // 2
            self.lorder = 0
        self.depthwise_conv = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            stride=1,
            padding=padding,
            groups=channels,
            bias=bias,
        )

        assert norm in ['batch_norm', 'layer_norm']
        if norm == "batch_norm":
            self.use_layer_norm = False
            self.norm = nn.BatchNorm1d(channels)
        else:
            self.use_layer_norm = True
            self.norm = nn.LayerNorm(channels)

        self.pointwise_conv2 = nn.Conv1d(
            channels,
            channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=bias,
        )
        self.activation = activation

    def forward(
        self,
        x: torch.Tensor,
        mask_pad: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
        cache: torch.Tensor = torch.zeros((0, 0, 0)),
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute convolution module.
        Args:
            x (torch.Tensor): Input tensor (#batch, time, channels).
            mask_pad (torch.Tensor): used for batch padding (#batch, 1, time),
                (0, 0, 0) means fake mask.
            cache (torch.Tensor): left context cache, it is only
                used in causal convolution (#batch, channels, cache_t),
                (0, 0, 0) meas fake cache.
        Returns:
            torch.Tensor: Output tensor (#batch, time, channels).
        """
        # exchange the temporal dimension and the feature dimension
        x = x.transpose(1, 2)  # (#batch, channels, time)

        # mask batch padding
        if mask_pad.size(2) > 0:  # time > 0
            x.masked_fill_(~mask_pad, 0.0)

        if self.lorder > 0:
            if cache.size(2) == 0:  # cache_t == 0
                x = nn.functional.pad(x, (self.lorder, 0), 'constant', 0.0)
            else:
                assert cache.size(0) == x.size(0)  # equal batch
                assert cache.size(1) == x.size(1)  # equal channel
                x = torch.cat((cache, x), dim=2)
            assert (x.size(2) > self.lorder)
            new_cache = x[:, :, -self.lorder:]
        else:
            # It's better we just return None if no cache is required,
            # However, for JIT export, here we just fake one tensor instead of
            # None.
            new_cache = torch.zeros((0, 0, 0), dtype=x.dtype, device=x.device)

        # GLU mechanism
        x = self.pointwise_conv1(x)  # (batch, 2*channel, dim)
        x = nn.functional.glu(x, dim=1)  # (batch, channel, dim)

        # 1D Depthwise Conv
        x = self.depthwise_conv(x)
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.activation(self.norm(x))
        if self.use_layer_norm:
            x = x.transpose(1, 2)
        x = self.pointwise_conv2(x)
        # mask batch padding
        if mask_pad.size(2) > 0:  # time > 0
            x.masked_fill_(~mask_pad, 0.0)

        return x.transpose(1, 2), new_cache

class ConformerEncoder(torch.nn.Module):

    def __init__(
        self,
        input_size: int,
        output_size: int = 80,
        attention_heads: int = 4,
        linear_units: int = 2048,
        num_blocks: int = 6,
        dropout_rate: float = 0.1,
        positional_dropout_rate: float = 0.1,
        attention_dropout_rate: float = 0.0,
        normalize_before: bool = True,
        static_chunk_size: int = 0,
        use_dynamic_chunk: bool = False,
        global_cmvn: torch.nn.Module = None,
        use_dynamic_left_chunk: bool = False,
        macaron_style: bool = True,
        use_cnn_module: bool = True,
        cnn_module_kernel: int = 15,
        causal: bool = False,
        cnn_module_norm: str = "batch_norm",
        key_bias: bool = True,
        gradient_checkpointing: bool = False,
    ):
        """
        Args:
            input_size (int): input dim
            output_size (int): dimension of attention
            attention_heads (int): the number of heads of multi head attention
            linear_units (int): the hidden units number of position-wise feed
                forward
            num_blocks (int): the number of decoder blocks
            dropout_rate (float): dropout rate
            attention_dropout_rate (float): dropout rate in attention
            positional_dropout_rate (float): dropout rate after adding
                positional encoding
            input_layer (str): input layer type.
                optional [linear, conv2d, conv2d6, conv2d8]
            pos_enc_layer_type (str): Encoder positional encoding layer type.
                opitonal [abs_pos, scaled_abs_pos, rel_pos, no_pos]
            normalize_before (bool):
                True: use layer_norm before each sub-block of a layer.
                False: use layer_norm after each sub-block of a layer.
            static_chunk_size (int): chunk size for static chunk training and
                decoding
            use_dynamic_chunk (bool): whether use dynamic chunk size for
                training or not, You can only use fixed chunk(chunk_size > 0)
                or dyanmic chunk size(use_dynamic_chunk = True)
            global_cmvn (Optional[torch.nn.Module]): Optional GlobalCMVN module
            use_dynamic_left_chunk (bool): whether use dynamic left chunk in
                dynamic chunk training
            key_bias: whether use bias in attention.linear_k, False for whisper models.
            gradient_checkpointing: rerunning a forward-pass segment for each
                checkpointed segment during backward.
        """
        super().__init__()
        self._output_size = output_size

        self.global_cmvn = global_cmvn
        self.embed = LinearNoSubsampling(
            input_size,
            output_size,
            dropout_rate,
            EspnetRelPositionalEncoding(output_size, positional_dropout_rate),
        )

        self.normalize_before = normalize_before
        self.after_norm = torch.nn.LayerNorm(output_size, eps=1e-5)
        self.static_chunk_size = static_chunk_size
        self.use_dynamic_chunk = use_dynamic_chunk
        self.use_dynamic_left_chunk = use_dynamic_left_chunk
        self.gradient_checkpointing = gradient_checkpointing
        activation = getattr(torch.nn, "SiLU", Swish)()
        # self-attention module definition
        encoder_selfattn_layer_args = (
            attention_heads,
            output_size,
            attention_dropout_rate,
            key_bias,
        )
        # feed-forward module definition
        positionwise_layer_args = (
            output_size,
            linear_units,
            dropout_rate,
            activation,
        )
        # convolution module definition
        convolution_layer_args = (output_size, cnn_module_kernel, activation,
                                  cnn_module_norm, causal)
        self.pre_lookahead_layer = PreLookaheadLayer(channels=output_size, pre_lookahead_len=3)
        self.encoders = torch.nn.ModuleList([
            ConformerEncoderLayer(
                output_size,
                RelPositionMultiHeadedAttention(
                    *encoder_selfattn_layer_args),
                PositionwiseFeedForward(*positionwise_layer_args),
                PositionwiseFeedForward(
                    *positionwise_layer_args) if macaron_style else None,
                ConvolutionModule(
                    *convolution_layer_args) if use_cnn_module else None,
                dropout_rate,
                normalize_before,
            ) for _ in range(num_blocks)
        ])

    def output_size(self) -> int:
        return self._output_size

    def forward(
        self,
        xs: torch.Tensor,
        masks: torch.Tensor,
        context: torch.Tensor = torch.zeros(0, 0, 0),
        decoding_chunk_size: int = 0,
        num_decoding_left_chunks: int = -1,
        streaming: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Embed positions in tensor.

        Args:
            xs: padded input tensor (B, T, D)
            xs_lens: input length (B)
            decoding_chunk_size: decoding chunk size for dynamic chunk
                0: default for training, use random dynamic chunk.
                <0: for decoding, use full chunk.
                >0: for decoding, use fixed chunk size as set.
            num_decoding_left_chunks: number of left chunks, this is for decoding,
            the chunk size is decoding_chunk_size.
                >=0: use num_decoding_left_chunks
                <0: use all left chunks
        Returns:
            encoder output tensor xs, and subsampled masks
            xs: padded output tensor (B, T' ~= T/subsample_rate, D)
            masks: torch.Tensor batch padding mask after subsample
                (B, 1, T' ~= T/subsample_rate)
        NOTE(xcsong):
            We pass the `__call__` method of the modules instead of `forward` to the
            checkpointing API because `__call__` attaches all the hooks of the module.
            https://discuss.pytorch.org/t/any-different-between-model-input-and-model-forward-input/3690/2
        """

        xs = xs.transpose(2, 1)
        context = context.transpose(2, 1)
        masks = masks.bool()
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, masks = self.embed(xs, masks)
        if context.size(1) != 0:
            assert self.training is False, 'you have passed context, make sure that you are running inference mode'
            context_masks = torch.ones(1, 1, context.size(1)).to(masks)
            context, _, _ = self.embed(context, context_masks, offset=xs.size(1))
        mask_pad = masks  # (B, 1, T/subsample_rate)
        chunk_masks = add_optional_chunk_mask(xs, masks, False, False, 0, self.static_chunk_size if streaming is True else 0, -1)
        # lookahead + conformer encoder
        xs = self.pre_lookahead_layer(xs, context=context)
        xs = self.forward_layers(xs, chunk_masks, pos_emb, mask_pad)

        if self.normalize_before:
            xs = self.after_norm(xs)
        xs = xs.transpose(2, 1)
        # Here we assume the mask is not changed in encoder layers, so just
        # return the masks before encoder layers, and the masks will be used
        # for cross attention with decoder later
        return xs

    def forward_layers(self, xs: torch.Tensor, chunk_masks: torch.Tensor,
                       pos_emb: torch.Tensor,
                       mask_pad: torch.Tensor) -> torch.Tensor:
        for layer in self.encoders:
            xs, chunk_masks, _, _ = layer(xs, chunk_masks, pos_emb, mask_pad)
        return xs

    def forward_layers_chunk(
        self,
        xs: torch.Tensor,
        att_mask: torch.Tensor,
        pos_emb: torch.Tensor,
        att_cache: torch.Tensor,
        cnn_cache: torch.Tensor,
        required_cache_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Run encoder layers with KV-cache for streaming chunk inference."""
        elayers, cache_t1 = att_cache.size(0), att_cache.size(2)
        chunk_size = xs.size(1)
        attention_key_size = cache_t1 + chunk_size
        if required_cache_size < 0:
            next_cache_start = 0
        elif required_cache_size == 0:
            next_cache_start = attention_key_size
        else:
            next_cache_start = max(attention_key_size - required_cache_size, 0)

        r_att_cache = []
        r_cnn_cache = []
        for i, layer in enumerate(self.encoders):
            xs, _, new_att_cache, new_cnn_cache = layer(
                xs,
                att_mask,
                pos_emb,
                mask_pad=torch.ones((0, 0, 0), dtype=torch.bool, device=xs.device),
                att_cache=att_cache[i:i + 1] if elayers > 0 else att_cache,
                cnn_cache=cnn_cache[i] if cnn_cache.size(0) > 0 else cnn_cache,
            )
            r_att_cache.append(new_att_cache[:, :, next_cache_start:, :])
            r_cnn_cache.append(new_cnn_cache.unsqueeze(0))

        r_att_cache = torch.cat(r_att_cache, dim=0)
        r_cnn_cache = torch.cat(r_cnn_cache, dim=0)
        return xs, r_att_cache, r_cnn_cache

    @torch.inference_mode()
    def forward_chunk(
        self,
        xs: torch.Tensor,
        offset: int,
        required_cache_size: int,
        att_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        cnn_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        context: torch.Tensor = torch.zeros(0, 0, 0),
        att_mask: torch.Tensor = torch.ones((0, 0, 0), dtype=torch.bool),
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward one streaming chunk with attention / conv KV-cache.

        Args:
            xs: chunk input (1, time, input_size), raw mu features before embed.
            offset: absolute time offset of ``xs`` in the full utterance.
            required_cache_size: max attention cache length for next chunk;
                <0 keeps full history, 0 drops cache, >0 trims to that size.
            att_cache: (elayers, head, cache_t1, d_k * 2).
            cnn_cache: (elayers, 1, hidden-dim, cache_t2).
            context: optional pre-lookahead raw context (1, pre_lookahead, input_size).
        Returns:
            encoded chunk (1, time, output_size), new att_cache, new cnn_cache.
        """
        assert xs.size(0) == 1
        tmp_masks = torch.ones(1, xs.size(1), device=xs.device, dtype=torch.bool).unsqueeze(1)
        if self.global_cmvn is not None:
            xs = self.global_cmvn(xs)
        xs, pos_emb, _ = self.embed(xs, tmp_masks, offset)
        if context.size(1) != 0:
            context_masks = torch.ones(
                1, 1, context.size(1), device=xs.device, dtype=torch.bool
            )
            context, _, _ = self.embed(context, context_masks, offset=offset + xs.size(1))
        xs = self.pre_lookahead_layer(xs, context=context)
        return self.forward_chunk_encoded(
            xs, offset, required_cache_size, att_cache, cnn_cache, att_mask
        )

    @torch.inference_mode()
    def forward_chunk_encoded(
        self,
        xs: torch.Tensor,
        offset: int,
        required_cache_size: int,
        att_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        cnn_cache: torch.Tensor = torch.zeros(0, 0, 0, 0),
        att_mask: torch.Tensor = None,
        num_decoding_left_chunks: int = -1,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward one chunk that is already embedded and pre-lookahead processed."""
        elayers, cache_t1 = att_cache.size(0), att_cache.size(2)
        attention_key_size = cache_t1 + xs.size(1)
        # Streaming calls position_encoding directly (not embed.forward), so extend
        # the relative-pos table when attention span exceeds the default max_len.
        self.embed.pos_enc.extend_pe(
            torch.zeros(1, attention_key_size, device=xs.device, dtype=xs.dtype)
        )
        pos_emb = self.embed.position_encoding(
            offset=0, size=attention_key_size
        )
        if att_mask is None:
            if self.static_chunk_size > 0:
                full_len = offset + xs.size(1)
                full_mask = subsequent_chunk_mask(
                    full_len,
                    self.static_chunk_size,
                    num_decoding_left_chunks,
                    xs.device,
                )
                att_mask = full_mask[offset:full_len, :full_len].unsqueeze(0)
            else:
                att_mask = torch.ones((0, 0, 0), dtype=torch.bool, device=xs.device)
        xs, r_att_cache, r_cnn_cache = self.forward_layers_chunk(
            xs, att_mask, pos_emb, att_cache, cnn_cache, required_cache_size
        )
        if self.normalize_before:
            xs = self.after_norm(xs)
        return xs, r_att_cache, r_cnn_cache
