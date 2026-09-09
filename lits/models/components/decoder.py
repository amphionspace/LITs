import math
from typing import Optional, Tuple, List, Dict

import torch
import torch.nn as nn  # pylint: disable=consider-using-from-import
import torch.nn.functional as F
from conformer import ConformerBlock
from diffusers.models.activations import get_activation

from lits.models.components.transformer import BasicTransformerBlock
from lits.models.components.utils import (
    SinusoidalPosEmb,
    TimestepEmbedding,
    Block1D,
    ResnetBlock1D,
    Downsample1D,
    Upsample1D,
    CausalConv1d,
    CausalResnetBlock1D,
    CausalBlock1D,
    concat_channels,
    channels_to_time,
    time_to_channels,
    expand_spk_emb,
    build_decoder_attn_mask,
    build_streaming_decoder_attn_mask,
    decoder_kv_cache_limit,
    trim_decoder_kv_cache,
)


class ConformerWrapper(ConformerBlock):
    def __init__(  # pylint: disable=useless-super-delegation
        self,
        *,
        dim,
        dim_head=64,
        heads=8,
        ff_mult=4,
        conv_expansion_factor=2,
        conv_kernel_size=31,
        attn_dropout=0,
        ff_dropout=0,
        conv_dropout=0,
        conv_causal=False,
    ):
        super().__init__(
            dim=dim,
            dim_head=dim_head,
            heads=heads,
            ff_mult=ff_mult,
            conv_expansion_factor=conv_expansion_factor,
            conv_kernel_size=conv_kernel_size,
            attn_dropout=attn_dropout,
            ff_dropout=ff_dropout,
            conv_dropout=conv_dropout,
            conv_causal=conv_causal,
        )

    def forward(
        self,
        hidden_states,
        attention_mask,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        timestep=None,
    ):
        return super().forward(x=hidden_states, mask=attention_mask.bool())


class BaseDecoder(nn.Module):
    """
    Base class for all decoders. Handles block construction, weight initialization, and forward skeleton.
    Subclasses should implement build_blocks() and forward_core().
    """
    def __init__(self, in_channels, out_channels, channels, dropout, attention_head_dim, n_blocks, num_mid_blocks, num_heads, act_fn):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.channels = tuple(channels)
        self.dropout = dropout
        self.attention_head_dim = attention_head_dim
        self.n_blocks = n_blocks
        self.num_mid_blocks = num_mid_blocks
        self.num_heads = num_heads
        self.act_fn = act_fn
        self.time_embeddings = SinusoidalPosEmb(in_channels)
        self.time_embed_dim = self.channels[0] * 4
        self.time_mlp = TimestepEmbedding(
            in_channels=in_channels,
            time_embed_dim=self.time_embed_dim,
            act_fn="silu",
        )
        self.down_blocks = nn.ModuleList([])
        self.mid_blocks = nn.ModuleList([])
        self.up_blocks = nn.ModuleList([])
        self.final_block = None
        self.final_proj = None
        self.build_blocks()
        self.initialize_weights()

    def build_blocks(self):
        raise NotImplementedError

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv1d):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x, mask, mu, t, spks=None, cond=None, streaming=False):
        t_emb = self.time_embeddings(t)
        t_emb = self.time_mlp(t_emb)
        return self.forward_core(x, mask, mu, t_emb, spks, cond, streaming)

    def forward_core(self, x, mask, mu, t_emb, spks, cond, streaming):
        raise NotImplementedError

class Decoder(BaseDecoder):
    """Standard Decoder supporting conformer/transformer blocks."""
    def __init__(self, in_channels, out_channels, channels=(256, 256), dropout=0.05, attention_head_dim=64, n_blocks=1, num_mid_blocks=2, num_heads=4, act_fn="snake", down_block_type="transformer", mid_block_type="transformer", up_block_type="transformer"):
        self.down_block_type = down_block_type
        self.mid_block_type = mid_block_type
        self.up_block_type = up_block_type
        super().__init__(in_channels, out_channels, channels, dropout, attention_head_dim, n_blocks, num_mid_blocks, num_heads, act_fn)

    def get_block(self, block_type, dim, attention_head_dim, num_heads, dropout, act_fn):
        if block_type == "conformer":
            return ConformerWrapper(
                dim=dim,
                dim_head=attention_head_dim,
                heads=num_heads,
                ff_mult=1,
                conv_expansion_factor=2,
                ff_dropout=dropout,
                attn_dropout=dropout,
                conv_dropout=dropout,
                conv_kernel_size=31,
            )
        elif block_type == "transformer":
            return BasicTransformerBlock(
                dim=dim,
                num_attention_heads=num_heads,
                attention_head_dim=attention_head_dim,
                dropout=dropout,
                activation_fn=act_fn,
            )
        else:
            raise ValueError(f"Unknown block type {block_type}")

    def build_blocks(self):
        channels = self.channels
        time_embed_dim = self.time_embed_dim
        output_channel = self.in_channels
        for i in range(len(channels)):
            input_channel = output_channel
            output_channel = channels[i]
            is_last = i == len(channels) - 1
            resnet = ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
            transformer_blocks = nn.ModuleList([
                self.get_block(self.down_block_type, output_channel, self.attention_head_dim, self.num_heads, self.dropout, self.act_fn)
                for _ in range(self.n_blocks)
            ])
            downsample = Downsample1D(output_channel) if not is_last else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            self.down_blocks.append(nn.ModuleList([resnet, transformer_blocks, downsample]))
        for i in range(self.num_mid_blocks):
            input_channel = channels[-1]
            resnet = ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
            transformer_blocks = nn.ModuleList([
                self.get_block(self.mid_block_type, output_channel, self.attention_head_dim, self.num_heads, self.dropout, self.act_fn)
                for _ in range(self.n_blocks)
            ])
            self.mid_blocks.append(nn.ModuleList([resnet, transformer_blocks]))
        up_channels = channels[::-1] + (channels[0],)
        for i in range(len(up_channels) - 1):
            input_channel = up_channels[i]
            output_channel = up_channels[i + 1]
            is_last = i == len(up_channels) - 2
            resnet = ResnetBlock1D(dim=2 * input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
            transformer_blocks = nn.ModuleList([
                self.get_block(self.up_block_type, output_channel, self.attention_head_dim, self.num_heads, self.dropout, self.act_fn)
                for _ in range(self.n_blocks)
            ])
            upsample = Upsample1D(output_channel, use_conv_transpose=True) if not is_last else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            self.up_blocks.append(nn.ModuleList([resnet, transformer_blocks, upsample]))
        self.final_block = Block1D(up_channels[-1], up_channels[-1])
        self.final_proj = nn.Conv1d(up_channels[-1], self.out_channels, 1)

    def forward_core(self, x, mask, mu, t_emb, spks, cond, streaming):
        x = concat_channels(x, mu)
        if spks is not None:
            spks = expand_spk_emb(spks, x.shape[-1])
            x = concat_channels(x, spks)
        hiddens = []
        masks = [mask]
        for resnet, transformer_blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            x = resnet(x, mask_down, t_emb)
            x = channels_to_time(x)
            attn_mask = mask_down.squeeze(1)
            for transformer_block in transformer_blocks:
                x = transformer_block(hidden_states=x, attention_mask=attn_mask, timestep=t_emb)
            x = time_to_channels(x)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])
        masks = masks[:-1]
        mask_mid = masks[-1]
        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t_emb)
            x = channels_to_time(x)
            attn_mask = mask_mid.squeeze(1)
            for transformer_block in transformer_blocks:
                x = transformer_block(hidden_states=x, attention_mask=attn_mask, timestep=t_emb)
            x = time_to_channels(x)
        for resnet, transformer_blocks, upsample in self.up_blocks:
            mask_up = masks.pop()
            x = resnet(concat_channels(x, hiddens.pop()), mask_up, t_emb)
            x = channels_to_time(x)
            attn_mask = mask_up.squeeze(1)
            for transformer_block in transformer_blocks:
                x = transformer_block(hidden_states=x, attention_mask=attn_mask, timestep=t_emb)
            x = time_to_channels(x)
            x = upsample(x * mask_up)
        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up)
        return output * mask


class ConditionalDecoder(BaseDecoder):
    """Conditional Decoder supporting additional input conditions."""
    def build_blocks(self):
        channels = self.channels
        time_embed_dim = self.time_embed_dim
        output_channel = self.in_channels
        for i in range(len(channels)):
            input_channel = output_channel
            output_channel = channels[i]
            is_last = i == len(channels) - 1
            resnet = ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
            transformer_blocks = nn.ModuleList([
                BasicTransformerBlock(
                    dim=output_channel,
                    num_attention_heads=self.num_heads,
                    attention_head_dim=self.attention_head_dim,
                    dropout=self.dropout,
                    activation_fn=self.act_fn,
                ) for _ in range(self.n_blocks)
            ])
            downsample = Downsample1D(output_channel) if not is_last else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            self.down_blocks.append(nn.ModuleList([resnet, transformer_blocks, downsample]))
        for _ in range(self.num_mid_blocks):
            input_channel = channels[-1]
            resnet = ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
            transformer_blocks = nn.ModuleList([
                BasicTransformerBlock(
                    dim=output_channel,
                    num_attention_heads=self.num_heads,
                    attention_head_dim=self.attention_head_dim,
                    dropout=self.dropout,
                    activation_fn=self.act_fn,
                ) for _ in range(self.n_blocks)
            ])
            self.mid_blocks.append(nn.ModuleList([resnet, transformer_blocks]))
        up_channels = channels[::-1] + (channels[0],)
        for i in range(len(up_channels) - 1):
            input_channel = up_channels[i] * 2
            output_channel = up_channels[i + 1]
            is_last = i == len(up_channels) - 2
            resnet = ResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
            transformer_blocks = nn.ModuleList([
                BasicTransformerBlock(
                    dim=output_channel,
                    num_attention_heads=self.num_heads,
                    attention_head_dim=self.attention_head_dim,
                    dropout=self.dropout,
                    activation_fn=self.act_fn,
                ) for _ in range(self.n_blocks)
            ])
            upsample = Upsample1D(output_channel, use_conv_transpose=True) if not is_last else nn.Conv1d(output_channel, output_channel, 3, padding=1)
            self.up_blocks.append(nn.ModuleList([resnet, transformer_blocks, upsample]))
        self.final_block = Block1D(up_channels[-1], up_channels[-1])
        self.final_proj = nn.Conv1d(up_channels[-1], self.out_channels, 1)

    def _transformer_blocks(self, x, mask, transformer_blocks, t_emb, streaming):
        x = channels_to_time(x)
        attn_mask = build_decoder_attn_mask(
            x, mask, streaming=streaming, static_chunk_size=0
        )
        for transformer_block in transformer_blocks:
            x = transformer_block(hidden_states=x, attention_mask=attn_mask, timestep=t_emb)
        return time_to_channels(x)

    def forward_core(self, x, mask, mu, t_emb, spks, cond, streaming):
        x = concat_channels(x, mu)
        hiddens = []
        masks = [mask]
        for resnet, transformer_blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            x = resnet(x, mask_down, t_emb)
            x = self._transformer_blocks(x, mask_down, transformer_blocks, t_emb, streaming)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])
        masks = masks[:-1]
        mask_mid = masks[-1]
        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t_emb)
            x = self._transformer_blocks(x, mask_mid, transformer_blocks, t_emb, streaming)
        for resnet, transformer_blocks, upsample in self.up_blocks:
            mask_up = masks.pop()
            skip = hiddens.pop()
            x = concat_channels(x[:, :, :skip.shape[-1]], skip)
            x = resnet(x, mask_up, t_emb)
            x = self._transformer_blocks(x, mask_up, transformer_blocks, t_emb, streaming)
            x = upsample(x * mask_up)
        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up)
        return output * mask


class CausalConditionalDecoder(ConditionalDecoder):
    """Causal Conditional Decoder supporting streaming and causal blocks."""
    def __init__(self, in_channels, out_channels, channels=(256, 256), dropout=0.05, attention_head_dim=64, n_blocks=1, num_mid_blocks=2, num_heads=4, act_fn="snake", static_chunk_size=50, num_decoding_left_chunks=2, decoder_left_frames=-1):
        self.static_chunk_size = static_chunk_size
        self.num_decoding_left_chunks = num_decoding_left_chunks
        self.decoder_left_frames = decoder_left_frames
        super().__init__(in_channels, out_channels, channels, dropout, attention_head_dim, n_blocks, num_mid_blocks, num_heads, act_fn)

    def build_blocks(self):
        channels = self.channels
        time_embed_dim = self.time_embed_dim
        output_channel = self.in_channels
        for i in range(len(channels)):
            input_channel = output_channel
            output_channel = channels[i]
            is_last = i == len(channels) - 1
            resnet = CausalResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
            transformer_blocks = nn.ModuleList([
                BasicTransformerBlock(
                    dim=output_channel,
                    num_attention_heads=self.num_heads,
                    attention_head_dim=self.attention_head_dim,
                    dropout=self.dropout,
                    activation_fn=self.act_fn,
                ) for _ in range(self.n_blocks)
            ])
            downsample = Downsample1D(output_channel) if not is_last else CausalConv1d(output_channel, output_channel, 3)
            self.down_blocks.append(nn.ModuleList([resnet, transformer_blocks, downsample]))
        for _ in range(self.num_mid_blocks):
            input_channel = channels[-1]
            resnet = CausalResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
            transformer_blocks = nn.ModuleList([
                BasicTransformerBlock(
                    dim=output_channel,
                    num_attention_heads=self.num_heads,
                    attention_head_dim=self.attention_head_dim,
                    dropout=self.dropout,
                    activation_fn=self.act_fn,
                ) for _ in range(self.n_blocks)
            ])
            self.mid_blocks.append(nn.ModuleList([resnet, transformer_blocks]))
        up_channels = channels[::-1] + (channels[0],)
        for i in range(len(up_channels) - 1):
            input_channel = up_channels[i] * 2
            output_channel = up_channels[i + 1]
            is_last = i == len(up_channels) - 2
            resnet = CausalResnetBlock1D(dim=input_channel, dim_out=output_channel, time_emb_dim=time_embed_dim)
            transformer_blocks = nn.ModuleList([
                BasicTransformerBlock(
                    dim=output_channel,
                    num_attention_heads=self.num_heads,
                    attention_head_dim=self.attention_head_dim,
                    dropout=self.dropout,
                    activation_fn=self.act_fn,
                ) for _ in range(self.n_blocks)
            ])
            upsample = Upsample1D(output_channel, use_conv_transpose=True) if not is_last else CausalConv1d(output_channel, output_channel, 3)
            self.up_blocks.append(nn.ModuleList([resnet, transformer_blocks, upsample]))
        self.final_block = CausalBlock1D(up_channels[-1], up_channels[-1])
        self.final_proj = nn.Conv1d(up_channels[-1], self.out_channels, 1)

    def _transformer_blocks(self, x, mask, transformer_blocks, t_emb, streaming):
        x = channels_to_time(x)
        attn_mask = build_decoder_attn_mask(
            x,
            mask,
            streaming=streaming,
            static_chunk_size=self.static_chunk_size,
            num_decoding_left_chunks=-1,
            max_left_frames=self.decoder_left_frames,
        )
        for transformer_block in transformer_blocks:
            x = transformer_block(hidden_states=x, attention_mask=attn_mask, timestep=t_emb)
        return time_to_channels(x)

    def forward_core(self, x, mask, mu, t_emb, spks, cond, streaming):
        x = concat_channels(x, mu)
        if spks is not None:
            spks = expand_spk_emb(spks, x.shape[-1])
            x = concat_channels(x, spks)
        if cond is not None:
            x = concat_channels(x, cond)
        hiddens = []
        masks = [mask]
        for resnet, transformer_blocks, downsample in self.down_blocks:
            mask_down = masks[-1]
            x = resnet(x, mask_down, t_emb)
            x = self._transformer_blocks(x, mask_down, transformer_blocks, t_emb, streaming)
            hiddens.append(x)
            x = downsample(x * mask_down)
            masks.append(mask_down[:, :, ::2])
        masks = masks[:-1]
        mask_mid = masks[-1]
        for resnet, transformer_blocks in self.mid_blocks:
            x = resnet(x, mask_mid, t_emb)
            x = self._transformer_blocks(x, mask_mid, transformer_blocks, t_emb, streaming)
        for resnet, transformer_blocks, upsample in self.up_blocks:
            mask_up = masks.pop()
            skip = hiddens.pop()
            x = concat_channels(x[:, :, :skip.shape[-1]], skip)
            x = resnet(x, mask_up, t_emb)
            x = self._transformer_blocks(x, mask_up, transformer_blocks, t_emb, streaming)
            x = upsample(x * mask_up)
        x = self.final_block(x, mask_up)
        output = self.final_proj(x * mask_up)
        return output * mask

    # ------------------------------------------------------------------
    # Streaming decoder with per-ODE-step KV + conv-tail cache
    # ------------------------------------------------------------------

    def _transformer_blocks_streaming(
        self,
        x: torch.Tensor,
        transformer_blocks: nn.ModuleList,
        att_caches: List[Optional[torch.Tensor]],
        att_offsets: Optional[List[int]] = None,
    ) -> Tuple[torch.Tensor, list, list]:
        """Run transformer blocks with streaming KV-cache.

        Args:
            x: (B, C, T_new)  channel-major.
            att_caches: one KV-cache per transformer block (or None).
            att_offsets: global frame index of each cache's first key.

        Returns:
            (x_out, new_att_caches, new_att_offsets)
        """
        x = channels_to_time(x)
        new_caches: list = []
        new_offsets: list = []
        max_cache_size = decoder_kv_cache_limit(
            self.static_chunk_size, self.num_decoding_left_chunks,
            self.decoder_left_frames,
        )
        for i, tb in enumerate(transformer_blocks):
            cache = att_caches[i] if i < len(att_caches) and att_caches[i] is not None \
                else torch.zeros(0, 0, 0, 0, device=x.device)
            cache_offset = att_offsets[i] if att_offsets and i < len(att_offsets) else 0
            cached_len = cache.size(2) if cache.size(0) > 0 else 0
            attn_mask = build_streaming_decoder_attn_mask(
                x.size(1), cached_len,
                self.static_chunk_size, self.num_decoding_left_chunks,
                x.dtype, x.device,
                cache_offset=cache_offset,
                max_left_frames=self.decoder_left_frames,
            )
            x, new_cache = tb.forward_streaming(x, attention_mask=attn_mask, kv_cache=cache)
            new_cache, new_offset = trim_decoder_kv_cache(
                new_cache, max_cache_size, cache_offset,
            )
            new_caches.append(new_cache)
            new_offsets.append(new_offset)
        return time_to_channels(x), new_caches, new_offsets

    def forward_streaming(
        self,
        x_new: torch.Tensor,
        mask_new: torch.Tensor,
        mu_new: torch.Tensor,
        t: torch.Tensor,
        spks: Optional[torch.Tensor] = None,
        cond: Optional[torch.Tensor] = None,
        step_cache: Optional[dict] = None,
    ) -> Tuple[torch.Tensor, dict]:
        """Streaming forward: only process *new* frames using cached state."""
        t_emb = self.time_embeddings(t)
        t_emb = self.time_mlp(t_emb)
        return self.forward_core_streaming(x_new, mask_new, mu_new, t_emb, spks, cond, step_cache)

    def forward_core_streaming(
        self,
        x_new: torch.Tensor,
        mask_new: torch.Tensor,
        mu_new: torch.Tensor,
        t_emb: torch.Tensor,
        spks: Optional[torch.Tensor],
        cond: Optional[torch.Tensor],
        step_cache: Optional[dict],
    ) -> Tuple[torch.Tensor, dict]:
        """Core streaming forward with per-step cache management.

        *step_cache* layout (all lists are flat, indexed by component order)::

            att        – KV-cache per BasicTransformerBlock
            att_offset – global frame index of each att cache's first key
            conv       – tail-buffer per CausalConv1d
            ds         – tail-buffer per Downsample1D
            us         – tail-buffer per Upsample1D
        """
        att_c = step_cache.get('att', []) if step_cache else []
        att_off = step_cache.get('att_offset', []) if step_cache else []
        conv_c = step_cache.get('conv', []) if step_cache else []
        ds_c = step_cache.get('ds', []) if step_cache else []
        us_c = step_cache.get('us', []) if step_cache else []

        ai = ci = di = ui = 0
        na: list = []
        naoff: list = []
        nc: list = []
        nd: list = []
        nu: list = []

        # ---------- prepare input channels ----------
        x = concat_channels(x_new, mu_new)
        if spks is not None:
            x = concat_channels(x, expand_spk_emb(spks, x.shape[-1]))
        if cond is not None:
            x = concat_channels(x, cond)

        new_skips: list = []
        masks = [mask_new]

        # ---------- down path ----------
        for i, (resnet, tblocks, downsample) in enumerate(self.down_blocks):
            is_last = (i == len(self.down_blocks) - 1)
            mask = masks[-1]

            bufs = [conv_c[ci] if ci < len(conv_c) else None,
                    conv_c[ci + 1] if ci + 1 < len(conv_c) else None]
            x, nb = resnet.forward_streaming(x, mask, t_emb, bufs)
            nc.extend(nb); ci += 2

            ac = [att_c[ai + j] if ai + j < len(att_c) else None for j in range(len(tblocks))]
            aoff = [att_off[ai + j] if ai + j < len(att_off) else 0 for j in range(len(tblocks))]
            x, nac, nacoff = self._transformer_blocks_streaming(x, tblocks, ac, aoff)
            na.extend(nac); naoff.extend(nacoff); ai += len(tblocks)

            new_skips.append(x)

            if not is_last:
                db = ds_c[di] if di < len(ds_c) else None
                x, ndb = downsample.forward_streaming(x * mask, db)
                nd.append(ndb); di += 1
            else:
                cb = conv_c[ci] if ci < len(conv_c) else None
                x, ncb = downsample.forward_streaming(x * mask, cb)
                nc.append(ncb); ci += 1

            masks.append(mask[:, :, ::2])

        masks = masks[:-1]
        mask_mid = masks[-1]

        # ---------- mid path ----------
        for resnet, tblocks in self.mid_blocks:
            bufs = [conv_c[ci] if ci < len(conv_c) else None,
                    conv_c[ci + 1] if ci + 1 < len(conv_c) else None]
            x, nb = resnet.forward_streaming(x, mask_mid, t_emb, bufs)
            nc.extend(nb); ci += 2

            ac = [att_c[ai + j] if ai + j < len(att_c) else None for j in range(len(tblocks))]
            aoff = [att_off[ai + j] if ai + j < len(att_off) else 0 for j in range(len(tblocks))]
            x, nac, nacoff = self._transformer_blocks_streaming(x, tblocks, ac, aoff)
            na.extend(nac); naoff.extend(nacoff); ai += len(tblocks)

        # ---------- up path ----------
        for i, (resnet, tblocks, upsample) in enumerate(self.up_blocks):
            is_last = (i == len(self.up_blocks) - 1)
            mask_up = masks.pop()
            skip = new_skips.pop()

            x = concat_channels(x[:, :, :skip.shape[-1]], skip)

            bufs = [conv_c[ci] if ci < len(conv_c) else None,
                    conv_c[ci + 1] if ci + 1 < len(conv_c) else None]
            x, nb = resnet.forward_streaming(x, mask_up, t_emb, bufs)
            nc.extend(nb); ci += 2

            ac = [att_c[ai + j] if ai + j < len(att_c) else None for j in range(len(tblocks))]
            aoff = [att_off[ai + j] if ai + j < len(att_off) else 0 for j in range(len(tblocks))]
            x, nac, nacoff = self._transformer_blocks_streaming(x, tblocks, ac, aoff)
            na.extend(nac); naoff.extend(nacoff); ai += len(tblocks)

            if not is_last:
                ub = us_c[ui] if ui < len(us_c) else None
                x, nub = upsample.forward_streaming(x * mask_up, ub)
                nu.append(nub); ui += 1
            else:
                cb = conv_c[ci] if ci < len(conv_c) else None
                x, ncb = upsample.forward_streaming(x * mask_up, cb)
                nc.append(ncb); ci += 1

        # ---------- final ----------
        fb = conv_c[ci] if ci < len(conv_c) else None
        x, nfb = self.final_block.forward_streaming(x, mask_new, fb)
        nc.append(nfb)

        output = self.final_proj(x * mask_new)

        new_cache = {'att': na, 'att_offset': naoff, 'conv': nc, 'ds': nd, 'us': nu}
        return output * mask_new, new_cache
