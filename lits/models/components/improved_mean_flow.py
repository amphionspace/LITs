"""Conditional Mel iMF adapted from Lyy-iiis/imeanflow's official implementation.

Training: main@bf60cd7cb653f6628e59d48034b333c5eba445e2, imf.py.
Sampling: torch@04687983e821b3ad01f54f03dc44194a33c20c54, imf.py.
The upstream MIT license is in training/imf/UPSTREAM_LICENSE.

The objective uses the upstream data->noise time convention (t=1 is noise).
The LITs decoder/cache interface keeps noise->data time s=1-t and positive
velocity, so an FM checkpoint can initialize every existing acoustic tensor.
Guidance is the exact omega=1, no-condition-dropout specialization of upstream.
"""

from copy import deepcopy
import math
import random

import torch
from torch import nn
from torch.nn.attention import SDPBackend, sdpa_kernel

from lits.models.components.decoder import CausalConditionalDecoder
from lits.models.components.flow_matching import CFM_Causal
from lits.models.components.utils import TimestepEmbedding, concat_channels, expand_spk_emb


class IMFCausalEstimator(CausalConditionalDecoder):
    """FM-compatible trunk and u decoder, with an interval MLP and auxiliary v tail.

    The down/mid blocks are shared; each head has its own up blocks and final
    projection. Only the u head executes during inference, including cached calls.
    All existing FM state-dict names remain unchanged.
    """

    def __init__(self, *args, interval_time_scale=1.0, **kwargs):
        # Upstream has no network dropout. A deterministic estimator also makes
        # the detached JVP and gradient-enabled forward evaluate the same field.
        kwargs['dropout'] = 0.0
        super().__init__(*args, **kwargs)
        self.interval_time_scale = float(interval_time_scale)
        if not math.isfinite(self.interval_time_scale) or self.interval_time_scale <= 0:
            raise ValueError('interval_time_scale must be finite and positive')
        self.interval_projector = TimestepEmbedding(self.in_channels, self.time_embed_dim)
        nn.init.zeros_(self.interval_projector.linear_2.weight)
        nn.init.zeros_(self.interval_projector.linear_2.bias)
        self.v_up_blocks = deepcopy(self.up_blocks)
        self.v_final_block = deepcopy(self.final_block)
        self.v_final_proj = deepcopy(self.final_proj)

    def interval_embedding(self, start, end):
        return (self.time_mlp(self.time_embeddings(start))
                + self.interval_projector(self.time_embeddings(end - start, scale=self.interval_time_scale)))

    def forward(self, x, mask, mu, t, spks=None, cond=None, streaming=False, r=None):
        # This is the existing LITs interval-estimator calling convention:
        # x is the state at r (start), and t is the destination.
        start = t if r is None else r
        emb = self.interval_embedding(start, t)
        return self.forward_core(x, mask, mu, emb, spks, cond, streaming)

    def forward_streaming(self, x_new, mask_new, mu_new, t, spks=None, cond=None,
                          step_cache=None, r=None):
        start = t if r is None else r
        emb = self.interval_embedding(start, t)
        return self.forward_core_streaming(x_new, mask_new, mu_new, emb, spks, cond, step_cache)

    def forward_uv(self, x, mask, mu, t, r, spks=None, cond=None, streaming=False):
        """Return (u, v) in upstream's time/sign convention, with t >= r."""
        emb = self.interval_embedding((1 - t).reshape(-1), (1 - r).reshape(-1))
        hidden = concat_channels(x, mu)
        if spks is not None:
            hidden = concat_channels(hidden, expand_spk_emb(spks, hidden.shape[-1]))
        if cond is not None:
            hidden = concat_channels(hidden, cond)
        skips, masks = [], [mask]
        for resnet, blocks, downsample in self.down_blocks:
            current_mask = masks[-1]
            hidden = resnet(hidden, current_mask, emb)
            hidden = self._transformer_blocks(hidden, current_mask, blocks, emb, streaming)
            skips.append(hidden)
            hidden = downsample(hidden * current_mask)
            masks.append(current_mask[:, :, ::2])
        masks = masks[:-1]
        for resnet, blocks in self.mid_blocks:
            hidden = resnet(hidden, masks[-1], emb)
            hidden = self._transformer_blocks(hidden, masks[-1], blocks, emb, streaming)

        def head(up_blocks, final_block, final_proj):
            out = hidden
            for (resnet, blocks, upsample), skip, current_mask in zip(
                    up_blocks, reversed(skips), reversed(masks)):
                out = concat_channels(out[:, :, :skip.shape[-1]], skip)
                out = resnet(out, current_mask, emb)
                out = self._transformer_blocks(out, current_mask, blocks, emb, streaming)
                out = upsample(out * current_mask)
            out = final_block(out, mask)
            # Reversing time reverses velocity. Keep stored weights in FM sign.
            return -final_proj(out * mask) * mask

        return (head(self.up_blocks, self.final_block, self.final_proj),
                head(self.v_up_blocks, self.v_final_block, self.v_final_proj))


def sample_tr(batch_size, device, p_mean=-0.4, p_std=1.0, data_proportion=0.5):
    """Port of upstream sample_tr: ordered logit normals and a diagonal FM slice."""
    pair = torch.sigmoid(torch.randn(2, batch_size, 1, 1, device=device) * p_std + p_mean)
    t, r = pair.max(dim=0).values, pair.min(dim=0).values
    fm_mask = (torch.arange(batch_size, device=device) < int(batch_size * data_proportion))[:, None, None]
    return t, torch.where(fm_mask, t, r), fm_mask


def imf_loss(uv_fn, x, mask, t, r, noise, norm_p=1.0, norm_eps=0.01):
    """Official dual-head iMF loss, specialized to omega=1 and variable-length Mel.

uv_fn returns (u, v); t/r and JVP directions use upstream's data->noise time.
Conditions are closure variables, hence held fixed in the directional derivative.
The separate grad-enabled forward preserves gradients to the condition encoders.
"""
    x, mask, noise = x.float(), mask.float(), noise.float()
    z = ((1 - t) * x + t * noise) * mask
    target = (noise - x) * mask
    with torch.no_grad():
        # The tangent is the *predicted* v at h=0, never the conditional target.
        _, tangent = uv_fn(z, t, t)
        _, du_dt, _ = torch.func.jvp(
            uv_fn, (z, t, r), (tangent.detach(), torch.ones_like(t), torch.zeros_like(r)),
            has_aux=True,
        )
    u, v = uv_fn(z, t, r)
    compound = u.float() + (t - r) * du_dt.detach().float()
    error_u = ((compound - target).square() * mask).sum(dim=(1, 2))
    error_v = ((v.float() - target).square() * mask).sum(dim=(1, 2))

    def adaptive(error):
        return error / (error + norm_eps).pow(norm_p).detach()

    loss_u, loss_v = adaptive(error_u), adaptive(error_v)
    valid = (mask.sum(dim=(1, 2)) * x.shape[1]).clamp_min(1)
    stats = {
        'u_mse': (error_u / valid).mean().detach(),
        'v_mse': (error_v / valid).mean().detach(),
        'u_weighted': loss_u.mean().detach(),
        'v_weighted': loss_v.mean().detach(),
        'jvp_rms': ((du_dt.float().square() * mask).sum(dim=(1, 2)) / valid).mean().sqrt().detach(),
        'fm_fraction': (t == r).float().mean().detach(),
    }
    return (loss_u + loss_v).mean(), z, stats


class IMF_Causal(CFM_Causal):
    """Opt-in iMF objective and interval sampler; original CFM remains independent."""

    objective = 'imf'

    def __init__(self, in_channels, out_channel, cfm_params, decoder_params,
                 n_spks=1, spk_emb_dim=64, pre_lookahead_len=3, streaming=True):
        if float(cfm_params.get('sigma_min', 0.0)) != 0.0:
            raise ValueError('iMF uses the upstream exact linear path; sigma_min must be 0')
        if float(cfm_params.get('guidance_scale', 1.0)) != 1.0:
            raise ValueError('LITs iMF currently supports guidance_scale=1 only (no CFG)')
        super().__init__(in_channels, out_channel, cfm_params, decoder_params,
                         n_spks, spk_emb_dim, pre_lookahead_len, streaming)
        original = self.estimator
        self.estimator = IMFCausalEstimator(
            in_channels=original.in_channels, out_channels=original.out_channels,
            channels=original.channels, attention_head_dim=original.attention_head_dim,
            n_blocks=original.n_blocks, num_mid_blocks=original.num_mid_blocks,
            num_heads=original.num_heads, act_fn=original.act_fn,
            static_chunk_size=original.static_chunk_size,
            num_decoding_left_chunks=original.num_decoding_left_chunks,
            decoder_left_frames=original.decoder_left_frames,
            # Old checkpoints predate this field and used the FM scale for h.
            # New recipes explicitly select normalized interval time (scale 1).
            interval_time_scale=cfm_params.get('interval_time_scale', 1000.0),
        )
        # Preserve even the initial random FM trunk; warm-start loading later
        # also initializes the auxiliary tail from the loaded FM tail.
        self.estimator.load_state_dict(self.expand_estimator_state(original.state_dict()), strict=True)
        self.p_mean = float(cfm_params.get('P_mean', -0.4))
        self.p_std = float(cfm_params.get('P_std', 1.0))
        self.data_proportion = float(cfm_params.get('data_proportion', 0.5))
        self.norm_p = float(cfm_params.get('norm_p', 1.0))
        self.norm_eps = float(cfm_params.get('norm_eps', 0.01))
        self.default_n_timesteps = int(cfm_params.get('num_steps', 2))
        self.sampling_time_grid = tuple(float(x) for x in cfm_params.get('sampling_time_grid', [0.0, 0.5, 1.0]))
        if not (math.isfinite(self.p_mean) and math.isfinite(self.p_std) and self.p_std > 0):
            raise ValueError('P_mean must be finite and P_std must be positive')
        if not 0 <= self.data_proportion < 1:
            raise ValueError('data_proportion must be in [0, 1); iMF requires off-diagonal samples')
        if not (math.isfinite(self.norm_p) and self.norm_p >= 0 and math.isfinite(self.norm_eps) and self.norm_eps > 0):
            raise ValueError('norm_p must be nonnegative and norm_eps must be positive')
        grid = self.sampling_time_grid
        if (self.default_n_timesteps < 1 or len(grid) != self.default_n_timesteps + 1
                or grid[0] != 0 or grid[-1] != 1 or not all(a < b for a, b in zip(grid, grid[1:]))):
            raise ValueError('sampling_time_grid must increase from 0 to 1 with num_steps intervals')
        self.last_loss_stats = {}
        self._imf_cache_grid = None

    def expand_estimator_state(self, fm_state):
        """Expand a complete FM estimator state; fail on unrelated missing keys."""
        result = {}
        expected_fm = {k for k in self.estimator.state_dict()
                       if not k.startswith(('interval_projector.', 'v_up_blocks.', 'v_final_block.', 'v_final_proj.'))}
        if set(fm_state) != expected_fm:
            raise ValueError(f'FM estimator keys differ: missing={sorted(expected_fm - set(fm_state))}, '
                             f'unexpected={sorted(set(fm_state) - expected_fm)}')
        for name, value in self.estimator.state_dict().items():
            if name.startswith('interval_projector.'):
                result[name] = value
            else:
                source_name = name[2:] if name.startswith(('v_up_blocks.', 'v_final_block.', 'v_final_proj.')) else name
                result[name] = fm_state[source_name]
        return result

    def compute_loss(self, x1, mask, mu, spks=None, cond=None):
        streaming = random.random() < 0.5
        mu = self.encoder(mu, mask, streaming=streaming)
        grad_enabled = torch.is_grad_enabled()
        # Lightning validation normally enables inference_mode, which disables
        # forward AD. Clone its tensors inside a normal tensor context for JVP.
        with torch.inference_mode(False), torch.set_grad_enabled(grad_enabled), \
                torch.autocast(device_type=x1.device.type, enabled=False), sdpa_kernel(SDPBackend.MATH):
            def normal(tensor):
                if tensor is None:
                    return None
                tensor = tensor.clone() if torch.is_inference(tensor) else tensor
                return tensor.float()

            x1, mask, mu, spks, cond = map(normal, (x1, mask, mu, spks, cond))
            t, r, _ = sample_tr(x1.shape[0], x1.device, self.p_mean, self.p_std, self.data_proportion)
            noise = torch.randn_like(x1)

            def uv_fn(z, time, end):
                return self.estimator.forward_uv(z, mask, mu, time, end, spks, cond, streaming)

            loss, z, self.last_loss_stats = imf_loss(uv_fn, x1, mask, t, r, noise, self.norm_p, self.norm_eps)
        return loss, z

    @torch.inference_mode()
    def forward(self, mu, mask, n_timesteps=None, finalize=True, temperature=1.0,
                spks=None, cond=None, streaming=False, z=None, chunk_start=0):
        steps = self.default_n_timesteps if n_timesteps is None else n_timesteps
        if not isinstance(steps, int) or isinstance(steps, bool) or steps < 1:
            raise ValueError('n_timesteps must be a positive integer')
        return super().forward(mu, mask, steps, finalize, temperature, spks, cond,
                               streaming, z, chunk_start)

    def solve_euler(self, x, t_span, mu, mask, spks, cond, streaming,
                    chunk_start=0, use_kv_cache=None):
        # Parent interval dispatch executes u only, once per interval, and keeps
        # separate convolution/KV caches for each sampling step.
        if len(t_span) - 1 == self.default_n_timesteps:
            t_span = t_span.new_tensor(self.sampling_time_grid)
        grid = tuple(t_span.detach().cpu().tolist())
        if chunk_start > 0 and self._decoder_caches is not None and grid != self._imf_cache_grid:
            raise ValueError('Cannot change the iMF time grid during a cached utterance; reset caches first')
        self._imf_cache_grid = grid
        return super().solve_euler(x, t_span, mu, mask, spks, cond, streaming,
                                   chunk_start, use_kv_cache) * mask
