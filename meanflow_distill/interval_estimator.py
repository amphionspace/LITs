"""Lightweight IntMeanFlow interval-conditioned estimator wrapper.

Kept separate from train_intmeanflow_distill.py so inference does not import
TensorBoard / TensorFlow through the training script.
"""

from __future__ import annotations

import torch


class IntervalConditionedEstimator(torch.nn.Module):
    """Wrap the LITS estimator with an IntMeanFlow-style interval condition.

    The pretrained estimator receives a single time embedding. The projection
    consumes start/end embeddings ``(r, t)`` and is initialized to select the
    original end-time embedding, preserving teacher behavior at init.
    """

    def __init__(self, base: torch.nn.Module):
        super().__init__()
        self.base = base
        in_channels = int(base.in_channels)
        self.interval_projector = torch.nn.Linear(2 * in_channels, in_channels)
        with torch.no_grad():
            self.interval_projector.weight.zero_()
            self.interval_projector.bias.zero_()
            self.interval_projector.weight[:, in_channels:].copy_(torch.eye(in_channels))

    def _interval_time_emb(self, r, t):
        t_emb = self.base.time_embeddings(t)
        r_emb = self.base.time_embeddings(r)
        pair = self.interval_projector(torch.cat([r_emb, t_emb], dim=-1))
        return self.base.time_mlp(pair)

    def forward(self, x, mask, mu, t, spks=None, cond=None, streaming=False, r=None):
        if r is None:
            r = t
        time_emb = self._interval_time_emb(r, t)
        return self.base.forward_core(x, mask, mu, time_emb, spks, cond, streaming)

    def forward_streaming(
        self,
        x_new,
        mask_new,
        mu_new,
        t,
        spks=None,
        cond=None,
        step_cache=None,
        r=None,
    ):
        """Streaming forward with interval time embedding (r, t)."""
        if r is None:
            r = t
        time_emb = self._interval_time_emb(r, t)
        return self.base.forward_core_streaming(
            x_new, mask_new, mu_new, time_emb, spks, cond, step_cache,
        )
