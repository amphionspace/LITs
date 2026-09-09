from abc import ABC
import random
import torch
import torch.nn.functional as F
from lits.models.components.decoder import Decoder, CausalConditionalDecoder
from lits.utils.pylogger import get_pylogger
from lits.models.components.utils import ConformerEncoder


log = get_pylogger(__name__)


def _estimator_supports_interval(estimator) -> bool:
    return hasattr(estimator, "interval_projector")


def _euler_step_nonstreaming(estimator, x, mask, mu, spks, cond, streaming, r, t_end, dt):
    if _estimator_supports_interval(estimator):
        dphi_dt = estimator(x, mask, mu, t_end, spks, cond, streaming, r=r)
    else:
        dphi_dt = estimator(x, mask, mu, r, spks, cond, streaming)
    return x + dt * dphi_dt


def _euler_step_streaming(estimator, x, mask, mu, spks, cond, r, t_end, dt, step_cache):
    if _estimator_supports_interval(estimator):
        dphi_dt, sc = estimator.forward_streaming(
            x, mask, mu, t_end, spks, cond, step_cache=step_cache, r=r,
        )
    else:
        dphi_dt, sc = estimator.forward_streaming(
            x, mask, mu, r, spks, cond, step_cache=step_cache,
        )
    return x + dt * dphi_dt, sc

class BASECFM(torch.nn.Module, ABC):
    """
    Base class for Conditional Flow Matching (CFM) models.
    Handles diffusion process and loss computation.
    """
    def __init__(
        self,
        n_feats: int,
        cfm_params,
        n_spks: int = 1,
        spk_emb_dim: int = 128,
        streaming: bool = False,
    ):
        super().__init__()
        self.n_feats = n_feats
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.solver = cfm_params.solver
        self.streaming = streaming
        self.sigma_min = getattr(cfm_params, "sigma_min", 1e-4)
        self.estimator = None

    @torch.inference_mode()
    def forward(self, mu: torch.Tensor, mask: torch.Tensor, n_timesteps: int, temperature: float = 1.0, spks: torch.Tensor = None, cond=None, streaming=False) -> torch.Tensor:
        """
        Forward diffusion process.
        Args:
            mu: (batch_size, n_feats, mel_timesteps)
            mask: (batch_size, 1, mel_timesteps)
            n_timesteps: number of diffusion steps
            temperature: noise scale
            spks: (batch_size, spk_emb_dim), optional
            cond: reserved for future use
        Returns:
            sample: (batch_size, n_feats, mel_timesteps)
        """
        z = torch.randn_like(mu) * temperature
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        return self.solve_euler(z, t_span=t_span, mu=mu, mask=mask, spks=spks, cond=cond, streaming=streaming)

    def solve_euler(self, x: torch.Tensor, t_span: torch.Tensor, mu: torch.Tensor, mask: torch.Tensor, spks: torch.Tensor, cond, streaming) -> torch.Tensor:
        """
        Fixed Euler solver for ODEs.
        Args:
            x: random noise
            t_span: (n_timesteps + 1,)
            mu: (batch_size, n_feats, mel_timesteps)
            mask: (batch_size, 1, mel_timesteps)
            spks: (batch_size, spk_emb_dim), optional
            cond: reserved for future use
        Returns:
            Final solution tensor
        """
        t, _, dt = t_span[0], t_span[-1], t_span[1] - t_span[0]
        sol = []
        for step in range(1, len(t_span)):
            dphi_dt = self.estimator(x, mask, mu, t, spks, cond, streaming)
            x = x + dt * dphi_dt
            t = t + dt
            sol.append(x)
            if step < len(t_span) - 1:
                dt = t_span[step + 1] - t
        return sol[-1]

    def compute_loss(self, x1: torch.Tensor, mask: torch.Tensor, mu: torch.Tensor, spks: torch.Tensor = None, cond=None):
        """
        Computes diffusion loss.
        Args:
            x1: Target (batch_size, n_feats, mel_timesteps)
            mask: (batch_size, 1, mel_timesteps)
            mu: (batch_size, n_feats, mel_timesteps)
            spks: (batch_size, spk_emb_dim), optional
            cond: reserved for future use
        Returns:
            loss: conditional flow matching loss
            y: conditional flow (batch_size, n_feats, mel_timesteps)
        """

class CFM(BASECFM):
    """
    Conditional Flow Matching model with Decoder estimator.
    """
    def __init__(self, in_channels: int, out_channel: int, cfm_params, decoder_params, n_spks: int = 1, spk_emb_dim: int = 64, streaming: bool = False):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
            streaming=streaming,
        )
        in_channels = in_channels + (spk_emb_dim if n_spks > 1 else 0)
        self.estimator = Decoder(in_channels=in_channels, out_channels=out_channel, **decoder_params)

    def compute_loss(self, x1: torch.Tensor, mask: torch.Tensor, mu: torch.Tensor, spks: torch.Tensor = None, cond=None):
        """
        Computes diffusion loss.
        Args:
            x1: Target (batch_size, n_feats, mel_timesteps)
            mask: (batch_size, 1, mel_timesteps)
            mu: (batch_size, n_feats, mel_timesteps)
            spks: (batch_size, spk_emb_dim), optional
            cond: reserved for future use
        Returns:
            loss: conditional flow matching loss
            y: conditional flow (batch_size, n_feats, mel_timesteps)
        """
        b, _, t = mu.shape

        t_rand = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        z = torch.randn_like(x1)
        y = (1 - (1 - self.sigma_min) * t_rand) * z + t_rand * x1
        u = x1 - (1 - self.sigma_min) * z
        pred = self.estimator(y, mask, mu, t_rand.squeeze(), spks, cond)
        loss = F.mse_loss(pred, u, reduction="sum") / (torch.sum(mask) * u.shape[1])
        return loss, y

class CFM_Causal(BASECFM):
    """
    Conditional Flow Matching model with Causal Decoder estimator.
    """
    def __init__(self, in_channels: int, out_channel: int, cfm_params, decoder_params, n_spks: int = 1, spk_emb_dim: int = 64, pre_lookahead_len: int = 3, streaming: bool = True):
        super().__init__(
            n_feats=in_channels,
            cfm_params=cfm_params,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
            streaming=streaming,
        )
        in_channels = in_channels + (spk_emb_dim if n_spks > 1 else 0)
        self.pre_lookahead_len = pre_lookahead_len
        decoder_params = dict(decoder_params)
        legacy_mu_encoder_left_chunks = decoder_params.pop("mu_encoder_left_chunks", None)
        encoder_left_chunks = decoder_params.pop("encoder_num_decoding_left_chunks", None)
        self.estimator = CausalConditionalDecoder(in_channels=in_channels, out_channels=out_channel, **decoder_params)
        self.num_decoding_left_chunks = decoder_params.get(
            "num_decoding_left_chunks",
            self.estimator.num_decoding_left_chunks,
        )
        self.estimator.num_decoding_left_chunks = self.num_decoding_left_chunks
        self.decoder_left_frames = decoder_params.get(
            "decoder_left_frames",
            getattr(self.estimator, "decoder_left_frames", -1),
        )
        self.estimator.decoder_left_frames = self.decoder_left_frames
        # encoder_num_decoding_left_chunks controls how many left chunks the mu
        # encoder (ConformerEncoder) attends to during streaming inference.
        # Defaults to the same value as the decoder but can be overridden at
        # inference time (e.g. set to -1 for unlimited encoder context while
        # keeping the decoder bounded) without re-training.
        if encoder_left_chunks is None:
            encoder_left_chunks = legacy_mu_encoder_left_chunks
        if encoder_left_chunks is None:
            encoder_left_chunks = self.num_decoding_left_chunks
        self.encoder_num_decoding_left_chunks = encoder_left_chunks
        self.encoder = ConformerEncoder(
            input_size=out_channel,
            output_size=out_channel,
            attention_heads=decoder_params["num_heads"],
            linear_units=2048,
            num_blocks=6,
            dropout_rate=0.1,
            positional_dropout_rate=0.1,
            attention_dropout_rate=0.1,
            normalize_before=True,
            use_cnn_module=False,
            macaron_style=False,
            static_chunk_size=decoder_params["static_chunk_size"],
        )
        self.reset_encoder_cache()

    def reset_encoder_cache(self) -> None:
        """Clear ConformerEncoder KV-cache and Decoder streaming caches."""
        self._enc_att_cache = torch.zeros(0, 0, 0, 0)
        self._enc_cnn_cache = torch.zeros(0, 0, 0, 0)
        self._enc_offset = 0
        self._encoded_mu = None
        self._decoder_caches = None

    def _compute_ode_window_start(self, chunk_start: int) -> int:
        """Left index of the ODE window for one streaming chunk.

        ``decoder_left_frames`` uses a frame-level left window and overrides
        the legacy chunk-level ``num_decoding_left_chunks`` setting.
        """
        left_frames = getattr(self.estimator, "decoder_left_frames", -1)
        if left_frames >= 0:
            return max(0, chunk_start - left_frames)
        if self.num_decoding_left_chunks < 0:
            return 0
        static = self.estimator.static_chunk_size
        max_left = static * self.num_decoding_left_chunks
        return max(0, chunk_start - max_left)

    def encode_mu(
        self,
        mu: torch.Tensor,
        mask: torch.Tensor,
        *,
        context: torch.Tensor = None,
        context_embedded: bool = False,
        finalize: bool = False,
        streaming: bool = False,
    ) -> torch.Tensor:
        """Encode duration-aligned hidden states through ConformerEncoder."""
        if streaming and not self.training:
            return self._encode_mu_streaming(mu, mask, context, context_embedded, finalize)
        if finalize:
            return self.encoder(mu, mask, streaming=streaming)
        if context is None:
            mu_body, context = mu[:, :, :-self.pre_lookahead_len], mu[:, :, -self.pre_lookahead_len:]
        else:
            mu_body = mu
        return self.encoder(mu_body, mask, context=context, streaming=streaming)

    def _encode_mu_streaming(
        self,
        mu: torch.Tensor,
        mask: torch.Tensor,
        context: torch.Tensor,
        context_embedded: bool,
        finalize: bool,
    ) -> torch.Tensor:
        if finalize:
            mu_main = mu
            context_raw = torch.zeros(0, 0, 0, device=mu.device, dtype=mu.dtype)
            context_embedded = False
        elif context is None:
            mu_main = mu[:, :, :-self.pre_lookahead_len]
            context_raw = mu[:, :, -self.pre_lookahead_len:]
            context_embedded = False
        else:
            mu_main = mu
            context_raw = context

        cur_len = mu_main.shape[-1]
        if cur_len == 0:
            return mu_main

        if cur_len < self._enc_offset:
            return self._encoded_mu[:, :, :cur_len]

        if cur_len == self._enc_offset:
            return self._encoded_mu[:, :, :cur_len]

        overlap = self.encoder.pre_lookahead_layer.pre_lookahead_len
        start = self._enc_offset
        chunk_start = max(0, start - overlap)
        chunk_mu = mu_main[:, :, chunk_start:cur_len]

        if self._enc_att_cache.numel() == 0:
            device = mu.device
            self._enc_att_cache = torch.zeros(0, 0, 0, 0, device=device)
            self._enc_cnn_cache = torch.zeros(0, 0, 0, 0, device=device)

        static_chunk = self.encoder.static_chunk_size
        enc_left = self.encoder_num_decoding_left_chunks
        if enc_left < 0:
            required_cache_size = -1
        else:
            required_cache_size = static_chunk * enc_left

        chunk_mu_t = chunk_mu.transpose(2, 1)
        context_t = (
            context_raw.transpose(2, 1)
            if context_raw.numel() > 0
            else torch.zeros(0, 0, 0, device=mu.device, dtype=mu.dtype)
        )

        tmp_masks = torch.ones(
            1, chunk_mu_t.size(1), device=mu.device, dtype=torch.bool
        ).unsqueeze(1)
        xs = chunk_mu_t
        if self.encoder.global_cmvn is not None:
            xs = self.encoder.global_cmvn(xs)
        xs, _, _ = self.encoder.embed(xs, tmp_masks, offset=chunk_start)
        if context_embedded and context_raw.numel() > 0:
            context_emb = context_raw.transpose(2, 1).contiguous()
        elif context_t.size(1) != 0:
            context_masks = torch.ones(
                1, 1, context_t.size(1), device=mu.device, dtype=torch.bool
            )
            context_emb, _, _ = self.encoder.embed(
                context_t, context_masks, offset=chunk_start + xs.size(1)
            )
        else:
            context_emb = torch.zeros(0, 0, 0, device=mu.device, dtype=mu.dtype)
        xs = self.encoder.pre_lookahead_layer(xs, context=context_emb)
        xs_new = xs[:, start - chunk_start:, :]

        y, self._enc_att_cache, self._enc_cnn_cache = self.encoder.forward_chunk_encoded(
            xs_new,
            offset=start,
            required_cache_size=required_cache_size,
            att_cache=self._enc_att_cache,
            cnn_cache=self._enc_cnn_cache,
            num_decoding_left_chunks=enc_left,
        )
        new_frames = y.transpose(2, 1)
        if self._encoded_mu is None:
            self._encoded_mu = new_frames
        else:
            self._encoded_mu = torch.cat([self._encoded_mu, new_frames], dim=-1)
        self._enc_offset = cur_len
        return self._encoded_mu[:, :, :cur_len]

    def solve_euler(
        self,
        x: torch.Tensor,
        t_span: torch.Tensor,
        mu: torch.Tensor,
        mask: torch.Tensor,
        spks: torch.Tensor,
        cond,
        streaming: bool,
        chunk_start: int = 0,
        use_kv_cache: bool | None = None,
    ) -> torch.Tensor:
        """Euler ODE solver with streaming decoder KV / conv-tail cache.

        On the **first chunk** (``chunk_start == 0``), the full forward is run
        through ``estimator.forward_streaming`` so that per-step caches are
        saved.  On subsequent chunks the cached K/V and conv-tail buffers allow
        the decoder to process *only* the new frames – eliminating the
        quadratic recomputation that the old ODE-window approach could only
        partially mitigate.

        Falls back to the ODE-window heuristic when the decoder cache is
        unavailable (e.g. after a mid-utterance reset).

        Args:
            use_kv_cache: When True, always use the KV-cache streaming path
                (including during training).  Default: ``streaming and not
                self.training``.
        """
        if use_kv_cache is None:
            use_kv_cache = streaming and not self.training
        cur_len = mu.shape[-1]
        batch_size, n_feats = x.shape[0], x.shape[1]

        # ------ cached path: subsequent chunks ------
        if (streaming and use_kv_cache and chunk_start > 0
                and self._decoder_caches is not None):
            x_new = x[:, :, chunk_start:].contiguous()
            mu_new = mu[:, :, chunk_start:].contiguous()
            mask_new = mask[:, :, chunk_start:].contiguous()

            new_caches = []
            for step in range(len(t_span) - 1):
                r = t_span[step]
                t_end = t_span[step + 1]
                dt = t_end - r
                x_new, sc = _euler_step_streaming(
                    self.estimator,
                    x_new,
                    mask_new,
                    mu_new,
                    spks,
                    cond,
                    r,
                    t_end,
                    dt,
                    self._decoder_caches[step],
                )
                new_caches.append(sc)

            self._decoder_caches = new_caches

            full_x = torch.zeros(batch_size, n_feats, cur_len,
                                 device=x.device, dtype=x.dtype)
            full_x[:, :, chunk_start:] = x_new
            return full_x

        # ------ first chunk: build caches ------
        win_start = 0
        if streaming and use_kv_cache and chunk_start > 0:
            win_start = self._compute_ode_window_start(chunk_start)

        if win_start > 0:
            x = x[:, :, win_start:].contiguous()
            mu = mu[:, :, win_start:].contiguous()
            mask = mask[:, :, win_start:].contiguous()

        save_caches = streaming and use_kv_cache and chunk_start == 0
        caches: list = [] if save_caches else []

        for step in range(len(t_span) - 1):
            r = t_span[step]
            t_end = t_span[step + 1]
            dt = t_end - r
            if save_caches:
                x, sc = _euler_step_streaming(
                    self.estimator, x, mask, mu, spks, cond, r, t_end, dt, None,
                )
                caches.append(sc)
            else:
                x = _euler_step_nonstreaming(
                    self.estimator, x, mask, mu, spks, cond, streaming, r, t_end, dt,
                )

        if save_caches:
            self._decoder_caches = caches

        if win_start > 0:
            full_x = torch.zeros(batch_size, n_feats, cur_len,
                                 device=x.device, dtype=x.dtype)
            full_x[:, :, win_start:] = x
            return full_x
        return x

    @torch.inference_mode()
    def forward(self, mu: torch.Tensor, mask: torch.Tensor, n_timesteps: int, finalize: bool, temperature: float = 1.0, spks: torch.Tensor = None, cond=None, streaming=False, z: torch.Tensor = None, chunk_start: int = 0) -> torch.Tensor:
        """
        Forward diffusion process.
        Args:
            mu: (batch_size, n_feats, mel_timesteps)
            mask: (batch_size, 1, mel_timesteps)
            n_timesteps: number of diffusion steps
            temperature: noise scale
            spks: (batch_size, spk_emb_dim), optional
            cond: reserved for future use
            z: optional pre-generated noise (batch_size, n_feats, mel_timesteps).
               When provided, reuses this noise instead of sampling fresh noise,
               ensuring consistency across streaming chunks.
        Returns:
            sample: (batch_size, n_feats, mel_timesteps)
        """

        mu = self.encode_mu(mu, mask, finalize=finalize, streaming=streaming)

        if z is None:
            z = torch.randn_like(mu) * temperature
        else:
            z = z[:, :, :mu.shape[-1]]
        t_span = torch.linspace(0, 1, n_timesteps + 1, device=mu.device)
        return self.solve_euler(
            z,
            t_span=t_span,
            mu=mu,
            mask=mask,
            spks=spks,
            cond=cond,
            streaming=streaming,
            chunk_start=chunk_start,
        )
    
    def compute_loss(self, x1: torch.Tensor, mask: torch.Tensor, mu: torch.Tensor, spks: torch.Tensor = None, cond=None):
        """
        Computes diffusion loss.
        Args:
            x1: Target (batch_size, n_feats, mel_timesteps)
            mask: (batch_size, 1, mel_timesteps)
            mu: (batch_size, n_feats, mel_timesteps)
            spks: (batch_size, spk_emb_dim), optional
            cond: reserved for future use
        Returns:
            loss: conditional flow matching loss
            y: conditional flow (batch_size, n_feats, mel_timesteps)
        """

        assert self.streaming, "CFM_Causal is only supported for streaming"
        streaming = True if random.random() < 0.5 else False

        mu = self.encoder(mu, mask, streaming=streaming)

        b, _, t = mu.shape
        t = torch.rand([b, 1, 1], device=mu.device, dtype=mu.dtype)
        z = torch.randn_like(x1)

        y = (1 - (1 - self.sigma_min) * t) * z + t * x1
        u = x1 - (1 - self.sigma_min) * z

        loss = F.mse_loss(self.estimator(y, mask, mu, t.squeeze(), spks, cond, streaming=streaming), u, reduction="sum") / (
            torch.sum(mask) * u.shape[1]
        )
        return loss, y
