import datetime as dt
import math
import random
import torch
import lits.utils.monotonic_align as monotonic_align
from lits import utils
from lits.models.base import BaseLits
from lits.models.components.flow_matching import CFM, CFM_Causal
from lits.models.components.text_encoder import TextEncoder
from lits.utils.infer_duration_floor import apply_infer_duration_patches
from lits.utils.model import (
    denormalize,
    duration_loss,
    fix_len_compatibility,
    generate_path,
    sequence_mask,
)

log = utils.get_pylogger(__name__)

class LITS(BaseLits):
    """
    LITS model: integrates text encoder, flow-matching decoder, and alignment for TTS.
    """
    def __init__(
        self,
        n_vocab: int,
        n_spks: int,
        spk_emb_dim: int,
        n_feats: int,
        encoder,
        decoder,
        cfm,
        data_statistics,
        out_size: int,
        optimizer=None,
        scheduler=None,
        prior_loss: bool = True,
        use_precomputed_durations=False,
        n_tones: int = 0,
    ):
        super().__init__()
        self.save_hyperparameters(logger=False)
        self.n_vocab = n_vocab
        self.n_tones = n_tones
        self.n_spks = n_spks
        self.spk_emb_dim = spk_emb_dim
        self.n_feats = n_feats
        self.out_size = out_size
        self.prior_loss = prior_loss
        self.use_precomputed_durations = use_precomputed_durations
        
        if n_spks > 1:
            self.spk_emb = torch.nn.Embedding(n_spks, spk_emb_dim)
        self.encoder = TextEncoder(
            encoder.encoder_type,
            encoder.encoder_params,
            encoder.duration_predictor_params,
            n_vocab,
            n_spks,
            spk_emb_dim,
            n_tones=n_tones,
        )
        self.decoder = CFM_Causal(
            in_channels=2 * encoder.encoder_params.n_feats,
            out_channel=encoder.encoder_params.n_feats,
            cfm_params=cfm,
            decoder_params=decoder,
            n_spks=n_spks,
            spk_emb_dim=spk_emb_dim,
        )
        # self.decoder = CFM(
        #     in_channels=2 * encoder.encoder_params.n_feats,
        #     out_channel=encoder.encoder_params.n_feats,
        #     cfm_params=cfm,
        #     decoder_params=decoder,
        #     n_spks=n_spks,
        #     spk_emb_dim=spk_emb_dim,
        # )
        self.update_data_statistics(data_statistics)

    def _maybe_apply_infer_duration_patches(
        self,
        w_ceil: torch.Tensor,
        x: torch.Tensor,
        x_mask: torch.Tensor,
    ) -> torch.Tensor:
        if getattr(self, "apply_infer_duration_patches", False):
            return apply_infer_duration_patches(w_ceil, x, x_mask, n_vocab=self.n_vocab)
        return w_ceil

    @torch.inference_mode()
    def synthesise(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        n_timesteps: int,
        temperature: float = 1.0,
        spks: torch.Tensor = None,
        length_scale: float = 1.0,
        x_tones: torch.Tensor = None,
    ) -> dict:
        """
        Generates mel-spectrogram from text.
        Returns encoder outputs, decoder outputs, alignment, denormalized mel, mel lengths, and RTF.
        Args:
            x: (batch_size, max_text_length)
            x_lengths: (batch_size,)
            n_timesteps: number of steps for reverse diffusion
            temperature: controls variance of terminal distribution
            spks: (batch_size,), optional
            length_scale: controls speech pace
        Returns:
            dict with keys: encoder_outputs, decoder_outputs, attn, mel, mel_lengths, rtf
        """
        t = dt.datetime.now()
        if self.n_spks > 1 and spks is not None:
            spks = self.spk_emb(spks.long())
        if spks is not None and len(spks.shape) == 1 and len(spks.shape) < len(x.shape):
            spks = spks.unsqueeze(0)
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spks, x_tones=x_tones)
        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        w_ceil = self._maybe_apply_infer_duration_patches(w_ceil, x, x_mask)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = y_lengths.max()
        y_max_length_ = fix_len_compatibility(y_max_length)
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        encoder_outputs = mu_y[:, :, :y_max_length]
        decoder_outputs = self.decoder(mu_y, y_mask, n_timesteps, True, temperature, spks)
        decoder_outputs = decoder_outputs[:, :, :y_max_length]
        t = (dt.datetime.now() - t).total_seconds()
        rtf = t * 22050 / (decoder_outputs.shape[-1] * 256)
        return {
            "encoder_outputs": encoder_outputs,
            "decoder_outputs": decoder_outputs,
            "attn": attn[:, :, :y_max_length],
            "mel": denormalize(decoder_outputs, self.mel_mean, self.mel_std),
            "mel_lengths": y_lengths,
            "rtf": rtf,
        }
    
    @torch.inference_mode()
    def get_hidden_mel(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        spks: torch.Tensor = None,
        length_scale: float = 1.0,
        x_tones: torch.Tensor = None,
    ) -> dict:
        """
        Generate hidden mel features and masks from text input.
        Args:
            x: (batch_size, max_text_length)
            x_lengths: (batch_size,)
            n_timesteps: number of steps for reverse diffusion
            temperature: controls variance of terminal distribution
            spks: (batch_size,), optional
            length_scale: controls speech pace
            streaming: unused, for interface compatibility
        Returns:
            dict with keys: mu_y, y_mask, y_max_length, spks
        """
        t = dt.datetime.now()
        if self.n_spks > 1 and spks is not None:
            spks = self.spk_emb(spks.long())
            if spks.dim() == 1:
                spks = spks.unsqueeze(0)
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spks, x_tones=x_tones)
        w = torch.exp(logw) * x_mask
        w_ceil = torch.ceil(w) * length_scale
        w_ceil = self._maybe_apply_infer_duration_patches(w_ceil, x, x_mask)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_max_length = y_lengths.max()
        y_max_length_ = fix_len_compatibility(y_max_length)
        y_mask = sequence_mask(y_lengths, y_max_length_).unsqueeze(1).to(x_mask.dtype)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        attn = generate_path(w_ceil.squeeze(1), attn_mask.squeeze(1)).unsqueeze(1)
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        return {
            "mu_y": mu_y,
            "y_mask": y_mask,
            "y_max_length": y_max_length,
            "spks": spks,
        }

    def get_mel(
        self,
        mu_y: torch.Tensor,
        y_mask: torch.Tensor,
        n_timesteps: int,
        temperature: float,
        spks: torch.Tensor = None,
        finalize = True,
        streaming: bool = True,
        z: torch.Tensor = None,
        chunk_start: int = 0,
    ) -> torch.Tensor:
        """
        Generate denormalized mel-spectrogram from hidden features.
        Args:
            mu_y: (batch_size, n_feats, max_mel_length)
            y_mask: (batch_size, 1, max_mel_length)
            n_timesteps: number of steps for reverse diffusion
            temperature: controls variance of terminal distribution
            spks: (batch_size,), optional
            streaming: whether to use streaming mode in decoder
            z: optional pre-generated noise for consistent streaming chunks
            chunk_start: first mel frame index of the current streaming chunk
        Returns:
            mel: (batch_size, n_feats, max_mel_length)
        """
        decoder_outputs = self.decoder(
            mu_y,
            y_mask,
            n_timesteps,
            finalize,
            temperature,
            spks,
            streaming=streaming,
            z=z,
            chunk_start=chunk_start,
        )
        return denormalize(decoder_outputs, self.mel_mean, self.mel_std)

    def forward(
        self,
        x: torch.Tensor,
        x_lengths: torch.Tensor,
        y: torch.Tensor,
        y_lengths: torch.Tensor,
        spks: torch.Tensor = None,
        out_size: int = None,
        cond=None,
        durations=None,
    ):
        """
        Computes duration loss, prior loss, flow matching loss, and alignment.
        Args:
            x: (batch_size, max_text_length)
            x_lengths: (batch_size,)
            y: (batch_size, n_feats, max_mel_length)
            y_lengths: (batch_size,)
            spks: (batch_size,), optional
            out_size: segment length for decoder training
        Returns:
            dur_loss, prior_loss, diff_loss, attn
        """
        if self.n_spks > 1:
            spks = self.spk_emb(spks)
        mu_x, logw, x_mask = self.encoder(x, x_lengths, spks)
        y_max_length = y.shape[-1]
        y_mask = sequence_mask(y_lengths, y_max_length).unsqueeze(1).to(x_mask)
        attn_mask = x_mask.unsqueeze(-1) * y_mask.unsqueeze(2)
        if self.use_precomputed_durations:
            attn = generate_path(durations.squeeze(1), attn_mask.squeeze(1))
        else:
            with torch.no_grad():
                const = -0.5 * math.log(2 * math.pi) * self.n_feats
                factor = -0.5 * torch.ones(mu_x.shape, dtype=mu_x.dtype, device=mu_x.device)
                y_square = torch.matmul(factor.transpose(1, 2), y**2)
                y_mu_double = torch.matmul(2.0 * (factor * mu_x).transpose(1, 2), y)
                mu_square = torch.sum(factor * (mu_x**2), 1).unsqueeze(-1)
                log_prior = y_square - y_mu_double + mu_square + const
                attn = monotonic_align.maximum_path(log_prior, attn_mask.squeeze(1))
                attn = attn.detach()
        logw_ = torch.log(1e-8 + torch.sum(attn.unsqueeze(1), -1)) * x_mask
        dur_loss = duration_loss(logw, logw_, x_lengths)
        mu_y = torch.matmul(attn.squeeze(1).transpose(1, 2), mu_x.transpose(1, 2))
        mu_y = mu_y.transpose(1, 2)
        diff_loss, _ = self.decoder.compute_loss(x1=y, mask=y_mask, mu=mu_y, spks=spks, cond=cond)
        if self.prior_loss:
            prior_loss = torch.sum(0.5 * ((y - mu_y) ** 2 + math.log(2 * math.pi)) * y_mask)
            prior_loss = prior_loss / (torch.sum(y_mask) * self.n_feats)
        else:
            prior_loss = 0
        return dur_loss, prior_loss, diff_loss, attn
