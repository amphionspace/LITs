""" from https://github.com/jaywalnut310/glow-tts """

import math
import torch
import torch.nn as nn
from einops import rearrange
import lits.utils as utils
from lits.utils.model import sequence_mask

log = utils.get_pylogger(__name__)

class LayerNorm(nn.Module):
    """Custom LayerNorm for 1D/2D tensors."""
    def __init__(self, channels: int, eps: float = 1e-4):
        super().__init__()
        self.channels = channels
        self.eps = eps
        self.gamma = nn.Parameter(torch.ones(channels))
        self.beta = nn.Parameter(torch.zeros(channels))
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n_dims = len(x.shape)
        mean = torch.mean(x, 1, keepdim=True)
        variance = torch.mean((x - mean) ** 2, 1, keepdim=True)
        x = (x - mean) * torch.rsqrt(variance + self.eps)
        shape = [1, -1] + [1] * (n_dims - 2)
        x = x * self.gamma.view(*shape) + self.beta.view(*shape)
        return x

class ConvReluNorm(nn.Module):
    """Stacked Conv1d + LayerNorm + ReLU + Dropout blocks."""
    def __init__(self, in_channels: int, hidden_channels: int, out_channels: int, kernel_size: int, n_layers: int, p_dropout: float):
        super().__init__()
        self.conv_layers = nn.ModuleList()
        self.norm_layers = nn.ModuleList()
        self.conv_layers.append(nn.Conv1d(in_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
        self.norm_layers.append(LayerNorm(hidden_channels))
        self.relu_drop = nn.Sequential(nn.ReLU(), nn.Dropout(p_dropout))
        for _ in range(n_layers - 1):
            self.conv_layers.append(nn.Conv1d(hidden_channels, hidden_channels, kernel_size, padding=kernel_size // 2))
            self.norm_layers.append(LayerNorm(hidden_channels))
        self.proj = nn.Conv1d(hidden_channels, out_channels, 1)
        self.proj.weight.data.zero_()
        self.proj.bias.data.zero_()
    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x_org = x
        for i in range(len(self.conv_layers)):
            x = self.conv_layers[i](x * x_mask)
            x = self.norm_layers[i](x)
            x = self.relu_drop(x)
        x = x_org + self.proj(x)
        return x * x_mask

class DurationPredictor(nn.Module):
    """Predicts duration for each input token."""
    def __init__(self, in_channels: int, filter_channels: int, kernel_size: int, p_dropout: float):
        super().__init__()
        self.drop = nn.Dropout(p_dropout)
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_1 = LayerNorm(filter_channels)
        self.conv_2 = nn.Conv1d(filter_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.norm_2 = LayerNorm(filter_channels)
        self.proj = nn.Conv1d(filter_channels, 1, 1)
    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.norm_1(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        x = torch.relu(x)
        x = self.norm_2(x)
        x = self.drop(x)
        x = self.proj(x * x_mask)
        return x * x_mask

class RotaryPositionalEmbeddings(nn.Module):
    """Rotary positional encoding (RoPE) for attention."""
    def __init__(self, d: int, base: int = 10_000):
        super().__init__()
        self.base = base
        self.d = int(d)
        self.cos_cached = None
        self.sin_cached = None
    def _build_cache(self, x: torch.Tensor):
        if self.cos_cached is not None and x.shape[0] <= self.cos_cached.shape[0]:
            return
        seq_len = x.shape[0]
        theta = 1.0 / (self.base ** (torch.arange(0, self.d, 2).float() / self.d)).to(x.device)
        seq_idx = torch.arange(seq_len, device=x.device).float().to(x.device)
        idx_theta = torch.einsum("n,d->nd", seq_idx, theta)
        idx_theta2 = torch.cat([idx_theta, idx_theta], dim=1)
        self.cos_cached = idx_theta2.cos()[:, None, None, :]
        self.sin_cached = idx_theta2.sin()[:, None, None, :]
    def _neg_half(self, x: torch.Tensor):
        d_2 = self.d // 2
        return torch.cat([-x[:, :, :, d_2:], x[:, :, :, :d_2]], dim=-1)
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = rearrange(x, "b h t d -> t b h d")
        self._build_cache(x)
        x_rope, x_pass = x[..., : self.d], x[..., self.d :]
        neg_half_x = self._neg_half(x_rope)
        x_rope = (x_rope * self.cos_cached[: x.shape[0]]) + (neg_half_x * self.sin_cached[: x.shape[0]])
        return rearrange(torch.cat((x_rope, x_pass), dim=-1), "t b h d -> b h t d")

class MultiHeadAttention(nn.Module):
    """Multi-head self-attention with rotary positional encoding."""
    def __init__(self, channels: int, out_channels: int, n_heads: int, heads_share: bool = True, p_dropout: float = 0.0, proximal_bias: bool = False, proximal_init: bool = False):
        super().__init__()
        assert channels % n_heads == 0
        self.channels = channels
        self.out_channels = out_channels
        self.n_heads = n_heads
        self.heads_share = heads_share
        self.proximal_bias = proximal_bias
        self.p_dropout = p_dropout
        self.attn = None
        self.k_channels = channels // n_heads
        self.conv_q = nn.Conv1d(channels, channels, 1)
        self.conv_k = nn.Conv1d(channels, channels, 1)
        self.conv_v = nn.Conv1d(channels, channels, 1)
        self.query_rotary_pe = RotaryPositionalEmbeddings(int(self.k_channels * 0.5))
        self.key_rotary_pe = RotaryPositionalEmbeddings(int(self.k_channels * 0.5))
        self.conv_o = nn.Conv1d(channels, out_channels, 1)
        self.drop = nn.Dropout(p_dropout)
        nn.init.xavier_uniform_(self.conv_q.weight)
        nn.init.xavier_uniform_(self.conv_k.weight)
        if proximal_init:
            self.conv_k.weight.data.copy_(self.conv_q.weight.data)
            self.conv_k.bias.data.copy_(self.conv_q.bias.data)
        nn.init.xavier_uniform_(self.conv_v.weight)
    def forward(self, x: torch.Tensor, c: torch.Tensor, attn_mask: torch.Tensor = None) -> torch.Tensor:
        q = self.conv_q(x)
        k = self.conv_k(c)
        v = self.conv_v(c)
        x, self.attn = self.attention(q, k, v, mask=attn_mask)
        x = self.conv_o(x)
        return x
    def attention(self, query, key, value, mask=None):
        b, d, t_s, t_t = (*key.size(), query.size(2))
        query = rearrange(query, "b (h c) t-> b h t c", h=self.n_heads)
        key = rearrange(key, "b (h c) t-> b h t c", h=self.n_heads)
        value = rearrange(value, "b (h c) t-> b h t c", h=self.n_heads)
        query = self.query_rotary_pe(query).contiguous()
        key = self.key_rotary_pe(key).contiguous()
        value = value.contiguous()
        scores = torch.matmul(query, key.transpose(-2, -1).contiguous()) / math.sqrt(self.k_channels)
        if self.proximal_bias:
            assert t_s == t_t, "Proximal bias is only available for self-attention."
            scores = scores + self._attention_bias_proximal(t_s).to(device=scores.device, dtype=scores.dtype)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e4)
        p_attn = torch.nn.functional.softmax(scores, dim=-1)
        p_attn = self.drop(p_attn)
        output = torch.matmul(p_attn, value)
        output = output.transpose(2, 3).contiguous().view(b, d, t_t)
        return output, p_attn
    @staticmethod
    def _attention_bias_proximal(length):
        r = torch.arange(length, dtype=torch.float32)
        diff = torch.unsqueeze(r, 0) - torch.unsqueeze(r, 1)
        return torch.unsqueeze(torch.unsqueeze(-torch.log1p(torch.abs(diff)), 0), 0)

class FFN(nn.Module):
    """Feed-forward network for transformer encoder."""
    def __init__(self, in_channels: int, out_channels: int, filter_channels: int, kernel_size: int, p_dropout: float = 0.0):
        super().__init__()
        self.conv_1 = nn.Conv1d(in_channels, filter_channels, kernel_size, padding=kernel_size // 2)
        self.conv_2 = nn.Conv1d(filter_channels, out_channels, kernel_size, padding=kernel_size // 2)
        self.drop = nn.Dropout(p_dropout)
    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        x = self.conv_1(x * x_mask)
        x = torch.relu(x)
        x = self.drop(x)
        x = self.conv_2(x * x_mask)
        return x * x_mask

class Encoder(nn.Module):
    """Transformer-style encoder with multi-head attention and FFN."""
    def __init__(self, hidden_channels: int, filter_channels: int, n_heads: int, n_layers: int, kernel_size: int = 1, p_dropout: float = 0.0, **kwargs):
        super().__init__()
        self.drop = nn.Dropout(p_dropout)
        self.attn_layers = nn.ModuleList()
        self.norm_layers_1 = nn.ModuleList()
        self.ffn_layers = nn.ModuleList()
        self.norm_layers_2 = nn.ModuleList()
        for _ in range(n_layers):
            self.attn_layers.append(MultiHeadAttention(hidden_channels, hidden_channels, n_heads, p_dropout=p_dropout))
            self.norm_layers_1.append(LayerNorm(hidden_channels))
            self.ffn_layers.append(FFN(hidden_channels, hidden_channels, filter_channels, kernel_size, p_dropout=p_dropout))
            self.norm_layers_2.append(LayerNorm(hidden_channels))
    def forward(self, x: torch.Tensor, x_mask: torch.Tensor) -> torch.Tensor:
        attn_mask = x_mask.unsqueeze(2) * x_mask.unsqueeze(-1)
        for i in range(len(self.attn_layers)):
            x = x * x_mask
            y = self.attn_layers[i](x, x, attn_mask)
            y = self.drop(y)
            x = self.norm_layers_1[i](x + y)
            y = self.ffn_layers[i](x, x_mask)
            y = self.drop(y)
            x = self.norm_layers_2[i](x + y)
        x = x * x_mask
        return x

class TextEncoder(nn.Module):
    """Text encoder with embedding, optional prenet, transformer encoder, and duration predictor."""
    def __init__(self, encoder_type, encoder_params, duration_predictor_params, n_vocab, n_spks=1, spk_emb_dim=128, n_tones=0):
        super().__init__()
        self.encoder_type = encoder_type
        self.n_vocab = n_vocab
        self.n_tones = n_tones
        self.n_feats = encoder_params.n_feats
        self.n_channels = encoder_params.n_channels
        self.spk_emb_dim = spk_emb_dim
        self.n_spks = n_spks
        # 一些旧 checkpoint 中：全局 n_spks > 1，但 encoder_params.n_spks == 1。
        # 这类模型并不在 encoder 通道维拼接 speaker embedding。
        encoder_n_spks = getattr(encoder_params, "n_spks", n_spks)
        self.use_spk_cond_in_encoder = encoder_n_spks > 1
        encoder_in_channels = self.n_channels + (spk_emb_dim if self.use_spk_cond_in_encoder else 0)

        self.emb = nn.Embedding(n_vocab, self.n_channels)
        nn.init.normal_(self.emb.weight, 0.0, self.n_channels**-0.5)
        if n_tones > 0:
            # Tone id 0 = "no tone" (initials, English, punctuation, blanks);
            # padding_idx pins that row to zero so toneless positions behave
            # exactly as without the tone embedding.
            self.tone_emb = nn.Embedding(n_tones + 1, self.n_channels, padding_idx=0)
            nn.init.normal_(self.tone_emb.weight, 0.0, self.n_channels**-0.5)
            with torch.no_grad():
                self.tone_emb.weight[0].zero_()
        else:
            self.tone_emb = None
        if encoder_params.prenet:
            self.prenet = ConvReluNorm(
                self.n_channels,
                self.n_channels,
                self.n_channels,
                kernel_size=5,
                n_layers=3,
                p_dropout=0.5,
            )
        else:
            self.prenet = lambda x, x_mask: x
        self.encoder = Encoder(
            encoder_in_channels,
            encoder_params.filter_channels,
            encoder_params.n_heads,
            encoder_params.n_layers,
            encoder_params.kernel_size,
            encoder_params.p_dropout,
        )
        self.proj_m = nn.Conv1d(encoder_in_channels, self.n_feats, 1)
        self.proj_w = DurationPredictor(
            encoder_in_channels,
            duration_predictor_params.filter_channels_dp,
            duration_predictor_params.kernel_size,
            duration_predictor_params.p_dropout,
        )
    def forward(self, x: torch.Tensor, x_lengths: torch.Tensor, spks: torch.Tensor = None, x_tones: torch.Tensor = None):
        """
        Run forward pass to the transformer based encoder and duration predictor.
        Args:
            x: (batch_size, max_text_length)
            x_lengths: (batch_size,)
            spks: (batch_size,), optional
            x_tones: (batch_size, max_text_length) tone ids, optional
        Returns:
            mu: (batch_size, n_feats, max_text_length)
            logw: (batch_size, 1, max_text_length)
            x_mask: (batch_size, 1, max_text_length)
        """
        emb = self.emb(x)
        if self.tone_emb is not None and x_tones is not None:
            emb = emb + self.tone_emb(x_tones.long())
        x = emb * math.sqrt(self.n_channels)
        x = torch.transpose(x, 1, -1)
        x_mask = torch.unsqueeze(sequence_mask(x_lengths, x.size(2)), 1).to(x.dtype)
        x = self.prenet(x, x_mask)
        if self.use_spk_cond_in_encoder and spks is not None:
            x = torch.cat([x, spks.unsqueeze(-1).repeat(1, 1, x.shape[-1])], dim=1)
        x = self.encoder(x, x_mask)
        mu = self.proj_m(x) * x_mask
        x_dp = torch.detach(x)
        logw = self.proj_w(x_dp, x_mask)
        return mu, logw, x_mask
