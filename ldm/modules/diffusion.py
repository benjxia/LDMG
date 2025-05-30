import math
from typing import Union
import torch
from torch import nn
import lightning as L
import numpy as np

from . import DEFAULT_1D_KERNEL_SIZE, DEFAULT_1D_PADDING, DEFAULT_AUDIO_DUR, DEFAULT_LATENT_CHANNELS, DEFAULT_LATENT_SR
from .attention import CrossAttention, sinu_posn_embedding, SelfAttention

def modulate(x, scale, shift):
    return x * (1 + scale) + shift

class DiffusionTransformerBlock(nn.Module):
    """
    DiT block for audio waveforms
    Heavily inspired from https://arxiv.org/pdf/2212.09748
    """
    def __init__(self, input_channels: int = DEFAULT_LATENT_CHANNELS, n_attn_heads=6, cross_attn_enabled=False):
        super(DiffusionTransformerBlock, self).__init__()
        self.cross_attn_enabled = cross_attn_enabled

        self.ln1 = nn.LayerNorm(input_channels, elementwise_affine=False)
        self.attn = SelfAttention(input_channels, n_attn_heads, None)

        self.ln_ca = nn.LayerNorm(input_channels, elementwise_affine=False) if cross_attn_enabled else nn.Identity()
        self.cross_attn = CrossAttention(input_channels, n_attn_heads, None) if cross_attn_enabled else nn.Identity()

        self.ln2 = nn.LayerNorm(input_channels, elementwise_affine=False)
        self.ff = nn.Sequential(
            nn.Conv1d(input_channels, 4 * input_channels, DEFAULT_1D_KERNEL_SIZE, padding=DEFAULT_1D_PADDING),
            nn.GELU(),
            nn.Conv1d(4 * input_channels, input_channels, DEFAULT_1D_KERNEL_SIZE, padding=DEFAULT_1D_PADDING),
        )

        # adaptive layer norm
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(input_channels, 9 * input_channels, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, kv: Union[None, torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): [B, L, T'] latent input
            c (torch.Tensor): [B, L] conditioning (e.g., timestep embedding)
            kv (torch.Tensor | None): [B, num_tokens, L] for cross-attn (e.g., text embeddings)

        Returns:
            torch.Tensor: [B, L, T']
        """
        # AdaLN modulation, all shape [B, L, 1]
        shift_msa, scale_msa, gate_msa, \
        shift_mlp, scale_mlp, gate_mlp, \
        shift_ca, scale_ca, gate_ca = self.adaLN_modulation(c).unsqueeze(-1).chunk(9, dim=-2)

        # Self-attention with modulation and residual
        x_ln1 = self.ln1(x.transpose(1, 2)).transpose(1, 2)
        x = x + gate_msa * self.attn(modulate(x_ln1, scale_msa, shift_msa))

        # Cross-attention if enabled
        if self.cross_attn_enabled:
            x_ln_ca = self.ln_ca(x.transpose(1, 2)).transpose(1, 2)
            # hope we didnt set kv to None, otherwise everything implodes
            x = x + gate_ca * self.cross_attn(modulate(x_ln_ca, scale_ca, shift_ca), kv)

        # Feedforward with modulation and residual
        x_ln2 = self.ln2(x.transpose(1, 2)).transpose(1, 2)
        x = x + gate_mlp * self.ff(modulate(x_ln2, scale_mlp, shift_mlp))

        return x

class DiffusionTransformerFinalLayer(nn.Module):
    def __init__(self, input_channels: int = DEFAULT_LATENT_CHANNELS):
        super(DiffusionTransformerFinalLayer, self).__init__()
        self.ln1 = nn.LayerNorm(input_channels, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Conv1d(input_channels, 4 * input_channels, DEFAULT_1D_KERNEL_SIZE, padding=DEFAULT_1D_PADDING),
            nn.GELU(),
            nn.Conv1d(input_channels * 4, input_channels, DEFAULT_1D_KERNEL_SIZE, padding=DEFAULT_1D_PADDING)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(input_channels, 2 * input_channels, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor, _) -> torch.Tensor:
        """
        Same input shape as DiffusionTransformerBlock
        _ is a dummy parameter that does nothing, cus i want to loop over layers in DiffusionTransformer with possible text conditioning
        """
        scale, shift = self.adaLN_modulation(c).unsqueeze(-1).chunk(2, dim=-2) # Each now of shape [B, L, 1]
        x_ln1 = self.ln1(x.transpose(1, 2)).transpose(1, 2)
        x = self.mlp(modulate(x_ln1, scale, shift))
        return x

class TimestepEmbedder(nn.Module):
    """
    Embeds scalar timesteps into vector representations.

    I stole this code from https://github.com/chuanyangjin/fast-DiT/blob/main/models.py#L27 :)
    """
    def __init__(self,
                 hidden_size: int,
                 frequency_embedding_size: int=256):
        super(TimestepEmbedder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            t (torch.Tensor): 1D Ttnsor of integer timesteps of shape [B]

        Returns:
            torch.Tensor: Timestep embedding tensor fo shape [B, D]
        """
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb

class DiffusionTransformer(nn.Module):
    def __init__(self,
                 n_layers: int = 8,
                 input_channels: int = DEFAULT_LATENT_CHANNELS,
                 hidden_channels=128,
                 n_attn_heads: int=8,
                 audio_dur: int = DEFAULT_AUDIO_DUR,
                 cross_attn_enabled=False):
        super(DiffusionTransformer, self).__init__()
        pe = sinu_posn_embedding(audio_dur * DEFAULT_LATENT_SR, hidden_channels)
        # pe is normally shape [len, D], make it [1, D, len] to make it compatible with waveform shape
        self.register_buffer('pe', pe.unsqueeze(0).permute(0, 2, 1), persistent=False)

        self.timestep_embedding = TimestepEmbedder(hidden_channels)
        n_upsamples = np.ceil(np.log2(hidden_channels / input_channels)).astype(np.int32)
        assert (2 ** n_upsamples) * input_channels == hidden_channels

        up_layers = []
        in_ch = input_channels
        for i in range(n_upsamples):
            out_ch = min(in_ch * 2, hidden_channels)
            up_layers.append(nn.Conv1d(in_ch, out_ch, DEFAULT_1D_KERNEL_SIZE, stride=1, padding=DEFAULT_1D_PADDING))
            in_ch = out_ch
            up_layers.append(nn.GELU())

        self.upsample = nn.Sequential(*up_layers)

        layers = [
            DiffusionTransformerBlock(hidden_channels, n_attn_heads, cross_attn_enabled and i % 4 == 0) for i in range(n_layers)
        ]
        layers.append(
            DiffusionTransformerFinalLayer(hidden_channels)
        )
        self.layers = nn.ModuleList(layers)

        down_layers = []
        in_ch = hidden_channels
        for i in range(n_upsamples):
            out_ch = min(in_ch // 2, hidden_channels)
            down_layers.append(nn.Conv1d(in_ch, out_ch, DEFAULT_1D_KERNEL_SIZE, stride=1, padding=DEFAULT_1D_PADDING))
            in_ch = out_ch
            down_layers.append(nn.GELU())
        self.downsample = nn.Sequential(*down_layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor, tc: Union[None, torch.Tensor]=None) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor of latent audio of shape [B, L, T]
            t (torch.Tensor): Tensor of timesteps of shape [B]
            tc (torch.Tensor): Optional text conditioning of shape [B, text_len, L]
        Returns:
            torch.Tensor: Predicted noise for timestep t - 1
        """
        B, C, T = x.shape
        x = self.upsample(x)
        x = x + self.pe[:, :T, :]
        c = self.timestep_embedding(t)
        for layer in self.layers:
            x = layer(x, c, tc)

        x = self.downsample(x)
        return x

# TODO: add a conditioned DiT, im lazy tho
