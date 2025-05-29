import math
import torch
from torch import nn
import lightning as L

from . import DEFAULT_AUDIO_DUR, DEFAULT_LATENT_CHANNELS, DEFAULT_LATENT_SR
from .attention import sinu_posn_embedding, SelfAttention

def modulate(x, scale, shift):
    return x * (1 + scale) + shift

class DiffusionTransformerBlock(nn.Module):
    """
    DiT block for audio waveforms
    Heavily inspired from https://arxiv.org/pdf/2212.09748
    """
    def __init__(self, input_channels: int=DEFAULT_LATENT_CHANNELS, n_attn_heads=6):
        super(DiffusionTransformerBlock, self).__init__()
        self.ln1 = nn.LayerNorm(input_channels, elementwise_affine=False)
        self.attn = SelfAttention(input_channels, n_attn_heads, None, False)
        self.ln2 = nn.LayerNorm(input_channels, elementwise_affine=False)
        self.ff = nn.Sequential(
            nn.Linear(input_channels, 4 * input_channels),
            nn.GELU(),
            nn.Linear(4 * input_channels, input_channels),
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(input_channels, 6 * input_channels, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        L = input channels
        Args:
            x (torch.Tensor): Input latent stuff of shape [B, T', L] - Note: we this is not the standard [B, L, T'] shape!
            c (torch.Tensor): (Time) Conditioning of shape [B, L]

        Returns:
            torch.Tensor: Output tensor of shape [B, L, T']
        """
        shift_msa, scale_msa, gate_msa, shift_mlp, scale_mlp, gate_mlp = self.adaLN_modulation(c.unsqueeze(1)).chunk(6, dim=-1) # Each now of shape [B, 1, L]

        x = x + gate_msa * self.attn(modulate(self.ln1(x), scale_msa, shift_msa))
        x = x + gate_mlp * self.ff(modulate(self.ln2(x), scale_mlp, shift_mlp))

        return x

class DiffusionTransformerFinalLayer(nn.Module):
    def __init__(self, input_channels: int = DEFAULT_LATENT_CHANNELS):
        super(DiffusionTransformerFinalLayer, self).__init__()
        self.ln1 = nn.LayerNorm(input_channels, elementwise_affine=False)
        self.mlp = nn.Sequential(
            nn.Linear(input_channels, 4 * input_channels),
            nn.GELU(),
            nn.Linear(input_channels * 4, input_channels)
        )
        self.adaLN_modulation = nn.Sequential(
            nn.SiLU(),
            nn.Linear(input_channels, 2 * input_channels, bias=True)
        )

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Same input shape as DiffusionTransformerBlock
        """
        scale, shift = self.adaLN_modulation(c.unsqueeze(1)).chunk(2, dim=-1) # Each now of shape [B, 1, L]
        x = self.mlp(modulate(self.ln1(x), scale, shift))
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
                 hidden_channels=512,
                 n_attn_heads: int=8,
                 audio_dur: int = DEFAULT_AUDIO_DUR):
        super(DiffusionTransformer, self).__init__()
        pe = sinu_posn_embedding(audio_dur * DEFAULT_LATENT_SR, hidden_channels)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

        self.timestep_embedding = TimestepEmbedder(hidden_channels)
        self.up_project = nn.Conv1d(input_channels, hidden_channels, 1)
        layers = [
            DiffusionTransformerBlock(hidden_channels, n_attn_heads) for _ in range(n_layers)
        ]
        layers.append(
            DiffusionTransformerFinalLayer(hidden_channels)
        )
        self.layers = nn.ModuleList(layers)
        self.down_project = nn.Conv1d(hidden_channels, input_channels, 1)

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor of latent audio of shape [B, L, T]
            t (torch.Tensor): Tensor of timesteps of shape [B]

        Returns:
            torch.Tensor: Predicted noise for timestep t - 1
        """
        B, C, T = x.shape
        x = self.up_project(x)
        x = x.permute(0, 2, 1)
        x = x + self.pe[:, :T, :]
        c = self.timestep_embedding(t)
        for layer in self.layers:
            x = layer(x, c)

        x = x.permute(0, 2, 1)
        x = self.down_project(x)
        return x

# TODO: add a conditioned DiT, im lazy tho
