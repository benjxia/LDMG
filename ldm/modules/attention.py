
import math
import torch
from torch import nn
from typing import Union

from . import DEFAULT_AUDIO_DUR, DEFAULT_LATENT_SR

def sinu_posn_embedding(max_len: int, dim: int) -> torch.Tensor:
    position = torch.arange(0, max_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, dim, 2) * -(math.log(10000.0) / dim))
    pe = torch.zeros(max_len, dim)
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    return pe

class SelfAttention(nn.Module):
    """
    Self attention with positional embedding
    """
    def __init__(self, channels: int, n_heads: int, max_len: Union[None, int] = DEFAULT_AUDIO_DUR * DEFAULT_LATENT_SR, permute=True):
        """
        Multiheaded self-attention (with residual connections)

        Args:
            channels (int): Channels for input sequence
            n_heads (int): Number of attention heads
            max_len: Max sequence length, if none, this module will not include positional embeddings
        """
        super().__init__()
        self.dim = channels
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)
        self.permute = permute

        # Precompute positional encodings
        self.max_len = max_len
        if max_len is not None:
            pe = sinu_posn_embedding(max_len, channels)
            self.register_buffer('pe', pe.unsqueeze(0), persistent=False) # shape [1, max_len, channels]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for self-attention (with residual connections)

        B = batch
        C = channels
        T = time
        Args:
            x (torch.Tensor): Sequence tensor of shape [B, C, T]
            Could be shape [B, T, C] if permute = False

        Returns:
            torch.Tensor: Tensor of shape [B, C, T]
        """
        B, C, T = x.shape
        if self.permute:
            x = x.permute(0, 2, 1)  # Reshape to [B, T, C] (expected shape for attention)
        if self.max_len is not None:
            pos_emb = self.pe[:, :T, :] # self.register_buffer adds self.pe
            attn_in = x + pos_emb
        else:
            attn_in = x
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        out = (x + attn_out)
        if self.permute:
            x = x.permute(0, 2, 1)
        return out