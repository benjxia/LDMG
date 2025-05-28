from lightning import LightningModule
import torch
from torch import nn
from torch import optim
import numpy as np

import math

from ldm.modules import DEFAULT_1D_KERNEL_SIZE, DEFAULT_1D_PADDING, DEFAULT_AUDIO_DUR, DEFAULT_INPUT_SR, DEFAULT_LATENT_CHANNELS, DEFAULT_LATENT_SR, DEFAULT_MAX_CHANNELS
from ldm.modules.loss import ELBO_Loss

class SelfAttention(nn.Module):
    def __init__(self, channels: int, n_heads: int, max_len: int = DEFAULT_AUDIO_DUR * DEFAULT_LATENT_SR):
        """
        Multiheaded self-attention (with residual connections)

        Args:
            channels (int): Channels for input sequence
            n_heads (int): Number of attention heads
        """
        super().__init__()
        self.dim = channels
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)

        # Precompute positional encodings
        position = torch.arange(0, max_len).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, channels, 2) * -(math.log(10000.0) / channels))
        pe = torch.zeros(max_len, channels)
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)  # shape [1, max_len, channels]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass for self-attention (with residual connections)

        B = batch
        C = channels
        T = time
        Args:
            x (torch.Tensor): Sequence tensor of shape [B, C, T]

        Returns:
            torch.Tensor: Tensor of shape [B, C, T]
        """
        B, C, T = x.shape
        x = x.permute(0, 2, 1)  # Reshape to [B, T, C] (expected shape for attention)
        pos_emb = self.pe[:, :T, :] # self.register_buffer adds self.pe
        attn_in = x + pos_emb
        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        return (x + attn_out).permute(0, 2, 1)  # Residual connection, Back to [B, C, T]

class UpsampleLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(UpsampleLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample = nn.ConvTranspose1d(in_channels, out_channels, DEFAULT_1D_KERNEL_SIZE, stride=2, padding=DEFAULT_1D_PADDING, output_padding=1)
        self.conv = nn.Conv1d(out_channels, out_channels, DEFAULT_1D_KERNEL_SIZE, stride=1, padding=DEFAULT_1D_PADDING)
        self.norm = nn.GroupNorm(out_channels // 4, out_channels)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.activation(x)
        x = self.conv(x) + x
        x = self.norm(x)
        x = self.activation(x)
        return x

class DownsampleLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super(DownsampleLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv1d(in_channels, out_channels, DEFAULT_1D_KERNEL_SIZE, stride=2, padding=DEFAULT_1D_PADDING)
        self.norm = nn.GroupNorm(out_channels // 4, out_channels)
        self.activation = nn.GELU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.conv(x)
        x = self.norm(x)
        x = self.activation(x)
        return x

class VAE_Encoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 latent_channels: int=DEFAULT_LATENT_CHANNELS,
                 input_sr: int=DEFAULT_INPUT_SR,
                 latent_sr: int=DEFAULT_LATENT_SR):
        """
        Conditional Variational Autoencoder Encoder
        Args:
            input_channels (int): Number of channels for input audio waveforms (ex. stereo vs. mono)
            latent_channels (int): Number of channels for latent audio waveforms
            input_sr (int): Input audio waveform sample rate (16000Hz default)
            latent_sr (int): Target Latent audio sample rate (125Hz default) - No guarantees it'll actually reach this
        """
        super(VAE_Encoder, self).__init__()

        self.input_channels = input_channels
        self.latent_channels = latent_channels
        self.input_sr = input_sr
        self.latent_sr = latent_sr

        # Input dimension must be some power of 2 multiple of latent dim
        self.n_downsamples = np.ceil(np.log2(self.input_sr / self.latent_sr)).astype(np.int32)
        assert (2 ** self.n_downsamples) * latent_sr == self.input_sr

        starter_channels = 16
        layers = [
            nn.Conv1d(input_channels, starter_channels, DEFAULT_1D_KERNEL_SIZE, stride=1, padding=DEFAULT_1D_PADDING),
            nn.GELU(),
        ]

        # Channels go from 16 -> 32 -> 64 -> DEFAULT_MAX_CHANNELS ... n_downsamples layers
        in_ch = starter_channels
        for i in range(self.n_downsamples):
            out_ch = min(in_ch * 2, DEFAULT_MAX_CHANNELS)
            layers.append(DownsampleLayer(in_ch, out_ch))  # Downsample by factor of 2
            in_ch = out_ch
            layers.append(nn.Conv1d(in_ch, in_ch, DEFAULT_1D_KERNEL_SIZE, stride=1, padding=DEFAULT_1D_PADDING))
            layers.append(nn.GELU())

        layers.append(SelfAttention(in_ch, 4))

        self.layers = nn.Sequential(*layers)

        self.mu_proj = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size=DEFAULT_1D_KERNEL_SIZE, padding=DEFAULT_1D_PADDING),
            nn.GELU(),
            nn.Conv1d(in_ch, latent_channels, kernel_size=1)
        )

        self.logvar_proj = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size=DEFAULT_1D_KERNEL_SIZE, padding=DEFAULT_1D_PADDING),
            nn.GELU(),
            nn.Conv1d(in_ch, latent_channels, kernel_size=1)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        VAE encoder forward pass, input waveforms are expected to be of the correct sample rate (from VAE encoder constructor)

        B = batch size
        C = channels
        T = timesteps

        L = latent space channels
        T' = T * latent_sr / input_sr
        Args:
            x (torch.Tensor): Batch of waveforms of shape [B, C, T]
        Returns:
            Parameters to a diagonal Gaussian in latent space
            torch.Tensor: Latent space tensor of shape [B, L, T'] mean
            torch.Tensor: Latent space tensor of shape [B, L, T'] log variances
        """
        x = self.layers(x)
        return self.mu_proj(x), self.logvar_proj(x)

class VAE_Decoder(nn.Module):
    def __init__(self,
                 input_channels: int,
                 latent_channels: int=DEFAULT_LATENT_CHANNELS,
                 input_sr: int=DEFAULT_INPUT_SR,
                 latent_sr: int=DEFAULT_LATENT_SR):
        """
        Conditional Variational Autoencoder Decoder
        Args:
            input_channels (int): Number of channels for input audio waveforms (ex. stereo vs. mono)
            latent_channels (int): Number of channels for latent audio waveforms
            input_sr (int): Input audio waveform sample rate (16000Hz default)
            latent_sr (int): Target Latent audio sample rate (125Hz default) - No guarantees it'll actually reach this
        """
        super(VAE_Decoder, self).__init__()

        self.input_channels = input_channels
        self.latent_channels = latent_channels
        self.input_sr = input_sr
        self.latent_sr = latent_sr

        # Input dimensions must be some power of 2 multiple of latent dim
        self.n_upsamples = np.ceil(np.log2(self.input_sr / self.latent_sr)).astype(np.int32)
        assert (2 ** self.n_upsamples) * latent_sr == self.input_sr

        channels = DEFAULT_MAX_CHANNELS
        layers = [
            nn.Conv1d(latent_channels, channels, DEFAULT_1D_KERNEL_SIZE, stride=1, padding=DEFAULT_1D_PADDING),
            nn.GELU(),
        ]

        layers.append(SelfAttention(channels, 4))

        for i in range(self.n_upsamples):
            layers.append(UpsampleLayer(channels, channels))
            layers.append(nn.Conv1d(channels, channels, DEFAULT_1D_KERNEL_SIZE, stride=1, padding=DEFAULT_1D_PADDING))
            layers.append(nn.GELU())

        layers.append(nn.Conv1d(channels, input_channels, kernel_size=1))

        self.layers = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        VAE decoder forward pass, input latent space waveforms and outputs waveforms in original input space

        B = batch size
        Z = latent channels
        T = timesteps
        Args:
            x (torch.Tensor): Batch of latent waveforms of shape [B, Z, T']

        Returns:
            torch.Tensor: Reconstruction of waveforms in input space of shape [B, C, T]
        """
        return self.layers(x)

class VAE(nn.Module):
    def __init__(self, audio_channels: int, input_sr: int=DEFAULT_INPUT_SR):
        super(VAE, self).__init__()
        self.channels = audio_channels
        self.input_sr = input_sr
        self.encoder = VAE_Encoder(audio_channels, input_sr=DEFAULT_INPUT_SR)
        self.decoder = VAE_Decoder(audio_channels, input_sr=DEFAULT_INPUT_SR)
        self.latent_dim = self.decoder.latent_channels
        self.latent_sr = self.decoder.latent_sr

    def _sample(self, mu: torch.Tensor, log_var: torch.Tensor):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def generate(self, n_samples: int=1, dur: int = DEFAULT_AUDIO_DUR) -> torch.Tensor:
        z = torch.randn([n_samples, self.latent_dim, self.latent_sr * dur]).to(device='cuda:0')
        audio = self.decoder(z)
        return audio

    def forward(self, input: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Full VAE encoder + decoder forward pass

        Args:
            x (torch.Tensor): Batch of waveforms of shape [B, C, T]

        Returns:
            torch.Tensor: Reconstruction of waveforms in input space of shape [B, C, T]
            torch.Tensor: Mean of Gaussian distribution over latent space
            torch.Tensor: Log variance of Gaussian distribution over latent space

        """
        mu, log_var = self.encoder(input)
        sample = self._sample(mu, log_var)
        reconstruction = self.decoder(sample)
        return reconstruction, mu, log_var

# Unused now
class AudioVAE(LightningModule):
    def __init__(self, channels: int, kl_weight: float = 1e-3, lr=1e-4):
        super(AudioVAE, self).__init__()
        self.vae = VAE(channels)
        self.loss = ELBO_Loss(kl_weight)
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vae(x)

    def training_step(self, batch, batch_idx=None, dataloader_idx=None) -> torch.Tensor:
        reconstruction, mu, log_var = self.vae(batch)
        loss = self.loss(reconstruction, batch, mu, log_var)
        self.log('training_elbo_loss', loss, prog_bar=True)
        return loss

    def generate(self, n_samples: int):
        return self.vae.generate(n_samples)

    def configure_optimizers(self):
        return optim.Adam(self.vae.parameters(), self.lr)
