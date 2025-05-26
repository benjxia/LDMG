import torch
from torch import nn
import torch.nn.functional as F
from torch import optim
import lightning as L
import utility as U
import numpy as np

DEFAULT_INPUT_SR = 16000
DEFAULT_LATENT_SR = 125 # Chosen because 16000 / 2^7 = 125, and we have an even number of 0.5x downsamples
DEFAULT_LATENT_CHANNELS = 16 # Seems to be a pretty standard value for this

DEFAULT_1D_KERNEL_SIZE = 7 # This seems to be standard practice for waveforms
DEFAULT_1D_PADDING = 3 # Padding necessary for kernel size 7 for exact halving of dimensions

DEFAULT_MAX_CHANNELS = 256

DEFAULT_AUDIO_DUR = 10 # In seconds
MAX_SEQ_LEN = 20000

class ELBO_Loss(nn.Module):
    def __init__(self, KL_weight=1e-3):
        self.KL = KL_weight

    def forward(self, recon_x: torch.Tensor, x: torch.Tensor, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        MSE = nn.MSELoss(reduction='sum')(recon_x, x)
        KLD = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())
        return MSE + self.KL * KLD

class SelfAttention(nn.Module):
    def __init__(self,
                 channels: int,
                 n_heads: int):
        """
        Multiheaded self-attention (with residual connections)

        Args:
            channels (int): Channels for input sequence
            n_heads (int): Number of attention heads
        """
        super(SelfAttention, self).__init__()
        self.dim = channels
        self.attn = nn.MultiheadAttention(channels, n_heads, batch_first=True)


    def _posn_encoding(self, seq_len: int) -> torch.Tensor:
        """
        Positional encoding
        Args:
            seq_len (int): Sequence length

        Returns:
            torch.Tensor: Positional encoding of shape [seq_len, dim]
        """
        position = torch.arange(0, seq_len, 1).unsqueeze(0).unsqueeze(-1)
        denom = torch.pow(10000, -2 * torch.arange(0, self.dim, 1) / self.dim).unsqueeze(0).unsqueeze(0)
        pe = torch.zeros((1, seq_len, self.dim))
        pe[:, :, 0::2] = torch.sin(position * denom[:, :, 0::2])
        pe[:, :, 1::2] = torch.cos(position * denom[:, :, 1::2])
        self.register_buffer('pe', pe)
        return pe

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
        x = x.permute(0, 2, 1)  # [B, T, C]
        B, T, C = x.shape

        embeddings = self._posn_encoding(T)  # [B, T, C]
        attn_in = x + embeddings

        attn_out, _ = self.attn(attn_in, attn_in, attn_in, need_weights=False)
        out = x + attn_out  # Residual connection
        return out.permute(0, 2, 1)  # Back to [B, C, T]

class DownsampleLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: str='gelu'):
        super(DownsampleLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv = nn.Conv1d(in_channels, out_channels, DEFAULT_1D_KERNEL_SIZE, stride=2, padding=DEFAULT_1D_PADDING)
        self.norm = nn.GroupNorm(out_channels // 4, out_channels)
        self.activation = U.get_activation(activation)

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

        self.input_channels = input_channels,
        self.latent_channels = latent_channels
        self.input_sr = input_sr
        self.latent_sr = latent_sr
        self.activation = 'gelu'

        # Input dimension must be some power of 2 multiple of latent dim
        self.n_downsamples = np.ceil(np.log2(self.input_sr / self.latent_sr)).astype(np.int32)
        assert (2 ** self.n_downsamples) * latent_sr == self.input_sr

        starter_channels = 16
        layers = [
            nn.Conv1d(input_channels, starter_channels, DEFAULT_1D_KERNEL_SIZE, stride=1, padding=DEFAULT_1D_PADDING),
            U.get_activation_module('gelu'),
        ]

        # Channels go from 16 -> 32 -> 64 -> DEFAULT_MAX_CHANNELS ... n_downsamples layers
        in_ch = starter_channels
        for i in range(self.n_downsamples):
            out_ch = min(in_ch * 2, DEFAULT_MAX_CHANNELS)
            layers.append(DownsampleLayer(in_ch, out_ch))  # Downsample by factor of 2
            in_ch = out_ch
            if (i + 1) >= 4 and (i + 1) % 2 == 0:
                layers.append(SelfAttention(in_ch, 4))

        self.layers = nn.Sequential(*layers)

        self.mu_proj = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size=DEFAULT_1D_KERNEL_SIZE, padding=DEFAULT_1D_PADDING),
            U.get_activation_module('gelu'),
            nn.Conv1d(in_ch, latent_channels, kernel_size=1)
        )

        self.logvar_proj = nn.Sequential(
            nn.Conv1d(in_ch, in_ch, kernel_size=DEFAULT_1D_KERNEL_SIZE, padding=DEFAULT_1D_PADDING),
            U.get_activation_module('gelu'),
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

class UpsampleLayer(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, activation: str='gelu'):
        super(UpsampleLayer, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.upsample = nn.ConvTranspose1d(in_channels, out_channels, DEFAULT_1D_KERNEL_SIZE, stride=2, padding=DEFAULT_1D_PADDING, output_padding=1)
        self.conv = nn.Conv1d(out_channels, out_channels, DEFAULT_1D_KERNEL_SIZE, stride=1, padding=DEFAULT_1D_PADDING)
        self.norm = nn.GroupNorm(out_channels // 4, out_channels)
        self.activation = U.get_activation(activation)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        x = self.activation(x)
        x = self.conv(x) + x
        x = self.norm(x)
        x = self.activation(x)
        return x

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
        self.activation = 'gelu'

        # Input dimensions must be some power of 2 multiple of latent dim
        self.n_upsamples = np.ceil(np.log2(self.input_sr / self.latent_sr)).astype(np.int32)
        assert (2 ** self.n_upsamples) * latent_sr == self.input_sr

        channels = DEFAULT_MAX_CHANNELS
        layers = [
            nn.Conv1d(latent_channels, channels, DEFAULT_1D_KERNEL_SIZE, stride=1, padding=DEFAULT_1D_PADDING),
            U.get_activation_module('gelu'),
        ]

        for i in range(self.n_upsamples):
            layers.append(UpsampleLayer(channels, channels))
            if (i + 1) >= 4 and (i + 1) % 2 == 0:
                layers.append(SelfAttention(channels, 4))

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
    def __init__(self, audio_channels: int):
        super(VAE, self).__init__()
        self.channels = audio_channels
        self.encoder = VAE_Encoder(audio_channels)
        self.decoder = VAE_Decoder(audio_channels)
        self.latent_dim = self.decoder.latent_channels
        self.latent_sr = self.decoder.latent_sr

    def _sample(self, mu: torch.Tensor, log_var: torch.Tensor):
        std = torch.exp(0.5 * log_var)
        eps = torch.randn_like(std)
        return eps.mul(std).add_(mu)

    def generate(self, n_samples: int=1) -> torch.Tensor:
        z = torch.randn([n_samples, self.latent_dim, self.latent_sr])
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

class AudioVAE(L.LightningModule):
    def __init__(self, channels: int, kl_weight: float = 1e-3, lr=1e-4):
        self.vae = VAE(channels)
        self.loss = ELBO_Loss(kl_weight)
        self.lr = lr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.vae(x)

    def training_step(self, batch, batch_idx=None, dataloader_idx=None) -> torch.Tensor:
        data, labels = batch
        reconstruction, mu, log_var = self.vae(batch)
        loss = self.loss(reconstruction, batch, mu, log_var)
        self.log('training_elbo_loss', loss, prog_bar=True)
        return loss

    def generate(self, n_samples: int):
        return self.vae.generate(n_samples)

    def configure_optimizers(self):
        return optim.Adam(self.vae.parameters(), self.lr)


class Discriminator(nn.Module):
    def __init__(self):
        pass
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

class GAN(nn.Module):
    def __init__(self):
        pass

    def forward(self, x):
        pass

if __name__ == '__main__':
    vae = AudioVAE(1)
    trainer = L.Trainer()
    trainer.fit(model=vae, train_dataloaders=None)
    # audio = torch.randn([1, 1, 16000 * 1])
    # recon, mu, log_var = vae(audio)
    # print(recon.size())
    # print(sum(p.numel() for p in vae.parameters()))
    # torch.save(vae.state_dict(), 'tmp.pt')

