import torch
from torch import nn
import torch.nn.functional as F

from . import DEFAULT_AUDIO_DUR, DEFAULT_LATENT_CHANNELS, DEFAULT_LATENT_SR
from .vae_gan import AudioVAEGAN
from .diffusion import DiffusionTransformer
from .gaussian_diffusion import GaussianDiffusion

import lightning as L
from lightning import LightningModule

class AudioLDM(LightningModule):
    def __init__(self,
                 n_dit_layers: int=4,
                 audiovae_ckpt_path: str = None,
                 lr: float = 1e-4
                 ):
        """
        Properties such as latent space size and shit are determined by loaded vae-gan - and assumed to be the default values

        I'm too tired to generalize all this code...
        Args:
            n_dit_layers (int, optional): Number of Diffusion Transformer layers
            audiovae_ckpt_path (str, optional): Path to VAE-GAN checkpoint (REQUIRED)
            lr (float, optional): Learning rate
        """
        super(AudioLDM, self).__init__()
        assert audiovae_ckpt_path is not None
        self.vae = AudioVAEGAN.load_from_checkpoint(audiovae_ckpt_path)

        self.dit = DiffusionTransformer(n_dit_layers)
        self.diffusion = GaussianDiffusion()
        self.vae.freeze()
        self.lr = lr

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Tensor of latent audio of shape [B, L, T]
            t (torch.Tensor): Tensor of timesteps of shape [B]

        Returns:
            torch.Tensor: Predicted noise for timestep t - 1
        """
        return self.dit(x, t)

    def _encode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Waveform in input space of shape [B, C, T]

        Returns:
            torch.Tensor: Waveform in latent space of shape [B, L, T']
        """
        encoder = self.vae.vae.encoder
        return encoder(x)

    def _decode(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): Waveform in latent space of shape [B, L, T']

        Returns:
            torch.Tensor: Waveform in input space of shape [B, C, T]
        """
        decoder = self.vae.vae.decoder
        return decoder(x)

    def generate(self, n_samples, dur: int = DEFAULT_AUDIO_DUR):
        latent_len = dur * DEFAULT_LATENT_SR
        shape = (n_samples, DEFAULT_LATENT_CHANNELS, latent_len)
        latents = self.diffusion.sample(self.dit, shape, device=self.device)
        wavs = self._decode(latents)
        return wavs

    def training_step(self, batch, batch_idx):
        x = batch  # raw waveform [B, 1, T]
        x_latent = self._encode(x)
        B = x_latent.shape[0]
        t = torch.randint(0, self.diffusion.timesteps, (B,), device=self.device)
        loss = self.diffusion.p_losses(self.dit, x_latent, t)
        self.log("train_loss", loss)
        return loss

    def configure_optimizers(self):
        opt = torch.optim.Adam(self.dit.parameters(), lr=self.lr)
        return opt

# TODO: add a conditioned version of AudioLDM
