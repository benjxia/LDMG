import torch
from torch import nn
import torch.nn.functional as F

import lightning as L
from lightning import LightningModule

from . import DEFAULT_INPUT_SR
from .autoencoder import DEFAULT_AUDIO_DUR, VAE
from .loss import ELBO_Loss, feature_matching_loss

import numpy as np

class PatchDiscriminator(nn.Module):
    def __init__(self, input_channels: int):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Conv1d(input_channels, 128, 15, stride=1, padding=7),   # preserves resolution
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 128, 41, stride=4, padding=20, groups=4),   # grouped conv like HiFi-GAN
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 128, 41, stride=4, padding=20, groups=16),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 128, 41, stride=4, padding=20, groups=16),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 128, 41, stride=4, padding=20, groups=16),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 128, 5, stride=1, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(128, 1, kernel_size=3, stride=1, padding=1)     # patch discriminator output
        ])

    def forward(self, x, return_features=False):
        features = []
        for layer in self.layers:
            x = layer(x)
            features.append(x)
        return (x, features) if return_features else x

class AudioVAEGAN(LightningModule):
    """
    Heavily inspired by https://arxiv.org/pdf/2404.10301v2
    """
    def __init__(self, channels: int, kl_weight: float = 1e-3, adv_weight: float = 1.0, lr: float=1e-4, discriminator_pause: int=0, sample_rate=DEFAULT_INPUT_SR):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.automatic_optimization = False # We define the optimization routine in training_step() instead of using lightning's automatic one

        self.vae = VAE(channels)
        self.discriminator = PatchDiscriminator(channels)

        self.recon_loss = ELBO_Loss(kl_weight, sample_rate)
        self.adv_weight = adv_weight
        self.discriminator_pause = discriminator_pause

    def forward(self, x):
        return self.vae(x)

    # # Cross entropy adversarial loss, I found hinge loss to perform better
    # def adversarial_loss(self, pred, target_is_real=True):
    #     target = torch.ones_like(pred) if target_is_real else torch.zeros_like(pred)
    #     return F.binary_cross_entropy_with_logits(pred, target)

    # Hinge adversarial loss
    def adversarial_loss(self, pred, target_is_real=True):
        if target_is_real:
            return torch.mean(F.relu(1.0 - pred))
        else:
            return torch.mean(F.relu(1.0 + pred))

    def training_step(self, batch, batch_idx):
        real = batch
        opt_vae, opt_disc = self.optimizers()

        # === Train Generator ===
        self.toggle_optimizer(opt_vae)
        recon, mu, logvar = self.vae(real)

        elbo = self.recon_loss(recon, real, mu, logvar)

        d_fake, fake_feats = self.discriminator(recon, return_features=True)
        _, real_feats = self.discriminator(real, return_features=True)

        adv_loss = self.adversarial_loss(d_fake, True)
        fm_loss = feature_matching_loss(real_feats, fake_feats)

        total_gen_loss = elbo + self.adv_weight * adv_loss + fm_loss

        self.manual_backward(total_gen_loss)
        torch.nn.utils.clip_grad_norm_(self.vae.parameters(), max_norm=1.0)
        opt_vae.step()
        opt_vae.zero_grad()
        self.untoggle_optimizer(opt_vae)

        # === Train Discriminator ===
        self.toggle_optimizer(opt_disc)
        recon_detached = recon.detach()

        d_real = self.discriminator(real)
        d_fake = self.discriminator(recon_detached)

        real_loss = self.adversarial_loss(d_real, True)
        fake_loss = self.adversarial_loss(d_fake, False)
        d_loss = 0.5 * (real_loss + fake_loss)

        # Discriminator's prety strong, let the generator have some fun
        if self.discriminator_pause != 0 and batch_idx % self.discriminator_pause == 0:
            opt_disc.zero_grad()
            self.log_dict({
                "gen/elbo": elbo,
                "gen/adv": adv_loss,
                "gen/fm": fm_loss,
                "gen/total": total_gen_loss,
                "disc/loss": d_loss,
            }, prog_bar=True, on_step=True, on_epoch=True)
            self.untoggle_optimizer(opt_disc)
            return

        self.manual_backward(d_loss)
        torch.nn.utils.clip_grad_norm_(self.discriminator.parameters(), max_norm=1.0)
        opt_disc.step()
        opt_disc.zero_grad()
        self.untoggle_optimizer(opt_disc)

        # === Logging ===
        self.log_dict({
            "gen/elbo": elbo,
            "gen/adv": adv_loss,
            "gen/fm": fm_loss,
            "gen/total": total_gen_loss,
            "disc/loss": d_loss,
        }, prog_bar=True, on_step=True, on_epoch=True)

    @torch.no_grad()
    def generate(self, n_samples: int, dur: int = DEFAULT_AUDIO_DUR):
        """
        This is kinda shit - the Gaussian prior on the latent space is a little too weak
        """
        return self.vae.generate(n_samples, dur)

    def configure_optimizers(self):
        opt_g = torch.optim.Adam(self.vae.parameters(), lr=self.lr)
        opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=self.lr * 2)
        return [opt_g, opt_d]
