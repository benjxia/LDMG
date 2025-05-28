import torch
from torch import nn
import torch.nn.functional as F
import auraloss

from ldm.modules import DEFAULT_INPUT_SR

class ELBO_Loss(nn.Module):
    def __init__(self, kl_weight=1e-3, sample_rate: int=DEFAULT_INPUT_SR):
        super().__init__()
        self.stft_loss = auraloss.freq.MultiResolutionSTFTLoss(perceptual_weighting=True, sample_rate=sample_rate)
        self.kl_weight = kl_weight

    def forward(self, recon, real, mu, logvar):
        stft = self.stft_loss(recon, real)
        kl = -0.5 * torch.mean(torch.sum(1 + logvar - mu.pow(2) - logvar.exp(), dim=1))
        return stft + self.kl_weight * kl

# TODO: make this a proper pytorch module
def feature_matching_loss(real_feats, fake_feats):
    loss = 0
    for real, fake in zip(real_feats, fake_feats):
        loss += F.l1_loss(real, fake)
    return loss
