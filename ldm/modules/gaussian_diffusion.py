import torch
import torch.nn.functional as F
from tqdm import tqdm

class GaussianDiffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alpha_cumprod_prev = torch.cat([torch.tensor([1.0]), self.alpha_cumprod[:-1]], dim=0)

        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)
        self.posterior_variance = self.betas * (1.0 - self.alpha_cumprod_prev) / (1.0 - self.alpha_cumprod)

    def to(self, device):
        for name in dir(self):
            attr = getattr(self, name)
            if isinstance(attr, torch.Tensor):
                setattr(self, name, attr.to(device))

    def q_sample(self, x_start, t, noise=None):
        self.to(x_start.device)
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alpha_cumprod_t = self._extract(self.sqrt_alpha_cumprod, t, x_start.shape)
        sqrt_one_minus_alpha_cumprod_t = self._extract(self.sqrt_one_minus_alpha_cumprod, t, x_start.shape)

        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

    def p_losses(self, model, x_start, t):
        self.to(x_start.device)
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise)
        predicted_noise = model(x_noisy, t)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(self, model, shape, device):
        self.to(device)
        x = torch.randn(shape, device=device)

        for i in tqdm(reversed(range(self.timesteps)), total=self.timesteps):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)

            betas_t = self._extract(self.betas, t, x.shape)
            sqrt_recip_alphas_t = self._extract(self.sqrt_recip_alphas, t, x.shape)
            sqrt_one_minus_alpha_cumprod_t = self._extract(self.sqrt_one_minus_alpha_cumprod, t, x.shape)

            model_mean = sqrt_recip_alphas_t * (
                x - betas_t * model(x, t) / sqrt_one_minus_alpha_cumprod_t
            )

            if i > 0:
                posterior_variance_t = self._extract(self.posterior_variance, t, x.shape)
                noise = torch.randn_like(x)
                x = model_mean + torch.sqrt(posterior_variance_t) * noise
            else:
                x = model_mean

        return x

    def _extract(self, a, t, x_shape):
        """Extract values from 1-D tensor `a` at positions `t` and reshape to `x_shape`."""
        out = a.gather(0, t)
        while len(out.shape) < len(x_shape):
            out = out.unsqueeze(-1)
        return out
