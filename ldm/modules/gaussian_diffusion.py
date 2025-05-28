import torch
import torch.nn.functional as F

class GaussianDiffusion:
    def __init__(self, timesteps=1000, beta_start=1e-4, beta_end=0.02):
        self.timesteps = timesteps
        self.betas = torch.linspace(beta_start, beta_end, timesteps)
        self.alphas = 1.0 - self.betas
        self.alpha_cumprod = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_cumprod = torch.sqrt(self.alpha_cumprod)
        self.sqrt_one_minus_alpha_cumprod = torch.sqrt(1.0 - self.alpha_cumprod)

    def q_sample(self, x_start, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x_start)
        sqrt_alpha_cumprod_t = self.sqrt_alpha_cumprod[t].view(-1, 1, 1)
        sqrt_one_minus_alpha_cumprod_t = self.sqrt_one_minus_alpha_cumprod[t].view(-1, 1, 1)
        return sqrt_alpha_cumprod_t * x_start + sqrt_one_minus_alpha_cumprod_t * noise

    def p_losses(self, model, x_start, t):
        noise = torch.randn_like(x_start)
        x_noisy = self.q_sample(x_start, t, noise=noise)
        predicted_noise = model(x_noisy, t)
        return F.mse_loss(predicted_noise, noise)

    @torch.no_grad()
    def sample(self, model, shape, device):
        x = torch.randn(shape, device=device)
        for i in reversed(range(self.timesteps)):
            t = torch.full((shape[0],), i, device=device, dtype=torch.long)
            alpha_cumprod = self.alpha_cumprod[t].view(-1, 1, 1).to(device)
            one_minus_alpha_cumprod = 1.0 - alpha_cumprod
            sqrt_one_minus_alpha_cumprod = torch.sqrt(one_minus_alpha_cumprod)

            noise_pred = model(x, t)
            x0_pred = (x - sqrt_one_minus_alpha_cumprod * noise_pred) / torch.sqrt(alpha_cumprod)

            if i > 0:
                noise = torch.randn_like(x)
                x = torch.sqrt(self.alphas[t]) * x0_pred + torch.sqrt(self.betas[t]) * noise
            else:
                x = x0_pred
        return x
