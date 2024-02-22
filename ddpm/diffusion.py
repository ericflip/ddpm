from typing import NamedTuple

import torch
from tqdm import tqdm


def linear_beta_schedule(beta_start: float, beta_end: float, timesteps: int):
    """
    Linear schedule, proposed in original ddpm paper
    """
    return torch.linspace(beta_start, beta_end, timesteps)


def extract(a: torch.Tensor, shape: torch.Size):
    """
    Make `a` broadcastable to tensor of size `shape`. The first dimension of `a` and `shape` must match.

    ie.
    a.shape = (3, )
    shape = (3, 32, 16, 16)

    a.shape -> (3, 1, 1, 1)
    """

    assert a.shape[0] == shape[0]

    while a.dim() < len(shape):
        a = a.unsqueeze(-1)

    return a


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models
    """

    def __init__(
        self,
        beta_start: float,
        beta_end: float,
        timesteps: int,
        device: torch.device = "cpu",
    ):
        self.betas = linear_beta_schedule(beta_start, beta_end, timesteps).to(device)
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.timesteps = timesteps

    @property
    def T(self):
        return self.timesteps

    def q_mean_var(self, x_0: torch.Tensor, t: int):
        """
        Get the mean and variance for q(x_t | x_0)

        - mu = sqrt(alpha_bar_t) * x_t
        - var = 1 - alpha_bat_t
        """
        mean = extract((self.alpha_bar[t] ** 0.5), x_0.shape) * x_0
        var = extract(1 - self.alpha_bar[t], x_0.shape)

        return mean, var

    def q_sample(self, x_0: torch.Tensor, t: torch.Tensor, noise=None):
        """
        Sample from q(x_t | x_0)

        Params:
            - x_0: noiseless sample from distribution (N, C, H, W)
            - t: timestep, t=1...T (N, )
        """

        if noise is None:
            noise = torch.randn_like(x_0).to(x_0.device)

        mean, var = self.q_mean_var(x_0, t)
        sample = mean + (var**0.5) * noise

        return sample

    def q_posterior_mean_var(
        self, x_t: torch.Tensor, x_0: torch.Tensor, t: torch.Tensor
    ):
        """
        Gets the mean and variance for posterior q(x_t-1 | x_t, x_0)
        """

        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bar[t]
        alpha_bar_t_1 = self.alpha_bar[t - 1]
        beta_t = self.betas[t]

        mean = (((alpha_t**0.5) * (1 - alpha_bar_t_1)) / (1 - alpha_bar_t)) * x_t + (
            (alpha_bar_t_1**0.5) / (1 - alpha_bar_t) * beta_t
        ) * x_0

        variance = (1 - alpha_bar_t_1) / (1 - alpha_bar_t) * beta_t

        return mean, variance

    def p_mean_variance(
        self, model: torch.nn.Module, x_t: torch.Tensor, t: torch.Tensor
    ):
        """ """

        alpha_bar_t_1 = self.alpha_bar[t - 1]
        alpha_bar_t = self.alpha_bar[t]
        beta_t = self.betas[t]

        # predict noise
        eps = model(x_t, t)

        # calculate parameterized mean and variance
        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bar[t]

        mean = extract(1 / alpha_t**0.5, x_t.shape) * (
            x_t - extract((1 - alpha_t) / ((1 - alpha_bar_t) ** 0.5), x_t.shape) * eps
        )

        var = (1 - alpha_bar_t_1) / (1 - alpha_bar_t) * beta_t
        var = extract(var, x_t.shape)

        return mean, var

    def p_sample(
        self,
        model: torch.nn.Module,
        x_t: torch.Tensor,
        t: torch.Tensor,
    ):
        """ """
        z = torch.randn_like(x_t)
        mean, var = self.p_mean_variance(model, x_t, t)

        sample = mean + (var**0.5) * z

        return sample

    def sample(
        self, model: torch.nn.Module, noise=None, clip_denoised=True, device="cuda"
    ):
        if noise is None:
            noise = torch.randn((1, 3, 32, 32)).to(device)

        N = noise.shape[0]
        x_t = noise

        model.to(device)
        model.eval()

        with torch.no_grad():
            for t in tqdm(range(self.T - 1, 0, -1)):
                t = torch.tensor([t] * N).to(device)
                x_t = self.p_sample(model, x_t, t)

        if clip_denoised:
            x_t = x_t.clamp(-1, 1)

        return x_t

    def sample_timestep(self, num_time_steps=1):
        return torch.randint(0, self.T, (num_time_steps,))
