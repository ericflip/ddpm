import torch


def linear_beta_schedule(beta_start: float, beta_end: float, timesteps: int):
    """
    Linear schedule, proposed in original ddpm paper
    """
    return torch.linspace(beta_start, beta_end, timesteps)


class GaussianDiffusion:
    """
    Utilities for training and sampling diffusion models
    """

    def __init__(self, beta_start: float, beta_end: float, timesteps: int):
        self.betas = linear_beta_schedule(beta_start, beta_end, timesteps)
        self.alphas = 1 - self.betas
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)

    @property
    def T(self):
        return self.betas.shape[0]

    def q_mean_var(self, t: int):
        """
        Get the mean and variance for q(x_t | x_0)
        """
        assert 1 <= t <= self.T

        mean = self.alpha_bar[t]
        var = 1 - self.alpha_bar[t]

        return mean, var

    def q_sample(self, x_start: torch.Tensor, t: int):
        """
        Sample from q(x_t | x_0)

        Params:
            - x_start: noiseless sample from distribution (N, C, H, W)
            - t: timestep, t=1...T
        """
        assert 1 <= t <= self.T

        mean, var = self.q_mean_var(t)
        eps = torch.rand_like(x_start)
        sample = mean * x_start + (var**0.5) * eps

        return sample
