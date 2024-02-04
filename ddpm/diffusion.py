import torch


def linear_beta_schedule(beta_start: float, beta_end: float, timesteps: int):
    """
    Linear schedule, proposed in original ddpm paper
    """
    return torch.linspace(beta_start, beta_end, timesteps)


# TODO: make t a tensor
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

    def q_sample(self, x_0: torch.Tensor, t: int):
        """
        Sample from q(x_t | x_0)

        Params:
            - x_0: noiseless sample from distribution (N, C, H, W)
            - t: timestep, t=1...T
        """
        assert 1 <= t <= self.T

        mean, var = self.q_mean_var(t)
        eps = torch.rand_like(x_0)
        sample = mean * x_0 + (var**0.5) * eps

        return sample

    def q_posterior_mean_var(self, x_t: torch.Tensor, x_0: torch.Tensor, t: int):
        """
        Gets the mean and variance for posterior q(x_t-1 | x_t, x_0)
        """
        assert 1 < t <= self.T

        t = t - 1

        alpha_t = self.alphas[t]
        alpha_bar_t = self.alpha_bar[t]
        alpha_bar_t_1 = self.alpha_bar[t - 1]
        beta_t = self.betas[t]

        mean = (((alpha_t**0.5) * (1 - alpha_bar_t_1)) / (1 - alpha_bar_t)) * x_t + (
            (alpha_bar_t_1**0.5) / (1 - alpha_bar_t) * beta_t
        ) * x_0

        variance = (1 - alpha_bar_t_1) / (1 - alpha_bar_t) * beta_t

        return mean, variance
    
    def p_mean_variance(self, model, x: torch.Tensor):
        pass
    
    def p_sample(self, model, x: torch.Tensor, ):
        pass
