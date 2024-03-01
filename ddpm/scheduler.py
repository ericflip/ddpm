import torch


def linear_beta_schedule(beta_start: float, beta_end: float, timesteps: int):
    """
    Linear schedule, proposed in original ddpm paper
    """
    return torch.linspace(beta_start, beta_end, timesteps)


class NoiseScheduler:
    def __init__(
        self, beta_start: float, beta_end: float, timesteps: int, schedule_type="linear"
    ):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps
        self.schedule_type = schedule_type

        if schedule_type == "linear":
            self.beta = linear_beta_schedule(beta_start, beta_end, timesteps)

        self.alpha = 1 - self.beta
        self.alpha_bar = torch.cumprod(self.alphas, dim=0)
        self.device = "cpu"

    @property
    def config(self):
        return {
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "timesteps": self.timesteps,
            "schedule_type": self.schedule_type,
        }

    def to(self, device):
        self.device = device

        self.beta = self.beta.to(device)
        self.alpha = self.alpha.to(device)
        self.alpha_bar = self.alpha_bar.to(device)

        return self
