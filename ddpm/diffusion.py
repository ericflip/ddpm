import torch


def linear_beta_schedule(beta_start: float, beta_end: float, timesteps: int):
    return torch.linspace(beta_start, beta_end, timesteps)
