class NoiseScheduler:
    def __init__(self, beta_start: float, beta_end: float, timesteps: int):
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.timesteps = timesteps

    @property
    def config(self):
        return {
            "beta_start": self.beta_start,
            "beta_end": self.beta_end,
            "timesteps": self.timesteps,
        }
    

    def to(self, device):
        pass
