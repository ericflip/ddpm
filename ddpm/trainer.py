from torch.utils.data import DataLoader

from diffusion import GaussianDiffusion


class Trainer:
    def __init__(
        self,
        diffusion: GaussianDiffusion,
        train_loader: DataLoader,
        train_batch_size=32,
        train_lr=3e-4,
        epochs=50,
        save_dir: str = "./train_checkpoints",
    ):
        pass

    def save_model_checkpoint(self):
        pass

    def train(self):
        pass
