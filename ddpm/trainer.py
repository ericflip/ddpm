import json
import os

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from ddpm.gaussian_diffusion import GaussianDiffusion
from ddpm.pipeline import DDPMPipeline
from ddpm.unet import UNet
from ddpm.utils import make_image_grid


def generate_samples(diffusion: GaussianDiffusion, grid_size=8):
    pipeline = DDPMPipeline(diffusion)
    images = pipeline(num_images=grid_size**2, image_size=32).images

    grid = make_image_grid(images, grid_size, grid_size, resize=128)

    return grid


class Trainer:
    def __init__(
        self,
        diffusion: GaussianDiffusion,
        train_loader: DataLoader,
        batch_size=32,
        lr=3e-4,
        epochs=50,
        checkpoint_epochs=10,
        save_dir: str = "./train_checkpoints",
        generate_samples: bool = True,
        num_samples: int = 25,
        device="cuda",
    ):
        self.diffusion = diffusion
        self.train_loader = train_loader
        self.batch_size = batch_size
        self.lr = lr
        self.epochs = epochs
        self.checkpoint_epochs = checkpoint_epochs
        self.save_dir = save_dir
        self.generate_samples = generate_samples

        if self.generate_samples:
            assert (
                int(num_samples**0.5) ** 2 == num_samples
            ), "`num_samples` must be a perfect square"

        self.num_samples = num_samples
        self.optimizer = torch.optim.Adam(self.diffusion.model.parameters(), lr=self.lr)
        self.device = device

    def save_model_checkpoint(self, epoch: int = None):
        save_dir = (
            self.save_dir
            if epoch is None
            else os.path.join(self.save_dir, f"epoch-{epoch}")
        )

        os.makedirs(save_dir, exist_ok=True)

        # save model state dict
        model_path = os.path.join(save_dir, "model.pt")
        torch.save(self.diffusion.model.state_dict(), model_path)

        # save model config
        config_path = os.path.join(save_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(self.diffusion.model.config, f, indent=4)

        # save noise schedule
        noise_schedule_path = os.path.join(save_dir, "schedule.json")
        with open(noise_schedule_path, "w") as f:
            json.dump(self.diffusion.noise_schedule.config, f, indent=4)

    def generate_and_save_samples(self, epoch: int = None):
        save_dir = (
            self.save_dir
            if epoch is None
            else os.path.join(self.save_dir, f"epoch-{epoch}")
        )

        os.makedirs(save_dir, exist_ok=True)
        samples = generate_samples(self.diffusion, grid_size=int(self.num_samples**0.5))
        samples.save(os.path.join(save_dir, "samples.png"))

    def train(self):
        print("[Training...]")

        # create dir for checkpointing
        os.makedirs(self.save_dir, exist_ok=True)

        # create progress bar
        total_steps = self.epochs * len(self.train_loader)
        pbar = tqdm(total=total_steps, position=0)

        # move model and diffusion to device
        self.diffusion.to(self.device)

        # train loop
        for i in range(self.epochs):
            for X, _ in self.train_loader:
                N = X.shape[0]
                x_0 = X.to(self.device)

                # normalize x between -1 and 1
                x_0 = (x_0 - (1 / 2)) * 2

                # sample timesteps uniformly
                t = torch.randint(0, self.diffusion.noise_schedule.timesteps, (N,)).to(
                    self.device
                )

                # noise inputs
                noise = torch.randn_like(x_0).to(self.device)
                x_t = self.diffusion.q_sample(x_0, t, noise)

                # predict noise
                pred = self.diffusion.model(x_t, t)
                loss = F.mse_loss(pred, noise)

                # backprop
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

                pbar.update(1)
                pbar.set_postfix(loss=loss.item())

            epoch = i + 1

            if epoch % self.checkpoint_epochs == 0:
                # save model state
                tqdm.write(f"saving model at epoch {epoch}...")
                self.save_model_checkpoint(epoch)

                # generate samples
                if self.generate_samples:
                    tqdm.write("Generating Samples...")
                    self.generate_and_save_samples(epoch)

        # save final model
        tqdm.write("Saving final model...")
        self.save_model_checkpoint()

        if self.generate_samples:
            tqdm.write("Generating Samples...")
            self.generate_and_save_samples()

        tqdm.write("Done!")
