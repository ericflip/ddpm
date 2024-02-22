import argparse
import json
import logging
import os

import torch
import torch.nn.functional as F
import torchvision.transforms.v2 as v2
from pipeline import DDPMPipeline
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import FashionMNIST
from tqdm import tqdm
from unet import UNet
from utils import make_image_grid

from diffusion import GaussianDiffusion


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpointing_epochs", type=float, default=100)
    parser.add_argument("--outdir", type=str, default="./train_fasion_mnist")

    # model args
    parser.add_argument("--model_channels", type=int, default=64)
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--attention_resolutions", type=int, nargs="+", default=[2])
    parser.add_argument("--channel_mult", type=int, nargs="+", default=[1, 2, 4, 8])

    args = parser.parse_args()

    return args


def save_model_checkpoint(model: UNet, epoch: int, outdir: str):
    folder_path = os.path.join(outdir, f"epoch-{epoch}")
    os.makedirs(folder_path, exist_ok=True)

    # save model state dict
    model_path = os.path.join(folder_path, "model.pt")
    torch.save(model.state_dict(), model_path)

    # save model config
    config_path = os.path.join(folder_path, "config.json")
    with open(config_path, "w") as f:
        json.dump(model.config, f, indent=4)


def save_model(model: UNet, outdir: str):
    # save model state dict
    model_path = os.path.join(outdir, "model.pt")
    torch.save(model.state_dict(), model_path)

    # save model config
    config_path = os.path.join(outdir, "config.json")
    with open(config_path, "w") as f:
        json.dump(model.config, f, indent=4)


def generate_samples(unet: UNet, diffusion: GaussianDiffusion, grid_size=8):
    pipeline = DDPMPipeline(unet, diffusion)
    images = pipeline(num_images=grid_size**2, image_size=32)

    grid = make_image_grid(images, grid_size, grid_size, resize=128)

    return grid


if __name__ == "__main__":
    args = parse_args()

    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    checkpointing_epochs = args.checkpointing_epochs
    outdir = args.outdir

    # make outdir
    os.makedirs(outdir, exist_ok=True)

    # prepare dataset and dataloader
    dataset_path = "./data"
    dataset = FashionMNIST(
        root=dataset_path,
        train=True,
        transform=v2.Compose(
            [
                v2.ToImage(),
                v2.ToDtype(torch.float32, scale=True),
                v2.Resize((32, 32), antialias=False),
            ]
        ),
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # prepare model
    unet = UNet(
        in_channels=1,
        model_channels=args.model_channels,
        out_channels=1,
        channel_mult=args.channel_mult,
        attention_resolutions=args.attention_resolutions,
        num_res_blocks=args.num_res_blocks,
    ).to("cuda")

    # create diffusion model
    beta_start = 1e-4
    beta_end = 0.02
    diffusion = GaussianDiffusion(
        beta_start=beta_start,
        beta_end=beta_end,
        timesteps=1000,
        device="cuda",
    )

    # prepare optimizer
    optimizer = torch.optim.Adam(unet.parameters(), lr=lr)

    # create progress bar
    total_steps = epochs * len(loader)
    pbar = tqdm(total=total_steps)

    # train loop
    for i in range(epochs):
        for X, _ in loader:
            N = X.shape[0]
            x_0 = X.to("cuda")

            # normalize x between -1 and 1
            x_0 = (x_0 - (1 / 2)) * 2

            # sample timesteps uniformly
            t = diffusion.sample_timestep(N).to("cuda")

            # noise inputs
            noise = torch.randn_like(x_0).to("cuda")
            x_t = diffusion.q_sample(x_0, t, noise)

            # predict noise
            pred = unet(x_t, t)

            loss = F.mse_loss(pred, noise)

            # backprop
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            pbar.update(1)

            pbar.set_postfix(loss=loss.item())

        epoch = i + 1

        if epoch % checkpointing_epochs == 0:
            # save model state
            print(f"saving model at epoch {epoch}...")
            save_model_checkpoint(unet, epoch, outdir)

            # generate samples
            samples = generate_samples(unet, diffusion)
            samples_path = os.path.join(outdir, f"epoch-{epoch}", "samples.png")
            samples.save(samples_path)

    # save final model
    save_model(unet, outdir)

    samples = generate_samples(unet, diffusion)
    samples_path = os.path.join(outdir, "samples.png")
    samples.save(samples_path)

    print("Done!")
