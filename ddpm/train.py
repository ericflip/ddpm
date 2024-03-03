import argparse
import os

import torch
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader, Subset
from torchvision.datasets import FashionMNIST

from ddpm.gaussian_diffusion import GaussianDiffusion
from ddpm.schedule import NoiseSchedule
from ddpm.trainer import Trainer
from ddpm.unet import UNet


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--checkpoint_epochs", type=float, default=100)
    parser.add_argument("--outdir", type=str, default="./train_fasion_mnist")

    # schedule args
    parser.add_argument("--beta_start", type=float, default=1e-4)
    parser.add_argument("--beta_end", type=float, default=0.02)
    parser.add_argument("--timesteps", type=int, default=1000)
    parser.add_argument("--schedule_type", type=str, default="linear")

    # model args
    parser.add_argument("--model_channels", type=int, default=64)
    parser.add_argument("--num_res_blocks", type=int, default=2)
    parser.add_argument("--attention_resolutions", type=int, nargs="+", default=[2])
    parser.add_argument("--channel_mult", type=int, nargs="+", default=[1, 2, 4, 8])

    args = parser.parse_args()

    return args


def main(args):
    args = parse_args()

    # make outdir
    os.makedirs(args.outdir, exist_ok=True)

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

    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True)

    # prepare model
    unet = UNet(
        in_channels=1,
        model_channels=args.model_channels,
        out_channels=1,
        channel_mult=args.channel_mult,
        attention_resolutions=args.attention_resolutions,
        num_res_blocks=args.num_res_blocks,
    ).to("cuda")

    # prepare noise schedule
    noise_schedule = NoiseSchedule(
        beta_end=args.beta_end,
        beta_start=args.beta_start,
        timesteps=args.timesteps,
        schedule_type=args.schedule_type,
    )

    # prepare diffusion
    diffusion = GaussianDiffusion(noise_schedule=noise_schedule, model=unet)

    # initialize trainer and train
    trainer = Trainer(
        diffusion=diffusion,
        train_loader=loader,
        batch_size=args.batch_size,
        lr=args.lr,
        epochs=args.epochs,
        checkpoint_epochs=args.checkpoint_epochs,
        save_dir=args.outdir,
    )

    trainer.train()


if __name__ == "__main__":
    args = parse_args()
    main(args)
