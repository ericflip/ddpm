import argparse
from torchvision.datasets import FashionMNIST
import torchvision.transforms.v2 as v2
from torch.utils.data import DataLoader
from unet import UNet
from tqdm import tqdm
from diffusion import GaussianDiffusion
import torch
import torch.nn.functional as F


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--checkpointing_steps", type=float, default=100)

    args = parser.parse_args()

    return args


if __name__ == "__main__":
    args = parse_args()

    batch_size = args.batch_size
    epochs = args.epochs
    lr = args.lr
    checkpointing_steps = args.checkpointing_steps

    # prepare dataset and dataloader
    dataset_path = "./data"
    dataset = FashionMNIST(
        root=dataset_path,
        train=True,
        transform=v2.Compose([v2.ToTensor(), v2.Resize((32, 32))]),
    )

    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # prepare model
    unet = UNet(
        in_channels=1,
        model_channels=64,
        out_channels=1,
        channel_mult=(1, 2, 4),
        attention_resolutions=[16],
        num_res_blocks=2,
    ).to("cuda")

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
            # print(t.shape)
            # print(noise.shape)
            # print(x_0.shape)
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
