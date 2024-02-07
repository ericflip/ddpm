import math

import torch
import torch.nn as nn
import torch.nn.functional as F


class Swish(nn.Module):
    """
    swish(x) = x * sigmoid(x)
    """

    def forward(self, x: torch.Tensor):
        return x * F.sigmoid(x)


class Upsample(nn.Module):
    """
    Upsample dimensions by 2x using nearest neighbor and applies Conv2d if
    `with_conv` is `True`
    """

    def __init__(self, n_channels: int, with_conv: bool):
        super().__init__()
        self.with_conv = with_conv
        self.upsample = nn.Upsample(scale_factor=2, mode="nearest")

        if with_conv:
            self.conv = nn.Conv2d(
                n_channels, n_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x: torch.Tensor, t: torch.Tensor = None):
        x = self.upsample(x)

        if self.with_conv:
            x = self.conv(x)

        return x


class Downsample(nn.Module):
    """
    Downsample dimensions by 2x using conv if `with_conv` is `True`, otherwise average pooling
    """

    def __init__(self, n_channels: int, with_conv: bool):
        super().__init__()

        if with_conv:
            self.downsample = nn.Conv2d(
                n_channels, n_channels, kernel_size=3, stride=2, padding=1
            )
        else:
            self.downsample = nn.AvgPool2d(kernel_size=2, stride=2)

    def forward(self, x: torch.Tensor, t: torch.Tensor = None):
        return self.downsample(x)


# https://github.com/openai/improved-diffusion/blob/main/improved_diffusion/nn.py
def timestep_embedding(timesteps, dim, max_period=10000):
    """
    Create sinusoidal timestep embeddings.

    :param timesteps: a 1-D Tensor of N indices, one per batch element.
                      These may be fractional.
    :param dim: the dimension of the output.
    :param max_period: controls the minimum frequency of the embeddings.
    :return: an [N x dim] Tensor of positional embeddings.
    """
    half = dim // 2
    freqs = torch.exp(
        -math.log(max_period)
        * torch.arange(start=0, end=half, dtype=torch.float32)
        / half
    ).to(device=timesteps.device)
    args = timesteps[:, None].float() * freqs[None]
    embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
    return embedding


class ResNetBlock(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        time_channels: int,
        n_groups: int = 32,
        dropout=0.5,
    ):
        super().__init__()
        self.in_layers = nn.Sequential(
            nn.GroupNorm(n_groups, in_channels),
            Swish(),
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )
        self.time_layers = nn.Sequential(
            Swish(), nn.Linear(time_channels, out_channels)
        )
        self.out_layers = nn.Sequential(
            nn.GroupNorm(n_groups, out_channels),
            Swish(),
            nn.Dropout(dropout),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

        # make sure when u add residual connections, the channel dimensions line up
        if in_channels == out_channels:
            self.skip_connection = nn.Identity()
        else:
            self.skip_connection = nn.Conv2d(
                in_channels, out_channels, kernel_size=3, stride=1, padding=1
            )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Params:
            - x: (N, C, W, H) batch of images
            - t: (N, T) batch of time embeddings
        """

        # save for residual connection
        h = x
        h = self.in_layers(h)

        # add in timestep embeddings;
        time_embed = self.time_layers(t)[:, :, None, None]  # (N, C) -> (N, C, 1, 1)
        h += time_embed
        h = self.out_layers(h)

        # pass x through skip connection
        x = self.skip_connection(x)

        assert x.shape == h.shape

        return x + h


class AttentionBlock(nn.Module):
    def __init__(self, channels: int, num_heads=1):
        super().__init__()
        self.channels = channels
        self.norm = nn.GroupNorm(32, channels)
        self.attention = nn.MultiheadAttention(
            embed_dim=channels, num_heads=num_heads, batch_first=True
        )
        self.out = nn.Conv1d(channels, channels, kernel_size=1)

    def forward(self, x: torch.Tensor, t: torch.Tensor = None):
        """
        Params:
            - x: (N, C, H, W)
        """
        N, C, H, W = x.shape

        x = self.norm(x)

        # reshape for attention, each pixel is a "word" in the sequence
        h = x.view(N, self.channels, -1).permute(
            (0, 2, 1)
        )  # (N, C, H, W) -> (N, H*W, C)

        # perform self attention pixel wise
        h, _ = self.attention(h, h, h)

        # out projection
        h = h.permute((0, 2, 1))  # (N, H*W, C) -> (N, C, H*W)
        h = self.out(h)

        # add skip connection
        x = x + h.view(N, C, H, W)

        return x


class TimeSequential(nn.Module):
    def __init__(self, *layers):
        super().__init__()
        self.layers = nn.ModuleList(layers)

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        for layer in self.layers:
            x = layer(x, t)

        return x
