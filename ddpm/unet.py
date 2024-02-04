import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union


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

    def forward(self, x: torch.Tensor):
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

    def forward(self, x: torch.Tensor):
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

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Params:
            - x: (N, C, H, W)
        """
        pass


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: int,
        attention_resolutions: list,
        dropout=0,
        channel_mult: Union[list[int], tuple[int]] = (1, 2, 4, 8),
        conv_resample=True,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.model_channels = model_channels

        # time embeddings
        time_embed_channels = model_channels * 4
        self.time_embed = nn.Sequential(
            nn.Linear(model_channels, time_embed_channels),
            Swish(),
            nn.Linear(time_embed_channels, time_embed_channels),
        )

        # downsampling
        self.in_layer = nn.Conv2d(
            in_channels, model_channels, kernel_size=3, stride=1, padding=1
        )
        self.downsamples = nn.ModuleList([])

        num_resolutions = len(channel_mult)
        ch = model_channels

        for i, ch_mult in enumerate(channel_mult):
            layers = []
            for j in range(num_res_blocks):
                in_channels = ch
                out_channels = model_channels * ch_mult

                resnet_block = ResNetBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    time_channels=time_embed_channels,
                    dropout=dropout,
                )

                layers.append(resnet_block)

                ch = out_channels
                resolution = 2**i

                if resolution in attention_resolutions:
                    layers.append(AttentionBlock(out_channels))

            # downsample if not last layer
            if i != num_resolutions - 1:
                layers.append(Downsample(out_channels, with_conv=conv_resample))

            self.downsamples.append(nn.ModuleList(layers))

        # middle
        self.middle_blocks = nn.ModuleList(
            [
                ResNetBlock(
                    ch,
                    ch,
                    time_embed_channels,
                    dropout=dropout,
                ),
                AttentionBlock(ch),
                ResNetBlock(
                    ch,
                    ch,
                    time_embed_channels,
                    dropout=dropout,
                ),
            ]
        )

        # upsampling
        # TODO: understand what the fuck is going on herew
        for i, ch_mult in enumerate(channel_mult[::-1]):
            layers = []

            for j in range(num_res_blocks + 1):
                pass

            if i != 0:
                layers.append(Upsample())

        # out
        self.out = nn.Sequential(
            # nn.GroupNorm(32, )
            Swish(),
            nn.Conv2d(model_channels, out_channels, kernel_size=3, stride=1, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Params:
            - x: (N, C, H, W) batch of images
            - t: (N, ) batch of timesteps
        """

        assert x.shape[0] == t.shape[0]
        N = x.shape[0]

        # timestep embeddings
        t = self.time_embed(timestep_embedding(t, self.model_channels))
        assert t.shape == torch.Size([N, self.model_channels * 4])

        # downsampling

        # middle block

        # upsampling
