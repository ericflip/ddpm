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
        -torch.log(max_period)
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

    def forward(self, x: torch.Tensor, t: torch.Tensor):
        """
        Params:
            - x: (N, C, W, H) batch of images
            - t: (N, T) batch of time embeddings
        """

        # save for residual connection
        h = x
        h = self.in_layers(h)

        # add in timestep embeddings
        h += self.time_layers(t)

        h = self.out_layers(h)

        assert x.shape == h.shape

        return x + h


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        model_channels: int,
        out_channels: int,
        num_res_blocks: list[int],
        attention_resolutions: list[int],
        dropout=0,
        channel_mult=(1, 2, 4, 8),
        conv_resample=True,
    ):
        pass
