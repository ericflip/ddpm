import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Union
from nn import Swish, ResNetBlock, Upsample, Downsample, AttentionBlock, TimeSequential


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
        input_block_chs = [model_channels]
        num_resolutions = len(channel_mult)
        ch = model_channels

        # (N, in_channels, H, W) -> (N, model_channels, H, W)
        self.in_layer = nn.Conv2d(
            in_channels, model_channels, kernel_size=3, stride=1, padding=1
        )

        self.downsamples = nn.ModuleList([])

        for i, ch_mult in enumerate(channel_mult):
            # add residual blocks
            for j in range(num_res_blocks):
                layers = [
                    ResNetBlock(
                        ch,
                        model_channels * ch_mult,
                        time_channels=time_embed_channels,
                        dropout=dropout,
                    )
                ]

                # set output channels of this block to input channels for next block
                ch = model_channels * ch_mult

                # apply attention at resolution
                resolution = 2**i

                if resolution in attention_resolutions:
                    layers.append(AttentionBlock(ch))

                # keep track of output channels
                input_block_chs.append(ch)
                self.downsamples.append(TimeSequential(*layers))

            # downsample if not last layer
            if i != num_resolutions - 1:
                self.downsamples.append(Downsample(ch, with_conv=conv_resample))
                input_block_chs.append(ch)

        # middle
        self.middle_blocks = TimeSequential(
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
        )

        # print(input_block_chs)

        self.upsamples = nn.ModuleList([])

        # upsampling
        for i, ch_mult in enumerate(channel_mult[::-1]):
            for j in range(num_res_blocks + 1):
                layers = [
                    ResNetBlock(
                        ch + input_block_chs.pop(),  # add b/c of skip connections
                        model_channels * ch_mult,
                        time_channels=time_embed_channels,
                        dropout=dropout,
                    )
                ]

                # set input channels for next block
                ch = model_channels * ch_mult

                # apply attention at resolution
                resolution = 2 ** (num_resolutions - 1 - i)
                if resolution in attention_resolutions:
                    layers.append(AttentionBlock(ch))

                # upsample if not last layer
                if i != num_resolutions - 1 and j == num_res_blocks:
                    layers.append(Upsample(ch, with_conv=conv_resample))

                self.upsamples.append(TimeSequential(*layers))

        # print(len(self.upsamples))

        # out
        self.out_layer = nn.Sequential(
            nn.GroupNorm(32, ch),
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
        temb = self.time_embed(timestep_embedding(t, self.model_channels))
        assert temb.shape == torch.Size([N, self.model_channels * 4])

        # downsampling
        x = self.in_layer(x)

        hs = [x]
        for layer in self.downsamples:
            x = layer(x, temb)
            hs.append(x)

        # middle block
        x = self.middle_blocks(x, temb)

        # print([z.shape for z in hs])

        # print(x.shape)

        # upsampling
        for layer in self.upsamples:
            # skip connection
            # print(x.shape, hs[-1].shape)
            x = torch.cat((x, hs.pop()), dim=1)
            x = layer(x, temb)

            # print(x.shape)

        # out layer
        x = self.out_layer(x)

        # print(x.shape)

        return x
