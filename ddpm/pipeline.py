from typing import NamedTuple, Union

import torch
from PIL import Image

from ddpm.gaussian_diffusion import GaussianDiffusion
from ddpm.unet import UNet
from ddpm.utils import batch_to_images


class DDPMPipelineOutput(NamedTuple):
    images: list[Image.Image]
    samples: Union[list[list[torch.Tensor]], None]


class DDPMPipeline:
    @staticmethod
    def from_checkpoint(checkpoint_path: str):
        pass
        # unet = UNet.from_checkpoint(checkpoint_path)

        # # TODO: load diffusion from config
        # beta_start = 1e-4
        # beta_end = 0.02
        # diffusion = GaussianDiffusion(
        #     beta_start=beta_start,
        #     beta_end=beta_end,
        #     timesteps=1000,
        # )

        # return DDPMPipeline(unet, diffusion)

    def __init__(self, diffusion: GaussianDiffusion):
        self.diffusion = diffusion
        self.device = diffusion.device

    def __call__(
        self,
        num_images: int = 1,
        image_size: int = 32,
        clip_denoised: bool = True,
        output_samples: bool = False,
    ) -> DDPMPipelineOutput:
        """
        Generate samples from DDPM
        """
        # initialize noise
        noise = torch.randn(
            (num_images, self.diffusion.model.in_channels, image_size, image_size)
        ).to(self.device)

        # sample x_0 from diffusion
        samples_output = self.diffusion.sample(
            noise, clip_denoised=clip_denoised, output_samples=output_samples
        )

        x_0 = samples_output.x_0
        samples = samples_output.samples

        # normalize grayscale images between 0 and 1
        C = x_0.shape[1]
        if C == 1:
            x_0 = (x_0 + 1) / 2

        # convert tensors to images
        images = batch_to_images(x_0)

        return DDPMPipelineOutput(images=images, samples=samples)

    def to(self, device: str):
        self.device = device
        self.unet = self.unet.to(device)
        self.diffusion = self.diffusion.to(device)

        return self
