import torch
from PIL import Image
from unet import UNet
from utils import batch_to_images

from diffusion import GaussianDiffusion


class DDPMPipeline:
    def __init__(self, unet: UNet, diffusion: GaussianDiffusion, device="cuda"):
        self.unet = unet
        self.diffusion = diffusion
        self.device = device

    def __call__(self, num_images: int = 1, image_size=32) -> list[Image.Image]:
        """
        Generate samples from DDPM
        """
        noise = torch.randn(
            (num_images, self.unet.in_channels, image_size, image_size)
        ).to(self.device)
        # samples = self.diffusion.sample(self.unet, noise, clip_denoised=True)
        samples = self.diffusion.sample(self.unet, noise, clip_denoised=False)
        images = batch_to_images(samples)

        return images
