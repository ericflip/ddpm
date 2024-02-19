import torch
import torchvision.transforms.functional as F
from PIL import Image


def batch_to_images(batch: torch.Tensor) -> list[Image.Image]:
    images = []

    for tensor in batch:
        mode = "RGB" if tensor.shape[0] == 3 else "L"
        image = F.to_pil_image(tensor, mode=mode)
        images.append(image)

    return images


def make_image_grid(
    images: list[Image.Image], rows: int, cols: int, resize: int = None
) -> Image.Image:
    """
    Prepares a single grid of images. Useful for visualization purposes.
    """
    assert len(images) == rows * cols

    if resize is not None:
        images = [img.resize((resize, resize)) for img in images]

    w, h = images[0].size
    grid = Image.new("RGB", size=(cols * w, rows * h))

    for i, img in enumerate(images):
        grid.paste(img, box=(i % cols * w, i // cols * h))
    return grid
