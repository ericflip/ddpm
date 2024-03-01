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


def make_video(arrays):
    import imageio
    import numpy as np
    from PIL import Image

    arrays = [array.permute((1, 2, 0)).squeeze(-1) * 255 for array in arrays]
    arrays = [array.cpu().numpy().astype(np.uint8) for array in arrays]

    arrays = [Image.fromarray(array).resize((256, 256), 0) for array in arrays]

    # Convert tensors to numpy arrays and ensure they are in uint8 format
    # images = [array.numpy().astype(np.uint8) for array in arrays]
    images = [np.array(array) for array in arrays]

    # Create a video writer object using imageio
    writer = imageio.get_writer(
        "output_video.mp4", fps=120
    )  # You can change fps to your desired frame rate

    # Write each frame to the video
    for img in images:
        writer.append_data(img)

    # Close the writer to finalize the video
    writer.close()
