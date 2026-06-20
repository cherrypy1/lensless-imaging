from pathlib import Path

import numpy as np
from PIL import Image


def load_rgb(path):
    return Image.open(path).convert("RGB")


def load_mask(path):
    return np.load(path)


def find_image(path):
    path = Path(path)
    for suffix in [".png", ".jpg", ".jpeg"]:
        image_path = path.with_suffix(suffix)
        if image_path.exists():
            return image_path
    return None


def chw(image):
    return image.permute(2, 0, 1).float()


def psf_chw(psf):
    if psf.ndim == 4:
        psf = psf[0]
    return psf.permute(2, 0, 1).float()


def image_to_numpy(tensor):
    tensor = tensor.detach().cpu().clamp(0, 1)
    if tensor.ndim == 4:
        tensor = tensor[0]
    tensor = tensor.permute(1, 2, 0)
    return (tensor.numpy() * 255).round().astype(np.uint8)


def save_tensor_image(tensor, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(image_to_numpy(tensor)).save(path)
