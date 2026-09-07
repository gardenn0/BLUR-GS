"""Image I/O and radiometric conversions; pixels are HWC float RGB in [0, 1]."""

from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch import Tensor


def read_image(path: Path) -> Tensor:
    return torch.from_numpy(np.array(Image.open(path).convert("RGB"), dtype=np.float32) / 255)


def save_image(path: Path, image: Tensor) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    array = (image.detach().cpu().clamp(0, 1).numpy() * 255).round().astype(np.uint8)
    Image.fromarray(array).save(path)


def srgb_to_linear(rgb: Tensor) -> Tensor:
    return torch.where(rgb <= 0.04045, rgb / 12.92, ((rgb + 0.055) / 1.055).clamp_min(0).pow(2.4))


def linear_to_srgb(rgb: Tensor) -> Tensor:
    return torch.where(
        rgb <= 0.0031308, 12.92 * rgb, 1.055 * rgb.clamp_min(0.0031308).pow(1 / 2.4) - 0.055
    )
