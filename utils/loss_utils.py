"""Photometric and robust elementary loss functions."""

import torch
from torch import Tensor
from torch.nn import functional as F


def charbonnier(residual: Tensor, eps: float = 1e-3) -> Tensor:
    return (residual.square() + eps * eps).sqrt() - eps


def weighted_mean(value: Tensor, weight: Tensor) -> Tensor:
    weight = weight.detach()
    value = torch.where(weight > 0, value, torch.zeros_like(value))
    return (value * weight).sum() / weight.sum().clamp_min(1e-8)


def ssim(a: Tensor, b: Tensor) -> Tensor:
    a, b = a.permute(2, 0, 1)[None], b.permute(2, 0, 1)[None]
    size = min(11, a.shape[-2], a.shape[-1])
    size -= 1 - size % 2
    coords = torch.arange(size, device=a.device, dtype=a.dtype) - (size - 1) / 2
    kernel = torch.exp(-coords.square() / (2 * 1.5**2))
    kernel = kernel / kernel.sum()
    kernel = (kernel[:, None] * kernel[None, :])[None, None].expand(3, 1, -1, -1)

    def pool(x):
        return F.conv2d(x, kernel, groups=3)

    ma, mb = pool(a), pool(b)
    va, vb = pool(a * a) - ma * ma, pool(b * b) - mb * mb
    cov = pool(a * b) - ma * mb
    return (
        ((2 * ma * mb + 0.01**2) * (2 * cov + 0.03**2))
        / ((ma * ma + mb * mb + 0.01**2) * (va + vb + 0.03**2))
    ).mean()


def rgb_loss(predicted: Tensor, observed: Tensor, dssim_weight: float = 0.2) -> Tensor:
    return (1 - dssim_weight) * (predicted - observed).abs().mean() + dssim_weight * (
        1 - ssim(predicted, observed)
    ) / 2
