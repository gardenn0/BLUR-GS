"""Integrate virtual-pose renders over an exposure without changing trajectory sampling."""

import torch
from torch import Tensor

from scene.gaussian_model import GaussianScene
from utils.image_utils import linear_to_srgb, srgb_to_linear


def render_blur(
    renderer,
    scene: GaussianScene,
    poses: Tensor,
    K: Tensor,
    height: int,
    width: int,
    weights: Tensor | None = None,
    linear_exposure: bool = True,
) -> Tensor:
    n = len(poses)
    if weights is None:
        weights = poses.new_full((n,), 1 / n)
    if weights.shape != (n,) or not torch.isfinite(weights).all() or (weights < 0).any():
        raise ValueError("Shutter weights must be finite nonnegative temporal weights")
    if weights.sum() <= 0:
        raise ValueError("At least one shutter weight must be positive")
    weights = weights / weights.sum()
    accumulated = None
    for pose, weight in zip(poses, weights):
        rgb = renderer(scene, pose, K, height, width).rgb
        if linear_exposure:
            rgb = srgb_to_linear(rgb)
        value = weight * rgb
        accumulated = value if accumulated is None else accumulated + value
    return linear_to_srgb(accumulated) if linear_exposure else accumulated
