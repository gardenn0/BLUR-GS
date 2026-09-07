"""Camera observations and the explicit OpenCV world-to-camera contract."""

from dataclasses import dataclass

import torch
from torch import Tensor


@dataclass
class Frame:
    name: str
    image: Tensor
    K: Tensor
    w2c: Tensor
    flow: Tensor | None = None
    confidence: Tensor | None = None
    initial_depth: Tensor | None = None
    observed_depth: Tensor | None = None
    flow_reference: str = "start"
    sign_ambiguous: bool = True
    exposure_seconds: float | None = None

    def to(self, device: str) -> "Frame":
        return Frame(
            **{
                key: value.to(device) if isinstance(value, Tensor) else value
                for key, value in vars(self).items()
            }
        )


def validate_camera(K: Tensor, w2c: Tensor) -> None:
    if K.shape != (3, 3) or w2c.shape != (4, 4):
        raise ValueError("Expected 3x3 K and 4x4 world-to-camera matrix")
    if not torch.isfinite(K).all() or not torch.isfinite(w2c).all():
        raise ValueError("Camera matrices must be finite")
    if K[0, 0] <= 0 or K[1, 1] <= 0 or K[0, 1] != 0 or K[1, 0] != 0:
        raise ValueError("Only positive-focal, zero-skew pinhole cameras are supported")
    if not torch.allclose(K[2], K.new_tensor([0, 0, 1]), atol=1e-5):
        raise ValueError("Invalid pinhole intrinsic matrix")
    if not torch.allclose(w2c[3], w2c.new_tensor([0, 0, 0, 1]), atol=1e-5):
        raise ValueError("Invalid homogeneous pose row")
    R = w2c[:3, :3]
    if not torch.allclose(R.T @ R, torch.eye(3), atol=1e-3) or torch.det(R) < 0.999:
        raise ValueError("Camera rotation must be a proper SO(3) rotation")
