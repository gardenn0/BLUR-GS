"""Differentiable cubic Bezier trajectory in se(3), allowed by BLUR-GS section 1.3."""

import torch
from torch import Tensor, nn

from .geometry import adjoint, inverse_pose, se3_exp


class ExposureTrajectory(nn.Module):
    def __init__(self, initial_w2c: Tensor):
        super().__init__()
        self.register_buffer("initial_w2c", initial_w2c.clone())
        self.controls = nn.Parameter(initial_w2c.new_zeros(4, 6))

    def twists(self, times: Tensor) -> Tensor:
        t = times.reshape(-1, 1)
        basis = torch.cat(((1 - t) ** 3, 3 * (1 - t) ** 2 * t, 3 * (1 - t) * t * t, t**3), -1)
        return basis @ self.controls

    def forward(self, times: Tensor) -> Tensor:
        return self.initial_w2c[None] @ se3_exp(self.twists(times))

    def at(self, time: float) -> Tensor:
        return self(self.controls.new_tensor([time]))[0]

    @torch.no_grad()
    def initialize_from_camera_motion(self, motion: Tensor) -> None:
        # The solver returns camera motion, whose world-to-camera increment has opposite sign.
        local = adjoint(inverse_pose(self.initial_w2c), -motion)
        factors = torch.linspace(-0.5, 0.5, 4, device=local.device, dtype=local.dtype)
        self.controls.copy_(factors[:, None] * local[None])

    def regularizers(self, samples: int = 9) -> tuple[Tensor, Tensor]:
        xi = self.twists(
            torch.linspace(0, 1, samples, device=self.controls.device, dtype=self.controls.dtype)
        )
        acceleration = (xi[2:] - 2 * xi[1:-1] + xi[:-2]).square().mean()
        # Twist norm is a local SE(3) anchor, not a globally bi-invariant metric.
        anchor = self.twists(self.controls.new_tensor([0.5])).square().mean()
        return acceleration, anchor
