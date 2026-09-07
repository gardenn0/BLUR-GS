"""Linear interpolation of two endpoint twists, mapped to valid SE(3) poses."""

import torch
from torch import Tensor, nn

from utils.pose_utils import adjoint, inverse_pose, se3_exp

TRAJECTORY_MODEL = "linear_se3"
CHECKPOINT_VERSION = 2


class ExposureTrajectory(nn.Module):
    def __init__(self, initial_w2c: Tensor):
        super().__init__()
        self.register_buffer("initial_w2c", initial_w2c.clone())
        # Two endpoints parameterize the curve; exposure_samples controls render count.
        self.controls = nn.Parameter(initial_w2c.new_zeros(2, 6))

    def twists(self, times: Tensor) -> Tensor:
        t = times.reshape(-1, 1)
        return (1 - t) * self.controls[0] + t * self.controls[1]

    def forward(self, times: Tensor) -> Tensor:
        return self.initial_w2c[None] @ se3_exp(self.twists(times))

    def at(self, time: float) -> Tensor:
        return self(self.controls.new_tensor([time]))[0]

    @torch.no_grad()
    def initialize_from_camera_motion(self, motion: Tensor) -> None:
        # The solver returns camera motion, whose world-to-camera increment has opposite sign.
        local = adjoint(inverse_pose(self.initial_w2c), -motion)
        factors = local.new_tensor([-0.5, 0.5])
        self.controls.copy_(factors[:, None] * local[None])

    def regularizers(self) -> tuple[Tensor, Tensor]:
        # d^2 xi / dt^2 is analytically zero. Avoid penalizing floating-point differences.
        acceleration = self.controls.sum() * 0
        # Twist norm is a local SE(3) anchor, not a globally bi-invariant metric.
        anchor = self.twists(self.controls.new_tensor([0.5])).square().mean()
        return acceleration, anchor


def require_linear_checkpoint(checkpoint: dict) -> None:
    """Do not silently reinterpret four-control optimizer state as a linear trajectory."""
    if (
        checkpoint.get("format_version") != CHECKPOINT_VERSION
        or checkpoint.get("trajectory_model") != TRAJECTORY_MODEL
    ):
        raise ValueError(
            "Resume requires a linear_se3 checkpoint (format_version=2). "
            "Legacy Bezier checkpoints can still be rendered, but start a new run for linear training."
        )


def checkpoint_midpoint(checkpoint: dict, index: int) -> Tensor:
    """Recover the exact saved midpoint, including legacy Bezier render-only support."""
    state = checkpoint["trajectories"]
    initial = state[f"{index}.initial_w2c"]
    controls = state[f"{index}.controls"]
    if checkpoint.get("format_version") == 1:
        if controls.shape != (4, 6):
            raise ValueError("Legacy Bezier checkpoint requires four 6D controls")
        midpoint = controls.new_tensor([0.125, 0.375, 0.375, 0.125]) @ controls
    else:
        require_linear_checkpoint(checkpoint)
        if controls.shape != (2, 6):
            raise ValueError("Linear checkpoint requires two 6D endpoint controls")
        midpoint = controls.mean(0)
    return initial @ se3_exp(midpoint)
