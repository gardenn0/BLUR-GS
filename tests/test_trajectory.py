import pytest
import torch

from utils.pose_utils import se3_exp
from scene.trajectory import (
    CHECKPOINT_VERSION,
    TRAJECTORY_MODEL,
    ExposureTrajectory,
    checkpoint_midpoint,
    require_linear_checkpoint,
)


def test_linear_endpoints_interior_and_endpoint_gradients():
    base = se3_exp(torch.tensor([0.3, -0.2, 0.1, 0.05, -0.1, 0.15], dtype=torch.float64))
    trajectory = ExposureTrajectory(base)
    start = base.new_tensor([0.01, 0.02, -0.03, 0.04, -0.05, 0.06])
    end = base.new_tensor([-0.02, 0.04, 0.01, -0.03, 0.02, -0.01])
    with torch.no_grad():
        trajectory.controls.copy_(torch.stack((start, end)))
    times = base.new_tensor([0.0, 0.25, 0.5, 0.75, 1.0])
    expected = start[None] + times[:, None] * (end - start)[None]
    assert trajectory.controls.shape == (2, 6)
    torch.testing.assert_close(trajectory.twists(times), expected)
    torch.testing.assert_close(trajectory(times), base[None] @ se3_exp(expected))
    trajectory.twists(base.new_tensor([0.25])).sum().backward()
    torch.testing.assert_close(trajectory.controls.grad[0], torch.full_like(start, 0.75))
    torch.testing.assert_close(trajectory.controls.grad[1], torch.full_like(end, 0.25))


def test_linear_acceleration_is_exactly_zero_with_finite_gradient():
    trajectory = ExposureTrajectory(torch.eye(4))
    with torch.no_grad():
        trajectory.controls.copy_(torch.randn_like(trajectory.controls) * 0.1)
    acceleration, anchor = trajectory.regularizers()
    assert acceleration.item() == 0
    torch.testing.assert_close(anchor, trajectory.controls.mean(0).square().mean())
    acceleration.backward()
    torch.testing.assert_close(trajectory.controls.grad, torch.zeros_like(trajectory.controls))


@pytest.mark.parametrize(
    "version,control_count,weights",
    [
        (1, 4, [0.125, 0.375, 0.375, 0.125]),
        (CHECKPOINT_VERSION, 2, [0.5, 0.5]),
    ],
)
def test_checkpoint_midpoint_preserves_saved_geometry(version, control_count, weights):
    base = se3_exp(torch.tensor([0.3, -0.2, 0.1, 0.05, -0.1, 0.15], dtype=torch.float64))
    controls = torch.arange(control_count * 6, dtype=base.dtype).reshape(control_count, 6) * 0.002
    checkpoint = {
        "format_version": version,
        "trajectories": {"0.initial_w2c": base, "0.controls": controls},
    }
    if version == CHECKPOINT_VERSION:
        checkpoint["trajectory_model"] = TRAJECTORY_MODEL
    expected = base @ se3_exp(base.new_tensor(weights) @ controls)
    torch.testing.assert_close(checkpoint_midpoint(checkpoint, 0), expected)


@pytest.mark.parametrize(
    "checkpoint",
    [
        {"format_version": 1},
        {"format_version": CHECKPOINT_VERSION, "trajectory_model": "cubic_bezier"},
    ],
)
def test_resume_rejects_incompatible_trajectory(checkpoint):
    with pytest.raises(ValueError, match="start a new run for linear training"):
        require_linear_checkpoint(checkpoint)
