import pytest
import torch

from blur_gs.geometry import (
    adjoint,
    exposure_path,
    inverse_pose,
    motion_jacobian,
    se3_exp,
    solve_camera_motion,
)
from blur_gs.trajectory import ExposureTrajectory


def camera(dtype=torch.float64):
    return torch.tensor([[27.0, 0, 8.1], [0, 31.0, 7.3], [0, 0, 1.0]], dtype=dtype)


def test_se3_zero_and_nonzero_gradients():
    xi = torch.zeros(6, dtype=torch.float64, requires_grad=True)
    assert torch.equal(se3_exp(xi), torch.eye(4, dtype=xi.dtype))
    assert torch.autograd.gradcheck(se3_exp, (xi,), eps=1e-6)
    assert torch.autograd.gradgradcheck(se3_exp, (xi,), eps=1e-6)
    xi = (torch.randn(6, dtype=torch.float64) * 0.1).requires_grad_()
    assert torch.autograd.gradcheck(se3_exp, (xi,))
    T = se3_exp(xi)
    torch.testing.assert_close(T @ inverse_pose(T), torch.eye(4, dtype=xi.dtype))


def test_imu_jacobian_matches_perspective_finite_difference():
    K = camera()
    depth = torch.linspace(1.0, 4.0, 16 * 18, dtype=K.dtype).reshape(16, 18)
    A = motion_jacobian(depth, K)
    identity = torch.eye(4, dtype=K.dtype)
    eps = 1e-7
    for i in range(6):
        camera_step = torch.zeros(6, dtype=K.dtype)
        camera_step[i] = eps
        poses = torch.stack((identity, se3_exp(-camera_step)))
        path, _ = exposure_path(depth, K, identity, poses)
        torch.testing.assert_close((path[1] - path[0]) / eps, A[..., i], atol=1e-4, rtol=1e-5)


def test_weighted_motion_solver_and_invalid_points():
    K = camera()
    depth = torch.linspace(1.0, 4.0, 16 * 18, dtype=K.dtype).reshape(16, 18)
    truth = torch.tensor([0.012, -0.023, 0.006, 0.008, -0.012, 0.003], dtype=K.dtype)
    flow = motion_jacobian(depth, K) @ truth
    confidence = torch.ones_like(depth)
    flow[0, 0] = float("nan")
    depth[0, 1] = 0
    confidence[1] = 0
    flow[1] = 1000
    estimate = solve_camera_motion(flow, depth, K, confidence, damping=1e-10)
    torch.testing.assert_close(estimate, truth, atol=1e-7, rtol=1e-5)
    with pytest.raises(ValueError, match="six valid"):
        solve_camera_motion(flow, depth, K, torch.zeros_like(confidence))


def test_inverse_depth_translation_and_rotation_independence():
    K = camera()
    depth = torch.ones(16, 18, dtype=K.dtype)
    A1, A2 = motion_jacobian(depth, K), motion_jacobian(2 * depth, K)
    torch.testing.assert_close(A1[..., :3], 2 * A2[..., :3])
    torch.testing.assert_close(A1[..., 3:], A2[..., 3:])


def test_nonidentity_initial_pose_and_midpoint_anchor():
    base = se3_exp(torch.tensor([0.3, -0.4, 0.2, 0.1, -0.2, 0.15], dtype=torch.float64))
    camera_motion = torch.tensor([0.03, 0.01, -0.02, 0.002, -0.004, 0.003], dtype=base.dtype)
    curve = ExposureTrajectory(base)
    curve.initialize_from_camera_motion(camera_motion)
    torch.testing.assert_close(curve.at(0.5), base)
    relative = curve.at(1) @ inverse_pose(curve.at(0))
    torch.testing.assert_close(relative, se3_exp(-camera_motion))
    local = adjoint(inverse_pose(base), -camera_motion)
    torch.testing.assert_close(base @ se3_exp(local) @ inverse_pose(base), se3_exp(-camera_motion))
