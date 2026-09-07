import torch

from utils.pose_utils import pixel_grid
from utils.motion_loss_utils import motion_loss, path_consistency
from scene.motion_prior import restore_prediction


def test_start_grid_alignment_for_spatially_varying_flow():
    grid = pixel_grid(12, 16, torch.ones(1))
    observed = torch.stack((0.05 * grid[..., 0] + 0.1, torch.zeros_like(grid[..., 0])), -1)
    start = grid + grid.new_tensor([0.5, 0])
    target_at_start = torch.stack((0.05 * start[..., 0] + 0.1, torch.zeros_like(start[..., 0])), -1)
    path = torch.stack((start, start + target_at_start))
    mask = torch.ones(12, 16, dtype=torch.bool)
    result = motion_loss(
        path,
        mask[None].expand(2, -1, -1),
        observed,
        mask.float(),
        mask,
        reference="start",
        ambiguous=False,
        direction_weight=0,
    )
    assert result["motion"].item() < 1e-6


def test_global_direction_ambiguity_not_per_pixel():
    grid = pixel_grid(12, 16, torch.ones(1))
    flow = torch.zeros_like(grid)
    flow[..., 0] = 1
    path = torch.stack((grid, grid + flow))
    mask = torch.ones(12, 16, dtype=torch.bool)
    kwargs = dict(
        path_valid=mask[None].expand(2, -1, -1),
        confidence=mask.float(),
        depth_valid=mask,
        reference="midpoint",
        direction_weight=0,
    )
    result = motion_loss(path, observed=-flow, **kwargs)
    assert result["motion"] < 1e-6 and result["reversed"] == 1
    mixed = flow.clone()
    mixed[:, :8] *= -1
    assert motion_loss(path, observed=mixed, **kwargs)["motion"] > 0.1


def test_nonlinear_path_length_not_endpoint_length():
    grid = pixel_grid(12, 16, torch.ones(1))
    path = torch.stack((grid, grid + 2, grid))
    mask = torch.ones(12, 16, dtype=torch.bool)
    kwargs = dict(
        path_valid=mask[None].expand(3, -1, -1),
        observed=torch.zeros_like(grid),
        confidence=mask.float(),
        depth_valid=mask,
        reference="midpoint",
        direction_weight=0,
        flow_weight=0,
        magnitude_weight=1,
    )
    assert motion_loss(path, magnitude="endpoint", **kwargs)["motion"] == 0
    assert motion_loss(path, magnitude="path", **kwargs)["motion"] > 5
    assert path_consistency(path, path, mask.float()) == 0


def test_zero_confidence_returns_finite_zero_and_gradients():
    path = torch.zeros(3, 12, 16, 2, requires_grad=True)
    result = motion_loss(
        path,
        torch.ones(3, 12, 16, dtype=torch.bool),
        torch.zeros(12, 16, 2),
        torch.zeros(12, 16),
        torch.ones(12, 16, dtype=torch.bool),
    )
    result["motion"].backward()
    assert result["motion"] == 0
    assert torch.isfinite(path.grad).all()


def test_official_crop_restoration_scaling_and_invalid_borders():
    flow = torch.ones(1, 2, 224, 320)
    depth = torch.full((1, 1, 224, 320), 7.0)
    restored, d, confidence = restore_prediction(flow, depth, 448, 800)
    assert confidence[:, :80].sum() == 0
    assert confidence[:, 720:].sum() == 0
    torch.testing.assert_close(restored[:, 80:720], torch.full((448, 640, 2), 2.0))
    torch.testing.assert_close(d[:, 80:720], torch.full((448, 640), 7.0))
    assert confidence.sum() == 448 * 640


def test_unsupported_reverse_direction_does_not_win_with_zero_loss():
    grid = pixel_grid(12, 16, torch.ones(1))
    path = torch.stack((grid, grid + 100))
    mask = torch.ones(12, 16, dtype=torch.bool)
    result = motion_loss(
        path,
        mask[None].expand(2, -1, -1),
        torch.ones_like(grid),
        mask.float(),
        mask,
        reference="start",
    )
    assert result["reversed"] == 0
    assert result["motion"] > 1
