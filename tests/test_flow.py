from types import SimpleNamespace
import torch
from blur_gs.loss import global_direction_loss, blur_flow_loss


def test_global_direction_not_per_pixel_sign():
    observed = torch.zeros(2, 2, 4)
    observed[0] = 1
    forward = observed.clone()
    forward[0, :, 2:] = -1
    backward = -forward
    valid = torch.ones(2, 4, dtype=torch.bool)
    loss, _ = global_direction_loss(forward, backward, observed, valid.float(), valid, valid)
    assert loss.item() > .9  # Per-pixel sign selection would incorrectly produce zero.


def test_backward_candidate_selected():
    observed = torch.ones(2, 4, 4)
    valid = torch.ones(4, 4, dtype=torch.bool)
    loss, stats = global_direction_loss(-observed, observed, observed, valid.float(), valid, valid)
    assert stats["direction"] == 1 and loss == 0


def test_empty_mask_is_finite_and_has_zero_gradient():
    forward = torch.randn(2, 3, 3, requires_grad=True)
    valid = torch.zeros(3, 3, dtype=torch.bool)
    loss, stats = global_direction_loss(forward, -forward, torch.zeros_like(forward),
                                        valid.float(), valid, valid)
    loss.backward()
    assert stats["skipped"] == 1 and torch.isfinite(loss)
    assert torch.count_nonzero(forward.grad) == 0


def camera(w2c, h=16, w=24):
    return SimpleNamespace(world_view_transform=w2c.T, image_width=w, image_height=h,
                           focal_x=20., focal_y=20.)


def test_endpoint_flow_full_pipeline_and_gradients():
    w0 = torch.eye(4)
    t = torch.tensor(-.1, requires_grad=True)
    w1 = torch.eye(4) + torch.nn.functional.pad(t.reshape(1, 1), (3, 0, 0, 3))
    z = torch.tensor(2., requires_grad=True)
    depth = z.expand(1, 16, 24)
    alpha = torch.ones_like(depth)
    observed = torch.zeros(2, 16, 24)
    observed[0] = -.8  # True induced flow is -1 pixel: nonzero residual.
    crop = dict(original_width=24, original_height=16, left=0, top=0, width=24, height=16)
    loss, stats = blur_flow_loss([camera(w0), camera(w1)], [(depth, alpha), (depth, alpha)],
                                observed, torch.ones(16, 24), crop)
    loss.backward()
    assert stats["direction"] == 0 and stats["skipped"] == 0
    assert t.grad.abs() > 0 and z.grad.abs() > 0


def test_occlusion_mask_rejects_inconsistent_target_depth():
    from blur_gs.geometry import pixel_grid
    from blur_gs.loss import directional_flow
    depth = torch.ones(1, 4, 4)
    _, valid = directional_flow(depth, depth, depth * 10, depth, torch.eye(4), torch.eye(4),
                                torch.eye(3), pixel_grid(4, 4), torch.ones(2), .5, .1)
    assert not valid.any()
