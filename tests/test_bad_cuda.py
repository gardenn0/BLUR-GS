"""Run explicitly on the training server: pytest -q tests/test_bad_cuda.py.

These tests exercise the real gsplat backend, not the CPU reference renderer.
"""
import pytest
import torch
from dataclasses import replace
from bad_blur_gs.config import BadConfig
from bad_blur_gs.renderer import render, render_depth
from bad_blur_gs.training import training_step
from bad_blur_gs.flow import FlowSupervisor
from blur_gs.training import BlurConfig
from test_bad_core import tiny_scene, write_cache
from reference.bad_render import render_single

pytestmark = [pytest.mark.cuda, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def test_real_rasterizer_pose_derivative_and_depth():
    cfg = BadConfig(sh_degree=0)
    model, camera, trajectory, _ = tiny_scene(cfg, "cuda")
    with torch.no_grad():
        model.params["scales"][:, 1] -= .4
        trajectory.controls[0, 1, 0] = .02
    image = render(trajectory.cameras(camera)[-1], model, cfg, torch.zeros(3, device="cuda"))["render"]
    weights = torch.linspace(0, 1, 48, device="cuda")[None, None, :]
    objective = (image*weights).sum()
    objective.backward()
    assert trajectory.controls.grad is not None and trajectory.controls.grad.norm() > 0
    assert torch.isfinite(trajectory.controls.grad).all()
    # Legacy gsplat deliberately approximates the rotation backward.
    # Check translation finite differences; rotation is compared to upstream below.
    for index in (0,):
        analytic = trajectory.controls.grad[0, 1, index].item()
        epsilon = 2e-4
        values = []
        original = trajectory.controls.detach().clone()
        for sign in (1, -1):
            with torch.no_grad():
                trajectory.controls.copy_(original)
                trajectory.controls[0, 1, index] += sign*epsilon
                out = render(trajectory.cameras(camera)[-1], model, cfg, torch.zeros(3, device="cuda"))["render"]
                values.append(float((out*weights).sum()))
        with torch.no_grad():
            trajectory.controls.copy_(original)
        finite = (values[0]-values[1])/(2*epsilon)
        assert analytic == pytest.approx(finite, rel=.08, abs=.15)
    model.optimizer.zero_grad(set_to_none=True)
    depth, alpha, valid = render_depth(camera, model, cfg, geometry_grad=True, min_alpha=.01)
    assert valid.any() and depth[valid].min() > 1.5 and depth[valid].max() < 2.6
    depth[valid].mean().backward()
    assert model.get_xyz.grad is not None and torch.isfinite(model.get_xyz.grad).all()


@pytest.mark.parametrize("mode", ["como", "trajectory"])
def test_real_training_flow_and_refinement(tmp_path, mode):
    write_cache(tmp_path)
    cfg = BadConfig(sh_degree=0, num_virtual_views=3, camera_accumulation=2,
                    warmup=0, refine_every=2, opacity_reset_every=20, stop_split=10)
    model, camera, trajectory, optimizer = tiny_scene(cfg, "cuda")
    with torch.no_grad():
        model.params["opacities"].fill_(2.)
        trajectory.controls[0, 1, 0] = .01
    fc = BlurConfig(flow_cache=str(tmp_path), flow_start=0, flow_ramp=1,
                    flow_mode=mode, flow_min_alpha=.01, flow_min_valid_fraction=.01)
    flow = FlowSupervisor(fc, [camera])
    for step in range(6):
        stats = training_step(step, camera, torch.full((3, 32, 48), .1, device="cuda"),
                              model, trajectory, optimizer, cfg, flow, torch.zeros(3, device="cuda"))
    assert stats["skipped"] == 0 and stats["weight"] == pytest.approx(.01)
    assert torch.isfinite(model.get_xyz).all()


@pytest.mark.parametrize("degree", [0, 3])
def test_cuda_forward_backward_and_adam_step_match_official_render(degree):
    from bad_blur_gs.renderer import require_backend
    cfg = BadConfig(sh_degree=degree, num_virtual_views=3)
    model, camera, trajectory, camera_optimizer = tiny_scene(cfg, "cuda")
    model.active_sh_degree = degree
    with torch.no_grad():
        model.params["scales"][:, 1] -= .4
        trajectory.controls[0, 1, :].add_(.015)
    bg = torch.tensor([.1490, .1647, .2157], device="cuda")
    k = camera.K.clone()
    k[:2, 2] += .5
    flip = torch.diag(torch.tensor([1., -1, -1, 1], device="cuda"))
    views = trajectory.cameras(camera)
    actual = torch.stack([render(c, model, cfg, bg)["render"] for c in views]).mean(0)
    reference = torch.stack([render_single(c.c2w @ flip, k, model, cfg, 48, 32, bg,
                                           require_backend())[0] for c in views]).mean(0)
    torch.testing.assert_close(actual, reference, rtol=1e-6, atol=1e-6)
    from pytorch_msssim import ssim
    gt = torch.full_like(actual, .1)
    def loss(im):
        return .8*(im-gt).abs().mean()+.2*(1-ssim(im[None], gt[None], data_range=1))
    torch.testing.assert_close(loss(actual), loss(reference))
    params = tuple(model.parameters())+(trajectory.controls,)
    ga = torch.autograd.grad(loss(actual), params, retain_graph=True, allow_unused=True)
    gb = torch.autograd.grad(loss(reference), params, allow_unused=True)
    for a, b in zip(ga, gb):
        if a is None:
            assert b is None
        else:
            torch.testing.assert_close(a, b, rtol=2e-4, atol=2e-6)
    # Compare optimizer updates with the same parameters and gradient signals.
    aa = [torch.nn.Parameter(p.detach().clone()) for p in params]
    bb = [torch.nn.Parameter(p.detach().clone()) for p in params]
    oa = torch.optim.Adam(aa, lr=.001, eps=1e-15)
    ob = torch.optim.Adam(bb, lr=.001, eps=1e-15)
    for p, q, a, b in zip(aa, bb, ga, gb):
        p.grad, q.grad = a, b
    oa.step()
    ob.step()
    for p, q in zip(aa, bb):
        torch.testing.assert_close(p, q, rtol=2e-4, atol=2e-6)
