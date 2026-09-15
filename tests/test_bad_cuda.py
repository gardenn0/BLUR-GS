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
    # Translation and rotation finite differences; avoid visibility boundaries.
    for index in (0, 4):
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
