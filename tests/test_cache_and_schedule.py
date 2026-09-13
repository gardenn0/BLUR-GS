import json
from types import SimpleNamespace
import numpy as np
import pytest
import torch
from blur_gs.cache import FlowCache, iaai_crop
from blur_gs.training import BlurConfig, BlurSupervisor, phase_at


def make_cache(tmp_path):
    np.savez(tmp_path / "a.npz", flow=np.ones((2, 4, 6)), mask=np.ones((4, 6)))
    manifest = dict(version=1, units="network_pixels", images={"a": dict(file="a.npz", crop=iaai_crop(640, 480))})
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    return manifest


def test_cache_roundtrip_and_missing_camera(tmp_path):
    make_cache(tmp_path)
    cache = FlowCache(tmp_path, ["a"])
    flow, mask, crop = cache.get("a", "cpu")
    assert flow.shape == (2, 4, 6) and mask.shape == (4, 6)
    assert not flow.requires_grad
    assert crop["height"] == 448 and crop["top"] == 16
    with pytest.raises(ValueError, match="Missing"):
        FlowCache(tmp_path, ["absent"])


def test_cache_rejects_nonfinite_values(tmp_path):
    make_cache(tmp_path)
    np.savez(tmp_path / "a.npz", flow=np.full((2, 4, 6), np.nan), mask=np.ones((4, 6)))
    with pytest.raises(ValueError, match="Nonfinite"):
        FlowCache(tmp_path).get("a", "cpu")


def test_cache_rejects_path_escape(tmp_path):
    manifest = make_cache(tmp_path)
    manifest["images"]["a"]["file"] = "../outside.npz"
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(ValueError, match="escapes"):
        FlowCache(tmp_path).get("a", "cpu")


def test_schedule_boundaries():
    cfg = BlurConfig(flow_cache="cache", flow_start=10, geometry_start=20, alternate_every=3, flow_mode="alternating")
    assert [phase_at(i, cfg) for i in [10, 11, 20, 21, 23, 24, 26, 27]] == [
        "baseline", "trajectory_prior", "trajectory_prior", "trajectory", "trajectory", "geometry", "geometry", "trajectory"]
    assert phase_at(100, BlurConfig()) == "baseline"
    assert phase_at(100, BlurConfig(flow_cache="c", flow_mode="trajectory", flow_start=0)) == "trajectory_prior"


def test_phase_switch_clears_grads_and_freezes_correct_group(tmp_path):
    make_cache(tmp_path)
    cfg = BlurConfig(flow_cache=str(tmp_path), flow_start=0, geometry_start=1, alternate_every=1, flow_mode="alternating")
    supervisor = BlurSupervisor(cfg, [SimpleNamespace(image_name="a")], start_warp=0)
    gp, kp = torch.nn.Parameter(torch.tensor(1.)), torch.nn.Parameter(torch.tensor(1.))
    g = SimpleNamespace(optimizer=torch.optim.Adam([gp]))
    k = SimpleNamespace(optimizer=torch.optim.Adam([kp]))
    gp.grad, kp.grad = torch.ones_like(gp), torch.ones_like(kp)
    supervisor.begin(2, g, k)
    assert not gp.requires_grad and kp.requires_grad and gp.grad is None
    supervisor.begin(3, g, k)
    assert gp.requires_grad and not kp.requires_grad and kp.grad is None


def test_como_default_preserves_update_and_gradient_lifecycle(tmp_path):
    make_cache(tmp_path)
    cfg = BlurConfig(flow_cache=str(tmp_path), flow_start=10, geometry_start=2)
    assert cfg.flow_mode == "como"
    supervisor = BlurSupervisor(cfg, [SimpleNamespace(image_name="a")], start_warp=1)
    gp = torch.nn.Parameter(torch.tensor(1.))
    kp = torch.nn.Parameter(torch.tensor(1.))
    # A deliberately frozen parameter must not be enabled by the controller.
    frozen = torch.nn.Parameter(torch.tensor(2.), requires_grad=False)
    g = SimpleNamespace(optimizer=torch.optim.Adam([gp, frozen]))
    k = SimpleNamespace(optimizer=torch.optim.Adam([kp]))
    gp.grad, kp.grad = torch.ones_like(gp), torch.ones_like(kp)
    for iteration in (1, 10, 11, 20000, 20001, 20051, 40000):
        supervisor.begin(iteration, g, k)
        assert supervisor.update_geometry and supervisor.update_kernel
        assert gp.requires_grad and kp.requires_grad and not frozen.requires_grad
        assert gp.grad.item() == 1 and kp.grad.item() == 1
        assert supervisor.phase == ("baseline" if iteration <= 10 else "joint")


def test_como_zero_weight_matches_baseline_optimizer_updates(tmp_path):
    make_cache(tmp_path)
    cfg = BlurConfig(flow_cache=str(tmp_path), flow_start=0, flow_weight=0)
    controlled = BlurSupervisor(cfg, [SimpleNamespace(image_name="a")], start_warp=0)
    baseline = BlurSupervisor(BlurConfig(), [], start_warp=0)
    def run(supervisor):
        gp, kp = torch.nn.Parameter(torch.tensor(1.)), torch.nn.Parameter(torch.tensor(2.))
        g = SimpleNamespace(get_xyz=gp, optimizer=torch.optim.Adam([gp], lr=.01))
        k = SimpleNamespace(optimizer=torch.optim.Adam([kp], lr=.02))
        for iteration in range(1, 6):
            supervisor.begin(iteration, g, k)
            # Missing view/cameras ensure the zero-weight path performs no flow work.
            flow, stats = supervisor.loss(iteration, None, [], g, None, .1)
            assert not stats
            ((gp + 2 * kp - .5).square() + flow).backward()
            if supervisor.update_geometry:
                g.optimizer.step()
            g.optimizer.zero_grad(set_to_none=True)
            if supervisor.update_kernel:
                k.optimizer.step()
                k.optimizer.zero_grad()
        return torch.stack([gp.detach(), kp.detach()])
    assert torch.equal(run(baseline), run(controlled))
