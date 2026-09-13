"""Exercise the supervisor and preprocessing boundary without CUDA/weights."""
import json
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
import torch
from PIL import Image
from blur_gs.cache import FlowCache
from blur_gs.training import BlurConfig, BlurSupervisor
from blur_gs.depth import render_z_depth


def test_auxiliary_pass_blocks_camera_but_keeps_geometry_gradient():
    xyz = torch.tensor([[0., 0., 2.], [0., 0., 3.]], requires_grad=True)
    matrix = torch.eye(4, requires_grad=True)
    camera = SimpleNamespace(world_view_transform=matrix, full_proj_transform=matrix,
                             projection_matrix=matrix, camera_center=matrix[3, :3])
    pc = SimpleNamespace(get_xyz=xyz)
    def fake_render(camera, pc, pipe, bg, **kwargs):
        assert not camera.world_view_transform.requires_grad
        assert not torch.count_nonzero(bg)
        return {"render": kwargs["override_color"].mean(0).reshape(3, 1, 1)}
    z, _, _ = render_z_depth(camera, pc, None, .1, geometry_grad=True, render_fn=fake_render)
    z.sum().backward()
    assert xyz.grad[:, 2].abs().sum() > 0 and matrix.grad is None
    z_fixed, _, _ = render_z_depth(camera, pc, None, .1, geometry_grad=False, render_fn=fake_render)
    assert not z_fixed.requires_grad


def test_supervisor_routes_loss_in_all_phases(tmp_path, monkeypatch):
    import blur_gs.training as training
    np.savez(tmp_path / "a.npz", flow=np.stack([np.full((16, 24), -.8), np.zeros((16, 24))]),
             mask=np.ones((16, 24)))
    crop = dict(original_width=24, original_height=16, left=0, top=0, width=24, height=16)
    (tmp_path / "manifest.json").write_text(json.dumps(dict(version=1, units="network_pixels",
        images={"a": dict(file="a.npz", crop=crop)})))
    z = torch.nn.Parameter(torch.tensor(2.))
    t = torch.nn.Parameter(torch.tensor(-.1))
    pc = SimpleNamespace(get_xyz=z, optimizer=torch.optim.Adam([z]))
    kernel = SimpleNamespace(optimizer=torch.optim.Adam([t]))
    view = SimpleNamespace(image_name="a", image_width=24, image_height=16)
    cfg = BlurConfig(flow_cache=str(tmp_path), flow_start=0, geometry_start=2,
                     flow_ramp=1, alternate_every=1)
    supervisor = BlurSupervisor(cfg, [view], 0)
    def depth_stub(camera, pc, pipe, kernel_size, *, geometry_grad, min_alpha):
        d = (z if geometry_grad else z.detach()).expand(1, 16, 24)
        return d, torch.ones_like(d), torch.ones_like(d, dtype=torch.bool)
    monkeypatch.setattr(training, "render_z_depth", depth_stub)
    for step, expected_geometry, expected_kernel in [(1, False, True), (3, False, True), (4, True, False)]:
        supervisor.begin(step, pc, kernel)
        w0 = torch.eye(4)
        w1 = w0 + torch.nn.functional.pad(t.reshape(1, 1), (3, 0, 0, 3))
        cameras = [SimpleNamespace(world_view_transform=w.T, focal_x=20., focal_y=20.,
                                   image_width=24, image_height=16) for w in (w0, w1)]
        loss, stats = supervisor.loss(step, view, cameras, pc, None, .1)
        loss.backward()
        assert stats["skipped"] == 0
        assert (z.grad is not None and z.grad.abs() > 0) == expected_geometry
        assert (t.grad is not None and t.grad.abs() > 0) == expected_kernel


def test_precompute_cli_cache_contract(tmp_path):
    root = Path(__file__).resolve().parents[1]
    package = tmp_path / "stub" / "iaai"
    package.mkdir(parents=True)
    (package / "__init__.py").write_text("")
    (package / "model.py").write_text('''import torch
class Blur2PoseSegNeXtBackbone(torch.nn.Module):
    def __init__(self, supervise_pose, device):
        super().__init__()
        assert not supervise_pose
    def infer(self, data):
        return {"flow_field": torch.ones(1, 2, 224, 320)}
''')
    images = tmp_path / "images"
    images.mkdir()
    Image.new("RGB", (640, 480)).save(images / "frame.png")
    checkpoint = tmp_path / "test.pth"
    torch.save({}, checkpoint)
    output = tmp_path / "cache"
    result = subprocess.run([sys.executable, str(root / "precompute_blur_flow.py"),
        "--images", str(images), "--output", str(output), "--checkpoint", str(checkpoint),
        "--iaai-root", str(package.parent), "--device", "cpu"], capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    flow, mask, crop = FlowCache(output, ["frame"]).get("frame", "cpu")
    assert flow.shape == (2, 224, 320) and flow.mean() == 1
    assert crop["top"] == 16 and mask[0].sum() == 0
