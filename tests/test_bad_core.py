import copy
from dataclasses import replace
import json
import random
from pathlib import Path
import numpy as np
import pytest
import torch
import pypose as pp

from bad_blur_gs.config import BadConfig, exponential_lr
from bad_blur_gs.trajectory import Camera, ExposureTrajectory, interpolate
from bad_blur_gs.gaussians import Gaussians
from bad_blur_gs.data import load_scene
from bad_blur_gs.flow import FlowSupervisor
from bad_blur_gs.checkpoint import capture, restore, save_atomic
from bad_blur_gs.training import training_step, seed_all, make_parser, flow_config
from blur_gs.training import BlurConfig
from blur_gs.geometry import camera_intrinsics, pixel_grid, reproject


@pytest.fixture(autouse=True)
def small_cpu_threadpool():
    old = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(old)


def tiny_scene(config, device="cpu"):
    xyz = torch.tensor([[-.12, -.08, 2.], [.1, -.1, 2.1], [-.1, .15, 2.2], [.18, .12, 2.3]], device=device)
    colors = torch.tensor([[.9, .2, .3], [.1, .8, .4], [.2, .3, .9], [.9, .8, .1]], device=device)
    model = Gaussians.from_points(xyz, colors, config)
    camera = Camera("a", "", torch.eye(4, device=device),
                    torch.tensor([[40., 0, 23.5], [0, 40., 15.5], [0, 0, 1]], device=device), 48, 32)
    trajectory = ExposureTrajectory([camera], config, device)
    optimizer = torch.optim.Adam([trajectory.controls], lr=config.camera_lr, eps=1e-15)
    return model, camera, trajectory, optimizer


def reference_render(camera, model, config, background, override_color=None, retain_stats=False):
    """Tiny differentiable test renderer; never used by production training."""
    xyz = model.get_xyz @ camera.world_view_transform.T[:3, :3].T + camera.world_view_transform.T[:3, 3]
    projected = xyz @ camera.K.T
    xy = (projected[:, :2]/projected[:, 2:])[None]
    if retain_stats:
        xy.retain_grad()
    grid = pixel_grid(camera.image_height, camera.image_width, device=xyz.device)
    sigma = model.get_scaling.mean(-1) * camera.focal_x / xyz[:, 2]
    weight = torch.exp(-((grid[None]-xy[0, :, None, None]).square().sum(-1))/(2*sigma[:, None, None].square()))
    weight = weight * model.get_opacity[:, :, None]
    colors = model.params["features_dc"].sigmoid() if override_color is None else override_color
    image = torch.einsum("nhw,nc->chw", weight, colors)
    info = dict(means2d=xy, radii=torch.ones(1, len(xyz), device=xyz.device), width=camera.image_width, height=camera.image_height)
    return dict(render=image, info=info)


@pytest.mark.parametrize("mode", ["linear", "cubic"])
def test_rigid_trajectory_and_gradients(mode):
    config = BadConfig(trajectory=mode)
    _, camera, trajectory, _ = tiny_scene(config)
    poses = trajectory.poses("a")
    r = poses[:, :3, :3]
    torch.testing.assert_close(r.transpose(1, 2) @ r, torch.eye(3).expand(10, 3, 3), atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(torch.linalg.det(r), torch.ones(10))
    (poses[:, :3, 3].sum()+poses[:, 0, 1].sum()).backward()
    assert torch.isfinite(trajectory.controls.grad).all() and trajectory.controls.grad.norm() > 0
    assert len(trajectory.cameras(camera, [0.5])) == 1


def test_interpolation_analytic_endpoints_and_cubic_translation():
    controls = pp.se3(torch.tensor([[0., 0, 0, 0, 0, 0], [1., 0, 0, 0, 0, .4]])).Exp()
    out = interpolate(controls, torch.tensor([0., .5, 1.]), "linear")
    torch.testing.assert_close(out[0].matrix(), controls[0].matrix())
    torch.testing.assert_close(out[-1].matrix(), controls[-1].matrix())
    torch.testing.assert_close(out[1].translation(), controls.translation().mean(0))
    four = pp.se3(torch.tensor([[0., 0, 0, 0, 0, 0], [1., 0, 0, 0, 0, 0],
                               [2., 0, 0, 0, 0, 0], [3., 0, 0, 0, 0, 0]])).Exp()
    out = interpolate(four, torch.tensor([0., .5, 1.]), "cubic")
    torch.testing.assert_close(out.translation()[:, 0], torch.tensor([1., 1.5, 2.]))


def test_intrinsics_and_resize_reprojection():
    _, camera, _, _ = tiny_scene(BadConfig())
    camera.K[0, 2] = 21.25  # principal point must not silently become centered
    torch.testing.assert_close(camera_intrinsics(camera), camera.K)
    half = camera.scaled(24, 16)
    torch.testing.assert_close(half.K[:2, 2], (camera.K[:2, 2]+.5)/2-.5)
    pixels = pixel_grid(16, 24)
    target = torch.eye(4)
    target[0, 3] = .1
    uv, _ = reproject(pixels, torch.full((16, 24), 2.), torch.eye(4), target, half.K)
    torch.testing.assert_close(uv-pixels, torch.tensor([1., 0.]).expand(16, 24, 2), atol=2e-6, rtol=1e-5)


def test_camera_accumulation_and_rgb_pose_learning():
    cfg = BadConfig(sh_degree=0, camera_accumulation=3, stop_split=0, num_virtual_views=3)
    model, camera, trajectory, optimizer = tiny_scene(cfg)
    flow = FlowSupervisor(BlurConfig(), [camera])
    original = trajectory.controls.detach().clone()
    for step in range(3):
        training_step(step, camera, torch.full((3, 32, 48), .3), model, trajectory, optimizer,
                      cfg, flow, torch.zeros(3), reference_render)
        if step < 2:
            torch.testing.assert_close(trajectory.controls, original)
    assert not torch.equal(trajectory.controls, original)
    assert optimizer.state[trajectory.controls]["step"] == 1
    assert exponential_lr(.001, .00001, 30000, 30000) == pytest.approx(.00001)


def test_resume_mid_accumulation_is_exact(tmp_path):
    cfg = BadConfig(sh_degree=0, camera_accumulation=3, stop_split=0, num_virtual_views=2)
    seed_all(7)
    model, camera, trajectory, optimizer = tiny_scene(cfg)
    flow_cfg = BlurConfig()
    flow = FlowSupervisor(flow_cfg, [camera])
    def advance(start, end, m, t, o):
        for step in range(start, end):
            target = torch.rand(3, 32, 48)*.3 + random.random()*.1
            training_step(step, camera, target, m, t, o, cfg, flow, torch.zeros(3), reference_render)
    advance(0, 2, model, trajectory, optimizer)
    path = tmp_path/"checkpoint.pth"
    save_atomic(capture(1, model, trajectory, optimizer, cfg, flow_cfg, {"id": "same"}, [0]), path)
    advance(2, 5, model, trajectory, optimizer)
    state = torch.load(path, weights_only=False)
    restored = Gaussians(state["gaussians"], cfg)
    t2 = ExposureTrajectory([camera], cfg, "cpu")
    o2 = torch.optim.Adam([t2.controls], lr=cfg.camera_lr, eps=1e-15)
    step, stack = restore(state, restored, t2, o2, cfg, flow_cfg, {"id": "same"})
    assert step == 2 and stack == [0]
    advance(step, 5, restored, t2, o2)
    for a, b in zip(model.parameters(), restored.parameters()):
        torch.testing.assert_close(a, b, rtol=0, atol=0)
    torch.testing.assert_close(trajectory.controls, t2.controls, rtol=0, atol=0)
    torch.testing.assert_close(trajectory.controls.grad, t2.controls.grad, rtol=0, atol=0)
    with pytest.raises(ValueError, match="data"):
        restore(state, restored, t2, o2, cfg, flow_cfg, {"id": "changed"})


def write_cache(root):
    root.mkdir(exist_ok=True)
    crop = dict(original_width=48, original_height=32, width=48, height=32, left=0, top=0)
    np.savez(root/"a.npz", flow=np.full((2, 32, 48), .25, dtype=np.float32), mask=np.ones((32, 48), np.float32))
    (root/"manifest.json").write_text(json.dumps(dict(version=1, units="network_pixels", images={"a": {"file": "a.npz", "crop": crop}})))


@pytest.mark.parametrize("mode,geometry", [("como", True), ("trajectory", False)])
def test_flow_routes_gradients_without_touching_camera_accumulation(tmp_path, mode, geometry):
    write_cache(tmp_path)
    cfg = BadConfig(sh_degree=0)
    model, camera, trajectory, optimizer = tiny_scene(cfg)
    with torch.no_grad():
        trajectory.controls[0, 1, 0] = .03
    def depth(cam, m, config, geometry_grad, min_alpha):
        z = m.get_xyz[:, 2].mean()
        if not geometry_grad:
            z = z.detach()
        d = z.expand(1, 32, 48)
        return d, torch.ones_like(d), torch.ones_like(d, dtype=torch.bool)
    fc = BlurConfig(flow_cache=str(tmp_path), flow_mode=mode, flow_start=0, flow_ramp=1)
    flow = FlowSupervisor(fc, [camera], depth_fn=depth)
    value, stats = flow.loss(1, camera, trajectory.cameras(camera), model, cfg)
    assert stats["skipped"] == 0
    value.backward()
    assert trajectory.controls.grad.norm() > 0
    grad = model.get_xyz.grad
    assert (grad is not None and grad.norm() > 0) == geometry


def test_zero_weight_skips_depth_and_cache_equivalent_gradients(tmp_path):
    write_cache(tmp_path)
    def forbidden(*args, **kwargs):
        raise AssertionError("depth must not be rendered at zero flow weight")
    cfg = BadConfig(sh_degree=0, stop_split=0, num_virtual_views=2, camera_accumulation=2)
    outputs = []
    for flow_cfg in (BlurConfig(), BlurConfig(flow_cache=str(tmp_path), flow_weight=0, flow_start=0)):
        seed_all(10)
        model, camera, trajectory, optimizer = tiny_scene(cfg)
        flow = FlowSupervisor(flow_cfg, [camera], depth_fn=forbidden)
        for step in range(3):
            training_step(step, camera, torch.full((3, 32, 48), .2), model, trajectory, optimizer,
                          cfg, flow, torch.zeros(3), reference_render)
        outputs.append([p.detach().clone() for p in model.parameters()] + [trajectory.controls.detach().clone()])
    for a, b in zip(*outputs):
        torch.testing.assert_close(a, b, rtol=0, atol=0)


def test_refinement_migrates_adam_and_resets_opacity():
    cfg = BadConfig(sh_degree=0, warmup=0, refine_every=2, opacity_reset_every=10, stop_split=30,
                    stop_screen_size=0, cull_scale=100)
    model, _, _, _ = tiny_scene(cfg)
    sum(p.square().sum() for p in model.parameters()).backward()
    model.optimizer.step()
    model.grad_sum, model.vis_count, model.max_radii = torch.ones(4), torch.ones(4), torch.zeros(4)
    model.refine(6, cfg, 1, 48, 32)
    assert len(model.get_xyz) == 8
    for group in model.optimizer.param_groups:
        p = group["params"][0]
        assert p is model.params[group["name"]]
        assert model.optimizer.state[p]["exp_avg"].shape == p.shape
        assert model.optimizer.state[p]["exp_avg"].count_nonzero() == 0
    model.refine(12, cfg, 1, 48, 32)
    assert model.get_opacity.max() <= 2*cfg.cull_alpha+1e-7


def test_colmap_split_intrinsics_scale_and_no_source_mutation(tmp_path):
    from PIL import Image
    sparse = tmp_path/"sparse"/"0"
    sparse.mkdir(parents=True)
    images = tmp_path/"images_1"
    images.mkdir()
    (sparse/"cameras.txt").write_text("1 PINHOLE 48 32 40 41 23 15\n")
    entries = []
    for i in range(17):
        Image.new("RGB", (48, 32), (i, i, i)).save(images/f"{i:03}.png")
        entries.append(f"{i+1} 1 0 0 0 {i*.1} 0 0 1 {i:03}.png\n\n")
    (sparse/"images.txt").write_text("".join(entries))
    (sparse/"points3D.txt").write_text("\n".join(f"{i} {i*.01} 0 2 128 64 32 0" for i in range(4)))
    before = set(tmp_path.rglob("*"))
    train, test, xyz, colors, manifest = load_scene(tmp_path, "", 1, True, 8, .25, "cpu")
    assert [c.image_name for c in test] == ["000", "008", "016"]
    assert len(train) == 14 and set(tmp_path.rglob("*")) == before
    assert train[0].K[0, 2] == 22.5
    # BAD scales all poses BEFORE splitting; heldout endpoints set the extent.
    assert max(abs(c.c2w[0, 3]) for c in train) == pytest.approx(.21875)
    assert max(abs(c.c2w[0, 3]) for c in train+test) == pytest.approx(.25)
    assert xyz.shape == colors.shape == (4, 3)
    assert manifest["train"] == [c.image_name for c in train]


def test_cli_rejects_unsupported_schedule_and_invalid_settings():
    parser = make_parser()
    args = parser.parse_args(["-s", "s", "-m", "m", "--baseline"])
    assert flow_config(args).flow_weight == 0
    with pytest.raises(SystemExit):
        parser.parse_args(["-s", "s", "-m", "m", "--flow_mode", "alternating"])
    with pytest.raises(ValueError):
        BadConfig(camera_accumulation=0).validate()
    with pytest.raises(ValueError):
        BadConfig(num_virtual_views=1).validate()
