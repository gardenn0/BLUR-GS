"""Pinned-upstream coordinate/renderer comparison, without CUDA for this file."""
import torch
import pypose as pp
import pytest
from bad_blur_gs.coordinates import normalize_colmap
from bad_blur_gs.config import BadConfig
from bad_blur_gs.trajectory import interpolate
from bad_blur_gs import renderer
from bad_blur_gs.checkpoint import restore
from bad_blur_gs.data import load_scene, load_image
from test_bad_core import tiny_scene
from reference.ns_camera_utils_v103 import auto_orient_and_center_poses
from reference.bad_render import render_single


@pytest.mark.parametrize("seed", [0, 7, 19])
def test_coordinates_against_pinned_nerfstudio(seed):
    torch.manual_seed(seed)
    original = pp.randn_SE3(11, sigma=.3).matrix().detach().cpu()
    xyz = torch.randn(8, 3)
    actual, points, transform, scale = normalize_colmap(original, xyz, .25)
    # Original ColmapDataParser sequence, using unmodified upstream helper.
    poses = original.clone()
    poses[:, :3, 1:3] *= -1
    poses = poses[:, [0, 2, 1, 3], :]
    poses[:, 2, :] *= -1
    expected, orient = auto_orient_and_center_poses(poses, method="up", center_method="poses")
    expected_scale = 1.0 / float(torch.max(torch.abs(expected[:, :3, 3])))
    expected_scale *= .25
    expected[:, :3, 3] *= expected_scale
    expected[:, :3, 1:3] *= -1
    applied = torch.eye(4)[:3, :][[0, 2, 1], :]
    applied[2, :] *= -1
    total = orient @ torch.cat([applied, torch.tensor([[0, 0, 0, 1]])], dim=0)
    expected_points = torch.cat([xyz, torch.ones_like(xyz[:, :1])], -1) @ total.T
    expected_points *= expected_scale
    torch.testing.assert_close(actual[:, :3, :], expected, rtol=0, atol=0)
    torch.testing.assert_close(points, expected_points, rtol=0, atol=0)
    torch.testing.assert_close(transform, total, rtol=0, atol=0)
    assert scale == expected_scale


@pytest.mark.parametrize("mode", ["linear", "cubic"])
def test_local_opengl_controls_match_official_composition(mode):
    _, camera, trajectory, _ = tiny_scene(BadConfig(trajectory=mode))
    with torch.no_grad():
        trajectory.controls.copy_(pp.randn_se3(*trajectory.controls.shape[:-1], sigma=.1))
    flip = torch.diag(torch.tensor([1., -1, -1, 1]))
    native_base = trajectory.base_c2w[0] @ flip
    times = torch.linspace(0, 1, 10)
    native = native_base @ interpolate(trajectory.controls[0].Exp(), times, mode).matrix()
    torch.testing.assert_close(trajectory.poses("a") @ flip, native, rtol=0, atol=0)


def fake_backend():
    # Differentiable stand-in for call-contract tests, NOT the production renderer.
    def project(means, scales, glob, quats, view, fx, fy, cx, cy, h, w, block):
        assert view.shape == (3, 4) and glob == 1 and block == 16
        m = means @ view[:3, :3].T + view[:3, 3]
        xy = m[:, :2]/m[:, 2:] * m.new_tensor([fx, fy]) + m.new_tensor([cx, cy])
        return xy, m[:, 2], torch.ones(len(m)), scales, scales.mean(-1), torch.ones(len(m), dtype=torch.int32), quats
    def raster(xy, depth, radii, conics, tiles, colors, alpha, h, w, block, background, return_alpha):
        a = alpha.mean()
        rgb = (colors*alpha).mean(0) + background*(1-a) + .00001*(xy.sum()+conics.sum())
        return rgb[None, None].expand(h, w, 3), a.expand(h, w)
    def sh(degree, directions, colors):
        from utils.sh_utils import eval_sh
        return eval_sh(degree, colors.transpose(1, 2), directions)
    return project, raster, sh


@pytest.mark.parametrize("raster_mode", ["classic", "antialiased"])
@pytest.mark.parametrize("degree", [0, 3])
def test_renderer_call_contract_and_backward_matches_bad(monkeypatch, raster_mode, degree):
    cfg = BadConfig(sh_degree=degree, rasterize_mode=raster_mode)
    model, camera, trajectory, _ = tiny_scene(cfg)
    model.active_sh_degree = degree
    backend = fake_backend()
    monkeypatch.setattr(renderer, "require_backend", lambda: backend)
    camera = trajectory.cameras(camera)[-1]
    bg = torch.tensor([.2, .3, .1])
    actual = renderer.render(camera, model, cfg, bg)
    flip = torch.diag(torch.tensor([1., -1, -1, 1]))
    k = camera.K.clone()
    k[:2, 2] += .5
    expected, alpha = render_single(camera.c2w @ flip, k, model, cfg, 48, 32, bg, backend)
    torch.testing.assert_close(actual["render"], expected, rtol=0, atol=0)
    torch.testing.assert_close(actual["alpha"], alpha, rtol=0, atol=0)
    params = tuple(model.parameters())+(trajectory.controls,)
    ga = torch.autograd.grad(actual["render"].sum(), params, retain_graph=True, allow_unused=True)
    gb = torch.autograd.grad(expected.sum(), params, allow_unused=True)
    for a, b in zip(ga, gb):
        if a is None:
            assert b is None
        else:
            torch.testing.assert_close(a, b, atol=1e-6, rtol=1e-6)


def test_old_checkpoints_are_rejected():
    with pytest.raises(ValueError, match="v2"):
        restore({"bad_blur_gs_version": 1}, None, None, None, None, None, None)


def make_colmap(root, actual_size=(96, 64)):
    from PIL import Image
    sparse = root/"sparse"/"0"
    sparse.mkdir(parents=True)
    images = root/"images"
    images.mkdir()
    (sparse/"cameras.txt").write_text("1 PINHOLE 48 32 40 41 24 16\n")
    (sparse/"images.txt").write_text("".join(
        f"{i+1} 1 0 0 0 {i*.1} 0 0 1 {i:03}.png\n\n" for i in range(9)))
    (sparse/"points3D.txt").write_text("\n".join(
        f"{i} {i*.01} 0 2 128 64 32 0" for i in range(4)))
    for i in range(9):
        Image.new("RGB", actual_size, (i, 40, 60)).save(images/f"{i:03}.png")


def test_bad_intrinsic_correction_and_precomputed_resolution(tmp_path):
    from PIL import Image
    make_colmap(tmp_path)
    train, test, _, _, _ = load_scene(tmp_path, "", 1, True, 8, .25, "cpu")
    assert train[0].image_width == 96 and train[0].K[0, 0] == 80
    assert train[0].K[0, 2] == 47.5 and train[0].K[1, 2] == 31.5
    image, camera = load_image(train[0], "cpu")
    assert image.shape == (3, 64, 96)
    with pytest.raises(FileNotFoundError, match="precomputed"):
        load_scene(tmp_path, "", 2, True, 8, .25, "cpu")
    (tmp_path/"images_2").mkdir()
    for path in (tmp_path/"images").glob("*.png"):
        Image.new("RGB", (48, 32), (7, 8, 9)).save(tmp_path/"images_2"/path.name)
    small, _, _, _, _ = load_scene(tmp_path, "", 2, True, 8, .25, "cpu")
    assert "images_2" in small[0].image_path
    assert small[0].K[0, 0] == 40 and small[0].K[0, 2] == 23.5
    pixels, _ = load_image(small[0], "cpu")
    torch.testing.assert_close(pixels[:, 0, 0], torch.tensor([7, 8, 9])/255)


def test_hold_file_and_normalization_independent_of_split(tmp_path):
    make_colmap(tmp_path, actual_size=(48, 32))
    _, _, points8, _, data8 = load_scene(tmp_path, "", 1, True, 8, .25, "cpu")
    (tmp_path/"hold=3").touch()
    _, test, points3, _, data3 = load_scene(tmp_path, "", 1, True, 8, .25, "cpu")
    assert [c.image_name for c in test] == ["000", "003", "006"]
    assert data3["llffhold"] == 3
    torch.testing.assert_close(points3, points8, atol=0, rtol=0)
    assert data3["transform"] == data8["transform"] and data3["scale"] == data8["scale"]
