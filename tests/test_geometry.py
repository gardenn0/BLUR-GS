import torch
from torch.autograd import gradcheck
from blur_gs.geometry import pixel_grid, reproject, network_pixels_in_camera, sample_map
from blur_gs.depth import z_features, normalize_z_features


def setup(dtype=torch.float64):
    pixels = pixel_grid(5, 7, dtype=dtype)
    k = torch.tensor([[100., 0, 3], [0, 100., 2], [0, 0, 1]], dtype=dtype)
    w = torch.eye(4, dtype=dtype)
    return pixels, k, w


def test_identity_and_translation():
    pixels, k, w = setup()
    z = torch.full((5, 7), 2., dtype=w.dtype)
    uv, _ = reproject(pixels, z, w, w, k)
    torch.testing.assert_close(uv, pixels)
    target = w.clone()
    target[0, 3] = -0.1
    uv, _ = reproject(pixels, z, w, target, k)
    torch.testing.assert_close(uv - pixels, torch.tensor([-5., 0.], dtype=w.dtype).expand_as(uv))


def test_depth_and_pose_gradcheck():
    pixels, k, w = setup()
    z = torch.full((5, 7), 2., dtype=w.dtype, requires_grad=True)
    t = torch.tensor([-.1, .02, .03], dtype=w.dtype, requires_grad=True)
    def fn(z, t):
        target = torch.cat((torch.cat((w[:3, :3], t[:, None]), 1), w[3:]), 0)
        return reproject(pixels, z, w, target, k)[0]
    assert gradcheck(fn, (z, t))


def test_rotation_flow_is_depth_independent():
    pixels, k, w = setup()
    angle = torch.tensor(.1, dtype=w.dtype)
    target = w.clone()
    target[:3, :3] = torch.tensor([[angle.cos(), 0, angle.sin()], [0, 1, 0],
                                 [-angle.sin(), 0, angle.cos()]], dtype=w.dtype)
    a, _ = reproject(pixels, torch.ones(5, 7, dtype=w.dtype), w, target, k)
    b, _ = reproject(pixels, torch.full((5, 7), 3., dtype=w.dtype), w, target, k)
    torch.testing.assert_close(a, b)


def test_crop_resize_half_pixel_mapping():
    crop = dict(original_width=640, original_height=480, left=0, top=16, width=640, height=448)
    pixels, scale = network_pixels_in_camera(crop, (240, 320), (224, 320), device="cpu", dtype=torch.float64)
    torch.testing.assert_close(pixels[0, 0], torch.tensor([0., 8.], dtype=pixels.dtype))
    torch.testing.assert_close(scale, torch.ones(2, dtype=pixels.dtype))
    # Native training resolution converts a 2-pixel displacement to 1 cache pixel.
    _, full_scale = network_pixels_in_camera(crop, (480, 640), (224, 320), device="cpu", dtype=torch.float64)
    torch.testing.assert_close(full_scale, torch.full((2,), .5, dtype=pixels.dtype))


def test_sampling_pixel_centers():
    image = torch.arange(35, dtype=torch.float64).reshape(1, 5, 7)
    torch.testing.assert_close(sample_map(image, pixel_grid(5, 7, dtype=image.dtype)), image)


def test_auxiliary_z_gradcheck_nonrigid_camera():
    xyz = torch.tensor([[.2, .3, 2.], [.5, .1, 4.]], dtype=torch.float64, requires_grad=True)
    w = torch.eye(4, dtype=xyz.dtype)
    w[2, 0] = .15
    weights = torch.tensor([.4, .36], dtype=xyz.dtype)
    def fn(xyz):
        channels = (weights[:, None] * z_features(xyz, w)).sum(0).reshape(3, 1, 1)
        return normalize_z_features(channels)[0]
    expected = ((xyz.detach() @ w[:3, :3].T)[:, 2] * weights).sum() / weights.sum()
    torch.testing.assert_close(fn(xyz).squeeze(), expected)
    assert gradcheck(fn, (xyz,))


def test_depth_optimization_from_known_translation():
    pixels, k, w = setup()
    target = w.clone()
    target[0, 3] = -.1
    log_z = torch.tensor(0., dtype=w.dtype, requires_grad=True)
    opt = torch.optim.Adam([log_z], lr=.05)
    for _ in range(200):
        uv, _ = reproject(pixels, log_z.exp().expand(5, 7), w, target, k)
        loss = ((uv[..., 0] - pixels[..., 0] + 5) ** 2).mean()
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert abs(log_z.exp().item() - 2.) < .005
