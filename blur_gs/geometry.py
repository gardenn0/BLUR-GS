"""Differentiable pinhole geometry; all matrices here are column-vector w2c.

Pixel centers follow the CUDA rasterizer: principal point ((W-1)/2, (H-1)/2).
The matrices need not be rigid: CoMoGaussian's CMR can be non-orthogonal.
"""
import torch
import torch.nn.functional as F


def camera_intrinsics(camera):
    # Standalone BAD port carries full pixel-center intrinsics, including principal
    # point and crop/downscale transforms. Legacy CoMo cameras retain their path.
    if hasattr(camera, "K"):
        return camera.K
    ref = camera.world_view_transform
    w, h = camera.image_width, camera.image_height
    return ref.new_tensor([[camera.focal_x, 0, (w - 1) / 2],
                           [0, camera.focal_y, (h - 1) / 2], [0, 0, 1]])


def pixel_grid(height, width, *, device=None, dtype=torch.float32):
    y, x = torch.meshgrid(torch.arange(height, device=device, dtype=dtype),
                          torch.arange(width, device=device, dtype=dtype), indexing="ij")
    return torch.stack((x, y), -1)


def sample_map(image, pixels):
    """CHW map, ...x2 pixel coordinates -> Cx... (grid must be HxWx2)."""
    h, w = image.shape[-2:]
    grid = (pixels + 0.5) * pixels.new_tensor([2 / w, 2 / h]) - 1
    return F.grid_sample(image[None], grid[None], mode="bilinear",
                         padding_mode="zeros", align_corners=False)[0]


def in_bounds(pixels, height, width):
    return ((pixels[..., 0] >= 0) & (pixels[..., 0] <= width - 1)
            & (pixels[..., 1] >= 0) & (pixels[..., 1] <= height - 1))


def backproject(pixels, z, intrinsics):
    homogeneous = torch.cat((pixels, torch.ones_like(pixels[..., :1])), -1)
    rays = homogeneous @ torch.linalg.inv(intrinsics).T
    return rays * z[..., None]


def reproject(pixels, z, source_w2c, target_w2c, intrinsics):
    points = backproject(pixels, z, intrinsics)
    points = torch.cat((points, torch.ones_like(points[..., :1])), -1)
    relative = target_w2c @ torch.linalg.inv(source_w2c)
    target = (points @ relative.T)[..., :3]
    projected = target @ intrinsics.T
    target_z = target[..., 2]
    uv = projected[..., :2] / projected[..., 2:].clamp_min(1e-8)
    return uv, target_z


def network_pixels_in_camera(crop, camera_hw, network_hw, *, device, dtype):
    """Map cached IAAI crop pixels into a possibly downscaled training camera.

    crop: original_width/height, left/top/width/height; half-pixel resize mapping.
    Returns camera pixel positions and conversion of camera flow to cache pixels.
    """
    hc, wc = camera_hw
    hn, wn = network_hw
    pixels = pixel_grid(hn, wn, device=device, dtype=dtype)
    crop_size = pixels.new_tensor([crop["width"], crop["height"]])
    full_size = pixels.new_tensor([crop["original_width"], crop["original_height"]])
    net_size = pixels.new_tensor([wn, hn])
    camera_size = pixels.new_tensor([wc, hc])
    original = (pixels + 0.5) * crop_size / net_size + pixels.new_tensor([crop["left"], crop["top"]]) - 0.5
    camera_pixels = (original + 0.5) * camera_size / full_size - 0.5
    flow_scale = full_size / camera_size * net_size / crop_size
    return camera_pixels, flow_scale
