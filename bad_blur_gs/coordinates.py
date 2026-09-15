"""Default DeblurNerf/Colmap parser coordinates from Nerfstudio v1.0.3.

Adapted from camera_utils.py and colmap_dataparser.py (Apache-2.0).
Copyright 2022 the Regents of the University of California, Nerfstudio Team
and contributors. Defaults: assume COLMAP world, up, poses, auto_scale=True.
"""
import torch


def rotation_between(a, b):
    # Preserve upstream operation order, including its parallel-axis fallback.
    a = a / torch.linalg.norm(a)
    b = b / torch.linalg.norm(b)
    v = torch.linalg.cross(a, b)
    eps = 1e-6
    if torch.sum(torch.abs(v)) < eps:
        x = torch.tensor([1.0, 0, 0]) if abs(a[0]) < eps else torch.tensor([0, 1.0, 0])
        v = torch.linalg.cross(a, x)
    v = v / torch.linalg.norm(v)
    skew = torch.Tensor([[0, -v[2], v[1]], [v[2], 0, -v[0]], [-v[1], v[0], 0]])
    theta = torch.acos(torch.clip(torch.dot(a, b), -1, 1))
    return torch.eye(3) + torch.sin(theta)*skew + (1-torch.cos(theta))*(skew @ skew)


def normalize_colmap(c2w_cv, xyz, scale_factor):
    """Normalize ALL camera poses before splitting; return OpenCV adapters.

    The returned 3x4 transform maps original COLMAP world points into oriented
    coordinates BEFORE multiplication by scale. Local controls remain OpenGL.
    """
    if c2w_cv.device.type != "cpu" or xyz.device.type != "cpu":
        raise ValueError("Run official parser normalization on CPU before GPU allocation")
    poses = c2w_cv.clone()
    poses[:, :3, 1:3] *= -1  # OpenCV -> OpenGL camera axes
    poses = poses[:, [0, 2, 1, 3], :]
    poses[:, 2, :] *= -1  # COLMAP world -> Nerfstudio world
    center = poses[:, :3, 3].mean(0)
    up = poses[:, :3, 1].mean(0)
    if up.norm() < 1e-8:
        raise ValueError("Degenerate average camera up vector for BAD default orientation")
    up = up / torch.linalg.norm(up)
    rotation = rotation_between(up, torch.Tensor([0, 0, 1]))
    transform = torch.cat([rotation, rotation @ -center[..., None]], dim=-1)
    oriented = transform @ poses
    extent = float(torch.max(torch.abs(oriented[:, :3, 3])))
    if extent < 1e-8:
        raise ValueError("Need distinct camera centers for BAD auto scaling")
    scale = (1.0 / extent) * scale_factor
    oriented[:, :3, 3] *= scale
    applied = torch.eye(4)[:3, :][[0, 2, 1], :]
    applied[2, :] *= -1
    total_transform = transform @ torch.cat([applied, torch.Tensor([[0, 0, 0, 1]])], dim=0)
    points = torch.cat([xyz, torch.ones_like(xyz[:, :1])], -1) @ total_transform.T
    points *= scale
    output = torch.eye(4).repeat(len(poses), 1, 1)
    output[:, :3, :] = oriented
    output[:, :3, 1:3] *= -1  # OpenCV adapter used by shared flow geometry
    return output, points, total_transform, scale
