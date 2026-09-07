"""OpenCV coordinates, world-to-camera poses, twists ordered (v, omega).

Pixel centers are integer coordinates. Depth is camera z, not distance along a ray.
"""

import torch
from torch import Tensor
from torch.nn import functional as F


def skew(v: Tensor) -> Tensor:
    x, y, z = v.unbind(-1)
    o = torch.zeros_like(x)
    return torch.stack((o, -z, y, z, o, -x, -y, x, o), -1).reshape(*v.shape[:-1], 3, 3)


def se3_exp(xi: Tensor) -> Tensor:
    """SE(3) exponential with finite first/second derivatives at zero rotation."""
    v, w = xi[..., :3], xi[..., 3:]
    theta2 = (w * w).sum(-1, keepdim=True)
    # Clamp the unused analytic branch too: torch.where alone does not prevent NaN gradients.
    safe2 = theta2.clamp_min(1e-8)
    theta = safe2.sqrt()
    a = torch.where(theta2 < 1e-6, 1 - theta2 / 6 + theta2.square() / 120, theta.sin() / theta)
    b = torch.where(
        theta2 < 1e-6, 0.5 - theta2 / 24 + theta2.square() / 720, (1 - theta.cos()) / safe2
    )
    c = torch.where(
        theta2 < 1e-6,
        1 / 6 - theta2 / 120 + theta2.square() / 5040,
        (theta - theta.sin()) / (safe2 * theta),
    )
    W = skew(w)
    W2 = W @ W
    eye = torch.eye(3, dtype=xi.dtype, device=xi.device).expand(*xi.shape[:-1], 3, 3)
    R = eye + a[..., None] * W + b[..., None] * W2
    V = eye + b[..., None] * W + c[..., None] * W2
    top = torch.cat((R, (V @ v[..., None])), -1)
    bottom = xi.new_tensor([0, 0, 0, 1]).expand(*xi.shape[:-1], 1, 4)
    return torch.cat((top, bottom), -2)


def inverse_pose(T: Tensor) -> Tensor:
    R = T[..., :3, :3].transpose(-1, -2)
    top = torch.cat((R, -R @ T[..., :3, 3:4]), -1)
    bottom = T.new_tensor([0, 0, 0, 1]).expand(*T.shape[:-2], 1, 4)
    return torch.cat((top, bottom), -2)


def adjoint(T: Tensor, xi: Tensor) -> Tensor:
    """Twist for T exp(xi) inv(T)."""
    R, t = T[..., :3, :3], T[..., :3, 3]
    w = (R @ xi[..., 3:, None]).squeeze(-1)
    v = (R @ xi[..., :3, None]).squeeze(-1) + torch.cross(t, w, dim=-1)
    return torch.cat((v, w), -1)


def pixel_grid(height: int, width: int, like: Tensor) -> Tensor:
    y, x = torch.meshgrid(
        torch.arange(height, device=like.device, dtype=like.dtype),
        torch.arange(width, device=like.device, dtype=like.dtype),
        indexing="ij",
    )
    return torch.stack((x, y), -1)


def backproject(depth: Tensor, K: Tensor) -> Tensor:
    grid = pixel_grid(*depth.shape, depth)
    homogeneous = torch.cat((grid, torch.ones_like(depth[..., None])), -1)
    return (homogeneous @ torch.linalg.inv(K).T) * depth[..., None]


def project(points: Tensor, K: Tensor, near: float = 1e-4) -> tuple[Tensor, Tensor]:
    uvz = points @ K.T
    valid = torch.isfinite(points).all(-1) & (points[..., 2] > near)
    uv = uvz[..., :2] / uvz[..., 2:].clamp_min(near)
    return uv, valid


def exposure_path(
    depth: Tensor, K: Tensor, reference: Tensor, poses: Tensor
) -> tuple[Tensor, Tensor]:
    """Eq. 29: p(t) = pi(K T(t) inv(T_ref) X_ref), shape [S,H,W,2]."""
    X = backproject(depth, K)
    relative = poses @ inverse_pose(reference)
    moved = torch.einsum("sij,hwj->shwi", relative[:, :3, :3], X)
    moved = moved + relative[:, None, None, :3, 3]
    uv, valid = project(moved, K)
    return uv, valid & torch.isfinite(depth)[None] & (depth[None] > 0)


def sample_map(values: Tensor, coordinates: Tensor) -> tuple[Tensor, Tensor]:
    """Bilinearly sample HWC/HW on an HW2 pixel grid; invalid borders have zero weight."""
    scalar = values.ndim == 2
    if scalar:
        values = values[..., None]
    h, w = values.shape[:2]
    xy = torch.nan_to_num(coordinates)
    normalized = torch.stack((2 * (xy[..., 0] + 0.5) / w - 1, 2 * (xy[..., 1] + 0.5) / h - 1), -1)
    out = F.grid_sample(
        values.permute(2, 0, 1)[None],
        normalized[None],
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    )
    out = out[0].permute(1, 2, 0)
    valid = (
        torch.isfinite(coordinates).all(-1)
        & (xy[..., 0] >= 0)
        & (xy[..., 0] <= w - 1)
        & (xy[..., 1] >= 0)
        & (xy[..., 1] <= h - 1)
    )
    return (out[..., 0] if scalar else out), valid


def motion_jacobian(depth: Tensor, K: Tensor) -> Tensor:
    """Image-as-an-IMU Eqs. 6-9, generalized to fx != fy and arbitrary principal point.

    Maps *camera body displacement* [tx,ty,tz,rx,ry,rz] to image displacement.
    A positive camera translation produces a negative image displacement.
    """
    xy = pixel_grid(*depth.shape, depth)
    x = (xy[..., 0] - K[0, 2]) / K[0, 0]
    y = (xy[..., 1] - K[1, 2]) / K[1, 1]
    iz = depth.clamp_min(1e-8).reciprocal()
    o = torch.zeros_like(x)
    fx, fy = K[0, 0], K[1, 1]
    row_x = torch.stack((-fx * iz, o, fx * x * iz, fx * x * y, -fx * (1 + x * x), fx * y), -1)
    row_y = torch.stack((o, -fy * iz, fy * y * iz, fy * (1 + y * y), -fy * x * y, -fy * x), -1)
    return torch.stack((row_x, row_y), -2)


def solve_camera_motion(
    flow: Tensor, depth: Tensor, K: Tensor, confidence: Tensor, damping: float = 1e-6
) -> Tensor:
    """Differentiable, column-scaled weighted least squares; displacement per exposure.

    Damping keeps degenerate views finite. This cannot recover absolute metric scale
    from arbitrary-scale SfM depth. Divide by measured exposure seconds for velocity.
    """
    valid = (
        torch.isfinite(flow).all(-1)
        & torch.isfinite(depth)
        & (depth > 0)
        & torch.isfinite(confidence)
        & (confidence > 0)
    )
    if valid.sum() < 6:
        raise ValueError("At least six valid depth/flow measurements are required")
    A = motion_jacobian(torch.nan_to_num(depth, nan=1), K)[valid].reshape(-1, 6)
    b = flow[valid].reshape(-1)
    weight = confidence[valid].clamp(0, 1).sqrt().repeat_interleave(2)
    A, b = A * weight[:, None], b * weight
    scale = A.square().sum(0).sqrt().clamp_min(1e-8)
    A = A / scale
    normal = A.T @ A + damping * torch.eye(6, device=A.device, dtype=A.dtype)
    return torch.linalg.solve(normal, A.T @ b) / scale


def quaternion_matrix(q: Tensor) -> Tensor:
    q = F.normalize(q, dim=-1, eps=1e-8)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - z * w),
            2 * (x * z + y * w),
            2 * (x * y + z * w),
            1 - 2 * (x * x + z * z),
            2 * (y * z - x * w),
            2 * (x * z - y * w),
            2 * (y * z + x * w),
            1 - 2 * (x * x + y * y),
        ),
        -1,
    ).reshape(-1, 3, 3)
