"""Auxiliary normalized z-depth through the existing RGB rasterizer.

Inspired by SplaTAM's [z, silhouette, z^2] feature rendering. This implementation
uses [z, 1, 0] and explicitly normalizes by accumulated alpha. No raw CUDA depth
output participates in the loss. See THIRD_PARTY_NOTICES.md.
"""
import copy
import torch


def z_features(xyz, w2c):
    z = (xyz @ w2c[:3, :3].T + w2c[:3, 3])[:, 2]
    return torch.stack((z, torch.ones_like(z), torch.zeros_like(z)), -1)


def normalize_z_features(rendered, min_alpha=0.5):
    alpha = rendered[1:2]
    depth = rendered[:1] / alpha.clamp_min(1e-6)
    valid = ((alpha >= min_alpha) & torch.isfinite(depth) & (depth > 0)).detach()
    return depth, alpha, valid


def render_z_depth(camera, gaussians, pipe, kernel_size, *, geometry_grad,
                   min_alpha=0.5, render_fn=None):
    if render_fn is None:
        from gaussian_renderer import render as render_fn
    # Block auxiliary rasterizer camera gradients intentionally. Trajectory
    # gradients use the explicit reprojection; geometry gradients use z/features.
    fixed = copy.copy(camera)
    for name in ("world_view_transform", "full_proj_transform", "projection_matrix", "camera_center"):
        setattr(fixed, name, getattr(camera, name).detach())
    with torch.set_grad_enabled(geometry_grad):
        features = z_features(gaussians.get_xyz, fixed.world_view_transform.T)
        result = render_fn(fixed, gaussians, pipe, features.new_zeros(3),
                           kernel_size=kernel_size, override_color=features,
                           subpixel_offset=None, compute_grad_cov2d=True)
        return normalize_z_features(result["render"], min_alpha)
