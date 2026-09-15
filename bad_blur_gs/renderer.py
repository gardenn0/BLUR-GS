"""BAD bdd8b3e rendering calls, pinned to gsplat 0.1.11.

Adapted from BAD-Gaussians (Apache-2.0). The legacy backend's approximate
camera-rotation backward is deliberately preserved, not replaced or patched.
"""
import torch
from blur_gs.depth import normalize_z_features, z_features


def require_backend():
    from importlib.metadata import version
    if torch.__version__.split("+")[0] != "2.1.2" or version("torchvision").split("+")[0] != "0.16.2":
        raise RuntimeError("Use the BAD-aligned Python 3.10 environment: torch 2.1.2 / torchvision 0.16.2")
    if version("gsplat") != "0.1.11":
        raise RuntimeError("BAD alignment requires gsplat==0.1.11; reinstall requirements-bad.txt")
    from gsplat.project_gaussians import project_gaussians
    from gsplat.rasterize import rasterize_gaussians
    from gsplat.sh import spherical_harmonics
    return project_gaussians, rasterize_gaussians, spherical_harmonics


def render(camera, model, config, background, override_color=None, retain_stats=False):
    project, rasterize, spherical_harmonics = require_backend()
    # K uses integer pixel centers for flow; gsplat uses corner coordinates.
    k = camera.K.clone()
    k[:2, 2] += 0.5
    height, width = camera.image_height, camera.image_width
    xys, depths, radii, conics, comp, tiles, _ = project(
        model.get_xyz, model.get_scaling, 1,
        model.params["quats"] / model.params["quats"].norm(dim=-1, keepdim=True),
        camera.world_view_transform.T[:3, :],
        k[0, 0].item(), k[1, 1].item(), k[0, 2].item(), k[1, 2].item(),
        height, width, 16)
    info = dict(means2d=xys, radii=radii, width=width, height=height)
    if retain_stats and xys.requires_grad:
        xys.retain_grad()
    if radii.sum() == 0:
        return dict(render=background[:, None, None].expand(3, height, width),
                    alpha=background.new_zeros(1, height, width), info=info)
    if override_color is not None:
        colors = override_color
    elif model.max_sh_degree > 0:
        directions = model.get_xyz.detach() - camera.c2w[:3, 3].detach()
        directions = directions / directions.norm(dim=-1, keepdim=True)
        colors = (spherical_harmonics(model.active_sh_degree, directions, model.get_features)+0.5).clamp_min(0)
    else:
        colors = model.get_features[:, 0, :].sigmoid()
    opacity = model.get_opacity
    if config.rasterize_mode == "antialiased":
        opacity = opacity * comp[:, None]
    image, alpha = rasterize(xys, depths, radii, conics, tiles, colors, opacity,
                            height, width, 16, background=background, return_alpha=True)
    if override_color is None:
        image = image.clamp(max=1)
    return dict(render=image.permute(2, 0, 1), alpha=alpha[None], info=info)


def render_depth(camera, model, config, *, geometry_grad, min_alpha, render_fn=render):
    from dataclasses import replace
    fixed = replace(camera, c2w=camera.c2w.detach(), K=camera.K.detach())
    with torch.set_grad_enabled(geometry_grad):
        features = z_features(model.get_xyz, fixed.world_view_transform.T)
        result = render_fn(fixed, model, config, features.new_zeros(3), override_color=features)
        return normalize_z_features(result["render"], min_alpha)
