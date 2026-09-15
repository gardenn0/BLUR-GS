"""3DGS rasterization with pose gradients; no Nerfstudio/CoMo dependency.

gsplat 1.5.3 replaces the old gsplat 0.x backend. Parameter representation is
standard 3DGS. Rasterization version is an explicit deviation from official BAD.
"""
import torch
from blur_gs.depth import normalize_z_features, z_features


def require_backend():
    import gsplat
    if gsplat.__version__ != "1.5.3":
        raise RuntimeError(f"Install gsplat==1.5.3; found {gsplat.__version__}")
    return gsplat.rasterization


def render(camera, model, config, background, override_color=None, retain_stats=False):
    rasterization = require_backend()
    # SH viewing direction gradients are detached as in the BAD implementation.
    if override_color is None:
        if model.max_sh_degree == 0:
            colors = model.params["features_dc"].sigmoid()
        else:
            from utils.sh_utils import eval_sh
            directions = model.get_xyz.detach() - camera.c2w[:3, 3].detach()
            directions = torch.nn.functional.normalize(directions, dim=-1)
            colors = (eval_sh(model.active_sh_degree, model.get_features.transpose(1, 2), directions)+0.5).clamp_min(0)
    else:
        colors = override_color
    # gsplat rasterizes at x+0.5/y+0.5; shared flow geometry uses integer centers.
    k = camera.K.clone()
    k[:2, 2] += 0.5
    image, alpha, info = rasterization(
        means=model.get_xyz, quats=model.get_rotation, scales=model.get_scaling,
        opacities=model.get_opacity.flatten(), colors=colors,
        viewmats=camera.world_view_transform.T[None], Ks=k[None],
        width=camera.image_width, height=camera.image_height,
        near_plane=0.01, far_plane=1e10, eps2d=0.3, packed=False,
        backgrounds=background[None], rasterize_mode=config.rasterize_mode,
        render_mode="RGB", absgrad=False)
    if retain_stats and info["means2d"].requires_grad:
        info["means2d"].retain_grad()
    # Do not clamp z feature channels! RGB clipping is upstream BAD behavior.
    result = image[0].permute(2, 0, 1)
    if override_color is None:
        result = result.clamp(max=1)
    return {"render": result, "alpha": alpha[0].permute(2, 0, 1), "info": info}


def render_depth(camera, model, config, *, geometry_grad, min_alpha, render_fn=render):
    from dataclasses import replace
    fixed = replace(camera, c2w=camera.c2w.detach(), K=camera.K.detach())
    with torch.set_grad_enabled(geometry_grad):
        features = z_features(model.get_xyz, fixed.world_view_transform.T)
        result = render_fn(fixed, model, config, features.new_zeros(3), override_color=features)
        return normalize_z_features(result["render"], min_alpha)
