"""Independent single-view transcription of BAD bdd8b3e get_outputs.

Apache-2.0, WU-CVGL/BAD-Gaussians. Keeps native OpenGL camera inputs and
legacy gsplat calls so tests compare the adapter against the upstream path.
Only crop/viewer/empty-view handling is omitted for the visible test fixture.
"""
import torch


def render_single(c2w_gl, k_corner, model, config, width, height, background, backend):
    project_gaussians, rasterize_gaussians, spherical_harmonics = backend
    R = c2w_gl[:3, :3]
    T = c2w_gl[:3, 3:4]
    R_edit = torch.diag(torch.tensor([1, -1, -1], device=R.device, dtype=R.dtype))
    R = R @ R_edit
    R_inv = R.T
    T_inv = -R_inv @ T
    viewmat = torch.eye(4, device=R.device, dtype=R.dtype)
    viewmat[:3, :3] = R_inv
    viewmat[:3, 3:4] = T_inv
    means, scales, quats = (model.params[n] for n in ("means", "scales", "quats"))
    features_dc, features_rest, opacities = (model.params[n] for n in ("features_dc", "features_rest", "opacities"))
    colors = torch.cat((features_dc[:, None, :], features_rest), dim=1)
    xys, depths, radii, conics, comp, num_tiles_hit, _ = project_gaussians(
        means, torch.exp(scales), 1, quats/quats.norm(dim=-1, keepdim=True),
        viewmat.squeeze()[:3, :], k_corner[0, 0].item(), k_corner[1, 1].item(),
        k_corner[0, 2].item(), k_corner[1, 2].item(), height, width, 16)
    if config.sh_degree > 0:
        viewdirs = means.detach() - c2w_gl.detach()[:3, 3]
        viewdirs = viewdirs/viewdirs.norm(dim=-1, keepdim=True)
        rgbs = torch.clamp(spherical_harmonics(model.active_sh_degree, viewdirs, colors)+.5, min=0)
    else:
        rgbs = torch.sigmoid(colors[:, 0, :])
    alphas = torch.sigmoid(opacities)
    if config.rasterize_mode == "antialiased":
        alphas = alphas*comp[:, None]
    rgb, alpha = rasterize_gaussians(xys, depths, radii, conics, num_tiles_hit,
                                    rgbs, alphas, height, width, 16,
                                    background=background, return_alpha=True)
    return torch.clamp(rgb, max=1).permute(2, 0, 1), alpha[None]
