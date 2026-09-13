"""Endpoint blur-flow consistency with one global direction per image."""
import torch
from .geometry import camera_intrinsics, network_pixels_in_camera, reproject, sample_map, in_bounds


def directional_flow(depth_src, alpha_src, depth_dst, alpha_dst, source_w2c,
                     target_w2c, intrinsics, pixels, flow_scale, min_alpha,
                     occlusion_tolerance):
    z = sample_map(depth_src, pixels)[0]
    a = sample_map(alpha_src, pixels)[0]
    projected, target_z = reproject(pixels, z, source_w2c, target_w2c, intrinsics)
    # Sampling/visibility selection must never become a way to learn confidence.
    with torch.no_grad():
        dst_z = sample_map(depth_dst.detach(), projected.detach())[0]
        dst_a = sample_map(alpha_dst.detach(), projected.detach())[0]
        h, w = depth_src.shape[-2:]
        valid = (in_bounds(pixels, h, w) & in_bounds(projected, h, w)
                 & (a >= min_alpha) & (dst_a >= min_alpha) & (z > 0) & (target_z > 0)
                 & torch.isfinite(z) & torch.isfinite(target_z) & torch.isfinite(dst_z))
        if occlusion_tolerance > 0:
            valid &= (target_z - dst_z).abs() <= occlusion_tolerance * dst_z.abs().clamp_min(1e-6)
    flow = ((projected - pixels) * flow_scale).permute(2, 0, 1)
    return flow, valid


def global_direction_loss(forward, backward, observed, mask, valid_forward,
                          valid_backward, *, direction="auto", eps=1e-3,
                          min_valid_fraction=0.05):
    if direction not in ("auto", "forward", "backward"):
        raise ValueError(f"Unknown direction: {direction}")
    # Fair global comparison: both candidates use the same support in auto mode.
    valid = (valid_forward & valid_backward if direction == "auto" else
             valid_forward if direction == "forward" else valid_backward)
    support = (mask * valid).detach()
    fraction = float((support > 0).float().mean())
    # Make invalid projections finite before any nonlinear penalty (NaN * 0 is NaN).
    errors = []
    for candidate in (forward, backward):
        residual = torch.where(valid[None], candidate - observed, torch.zeros_like(candidate))
        robust = (residual.square().sum(0) + eps * eps).sqrt() - eps
        errors.append((robust * support).sum() / support.sum().clamp_min(1e-6))
    selected = (int(errors[1].detach() < errors[0].detach()) if direction == "auto"
                else int(direction == "backward"))
    if fraction < min_valid_fraction:
        return (torch.nan_to_num(forward).sum() + torch.nan_to_num(backward).sum()) * 0, dict(
            valid_fraction=fraction, direction=selected, skipped=1)
    return errors[selected], dict(valid_fraction=fraction, direction=selected, skipped=0,
                                  forward_error=float(errors[0].detach()), backward_error=float(errors[1].detach()))


def blur_flow_loss(cameras, depths, observed, mask, crop, *, min_alpha=0.5,
                   occlusion_tolerance=0.1, direction="auto", min_valid_fraction=0.05):
    first, last = cameras[0], cameras[-1]
    w0, w1 = first.world_view_transform.T, last.world_view_transform.T
    k = camera_intrinsics(first)
    pixels, scale = network_pixels_in_camera(crop, depths[0][0].shape[-2:], observed.shape[-2:],
                                             device=observed.device, dtype=observed.dtype)
    d0, a0 = depths[0][:2]
    d1, a1 = depths[1][:2]
    fw, vf = directional_flow(d0, a0, d1, a1, w0, w1, k, pixels, scale, min_alpha, occlusion_tolerance)
    bw, vb = directional_flow(d1, a1, d0, a0, w1, w0, k, pixels, scale, min_alpha, occlusion_tolerance)
    return global_direction_loss(fw, bw, observed, mask, vf, vb, direction=direction,
                                  min_valid_fraction=min_valid_fraction)
