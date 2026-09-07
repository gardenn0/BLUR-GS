"""Confidence-weighted motion consistency, including virtual-start flow alignment."""

import torch
from torch import Tensor
from torch.nn import functional as F

from .pose_utils import sample_map
from .loss_utils import charbonnier, weighted_mean


def motion_loss(
    path: Tensor,
    path_valid: Tensor,
    observed: Tensor,
    confidence: Tensor,
    depth_valid: Tensor,
    *,
    reference: str = "start",
    ambiguous: bool = True,
    flow_weight: float = 1.0,
    magnitude_weight: float = 0.1,
    direction_weight: float = 0.01,
    magnitude: str = "endpoint",
) -> dict[str, Tensor]:
    """One temporal direction per image; never choose an independent sign per pixel.

    Image-as-an-IMU lives on a virtual-start grid. Sample F_obs at p_start(X_ref)
    to compare it to the path of the *same* scene point. For the reversed hypothesis,
    sample at p_end and reverse the predicted path. This is not simply -F_obs(x).
    Weights/validity are detached so confidence cannot be reduced to hide residuals.
    """
    if reference not in {"start", "midpoint"} or magnitude not in {"endpoint", "path"}:
        raise ValueError("Invalid flow reference or magnitude mode")
    candidates = []
    for reverse in [False, True] if ambiguous else [False]:
        p = path.flip(0) if reverse else path
        flow = p[-1] - p[0]
        if reference == "start":
            target, inside = sample_map(observed, p[0])
            conf, _ = sample_map(confidence, p[0].detach())
        else:
            target, conf = observed, confidence
            inside = torch.ones_like(confidence, dtype=torch.bool)
        valid = inside & path_valid.all(0) & depth_valid & torch.isfinite(target).all(-1)
        weight = torch.nan_to_num(conf).clamp(0, 1) * valid
        target = torch.nan_to_num(target)
        lf = weighted_mean(charbonnier(flow - target).mean(-1), weight)
        pred_mag = torch.linalg.vector_norm(flow, dim=-1)
        if magnitude == "path":
            pred_mag = torch.linalg.vector_norm(p[1:] - p[:-1], dim=-1).sum(0)
        obs_mag = torch.linalg.vector_norm(target, dim=-1)
        lm = weighted_mean(charbonnier(pred_mag - obs_mag), weight)
        cosine = (F.normalize(flow, dim=-1, eps=1e-6) * F.normalize(target, dim=-1, eps=1e-6)).sum(
            -1
        )
        # No directional evidence exists for effectively zero motion.
        ld = weighted_mean(1 - cosine.clamp(-1, 1), weight * (obs_mag > 0.1))
        total = flow_weight * lf + magnitude_weight * lm + direction_weight * ld
        candidates.append(
            {
                "motion": total,
                "flow": lf,
                "magnitude": lm,
                "direction": ld,
                "valid_fraction": (weight > 0).float().mean(),
                "reversed": path.new_tensor(float(reverse)),
            }
        )
    supported = [i for i, candidate in enumerate(candidates) if candidate["valid_fraction"] > 0]
    # An out-of-frame hypothesis with no observations must not win by having zero loss.
    choice = min(supported or [0], key=lambda i: candidates[i]["motion"].detach().item())
    return candidates[choice]


def path_consistency(predicted: Tensor, observed: Tensor, confidence: Tensor) -> Tensor:
    """Eq. 63 for optional time-aligned paths on the SAME reference grid, pixel units.

    The Image-as-an-IMU checkpoint does not predict these paths; callers must provide
    real path observations. Shapes are [S,H,W,2] and [H,W].
    """
    if predicted.shape != observed.shape:
        raise ValueError("Path samples and timestamps must be aligned before comparison")
    valid = torch.isfinite(observed).all(dim=(0, 3))
    error = charbonnier(predicted - torch.nan_to_num(observed)).mean(dim=(0, 3))
    return weighted_mean(error, confidence * valid)
