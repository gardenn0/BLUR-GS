"""Frozen adapter for the official Image-as-an-IMU network; no substitute predictor."""

import hashlib
import json
from pathlib import Path

import numpy as np
import torch
from torch import Tensor
from torch.nn import functional as F

from .data import load_manifest, read_frame


def restore_prediction(
    flow: Tensor, depth: Tensor, height: int, width: int
) -> tuple[Tensor, Tensor, Tensor]:
    """Undo official infer() center crop + 320x224 resize, preserving pixel displacement.

    Unobserved crop borders get zero confidence. Depth values do not scale with resolution.
    Crop offsets follow torchvision center_crop's round((size-crop)/2) convention.
    """
    crop_h, crop_w = height, width
    if width / height > 320 / 224:
        crop_w = int(height * 320 / 224)
    elif width / height < 320 / 224:
        crop_h = int(width / (320 / 224))
    top, left = int(round((height - crop_h) / 2)), int(round((width - crop_w) / 2))
    if flow.shape != (1, 2, 224, 320) or depth.shape != (1, 1, 224, 320):
        raise ValueError("Official predictor must output flow [1,2,224,320], depth [1,1,224,320]")
    resized_flow = F.interpolate(flow, (crop_h, crop_w), mode="bilinear", align_corners=False)[0]
    resized_flow = resized_flow * flow.new_tensor([crop_w / 320, crop_h / 224])[:, None, None]
    resized_depth = F.interpolate(depth, (crop_h, crop_w), mode="bilinear", align_corners=False)[
        0, 0
    ]
    full_flow = flow.new_zeros(height, width, 2)
    full_depth, valid = depth.new_zeros(height, width), depth.new_zeros(height, width)
    full_flow[top : top + crop_h, left : left + crop_w] = resized_flow.permute(1, 2, 0)
    full_depth[top : top + crop_h, left : left + crop_w] = resized_depth
    valid[top : top + crop_h, left : left + crop_w] = 1
    return full_flow, full_depth, valid


def heuristic_confidence(image: Tensor, depth: Tensor, valid: Tensor) -> Tensor:
    """BLUR-GS heuristic, not a learned/calibrated confidence from the IAAI paper.

    Reject invalid depth and saturated pixels; softly downweight textureless regions.
    Does not claim to detect all occlusions or dynamic objects.
    """
    gray = image.mean(-1)
    dx = F.pad((gray[:, 1:] - gray[:, :-1]).abs(), (0, 1))
    dy = F.pad((gray[1:] - gray[:-1]).abs(), (0, 0, 0, 1))
    texture = F.avg_pool2d((dx + dy)[None, None], 5, stride=1, padding=2)[0, 0]
    texture = (0.1 + texture / 0.05).clamp_max(1)
    nonsaturated = (image.max(-1).values < 0.99) & (image.min(-1).values > 0.01)
    finite = torch.isfinite(depth) & (depth > 0)
    return valid * texture * nonsaturated * finite


class ImageAsIMU:
    def __init__(self, checkpoint: str | Path, device: str = "cuda"):
        checkpoint = Path(checkpoint)
        if not checkpoint.is_file():
            raise FileNotFoundError(f"Official Image-as-an-IMU checkpoint missing: {checkpoint}")
        try:
            from iaai.model import Blur2PoseSegNeXtBackbone
        except ImportError as exc:
            raise RuntimeError(
                "Install the official iaai package; see docs/image_as_imu.md"
            ) from exc
        self.device = device
        # BLUR-GS uses only the pretrained dense field/depth heads. The official pose head
        # hardcodes CUDA and a mean focal length; our geometry solver handles general K.
        self.model = Blur2PoseSegNeXtBackbone(supervise_pose=False, device=device)
        state = torch.load(checkpoint, map_location="cpu", weights_only=True)
        self.model.load_state_dict(state, strict=True)
        self.model.to(device).eval().requires_grad_(False)

    @torch.no_grad()
    def __call__(self, image: Tensor, K: Tensor) -> tuple[Tensor, Tensor, Tensor]:
        out = self.model.infer(
            {
                "image": image.permute(2, 0, 1)[None].to(self.device),
                "fx": float(K[0, 0]),
                "fy": float(K[1, 1]),
            }
        )
        flow, depth, valid = restore_prediction(out["flow_field"], out["depth"], *image.shape[:2])
        confidence = heuristic_confidence(image.to(self.device), depth, valid)
        confidence = confidence * torch.isfinite(flow).all(-1)
        return torch.nan_to_num(flow).cpu(), depth.cpu(), confidence.cpu()


def prepare_motion(
    manifest_path: str, checkpoint: str, device: str, output: str | None = None
) -> Path:
    root, manifest = load_manifest(manifest_path)
    model = ImageAsIMU(checkpoint, device)
    with Path(checkpoint).open("rb") as stream:
        digest = hashlib.sha256()
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    cache_dir = root / "motion"
    cache_dir.mkdir(exist_ok=True)
    for i, entry in enumerate(manifest["frames"]):
        if entry.get("split", "train") != "train":
            continue
        frame = read_frame(root, {k: v for k, v in entry.items() if k != "motion"}, False)
        flow, depth, confidence = model(frame.image, frame.K)
        path = cache_dir / f"{i:06d}.npz"
        np.savez_compressed(
            path,
            flow=flow.numpy(),
            depth=depth.numpy(),
            confidence=confidence.numpy(),
            reference="start",
            sign_ambiguous=True,
            source="official-image-as-an-imu",
            checkpoint_sha256=digest.hexdigest(),
            confidence_source="blur-gs-texture-saturation-heuristic",
        )
        entry["motion"] = path.relative_to(root).as_posix()
        print(
            f"Cached {frame.name}: {float((confidence > 0).float().mean()):.1%} valid pixels",
            flush=True,
        )
    destination = Path(output) if output else root / "scene.motion.json"
    # Manifest paths remain relative to the original dataset root.
    if destination.resolve().parent != root:
        raise ValueError("Output manifest must be in the same directory as the input manifest")
    destination.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return destination
