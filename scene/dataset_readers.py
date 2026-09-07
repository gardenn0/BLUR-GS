"""Explicit manifest, image and fixed motion-cache loading."""

import json
from pathlib import Path

import numpy as np
import torch

from .cameras import Frame, validate_camera
from utils.image_utils import read_image


def load_manifest(path: str | Path) -> tuple[Path, dict]:
    path = Path(path).resolve()
    manifest = json.loads(path.read_text(encoding="utf-8"))
    if manifest.get("version") != 1 or manifest.get("camera_convention") != "opencv_w2c":
        raise ValueError("Manifest requires version=1 and camera_convention='opencv_w2c'")
    frames = manifest.get("frames", [])
    if not frames or len({f["name"] for f in frames}) != len(frames):
        raise ValueError("Manifest must contain frames with unique names")
    return path.parent, manifest


def read_frame(root: Path, entry: dict, require_motion: bool = True) -> Frame:
    image = read_image(root / entry["image"])
    K, pose = (
        torch.tensor(entry["K"], dtype=torch.float32),
        torch.tensor(entry["w2c"], dtype=torch.float32),
    )
    validate_camera(K, pose)
    frame = Frame(entry["name"], image, K, pose, exposure_seconds=entry.get("exposure_seconds"))
    if frame.exposure_seconds is not None and frame.exposure_seconds <= 0:
        raise ValueError("Exposure seconds must be positive")
    if "initial_depth" in entry:
        frame.initial_depth = torch.from_numpy(
            np.load(root / entry["initial_depth"], allow_pickle=False)
        ).float()
        if frame.initial_depth.shape != image.shape[:2]:
            raise ValueError("Initial depth must match image resolution and midpoint camera")
    if "motion" not in entry:
        if require_motion:
            raise ValueError(
                f"{frame.name}: missing motion cache; run scripts/prepare_motion.py first"
            )
        return frame
    with np.load(root / entry["motion"], allow_pickle=False) as cache:
        frame.flow = torch.from_numpy(cache["flow"].copy()).float()
        frame.confidence = torch.from_numpy(cache["confidence"].copy()).float()
        frame.flow_reference = str(cache["reference"].item())
        frame.sign_ambiguous = bool(cache["sign_ambiguous"].item())
        if "depth" in cache:
            frame.observed_depth = torch.from_numpy(cache["depth"].copy()).float()
    if frame.flow.shape != (*image.shape[:2], 2) or frame.confidence.shape != image.shape[:2]:
        raise ValueError(f"{frame.name}: motion cache resolution does not match image")
    if frame.flow_reference not in {"start", "midpoint"}:
        raise ValueError("Flow cache must specify start or midpoint reference")
    valid = torch.isfinite(frame.flow).all(-1) & torch.isfinite(frame.confidence)
    frame.confidence = torch.where(valid, frame.confidence.clamp(0, 1), 0)
    frame.flow = torch.nan_to_num(frame.flow)
    if frame.observed_depth is not None and frame.observed_depth.shape != image.shape[:2]:
        raise ValueError("Observed depth must match image resolution")
    return frame


def load_dataset(
    path: str | Path, require_motion: bool = True, split: str = "train"
) -> tuple[list[Frame], dict]:
    root, manifest = load_manifest(path)
    entries = [f for f in manifest["frames"] if f.get("split", "train") == split]
    if not entries:
        raise ValueError(f"No frames in split {split!r}")
    frames = [read_frame(root, entry, require_motion) for entry in entries]
    with np.load(root / manifest["point_cloud"], allow_pickle=False) as data:
        cloud = {
            name: torch.from_numpy(data[name].copy()).float()
            for name in ("points", "colors", "scales")
            if name in data
        }
    if "points" not in cloud or "colors" not in cloud:
        raise ValueError("point_cloud NPZ requires points and colors")
    return frames, cloud
