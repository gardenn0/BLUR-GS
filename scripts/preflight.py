"""Validate a prepared real-scene dataset and the selected training backend."""

import argparse
import json
from pathlib import Path
import sys

if __package__ in (None, ""):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
import torch

from arguments import add_source_argument, resolve_source
from scene.dataset_readers import load_manifest, read_frame


def inspect_dataset(source: str) -> dict:
    manifest_path = resolve_source(source)
    root, manifest = load_manifest(manifest_path)
    point_path = root / manifest["point_cloud"]
    if not point_path.is_file():
        raise FileNotFoundError(f"Point cloud missing: {point_path}")
    with np.load(point_path, allow_pickle=False) as cloud:
        points = cloud["points"]
        colors = cloud["colors"]
        if points.ndim != 2 or points.shape[1] != 3 or not len(points):
            raise ValueError("Point cloud requires a nonempty points[N,3] array")
        if colors.shape != points.shape:
            raise ValueError("Point cloud colors must match points[N,3]")

    resolutions = set()
    valid_fractions = []
    train_frames = 0
    for entry in manifest["frames"]:
        training = entry.get("split", "train") == "train"
        frame = read_frame(root, entry, require_motion=training)
        resolutions.add(tuple(frame.image.shape[:2]))
        if training:
            train_frames += 1
            valid_fractions.append(float((frame.confidence > 0).float().mean()))
    if not train_frames:
        raise ValueError("Dataset has no training frames")
    if min(valid_fractions) <= 0:
        raise ValueError("At least one training frame has no valid motion supervision")

    return {
        "manifest": str(Path(manifest_path).resolve()),
        "frames": len(manifest["frames"]),
        "train_frames": train_frames,
        "gaussians": int(len(points)),
        "resolutions": [list(x) for x in sorted(resolutions)],
        "motion_valid_fraction": {
            "min": min(valid_fractions),
            "mean": sum(valid_fractions) / len(valid_fractions),
            "max": max(valid_fractions),
        },
    }


def inspect_runtime(backend: str, device: str) -> dict:
    cuda_requested = device.startswith("cuda") or backend == "gsplat"
    result = {
        "torch": torch.__version__,
        "device": device,
        "backend": backend,
        "cuda_available": torch.cuda.is_available(),
    }
    if cuda_requested and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but this PyTorch build cannot access CUDA")
    if backend == "gsplat":
        try:
            import gsplat
        except ImportError as exc:
            raise RuntimeError("gsplat is missing; install the cuda extra") from exc
        result["gsplat"] = getattr(gsplat, "__version__", "unknown")
        result["gpu"] = torch.cuda.get_device_name(torch.device(device))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    add_source_argument(parser)
    parser.add_argument("--backend", choices=("torch", "gsplat"), default="gsplat")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    report = {
        "dataset": inspect_dataset(args.source_path),
        "runtime": inspect_runtime(args.backend, args.device),
        "status": "ready",
    }
    print(json.dumps(report, indent=2))
    return report


if __name__ == "__main__":
    main()
