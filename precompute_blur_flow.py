"""Run the official frozen Image-as-an-IMU model once per source image.

Use its own environment; training only reads NPZ + JSON and needs no IAAI install.
"""
import argparse
import hashlib
import json
import sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from blur_gs.cache import iaai_crop


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--images", type=Path)
    inputs.add_argument("--image-list", type=Path, help="JSON list of exact camera name/path pairs from train.py")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--iaai-root", type=Path, help="Path to official image-as-an-imu checkout")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--name-mode", choices=("colmap", "stem"), default="colmap",
                        help="Match CoMo dataset loader: colmap splits at first dot; stem for Blender")
    parser.add_argument("--border", type=int, default=4, help="Mask this many network-border pixels")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()
    if not 0 <= args.border < 112:
        parser.error("--border must be between 0 and 111")
    if args.iaai_root:
        sys.path.insert(0, str(args.iaai_root.resolve()))
    from iaai.model import Blur2PoseSegNeXtBackbone

    if args.image_list:
        records = json.loads(args.image_list.read_text(encoding="utf-8"))
        paths = [Path(item["path"]) for item in records]
        names = [item["name"] for item in records]
    else:
        paths = sorted(p for p in args.images.rglob("*") if p.suffix.lower() in (".png", ".jpg", ".jpeg"))
        names = [p.name.split(".")[0] if args.name_mode == "colmap" else p.stem for p in paths]
    if not paths:
        parser.error("No images found")
    if len(set(names)) != len(names):
        parser.error("Duplicate camera names; supply only the actual source image directory")
    if (args.output / "manifest.json").exists() and not args.overwrite:
        parser.error("Cache already exists; use a fresh directory or --overwrite")
    args.output.mkdir(parents=True, exist_ok=True)
    model = Blur2PoseSegNeXtBackbone(supervise_pose=False, device=args.device)
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=True)
    if "state_dict" in state:
        state = state["state_dict"]
    model.load_state_dict(state, strict=True)
    model.to(args.device).eval().requires_grad_(False)
    digest = hashlib.sha256()
    with args.checkpoint.open("rb") as checkpoint_file:
        for block in iter(lambda: checkpoint_file.read(1024 * 1024), b""):
            digest.update(block)
    manifest = dict(version=1, units="network_pixels", estimator="image-as-an-imu",
                    checkpoint_sha256=digest.hexdigest(), name_mode=args.name_mode,
                    direction="ambiguous", images={})
    with torch.inference_mode():
        for index, (path, name) in enumerate(zip(paths, names)):
            with Image.open(path) as original:
                img = original.convert("RGB")
            crop = iaai_crop(*img.size)
            # Flow-only forward has no dependency on intrinsics; official infer
            # still requires fx/fy fields. Pose/depth outputs are not supervision.
            result = model.infer({"image": img, "fx": 1., "fy": 1.})
            flow = result["flow_field"].squeeze(0).float().cpu().numpy()
            if flow.shape != (2, 224, 320) or not np.isfinite(flow).all():
                raise ValueError(f"Unexpected IAAI flow output for {path}: {flow.shape}")
            mask = np.ones((224, 320), dtype=np.float32)
            if args.border:
                b = args.border
                mask[:b] = mask[-b:] = 0
                mask[:, :b] = mask[:, -b:] = 0
            # File hash prevents path traversal and filesystem-sensitive image names.
            filename = hashlib.sha256(name.encode()).hexdigest() + ".npz"
            np.savez_compressed(args.output / filename, flow=flow, mask=mask)
            manifest["images"][name] = dict(file=filename, crop=crop,
                                           source=str(path) if args.image_list else path.relative_to(args.images).as_posix())
            print(f"[{index + 1}/{len(paths)}] {name}")
    temporary = args.output / "manifest.json.tmp"
    temporary.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    temporary.replace(args.output / "manifest.json")
    print(f"Saved {len(paths)} frozen flow observations to {args.output}")


if __name__ == "__main__":
    main()
