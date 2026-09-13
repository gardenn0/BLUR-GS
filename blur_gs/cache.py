"""Portable, pickle-free observed-flow cache, indexed by CoMo camera.image_name."""
import json
from pathlib import Path
import numpy as np
import torch


def iaai_crop(width, height):
    """Match official infer(): integer aspect crop, torchvision center_crop rounding."""
    w, h = width, height
    ratio = 320 / 224
    if w / h > ratio:
        w = int(h * ratio)
    elif w / h < ratio:
        h = int(w / ratio)
    if min(w, h) < 1:
        raise ValueError("Image is too small for IAAI preprocessing")
    return dict(original_width=width, original_height=height,
                left=int(round((width - w) / 2)), top=int(round((height - h) / 2)), width=w, height=h)


class FlowCache:
    def __init__(self, root, camera_names=None):
        self.root = Path(root).resolve()
        self.manifest = json.loads((self.root / "manifest.json").read_text(encoding="utf-8"))
        if self.manifest.get("version") != 1 or self.manifest.get("units") != "network_pixels":
            raise ValueError("Unsupported flow cache format or units")
        self.entries = self.manifest["images"]
        if camera_names is not None:
            names = list(camera_names)
            if len(set(names)) != len(names):
                raise ValueError("Training camera image_name values must be unique")
            missing = set(names) - self.entries.keys()
            if missing:
                raise ValueError(f"Missing flow caches: {sorted(missing)[:10]}")
        self.loaded = {}

    def get(self, name, device):
        if name not in self.loaded:
            entry = self.entries[name]
            path = (self.root / entry["file"]).resolve()
            if not path.is_relative_to(self.root):
                raise ValueError("Cache path escapes the cache directory")
            crop = entry["crop"]
            for key in ("original_width", "original_height", "width", "height"):
                if crop[key] <= 0:
                    raise ValueError(f"Invalid crop for {name}")
            if (crop["left"] < 0 or crop["top"] < 0 or
                crop["left"] + crop["width"] > crop["original_width"] or
                crop["top"] + crop["height"] > crop["original_height"]):
                raise ValueError(f"Crop outside original image: {name}")
            with np.load(path, allow_pickle=False) as data:
                flow = np.asarray(data["flow"], dtype=np.float32).copy()
                mask = np.asarray(data["mask"], dtype=np.float32).copy()
            if flow.ndim != 3 or flow.shape[0] != 2 or mask.shape != flow.shape[1:]:
                raise ValueError(f"Invalid flow/mask shape for {name}")
            if not np.isfinite(flow).all() or not np.isfinite(mask).all() or (mask < 0).any() or (mask > 1).any():
                raise ValueError(f"Nonfinite flow or invalid mask: {name}")
            self.loaded[name] = torch.from_numpy(flow), torch.from_numpy(mask), crop
        flow, mask, crop = self.loaded[name]
        return flow.to(device), mask.to(device), crop
