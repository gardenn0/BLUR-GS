"""Deterministic analytic fixture for integration tests, not an IAAI prediction or benchmark."""

import json
from pathlib import Path

import numpy as np
import torch

from .data import save_image
from .geometry import exposure_path
from .rendering import TorchRenderer, render_blur
from .scene import GaussianScene
from .trajectory import ExposureTrajectory


@torch.no_grad()
def make_synthetic(output: str | Path, size: int = 32, views: int = 3, seed: int = 7) -> Path:
    root = Path(output)
    if size < 12 or views < 1:
        raise ValueError("Synthetic fixture needs size>=12 and views>=1")
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Synthetic output must be empty: {root}")
    for name in ("images", "sharp", "motion", "depth"):
        (root / name).mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    yy, xx = torch.meshgrid(
        torch.linspace(-0.8, 0.8, 5), torch.linspace(-0.8, 0.8, 5), indexing="ij"
    )
    points = torch.stack((xx.flatten(), yy.flatten(), 2.5 + 0.25 * torch.rand(25)), -1)
    colors = 0.1 + 0.8 * torch.rand_like(points)
    scales = torch.full_like(points, 0.16)
    truth = GaussianScene(points, colors, scales)
    truth.opacity_logits.fill_(2.0)
    K = torch.tensor(
        [[size * 1.15, 0, (size - 1) / 2], [0, size * 1.1, (size - 1) / 2], [0, 0, 1.0]]
    )
    renderer, entries = TorchRenderer(), []
    for i in range(views):
        pose = torch.eye(4)
        pose[0, 3] = (i - (views - 1) / 2) * 0.08
        trajectory = ExposureTrajectory(pose)
        trajectory.initialize_from_camera_motion(
            torch.tensor([0.05, -0.02, 0.005, 0.008, 0.018, -0.004])
        )
        # Nonzero curvature exercises the continuous trajectory and acceleration terms.
        trajectory.controls[1, 1] += 0.007
        trajectory.controls[2, 1] -= 0.007
        times = torch.linspace(0, 1, 9)
        ref = renderer(truth, trajectory.at(0.5), K, size, size)
        path, valid = exposure_path(ref.depth, K, trajectory.at(0.5), trajectory(times))
        blur = render_blur(renderer, truth, trajectory(times), K, size, size)
        name = f"{i:03d}"
        save_image(root / "images" / f"{name}.png", blur)
        save_image(root / "sharp" / f"{name}.png", ref.rgb)
        np.save(root / "depth" / f"{name}.npy", ref.depth.numpy())
        np.savez_compressed(
            root / "motion" / f"{name}.npz",
            flow=(path[-1] - path[0]).numpy(),
            confidence=((ref.alpha > 0.1) & valid.all(0)).float().numpy(),
            reference="midpoint",
            sign_ambiguous=True,
            source="analytic-synthetic-ground-truth-NOT-Image-as-an-IMU",
        )
        entries.append(
            {
                "name": name,
                "image": f"images/{name}.png",
                "sharp_image": f"sharp/{name}.png",
                "K": K.tolist(),
                "w2c": pose.tolist(),
                "motion": f"motion/{name}.npz",
                "initial_depth": f"depth/{name}.npy",
                "exposure_seconds": 0.02,
                "split": "train",
            }
        )
    np.savez(
        root / "points.npz",
        points=(points + 0.015 * torch.randn_like(points)).numpy(),
        colors=(colors + 0.03 * torch.randn_like(colors)).clamp(0, 1).numpy(),
        scales=scales.numpy(),
    )
    manifest = {
        "version": 1,
        "camera_convention": "opencv_w2c",
        "point_cloud": "points.npz",
        "source": "analytic test fixture; not a real-scene benchmark",
        "frames": entries,
    }
    path = root / "scene.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path
