"""COLMAP/LLFF input without importing CoMo or its CUDA extension.

Preserves the user's images and sorted holdout split. Only a recorded similarity
transform is applied to cameras and points; no COLMAP rerun or GT pose fitting.
"""
from pathlib import Path
from dataclasses import replace
import hashlib
import json
import numpy as np
import torch
from PIL import Image
from . import colmap_io
from .trajectory import Camera


def resolve_image(root, name):
    direct = root / name
    if direct.is_file():
        return direct.resolve()
    candidates = [direct.with_suffix(s) for s in (".png", ".jpg", ".jpeg", ".JPG", ".PNG")]
    candidates = list(dict.fromkeys(p for p in candidates if p.is_file()))
    if len(candidates) != 1:
        raise FileNotFoundError(f"Cannot uniquely resolve {direct}: {candidates}")
    return candidates[0].resolve()


def load_scene(source, images, resolution, evaluate, holdout, scene_scale, device="cuda"):
    source = Path(source).resolve()
    sparse = source / "sparse" / "0"
    if resolution not in (1, 2, 4, 8) or holdout < 2:
        raise ValueError("Use -r 1/2/4/8 and llffhold >= 2")
    if (sparse / "images.bin").is_file():
        extr = colmap_io.read_extrinsics_binary(sparse / "images.bin")
        intr = colmap_io.read_intrinsics_binary(sparse / "cameras.bin")
    else:
        extr = colmap_io.read_extrinsics_text(sparse / "images.txt")
        intr = colmap_io.read_intrinsics_text(sparse / "cameras.txt")
    if images:
        image_dir = source / images
    else:
        image_dir = source / "images"
        # Existing CoMo datasets often only contain images_1.
        if not image_dir.is_dir():
            image_dir = source / "images_1"
    cameras = []
    for e in sorted(extr.values(), key=lambda e: e.name):
        c = intr[e.camera_id]
        if c.model == "PINHOLE":
            fx, fy, cx, cy = c.params
        elif c.model == "SIMPLE_PINHOLE":
            fx, cx, cy = c.params
            fy = fx
        elif c.model == "SIMPLE_RADIAL" and abs(c.params[-1]) < 1e-10:
            fx, cx, cy = c.params[:3]
            fy = fx
        else:
            raise ValueError(f"Undistort images first; unsupported non-pinhole camera: {c.model}")
        path = resolve_image(image_dir, e.name)
        with Image.open(path) as im:
            width, height = im.size
        # COLMAP uses image-corner coordinates; our flow grid uses pixel centers.
        sx, sy = width / c.width, height / c.height
        k = torch.tensor([[fx*sx, 0, cx*sx-0.5], [0, fy*sy, cy*sy-0.5], [0, 0, 1]], dtype=torch.float32)
        w2c = torch.eye(4)
        w2c[:3, :3] = torch.from_numpy(colmap_io.qvec2rotmat(e.qvec)).float()
        w2c[:3, 3] = torch.from_numpy(e.tvec).float()
        camera = Camera(Path(e.name).with_suffix("").as_posix(), str(path),
                        torch.linalg.inv(w2c), k, width, height, len(cameras))
        cameras.append(camera.scaled(max(1, round(width/resolution)), max(1, round(height/resolution))))
    names = [c.image_name for c in cameras]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate image stems are not supported by the flow cache")
    train_indices = [i for i in range(len(cameras)) if not evaluate or i % holdout != 0]
    test_indices = [i for i in range(len(cameras)) if evaluate and i % holdout == 0]
    if not train_indices:
        raise ValueError("No training cameras")
    centers = torch.stack([cameras[i].c2w[:3, 3] for i in train_indices])
    center = centers.mean(0)
    extent = (centers-center).abs().max().item()
    if len(train_indices) < 2 or extent < 1e-8:
        raise ValueError("Need at least two distinct training camera centers for scene normalization")
    scale = scene_scale / extent
    for i, c in enumerate(cameras):
        pose = c.c2w.clone()
        pose[:3, 3] = (pose[:3, 3]-center) * scale
        cameras[i] = replace(c, c2w=pose.to(device), K=c.K.to(device))
    if (sparse / "points3D.bin").is_file():
        xyz, rgb, _ = colmap_io.read_points3D_binary(sparse / "points3D.bin")
    elif (sparse / "points3D.txt").is_file():
        xyz, rgb, _ = colmap_io.read_points3D_text(sparse / "points3D.txt")
    else:
        from plyfile import PlyData
        vertices = PlyData.read(sparse / "points3D.ply")["vertex"]
        xyz = np.stack([vertices[n] for n in ("x", "y", "z")], -1)
        rgb = np.stack([vertices[n] for n in ("red", "green", "blue")], -1)
    if len(xyz) < 4 or not np.isfinite(xyz).all():
        raise ValueError("Need at least four finite COLMAP points")
    xyz = (torch.as_tensor(xyz, dtype=torch.float32)-center) * scale
    rgb = torch.as_tensor(rgb, dtype=torch.float32) / 255
    train, test = [cameras[i] for i in train_indices], [cameras[i] for i in test_indices]
    # Hash actual files and calibration, so resuming cannot silently change data.
    from blur_gs.startup import digest_file
    manifest = {"train": [c.image_name for c in train], "test": [c.image_name for c in test],
                "center": center.tolist(), "scale": scale, "llffhold": holdout,
                "images": [{"name": c.image_name, "sha256": digest_file(c.image_path),
                            "K": c.K.cpu().tolist(), "c2w": c.c2w.cpu().tolist(),
                            "width": c.image_width, "height": c.image_height} for c in cameras],
                "points_sha256": hashlib.sha256(xyz.numpy().tobytes()+rgb.numpy().tobytes()).hexdigest()}
    manifest["fingerprint"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return train, test, xyz.to(device), rgb.to(device), manifest


def load_image(camera, device, downscale=1, background=None):
    width, height = max(1, camera.image_width//downscale), max(1, camera.image_height//downscale)
    with Image.open(camera.image_path) as im:
        im = im.convert("RGBA").resize((width, height), Image.Resampling.BILINEAR)
        pixels = torch.from_numpy(np.asarray(im).copy()).float().to(device) / 255
    bg = torch.zeros(3, device=device) if background is None else background
    pixels = pixels[..., :3]*pixels[..., 3:] + bg*(1-pixels[..., 3:])
    return pixels.permute(2, 0, 1), camera.scaled(width, height)
