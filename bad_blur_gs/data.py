"""COLMAP/LLFF input without importing CoMo or its CUDA extension.

Uses BAD's default DeblurNerf/Colmap all-pose normalization before holdout.
Input must be undistorted. No COLMAP rerun or GT pose fitting.
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
from .coordinates import normalize_colmap


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
    hold_files = sorted(source.glob("hold=*"))
    if len(hold_files) > 1:
        raise ValueError("Ambiguous hold= files")
    if hold_files and evaluate:
        holdout = int(hold_files[0].name.split("=")[-1])
        if holdout < 1:
            raise ValueError("This entrypoint requires an interval holdout; remove hold<1 or use --eval off")
    if (source/"images_test").is_dir() or any((source/f"{s}_list.txt").exists() for s in ("train", "test", "val", "validation")):
        raise ValueError("This port supports BAD's interval-split COLMAP input; images_test/list-based restoration protocols require the official parser")
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
        original = resolve_image(image_dir, e.name)
        scaled_dir = image_dir.with_name(image_dir.name + f"_{resolution}")
        if resolution > 1 and not scaled_dir.is_dir():
            raise FileNotFoundError(f"Use the same precomputed BAD images: missing {scaled_dir}; use -r 1 for original images")
        path = resolve_image(scaled_dir, e.name) if resolution > 1 else original
        with Image.open(path) as im:
            width, height = im.size
        # Do not silently change the pixel data with a different resize kernel.
        # COLMAP uses image-corner coordinates; our flow grid uses pixel centers.
        k = torch.tensor([[fx/resolution, 0, cx/resolution-0.5], [0, fy/resolution, cy/resolution-0.5], [0, 0, 1]], dtype=torch.float32)
        w2c = np.eye(4)
        w2c[:3, :3] = colmap_io.qvec2rotmat(e.qvec)
        w2c[:3, 3] = e.tvec
        camera = Camera(Path(e.name).with_suffix("").as_posix(), str(path),
                        torch.from_numpy(np.linalg.inv(w2c).astype(np.float32)), k,
                        int(c.width//resolution), int(c.height//resolution), len(cameras))
        cameras.append(camera)
        # Store actual target size for the upstream first-image calibration check.
        camera._actual_size = (width, height)
    names = [c.image_name for c in cameras]
    if len(names) != len(set(names)):
        raise ValueError("Duplicate image stems are not supported by the flow cache")
    train_indices = [i for i in range(len(cameras)) if not evaluate or i % holdout != 0]
    test_indices = [i for i in range(len(cameras)) if evaluate and i % holdout == 0]
    if not train_indices:
        raise ValueError("No training cameras")
    # BAD's _check_outputs corrects each split using its first image.
    for indices in (train_indices, test_indices):
        if not indices:
            continue
        first = cameras[indices[0]]
        for axis, size in enumerate(first._actual_size):
            principal = first.K[axis, 2] + .5
            ideal = torch.tensor(size/2)
            if not torch.allclose(principal, ideal, rtol=.3):
                ratio = principal/ideal
                factor = round(1/ratio.item()) if ratio < 1 else 1/round(ratio.item())
                for i in indices:
                    c = cameras[i]
                    c.K[axis, axis] *= factor
                    c.K[axis, 2] = (c.K[axis, 2]+.5)*factor-.5
                    if axis == 0:
                        c.image_width = int(c.image_width*factor)
                    else:
                        c.image_height = int(c.image_height*factor)
        for i in indices:
            c = cameras[i]
            if (c.image_width, c.image_height) != c._actual_size:
                raise ValueError(f"BAD calibration correction does not match image size for {c.image_name}; fix input calibration")
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
    poses, xyz, transform, scale = normalize_colmap(
        torch.stack([c.c2w for c in cameras]), torch.as_tensor(xyz, dtype=torch.float32), scene_scale)
    cameras = [replace(c, c2w=poses[i].to(device), K=c.K.to(device)) for i, c in enumerate(cameras)]
    rgb = torch.as_tensor(rgb, dtype=torch.float32) / 255
    train, test = [cameras[i] for i in train_indices], [cameras[i] for i in test_indices]
    # Hash actual files and calibration, so resuming cannot silently change data.
    from blur_gs.startup import digest_file
    manifest = {"train": [c.image_name for c in train], "test": [c.image_name for c in test],
                "coordinate_version": 2, "transform": transform.tolist(), "scale": scale, "llffhold": holdout,
                "images": [{"name": c.image_name, "sha256": digest_file(c.image_path),
                            "K": c.K.cpu().tolist(), "c2w": c.c2w.cpu().tolist(),
                            "width": c.image_width, "height": c.image_height} for c in cameras],
                "points_sha256": hashlib.sha256(xyz.numpy().tobytes()+rgb.numpy().tobytes()).hexdigest()}
    manifest["fingerprint"] = hashlib.sha256(json.dumps(manifest, sort_keys=True).encode()).hexdigest()
    return train, test, xyz.to(device), rgb.to(device), manifest


def load_image(camera, device, downscale=1, background=None):
    width, height = max(1, camera.image_width//downscale), max(1, camera.image_height//downscale)
    with Image.open(camera.image_path) as im:
        pixels = np.asarray(im.convert("RGBA")).copy()
        pixels = torch.from_numpy(pixels).float().to(device) / 255
    if pixels.shape[:2] != (camera.image_height, camera.image_width):
        raise ValueError(f"Image dimensions changed after calibration: {camera.image_name}")
    if downscale > 1:
        # Equivalent to NS v1.0.3 TF.resize(..., antialias=None) on a tensor.
        pixels = torch.nn.functional.interpolate(pixels.permute(2, 0, 1)[None],
            size=(height, width), mode="bilinear", align_corners=False,
            antialias=False)[0].permute(1, 2, 0)
    bg = torch.zeros(3, device=device) if background is None else background
    pixels = pixels[..., :3]*pixels[..., 3:] + bg*(1-pixels[..., 3:])
    return pixels.permute(2, 0, 1), camera.scaled(width, height, factor=1/downscale)
