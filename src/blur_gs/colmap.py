"""Read undistorted COLMAP text/binary models without importing third-party code.

Only SIMPLE_PINHOLE and PINHOLE are accepted. Run COLMAP image_undistorter first
for distorted models; silently dropping distortion would break geometric consistency.
"""

import json
import struct
from pathlib import Path

import numpy as np
from PIL import Image


def _read(stream, fmt):
    fmt = "<" + fmt
    payload = stream.read(struct.calcsize(fmt))
    if len(payload) != struct.calcsize(fmt):
        raise ValueError("Truncated COLMAP binary file")
    return struct.unpack(fmt, payload)


def _rotation(q):
    q = np.asarray(q, dtype=np.float64)
    q /= np.linalg.norm(q)
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def _camera(model, width, height, params):
    if model in {0, "SIMPLE_PINHOLE"}:
        fx, cx, cy = params
        fy = fx
    elif model in {1, "PINHOLE"}:
        fx, fy, cx, cy = params
    else:
        raise ValueError(f"Unsupported camera {model}. Run COLMAP image_undistorter first")
    return width, height, np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1.0]])


def _text_records(path):
    return [
        line.strip()
        for line in path.read_text(encoding="utf-8").splitlines()
        if not line.startswith("#") and line.strip()
    ]


def read_model(model: Path):
    cameras, images, points, colors = {}, [], [], []
    if (model / "cameras.bin").is_file():
        with (model / "cameras.bin").open("rb") as stream:
            for _ in range(_read(stream, "Q")[0]):
                identifier, kind, width, height = _read(stream, "iiQQ")
                if kind not in (0, 1):
                    raise ValueError("Distorted COLMAP camera; run image_undistorter first")
                cameras[identifier] = _camera(
                    kind, width, height, _read(stream, "d" * (3 if kind == 0 else 4))
                )
        with (model / "images.bin").open("rb") as stream:
            for _ in range(_read(stream, "Q")[0]):
                row = _read(stream, "idddddddi")
                filename = bytearray()
                while True:
                    char = stream.read(1)
                    if not char:
                        raise ValueError("Truncated COLMAP filename")
                    if char == b"\0":
                        break
                    filename.extend(char)
                images.append((row[0], row[1:5], row[5:8], row[8], filename.decode("utf-8")))
                stream.seek(24 * _read(stream, "Q")[0], 1)
        with (model / "points3D.bin").open("rb") as stream:
            for _ in range(_read(stream, "Q")[0]):
                row = _read(stream, "QdddBBBd")
                points.append(row[1:4])
                colors.append(row[4:7])
                stream.seek(8 * _read(stream, "Q")[0], 1)
    else:
        for line in _text_records(model / "cameras.txt"):
            row = line.split()
            cameras[int(row[0])] = _camera(
                row[1], int(row[2]), int(row[3]), list(map(float, row[4:]))
            )
        # The second line may be empty (no observations), so do not strip empty lines here.
        lines = iter((model / "images.txt").read_text(encoding="utf-8").splitlines())
        for line in lines:
            if line.startswith("#") or not line.strip():
                continue
            row = line.split(maxsplit=9)
            images.append(
                (
                    int(row[0]),
                    list(map(float, row[1:5])),
                    list(map(float, row[5:8])),
                    int(row[8]),
                    row[9],
                )
            )
            next(lines, None)
        for line in _text_records(model / "points3D.txt"):
            row = line.split()
            points.append(list(map(float, row[1:4])))
            colors.append(list(map(int, row[4:7])))
    if not images or not points:
        raise ValueError("COLMAP model needs registered images and a nonempty sparse point cloud")
    return (
        cameras,
        sorted(images),
        np.asarray(points, np.float32),
        np.asarray(colors, np.float32) / 255,
    )


def import_colmap(
    model_path: str,
    image_path: str,
    output: str,
    downscale: int = 1,
    max_points: int = 10000,
    holdout_every: int = 0,
) -> Path:
    if downscale < 1 or max_points < 1 or holdout_every < 0:
        raise ValueError("downscale/max_points must be positive, holdout_every nonnegative")
    root = Path(output)
    if root.exists() and any(root.iterdir()):
        raise FileExistsError(f"Import output must be empty: {root}")
    cameras, images, points, colors = read_model(Path(model_path))
    (root / "images").mkdir(parents=True, exist_ok=True)
    entries = []
    for index, (image_id, q, translation, camera_id, filename) in enumerate(images):
        width, height, K = cameras[camera_id]
        image = Image.open(Path(image_path) / filename).convert("RGB")
        if image.size != (width, height):
            raise ValueError(f"{filename}: image size differs from COLMAP calibration")
        size = (max(1, width // downscale), max(1, height // downscale))
        image = image.resize(size, Image.Resampling.LANCZOS)
        sx, sy = size[0] / width, size[1] / height
        K = K.copy()
        # COLMAP uses half-integer pixel centers. Convert to our integer-center convention,
        # with half-pixel resampling: u_new+.5 = s*(u_old+.5).
        K[0] *= sx
        K[1] *= sy
        K[0, 2] -= 0.5
        K[1, 2] -= 0.5
        pose = np.eye(4)
        pose[:3, :3], pose[:3, 3] = _rotation(q), translation
        destination = f"images/{index:06d}.png"
        image.save(root / destination)
        entries.append(
            {
                "name": str(image_id),
                "image": destination,
                "original_image": filename,
                "K": K.tolist(),
                "w2c": pose.tolist(),
                "split": "test" if holdout_every and index % holdout_every == 0 else "train",
            }
        )
    if not any(f["split"] == "train" for f in entries):
        raise ValueError("Holdout split leaves no training images")
    if len(points) > max_points:
        indices = np.random.default_rng(0).choice(len(points), max_points, replace=False)
        points, colors = points[indices], colors[indices]
    np.savez_compressed(root / "points.npz", points=points, colors=colors)
    manifest = {
        "version": 1,
        "camera_convention": "opencv_w2c",
        "point_cloud": "points.npz",
        "depth_units": "COLMAP arbitrary scene scale",
        "frames": entries,
    }
    path = root / "scene.json"
    path.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")
    return path
