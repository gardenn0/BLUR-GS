import json
import struct

import numpy as np
import pytest
from PIL import Image

from scene.colmap_loader import import_colmap, read_model


@pytest.mark.parametrize("binary", [False, True])
def test_colmap_import_text_binary_and_pixel_conventions(tmp_path, binary):
    model, images = tmp_path / "model", tmp_path / "source"
    model.mkdir()
    images.mkdir()
    Image.new("RGB", (20, 16), (128, 64, 32)).save(images / "a test.png")
    if binary:
        (model / "cameras.bin").write_bytes(
            struct.pack("<QiiQQdddd", 1, 3, 1, 20, 16, 20.0, 24.0, 10.0, 8.0)
        )
        (model / "images.bin").write_bytes(
            struct.pack("<Qidddddddi", 1, 7, 1.0, 0.0, 0.0, 0.0, 0.1, 0.2, 0.3, 3)
            + b"a test.png\0"
            + struct.pack("<Q", 0)
        )
        (model / "points3D.bin").write_bytes(
            struct.pack("<QQdddBBBdQ", 1, 44, 0.0, 0.0, 2.0, 128, 64, 32, 0.1, 0)
        )
    else:
        (model / "cameras.txt").write_text("# cameras\n3 PINHOLE 20 16 20 24 10 8\n")
        (model / "images.txt").write_text("# images\n7 1 0 0 0 .1 .2 .3 3 a test.png\n\n")
        (model / "points3D.txt").write_text("# points\n44 0 0 2 128 64 32 .1\n")
    path = import_colmap(str(model), str(images), str(tmp_path / "out"), downscale=2)
    document = json.loads(path.read_text())
    K = np.array(document["frames"][0]["K"])
    np.testing.assert_allclose(K, [[10, 0, 4.5], [0, 12, 3.5], [0, 0, 1]])
    np.testing.assert_allclose(np.array(document["frames"][0]["w2c"])[:3, 3], [0.1, 0.2, 0.3])
    assert Image.open(path.parent / "images/000000.png").size == (10, 8)
    with np.load(path.parent / "points.npz") as cloud:
        np.testing.assert_allclose(cloud["colors"], [[128 / 255, 64 / 255, 32 / 255]])


def test_distortion_is_not_silently_ignored(tmp_path):
    (tmp_path / "cameras.txt").write_text("1 SIMPLE_RADIAL 20 16 20 10 8 .01\n")
    with pytest.raises(ValueError, match="image_undistorter"):
        read_model(tmp_path)
