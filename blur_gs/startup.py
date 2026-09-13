"""Prepare frozen observations automatically before allocating the GS scene."""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
from .cache import FlowCache


def add_startup_arguments(parser):
    group = parser.add_argument_group("Automatic flow preparation")
    group.add_argument("--baseline", action="store_true", help="Run original CoMoGaussian without flow")
    group.add_argument("--iaai_python", default=os.environ.get("BLUR_GS_IAAI_PYTHON", sys.executable))
    group.add_argument("--iaai_root", default=os.environ.get("BLUR_GS_IAAI_ROOT", ""))
    group.add_argument("--iaai_checkpoint", default=os.environ.get("BLUR_GS_IAAI_CHECKPOINT", "checkpoints/image_as_imu.pth"))
    group.add_argument("--iaai_device", default="cuda")


def digest_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def prepare_flow(args, config, camera_infos, source_path):
    if args.baseline:
        if config.flow_cache:
            raise ValueError("--baseline cannot be combined with --flow_cache")
        print("[BLUR-GS] Explicit CoMoGaussian baseline")
        return
    images = sorted([dict(name=c.image_name, path=str(Path(c.image_path).resolve()))
                     for c in camera_infos], key=lambda item: item["name"])
    names = [item["name"] for item in images]
    if not names or len(set(names)) != len(names):
        raise ValueError("Training images must have nonempty, unique camera names")
    # Content hashes invalidate automatic caches when image bytes or splits change.
    sources = [dict(name=item["name"], sha256=digest_file(item["path"])) for item in images]
    checkpoint = Path(args.iaai_checkpoint).expanduser().resolve()
    checkpoint_sha = digest_file(checkpoint) if checkpoint.is_file() else None
    signature = dict(images=sources, preprocessing="official-infer-v1-border4")
    key = hashlib.sha256(json.dumps(signature, sort_keys=True).encode()).hexdigest()[:24]
    explicit = bool(config.flow_cache)
    root = (Path(config.flow_cache).expanduser() if explicit else
            Path(source_path) / "flow_cache" / key).resolve()

    def validate():
        cache = FlowCache(root, names)
        if checkpoint_sha and cache.manifest.get("checkpoint_sha256") != checkpoint_sha:
            raise ValueError("Flow cache checkpoint differs from --iaai_checkpoint; choose a new --flow_cache directory")
        for name in names:
            cache.get(name, "cpu")
            cache.loaded.clear()

    if (root / "manifest.json").is_file():
        validate()
        print(f"[BLUR-GS] Reusing flow cache: {root}")
    else:
        if not checkpoint.is_file():
            raise FileNotFoundError(
                "BLUR-GS needs the official Image-as-an-IMU checkpoint on its first run. "
                "Set --iaai_checkpoint or BLUR_GS_IAAI_CHECKPOINT, or place it at "
                "checkpoints/image_as_imu.pth. Use --baseline only for CoMoGaussian.")
        root.mkdir(parents=True, exist_ok=True)
        # Temporary request is outside the cache and disappears on inference failure.
        with tempfile.TemporaryDirectory(prefix="blur-gs-") as temporary:
            request = Path(temporary) / "images.json"
            request.write_text(json.dumps(images), encoding="utf-8")
            command = [args.iaai_python, str(Path(__file__).resolve().parents[1] / "precompute_blur_flow.py"),
                       "--image-list", str(request), "--output", str(root),
                       "--checkpoint", str(checkpoint), "--device", args.iaai_device]
            if args.iaai_root:
                command.extend(["--iaai-root", str(Path(args.iaai_root).expanduser().resolve())])
            print(f"[BLUR-GS] Generating flow for {len(images)} training images using {args.iaai_python}", flush=True)
            # A separate process frees the estimator's GPU memory before GS training.
            subprocess.run(command, check=True)
        validate()
    config.flow_cache = str(root)
    print("[BLUR-GS] Flow ready; starting reconstruction training", flush=True)
