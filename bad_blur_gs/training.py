"""Standalone train.py-style BAD loop. RGB and flow share one backward pass."""
from dataclasses import asdict, replace
from pathlib import Path
from argparse import ArgumentParser
import json
import math
import random
import sys
import time
import numpy as np
import torch
from .config import BadConfig, add_arguments, from_args, exponential_lr
from .trajectory import ExposureTrajectory
from .gaussians import Gaussians
from .flow import FlowSupervisor
from .data import load_scene, load_image
from .renderer import render, require_backend
from .checkpoint import capture, restore, save_atomic, rng_state, restore_rng
from blur_gs.training import BlurConfig
from blur_gs.startup import prepare_flow, digest_file


def make_parser():
    parser = ArgumentParser(description="BAD-BLUR-GS: standalone 3DGS + BAD trajectory + optional frozen IAAI")
    parser.add_argument("-s", "--source_path", required=True)
    parser.add_argument("-m", "--model_path", required=True)
    parser.add_argument("-i", "--images", default="")
    parser.add_argument("-r", "--resolution", type=int, default=1)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--llffhold", type=int, default=8)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--start_checkpoint", default="")
    parser.add_argument("--baseline", action="store_true", help="BAD-style baseline; no IAAI/cache/flow loss")
    parser.add_argument("--flow_mode", choices=("como", "trajectory"), default="como",
                        help="como: joint geometry/trajectory flow, trajectory: detach auxiliary depth only")
    for name in ("flow_cache", "flow_weight", "flow_start", "flow_ramp", "flow_direction",
                 "flow_min_alpha", "flow_occlusion_tolerance", "flow_min_valid_fraction"):
        default = getattr(BlurConfig(), name)
        options = dict(default=default, type=type(default))
        if name == "flow_direction":
            options["choices"] = ("auto", "forward", "backward")
        parser.add_argument("--"+name, **options)
    import os
    parser.add_argument("--iaai_checkpoint", default=os.environ.get("BLUR_GS_IAAI_CHECKPOINT", "checkpoints/image_as_imu.pth"))
    parser.add_argument("--iaai_python", default=os.environ.get("BLUR_GS_IAAI_PYTHON", sys.executable))
    parser.add_argument("--iaai_root", default=os.environ.get("BLUR_GS_IAAI_ROOT", ""))
    parser.add_argument("--iaai_device", default="cuda")
    add_arguments(parser)
    return parser


def flow_config(args):
    cfg = BlurConfig()
    for key in asdict(cfg):
        if hasattr(args, key):
            setattr(cfg, key, getattr(args, key))
    if args.baseline:
        if cfg.flow_cache:
            raise ValueError("--baseline cannot be combined with --flow_cache")
        cfg.flow_weight = 0.0
    for key in ("flow_weight", "flow_start", "flow_ramp", "flow_occlusion_tolerance"):
        value = getattr(cfg, key)
        if not math.isfinite(value) or value < 0:
            raise ValueError(f"Invalid {key}")
    if not 0 < cfg.flow_min_alpha <= 1 or not 0 < cfg.flow_min_valid_fraction <= 1:
        raise ValueError("Invalid flow coverage thresholds")
    return cfg


def seed_all(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def background_for(config, device, training=True):
    if config.background == "random":
        return torch.rand(3, device=device) if training else torch.tensor([0.1490, 0.1647, 0.2157], device=device)
    return torch.full((3,), float(config.background == "white"), device=device)


def training_step(step, camera, target, model, trajectory, camera_optimizer, config, flow,
                  background, render_fn=render):
    from pytorch_msssim import ssim
    model.train()
    model.schedule(step, config)
    camera_optimizer.param_groups[0]["lr"] = exponential_lr(
        config.camera_lr, config.camera_lr_final, max(step-1, 0), config.lr_max_steps)
    model.optimizer.zero_grad(set_to_none=True)
    if step % config.camera_accumulation == 0:
        camera_optimizer.zero_grad(set_to_none=True)
    cameras = trajectory.cameras(camera)
    images = []
    last_info = None
    for virtual in cameras:
        output = render_fn(virtual, model, config, background, retain_stats=True)
        images.append(output["render"])
        last_info = output["info"]
    image = torch.stack(images).mean(0)
    l1 = (image-target).abs().mean()
    rgb = (1-config.ssim_lambda)*l1 + config.ssim_lambda*(1-ssim(image[None], target[None], data_range=1))
    scale = model.scale_loss(step, config)
    flow_loss, stats = flow.loss(step, camera, cameras, model, config)
    total = rgb + scale + flow_loss
    if not torch.isfinite(total):
        raise FloatingPointError(f"Nonfinite loss at step {step}")
    total.backward()
    # A missing pose gradient must fail loudly rather than silently training only GS.
    if trajectory.controls.grad is None:
        raise RuntimeError("Renderer did not return camera-pose gradients")
    for p in list(model.parameters()) + [trajectory.controls]:
        if p.grad is not None and not torch.isfinite(p.grad).all():
            raise FloatingPointError(f"Nonfinite gradient at step {step}")
    model.optimizer.step()
    # Upstream sums (does not average) gradients and does not flush partial blocks.
    if step % config.camera_accumulation == config.camera_accumulation-1:
        camera_optimizer.step()
    model.collect_stats(last_info, config, step)
    model.refine(step, config, len(trajectory.names), camera.image_width, camera.image_height)
    return dict(total=float(total.detach()), rgb=float(rgb.detach()), l1=float(l1.detach()),
                scale=float(scale.detach()), flow=float(flow_loss.detach()), **stats)


def save_image(path, image):
    from PIL import Image
    array = (image.detach().clamp(0, 1).permute(1, 2, 0).cpu().numpy()*255).astype(np.uint8)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


@torch.no_grad()
def evaluate(model, cameras, config, lpips_model, output=None, writer=None, step=0):
    from skimage.metrics import structural_similarity
    from pytorch_msssim import ssim
    if not cameras:
        return {}
    background = background_for(config, model.get_xyz.device, False)
    rows = {}
    for i, camera in enumerate(cameras):
        gt, view = load_image(camera, model.get_xyz.device, background=background)
        image = render(view, model, config, background)["render"].clamp(0, 1)
        mse = (image-gt).square().mean()
        # Explicit range=2 preserves the earlier CoMo skimage metric convention.
        im_np = (image*2-1).permute(1, 2, 0).cpu().numpy()
        gt_np = (gt*2-1).permute(1, 2, 0).cpu().numpy()
        rows[camera.image_name] = dict(PSNR=float(-10*torch.log10(mse.clamp_min(1e-12))),
            SSIM=float(structural_similarity(im_np, gt_np, channel_axis=-1, data_range=2)),
            SSIM_BAD=float(ssim(image[None], gt[None], data_range=1)),
            LPIPS=float(lpips_model(image[None]*2-1, gt[None]*2-1).mean()))
        if output:
            name = camera.image_name.replace("/", "_")
            save_image(Path(output)/"renders"/(name+".png"), image)
            save_image(Path(output)/"gt"/(name+".png"), gt)
            save_image(Path(output)/"error"/(name+".png"), (image-gt).abs())
        if writer and i < 5:
            writer.add_image(f"Test_view_{camera.image_name}/render", image, step)
            writer.add_image(f"Test_view_{camera.image_name}/gt", gt, step)
    averages = {k: sum(r[k] for r in rows.values())/len(rows) for k in next(iter(rows.values()))}
    if output:
        Path(output).mkdir(parents=True, exist_ok=True)
        (Path(output)/"metrics.json").write_text(json.dumps(dict(mean=averages, per_view=rows), indent=2))
    if writer:
        for key, value in averages.items():
            writer.add_scalar("Test/loss_viewpoint - "+key.lower(), value, step)
    return averages


def main(argv=None):
    args = make_parser().parse_args(argv)
    args.source_path = str(Path(args.source_path).expanduser().resolve())
    config, flow_cfg = from_args(args), flow_config(args)
    if not torch.cuda.is_available():
        raise RuntimeError("Training requires CUDA. CPU mathematical tests: python -m pytest tests/test_bad*.py")
    require_backend()
    output = Path(args.model_path).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if (output/"cfg_args").exists():
        raise FileExistsError("Output contains a CoMo/3DGS run; use a separate -m for BAD-BLUR-GS")
    if (output/"bad_config.json").exists() and not args.start_checkpoint:
        raise FileExistsError("Output already contains a BAD run; use a new -m or --start_checkpoint")
    # Read camera metadata on CPU; prepare frozen IAAI observations before allocating GS.
    train, test, xyz, colors, manifest = load_scene(args.source_path, args.images, args.resolution,
        args.eval, args.llffhold, config.scene_scale, device="cpu")
    print(f"[BAD-BLUR-GS] {len(train)} train, {len(test)} test; holdout={args.llffhold}", flush=True)
    if not args.baseline and flow_cfg.flow_weight > 0:
        prepare_flow(args, flow_cfg, train, args.source_path)
    else:
        flow_cfg.flow_cache = ""
        print("[BAD-BLUR-GS] No-flow BAD baseline; IAAI is not loaded", flush=True)
    flow_digest = None
    if flow_cfg.enabled:
        from blur_gs.cache import FlowCache
        import hashlib
        cache = FlowCache(flow_cfg.flow_cache, [c.image_name for c in train])
        flow_hashes = {c.image_name: digest_file(cache.root/cache.entries[c.image_name]["file"]) for c in train}
        flow_digest = hashlib.sha256(json.dumps(dict(images=flow_hashes,
            crop={c.image_name: cache.entries[c.image_name]["crop"] for c in train}), sort_keys=True).encode()).hexdigest()
    manifest["flow_sha256"] = flow_digest
    seed_all(args.seed)
    train = [replace(c, c2w=c.c2w.cuda(), K=c.K.cuda()) for c in train]
    test = [replace(c, c2w=c.c2w.cuda(), K=c.K.cuda()) for c in test]
    state = torch.load(args.start_checkpoint, map_location="cuda", weights_only=False) if args.start_checkpoint else None
    if state and state.get("bad_blur_gs_version") != 1:
        raise ValueError("Use a BAD-BLUR-GS checkpoint, not a CoMo/Gaussian-only checkpoint")
    model = Gaussians(state["gaussians"], config) if state else Gaussians.from_points(xyz.cuda(), colors.cuda(), config)
    trajectory = ExposureTrajectory(train, config, "cuda")
    camera_optimizer = torch.optim.Adam([trajectory.controls], lr=config.camera_lr, eps=1e-15)
    supervisor = FlowSupervisor(flow_cfg, train)
    # Evaluation-network initialization must not perturb reconstruction randomness.
    saved_rng = rng_state()
    import lpips
    lpips_model = lpips.LPIPS(net="alex").eval().cuda()
    restore_rng(saved_rng)
    from torch.utils.tensorboard import SummaryWriter
    first, stack = restore(state, model, trajectory, camera_optimizer, config, flow_cfg, manifest) if state else (0, [])
    writer = SummaryWriter(str(output), purge_step=first if state else None)
    metadata = dict(model="bad_blur_gs", config=asdict(config), flow=asdict(flow_cfg), args=vars(args), data=manifest)
    import subprocess
    import gsplat
    import pypose
    runtime = dict(torch=torch.__version__, gsplat=gsplat.__version__, pypose=pypose.__version__)
    try:
        repo = Path(__file__).resolve().parents[1]
        runtime["git_commit"] = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=repo, text=True).strip()
        runtime["tracked_changes"] = bool(subprocess.check_output(["git", "diff", "HEAD", "--name-only"], cwd=repo, text=True).strip())
    except (OSError, subprocess.SubprocessError):
        runtime["git_commit"] = None
    metadata["runtime"] = runtime
    (output/"bad_config.json").write_text(json.dumps(metadata, indent=2))
    (output/"blur_gs_config.json").write_text(json.dumps(asdict(flow_cfg), indent=2))
    print(f"[BAD-BLUR-GS] {len(model.get_xyz)} Gaussians; steps {first}..{config.iterations-1}", flush=True)
    from tqdm import tqdm
    started = time.perf_counter()
    try:
        progress = tqdm(range(first, config.iterations), desc="BAD-BLUR-GS")
        for step in progress:
            if not stack:
                stack = list(range(len(train)))
            camera = train[stack.pop(random.randrange(len(stack)))]
            background = background_for(config, "cuda")
            downscale = 2**max(config.num_downscales-step//config.resolution_schedule, 0)
            image, camera = load_image(camera, "cuda", downscale, background)
            stats = training_step(step, camera, image, model, trajectory, camera_optimizer,
                                  config, supervisor, background)
            if step % 10 == 0:
                progress.set_postfix(loss=f"{stats['total']:.5f}", flow=f"{stats['flow']:.5f}",
                                     points=len(model.get_xyz))
                for key, value in stats.items():
                    writer.add_scalar("bad_blur_gs/"+key, value, step)
                writer.add_scalar("total_points", len(model.get_xyz), step)
            if step % 100 == 0 and "valid_fraction" in stats:
                tqdm.write(f"[BAD flow {step}] mode={flow_cfg.flow_mode} "
                           f"raw={stats['raw_loss']:.4f} weight={stats['weight']:.5f} "
                           f"valid={stats['valid_fraction']:.3f} skipped={stats['skipped']}")
            final = step == config.iterations-1
            if (step > 0 and config.eval_every and step % config.eval_every == 0) or final:
                metrics = evaluate(model, test, config, lpips_model, output/"test"/f"iteration_{step}", writer, step)
                print(f"[BAD {step}] {metrics}", flush=True)
            if (step > 0 and config.save_every and step % config.save_every == 0) or final:
                save_atomic(capture(step, model, trajectory, camera_optimizer, config, flow_cfg, manifest, stack),
                            output/f"bad_chkpnt{step}.pth")
                model.save_ply(output/"point_cloud"/f"iteration_{step}"/"point_cloud.ply")
        print(f"Reconstruction loop finished in {(time.perf_counter()-started)/60:.1f} min; artifacts: {output}", flush=True)
    finally:
        writer.close()
