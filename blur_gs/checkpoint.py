"""Full BLUR-GS state. Load only trusted local training checkpoints."""
import random
from dataclasses import asdict
import numpy as np
import torch


def make_checkpoint(iteration, gaussians, kernel, config, cameras, remaining_cameras):
    return dict(blur_gs_version=1, iteration=iteration, gaussians=gaussians.capture(),
                filter_3D=gaussians.filter_3D, kernel=kernel.state_dict(),
                kernel_optimizer=kernel.optimizer.state_dict(), config=asdict(config),
                cameras=[(c.uid, c.image_name) for c in cameras],
                remaining=[c.uid for c in (remaining_cameras or [])],
                python_rng=random.getstate(), numpy_rng=np.random.get_state(),
                torch_rng=torch.get_rng_state(), cuda_rng=torch.cuda.get_rng_state_all())


def restore_kernel(state, kernel, config, cameras):
    if state.get("blur_gs_version") != 1:
        raise ValueError("Unsupported BLUR-GS checkpoint")
    if state["cameras"] != [(c.uid, c.image_name) for c in cameras]:
        raise ValueError("Camera order/names changed; per-view CoMo parameters cannot be remapped silently")
    previous = dict(state["config"])
    current = asdict(config)
    # Cache can be relocated but all training semantics must match.
    previous.pop("flow_cache")
    current.pop("flow_cache")
    if previous != current:
        raise ValueError("BLUR-GS resume configuration differs from checkpoint")
    kernel.load_state_dict(state["kernel"])
    kernel.optimizer.load_state_dict(state["kernel_optimizer"])


def restore_rng_and_stack(state, cameras):
    random.setstate(state["python_rng"])
    np.random.set_state(state["numpy_rng"])
    torch.set_rng_state(state["torch_rng"].cpu())
    torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda_rng"]])
    lookup = {c.uid: c for c in cameras}
    return [lookup[uid] for uid in state["remaining"]]
