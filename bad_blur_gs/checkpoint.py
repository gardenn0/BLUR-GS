"""Resume includes partial camera-gradient accumulation and sampler/RNG state."""
from dataclasses import asdict
from pathlib import Path
import random
import numpy as np
import torch


def rng_state():
    return dict(python=random.getstate(), numpy=np.random.get_state(), torch=torch.get_rng_state(),
                cuda=torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None)


def restore_rng(state):
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"].cpu())
    if state["cuda"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all([s.cpu() for s in state["cuda"]])


def capture(step, model, trajectory, camera_optimizer, config, flow, manifest, stack):
    return dict(bad_blur_gs_version=1, step=step, config=asdict(config), flow=asdict(flow),
                data=manifest, gaussians={k: p.detach().clone() for k, p in model.params.items()},
                active_sh_degree=model.active_sh_degree, gaussian_optimizer=model.optimizer.state_dict(),
                refinement={k: getattr(model, k) for k in ("grad_sum", "vis_count", "max_radii")},
                trajectory=trajectory.state_dict(), camera_optimizer=camera_optimizer.state_dict(),
                camera_grad=None if trajectory.controls.grad is None else trajectory.controls.grad.detach().clone(),
                stack=list(stack), rng=rng_state())


def save_atomic(state, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix+".tmp")
    torch.save(state, temporary)
    temporary.replace(path)


def restore(state, model, trajectory, camera_optimizer, config, flow, manifest):
    if state.get("bad_blur_gs_version") != 1:
        raise ValueError("Not a BAD-BLUR-GS checkpoint (CoMo checkpoints are incompatible)")
    if state["config"] != asdict(config):
        raise ValueError("BAD training settings differ from checkpoint")
    old, new = dict(state["flow"]), asdict(flow)
    old.pop("flow_cache", None)
    new.pop("flow_cache", None)
    if old != new or bool(state["flow"]["flow_cache"]) != bool(flow.flow_cache):
        raise ValueError("Flow settings/baseline differ from checkpoint")
    if state["data"] != manifest:
        raise ValueError("Training data, calibration, split or flow cache changed")
    model.load_state_dict({"params."+k: v for k, v in state["gaussians"].items()})
    model.optimizer.load_state_dict(state["gaussian_optimizer"])
    model.active_sh_degree = state["active_sh_degree"]
    for key, value in state["refinement"].items():
        setattr(model, key, value)
    trajectory.load_state_dict(state["trajectory"])
    camera_optimizer.load_state_dict(state["camera_optimizer"])
    trajectory.controls.grad = state["camera_grad"]
    restore_rng(state["rng"])
    return state["step"]+1, list(state["stack"])
