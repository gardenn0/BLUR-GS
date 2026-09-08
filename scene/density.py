"""Exposure-aggregated screen-gradient refinement with Adam state preservation."""

import torch
from torch import nn

from utils.pose_utils import quaternion_matrix


class DensityController:
    def __init__(self, scene, extent):
        self.extent = float(extent)
        self.gradient_sum = scene.means.new_zeros(len(scene.means))
        self.count = torch.zeros_like(self.gradient_sum)

    @torch.no_grad()
    def accumulate(self, results, height, width):
        # Summing norms avoids cancellation between exposure views. Convert pixel
        # gradients to normalized-screen gradients; do not include the depth pass.
        for result in results:
            if result.means2d is None or result.means2d.grad is None:
                continue
            grad = result.means2d.grad.reshape(-1, 2).clone()
            grad *= grad.new_tensor([width / 2, height / 2]) * len(results)
            visible = result.visible.reshape(-1)
            indices = result.point_order
            if indices is None:
                indices = torch.arange(len(grad), device=grad.device)
            self.gradient_sum[indices] += grad.norm(dim=-1) * visible
            self.count[indices] += visible

    @staticmethod
    @torch.no_grad()
    def replace(scene, optimizer, indices, values, fresh):
        for name, old in list(scene.named_parameters()):
            new = nn.Parameter(values.get(name, old.detach()[indices]).clone())
            state = optimizer.state.pop(old, {})
            for key, value in list(state.items()):
                if torch.is_tensor(value) and value.shape == old.shape:
                    state[key] = value[indices].clone()
                    state[key][fresh] = 0
            for group in optimizer.param_groups:
                group["params"] = [new if p is old else p for p in group["params"]]
            setattr(scene, name, new)
            if state:
                optimizer.state[new] = state

    @torch.no_grad()
    def refine(self, scene, optimizer, config):
        score = self.gradient_sum / self.count.clamp_min(1)
        keep = scene.opacities >= config.prune_opacity
        # Never delete the entire representation.
        if not keep.any():
            keep[scene.opacities.argmax()] = True
        candidates = torch.where(
            keep & (self.count > 0) & (score >= config.densify_grad_threshold)
        )[0]
        budget = max(0, config.max_gaussians - int(keep.sum()))
        candidates = candidates[score[candidates].argsort(descending=True)[:budget]]
        large = scene.scales[candidates].amax(-1) > 0.01 * self.extent
        split, clone = candidates[large], candidates[~large]
        keep[split] = False
        retained = torch.where(keep)[0]
        indices = torch.cat([retained, clone, split, split])
        fresh = torch.arange(len(indices), device=indices.device) >= len(retained)
        values = {}
        if len(split):
            means = scene.means.detach()[indices].clone()
            scales = scene.log_scales.detach()[indices].clone()
            axis = scene.scales[split].argmax(-1)
            rotation = quaternion_matrix(scene.quats[split])
            offset = rotation[torch.arange(len(split), device=axis.device), :, axis]
            offset *= scene.scales[split].gather(1, axis[:, None]) * 0.5
            start = len(retained) + len(clone)
            means[start : start + len(split)] += offset
            means[start + len(split) :] -= offset
            scales[start:] -= torch.log(scales.new_tensor(1.6))
            values.update(means=means, log_scales=scales)
        self.replace(scene, optimizer, indices, values, fresh)
        self.gradient_sum = scene.means.new_zeros(len(indices))
        self.count = torch.zeros_like(self.gradient_sum)

    @torch.no_grad()
    def reset_opacity(self, scene, optimizer):
        scene.opacity_logits.copy_(torch.logit(scene.opacities.clamp_max(0.01)))
        for key in ("exp_avg", "exp_avg_sq", "max_exp_avg_sq"):
            value = optimizer.state.get(scene.opacity_logits, {}).get(key)
            if value is not None:
                value.zero_()

    def state_dict(self):
        return dict(extent=self.extent, gradient_sum=self.gradient_sum, count=self.count)

    def load_state_dict(self, state):
        self.extent = state["extent"]
        self.gradient_sum = state["gradient_sum"].clone()
        self.count = state["count"].clone()
