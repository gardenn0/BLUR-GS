"""Reuse BLUR-GS cache, crop transforms, visibility tests and endpoint loss."""
from blur_gs.cache import FlowCache
from blur_gs.loss import blur_flow_loss
from .renderer import render_depth


class FlowSupervisor:
    def __init__(self, config, cameras, depth_fn=render_depth):
        self.config = config
        self.depth_fn = depth_fn
        self.cache = FlowCache(config.flow_cache, [c.image_name for c in cameras]) if config.enabled else None

    def loss(self, step, source, cameras, model, bad_config):
        cfg = self.config
        if self.cache is None or cfg.flow_weight == 0 or step <= cfg.flow_start:
            return model.get_xyz.new_zeros(()), {}
        observed, mask, crop = self.cache.get(source.image_name, model.get_xyz.device)
        if abs(source.image_width/source.image_height-crop["original_width"]/crop["original_height"]) > 0.01:
            raise ValueError(f"Flow cache aspect ratio mismatch: {source.image_name}")
        depths = [self.depth_fn(c, model, bad_config, geometry_grad=cfg.flow_mode == "como",
                               min_alpha=cfg.flow_min_alpha) for c in (cameras[0], cameras[-1])]
        value, stats = blur_flow_loss(cameras, depths, observed, mask, crop,
                                     min_alpha=cfg.flow_min_alpha,
                                     occlusion_tolerance=cfg.flow_occlusion_tolerance,
                                     direction=cfg.flow_direction,
                                     min_valid_fraction=cfg.flow_min_valid_fraction)
        weight = cfg.flow_weight * min(1.0, (step-cfg.flow_start)/max(1, cfg.flow_ramp))
        stats.update(raw_loss=float(value.detach()), weight=weight)
        return value*weight, stats
