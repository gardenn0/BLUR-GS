import importlib.util

import pytest
import torch

from utils.pose_utils import se3_exp
from gaussian_renderer import GsplatRenderer, TorchRenderer
from gaussian_renderer.blur_renderer import render_blur
from scene.gaussian_model import GaussianScene
from scene.density import DensityController


def scene_and_camera(device="cpu"):
    scene = GaussianScene(
        torch.tensor([[-0.1, 0.0, 2.0], [0.1, 0.1, 3.0]], device=device),
        torch.tensor([[0.9, 0.1, 0.2], [0.1, 0.8, 0.3]], device=device),
        torch.full((2, 3), 0.12, device=device),
    )
    K = torch.tensor([[20.0, 0, 7.0], [0, 20.0, 7.0], [0, 0, 1.0]], device=device)
    return scene, K


def test_alpha_normalized_depth_and_pose_gradients():
    scene, K = scene_and_camera()
    twist = torch.zeros(6, requires_grad=True)
    result = TorchRenderer()(scene, se3_exp(twist), K, 15, 15)
    assert (result.alpha >= 0).all() and (result.alpha <= 1).all()
    valid = result.alpha > 0.01
    assert (result.depth[valid] >= 2 - 1e-5).all() and (result.depth[valid] <= 3 + 1e-5).all()
    (result.rgb[5:8].sum() + 0.1 * result.depth[valid].mean()).backward()
    assert torch.isfinite(twist.grad).all() and twist.grad.abs().sum() > 0
    assert scene.means.grad.abs().sum() > 0


def test_static_camera_blur_equals_sharp_render():
    scene, K = scene_and_camera()
    pose = torch.eye(4)
    renderer = TorchRenderer()
    blur = render_blur(renderer, scene, pose[None].expand(5, -1, -1), K, 15, 15)
    torch.testing.assert_close(blur, renderer(scene, pose, K, 15, 15).rgb)


@pytest.mark.skipif(
    not torch.cuda.is_available() or importlib.util.find_spec("gsplat") is None,
    reason="CUDA + gsplat required",
)
def test_cuda_renderer_pose_and_depth_gradients():
    scene, K = scene_and_camera("cuda")
    twist = torch.zeros(6, device="cuda", requires_grad=True)
    result = GsplatRenderer()(scene, se3_exp(twist), K, 15, 15)
    loss = result.rgb[5:8].sum() + 0.1 * result.depth[result.alpha > 0.01].mean()
    loss.backward()
    assert torch.isfinite(twist.grad).all() and twist.grad.abs().sum() > 0
    assert torch.isfinite(scene.means.grad).all() and scene.means.grad.abs().sum() > 0
    controller = DensityController(scene, 1.0)
    controller.accumulate([result], 15, 15)
    assert controller.count.sum() > 0
    assert torch.isfinite(controller.gradient_sum).all()
    assert controller.gradient_sum.sum() > 0
