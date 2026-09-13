"""Run on the actual compiled CoMo renderer before claiming GPU validation."""
from types import SimpleNamespace
import math
import pytest
import torch


@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA unavailable")
def test_real_renderer_depth_gradient():
    # Do not import-or-skip: on a GPU validation host, missing extensions must fail.
    from gaussian_renderer import render
    from utils.graphics_utils import getProjectionMatrix
    from blur_gs.depth import render_z_depth
    xyz = torch.tensor([[0., 0., 2.], [.1, 0., 2.2]], device="cuda", requires_grad=True)
    pc = SimpleNamespace(get_xyz=xyz, get_opacity_with_3D_filter=torch.full((2, 1), .9, device="cuda"),
                         get_scaling_with_3D_filter=torch.full((2, 3), .2, device="cuda"),
                         get_rotation=torch.tensor([[1., 0, 0, 0]] * 2, device="cuda"), active_sh_degree=0)
    proj = getProjectionMatrix(.01, 100., math.pi / 2, math.pi / 2).T.cuda()
    cam = SimpleNamespace(image_width=32, image_height=32, FoVx=math.pi / 2, FoVy=math.pi / 2,
                          world_view_transform=torch.eye(4, device="cuda"), projection_matrix=proj,
                          full_proj_transform=proj, camera_center=torch.zeros(3, device="cuda"))
    pipe = SimpleNamespace(compute_cov3D_python=False, convert_SHs_python=False, debug=False)
    def value():
        z, alpha, _ = render_z_depth(cam, pc, pipe, .1, geometry_grad=True, render_fn=render)
        return z[0, 16, 16]
    loss = value()
    loss.backward()
    analytic = xyz.grad[0, 2].item()
    h = 1e-3
    with torch.no_grad():
        xyz[0, 2] += h
    plus = value().item()
    with torch.no_grad():
        xyz[0, 2] -= 2 * h
    minus = value().item()
    with torch.no_grad():
        xyz[0, 2] += h
    numerical = (plus - minus) / (2 * h)
    assert abs(analytic) > 1e-4
    assert analytic == pytest.approx(numerical, rel=.1, abs=.02)
