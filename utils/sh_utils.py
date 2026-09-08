"""Real spherical harmonics in the conventional 3DGS coefficient order."""

import torch


def sh_basis(directions, degree):
    x, y, z = directions.unbind(-1)
    basis = [torch.ones_like(x) * 0.28209479177387814]
    if degree >= 1:
        basis += [-0.4886025119029199 * y, 0.4886025119029199 * z, -0.4886025119029199 * x]
    if degree >= 2:
        basis += [
            1.0925484305920792 * x * y,
            -1.0925484305920792 * y * z,
            0.31539156525252005 * (2 * z * z - x * x - y * y),
            -1.0925484305920792 * x * z,
            0.5462742152960396 * (x * x - y * y),
        ]
    if degree >= 3:
        basis += [
            -0.5900435899266435 * y * (3 * x * x - y * y),
            2.890611442640554 * x * y * z,
            -0.4570457994644658 * y * (4 * z * z - x * x - y * y),
            0.3731763325901154 * z * (2 * z * z - 3 * x * x - 3 * y * y),
            -0.4570457994644658 * x * (4 * z * z - x * x - y * y),
            1.445305721320277 * z * (x * x - y * y),
            -0.5900435899266435 * x * (x * x - 3 * y * y),
        ]
    return torch.stack(basis, -1)
