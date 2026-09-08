"""Linear, clamped cubic spline and Neural ODE trajectories in se(3)."""

import torch
from torch import Tensor, nn

from utils.pose_utils import adjoint, inverse_pose, se3_exp

TRAJECTORY_MODEL = "linear_se3"
CHECKPOINT_VERSION = 2


class ExposureTrajectory(nn.Module):
    model_name = TRAJECTORY_MODEL

    def __init__(self, initial_w2c: Tensor):
        super().__init__()
        self.register_buffer("initial_w2c", initial_w2c.clone())
        # Two endpoints parameterize the curve; exposure_samples controls render count.
        self.controls = nn.Parameter(initial_w2c.new_zeros(2, 6))

    def twists(self, times: Tensor) -> Tensor:
        t = times.reshape(-1, 1)
        return (1 - t) * self.controls[0] + t * self.controls[1]

    def forward(self, times: Tensor) -> Tensor:
        return self.initial_w2c[None] @ se3_exp(self.twists(times))

    def at(self, time: float) -> Tensor:
        return self(self.controls.new_tensor([time]))[0]

    @torch.no_grad()
    def initialize_from_camera_motion(self, motion: Tensor) -> None:
        # The solver returns camera motion, whose world-to-camera increment has opposite sign.
        local = adjoint(inverse_pose(self.initial_w2c), -motion)
        factors = local.new_tensor([-0.5, 0.5])
        self.controls.copy_(factors[:, None] * local[None])

    def regularizers(self) -> tuple[Tensor, Tensor]:
        # d^2 xi / dt^2 is analytically zero. Avoid penalizing floating-point differences.
        acceleration = self.controls.sum() * 0
        # Twist norm is a local SE(3) anchor, not a globally bi-invariant metric.
        anchor = self.twists(self.controls.new_tensor([0.5])).square().mean()
        return acceleration, anchor


def require_linear_checkpoint(checkpoint: dict) -> None:
    """Do not silently reinterpret four-control optimizer state as a linear trajectory."""
    if (
        checkpoint.get("format_version") != CHECKPOINT_VERSION
        or checkpoint.get("trajectory_model") != TRAJECTORY_MODEL
    ):
        raise ValueError(
            "Resume requires a linear_se3 checkpoint (format_version=2). "
            "Legacy Bezier checkpoints can still be rendered, but start a new run for linear training."
        )


def checkpoint_midpoint(checkpoint: dict, index: int) -> Tensor:
    """Recover the exact saved midpoint, including legacy Bezier render-only support."""
    state = checkpoint["trajectories"]
    initial = state[f"{index}.initial_w2c"]
    if checkpoint.get("format_version") == 3:
        config = checkpoint["config"]
        trajectory = make_trajectory(initial, config["trajectory"], config.get("ode_steps", 16))
        if checkpoint["trajectory_model"] != trajectory.model_name:
            raise ValueError("Checkpoint trajectory metadata mismatch")
        prefix = f"{index}."
        trajectory.load_state_dict(
            {k[len(prefix) :]: v for k, v in state.items() if k.startswith(prefix)}
        )
        return trajectory.at(0.5)
    controls = state[f"{index}.controls"]
    if checkpoint.get("format_version") == 1:
        if controls.shape != (4, 6):
            raise ValueError("Legacy Bezier checkpoint requires four 6D controls")
        midpoint = controls.new_tensor([0.125, 0.375, 0.375, 0.125]) @ controls
    else:
        require_linear_checkpoint(checkpoint)
        if controls.shape != (2, 6):
            raise ValueError("Linear checkpoint requires two 6D endpoint controls")
        midpoint = controls.mean(0)
    return initial @ se3_exp(midpoint)


class SplineTrajectory(ExposureTrajectory):
    """Clamped cubic B-spline with knots [0,0,0,0,1,1,1,1] (Bezier basis).

    Four independent twist controls represent one smooth exposure segment.
    Linear controls reproduce the linear trajectory exactly.
    """

    model_name = "spline_se3"

    def __init__(self, initial_w2c: Tensor):
        super().__init__(initial_w2c)
        self.controls = nn.Parameter(initial_w2c.new_zeros(4, 6))

    def twists(self, times: Tensor) -> Tensor:
        t = times.reshape(-1, 1)
        basis = torch.cat(((1 - t) ** 3, 3 * t * (1 - t) ** 2, 3 * t * t * (1 - t), t**3), dim=1)
        return basis @ self.controls

    @torch.no_grad()
    def initialize_from_camera_motion(self, motion: Tensor) -> None:
        local = adjoint(inverse_pose(self.initial_w2c), -motion)
        self.controls.copy_(
            torch.linspace(-0.5, 0.5, 4, device=local.device, dtype=local.dtype)[:, None] * local
        )

    def regularizers(self) -> tuple[Tensor, Tensor]:
        a = self.controls[2] - 2 * self.controls[1] + self.controls[0]
        b = self.controls[3] - 2 * self.controls[2] + self.controls[1]
        # Exact integral over [0,1] of the squared second twist derivative.
        acceleration = 12 * (a.square() + a * b + b.square()).mean()
        return acceleration, self.twists(self.controls.new_tensor([0.5])).square().mean()


class ODETrajectory(ExposureTrajectory):
    """Per-image Neural ODE: dxi/dt = velocity + MLP(t, xi).

    Fixed-step differentiable RK4, integrated independently from t=0 to each
    query time, makes the midpoint independent of the exposure sample count.
    This is a BLUR-GS parameterization, not a reproduction of CoMoGaussian.
    """

    model_name = "ode_se3"

    def __init__(self, initial_w2c: Tensor, steps: int = 16):
        super().__init__(initial_w2c)
        del self.controls
        if steps < 1:
            raise ValueError("ode_steps must be positive")
        self.steps = steps
        self.initial_twist = nn.Parameter(initial_w2c.new_zeros(6))
        self.velocity = nn.Parameter(initial_w2c.new_zeros(6))
        self.field = nn.Sequential(nn.Linear(7, 32), nn.Tanh(), nn.Linear(32, 6)).to(initial_w2c)
        nn.init.zeros_(self.field[-1].weight)
        nn.init.zeros_(self.field[-1].bias)

    def derivative(self, t: Tensor, xi: Tensor) -> Tensor:
        return self.velocity + self.field(torch.cat((t, xi), -1))

    def twists(self, times: Tensor) -> Tensor:
        h = times.reshape(-1, 1) / self.steps
        xi = self.initial_twist.expand(len(h), -1)
        for i in range(self.steps):
            t = i * h
            k1 = self.derivative(t, xi)
            k2 = self.derivative(t + h / 2, xi + h * k1 / 2)
            k3 = self.derivative(t + h / 2, xi + h * k2 / 2)
            k4 = self.derivative(t + h, xi + h * k3)
            xi = xi + h * (k1 + 2 * k2 + 2 * k3 + k4) / 6
        return xi

    def at(self, time: float) -> Tensor:
        return self(self.initial_twist.new_tensor([time]))[0]

    @torch.no_grad()
    def initialize_from_camera_motion(self, motion: Tensor) -> None:
        local = adjoint(inverse_pose(self.initial_w2c), -motion)
        self.initial_twist.copy_(-0.5 * local)
        self.velocity.copy_(local)
        self.field[-1].weight.zero_()
        self.field[-1].bias.zero_()

    def regularizers(self) -> tuple[Tensor, Tensor]:
        xi = self.twists(
            torch.linspace(
                0, 1, 9, device=self.initial_twist.device, dtype=self.initial_twist.dtype
            )
        )
        acceleration = ((xi[2:] - 2 * xi[1:-1] + xi[:-2]) * 64).square().mean()
        return acceleration, xi[4].square().mean()


def make_trajectory(initial_w2c: Tensor, kind: str = "linear", ode_steps: int = 16):
    if kind == "linear":
        return ExposureTrajectory(initial_w2c)
    if kind == "spline":
        return SplineTrajectory(initial_w2c)
    if kind == "ode":
        return ODETrajectory(initial_w2c, ode_steps)
    raise ValueError(f"Unknown trajectory: {kind}")
