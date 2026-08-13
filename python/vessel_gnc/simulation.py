"""Simulation orchestration: propagate the C++ core over time.

Numerical logic lives in the C++ kernel (``vessel_gnc._core``); this module
owns the loop, the time history and input validation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol

import numpy as np

from vessel_gnc import _core

__all__ = ["SimulationResult", "simulate"]


class ControlPolicy(Protocol):
    """Control policy, evaluated at the start of each integration step."""

    def __call__(self, t: float, state: _core.State) -> _core.Control: ...


class EnvironmentPolicy(Protocol):
    """Time-varying environment, sampled at each integration step."""

    def __call__(self, t: float) -> _core.Environment: ...


@dataclass(frozen=True)
class SimulationResult:
    """Time history of a simulation run: state and applied controls."""

    t: np.ndarray  # [s]
    x: np.ndarray  # [m]     inertial North position
    y: np.ndarray  # [m]     inertial East position
    psi: np.ndarray  # [rad]
    u: np.ndarray  # [m/s]
    v: np.ndarray  # [m/s]
    r: np.ndarray  # [rad/s]
    thrust: np.ndarray  # [N]     applied thrust (post-actuator)
    yaw_moment: np.ndarray  # [N m]  applied yaw moment (post-actuator)

    @property
    def n_steps(self) -> int:
        return len(self.t) - 1

    @property
    def dt(self) -> float:
        """Integration step [s] (uniform sampling)."""
        return float(self.t[1] - self.t[0])


def simulate(
    duration: float,
    dt: float,
    params: _core.ModelParams | None = None,
    state0: _core.State | None = None,
    control: ControlPolicy | _core.Control | None = None,
    environment: EnvironmentPolicy | _core.Environment | None = None,
    clamp: bool = True,
    control_period: float | None = None,
) -> SimulationResult:
    """Propagate the vessel dynamics with fixed-step RK4.

    Args:
        duration: simulation time [s].
        dt: integration step [s].
        params: vessel model parameters (default: ``_core.default_params()``).
        state0: initial state (default: rest at the origin, heading North).
        control: constant control (zero-order hold) or a policy
            ``(t, state) -> Control`` evaluated at the start of each step.
        environment: constant environment, or a policy ``t -> Environment``
            sampled at each integration step (default: calm).
        clamp: clamp the control to the actuator limits before each step.
        control_period: for policies, the evaluation period [s] (default:
            every step). The last command is held in between (sampled control).

    Returns:
        A SimulationResult with the state and applied-control histories.

    Example:
        >>> from vessel_gnc import _core, simulate
        >>> result = simulate(
        ...     30.0, 0.01, control=_core.Control(thrust=25.0, yaw_moment=0.5)
        ... )
        >>> result.x[-1]  # final North position [m]  # doctest: +SKIP
    """
    params = params if params is not None else _core.default_params()
    state = state0 if state0 is not None else _core.State()
    env_policy = environment if callable(environment) else None
    environment = environment if environment is not None else _core.Environment()
    control = control if control is not None else _core.Control()

    if duration <= 0.0 or dt <= 0.0:
        raise ValueError("duration and dt must be positive")
    period = control_period if control_period is not None else dt
    if period <= 0.0:
        raise ValueError("control_period must be positive")
    env0 = env_policy(0.0) if env_policy is not None else environment
    values = (
        state.x,
        state.y,
        state.psi,
        state.u,
        state.v,
        state.r,
        env0.current_north,
        env0.current_east,
        env0.wind_north,
        env0.wind_east,
    )
    if not np.isfinite(values).all():
        raise ValueError("state and environment must contain finite values")

    n_steps = int(round(duration / dt))
    if n_steps < 1:
        raise ValueError(f"duration/dt must be >= 1, got {n_steps} step(s)")

    is_policy = callable(control)
    last_ctrl_t = -np.inf
    n_samples = n_steps + 1
    t = np.arange(n_samples) * dt
    x = np.empty(n_samples)
    y = np.empty(n_samples)
    psi = np.empty(n_samples)
    u = np.empty(n_samples)
    v = np.empty(n_samples)
    r = np.empty(n_samples)
    thrust = np.empty(n_samples)
    yaw_moment = np.empty(n_samples)
    actuator = _core.ActuatorState()

    for k in range(n_steps):
        if env_policy is not None:
            environment = env_policy(t[k])
        if is_policy:
            if t[k] >= last_ctrl_t + period - 1e-9:
                cmd = control(t[k], state)
                last_ctrl_t = t[k]
        else:
            cmd = control
        if clamp:
            cmd = _core.clamp_control(cmd, params)
        # The plant applies the actuator output (lag, rate limit, saturation),
        # not the raw command (docs/model.md §5).
        actuator = _core.actuator_step(actuator, cmd, params, dt)
        applied = _core.Control(thrust=actuator.thrust, yaw_moment=actuator.yaw_moment)
        x[k] = state.x
        y[k] = state.y
        psi[k] = state.psi
        u[k] = state.u
        v[k] = state.v
        r[k] = state.r
        thrust[k] = applied.thrust
        yaw_moment[k] = applied.yaw_moment
        state = _core.rk4_step(state, applied, environment, params, dt)

    x[n_steps] = state.x
    y[n_steps] = state.y
    psi[n_steps] = state.psi
    u[n_steps] = state.u
    v[n_steps] = state.v
    r[n_steps] = state.r

    return SimulationResult(
        t=t,
        x=x,
        y=y,
        psi=psi,
        u=u,
        v=v,
        r=r,
        thrust=thrust,
        yaw_moment=yaw_moment,
    )
