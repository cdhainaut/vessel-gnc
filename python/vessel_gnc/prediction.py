"""Shared CasADi prediction dynamics for predictive vessel controllers.

The model reproduces the C++ ``actuator_step`` + ``rk4_step`` composition for
an 8-state vessel-plus-actuator state. It keeps actuator command clipping,
smooth ``tanh`` rate limits, per-substep actuator state projection, body-frame
current transport, and inertial current/wind conventions from ``docs/model.md``.
"""

from __future__ import annotations

from numbers import Integral

import casadi as ca
import numpy as np

from vessel_gnc import _core

# IPOPT final statuses treated as accepted solves by every predictive
# controller, benchmark workload and artifact check.
ACCEPTED_IPOPT_STATUSES = ("Solve_Succeeded", "Solved_To_Acceptable_Level")


def environment_vector(environment: _core.Environment | None) -> np.ndarray:
    """Return ``[current_N, current_E, wind_N, wind_E]`` or exact zeros.

    Args:
        environment: constant inertial environment estimate. ``None`` is the
            exact zero-disturbance case.

    Returns:
        Finite four-component disturbance vector.

    Raises:
        ValueError: if any environment component is non-finite.
    """
    if environment is None:
        return np.zeros(4)
    values = np.array(
        [
            environment.current_north,
            environment.current_east,
            environment.wind_north,
            environment.wind_east,
        ],
        dtype=float,
    )
    if not np.all(np.isfinite(values)):
        raise ValueError("disturbance_estimate must contain finite values")
    return values


def build_prediction_step(
    params: _core.ModelParams,
    dt: float,
    substeps: int,
) -> ca.Function:
    """Build the discrete 8-state vessel-plus-actuator prediction function.

    The actuator block is integrated and projected into its physical bounds
    first in every internal substep. The vessel block is then integrated with
    the resulting actuator forces held constant, matching the C++ reference
    composition exactly.

    Args:
        params: nominal vessel and actuator parameters.
        dt: complete prediction-step duration [s].
        substeps: internal RK4 steps per complete prediction step.

    Returns:
        CasADi function ``F(x, u, disturbance)`` with state shape ``(8,)``,
        commanded control shape ``(2,)``, and disturbance shape ``(4,)``.

    Raises:
        ValueError: if ``dt`` is not finite and positive, or ``substeps`` is
            not an integer greater than or equal to one.
    """
    if not np.isfinite(dt) or dt <= 0.0:
        raise ValueError("dt must be finite and positive")
    if isinstance(substeps, bool) or not isinstance(substeps, Integral) or substeps < 1:
        raise ValueError("substeps must be an integer greater than or equal to one")

    substep_dt = dt / substeps
    state = ca.MX.sym("x", 8)
    command = ca.MX.sym("u", 2)
    disturbance = ca.MX.sym("disturbance", 4)

    def rk4(ode: ca.Function, x: ca.MX, argument: ca.MX) -> ca.MX:
        """One internal RK4 step of ``x_dot = ode(x, argument)``."""
        k1 = ode(x, argument)
        k2 = ode(x + substep_dt / 2 * k1, argument)
        k3 = ode(x + substep_dt / 2 * k2, argument)
        k4 = ode(x + substep_dt * k3, argument)
        return x + substep_dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

    # Actuator model (docs/model.md §5): clipped commands and smooth tanh
    # approximation of the first-order rate limit.
    actuator_state = ca.MX.sym("actuator_state", 2)
    thrust_command = ca.fmin(ca.fmax(command[0], params.thrust_min), params.thrust_max)
    moment_command = ca.fmin(ca.fmax(command[1], params.moment_min), params.moment_max)
    thrust_dot = params.thrust_rate_limit * ca.tanh(
        (thrust_command - actuator_state[0])
        / (params.thrust_time_constant * params.thrust_rate_limit)
    )
    moment_dot = params.moment_rate_limit * ca.tanh(
        (moment_command - actuator_state[1])
        / (params.moment_time_constant * params.moment_rate_limit)
    )
    actuator_ode = ca.Function(
        "actuator_dot",
        [actuator_state, command],
        [ca.vertcat(thrust_dot, moment_dot)],
    )

    # Vessel model (docs/model.md §2-§3): absolute body-frame state velocity,
    # relative hydrodynamic velocity, and inertial current/wind inputs.
    vessel_state_symbol = ca.MX.sym("vessel_state", 6)
    vessel_input_symbol = ca.MX.sym("vessel_input", 6)

    def vessel_dot(vessel_state: ca.MX, vessel_input: ca.MX) -> ca.MX:
        applied = vessel_input[:2]
        environment = vessel_input[2:]
        cos_psi = ca.cos(vessel_state[2])
        sin_psi = ca.sin(vessel_state[2])
        current_u = cos_psi * environment[0] + sin_psi * environment[1]
        current_v = -sin_psi * environment[0] + cos_psi * environment[1]
        wind_u = cos_psi * environment[2] + sin_psi * environment[3]
        wind_v = -sin_psi * environment[2] + cos_psi * environment[3]
        u_relative = vessel_state[3] - current_u
        v_relative = vessel_state[4] - current_v
        r_relative = vessel_state[5]
        m11 = params.mass + params.added_mass_x
        m22 = params.mass + params.added_mass_y
        m33 = params.inertia_z + params.added_inertia_z
        coriolis_x = -m22 * v_relative * r_relative
        coriolis_y = m11 * u_relative * r_relative
        coriolis_r = (m22 - m11) * u_relative * v_relative
        damping_u = (
            params.lin_damping_u * u_relative
            + params.quad_damping_u * ca.fabs(u_relative) * u_relative
        )
        damping_v = (
            params.lin_damping_v * v_relative
            + params.quad_damping_v * ca.fabs(v_relative) * v_relative
        )
        damping_r = (
            params.lin_damping_r * r_relative
            + params.quad_damping_r * ca.fabs(r_relative) * r_relative
        )
        current_transport_u = r_relative * current_v
        current_transport_v = -r_relative * current_u
        return ca.vertcat(
            vessel_state[3] * cos_psi - vessel_state[4] * sin_psi,
            vessel_state[3] * sin_psi + vessel_state[4] * cos_psi,
            r_relative,
            current_transport_u + (applied[0] + wind_u - coriolis_x - damping_u) / m11,
            current_transport_v + (wind_v - coriolis_y - damping_v) / m22,
            (applied[1] - coriolis_r - damping_r) / m33,
        )

    vessel_ode = ca.Function(
        "vessel_ode",
        [vessel_state_symbol, vessel_input_symbol],
        [vessel_dot(vessel_state_symbol, vessel_input_symbol)],
    )

    vessel_state = state[:6]
    actuator = state[6:]
    for _ in range(substeps):
        actuator = rk4(actuator_ode, actuator, command)
        # Match actuator_step exactly: project the integrated actuator state
        # before its output is applied to the vessel in every substep.
        actuator = ca.vertcat(
            ca.fmin(ca.fmax(actuator[0], params.thrust_min), params.thrust_max),
            ca.fmin(ca.fmax(actuator[1], params.moment_min), params.moment_max),
        )
        vessel_input = ca.vertcat(actuator[0], actuator[1], disturbance)
        vessel_state = rk4(vessel_ode, vessel_state, vessel_input)

    return ca.Function(
        "F",
        [state, command, disturbance],
        [ca.vertcat(vessel_state, actuator)],
    )
