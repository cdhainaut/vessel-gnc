"""Nonlinear model predictive control with CasADi (docs/control.md §5).

The prediction model is an independent CasADi implementation of the 3-DOF
dynamics (docs/model.md §2-§3), discretized with RK4 — deliberately
duplicated from the C++ kernel because CasADi needs symbolic expressions;
the two implementations are cross-validated in tests/test_nmpc.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import casadi as ca
import numpy as np

from vessel_gnc import _core

__all__ = ["NmpcConfig", "VesselNmpc"]


@dataclass(frozen=True)
class NmpcConfig:
    """NMPC formulation parameters (docs/control.md §5)."""

    horizon: int = 25  # prediction steps
    dt: float = 0.4  # [s] model step within the horizon
    substeps: int = 2  # internal RK4 steps per model step (stability, see §5)
    q_position: float = 8.0  # position error weight [1/m^2]
    q_heading: float = 1.5  # heading error weight [1/rad^2]
    r_thrust: float = 2e-3  # control weight [1/N^2]
    r_moment: float = 1e-2  # control weight [1/(N m)^2]
    s_thrust: float = 5e-3  # control-rate weight [1/N^2]
    s_moment: float = 5e-2  # control-rate weight [1/(N m)^2]
    warm_start: bool = True  # shifted-solution initial guess, with automatic
    # fallback to the physical rollout (see docs/control.md §5)


class VesselNmpc:
    """Receding-horizon NMPC for the 3-DOF vessel.

    Decision variables: states ``X = [x_1..x_N]`` and controls
    ``U = [u_0..u_{N-1}]`` (the initial state is pinned by bounds).
    Parameters: the reference trajectory (positions and headings) and the
    previously applied control (for the rate cost).

    Args:
        params: vessel model parameters (also provide the control bounds).
        config: horizon, weights and model step.
        environment: nominal environment for the prediction model (default:
            calm; the filter/controller does not know the true disturbances).
    """

    def __init__(
        self,
        params: _core.ModelParams,
        config: NmpcConfig | None = None,
        environment: _core.Environment | None = None,
    ):
        self.params = params
        self.config = config if config is not None else NmpcConfig()
        self.environment = (
            environment if environment is not None else _core.Environment()
        )
        self.last_solve_time = 0.0  # [s]
        self.last_status = ""
        self.last_trajectory: np.ndarray | None = None  # (6, N+1)
        self.last_controls: np.ndarray | None = None  # (2, N)
        self._build()

    # --- public API ---------------------------------------------------------

    def solve(
        self,
        state: _core.State,
        actuator: _core.ActuatorState,
        refs: np.ndarray,
        psi_refs: np.ndarray,
        u_prev: _core.Control,
    ) -> _core.Control:
        """Solve the NMPC problem and return the first control action.

        Args:
            state: current vessel state estimate.
            actuator: current actuator state (nominal model, docs/model.md §5).
            refs: (N, 2) reference positions along the path.
            psi_refs: (N,) reference headings [rad] (path tangents).
            u_prev: previously applied control (rate-cost anchor).
        """
        n_steps = self.config.horizon
        x0 = np.array(
            [
                state.x,
                state.y,
                state.psi,
                state.u,
                state.v,
                state.r,
                actuator.thrust,
                actuator.yaw_moment,
            ]
        )
        self.lbw[:8] = x0
        self.ubw[:8] = x0
        p = np.concatenate(
            [
                np.asarray(refs, dtype=float).ravel(),
                psi_refs,
                [u_prev.thrust, u_prev.yaw_moment],
            ]
        )

        t0 = time.perf_counter()
        sol = None
        status = ""
        # Initial guess: shifted previous solution when warm start is enabled,
        # with automatic fallback to the physical rollout (a shifted guess can
        # make IPOPT diverge on this nonconvex problem, see docs/control.md §5).
        for guess in self._guesses(x0):
            sol = self.solver(
                x0=guess,
                lbx=self.lbw,
                ubx=self.ubw,
                lbg=self.lbg,
                ubg=self.ubg,
                p=p,
            )
            status = self.solver.stats()["return_status"]
            if self._ok(status):
                break
        assert sol is not None
        self.last_solve_time = time.perf_counter() - t0
        self.last_status = status

        w = np.array(sol["x"]).ravel()
        # CasADi stores MX column-major: reshape/ravel with Fortran order.
        self.last_trajectory = w[: 8 * (n_steps + 1)].reshape(8, n_steps + 1, order="F")
        self.last_controls = w[8 * (n_steps + 1) :].reshape(2, n_steps, order="F")
        self.w0 = w
        return _core.Control(
            thrust=float(self.last_controls[0, 0]),
            yaw_moment=float(self.last_controls[1, 0]),
        )

    def reset(self) -> None:
        """Drop all warm-start state (next solve starts from the rollout guess)."""
        self.w0 = None
        self.last_controls = None
        self.last_trajectory = None

    def model_step(self, x: np.ndarray, u: np.ndarray) -> np.ndarray:
        """One RK4 model step on the 8-state model (vessel + actuator).

        Used for cross-validation against the C++ kernel.
        """
        return np.array(self.F(ca.DM(x), ca.DM(u))).ravel()

    # --- internals ----------------------------------------------------------

    def _build(self) -> None:
        cfg = self.config
        n_steps = cfg.horizon
        cfg_min_t, cfg_max_t = self.params.thrust_min, self.params.thrust_max
        cfg_min_m, cfg_max_m = self.params.moment_min, self.params.moment_max

        # Decision variables: X (8, N+1), U (2, N); the initial state X_0 is
        # pinned by equal bounds in solve().
        x = ca.MX.sym("x", 8)
        u = ca.MX.sym("u", 2)
        F = ca.Function("F", [x, u], [self._discrete_step(x, u)])
        self.F = F

        X = ca.MX.sym("X", 8, n_steps + 1)
        U = ca.MX.sym("U", 2, n_steps)
        w = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

        # Parameters: reference positions/headings and the previous control.
        refs = ca.MX.sym("refs", 2, n_steps)
        psi_refs = ca.MX.sym("psi_refs", n_steps)
        u_prev = ca.MX.sym("u_prev", 2)
        p = ca.vertcat(ca.reshape(refs, -1, 1), psi_refs, u_prev)

        # Stage cost (docs/control.md §5).
        cost = 0.0
        for k in range(n_steps):
            e_pos = X[0:2, k + 1] - refs[:, k]
            cost += cfg.q_position * ca.dot(e_pos, e_pos)
            e_psi = X[2, k + 1] - psi_refs[k]
            cost += cfg.q_heading * ca.atan2(ca.sin(e_psi), ca.cos(e_psi)) ** 2
            cost += cfg.r_thrust * U[0, k] ** 2 + cfg.r_moment * U[1, k] ** 2
            du = U[:, k] - (u_prev if k == 0 else U[:, k - 1])
            cost += cfg.s_thrust * du[0] ** 2 + cfg.s_moment * du[1] ** 2

        # Dynamics constraints: X_{k+1} = F(X_k, U_k).
        g = ca.vertcat(*[X[:, k + 1] - F(X[:, k], U[:, k]) for k in range(n_steps)])

        opts = {
            "ipopt": {
                "print_level": 0,
                "sb": "yes",
                "max_iter": 300,
                "tol": 1e-4,
                "acceptable_tol": 1e-4,
                "acceptable_iter": 8,
                "max_wall_time": 0.4,
            },
            "print_time": False,
        }
        self.solver = ca.nlpsol(
            "nmpc", "ipopt", {"x": w, "f": cost, "g": g, "p": p}, opts
        )

        # Bounds: control saturation (static), initial state pinned per solve,
        # and generous state bounds (they never bind in practice but keep IPOPT
        # from exploring states where the quadratic damping overflows).
        n_x = 8 * (n_steps + 1)
        self.lbw = np.full(n_x + 2 * n_steps, -ca.inf)
        self.ubw = np.full(n_x + 2 * n_steps, ca.inf)
        state_lo = np.array(
            [-1e3, -1e3, -200.0, -10.0, -10.0, -5.0, cfg_min_t, cfg_min_m]
        )
        state_hi = np.array([1e3, 1e3, 200.0, 10.0, 10.0, 5.0, cfg_max_t, cfg_max_m])
        self.lbw[:n_x] = np.tile(state_lo, n_steps + 1)
        self.ubw[:n_x] = np.tile(state_hi, n_steps + 1)
        self.lbw[n_x::2] = self.params.thrust_min
        self.ubw[n_x::2] = self.params.thrust_max
        self.lbw[n_x + 1 :: 2] = self.params.moment_min
        self.ubw[n_x + 1 :: 2] = self.params.moment_max
        self.lbg = np.zeros(8 * n_steps)
        self.ubg = np.zeros(8 * n_steps)
        self.w0 = None

    def _discrete_step(self, x: ca.MX, u: ca.MX) -> ca.MX:
        """RK4 discretization of the continuous 8-state model.

        States: vessel (x, y, psi, u, v, r) plus actuator (thrust,
        yaw_moment); controls: commanded (thrust_cmd, moment_cmd). The
        actuator block is stepped first, then the vessel block with the
        applied forces held at their end-of-step values — the exact same
        composition as the C++ reference (actuator_step + rk4_step), which
        keeps the cross-validation test bit-tight. Each block integrates
        ``substeps`` internal RK4 steps of ``dt / substeps``: the yaw
        dynamics are fast (time constant m33/N_r ~ 0.2 s) and a single step
        of 0.4 s is outside RK4's stability margin once the Munk coupling is
        active.
        """
        p = self.params
        substep_dt = self.config.dt / self.config.substeps

        def rk4(ode: ca.Function, state: ca.MX, argument: ca.MX) -> ca.MX:
            """One internal RK4 step of ``state_dot = ode(state, argument)``."""
            k1 = ode(state, argument)
            k2 = ode(state + substep_dt / 2 * k1, argument)
            k3 = ode(state + substep_dt / 2 * k2, argument)
            k4 = ode(state + substep_dt * k3, argument)
            return state + substep_dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)

        # --- Actuator block (docs/model.md §5): rate-limited first-order
        # response, smooth tanh approximation of the rate limit (C1, keeps
        # IPOPT fast). The command is clamped to the actuator bounds.
        actuator_state = ca.MX.sym("actuator_state", 2)
        thrust_cmd = ca.fmin(ca.fmax(u[0], p.thrust_min), p.thrust_max)
        moment_cmd = ca.fmin(ca.fmax(u[1], p.moment_min), p.moment_max)
        thrust_dot = p.thrust_rate_limit * ca.tanh(
            (thrust_cmd - actuator_state[0])
            / (p.thrust_time_constant * p.thrust_rate_limit)
        )
        moment_dot = p.moment_rate_limit * ca.tanh(
            (moment_cmd - actuator_state[1])
            / (p.moment_time_constant * p.moment_rate_limit)
        )
        actuator_dot = ca.Function(
            "actuator_dot", [actuator_state, u], [ca.vertcat(thrust_dot, moment_dot)]
        )

        # --- Vessel block: the applied forces are the actuator states.
        vessel_state_sym = ca.MX.sym("vessel_state", 6)
        applied_sym = ca.MX.sym("applied", 2)

        def vessel_dot(state: ca.MX, applied: ca.MX) -> ca.MX:
            u_rel, v_rel, r_rel = state[3], state[4], state[5]
            m11 = p.mass + p.added_mass_x
            m22 = p.mass + p.added_mass_y
            m33 = p.inertia_z + p.added_inertia_z
            cx = -m22 * v_rel * r_rel
            cy = m11 * u_rel * r_rel
            cr = (m22 - m11) * u_rel * v_rel
            du = p.lin_damping_u * u_rel + p.quad_damping_u * ca.fabs(u_rel) * u_rel
            dv = p.lin_damping_v * v_rel + p.quad_damping_v * ca.fabs(v_rel) * v_rel
            dr = p.lin_damping_r * r_rel + p.quad_damping_r * ca.fabs(r_rel) * r_rel
            return ca.vertcat(
                u_rel * ca.cos(state[2]) - v_rel * ca.sin(state[2]),
                u_rel * ca.sin(state[2]) + v_rel * ca.cos(state[2]),
                r_rel,
                (applied[0] - cx - du) / m11,
                (-cy - dv) / m22,
                (applied[1] - cr - dr) / m33,
            )

        vessel_ode = ca.Function(
            "vessel_ode",
            [vessel_state_sym, applied_sym],
            [vessel_dot(vessel_state_sym, applied_sym)],
        )

        # --- Compose: actuator first (end-of-step forces), then the vessel.
        vessel_state = x[:6]
        actuator = x[6:]
        for _ in range(self.config.substeps):
            actuator = rk4(actuator_dot, actuator, u)
            applied = ca.vertcat(actuator[0], actuator[1])
            vessel_state = rk4(vessel_ode, vessel_state, applied)
        return ca.vertcat(vessel_state, actuator)

    def _ok(self, status: str) -> bool:
        return status in ("Solve_Succeeded", "Solved_To_Acceptable_Level")

    def _guesses(self, x0: np.ndarray) -> list[np.ndarray]:
        """Initial guesses in order of preference: the previously applied
        command (or a max-acceleration ramp on the very first solve), then a
        drag-balance cruise. The solver falls back down the list until one
        converges (start-up and turn transients can be hard for IPOPT)."""
        p = self.params
        if self.last_controls is not None:
            primary = np.array([self.last_controls[0, 0], self.last_controls[1, 0]])
        else:
            primary = np.array([p.thrust_max, 0.0])  # start-up: accelerate hard
        t_eq = p.lin_damping_u * x0[3] + p.quad_damping_u * abs(x0[3]) * x0[3]
        guesses = []
        if self.config.warm_start and self.w0 is not None:
            guesses.append(self._shifted_guess(x0))
        for u_const in (primary, np.array([t_eq, 0.0])):
            if not guesses or not np.allclose(guesses[-1][:2], u_const):
                guesses.append(self._rollout_guess(x0, u_const))
        return guesses

    def _rollout_guess(self, x0: np.ndarray, u_const: np.ndarray) -> np.ndarray:
        """Constant-control rollout through the model (dynamically consistent)."""
        n_steps = self.config.horizon
        X = np.empty((8, n_steps + 1))
        X[:, 0] = x0
        for k in range(n_steps):
            X[:, k + 1] = self.model_step(X[:, k], u_const)
        U = np.tile(u_const, (n_steps, 1)).T
        # CasADi stores MX column-major: ravel with Fortran order.
        return np.concatenate([X.ravel(order="F"), U.ravel(order="F")])

    def _shifted_guess(self, x0: np.ndarray) -> np.ndarray:
        """Previous solution shifted by one step, with the new state pinned."""
        n_steps = self.config.horizon
        w = self.w0
        X = w[: 8 * (n_steps + 1)].reshape(8, n_steps + 1, order="F")
        U = w[8 * (n_steps + 1) :].reshape(2, n_steps, order="F")
        X_new = np.empty_like(X)
        X_new[:, :n_steps] = X[:, 1:]
        X_new[:, n_steps] = X[:, n_steps]
        U_new = np.empty_like(U)
        U_new[:, : n_steps - 1] = U[:, 1:]
        U_new[:, n_steps - 1] = U[:, n_steps - 1]
        X_new[:, 0] = x0
        return np.concatenate([X_new.ravel(order="F"), U_new.ravel(order="F")])
