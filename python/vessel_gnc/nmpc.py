"""Nonlinear model predictive control with CasADi (docs/control.md §5).

The shared prediction model is an independent CasADi implementation of the
3-DOF dynamics (docs/model.md §2-§3), discretized with RK4 and cross-validated
against the C++ kernel in tests/test_nmpc.py.
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import casadi as ca
import numpy as np

from vessel_gnc import _core
from vessel_gnc.prediction import (
    ACCEPTED_IPOPT_STATUSES,
    build_prediction_step,
    environment_vector,
)

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
    """

    def __init__(
        self,
        params: _core.ModelParams,
        config: NmpcConfig | None = None,
    ):
        self.params = params
        self.config = config if config is not None else NmpcConfig()
        self.last_solve_time = 0.0  # [s]
        self.last_status = ""
        self.last_trajectory: np.ndarray | None = None  # (8, N+1)
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
        *,
        disturbance_estimate: _core.Environment | None = None,
    ) -> _core.Control:
        """Solve the NMPC problem and return the first control action.

        Args:
            state: current vessel state estimate.
            actuator: current actuator state (nominal model, docs/model.md §5).
            refs: (N, 2) reference positions along the path.
            psi_refs: (N,) reference headings [rad] (path tangents).
            u_prev: previously applied control (rate-cost anchor).
            disturbance_estimate: inertial current-equivalent velocity and
                optional force estimate, assumed constant over the prediction
                horizon. ``None`` gives the nominal calm-water model.
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
        disturbance = environment_vector(disturbance_estimate)
        p = np.concatenate(
            [
                np.asarray(refs, dtype=float).ravel(),
                psi_refs,
                [u_prev.thrust, u_prev.yaw_moment],
                disturbance,
            ]
        )

        t0 = time.perf_counter()
        sol = None
        status = ""
        # Initial guess: shifted previous solution when warm start is enabled,
        # with automatic fallback to the physical rollout (a shifted guess can
        # make IPOPT diverge on this nonconvex problem, see docs/control.md §5).
        for guess in self._guesses(x0, disturbance):
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

    @property
    def last_solve_succeeded(self) -> bool:
        """Whether the most recent solve ended with an accepted IPOPT status.

        True after a solve that returned with ``Solve_Succeeded`` or
        ``Solved_To_Acceptable_Level``; False before the first solve and
        after any other final status. Read-only: never changes solve
        behaviour.
        """
        return self._ok(self.last_status)

    def model_step(
        self,
        x: np.ndarray,
        u: np.ndarray,
        disturbance_estimate: _core.Environment | None = None,
    ) -> np.ndarray:
        """One RK4 model step on the 8-state model (vessel + actuator).

        Args:
            x: vessel plus actuator state, shape ``(8,)``.
            u: commanded thrust and yaw moment, shape ``(2,)``.
            disturbance_estimate: constant inertial environment over the step.

        Returns:
            The propagated 8-state vector, cross-validatable against C++.
        """
        disturbance = environment_vector(disturbance_estimate)
        return np.array(self.F(ca.DM(x), ca.DM(u), ca.DM(disturbance))).ravel()

    # --- internals ----------------------------------------------------------

    def _build(self) -> None:
        cfg = self.config
        n_steps = cfg.horizon
        cfg_min_t, cfg_max_t = self.params.thrust_min, self.params.thrust_max
        cfg_min_m, cfg_max_m = self.params.moment_min, self.params.moment_max

        # Decision variables: X (8, N+1), U (2, N); the initial state X_0 is
        # pinned by equal bounds in solve().
        disturbance = ca.MX.sym("disturbance", 4)
        F = build_prediction_step(
            params=self.params,
            dt=cfg.dt,
            substeps=cfg.substeps,
        )
        self.F = F

        X = ca.MX.sym("X", 8, n_steps + 1)
        U = ca.MX.sym("U", 2, n_steps)
        w = ca.vertcat(ca.reshape(X, -1, 1), ca.reshape(U, -1, 1))

        # Parameters: reference positions/headings, previous control and the
        # constant-over-horizon disturbance estimate [current N/E, force N/E].
        refs = ca.MX.sym("refs", 2, n_steps)
        psi_refs = ca.MX.sym("psi_refs", n_steps)
        u_prev = ca.MX.sym("u_prev", 2)
        p = ca.vertcat(ca.reshape(refs, -1, 1), psi_refs, u_prev, disturbance)

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
        g = ca.vertcat(*[X[:, k + 1] - F(X[:, k], U[:, k], disturbance) for k in range(n_steps)])

        opts = {
            "expand": True,
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
        self.solver = ca.nlpsol("nmpc", "ipopt", {"x": w, "f": cost, "g": g, "p": p}, opts)

        # Bounds: control saturation (static), initial state pinned per solve,
        # and generous state bounds (they never bind in practice but keep IPOPT
        # from exploring states where the quadratic damping overflows).
        n_x = 8 * (n_steps + 1)
        self.lbw = np.full(n_x + 2 * n_steps, -ca.inf)
        self.ubw = np.full(n_x + 2 * n_steps, ca.inf)
        state_lo = np.array([-1e3, -1e3, -200.0, -10.0, -10.0, -5.0, cfg_min_t, cfg_min_m])
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

    def _ok(self, status: str) -> bool:
        return status in ACCEPTED_IPOPT_STATUSES

    def _guesses(
        self,
        x0: np.ndarray,
        disturbance: np.ndarray,
    ) -> list[np.ndarray]:
        """Initial guesses in order of preference: the previously applied
        command (or a max-acceleration ramp on the very first solve), then a
        drag-balance cruise. The solver falls back down the list until one
        converges (start-up and turn transients can be hard for IPOPT)."""
        p = self.params
        if self.last_controls is not None:
            primary = np.array([self.last_controls[0, 0], self.last_controls[1, 0]])
        else:
            primary = np.array([p.thrust_max, 0.0])  # start-up: accelerate hard
        cos_psi = np.cos(x0[2])
        sin_psi = np.sin(x0[2])
        current_u = cos_psi * disturbance[0] + sin_psi * disturbance[1]
        relative_surge = x0[3] - current_u
        t_eq = (
            p.lin_damping_u * relative_surge
            + p.quad_damping_u * abs(relative_surge) * relative_surge
        )
        guesses = []
        if self.config.warm_start and self.w0 is not None:
            guesses.append(self._shifted_guess(x0))
        for u_const in (primary, np.array([t_eq, 0.0])):
            if not guesses or not np.allclose(guesses[-1][:2], u_const):
                guesses.append(self._rollout_guess(x0, u_const, disturbance))
        return guesses

    def _rollout_guess(
        self,
        x0: np.ndarray,
        u_const: np.ndarray,
        disturbance: np.ndarray,
    ) -> np.ndarray:
        """Constant-control rollout through the model (dynamically consistent)."""
        n_steps = self.config.horizon
        X = np.empty((8, n_steps + 1))
        X[:, 0] = x0
        for k in range(n_steps):
            X[:, k + 1] = np.array(
                self.F(ca.DM(X[:, k]), ca.DM(u_const), ca.DM(disturbance))
            ).ravel()
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
