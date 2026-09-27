"""Geometric model predictive contouring control for the 3-DOF vessel.

The controller follows a fixed :class:`~vessel_gnc.path.PathGeometry` without
an absolute mission clock. Its optimization state contains the shared physical
8-state vessel/actuator predictor and a monotone chord-progress state ``S``
[m], driven by the bounded virtual progress speed ``V_s`` [m/s].

North/East and sign conventions follow ``docs/model.md`` and ``plan.md``:
contouring error is positive to port/left, while lag error is positive when
the vessel is ahead of the candidate path point.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, fields
from numbers import Integral

import casadi as ca
import numpy as np

from vessel_gnc import _core
from vessel_gnc.path import PathGeometry
from vessel_gnc.prediction import build_prediction_step, environment_vector

__all__ = ["MPCC_COMPONENT_ID", "MpccConfig", "VesselMpcc"]

MPCC_COMPONENT_ID = "disturbance_aware_mpcc_v1"
_ACCEPTED_STATUSES = ("Solve_Succeeded", "Solved_To_Acceptable_Level")


@dataclass(frozen=True)
class MpccConfig:
    """Geometric MPCC formulation parameters.

    Cost weights multiply one stage of the unscaled objective. Contour and lag
    errors are in metres, heading error in radians, physical commands in N and
    N m, and virtual progress speed in m/s. ``r_vs`` weights departure from
    ``progress_speed_ref``; ``s_vs`` weights virtual-speed increments between
    stages (the first increment is anchored to the prior applied virtual speed,
    or to ``progress_speed_ref`` after reset). The default ``r_vs = 0.5`` is a
    deliberate *conditioning* term: it keeps the virtual-speed Hessian block
    positive definite (diagonal contribution ``2*r_vs`` dominates the
    rank-deficient increment penalty), without which IPOPT's dual infeasibility
    stalls near ``1e-2`` on the first-turn instances of the ``bench_mpcc``
    workload (see docs/control.md).
    """

    horizon: int = 25  # prediction steps
    dt: float = 0.4  # [s] prediction step
    substeps: int = 2  # internal RK4 steps per prediction step
    q_contour: float = 8.0  # contour-error weight [1/m^2]
    q_lag: float = 4.0  # lag-error weight [1/m^2]
    q_heading: float = 1.5  # wrapped heading-error weight [1/rad^2]
    q_progress: float = 8.25  # bounded virtual-progress reward [s/m]
    r_thrust: float = 2e-3  # command-effort weight [1/N^2]
    r_moment: float = 1e-2  # command-effort weight [1/(N m)^2]
    s_thrust: float = 5e-3  # command-increment weight [1/N^2]
    s_moment: float = 5e-2  # command-increment weight [1/(N m)^2]
    r_vs: float = 0.5  # virtual-speed reference weight [s^2/m^2];
    # > 0 keeps the V_s Hessian block positive definite (docs/control.md)
    s_vs: float = 0.1  # virtual-speed-increment weight [s^2/m^2]
    progress_speed_ref: float = 1.3  # [m/s] regularization anchor
    progress_speed_max: float = 2.0  # [m/s] hard upper bound
    warm_start: bool = True
    # Per-attempt IPOPT wall-time cap [s], the real-time safety net. Behaviour
    # tests lift it so accept/reject outcomes cannot depend on machine load.
    solver_max_wall_time_s: float = 0.4

    def __post_init__(self) -> None:
        """Reject invalid dimensions, bounds and non-finite weights."""
        for name in ("horizon", "substeps"):
            value = getattr(self, name)
            if isinstance(value, bool) or not isinstance(value, Integral) or value < 1:
                raise ValueError(
                    f"{name} must be an integer greater than or equal to one"
                )
        scalar_fields = (
            field.name
            for field in fields(self)
            if field.name not in {"horizon", "substeps", "warm_start"}
        )
        for name in scalar_fields:
            value = getattr(self, name)
            if not np.isfinite(value):
                raise ValueError(f"{name} must be finite")
        if self.dt <= 0.0:
            raise ValueError("dt must be positive")
        weights = (
            "q_contour",
            "q_lag",
            "q_heading",
            "q_progress",
            "r_thrust",
            "r_moment",
            "s_thrust",
            "s_moment",
            "r_vs",
            "s_vs",
        )
        if any(getattr(self, name) < 0.0 for name in weights):
            raise ValueError("MPCC weights must be non-negative")
        if self.progress_speed_max <= 0.0:
            raise ValueError("progress_speed_max must be positive")
        if self.solver_max_wall_time_s <= 0.0:
            raise ValueError("solver_max_wall_time_s must be positive")
        if not 0.0 <= self.progress_speed_ref <= self.progress_speed_max:
            raise ValueError("progress_speed_ref must lie in [0, progress_speed_max]")


def _contour_lag_errors(
    position: ca.MX,
    path_position: ca.MX,
    unit_tangent: ca.MX,
) -> tuple[ca.MX, ca.MX]:
    """Return port-positive contour and tangent-positive lag errors [m]."""
    delta = position - path_position
    lag = unit_tangent[0] * delta[0] + unit_tangent[1] * delta[1]
    contour = unit_tangent[1] * delta[0] - unit_tangent[0] * delta[1]
    return contour, lag


class VesselMpcc:
    """Disturbance-aware geometric MPCC for the vessel and actuator model.

    The column-major decision vector is exactly ``[vec(X), vec(S), vec(U),
    vec(V_s)]``, with ``X`` shaped ``(8, N+1)``, ``S`` ``(1, N+1)``, ``U``
    ``(2, N)`` and ``V_s`` ``(1, N)``. ``X[:, 0]`` and ``S[0]`` are pinned at
    every solve. The disturbance estimate is explicit and held constant over
    the horizon; no mission time or time-indexed reference enters this API.

    Args:
        params: nominal vessel parameters and physical actuator bounds.
        path: immutable geometric path fixed for the lifetime of the NLP.
        config: horizon, weights and virtual-progress bounds.
    """

    component_id = MPCC_COMPONENT_ID

    def __init__(
        self,
        params: _core.ModelParams,
        path: PathGeometry,
        config: MpccConfig | None = None,
    ):
        if not isinstance(path, PathGeometry):
            raise TypeError("path must be a PathGeometry")
        self.params = params
        self.path = path
        self.config = config if config is not None else MpccConfig()
        self.last_solve_time = 0.0  # [s]
        self.last_status = ""
        self.last_trajectory: np.ndarray | None = None  # (8, N+1)
        self.last_progress_trajectory: np.ndarray | None = None  # (N+1,) [m]
        self.last_controls: np.ndarray | None = None  # (2, N)
        self.last_progress_speed: np.ndarray | None = None  # (N,) [m/s]
        self.last_attempt_statuses: tuple[str, ...] = ()
        self.last_attempted_decision: np.ndarray | None = None
        self.last_attempted_trajectory: np.ndarray | None = None
        self.last_attempted_progress_trajectory: np.ndarray | None = None
        self.last_attempted_controls: np.ndarray | None = None
        self.last_attempted_progress_speed: np.ndarray | None = None
        self._accepted_progress: float | None = None
        self._previous_progress_speed: float | None = None
        self._build()

    def solve(
        self,
        state: _core.State,
        actuator: _core.ActuatorState,
        u_prev: _core.Control,
        *,
        disturbance_estimate: _core.Environment | None = None,
    ) -> _core.Control:
        """Solve from the estimated state and return the first bounded command.

        Args:
            state: current six-state vessel estimate.
            actuator: current physical actuator-state estimate.
            u_prev: previously applied command, used by the command-rate cost.
            disturbance_estimate: explicit constant inertial environment
                estimate over the prediction horizon. ``None`` is exactly zero
                current and zero wind force.

        Returns:
            First optimized thrust [N] and yaw-moment [N m] command after an
            accepted IPOPT status. If every attempt rejects or raises, the
            failed iterate is never applied and the deterministic fallback is
            ``u_prev`` clipped componentwise to the physical command bounds.
        """
        x0, previous_control = self._validated_initial_values(
            state=state,
            actuator=actuator,
            u_prev=u_prev,
        )
        disturbance = environment_vector(disturbance_estimate)
        candidate_progress = self._candidate_geometric_progress(x0[:2])
        progress_speed_anchor = (
            self.config.progress_speed_ref
            if self._previous_progress_speed is None
            else self._previous_progress_speed
        )
        solve_lbw = self.lbw.copy()
        solve_ubw = self.ubw.copy()
        solve_lbw[self._x_slice.start : self._x_slice.start + 8] = x0
        solve_ubw[self._x_slice.start : self._x_slice.start + 8] = x0
        solve_lbw[self._s_slice.start] = candidate_progress
        solve_ubw[self._s_slice.start] = candidate_progress
        parameters = np.concatenate(
            [previous_control, [progress_speed_anchor], disturbance]
        )

        self._clear_attempt_diagnostics()
        start = time.perf_counter()
        accepted_decision = None
        attempt_statuses = []
        for guess in self._guesses(
            x0=x0,
            progress=candidate_progress,
            disturbance=disturbance,
        ):
            try:
                solution = self.solver(
                    x0=guess,
                    lbx=solve_lbw,
                    ubx=solve_ubw,
                    lbg=self.lbg,
                    ubg=self.ubg,
                    p=parameters,
                )
                status = str(self.solver.stats()["return_status"])
                decision = np.array(solution["x"], dtype=float).ravel()
                self._record_attempt(decision)
            except Exception as error:  # CasADi raises on some rejected solves.
                status = f"Solver_Exception:{type(error).__name__}"
                attempt_statuses.append(status)
                continue
            attempt_statuses.append(status)
            if self._ok(status):
                accepted_decision = decision
                break

        self.last_solve_time = time.perf_counter() - start
        self.last_attempt_statuses = tuple(attempt_statuses)
        self.last_status = (
            attempt_statuses[-1] if attempt_statuses else "No_Solver_Attempt"
        )
        if accepted_decision is None:
            return self._bounded_fallback(previous_control)

        self.lbw = solve_lbw
        self.ubw = solve_ubw
        self._accepted_progress = candidate_progress
        self._record_solution(accepted_decision)
        self.w0 = accepted_decision.copy()
        self._previous_progress_speed = float(self.last_progress_speed[0])
        return self._bounded_fallback(self.last_controls[:, 0])

    def reset(self) -> None:
        """Clear warm starts, progress tracking and all recorded solve state."""
        self.w0 = None
        self._accepted_progress = None
        self._previous_progress_speed = None
        self.last_trajectory = None
        self.last_progress_trajectory = None
        self.last_controls = None
        self.last_progress_speed = None
        self._clear_attempt_diagnostics()
        self.last_attempt_statuses = ()
        self.last_solve_time = 0.0
        self.last_status = ""

    @property
    def last_solve_succeeded(self) -> bool:
        """Whether the latest IPOPT status has the existing accepted meaning."""
        return self._ok(self.last_status)

    @property
    def accepted_progress(self) -> float | None:
        """Latest monotone geometric progress accepted at the solve origin [m]."""
        return self._accepted_progress

    def model_step(
        self,
        x: np.ndarray,
        u: np.ndarray,
        disturbance_estimate: _core.Environment | None = None,
    ) -> np.ndarray:
        """Propagate the shared physical 8-state predictor by one MPCC step."""
        state = np.asarray(x, dtype=float)
        command = np.asarray(u, dtype=float)
        if state.shape != (8,) or command.shape != (2,):
            raise ValueError("x and u must have shapes (8,) and (2,)")
        if not np.all(np.isfinite(state)) or not np.all(np.isfinite(command)):
            raise ValueError("x and u must contain finite values")
        disturbance = environment_vector(disturbance_estimate)
        return np.array(
            self.F(ca.DM(state), ca.DM(command), ca.DM(disturbance))
        ).ravel()

    def _build(self) -> None:
        cfg = self.config
        n_steps = cfg.horizon
        self.F = build_prediction_step(
            params=self.params,
            dt=cfg.dt,
            substeps=cfg.substeps,
        )

        X = ca.MX.sym("X", 8, n_steps + 1)
        S = ca.MX.sym("S", 1, n_steps + 1)
        U = ca.MX.sym("U", 2, n_steps)
        progress_speed = ca.MX.sym("Vs", 1, n_steps)
        decision = ca.vertcat(
            ca.reshape(X, -1, 1),
            ca.reshape(S, -1, 1),
            ca.reshape(U, -1, 1),
            ca.reshape(progress_speed, -1, 1),
        )

        previous_control = ca.MX.sym("u_prev", 2)
        previous_progress_speed = ca.MX.sym("vs_prev")
        disturbance = ca.MX.sym("disturbance", 4)
        parameters = ca.vertcat(
            previous_control,
            previous_progress_speed,
            disturbance,
        )

        cost = 0.0
        physical_constraints = []
        progress_constraints = []
        for k in range(n_steps):
            physical_constraints.append(
                X[:, k + 1] - self.F(X[:, k], U[:, k], disturbance)
            )
            progress_constraints.append(
                S[0, k + 1] - S[0, k] - cfg.dt * progress_speed[0, k]
            )

            path_position = self.path.casadi_position(S[0, k + 1])
            tangent = self.path.casadi_unit_tangent(S[0, k + 1])
            contour_error, lag_error = _contour_lag_errors(
                X[0:2, k + 1],
                path_position,
                tangent,
            )
            heading_error = X[2, k + 1] - self.path.casadi_heading(S[0, k + 1])
            wrapped_heading_error = ca.atan2(
                ca.sin(heading_error),
                ca.cos(heading_error),
            )
            cost += cfg.q_contour * contour_error**2
            cost += cfg.q_lag * lag_error**2
            cost += cfg.q_heading * wrapped_heading_error**2
            cost -= cfg.q_progress * progress_speed[0, k]
            cost += cfg.r_thrust * U[0, k] ** 2
            cost += cfg.r_moment * U[1, k] ** 2
            control_increment = U[:, k] - (previous_control if k == 0 else U[:, k - 1])
            cost += cfg.s_thrust * control_increment[0] ** 2
            cost += cfg.s_moment * control_increment[1] ** 2
            cost += cfg.r_vs * (progress_speed[0, k] - cfg.progress_speed_ref) ** 2
            progress_speed_increment = progress_speed[0, k] - (
                previous_progress_speed if k == 0 else progress_speed[0, k - 1]
            )
            cost += cfg.s_vs * progress_speed_increment**2

        constraints = ca.vertcat(*physical_constraints, *progress_constraints)
        options = {
            "expand": True,
            "ipopt": {
                "print_level": 0,
                "sb": "yes",
                "max_iter": 300,
                # Adaptive barrier: the monotone mu path stalls on the
                # turn-2 instances of bench_mpcc (hundreds of iterations,
                # wall-time-capped), while the adaptive strategy converges
                # them in ~10-25 iterations without changing the optimum.
                "mu_strategy": "adaptive",
                "tol": 1e-4,
                "acceptable_tol": 1e-4,
                "acceptable_iter": 8,
                "max_wall_time": cfg.solver_max_wall_time_s,
            },
            "print_time": False,
        }
        self.solver = ca.nlpsol(
            "mpcc",
            "ipopt",
            {"x": decision, "f": cost, "g": constraints, "p": parameters},
            options,
        )

        x_size = 8 * (n_steps + 1)
        s_size = n_steps + 1
        u_size = 2 * n_steps
        vs_size = n_steps
        self._x_slice = slice(0, x_size)
        self._s_slice = slice(x_size, x_size + s_size)
        self._u_slice = slice(x_size + s_size, x_size + s_size + u_size)
        self._vs_slice = slice(
            x_size + s_size + u_size, x_size + s_size + u_size + vs_size
        )

        self.lbw = np.full(self._vs_slice.stop, -np.inf)
        self.ubw = np.full(self._vs_slice.stop, np.inf)
        state_lower = np.array(
            [
                -np.inf,
                -np.inf,
                -np.inf,
                -10.0,
                -10.0,
                -5.0,
                self.params.thrust_min,
                self.params.moment_min,
            ]
        )
        state_upper = np.array(
            [
                np.inf,
                np.inf,
                np.inf,
                10.0,
                10.0,
                5.0,
                self.params.thrust_max,
                self.params.moment_max,
            ]
        )
        self.lbw[self._x_slice] = np.tile(state_lower, n_steps + 1)
        self.ubw[self._x_slice] = np.tile(state_upper, n_steps + 1)
        self.lbw[self._s_slice] = 0.0
        self.ubw[self._s_slice] = self.path.length
        self.lbw[self._u_slice.start : self._u_slice.stop : 2] = self.params.thrust_min
        self.ubw[self._u_slice.start : self._u_slice.stop : 2] = self.params.thrust_max
        self.lbw[self._u_slice.start + 1 : self._u_slice.stop : 2] = (
            self.params.moment_min
        )
        self.ubw[self._u_slice.start + 1 : self._u_slice.stop : 2] = (
            self.params.moment_max
        )
        self.lbw[self._vs_slice] = 0.0
        self.ubw[self._vs_slice] = cfg.progress_speed_max
        self.lbg = np.zeros(9 * n_steps)
        self.ubg = np.zeros(9 * n_steps)
        self.w0: np.ndarray | None = None

    def _validated_initial_values(
        self,
        state: _core.State,
        actuator: _core.ActuatorState,
        u_prev: _core.Control,
    ) -> tuple[np.ndarray, np.ndarray]:
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
            ],
            dtype=float,
        )
        previous_control = np.array(
            [u_prev.thrust, u_prev.yaw_moment],
            dtype=float,
        )
        if not np.all(np.isfinite(x0)) or not np.all(np.isfinite(previous_control)):
            raise ValueError("state, actuator and u_prev must contain finite values")
        if not self.params.thrust_min <= actuator.thrust <= self.params.thrust_max:
            raise ValueError("actuator thrust must lie within the physical bounds")
        if not self.params.moment_min <= actuator.yaw_moment <= self.params.moment_max:
            raise ValueError("actuator yaw moment must lie within the physical bounds")
        if not self.params.thrust_min <= u_prev.thrust <= self.params.thrust_max:
            raise ValueError("u_prev thrust must lie within the command bounds")
        if not self.params.moment_min <= u_prev.yaw_moment <= self.params.moment_max:
            raise ValueError("u_prev yaw moment must lie within the command bounds")
        return x0, previous_control

    def _candidate_geometric_progress(self, position: np.ndarray) -> float:
        """Compute monotone solve-origin progress without committing it."""
        hint = self._accepted_progress
        projected = float(
            self.path.project(
                np.asarray(position, dtype=float),
                progress_hint_m=hint,
            ).progress[0]
        )
        candidate = projected if hint is None else max(hint, projected)
        return float(np.clip(candidate, 0.0, self.path.length))

    def _decision_blocks(
        self,
        decision: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Unpack raw solver values without concealing bound violations."""
        decision_array = np.asarray(decision, dtype=float).ravel()
        if decision_array.shape != (self._vs_slice.stop,):
            raise ValueError("solver returned an invalid decision-vector size")
        n_steps = self.config.horizon
        trajectory = decision_array[self._x_slice].reshape(
            8,
            n_steps + 1,
            order="F",
        )
        progress_trajectory = decision_array[self._s_slice]
        controls = decision_array[self._u_slice].reshape(
            2,
            n_steps,
            order="F",
        )
        progress_speed = decision_array[self._vs_slice]
        return tuple(
            block.copy()
            for block in (
                trajectory,
                progress_trajectory,
                controls,
                progress_speed,
            )
        )

    def _record_attempt(self, decision: np.ndarray) -> None:
        """Record the latest returned iterate exactly as IPOPT supplied it."""
        blocks = self._decision_blocks(decision)
        self.last_attempted_decision = np.asarray(decision, dtype=float).ravel().copy()
        (
            self.last_attempted_trajectory,
            self.last_attempted_progress_trajectory,
            self.last_attempted_controls,
            self.last_attempted_progress_speed,
        ) = blocks

    def _record_solution(self, decision: np.ndarray) -> None:
        """Record one accepted iterate without diagnostic clipping."""
        (
            self.last_trajectory,
            self.last_progress_trajectory,
            self.last_controls,
            self.last_progress_speed,
        ) = self._decision_blocks(decision)

    def _clear_attempt_diagnostics(self) -> None:
        self.last_attempted_decision = None
        self.last_attempted_trajectory = None
        self.last_attempted_progress_trajectory = None
        self.last_attempted_controls = None
        self.last_attempted_progress_speed = None

    def _bounded_fallback(self, control: np.ndarray) -> _core.Control:
        """Return a deterministic componentwise bounded command."""
        return _core.Control(
            thrust=float(
                np.clip(control[0], self.params.thrust_min, self.params.thrust_max)
            ),
            yaw_moment=float(
                np.clip(control[1], self.params.moment_min, self.params.moment_max)
            ),
        )

    def _ok(self, status: str) -> bool:
        return status in _ACCEPTED_STATUSES

    def _guesses(
        self,
        x0: np.ndarray,
        progress: float,
        disturbance: np.ndarray,
    ) -> list[np.ndarray]:
        """Shifted warm start followed by dynamically consistent rollouts."""
        guesses = []
        if self.config.warm_start and self.w0 is not None:
            guesses.append(self._shifted_guess(x0=x0, progress=progress))

        if self.last_controls is None:
            primary_control = np.array([self.params.thrust_max, 0.0])
        else:
            primary_control = self.last_controls[:, 0]
        relative_surge = x0[3] - (
            np.cos(x0[2]) * disturbance[0] + np.sin(x0[2]) * disturbance[1]
        )
        equilibrium_thrust = (
            self.params.lin_damping_u * relative_surge
            + self.params.quad_damping_u * abs(relative_surge) * relative_surge
        )
        equilibrium_control = np.array(
            [
                np.clip(
                    equilibrium_thrust,
                    self.params.thrust_min,
                    self.params.thrust_max,
                ),
                0.0,
            ]
        )
        for control in (primary_control, equilibrium_control):
            rollout = self._rollout_guess(
                x0=x0,
                progress=progress,
                control=np.asarray(control, dtype=float),
                disturbance=disturbance,
            )
            if not guesses or not np.allclose(rollout, guesses[-1]):
                guesses.append(rollout)
        return guesses

    def _rollout_guess(
        self,
        x0: np.ndarray,
        progress: float,
        control: np.ndarray,
        disturbance: np.ndarray,
    ) -> np.ndarray:
        """Physical-model rollout with feasible monotone progress dynamics."""
        n_steps = self.config.horizon
        X = np.empty((8, n_steps + 1))
        S = np.empty(n_steps + 1)
        U = np.tile(control, (n_steps, 1)).T
        progress_speed = np.empty(n_steps)
        X[:, 0] = x0
        S[0] = progress
        for k in range(n_steps):
            X[:, k + 1] = np.array(
                self.F(ca.DM(X[:, k]), ca.DM(control), ca.DM(disturbance))
            ).ravel()
            progress_speed[k] = min(
                self.config.progress_speed_ref,
                max(0.0, (self.path.length - S[k]) / self.config.dt),
            )
            S[k + 1] = S[k] + self.config.dt * progress_speed[k]
        return np.concatenate(
            [
                X.ravel(order="F"),
                S,
                U.ravel(order="F"),
                progress_speed,
            ]
        )

    def _shifted_guess(self, x0: np.ndarray, progress: float) -> np.ndarray:
        """Shift ``X``, ``S``, ``U`` and ``V_s`` independently by one stage."""
        n_steps = self.config.horizon
        decision = self.w0
        X = decision[self._x_slice].reshape(8, n_steps + 1, order="F")
        S = decision[self._s_slice]
        U = decision[self._u_slice].reshape(2, n_steps, order="F")
        progress_speed = decision[self._vs_slice]

        X_new = np.empty_like(X)
        X_new[:, :-1] = X[:, 1:]
        X_new[:, -1] = X[:, -1]
        X_new[:, 0] = x0

        S_new = np.empty_like(S)
        S_new[:-1] = S[1:]
        S_new[-1] = S[-1]
        S_new[0] = progress
        S_new = np.maximum.accumulate(np.maximum(S_new, progress))
        np.clip(S_new, 0.0, self.path.length, out=S_new)

        U_new = np.empty_like(U)
        U_new[:, :-1] = U[:, 1:]
        U_new[:, -1] = U[:, -1]

        progress_speed_new = np.empty_like(progress_speed)
        progress_speed_new[:-1] = progress_speed[1:]
        progress_speed_new[-1] = progress_speed[-1]

        return np.concatenate(
            [
                X_new.ravel(order="F"),
                S_new,
                U_new.ravel(order="F"),
                progress_speed_new,
            ]
        )
