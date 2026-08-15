"""Canonical flagship: LOS, nominal and disturbance-aware NMPC with EKF.

This module owns the entire flagship simulation (scenario
``scenario_v2_disturbance_aware``): the path, environment, sensor
suites, the EKFs, the PID/PI controllers, the NMPC instance and the closed
loop. ``examples/05_nmpc_demo.py`` is a thin entry point that only renders
the recorded run; nothing here is duplicated there.

Every call to ``run_reference_scenario`` constructs fresh controllers,
filters, sensor suites and NMPC instances — there is no module-level mutable
state, so repeated calls (or two runners interleaved) start clean. The two
controllers use separate RNGs initialized with the same seed, so LOS and
NMPC see identical sensor-noise realizations. The plant runs the perturbed
truth parameters while the filter and the controllers use the nominal set
(portfolio plan Phase C).

The history is recorded at every controller update (callback-aligned with
the true plant state), so estimator errors are computed from matched
true/estimated pairs, never from array slicing.

Coordinate frames and units follow docs/model.md §1: inertial position
``x`` North [m], ``y`` East [m], heading ``psi`` clockwise from North [rad],
body speeds ``u, v`` [m/s], yaw rate ``r`` [rad/s], ambient current
components [m/s].
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np

from vessel_gnc import _core
from vessel_gnc.ekf import VesselEKF
from vessel_gnc.environment import EnvironmentScenario
from vessel_gnc.guidance import los_heading, make_s_curve_path, path_reference
from vessel_gnc.metrics import path_following_metrics
from vessel_gnc.nmpc import NmpcConfig, VesselNmpc
from vessel_gnc.sensors import SensorConfig, SensorSuite
from vessel_gnc.simulation import SimulationResult, simulate

__all__ = [
    "ReferenceScenarioConfig",
    "EstimatorHistory",
    "ControllerReferenceRun",
    "ReferenceRun",
    "default_reference_config",
    "run_reference_scenario",
    "reference_metrics",
]

# Stable component IDs of the reference scenario (results/reference schema).
LOS_COMPONENT_ID = "los_pid_v1"
NMPC_COMPONENT_ID = "nominal_nmpc_v1"
DISTURBANCE_AWARE_NMPC_COMPONENT_ID = "disturbance_aware_nmpc_v1"

_SENSOR_NAMES = ("gnss", "compass", "speed", "gyro")

# Policy signature: (t, estimate, previously applied command, filter).
ControllerPolicy = Callable[
    [float, _core.State, _core.Control, VesselEKF], _core.Control
]


@dataclass(frozen=True)
class ReferenceScenarioConfig:
    """Frozen configuration of the flagship reference scenario.

    Defaults are the canonical reference scenario: 120 s at 0.01 s
    integration with seed 42, 5 Hz NMPC, 10 Hz LOS, 1.3 m/s speed
    reference, 8.0 m LOS lookahead, the rotating-current/gust environment,
    the default sensor suite and the tuned LOS/NMPC settings. The plant
    (truth) and the filter/controller (nominal) parameter sets are both
    recorded so the run stays interpretable if defaults change later.
    """

    duration_s: float = 120.0  # [s] run length
    integration_dt_s: float = 0.01  # [s] fixed RK4 step
    seed: int = 42  # sensor-noise seed (identical for LOS and NMPC)
    los_period_s: float = 0.1  # [s] 10 Hz LOS control / filter step
    nmpc_period_s: float = 0.2  # [s] 5 Hz NMPC control / filter step
    speed_ref_m_s: float = 1.3  # [m/s] surge speed reference
    lookahead_m: float = 8.0  # [m] LOS lookahead distance
    environment: EnvironmentScenario = field(default_factory=EnvironmentScenario)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    nmpc: NmpcConfig = field(default_factory=NmpcConfig)
    los_heading_gains: _core.PidGains = field(
        default_factory=_core.default_heading_gains
    )
    los_speed_gains: _core.PidGains = field(default_factory=_core.default_speed_gains)
    los_heading_moment_limit_Nm: float = 6.0  # [N m] heading-controller output limit
    los_speed_thrust_limit_N: float = 40.0  # [N] speed-controller output limit
    nominal_params: _core.ModelParams = field(default_factory=_core.default_params)
    truth_params: _core.ModelParams = field(default_factory=_core.truth_params)
    estimator_transient_s: float = 20.0  # [s] discarded before current-error stats
    render_fps: int = 12  # hero animation frame rate
    render_hero_stride_frames: int = 40  # hero frame period = stride * dt
    render_hero_wake_duration_s: float = 12.0  # [s] hero wake trail length


@dataclass(frozen=True)
class EstimatorHistory:
    """Callback-aligned estimation records at every controller update.

    Arrays share the row index ``k``: the estimate and the commanded
    control at time ``t[k]`` correspond to the true plant state
    ``state_true[k]`` (the state passed to the control callback). All
    components are SI: position [m], heading/yaw rate [rad]/[rad/s],
    speeds [m/s], current [m/s].
    """

    t: np.ndarray  # (N,) controller update times [s]
    state_true: np.ndarray  # (N, 6) true vessel state [x, y, psi, u, v, r]
    state_estimate: np.ndarray  # (N, 6) EKF vessel-state estimate
    current_true: np.ndarray  # (N, 2) true ambient current [V_cx, V_cy]
    current_estimate: np.ndarray  # (N, 2) EKF equivalent-current estimate


@dataclass(frozen=True)
class ControllerReferenceRun:
    """One controller's reference run (LOS baseline or NMPC).

    ``result`` is the plant history (truth model, applied post-actuator
    controls); ``estimator`` and ``command`` are the callback-aligned
    records at the controller period. NMPC-only fields (solve times, final
    statuses, recorded prediction horizons) are empty for the LOS run.
    """

    label: str  # human-readable controller name (for reports)
    period_s: float  # [s] controller period
    result: SimulationResult  # truth-plant history
    estimator: EstimatorHistory  # callback-aligned estimation records
    command: np.ndarray  # (N, 2) clamped commands [thrust N, yaw moment N m]
    solve_time_s: np.ndarray  # (N,) per-solve wall time [s] (zeros for LOS)
    solve_status: tuple[str, ...]  # final IPOPT status per solve (() for LOS)
    horizon: tuple[tuple[float, np.ndarray], ...]  # (t, (8, N+1) prediction)


@dataclass(frozen=True)
class ReferenceRun:
    """The flagship run: shared path plus three controller variants."""

    config: ReferenceScenarioConfig
    path: np.ndarray  # (M, 2) reference waypoints [m]
    los: ControllerReferenceRun
    nmpc: ControllerReferenceRun  # nominal, zero-disturbance prediction
    disturbance_aware_nmpc: ControllerReferenceRun


def default_reference_config() -> ReferenceScenarioConfig:
    """The canonical flagship configuration: exactly 120.0 s / 0.01 s / seed 42.

    Example:
        >>> from vessel_gnc.reference import default_reference_config
        >>> config = default_reference_config()
        >>> (config.duration_s, config.integration_dt_s, config.seed)
        (120.0, 0.01, 42)
    """
    return ReferenceScenarioConfig()


def run_reference_scenario(
    config: ReferenceScenarioConfig | None = None,
) -> ReferenceRun:
    """Run the flagship reference scenario with fresh per-run objects.

    Everything that carries state — sensor suites, EKFs, PID/PI controllers
    and the NMPC instance (including its warm start) — is constructed inside
    this call, so consecutive runs never share warm-start/controller/filter
    state. LOS and NMPC use separate RNGs initialized with the same seed.

    Args:
        config: scenario configuration (default: the canonical 120 s
            flagship, ``default_reference_config()``).

    Returns:
        A ReferenceRun with the shared path and three controller runs.

    Example:
        >>> from vessel_gnc.reference import run_reference_scenario
        >>> run = run_reference_scenario()  # doctest: +SKIP  (120 s flagship)
    """
    config = config if config is not None else default_reference_config()
    _validate_config(config)
    path = make_s_curve_path()
    los = _run_los(config, path)
    nmpc = _run_nmpc(config, path, disturbance_aware=False)
    disturbance_aware_nmpc = _run_nmpc(config, path, disturbance_aware=True)
    return ReferenceRun(
        config=config,
        path=path,
        los=los,
        nmpc=nmpc,
        disturbance_aware_nmpc=disturbance_aware_nmpc,
    )


def reference_metrics(run: ReferenceRun) -> dict[str, object]:
    """Deterministic flagship metrics (schema-shaped, docs/control.md §13).

    Per-controller metrics use the applied (post-actuator) histories and the
    physical actuator bounds of the truth plant (identical to the nominal
    bounds in the current parameter sets). Estimator errors are computed
    directly from the callback-aligned true/estimated records of the
    disturbance-aware NMPC run; the current-vector statistics discard the first
    ``config.estimator_transient_s`` seconds.

    Returns:
        A JSON-serializable dict ``{"controllers": {...}, "estimator": {...}}``
        with stable component IDs for LOS, nominal NMPC and
        disturbance-aware NMPC.

    Example:
        >>> from vessel_gnc.reference import run_reference_scenario, reference_metrics
        >>> metrics = reference_metrics(run_reference_scenario())  # doctest: +SKIP
    """
    controller_metrics = {}
    for component_id, controller in (
        (LOS_COMPONENT_ID, run.los),
        (NMPC_COMPONENT_ID, run.nmpc),
        (DISTURBANCE_AWARE_NMPC_COMPONENT_ID, run.disturbance_aware_nmpc),
    ):
        controller_metrics[component_id] = path_following_metrics(
            controller.result,
            run.path,
            run.config.lookahead_m,
            params=run.config.truth_params,
        )
    return {
        "controllers": controller_metrics,
        "estimator": _estimator_metrics(
            run.config,
            run.disturbance_aware_nmpc.estimator,
        ),
    }


# --- per-controller runs ----------------------------------------------------


def _run_los(
    config: ReferenceScenarioConfig, path: np.ndarray
) -> ControllerReferenceRun:
    """LOS baseline: PID heading + PI surge speed on EKF estimates."""
    heading = _core.HeadingController(
        config.los_heading_gains, config.los_heading_moment_limit_Nm
    )
    speed = _core.SpeedController(
        config.los_speed_gains, config.los_speed_thrust_limit_N
    )

    def policy(
        _t: float, xhat: _core.State, _prev: _core.Control, _ekf: VesselEKF
    ) -> _core.Control:
        (psi_los,) = los_heading(np.array([[xhat.x, xhat.y]]), path, config.lookahead_m)
        moment = heading.update(psi_los, xhat.psi, xhat.r, config.los_period_s)
        thrust = speed.update(config.speed_ref_m_s, xhat.u, config.los_period_s)
        return _core.Control(thrust=thrust, yaw_moment=moment)

    return _run_closed_loop(config, "LOS baseline", config.los_period_s, policy)


def _run_nmpc(
    config: ReferenceScenarioConfig,
    path: np.ndarray,
    *,
    disturbance_aware: bool,
) -> ControllerReferenceRun:
    """Mission-clock NMPC, with optional equivalent-current prediction."""
    nmpc = VesselNmpc(config.nominal_params, config.nmpc)

    def policy(
        t: float, xhat: _core.State, prev: _core.Control, ekf: VesselEKF
    ) -> _core.Control:
        refs, psi_refs = path_reference(
            path,
            config.speed_ref_m_s * t,
            config.speed_ref_m_s,
            nmpc.config.dt,
            nmpc.config.horizon,
        )
        # The model includes actuator states. The disturbance-aware variant
        # treats the EKF equivalent-current estimate as constant over the
        # finite horizon; nominal NMPC uses the exact zero-disturbance case.
        disturbance_estimate = (
            ekf.equivalent_current_estimate if disturbance_aware else None
        )
        return nmpc.solve(
            xhat,
            ekf.actuator,
            refs,
            psi_refs,
            prev,
            disturbance_estimate=disturbance_estimate,
        )

    label = "Disturbance-aware NMPC" if disturbance_aware else "Nominal NMPC"
    return _run_closed_loop(config, label, config.nmpc_period_s, policy, nmpc=nmpc)


def _run_closed_loop(
    config: ReferenceScenarioConfig,
    label: str,
    period_s: float,
    policy: ControllerPolicy,
    nmpc: VesselNmpc | None = None,
) -> ControllerReferenceRun:
    """Closed loop on EKF estimates for one controller (shared plumbing).

    Constructs a fresh sensor suite, EKF and filter-side actuator state for
    this run. The true environment is sampled deterministically at each
    controller update to record the callback-aligned current history.
    """
    params = config.nominal_params
    rng = np.random.default_rng(config.seed)
    sensors = SensorSuite(config.sensors, rng)
    r_cov = {name: config.sensors.covariance(name) for name in _SENSOR_NAMES}
    ekf = VesselEKF(params, dt=period_s)
    prev = _core.Control()

    t_rec: list[float] = []
    state_true: list[list[float]] = []
    state_estimate: list[list[float]] = []
    current_true: list[list[float]] = []
    current_estimate: list[list[float]] = []
    command: list[list[float]] = []
    solve_times: list[float] = []
    solve_status: list[str] = []
    horizon_shots: list[tuple[float, np.ndarray]] = []

    def callback(t: float, state: _core.State) -> _core.Control:
        nonlocal prev
        ekf.predict(prev)
        ekf.observe(sensors.sample(state, t), r_cov)
        xhat = ekf.estimate
        cmd = policy(t, xhat, prev, ekf)
        prev = _core.clamp_control(cmd, params)
        env = config.environment.sample(t)
        t_rec.append(t)
        state_true.append([state.x, state.y, state.psi, state.u, state.v, state.r])
        state_estimate.append([xhat.x, xhat.y, xhat.psi, xhat.u, xhat.v, xhat.r])
        current_true.append([env.current_north, env.current_east])
        equivalent_current = ekf.equivalent_current_estimate
        current_estimate.append(
            [equivalent_current.current_north, equivalent_current.current_east]
        )
        command.append([prev.thrust, prev.yaw_moment])
        if nmpc is not None:
            solve_times.append(nmpc.last_solve_time)
            solve_status.append(nmpc.last_status)
            horizon_shots.append((t, nmpc.last_trajectory.copy()))
        return prev

    result = simulate(
        config.duration_s,
        config.integration_dt_s,
        params=config.truth_params,
        control=callback,
        environment=config.environment.sample,
        control_period=period_s,
    )
    estimator = EstimatorHistory(
        t=np.asarray(t_rec),
        state_true=np.asarray(state_true),
        state_estimate=np.asarray(state_estimate),
        current_true=np.asarray(current_true),
        current_estimate=np.asarray(current_estimate),
    )
    return ControllerReferenceRun(
        label=label,
        period_s=period_s,
        result=result,
        estimator=estimator,
        command=np.asarray(command),
        solve_time_s=(
            np.asarray(solve_times) if nmpc is not None else np.zeros(len(t_rec))
        ),
        solve_status=tuple(solve_status),
        horizon=tuple(horizon_shots),
    )


# --- metrics -----------------------------------------------------------------


def _estimator_metrics(
    config: ReferenceScenarioConfig, history: EstimatorHistory
) -> dict[str, float]:
    """Full-run position/yaw-rate errors and post-transient current errors.

    Errors use the callback-aligned true/estimated records directly (no
    array slicing across different sampling rates).
    """
    if len(history.t) == 0:
        raise ValueError("estimator history is empty")
    pos_error = np.hypot(
        history.state_true[:, 0] - history.state_estimate[:, 0],
        history.state_true[:, 1] - history.state_estimate[:, 1],
    )
    yaw_rate_error = history.state_true[:, 5] - history.state_estimate[:, 5]
    current_error = np.hypot(
        history.current_true[:, 0] - history.current_estimate[:, 0],
        history.current_true[:, 1] - history.current_estimate[:, 1],
    )
    after_transient = history.t >= config.estimator_transient_s
    if not np.any(after_transient):
        raise ValueError(
            "run is shorter than the estimator transient "
            f"({config.estimator_transient_s:.1f} s); no post-transient samples"
        )
    return {
        "position_error_rms_m": float(np.sqrt(np.mean(pos_error**2))),
        "position_error_max_m": float(np.max(pos_error)),
        "yaw_rate_error_rms_rad_s": float(np.sqrt(np.mean(yaw_rate_error**2))),
        "current_error_rms_m_s": float(
            np.sqrt(np.mean(current_error[after_transient] ** 2))
        ),
        "current_error_max_m_s": float(np.max(current_error[after_transient])),
        "current_error_transient_s": float(config.estimator_transient_s),
    }


def _validate_config(config: ReferenceScenarioConfig) -> None:
    """Fail fast on configurations that cannot produce a finite run."""
    if config.duration_s <= 0.0 or config.integration_dt_s <= 0.0:
        raise ValueError("duration_s and integration_dt_s must be positive")
    if config.los_period_s <= 0.0 or config.nmpc_period_s <= 0.0:
        raise ValueError("los_period_s and nmpc_period_s must be positive")
    if config.speed_ref_m_s <= 0.0 or config.lookahead_m <= 0.0:
        raise ValueError("speed_ref_m_s and lookahead_m must be positive")
    if config.seed < 0:
        raise ValueError("seed must be a non-negative integer")
