"""Canonical comparison: LOS, two time-NMPC variants and geometric MPCC.

This module owns the entire flagship simulation (scenario
``scenario_v3_mpcc``): the smooth path geometry, environment, sensor suites,
EKFs and four fresh controller runs. ``examples/05_nmpc_demo.py`` is a thin
entry point that only renders the recorded run; nothing here is duplicated
there.

Every call to ``run_reference_scenario`` constructs fresh controllers,
filters, sensor suites, NMPC instances and one MPCC instance — there is no
module-level mutable state, so repeated calls (or two runners interleaved)
start clean. The four runs use separate RNGs initialized with the same seed,
so they receive matched sensor-noise realizations. The plant runs the same
perturbed truth parameters and environment in every run while each filter and
predictive controller uses the same nominal set (portfolio plan Phase C).

The estimator history is recorded at its fixed 10 Hz cadence, callback-aligned
with the true plant state. LOS updates at every estimator tick; each predictive
controller solves exactly at 5 Hz and its command is held between solves.
Estimator errors therefore use matched true/estimated pairs, never array
slicing.

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
from vessel_gnc.metrics import path_following_metrics
from vessel_gnc.mpcc import MPCC_COMPONENT_ID, MpccConfig, VesselMpcc
from vessel_gnc.nmpc import NmpcConfig, VesselNmpc
from vessel_gnc.path import PathGeometry, make_s_curve_geometry
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
    "ESTIMATOR_METRIC_KEYS",
]

# Metric-key contract of ``reference_metrics()["estimator"]`` (mirrors
# reference.schema.json #/$defs/estimatorMetrics).
ESTIMATOR_METRIC_KEYS = (
    "position_error_rms_m",
    "position_error_max_m",
    "yaw_rate_error_rms_rad_s",
    "current_error_rms_m_s",
    "current_error_max_m_s",
    "current_error_transient_s",
)

# Stable component IDs of the reference scenario (results/reference schema).
LOS_COMPONENT_ID = "los_pid_v1"
NMPC_COMPONENT_ID = "nominal_nmpc_v1"
DISTURBANCE_AWARE_NMPC_COMPONENT_ID = "disturbance_aware_nmpc_v1"

_SENSOR_NAMES = ("gnss", "compass", "speed", "gyro")

# Policy signature: (t, estimate, previously applied command, filter).
ControllerPolicy = Callable[[float, _core.State, _core.Control, VesselEKF], _core.Control]


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
    seed: int = 42  # sensor-noise seed (matched across all controller runs)
    estimator_period_s: float = 0.1  # [s] 10 Hz sensor sampling / EKF update
    los_period_s: float = 0.1  # [s] 10 Hz LOS command update
    nmpc_period_s: float = 0.2  # [s] 5 Hz time-NMPC solve / command update
    mpcc_period_s: float = 0.2  # [s] 5 Hz geometric-MPCC solve / command update
    speed_ref_m_s: float = 1.3  # [m/s] surge speed / mission-progress reference
    lookahead_m: float = 8.0  # [m] LOS lookahead in chord-progress
    environment: EnvironmentScenario = field(default_factory=EnvironmentScenario)
    sensors: SensorConfig = field(default_factory=SensorConfig)
    nmpc: NmpcConfig = field(default_factory=NmpcConfig)
    los_heading_gains: _core.PidGains = field(default_factory=_core.default_heading_gains)
    los_speed_gains: _core.PidGains = field(default_factory=_core.default_speed_gains)
    los_heading_moment_limit_Nm: float = 6.0  # [N m] heading-controller output limit
    los_speed_thrust_limit_N: float = 40.0  # [N] speed-controller output limit
    nominal_params: _core.ModelParams = field(default_factory=_core.default_params)
    truth_params: _core.ModelParams = field(default_factory=_core.truth_params)
    estimator_transient_s: float = 20.0  # [s] discarded before current-error stats
    render_fps: int = 12  # hero animation frame rate
    render_hero_stride_frames: int = 40  # hero frame period = stride * dt
    render_hero_wake_duration_s: float = 12.0  # [s] hero wake trail length
    mpcc: MpccConfig = field(default_factory=MpccConfig)
    path_render_samples: int = 501  # rendering-only smooth-path samples


@dataclass(frozen=True)
class EstimatorHistory:
    """Callback-aligned estimation records at every estimator update.

    Arrays share the row index ``k``: the estimate and held commanded control
    at time ``t[k]`` correspond to the true plant state ``state_true[k]`` (the
    state passed to the estimator callback). All
    components are SI: position [m], heading/yaw rate [rad]/[rad/s],
    speeds [m/s], current [m/s].
    """

    t: np.ndarray  # (N,) estimator update times [s]
    state_true: np.ndarray  # (N, 6) true vessel state [x, y, psi, u, v, r]
    state_estimate: np.ndarray  # (N, 6) EKF vessel-state estimate
    current_true: np.ndarray  # (N, 2) true ambient current [V_cx, V_cy]
    current_estimate: np.ndarray  # (N, 2) EKF equivalent-current estimate


@dataclass(frozen=True)
class ControllerReferenceRun:
    """One controller's reference run (LOS baseline or NMPC).

    ``result`` is the plant history (truth model, applied post-actuator
    controls); ``estimator`` and held ``command`` are callback-aligned records
    at the estimator period. Predictive solve arrays contain one entry per
    5 Hz solve. Prediction horizons contain accepted solves only and are empty
    for the LOS run.
    """

    label: str  # human-readable controller name (for reports)
    period_s: float  # [s] controller period
    result: SimulationResult  # truth-plant history
    estimator: EstimatorHistory  # callback-aligned estimation records
    command: np.ndarray  # (N, 2) clamped commands [thrust N, yaw moment N m]
    solve_time_s: np.ndarray  # (N_solve,) wall time [s] (zeros at EKF rate for LOS)
    solve_status: tuple[str, ...]  # final IPOPT status per solve (() for LOS)
    horizon: tuple[tuple[float, np.ndarray], ...]  # (t, (8, N+1) prediction)
    progress_horizon: tuple[tuple[float, np.ndarray], ...] = ()  # MPCC S [m]


@dataclass(frozen=True)
class ReferenceRun:
    """Matched controller runs on one authoritative smooth path geometry.

    ``path`` is a rendering-only dense sample retained for legacy plotting and
    artifact callers. All guidance, references and metrics in a newly executed
    run use ``path_geometry`` directly.
    """

    config: ReferenceScenarioConfig
    path: np.ndarray  # (M, 2) rendering-only smooth-path samples [m]
    los: ControllerReferenceRun
    nmpc: ControllerReferenceRun  # nominal, zero-disturbance prediction
    disturbance_aware_nmpc: ControllerReferenceRun
    disturbance_aware_mpcc: ControllerReferenceRun | None = None
    path_geometry: PathGeometry | None = None


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

    Everything that carries state — sensor suites, EKFs, PID/PI controllers,
    both NMPC instances and the MPCC instance (including warm starts) — is
    constructed inside this call, so consecutive runs never share controller
    or filter state. All four runs use separate RNGs initialized with the same
    seed.

    Args:
        config: scenario configuration (default: the canonical 120 s
            flagship, ``default_reference_config()``).

    Returns:
        A ReferenceRun with the shared geometry and four controller runs.

    Example:
        >>> from vessel_gnc.reference import run_reference_scenario
        >>> run = run_reference_scenario()  # doctest: +SKIP  (120 s flagship)
    """
    config = config if config is not None else default_reference_config()
    _validate_config(config)
    path_geometry = make_s_curve_geometry()
    path, _ = path_geometry.sample(config.path_render_samples)
    los = _run_los(config, path_geometry)
    nmpc = _run_nmpc(config, path_geometry, disturbance_aware=False)
    disturbance_aware_nmpc = _run_nmpc(
        config,
        path_geometry,
        disturbance_aware=True,
    )
    disturbance_aware_mpcc = _run_mpcc(config, path_geometry)
    return ReferenceRun(
        config=config,
        path=path,
        los=los,
        nmpc=nmpc,
        disturbance_aware_nmpc=disturbance_aware_nmpc,
        disturbance_aware_mpcc=disturbance_aware_mpcc,
        path_geometry=path_geometry,
    )


def reference_metrics(run: ReferenceRun) -> dict[str, object]:
    """Deterministic flagship metrics (schema-shaped, docs/control.md §6).

    Per-controller metrics use the applied (post-actuator) histories and the
    physical actuator bounds of the truth plant (identical to the nominal
    bounds in the current parameter sets). Estimator errors are computed
    directly from the callback-aligned true/estimated records of the
    disturbance-aware NMPC run; the current-vector statistics discard the first
    ``config.estimator_transient_s`` seconds.

    Returns:
        A JSON-serializable dict ``{"controllers": {...}, "estimator": {...}}``
        with stable component IDs for LOS, nominal NMPC, disturbance-aware
        NMPC and, for newly executed runs, disturbance-aware MPCC.

    Example:
        >>> from vessel_gnc.reference import run_reference_scenario, reference_metrics
        >>> metrics = reference_metrics(run_reference_scenario())  # doctest: +SKIP
    """
    controller_metrics = {}
    controllers = [
        (LOS_COMPONENT_ID, run.los),
        (NMPC_COMPONENT_ID, run.nmpc),
        (DISTURBANCE_AWARE_NMPC_COMPONENT_ID, run.disturbance_aware_nmpc),
    ]
    if run.disturbance_aware_mpcc is not None:
        controllers.append((MPCC_COMPONENT_ID, run.disturbance_aware_mpcc))
    metric_path = run.path_geometry if run.path_geometry is not None else run.path
    for component_id, controller in controllers:
        controller_metrics[component_id] = path_following_metrics(
            controller.result,
            metric_path,
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
    config: ReferenceScenarioConfig,
    path: PathGeometry,
) -> ControllerReferenceRun:
    """LOS baseline: PID heading + PI surge speed on EKF estimates."""
    heading = _core.HeadingController(config.los_heading_gains, config.los_heading_moment_limit_Nm)
    speed = _core.SpeedController(config.los_speed_gains, config.los_speed_thrust_limit_N)

    def policy(
        _t: float, xhat: _core.State, _prev: _core.Control, _ekf: VesselEKF
    ) -> _core.Control:
        (psi_los,) = path.los_heading(
            np.array([[xhat.x, xhat.y]]),
            config.lookahead_m,
        )
        moment = heading.update(psi_los, xhat.psi, xhat.r, config.los_period_s)
        thrust = speed.update(config.speed_ref_m_s, xhat.u, config.los_period_s)
        return _core.Control(thrust=thrust, yaw_moment=moment)

    return _run_closed_loop(config, "LOS baseline", config.los_period_s, policy)


def _run_nmpc(
    config: ReferenceScenarioConfig,
    path: PathGeometry,
    *,
    disturbance_aware: bool,
) -> ControllerReferenceRun:
    """Mission-clock NMPC, with optional equivalent-current prediction."""
    nmpc = VesselNmpc(config.nominal_params, config.nmpc)

    def policy(t: float, xhat: _core.State, prev: _core.Control, ekf: VesselEKF) -> _core.Control:
        # Legacy time-NMPC retains mission-clock progress, but both position
        # and heading references are evaluated on the exact same stored smooth
        # geometry as LOS and MPCC (never on a separate polyline).
        reference_progress = config.speed_ref_m_s * (
            t + nmpc.config.dt * np.arange(1, nmpc.config.horizon + 1)
        )
        refs = path.position(reference_progress)
        psi_refs = path.heading(reference_progress)
        # The model includes actuator states. The disturbance-aware variant
        # treats the EKF equivalent-current estimate as constant over the
        # finite horizon; nominal NMPC uses the exact zero-disturbance case.
        disturbance_estimate = ekf.equivalent_current_estimate if disturbance_aware else None
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


def _run_mpcc(
    config: ReferenceScenarioConfig,
    path: PathGeometry,
) -> ControllerReferenceRun:
    """Geometric MPCC using only EKF estimates, with no mission clock/reference."""
    mpcc = VesselMpcc(
        params=config.nominal_params,
        path=path,
        config=config.mpcc,
    )

    def policy(
        _t: float,
        xhat: _core.State,
        prev: _core.Control,
        ekf: VesselEKF,
    ) -> _core.Control:
        return mpcc.solve(
            state=xhat,
            actuator=ekf.actuator,
            u_prev=prev,
            disturbance_estimate=ekf.equivalent_current_estimate,
        )

    return _run_closed_loop(
        config,
        "Disturbance-aware MPCC",
        config.mpcc_period_s,
        policy,
        mpcc=mpcc,
    )


def _run_closed_loop(
    config: ReferenceScenarioConfig,
    label: str,
    period_s: float,
    policy: ControllerPolicy,
    nmpc: VesselNmpc | None = None,
    mpcc: VesselMpcc | None = None,
) -> ControllerReferenceRun:
    """Closed loop on EKF estimates for one controller (shared plumbing).

    Constructs a fresh sensor suite, EKF and filter-side actuator state for
    this run. Sensors and the EKF update at ``estimator_period_s`` in every
    controller run. The controller policy updates on its own ``period_s`` and
    its command is held between updates. The true environment is sampled
    deterministically at every estimator update for aligned current history.
    """
    params = config.nominal_params
    rng = np.random.default_rng(config.seed)
    sensors = SensorSuite(config.sensors, rng)
    r_cov = {name: config.sensors.covariance(name) for name in _SENSOR_NAMES}
    ekf = VesselEKF(params, dt=config.estimator_period_s)
    prev = _core.Control()
    last_policy_t = -np.inf

    t_rec: list[float] = []
    state_true: list[list[float]] = []
    state_estimate: list[list[float]] = []
    current_true: list[list[float]] = []
    current_estimate: list[list[float]] = []
    command: list[list[float]] = []
    solve_times: list[float] = []
    solve_status: list[str] = []
    horizon_shots: list[tuple[float, np.ndarray]] = []
    progress_horizon_shots: list[tuple[float, np.ndarray]] = []

    def callback(t: float, state: _core.State) -> _core.Control:
        nonlocal last_policy_t, prev
        ekf.predict(prev)
        ekf.observe(sensors.sample(state, t), r_cov)
        xhat = ekf.estimate
        policy_updated = t >= last_policy_t + period_s - 1e-9
        if policy_updated:
            cmd = policy(t, xhat, prev, ekf)
            prev = _core.clamp_control(cmd, params)
            last_policy_t = t
        env = config.environment.sample(t)
        t_rec.append(t)
        state_true.append([state.x, state.y, state.psi, state.u, state.v, state.r])
        state_estimate.append([xhat.x, xhat.y, xhat.psi, xhat.u, xhat.v, xhat.r])
        current_true.append([env.current_north, env.current_east])
        equivalent_current = ekf.equivalent_current_estimate
        current_estimate.append([equivalent_current.current_north, equivalent_current.current_east])
        command.append([prev.thrust, prev.yaw_moment])
        predictive_controller = nmpc if nmpc is not None else mpcc
        if predictive_controller is not None and policy_updated:
            solve_times.append(predictive_controller.last_solve_time)
            solve_status.append(predictive_controller.last_status)
            if nmpc is not None and nmpc.last_trajectory is not None:
                horizon_shots.append((t, nmpc.last_trajectory.copy()))
            if (
                mpcc is not None
                and mpcc.last_solve_succeeded
                and mpcc.last_trajectory is not None
                and mpcc.last_progress_trajectory is not None
            ):
                horizon_shots.append((t, mpcc.last_trajectory.copy()))
                progress_horizon_shots.append((t, mpcc.last_progress_trajectory.copy()))
        return prev

    result = simulate(
        config.duration_s,
        config.integration_dt_s,
        params=config.truth_params,
        control=callback,
        environment=config.environment.sample,
        control_period=config.estimator_period_s,
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
            np.asarray(solve_times)
            if nmpc is not None or mpcc is not None
            else np.zeros(len(t_rec))
        ),
        solve_status=tuple(solve_status),
        horizon=tuple(horizon_shots),
        progress_horizon=tuple(progress_horizon_shots),
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
        "current_error_rms_m_s": float(np.sqrt(np.mean(current_error[after_transient] ** 2))),
        "current_error_max_m_s": float(np.max(current_error[after_transient])),
        "current_error_transient_s": float(config.estimator_transient_s),
    }


def _validate_config(config: ReferenceScenarioConfig) -> None:
    """Fail fast on configurations that cannot produce a finite run."""
    if config.duration_s <= 0.0 or config.integration_dt_s <= 0.0:
        raise ValueError("duration_s and integration_dt_s must be positive")
    if (
        config.estimator_period_s <= 0.0
        or config.los_period_s <= 0.0
        or config.nmpc_period_s <= 0.0
        or config.mpcc_period_s <= 0.0
    ):
        raise ValueError(
            "estimator_period_s, los_period_s, nmpc_period_s and mpcc_period_s must be positive"
        )
    estimator_step_ratio = config.estimator_period_s / config.integration_dt_s
    if not np.isclose(
        estimator_step_ratio,
        round(estimator_step_ratio),
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("estimator_period_s must be an integer multiple of integration_dt_s")
    for name, period in (
        ("los_period_s", config.los_period_s),
        ("nmpc_period_s", config.nmpc_period_s),
        ("mpcc_period_s", config.mpcc_period_s),
    ):
        ratio = period / config.estimator_period_s
        if not np.isclose(ratio, round(ratio), rtol=0.0, atol=1e-12):
            raise ValueError(f"{name} must be an integer multiple of estimator_period_s")
    if not np.isclose(
        config.mpcc_period_s,
        config.nmpc_period_s,
        rtol=0.0,
        atol=1e-12,
    ):
        raise ValueError("nmpc_period_s and mpcc_period_s must match")
    sensor_periods = (
        config.sensors.gnss_period,
        config.sensors.compass_period,
        config.sensors.speed_period,
        config.sensors.gyro_period,
    )
    for period in sensor_periods:
        if period is None:
            continue
        if not np.isfinite(period) or period <= 0.0:
            raise ValueError("sensor periods must be finite and positive when enabled")
        ratio = period / config.estimator_period_s
        if ratio < 1.0 or not np.isclose(
            ratio,
            round(ratio),
            rtol=0.0,
            atol=1e-12,
        ):
            raise ValueError("sensor periods must be integer multiples of estimator_period_s")
    if config.speed_ref_m_s <= 0.0 or config.lookahead_m <= 0.0:
        raise ValueError("speed_ref_m_s and lookahead_m must be positive")
    if config.seed < 0:
        raise ValueError("seed must be a non-negative integer")
    if config.path_render_samples < 2:
        raise ValueError("path_render_samples must be at least two")
