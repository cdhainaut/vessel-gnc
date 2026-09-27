"""Focused tests for geometric vessel MPCC (Milestone 5, Ticket M5-D)."""

import inspect

import casadi as ca
import numpy as np
import pytest
import vessel_gnc
from vessel_gnc import MpccConfig, PathGeometry, VesselMpcc, _core
from vessel_gnc.mpcc import MPCC_COMPONENT_ID, _contour_lag_errors
from vessel_gnc.nmpc import VesselNmpc
from vessel_gnc.path import make_s_curve_geometry

PARAMS = _core.default_params()
NORTH_PATH = np.array([[0.0, 0.0], [100.0, 0.0]])
EAST_PATH = np.array([[0.0, 0.0], [0.0, 100.0]])
MAX_PREDICTED_PROGRESS_LEAD_M = 2.0
# Behaviour tests lift the per-attempt IPOPT wall-time cap: the production
# real-time safety net would make accept/reject outcomes depend on machine
# load, which must not decide a deterministic unit test.
BEHAVIOUR_CONFIG = MpccConfig(solver_max_wall_time_s=5.0)


class _RejectedSolver:
    def __init__(self, decision: np.ndarray, status: str):
        self.decision = decision
        self.status = status

    def __call__(self, **_kwargs):
        return {"x": ca.DM(self.decision)}

    def stats(self):
        return {"return_status": self.status}


class _RaisingSolver:
    def __call__(self, **_kwargs):
        raise RuntimeError("deterministic rejected solve")


def _mpcc(
    path: np.ndarray = NORTH_PATH,
    *,
    params: _core.ModelParams = PARAMS,
    config: MpccConfig | None = None,
) -> VesselMpcc:
    return VesselMpcc(
        params=params,
        path=PathGeometry(path),
        config=config,
    )


def _predicted_terminal_progress_lead(controller: VesselMpcc) -> float:
    """Virtual minus geometric physical progress at the horizon end [m]."""
    physical_progress = controller.path.project(
        controller.last_trajectory[:2, -1]
    ).progress[0]
    return float(controller.last_progress_trajectory[-1] - physical_progress)


def _run_closed_loop(
    duration: float,
    *,
    initial_state: _core.State | None = None,
    environment: _core.Environment | None = None,
    disturbance_estimate: _core.Environment | None = None,
) -> tuple[_core.State, VesselMpcc, np.ndarray, list[str]]:
    """Short deterministic truth-state loop; no sensor or truth-data shortcut."""
    integration_dt = 0.05
    control_period = 0.4
    controller = _mpcc()
    state = initial_state if initial_state is not None else _core.State()
    actuator = _core.ActuatorState()
    command = _core.Control()
    truth_environment = environment if environment is not None else _core.Environment()
    progress_history = []
    statuses = []
    control_stride = round(control_period / integration_dt)

    for step in range(round(duration / integration_dt)):
        if step % control_stride == 0:
            command = controller.solve(
                state=state,
                actuator=actuator,
                u_prev=command,
                disturbance_estimate=disturbance_estimate,
            )
            progress_history.append(controller.accepted_progress)
            statuses.append(controller.last_status)
        actuator = _core.actuator_step(
            actuator,
            command,
            PARAMS,
            integration_dt,
        )
        state = _core.rk4_step(
            state,
            _core.Control(
                thrust=actuator.thrust,
                yaw_moment=actuator.yaw_moment,
            ),
            truth_environment,
            PARAMS,
            integration_dt,
        )

    return state, controller, np.asarray(progress_history), statuses


def test_public_defaults_api_and_component_id():
    config = MpccConfig()
    assert config == MpccConfig(
        horizon=25,
        dt=0.4,
        substeps=2,
        q_contour=8.0,
        q_lag=4.0,
        q_heading=1.5,
        q_progress=8.25,
        r_thrust=2e-3,
        r_moment=1e-2,
        s_thrust=5e-3,
        s_moment=5e-2,
        r_vs=0.5,
        s_vs=0.1,
        progress_speed_ref=1.3,
        progress_speed_max=2.0,
    )
    assert vessel_gnc.MpccConfig is MpccConfig
    assert vessel_gnc.VesselMpcc is VesselMpcc
    assert vessel_gnc.MPCC_COMPONENT_ID == "disturbance_aware_mpcc_v1"
    assert VesselMpcc.component_id == MPCC_COMPONENT_ID

    parameters = inspect.signature(VesselMpcc.solve).parameters
    assert tuple(parameters) == (
        "self",
        "state",
        "actuator",
        "u_prev",
        "disturbance_estimate",
    )
    assert "time" not in parameters
    assert "t" not in parameters
    assert "refs" not in parameters
    assert "reference" not in parameters


@pytest.mark.parametrize(
    ("keyword", "value"),
    [
        ("horizon", 0),
        ("substeps", 0),
        ("dt", 0.0),
        ("q_lag", -1.0),
        ("s_vs", np.nan),
        ("progress_speed_max", np.inf),
        ("progress_speed_ref", 2.1),
    ],
)
def test_config_rejects_invalid_or_nonfinite_values(keyword, value):
    with pytest.raises(ValueError):
        MpccConfig(**{keyword: value})


def test_decision_vector_layout_dimensions_and_bounds():
    controller = _mpcc()
    n_steps = controller.config.horizon
    x_size = 8 * (n_steps + 1)
    s_size = n_steps + 1
    u_size = 2 * n_steps
    vs_size = n_steps

    assert controller._x_slice == slice(0, x_size)
    assert controller._s_slice == slice(x_size, x_size + s_size)
    assert controller._u_slice == slice(x_size + s_size, x_size + s_size + u_size)
    assert controller._vs_slice == slice(
        x_size + s_size + u_size,
        x_size + s_size + u_size + vs_size,
    )
    assert controller.lbw.shape == (309,)
    assert controller.ubw.shape == (309,)
    assert controller.lbg.shape == (225,)
    assert controller.ubg.shape == (225,)

    x_lower = controller.lbw[controller._x_slice].reshape(
        8,
        n_steps + 1,
        order="F",
    )
    x_upper = controller.ubw[controller._x_slice].reshape(
        8,
        n_steps + 1,
        order="F",
    )
    assert np.all(np.isneginf(x_lower[:3]))
    assert np.all(np.isposinf(x_upper[:3]))
    np.testing.assert_array_equal(x_lower[3], -10.0)
    np.testing.assert_array_equal(x_upper[3], 10.0)
    np.testing.assert_array_equal(x_lower[4], -10.0)
    np.testing.assert_array_equal(x_upper[4], 10.0)
    np.testing.assert_array_equal(x_lower[5], -5.0)
    np.testing.assert_array_equal(x_upper[5], 5.0)
    np.testing.assert_array_equal(x_lower[6], PARAMS.thrust_min)
    np.testing.assert_array_equal(x_upper[6], PARAMS.thrust_max)
    np.testing.assert_array_equal(x_lower[7], PARAMS.moment_min)
    np.testing.assert_array_equal(x_upper[7], PARAMS.moment_max)
    np.testing.assert_array_equal(controller.lbw[controller._s_slice], 0.0)
    np.testing.assert_array_equal(
        controller.ubw[controller._s_slice],
        controller.path.length,
    )

    u_lower = controller.lbw[controller._u_slice].reshape(2, n_steps, order="F")
    u_upper = controller.ubw[controller._u_slice].reshape(2, n_steps, order="F")
    np.testing.assert_array_equal(u_lower[0], PARAMS.thrust_min)
    np.testing.assert_array_equal(u_upper[0], PARAMS.thrust_max)
    np.testing.assert_array_equal(u_lower[1], PARAMS.moment_min)
    np.testing.assert_array_equal(u_upper[1], PARAMS.moment_max)
    np.testing.assert_array_equal(controller.lbw[controller._vs_slice], 0.0)
    np.testing.assert_array_equal(
        controller.ubw[controller._vs_slice],
        controller.config.progress_speed_max,
    )


def test_contour_and_lag_signs_on_analytical_north_and_east_paths():
    north_position = ca.DM([12.0, -3.0])
    contour, lag = _contour_lag_errors(
        north_position,
        ca.DM([10.0, 0.0]),
        ca.DM([1.0, 0.0]),
    )
    assert float(contour) == pytest.approx(3.0)  # west is port when heading North
    assert float(lag) == pytest.approx(2.0)  # North is ahead

    east_position = ca.DM([2.0, 13.0])
    contour, lag = _contour_lag_errors(
        east_position,
        ca.DM([0.0, 10.0]),
        ca.DM([0.0, 1.0]),
    )
    assert float(contour) == pytest.approx(2.0)  # North is port heading East
    assert float(lag) == pytest.approx(3.0)  # East is ahead

    assert PathGeometry(NORTH_PATH).heading(20.0) == pytest.approx(0.0)
    assert PathGeometry(EAST_PATH).heading(20.0) == pytest.approx(np.pi / 2)


def test_solution_is_invariant_to_path_and_state_translation():
    translation = np.array([5000.0, -7000.0])
    base = _mpcc()
    translated = _mpcc(path=NORTH_PATH + translation)
    base_state = _core.State(x=10.0, y=2.0, psi=0.2, u=1.1)
    translated_state = _core.State(
        x=base_state.x + translation[0],
        y=base_state.y + translation[1],
        psi=base_state.psi,
        u=base_state.u,
    )
    actuator = _core.ActuatorState(thrust=15.0, yaw_moment=0.4)
    previous = _core.Control(thrust=15.0, yaw_moment=0.4)

    base_command = base.solve(base_state, actuator, previous)
    translated_command = translated.solve(translated_state, actuator, previous)

    assert base.last_solve_succeeded
    assert translated.last_solve_succeeded
    np.testing.assert_allclose(
        [translated_command.thrust, translated_command.yaw_moment],
        [base_command.thrust, base_command.yaw_moment],
        atol=1e-5,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        translated.last_trajectory[:2] - translation[:, None],
        base.last_trajectory[:2],
        atol=1e-5,
        rtol=0.0,
    )


def test_solution_is_invariant_to_heading_plus_integer_turns():
    heading_turns = 100
    heading_offset = heading_turns * 2.0 * np.pi
    base = _mpcc()
    wrapped_equivalent = _mpcc()
    actuator = _core.ActuatorState(thrust=15.0, yaw_moment=0.4)
    previous = _core.Control(thrust=15.0, yaw_moment=0.4)

    base_command = base.solve(
        _core.State(x=10.0, y=2.0, psi=0.2, u=1.1),
        actuator,
        previous,
    )
    equivalent_command = wrapped_equivalent.solve(
        _core.State(x=10.0, y=2.0, psi=0.2 + heading_offset, u=1.1),
        actuator,
        previous,
    )

    assert base.last_solve_succeeded
    assert wrapped_equivalent.last_solve_succeeded
    np.testing.assert_allclose(
        [equivalent_command.thrust, equivalent_command.yaw_moment],
        [base_command.thrust, base_command.yaw_moment],
        atol=1e-5,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        wrapped_equivalent.last_trajectory[2] - heading_offset,
        base.last_trajectory[2],
        atol=1e-5,
        rtol=0.0,
    )


def test_none_equals_zero_disturbance_and_shared_predictor_is_sensitive():
    state = _core.State(y=1.0, u=0.8)
    actuator = _core.ActuatorState(thrust=15.0)
    previous = _core.Control(thrust=15.0)
    nominal = _mpcc()
    explicit_zero = _mpcc()

    command_none = nominal.solve(state, actuator, previous)
    command_zero = explicit_zero.solve(
        state,
        actuator,
        previous,
        disturbance_estimate=_core.Environment(),
    )
    assert command_none.thrust == pytest.approx(command_zero.thrust, abs=1e-12)
    assert command_none.yaw_moment == pytest.approx(command_zero.yaw_moment, abs=1e-12)
    np.testing.assert_array_equal(
        nominal.last_trajectory,
        explicit_zero.last_trajectory,
    )

    x = np.array([2.0, -1.0, 0.4, 1.1, -0.2, 0.08, 25.0, 0.3])
    u = np.array([30.0, -0.5])
    zero_step = nominal.model_step(x, u)
    np.testing.assert_array_equal(
        zero_step,
        nominal.model_step(x, u, _core.Environment()),
    )
    current_step = nominal.model_step(
        x,
        u,
        _core.Environment(current_east=0.2),
    )
    assert np.linalg.norm(current_step - zero_step) > 1e-3


def test_disturbance_estimate_changes_solution_without_truth_input():
    state = _core.State(u=0.8)
    actuator = _core.ActuatorState(thrust=15.0)
    previous = _core.Control(thrust=15.0)
    calm = _mpcc()
    aware = _mpcc()
    calm_command = calm.solve(state, actuator, previous)
    aware_command = aware.solve(
        state,
        actuator,
        previous,
        disturbance_estimate=_core.Environment(current_east=0.25),
    )

    assert abs(aware_command.yaw_moment - calm_command.yaw_moment) > 0.1
    assert np.linalg.norm(aware.last_trajectory - calm.last_trajectory) > 1e-3
    solve_parameters = inspect.signature(VesselMpcc.solve).parameters
    assert "environment" not in solve_parameters
    assert "truth_environment" not in solve_parameters


def test_geometric_projection_is_pinned_monotone_and_bounded():
    controller = _mpcc()
    first = _core.State(x=20.0, y=3.0, u=0.5)
    controller.solve(first, _core.ActuatorState(), _core.Control())
    x0 = np.array([20.0, 3.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0])
    np.testing.assert_array_equal(controller.lbw[:8], x0)
    np.testing.assert_array_equal(controller.ubw[:8], x0)
    assert controller.lbw[controller._s_slice.start] == pytest.approx(20.0)
    assert controller.ubw[controller._s_slice.start] == pytest.approx(20.0)
    assert controller.accepted_progress == pytest.approx(20.0)
    assert controller.last_progress_trajectory[0] == pytest.approx(20.0)
    np.testing.assert_allclose(
        np.diff(controller.last_progress_trajectory),
        controller.config.dt * controller.last_progress_speed,
        atol=2e-6,
        rtol=0.0,
    )
    assert np.all(np.diff(controller.last_progress_trajectory) >= -1e-9)
    assert np.all(controller.last_progress_trajectory >= 0.0)
    assert np.all(controller.last_progress_trajectory <= controller.path.length)

    controller.solve(
        _core.State(x=10.0, y=0.0, u=0.5),
        _core.ActuatorState(),
        _core.Control(),
    )
    assert controller.accepted_progress == pytest.approx(20.0)
    assert controller.last_progress_trajectory[0] == pytest.approx(20.0)


def test_command_and_predicted_actuator_bounds_and_finite_validation():
    controller = _mpcc()
    command = controller.solve(
        _core.State(y=3.0),
        _core.ActuatorState(),
        _core.Control(),
    )
    assert PARAMS.thrust_min <= command.thrust <= PARAMS.thrust_max
    assert PARAMS.moment_min <= command.yaw_moment <= PARAMS.moment_max
    assert np.all(controller.last_controls[0] >= PARAMS.thrust_min)
    assert np.all(controller.last_controls[0] <= PARAMS.thrust_max)
    assert np.all(controller.last_controls[1] >= PARAMS.moment_min)
    assert np.all(controller.last_controls[1] <= PARAMS.moment_max)
    assert np.all(controller.last_trajectory[6] >= PARAMS.thrust_min)
    assert np.all(controller.last_trajectory[6] <= PARAMS.thrust_max)
    assert np.all(controller.last_trajectory[7] >= PARAMS.moment_min)
    assert np.all(controller.last_trajectory[7] <= PARAMS.moment_max)

    with pytest.raises(ValueError, match="finite"):
        controller.solve(
            _core.State(x=np.nan),
            _core.ActuatorState(),
            _core.Control(),
        )
    with pytest.raises(ValueError, match="finite"):
        controller.solve(
            _core.State(),
            _core.ActuatorState(),
            _core.Control(thrust=np.inf),
        )
    with pytest.raises(ValueError, match="finite"):
        controller.solve(
            _core.State(),
            _core.ActuatorState(),
            _core.Control(),
            disturbance_estimate=_core.Environment(current_north=np.nan),
        )
    with pytest.raises(ValueError, match="physical bounds"):
        controller.solve(
            _core.State(),
            _core.ActuatorState(thrust=PARAMS.thrust_max + 1.0),
            _core.Control(),
        )


def test_shifted_warm_start_moves_all_decision_blocks_independently():
    controller = _mpcc()
    controller.solve(
        _core.State(x=5.0, y=1.0, u=1.0),
        _core.ActuatorState(thrust=10.0),
        _core.Control(thrust=10.0),
    )
    old_x = controller.last_trajectory.copy()
    old_s = controller.last_progress_trajectory.copy()
    old_u = controller.last_controls.copy()
    old_vs = controller.last_progress_speed.copy()
    x0 = np.array([5.4, 0.9, 0.1, 1.0, 0.0, 0.0, 10.0, 0.0])
    progress = float(old_s[1])
    guess = controller._shifted_guess(x0=x0, progress=progress)
    n_steps = controller.config.horizon
    shifted_x = guess[controller._x_slice].reshape(8, n_steps + 1, order="F")
    shifted_s = guess[controller._s_slice]
    shifted_u = guess[controller._u_slice].reshape(2, n_steps, order="F")
    shifted_vs = guess[controller._vs_slice]

    np.testing.assert_array_equal(shifted_x[:, 0], x0)
    np.testing.assert_allclose(shifted_x[:, 1], old_x[:, 2], atol=1e-12)
    np.testing.assert_allclose(shifted_s[1], old_s[2], atol=1e-12)
    np.testing.assert_allclose(shifted_u[:, 0], old_u[:, 1], atol=1e-12)
    np.testing.assert_allclose(shifted_vs[0], old_vs[1], atol=1e-12)
    np.testing.assert_allclose(shifted_x[:, -1], old_x[:, -1], atol=1e-12)
    np.testing.assert_allclose(shifted_s[-1], old_s[-1], atol=1e-12)
    np.testing.assert_allclose(shifted_u[:, -1], old_u[:, -1], atol=1e-12)
    np.testing.assert_allclose(shifted_vs[-1], old_vs[-1], atol=1e-12)


def test_physical_rollout_fallback_guess_is_dynamically_consistent():
    controller = _mpcc()
    x0 = np.array([2.0, -1.0, 0.2, 0.8, 0.0, 0.0, 10.0, 0.0])
    control = np.array([20.0, 0.5])
    disturbance = np.array([0.1, -0.05, 1.0, -0.5])
    guess = controller._rollout_guess(
        x0=x0,
        progress=2.0,
        control=control,
        disturbance=disturbance,
    )
    n_steps = controller.config.horizon
    X = guess[controller._x_slice].reshape(8, n_steps + 1, order="F")
    S = guess[controller._s_slice]
    U = guess[controller._u_slice].reshape(2, n_steps, order="F")
    progress_speed = guess[controller._vs_slice]

    for k in range(n_steps):
        expected = np.array(
            controller.F(
                ca.DM(X[:, k]),
                ca.DM(U[:, k]),
                ca.DM(disturbance),
            )
        ).ravel()
        np.testing.assert_allclose(X[:, k + 1], expected, atol=1e-12, rtol=0.0)
    np.testing.assert_allclose(
        np.diff(S),
        controller.config.dt * progress_speed,
        atol=1e-12,
        rtol=0.0,
    )


def test_rejected_status_is_transactional_and_returns_bounded_previous_command():
    controller = _mpcc()
    controller.solve(
        _core.State(x=5.0, y=1.0, u=0.8),
        _core.ActuatorState(),
        _core.Control(),
    )
    accepted_progress = controller.accepted_progress
    progress_speed_anchor = controller._previous_progress_speed
    warm_start = controller.w0.copy()
    accepted_trajectory = controller.last_trajectory.copy()
    accepted_progress_trajectory = controller.last_progress_trajectory.copy()
    accepted_controls = controller.last_controls.copy()
    accepted_progress_speed = controller.last_progress_speed.copy()
    accepted_lbw = controller.lbw.copy()
    accepted_ubw = controller.ubw.copy()

    rejected_decision = warm_start.copy()
    rejected_decision[controller._s_slice.start + 1] = controller.path.length + 7.0
    rejected_decision[controller._u_slice.start] = PARAMS.thrust_max + 11.0
    rejected_decision[controller._vs_slice.start] = (
        controller.config.progress_speed_max + 3.0
    )
    controller.solver = _RejectedSolver(
        rejected_decision,
        "Maximum_Iterations_Exceeded",
    )
    previous = _core.Control(thrust=-10.0, yaw_moment=4.0)
    command = controller.solve(
        _core.State(x=30.0, y=0.0, u=0.8),
        _core.ActuatorState(),
        previous,
    )

    assert command.thrust == pytest.approx(previous.thrust)
    assert command.yaw_moment == pytest.approx(previous.yaw_moment)
    assert controller.last_status == "Maximum_Iterations_Exceeded"
    assert not controller.last_solve_succeeded
    assert controller.last_solve_time >= 0.0
    assert set(controller.last_attempt_statuses) == {"Maximum_Iterations_Exceeded"}
    assert controller.accepted_progress == accepted_progress
    assert controller._previous_progress_speed == progress_speed_anchor
    np.testing.assert_array_equal(controller.w0, warm_start)
    np.testing.assert_array_equal(controller.last_trajectory, accepted_trajectory)
    np.testing.assert_array_equal(
        controller.last_progress_trajectory,
        accepted_progress_trajectory,
    )
    np.testing.assert_array_equal(controller.last_controls, accepted_controls)
    np.testing.assert_array_equal(
        controller.last_progress_speed,
        accepted_progress_speed,
    )
    np.testing.assert_array_equal(controller.lbw, accepted_lbw)
    np.testing.assert_array_equal(controller.ubw, accepted_ubw)
    assert controller.last_attempted_progress_trajectory[1] == pytest.approx(
        controller.path.length + 7.0
    )
    assert controller.last_attempted_controls[0, 0] == pytest.approx(
        PARAMS.thrust_max + 11.0
    )
    assert controller.last_attempted_progress_speed[0] == pytest.approx(
        controller.config.progress_speed_max + 3.0
    )


def test_solver_exceptions_are_transactional_and_return_bounded_previous_command():
    controller = _mpcc()
    controller.solve(
        _core.State(x=5.0, y=1.0, u=0.8),
        _core.ActuatorState(),
        _core.Control(),
    )
    accepted_progress = controller.accepted_progress
    progress_speed_anchor = controller._previous_progress_speed
    warm_start = controller.w0.copy()
    accepted_trajectory = controller.last_trajectory.copy()
    accepted_progress_trajectory = controller.last_progress_trajectory.copy()
    accepted_controls = controller.last_controls.copy()
    accepted_progress_speed = controller.last_progress_speed.copy()
    accepted_lbw = controller.lbw.copy()
    accepted_ubw = controller.ubw.copy()

    controller.solver = _RaisingSolver()
    previous = _core.Control(thrust=-10.0, yaw_moment=4.0)
    command = controller.solve(
        _core.State(x=30.0, y=0.0, u=0.8),
        _core.ActuatorState(),
        previous,
    )

    assert command.thrust == pytest.approx(previous.thrust)
    assert command.yaw_moment == pytest.approx(previous.yaw_moment)
    assert controller.last_status == "Solver_Exception:RuntimeError"
    assert not controller.last_solve_succeeded
    assert controller.last_solve_time >= 0.0
    assert set(controller.last_attempt_statuses) == {"Solver_Exception:RuntimeError"}
    assert controller.last_attempted_decision is None
    assert controller.accepted_progress == accepted_progress
    assert controller._previous_progress_speed == progress_speed_anchor
    np.testing.assert_array_equal(controller.w0, warm_start)
    np.testing.assert_array_equal(controller.last_trajectory, accepted_trajectory)
    np.testing.assert_array_equal(
        controller.last_progress_trajectory,
        accepted_progress_trajectory,
    )
    np.testing.assert_array_equal(controller.last_controls, accepted_controls)
    np.testing.assert_array_equal(
        controller.last_progress_speed,
        accepted_progress_speed,
    )
    np.testing.assert_array_equal(controller.lbw, accepted_lbw)
    np.testing.assert_array_equal(controller.ubw, accepted_ubw)


def test_repeated_fresh_solve_is_deterministic_with_explicit_tolerance():
    controller = _mpcc()
    state = _core.State(x=10.0, y=2.0, psi=0.2, u=1.2)
    actuator = _core.ActuatorState(thrust=20.0, yaw_moment=0.5)
    previous = _core.Control(thrust=20.0, yaw_moment=0.5)
    first = controller.solve(state, actuator, previous)
    first_trajectory = controller.last_trajectory.copy()
    first_progress = controller.last_progress_trajectory.copy()
    controller.reset()
    second = controller.solve(state, actuator, previous)

    assert first.thrust == pytest.approx(second.thrust, abs=1e-5)
    assert first.yaw_moment == pytest.approx(second.yaw_moment, abs=1e-5)
    np.testing.assert_allclose(
        first_trajectory,
        controller.last_trajectory,
        atol=1e-5,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        first_progress,
        controller.last_progress_trajectory,
        atol=1e-5,
        rtol=0.0,
    )


def test_reset_clears_every_recorded_and_progress_state():
    controller = _mpcc()
    controller.solve(
        _core.State(x=5.0, y=1.0),
        _core.ActuatorState(),
        _core.Control(),
    )
    assert controller.w0 is not None
    assert controller.accepted_progress is not None
    assert controller.last_solve_succeeded
    controller.reset()

    assert controller.w0 is None
    assert controller.accepted_progress is None
    assert controller._previous_progress_speed is None
    assert controller.last_trajectory is None
    assert controller.last_progress_trajectory is None
    assert controller.last_controls is None
    assert controller.last_progress_speed is None
    assert controller.last_attempt_statuses == ()
    assert controller.last_attempted_decision is None
    assert controller.last_attempted_trajectory is None
    assert controller.last_attempted_progress_trajectory is None
    assert controller.last_attempted_controls is None
    assert controller.last_attempted_progress_speed is None
    assert controller.last_solve_time == 0.0
    assert controller.last_status == ""
    assert controller.last_solve_succeeded is False


def test_tuned_mpcc_matches_curved_path_relative_progress_without_racing():
    path = make_s_curve_geometry()
    start_progress = 25.0  # [m] start near cruise, before the first turn
    start_position = path.position(start_progress)
    integration_dt = 0.05  # [s]
    controller_period = 0.2  # [s] canonical 5 Hz predictive solve cadence
    duration = 8.0  # [s], long enough to traverse the first curved section
    truth_environment = _core.Environment(current_east=0.15, wind_east=1.0)
    # Deliberately distinct from truth; both predictors receive only this
    # explicit held estimate.
    disturbance_estimate = _core.Environment(current_east=0.10)

    def run_controller(
        controller: VesselNmpc | VesselMpcc,
    ) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, np.ndarray]:
        state = _core.State(
            x=float(start_position[0]),
            y=float(start_position[1]),
            psi=float(path.heading(start_progress)),
            u=1.2,
        )
        actuator = _core.ActuatorState(thrust=32.0)
        command = _core.Control(thrust=32.0)
        points = []
        statuses = []
        commands = []
        horizon_leads = []
        control_stride = round(controller_period / integration_dt)

        for step in range(round(duration / integration_dt)):
            time_s = step * integration_dt
            if step % control_stride == 0:
                if isinstance(controller, VesselNmpc):
                    reference_progress = start_progress + 1.3 * (
                        time_s
                        + controller.config.dt
                        * np.arange(1, controller.config.horizon + 1)
                    )
                    command = controller.solve(
                        state=state,
                        actuator=actuator,
                        refs=path.position(reference_progress),
                        psi_refs=path.heading(reference_progress),
                        u_prev=command,
                        disturbance_estimate=disturbance_estimate,
                    )
                else:
                    command = controller.solve(
                        state=state,
                        actuator=actuator,
                        u_prev=command,
                        disturbance_estimate=disturbance_estimate,
                    )
                    physical_terminal_progress = path.project(
                        controller.last_trajectory[:2, -1]
                    ).progress[0]
                    horizon_leads.append(
                        controller.last_progress_trajectory[-1]
                        - physical_terminal_progress
                    )
                statuses.append(controller.last_status)
                commands.append([command.thrust, command.yaw_moment])
            actuator = _core.actuator_step(
                actuator,
                command,
                PARAMS,
                integration_dt,
            )
            state = _core.rk4_step(
                state,
                _core.Control(
                    thrust=actuator.thrust,
                    yaw_moment=actuator.yaw_moment,
                ),
                truth_environment,
                PARAMS,
                integration_dt,
            )
            points.append([state.x, state.y])

        projection = path.project(np.asarray(points))
        tracking = np.array(
            [
                projection.progress[-1],
                np.sqrt(np.mean(projection.cross_track**2)),
                np.max(np.abs(projection.cross_track)),
            ]
        )
        return (
            tracking,
            tuple(statuses),
            np.asarray(commands),
            np.asarray(horizon_leads),
        )

    aware_tracking, aware_statuses, _aware_commands, _ = run_controller(
        VesselNmpc(PARAMS)
    )
    mpcc_controller = VesselMpcc(PARAMS, path)
    mpcc_tracking, mpcc_statuses, mpcc_commands, horizon_leads = run_controller(
        mpcc_controller
    )

    achieved_progress_ratio = (mpcc_tracking[0] - start_progress) / (
        aware_tracking[0] - start_progress
    )
    assert mpcc_controller.config.q_progress == pytest.approx(8.25)
    assert achieved_progress_ratio >= 0.98
    assert abs(path.heading(mpcc_tracking[0])) > 0.1
    assert mpcc_tracking[1] <= 1.05 * aware_tracking[1]
    assert mpcc_tracking[2] <= 1.10 * aware_tracking[2]
    assert len(aware_statuses) == len(mpcc_statuses) == 40
    assert set(aware_statuses) <= {"Solve_Succeeded", "Solved_To_Acceptable_Level"}
    assert set(mpcc_statuses) <= {"Solve_Succeeded", "Solved_To_Acceptable_Level"}
    assert np.max(horizon_leads) < 1.5
    assert np.all(mpcc_commands[:, 0] >= PARAMS.thrust_min)
    assert np.all(mpcc_commands[:, 0] <= PARAMS.thrust_max)
    assert np.all(mpcc_commands[:, 1] >= PARAMS.moment_min)
    assert np.all(mpcc_commands[:, 1] <= PARAMS.moment_max)


def test_short_calm_straight_path_makes_monotone_progress():
    state, controller, progress, statuses = _run_closed_loop(8.0)
    assert np.all(np.isfinite([state.x, state.y, state.psi, state.u, state.v, state.r]))
    assert state.x > 3.0
    assert abs(state.y) < 1e-6
    assert progress[-1] > progress[0] + 3.0
    assert np.all(np.diff(progress) >= -1e-12)
    assert (
        0.0
        <= _predicted_terminal_progress_lead(controller)
        < MAX_PREDICTED_PROGRESS_LEAD_M
    )
    assert all(
        status in ("Solve_Succeeded", "Solved_To_Acceptable_Level")
        for status in statuses
    )
    assert controller.last_solve_succeeded


def test_short_lateral_offset_converges_toward_path():
    state, controller, progress, statuses = _run_closed_loop(
        10.0,
        initial_state=_core.State(y=3.0),
    )
    assert abs(state.y) < 0.5
    assert progress[-1] > 5.0
    assert (
        0.0
        <= _predicted_terminal_progress_lead(controller)
        < MAX_PREDICTED_PROGRESS_LEAD_M
    )
    assert np.all(np.isfinite(controller.last_trajectory))
    assert all(
        status in ("Solve_Succeeded", "Solved_To_Acceptable_Level")
        for status in statuses
    )


def test_short_nonzero_current_case_is_finite_and_accepted():
    state, controller, progress, statuses = _run_closed_loop(
        8.0,
        environment=_core.Environment(current_east=0.20, wind_east=1.0),
        # Deliberately not the truth value: only this explicit estimate enters
        # the predictor, and it is held constant over each horizon.
        disturbance_estimate=_core.Environment(current_east=0.15),
    )
    values = np.array([state.x, state.y, state.psi, state.u, state.v, state.r])
    assert np.all(np.isfinite(values))
    assert np.all(np.isfinite(controller.last_trajectory))
    assert np.all(np.isfinite(controller.last_progress_trajectory))
    assert progress[-1] > progress[0]
    assert all(
        status in ("Solve_Succeeded", "Solved_To_Acceptable_Level")
        for status in statuses
    )


def test_stall_diagnostic_exposes_virtual_vs_achieved_progress():
    slow_actuator_params = _core.default_params()
    slow_actuator_params.thrust_rate_limit = 0.01
    slow_actuator_params.moment_rate_limit = 0.01
    controller = _mpcc(params=slow_actuator_params, config=BEHAVIOUR_CONFIG)
    controller.solve(
        _core.State(),
        _core.ActuatorState(),
        _core.Control(),
    )
    assert controller.last_solve_succeeded
    horizon_duration = controller.config.horizon * controller.config.dt
    achieved_progress_rate = (
        controller.last_trajectory[0, -1] - controller.last_trajectory[0, 0]
    ) / horizon_duration
    virtual_progress_rate = float(np.mean(controller.last_progress_speed))

    assert achieved_progress_rate < 0.01
    progress_lead = _predicted_terminal_progress_lead(controller)
    assert virtual_progress_rate > achieved_progress_rate
    # The tuned controller remains below 2 m virtual lead even when the
    # physical actuator is intentionally stalled. This is less than 10 % of
    # the 20 m maximum-speed horizon advance and still catches progress racing.
    assert 0.0 < progress_lead < MAX_PREDICTED_PROGRESS_LEAD_M


def test_hard_first_turn_instance_solves_within_budget():
    """Regression: the default MPCC must solve the t=33.8 hard instance.

    The first-turn sample of the ``bench_mpcc`` workload (progress ~42.6 m,
    entering the first S-curve corner) used to stall with ``r_vs=0.0`` and the
    default monotone barrier: both rollout guesses ended
    ``Maximum_WallTime_Exceeded`` with IPOPT's dual infeasibility stuck near
    ``1e-2`` (ill-conditioned virtual-speed Hessian block). The default
    ``r_vs = 0.5`` keeps that block positive definite and the adaptive
    barrier converges the turn-2 samples in ~10-25 iterations, so the same
    captured hard state now converges on the first attempt within the
    per-attempt wall-time budget.
    """
    controller = VesselMpcc(PARAMS, make_s_curve_geometry(), config=BEHAVIOUR_CONFIG)
    state = _core.State(
        x=41.10,
        y=6.52,
        psi=0.502,
        u=1.42,
        v=0.16,
        r=-0.050,
    )
    actuator = _core.ActuatorState(thrust=38.0, yaw_moment=-1.04)
    previous = _core.Control(thrust=38.78, yaw_moment=-1.08)

    command = controller.solve(
        state=state,
        actuator=actuator,
        u_prev=previous,
        disturbance_estimate=_core.Environment(current_east=0.15),
    )

    assert controller.config.r_vs == pytest.approx(0.5)
    assert controller.last_solve_succeeded
    assert controller.last_status == "Solve_Succeeded"
    assert controller.last_attempt_statuses == ("Solve_Succeeded",)
    assert controller.last_solve_time < 1.0
    assert np.all(np.isfinite(controller.last_trajectory))
    assert PARAMS.thrust_min <= command.thrust <= PARAMS.thrust_max
    assert PARAMS.moment_min <= command.yaw_moment <= PARAMS.moment_max


def test_path_end_clamps_progress_and_drives_virtual_speed_to_zero():
    controller = _mpcc(path=np.array([[0.0, 0.0], [20.0, 0.0]]))
    command = controller.solve(
        _core.State(x=20.0, u=0.2),
        _core.ActuatorState(),
        _core.Control(),
    )
    assert controller.last_solve_succeeded
    assert controller.accepted_progress == pytest.approx(controller.path.length)
    # Raw IPOPT diagnostics are not clipped; tolerate its accepted feasibility
    # residual instead of concealing the 1e-7-scale terminal-bound overshoot.
    assert np.all(controller.last_progress_trajectory <= controller.path.length + 2e-6)
    assert np.all(controller.last_progress_trajectory >= -2e-6)
    np.testing.assert_allclose(
        controller.last_progress_trajectory,
        controller.path.length,
        atol=1e-6,
        rtol=0.0,
    )
    assert np.max(controller.last_progress_speed) < 1e-6
    assert PARAMS.thrust_min <= command.thrust <= PARAMS.thrust_max
    assert PARAMS.moment_min <= command.yaw_moment <= PARAMS.moment_max
