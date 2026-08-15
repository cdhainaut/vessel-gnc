"""Tests for the CasADi NMPC (docs/control.md §5-§6)."""

import numpy as np
import pytest
from vessel_gnc import _core
from vessel_gnc.guidance import make_s_curve_path, path_reference
from vessel_gnc.nmpc import NmpcConfig, VesselNmpc
from vessel_gnc.simulation import simulate

PARAMS = _core.default_params()


def _nmpc(config: NmpcConfig | None = None) -> VesselNmpc:
    return VesselNmpc(PARAMS, config)


def _refs(nmpc: VesselNmpc, t: float, path):
    # Mission-schedule reference: s0 advances at the reference speed.
    return path_reference(path, 1.3 * t, 1.3, nmpc.config.dt, nmpc.config.horizon)


def _run_nmpc(duration, path, env, config=None, state0=None, speed_ref=1.3, period=0.2):
    nmpc = _nmpc(config)
    prev = _core.Control()
    actuator = _core.ActuatorState()
    solve_times = []

    def policy(t: float, state: _core.State) -> _core.Control:
        nonlocal prev, actuator
        # Controller-side actuator state, stepped at the control rate with the
        # previously applied command (nominal actuator model).
        actuator = _core.actuator_step(actuator, prev, PARAMS, period)
        refs, psi_refs = path_reference(
            path, speed_ref * t, speed_ref, nmpc.config.dt, nmpc.config.horizon
        )
        cmd = nmpc.solve(state, actuator, refs, psi_refs, prev)
        solve_times.append(nmpc.last_solve_time)
        prev = _core.clamp_control(cmd, PARAMS)
        return prev

    result = simulate(
        duration,
        0.01,
        control=policy,
        environment=env,
        control_period=period,
        state0=state0,
    )
    return result, nmpc, solve_times


def test_model_matches_cpp_kernel():
    # The CasADi prediction model (8 states: vessel + actuator) must reproduce
    # the C++ kernel composed over the same internal sub-steps: this is the
    # cross-validation of the deliberate model duplication.
    nmpc = _nmpc()
    rng = np.random.default_rng(0)
    h = nmpc.config.dt / nmpc.config.substeps
    worst = 0.0
    for _ in range(20):
        x = rng.uniform(
            [-20, -20, -3, 0.5, -0.5, -0.3, -10.0, -3.0],
            [20, 20, 3, 2, 0.5, 0.3, 40.0, 3.0],
        )
        u = rng.uniform([-10, -3], [40, 3])
        cas = nmpc.model_step(x, u)
        s = _core.State(x=x[0], y=x[1], psi=x[2], u=x[3], v=x[4], r=x[5])
        actuator = _core.ActuatorState(thrust=x[6], yaw_moment=x[7])
        cmd = _core.Control(thrust=u[0], yaw_moment=u[1])
        env = _core.Environment()
        for _ in range(nmpc.config.substeps):
            actuator = _core.actuator_step(actuator, cmd, PARAMS, h)
            applied = _core.Control(
                thrust=actuator.thrust, yaw_moment=actuator.yaw_moment
            )
            s = _core.rk4_step(s, applied, env, PARAMS, h)
        cpp = np.array(
            [s.x, s.y, s.psi, s.u, s.v, s.r, actuator.thrust, actuator.yaw_moment]
        )
        worst = max(worst, float(np.max(np.abs(cas - cpp))))
    assert worst < 1e-8


def test_control_within_bounds():
    path = make_s_curve_path()
    env = _core.Environment(current_east=0.15, wind_east=3.0)
    result, _, _ = _run_nmpc(60.0, path, env)
    assert np.all(result.thrust >= PARAMS.thrust_min - 1e-9)
    assert np.all(result.thrust <= PARAMS.thrust_max + 1e-9)
    assert np.all(result.yaw_moment >= PARAMS.moment_min - 1e-9)
    assert np.all(result.yaw_moment <= PARAMS.moment_max + 1e-9)


def test_straight_line_tracking():
    # Straight path, calm water: the vessel reaches cruise and stays on track.
    path = np.array([[0.0, 0.0], [150.0, 0.0]])
    result, _, _ = _run_nmpc(80.0, path, _core.Environment())
    assert np.all(np.isfinite(result.x))
    assert np.abs(result.y[-1]) < 0.5  # no lateral drift
    assert result.u[-1] == pytest.approx(1.3, abs=0.05)
    assert result.r[-1] == pytest.approx(0.0, abs=0.02)


def test_s_curve_regression():
    # S-curve with current and wind (unknown to the NMPC): tighter tracking
    # than the LOS baseline thanks to the receding horizon.
    path = make_s_curve_path()
    env = _core.Environment(current_east=0.15, wind_east=3.0)
    result, _, _ = _run_nmpc(60.0, path, env)
    from vessel_gnc.metrics import path_following_metrics

    m = path_following_metrics(result, path, 8.0)
    assert np.all(np.isfinite(result.x))
    assert m["cross_track_rms_m"] < 1.0
    assert m["cross_track_max_m"] < 3.0


def test_solve_time_budget():
    # 5 Hz control budget with margin: mean solve well under 0.2 s.
    path = make_s_curve_path()
    env = _core.Environment(current_east=0.15, wind_east=3.0)
    _, _, solve_times = _run_nmpc(60.0, path, env)
    mean = float(np.mean(solve_times))
    assert mean < 0.2
    assert np.percentile(solve_times, 95) < 0.4


def test_deterministic_solve():
    # Same inputs, same warm start: two solves must return the same first
    # control action to abs=1e-5. IPOPT runs with tol=1e-4
    # (docs/control.md §5), so bit-identical iterates are not part of the
    # reproducibility contract; 1e-5 is two orders of magnitude tighter
    # than the solver tolerance and far below any physical effect
    # (commands are O(1-60) N / N m).
    nmpc = _nmpc()
    state = _core.State(x=10.0, y=2.0, psi=0.2, u=1.2, v=0.0, r=0.0)
    path = make_s_curve_path()
    refs, psi_refs = _refs(nmpc, 0.0, path)
    actuator = _core.ActuatorState(thrust=20.0, yaw_moment=0.5)
    c1 = nmpc.solve(state, actuator, refs, psi_refs, _core.Control())
    nmpc.reset()  # identical starting point for both solves
    c2 = nmpc.solve(state, actuator, refs, psi_refs, _core.Control())
    assert c1.thrust == pytest.approx(c2.thrust, abs=1e-5)
    assert c1.yaw_moment == pytest.approx(c2.yaw_moment, abs=1e-5)


def test_warm_start_is_shifted_solution():
    nmpc = _nmpc()
    state = _core.State(x=5.0, y=1.0, psi=0.1, u=1.2, v=0.0, r=0.0)
    path = make_s_curve_path()
    refs, psi_refs = _refs(nmpc, 0.0, path)
    nmpc.solve(state, _core.ActuatorState(), refs, psi_refs, _core.Control())
    assert nmpc.last_trajectory is not None
    assert nmpc.last_trajectory.shape == (8, nmpc.config.horizon + 1)
    # The next guess is the previous solution shifted by one step, with the
    # new initial state pinned.
    moved = _core.State(x=5.3, y=1.1, psi=0.12, u=1.2, v=0.0, r=0.0)
    w = nmpc._shifted_guess(
        np.array([moved.x, moved.y, moved.psi, moved.u, moved.v, moved.r, 0.0, 0.0])
    )
    n = nmpc.config.horizon
    X = w[: 8 * (n + 1)].reshape(8, n + 1, order="F")
    np.testing.assert_allclose(
        X[:, 0], [5.3, 1.1, 0.12, 1.2, 0.0, 0.0, 0.0, 0.0], atol=1e-12
    )
    np.testing.assert_allclose(X[:, 1], nmpc.last_trajectory[:, 2], atol=1e-12)
    U = w[8 * (n + 1) :].reshape(2, n, order="F")
    np.testing.assert_allclose(U[:, 0], nmpc.last_controls[:, 1], atol=1e-12)


def test_last_solve_succeeded_interpretation():
    # The read-only success property must reflect the accepted IPOPT statuses
    # without changing solver settings: False before the first solve, True
    # after a real successful solve, and the accepted-status mapping is exact
    # (interpretation is checked directly on the recorded status).
    nmpc = _nmpc()
    assert nmpc.last_solve_succeeded is False  # no solve has run yet
    state = _core.State(x=5.0, y=1.0, psi=0.1, u=1.2, v=0.0, r=0.0)
    path = make_s_curve_path()
    refs, psi_refs = _refs(nmpc, 0.0, path)
    nmpc.solve(state, _core.ActuatorState(), refs, psi_refs, _core.Control())
    assert nmpc.last_solve_succeeded is True
    for status, expected in (
        ("Solve_Succeeded", True),
        ("Solved_To_Acceptable_Level", True),
        ("Maximum_Iterations_Exceeded", False),
        ("Restoration_Failed", False),
        ("", False),
    ):
        nmpc.last_status = status
        assert nmpc.last_solve_succeeded is expected
