"""Focused short-configuration tests for the reusable reference runner.

These tests prove that the flagship runner (python/vessel_gnc/reference.py)
holds no module-level mutable state: every ``run_reference_scenario`` call
constructs fresh sensor suites, EKFs, controllers and NMPC instances, so
short runs can be interleaved without warm-start/filter contamination, and
that the default configuration is exactly the canonical 120.0 s / 0.01 s /
seed 42 scenario. No 120 s flagship run is executed here.
"""

import numpy as np
import pytest
import vessel_gnc.reference as reference_module
from vessel_gnc.metrics import CONTROLLER_METRIC_KEYS
from vessel_gnc.reference import (
    ESTIMATOR_METRIC_KEYS,
    ReferenceScenarioConfig,
    default_reference_config,
    reference_metrics,
    run_reference_scenario,
)

# Short deterministic scenario: 2 s at 0.01 s, LOS 0.1 s, NMPC 0.2 s, with a
# reduced estimator transient so reference_metrics() has post-transient data.
SHORT = ReferenceScenarioConfig(duration_s=2.0, estimator_transient_s=0.5)


def test_default_configuration_is_canonical():
    # The default configuration must be exactly the 120 s / 0.01 s / seed 42
    # flagship with the documented controller periods and path settings.
    config = default_reference_config()
    assert config.duration_s == 120.0
    assert config.integration_dt_s == 0.01
    assert config.seed == 42
    assert config.estimator_period_s == 0.1
    assert config.los_period_s == 0.1
    assert config.nmpc_period_s == 0.2
    assert config.mpcc_period_s == 0.2
    assert config.speed_ref_m_s == 1.3
    assert config.lookahead_m == 8.0


def test_short_run_records_callback_aligned_histories_and_fair_mpcc_inputs(
    monkeypatch,
):
    # Capture the actual orchestration inputs while retaining real short
    # closed loops. Every policy receives the same PathGeometry object, and
    # MPCC receives only the callback's EKF state/current estimate.
    path_ids = []
    mpcc_states = []
    mpcc_disturbances = []
    sensor_residual_records = {}
    real_run_los = reference_module._run_los
    real_run_nmpc = reference_module._run_nmpc
    real_run_mpcc = reference_module._run_mpcc
    real_mpcc_solve = reference_module.VesselMpcc.solve
    real_sensor_sample = reference_module.SensorSuite.sample

    def run_los_spy(config, path):
        path_ids.append(id(path))
        return real_run_los(config, path)

    def run_nmpc_spy(config, path, *, disturbance_aware):
        path_ids.append(id(path))
        return real_run_nmpc(
            config,
            path,
            disturbance_aware=disturbance_aware,
        )

    def run_mpcc_spy(config, path):
        path_ids.append(id(path))
        return real_run_mpcc(config, path)

    def sensor_sample_spy(suite, state, t):
        measurements = real_sensor_sample(suite, state, t)
        residuals = {}
        for name, measurement in measurements.items():
            if name == "gnss":
                residuals[name] = measurement - [state.x, state.y]
            elif name == "compass":
                difference = measurement[0] - state.psi
                residuals[name] = np.array([np.arctan2(np.sin(difference), np.cos(difference))])
            elif name == "speed":
                residuals[name] = measurement - state.u
            elif name == "gyro":
                residuals[name] = measurement - state.r
        sensor_residual_records.setdefault(suite, []).append(
            (t, {name: value.copy() for name, value in residuals.items()})
        )
        return measurements

    def mpcc_solve_spy(
        controller,
        state,
        actuator,
        u_prev,
        *,
        disturbance_estimate=None,
    ):
        mpcc_states.append([state.x, state.y, state.psi, state.u, state.v, state.r])
        mpcc_disturbances.append(
            [
                disturbance_estimate.current_north,
                disturbance_estimate.current_east,
                disturbance_estimate.wind_north,
                disturbance_estimate.wind_east,
            ]
        )
        return real_mpcc_solve(
            controller,
            state,
            actuator,
            u_prev,
            disturbance_estimate=disturbance_estimate,
        )

    monkeypatch.setattr(reference_module, "_run_los", run_los_spy)
    monkeypatch.setattr(reference_module, "_run_nmpc", run_nmpc_spy)
    monkeypatch.setattr(reference_module, "_run_mpcc", run_mpcc_spy)
    monkeypatch.setattr(reference_module.VesselMpcc, "solve", mpcc_solve_spy)
    monkeypatch.setattr(reference_module.SensorSuite, "sample", sensor_sample_spy)

    run = run_reference_scenario(SHORT)
    assert run.los.label == "LOS baseline"
    assert run.nmpc.label == "Nominal NMPC"
    assert run.disturbance_aware_nmpc.label == "Disturbance-aware NMPC"
    assert run.disturbance_aware_mpcc.label == "Disturbance-aware MPCC"
    assert run.path_geometry is not None
    assert len(set(path_ids)) == 1
    assert path_ids[0] == id(run.path_geometry)
    sampled_path, _ = run.path_geometry.sample(SHORT.path_render_samples)
    np.testing.assert_array_equal(run.path, sampled_path)

    estimator_t = np.arange(20) * SHORT.estimator_period_s  # 0.0 .. 1.9 s
    predictive_t = np.arange(10) * SHORT.nmpc_period_s  # 0.0 .. 1.8 s
    # Sensor sampling and EKF updates are controller-independent: every run
    # uses the same 10 Hz timestamps, while predictive solves stay at 5 Hz.
    controllers = (
        run.los,
        run.nmpc,
        run.disturbance_aware_nmpc,
        run.disturbance_aware_mpcc,
    )
    for controller in controllers:
        np.testing.assert_allclose(controller.estimator.t, estimator_t, atol=1e-12)
    assert len(sensor_residual_records) == 4
    sensor_records = list(sensor_residual_records.values())
    for records in sensor_records:
        np.testing.assert_allclose(
            [record[0] for record in records],
            estimator_t,
            atol=1e-12,
        )
    for records in sensor_records[1:]:
        for reference_record, record in zip(sensor_records[0], records, strict=True):
            assert reference_record[1].keys() == record[1].keys()
            for sensor_name in reference_record[1]:
                np.testing.assert_allclose(
                    record[1][sensor_name],
                    reference_record[1][sensor_name],
                    atol=2e-14,
                    rtol=0.0,
                )
    assert run.disturbance_aware_mpcc.period_s == SHORT.mpcc_period_s

    for controller in controllers:
        n = len(controller.estimator.t)
        assert controller.estimator.state_true.shape == (n, 6)
        assert controller.estimator.state_estimate.shape == (n, 6)
        assert controller.estimator.current_true.shape == (n, 2)
        assert controller.estimator.current_estimate.shape == (n, 2)
        assert controller.command.shape == (n, 2)
        assert np.all(np.isfinite(controller.result.x))
        assert np.all(np.isfinite(controller.estimator.state_true))
        assert np.all(np.isfinite(controller.estimator.state_estimate))
        assert np.all(np.isfinite(controller.command))

    # Identically seeded, independent filter/sensor runs agree before their
    # controller actions diverge. All runs sample the true environment at the
    # common estimator cadence.
    for controller in controllers[1:]:
        np.testing.assert_array_equal(
            controller.estimator.current_true,
            run.nmpc.estimator.current_true,
        )
    predictive_initial_estimates = np.array(
        [controller.estimator.state_estimate[0] for controller in controllers[1:]]
    )
    np.testing.assert_array_equal(
        predictive_initial_estimates,
        np.tile(
            predictive_initial_estimates[0],
            (len(predictive_initial_estimates), 1),
        ),
    )

    # The MPCC spy saw exactly the EKF estimate and equivalent-current proxy,
    # never the callback's truth state or sampled truth environment.
    predictive_indices = np.arange(0, len(estimator_t), 2)
    np.testing.assert_allclose(
        mpcc_states,
        run.disturbance_aware_mpcc.estimator.state_estimate[predictive_indices],
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(mpcc_disturbances)[:, :2],
        run.disturbance_aware_mpcc.estimator.current_estimate[predictive_indices],
        atol=0.0,
        rtol=0.0,
    )
    np.testing.assert_array_equal(np.asarray(mpcc_disturbances)[:, 2:], 0.0)
    assert not np.allclose(
        np.asarray(mpcc_disturbances)[:, :2],
        run.disturbance_aware_mpcc.estimator.current_true[predictive_indices],
    )

    # Commands respect the physical actuator bounds (clamped per update).
    p = SHORT.nominal_params
    for controller in controllers:
        assert np.all(controller.command[:, 0] >= p.thrust_min - 1e-12)
        assert np.all(controller.command[:, 0] <= p.thrust_max + 1e-12)
        assert np.all(controller.command[:, 1] >= p.moment_min - 1e-12)
        assert np.all(controller.command[:, 1] <= p.moment_max + 1e-12)

    # Predictive records remain exactly 5 Hz despite the 10 Hz estimator.
    accepted_statuses = ("Solve_Succeeded", "Solved_To_Acceptable_Level")
    for controller in (
        run.nmpc,
        run.disturbance_aware_nmpc,
        run.disturbance_aware_mpcc,
    ):
        assert controller.solve_time_s.shape == (10,)
        assert np.all(controller.solve_time_s >= 0.0)
        assert len(controller.solve_status) == 10
        assert all(status in accepted_statuses for status in controller.solve_status)
        assert len(controller.horizon) == 10
        assert controller.horizon[0][0] == pytest.approx(0.0)
        np.testing.assert_allclose(
            [shot[0] for shot in controller.horizon],
            predictive_t,
            atol=1e-12,
        )
        np.testing.assert_allclose(
            controller.command[0::2],
            controller.command[1::2],
            atol=0.0,
            rtol=0.0,
        )
        assert np.all(np.isfinite([shot[1] for shot in controller.horizon]))
    assert run.nmpc.horizon[0][1].shape == (8, SHORT.nmpc.horizon + 1)
    assert run.disturbance_aware_nmpc.horizon[0][1].shape == (
        8,
        SHORT.nmpc.horizon + 1,
    )
    assert run.disturbance_aware_mpcc.horizon[0][1].shape == (
        8,
        SHORT.mpcc.horizon + 1,
    )

    # MPCC records its progress prediction alongside each physical horizon.
    progress_horizons = run.disturbance_aware_mpcc.progress_horizon
    assert len(progress_horizons) == 10
    assert all(shot[1].shape == (SHORT.mpcc.horizon + 1,) for shot in progress_horizons)
    progress_array = np.array([shot[1] for shot in progress_horizons])
    assert np.all(np.isfinite(progress_array))
    assert np.all(np.diff(progress_array, axis=1) >= -2e-6)
    assert np.all(progress_array >= -2e-6)
    assert np.all(progress_array <= run.path_geometry.length + 2e-6)
    assert np.all(np.diff(progress_array[:, 0]) >= -1e-12)
    progress_speed = np.diff(progress_array, axis=1) / SHORT.mpcc.dt
    assert np.all(progress_speed >= -5e-6)
    assert np.all(progress_speed <= SHORT.mpcc.progress_speed_max + 5e-6)

    # LOS carries no predictive-controller records.
    assert run.los.solve_status == ()
    assert run.los.horizon == ()
    assert run.los.progress_horizon == ()
    assert run.nmpc.progress_horizon == ()
    assert run.disturbance_aware_nmpc.progress_horizon == ()
    np.testing.assert_array_equal(run.los.solve_time_s, np.zeros(20))

    # Deterministic metrics are schema-shaped with callback-aligned errors.
    metrics = reference_metrics(run)
    assert set(metrics["controllers"]) == {
        "los_pid_v1",
        "nominal_nmpc_v1",
        "disturbance_aware_nmpc_v1",
        "disturbance_aware_mpcc_v1",
    }
    for controller_metrics in metrics["controllers"].values():
        assert set(controller_metrics) == set(CONTROLLER_METRIC_KEYS)
        finite_values = [value for value in controller_metrics.values() if value is not None]
        assert all(np.isfinite(value) for value in finite_values)
        assert controller_metrics["route_completion_s"] is None
    assert set(metrics["estimator"]) == set(ESTIMATOR_METRIC_KEYS)
    assert metrics["estimator"]["current_error_transient_s"] == SHORT.estimator_transient_s
    assert all(np.isfinite(value) for value in metrics["estimator"].values())


def test_rejected_mpcc_solve_keeps_last_accepted_horizons_at_original_time():
    config = ReferenceScenarioConfig(duration_s=0.4, estimator_transient_s=0.1)
    accepted_trajectory = np.arange(24, dtype=float).reshape(8, 3)
    accepted_progress = np.array([0.0, 0.5, 1.0])

    class AcceptedThenRejectedMpcc:
        def __init__(self):
            self.calls = 0
            self.last_solve_time = 0.0
            self.last_status = ""
            self.last_trajectory = None
            self.last_progress_trajectory = None

        @property
        def last_solve_succeeded(self):
            return self.last_status == "Solve_Succeeded"

        def solve(self):
            self.calls += 1
            self.last_solve_time = 0.001 * self.calls
            if self.calls == 1:
                self.last_status = "Solve_Succeeded"
                self.last_trajectory = accepted_trajectory.copy()
                self.last_progress_trajectory = accepted_progress.copy()
            else:
                self.last_status = "Maximum_Iterations_Exceeded"
                # Transactional MPCC semantics retain the accepted arrays.
            return reference_module._core.Control()

    controller = AcceptedThenRejectedMpcc()

    def policy(_t, _state, _previous, _ekf):
        return controller.solve()

    run = reference_module._run_closed_loop(
        config,
        "accepted then rejected MPCC",
        config.nmpc_period_s,
        policy,
        mpcc=controller,
    )

    assert controller.calls == 2
    assert run.solve_status == (
        "Solve_Succeeded",
        "Maximum_Iterations_Exceeded",
    )
    np.testing.assert_allclose(run.solve_time_s, [0.001, 0.002])
    assert len(run.horizon) == 1
    assert len(run.progress_horizon) == 1
    assert run.horizon[0][0] == pytest.approx(0.0)
    assert run.progress_horizon[0][0] == pytest.approx(0.0)
    np.testing.assert_array_equal(run.horizon[0][1], accepted_trajectory)
    np.testing.assert_array_equal(run.progress_horizon[0][1], accepted_progress)


def test_separate_runs_share_no_state():
    # Two separately constructed runners must not share warm-start,
    # controller or filter state: interleaving a different run between two
    # identical short runs leaves the second within the reproducibility
    # tolerance of the first. NMPC-influenced arrays are compared with
    # rtol/atol = 1e-6 because IPOPT (tol=1e-4, docs/control.md §5) can
    # legitimately return slightly different full-precision iterates between
    # runs; solve statuses are compared through the accepted-status set
    # (Solve_Succeeded / Solved_To_Acceptable_Level), since a run may
    # legitimately flip between two accepted statuses. Wall-clock solve
    # times are machine-dependent and excluded.
    other = ReferenceScenarioConfig(duration_s=1.0, estimator_transient_s=0.5)
    first = run_reference_scenario(SHORT)
    run_reference_scenario(other)
    second = run_reference_scenario(SHORT)

    for controller_name in (
        "los",
        "nmpc",
        "disturbance_aware_nmpc",
        "disturbance_aware_mpcc",
    ):
        first_result = getattr(first, controller_name).result
        second_result = getattr(second, controller_name).result
        for field in ("x", "y", "psi", "u", "v", "r", "thrust", "yaw_moment"):
            np.testing.assert_allclose(
                getattr(first_result, field),
                getattr(second_result, field),
                rtol=1e-6,
                atol=1e-6,
            )
        first_est = getattr(first, controller_name).estimator
        second_est = getattr(second, controller_name).estimator
        np.testing.assert_allclose(
            first_est.state_estimate,
            second_est.state_estimate,
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            first_est.current_estimate,
            second_est.current_estimate,
            rtol=1e-6,
            atol=1e-6,
        )
        np.testing.assert_allclose(
            getattr(first, controller_name).command,
            getattr(second, controller_name).command,
            rtol=1e-6,
            atol=1e-6,
        )
    # Warm starts are per-run: prediction horizons agree within tolerance.
    accepted = ("Solve_Succeeded", "Solved_To_Acceptable_Level")
    for controller_name in (
        "nmpc",
        "disturbance_aware_nmpc",
        "disturbance_aware_mpcc",
    ):
        first_controller = getattr(first, controller_name)
        second_controller = getattr(second, controller_name)
        for k, (_, trajectory) in enumerate(first_controller.horizon):
            np.testing.assert_allclose(
                trajectory,
                second_controller.horizon[k][1],
                rtol=1e-6,
                atol=1e-6,
            )
        if controller_name == "disturbance_aware_mpcc":
            for k, (_, progress) in enumerate(first_controller.progress_horizon):
                np.testing.assert_allclose(
                    progress,
                    second_controller.progress_horizon[k][1],
                    rtol=1e-6,
                    atol=1e-6,
                )
        # Compare accepted/rejected outcomes, not the two accepted raw strings.
        assert all(
            (status in accepted) == (second_status in accepted)
            for status, second_status in zip(
                first_controller.solve_status,
                second_controller.solve_status,
                strict=True,
            )
        )


def test_invalid_configuration_is_rejected():
    with pytest.raises(ValueError):
        run_reference_scenario(ReferenceScenarioConfig(duration_s=0.0))
    with pytest.raises(ValueError):
        run_reference_scenario(ReferenceScenarioConfig(lookahead_m=-1.0))
    with pytest.raises(ValueError):
        run_reference_scenario(ReferenceScenarioConfig(seed=-1))
    with pytest.raises(ValueError):
        run_reference_scenario(ReferenceScenarioConfig(path_render_samples=1))
    with pytest.raises(ValueError, match="integer multiple"):
        run_reference_scenario(ReferenceScenarioConfig(nmpc_period_s=0.15))
    with pytest.raises(ValueError, match="must match"):
        run_reference_scenario(ReferenceScenarioConfig(mpcc_period_s=0.3))
