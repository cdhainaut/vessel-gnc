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
from vessel_gnc.reference import (
    ReferenceScenarioConfig,
    default_reference_config,
    reference_metrics,
    run_reference_scenario,
)

# Short deterministic scenario: 2 s at 0.01 s, LOS 0.1 s, NMPC 0.2 s, with a
# reduced estimator transient so reference_metrics() has post-transient data.
SHORT = ReferenceScenarioConfig(duration_s=2.0, estimator_transient_s=0.5)

CONTROLLER_METRIC_KEYS = (
    "cross_track_rms_m",
    "cross_track_p95_m",
    "cross_track_max_m",
    "heading_error_rms_rad",
    "heading_error_max_rad",
    "thrust_rms_N",
    "thrust_max_N",
    "moment_rms_Nm",
    "moment_max_Nm",
    "thrust_saturation_duration_s",
    "moment_saturation_duration_s",
    "any_saturation_duration_s",
)
ESTIMATOR_METRIC_KEYS = (
    "position_error_rms_m",
    "position_error_max_m",
    "yaw_rate_error_rms_rad_s",
    "current_error_rms_m_s",
    "current_error_max_m_s",
    "current_error_transient_s",
)


def test_default_configuration_is_canonical():
    # The default configuration must be exactly the 120 s / 0.01 s / seed 42
    # flagship with the documented controller periods and path settings.
    config = default_reference_config()
    assert config.duration_s == 120.0
    assert config.integration_dt_s == 0.01
    assert config.seed == 42
    assert config.los_period_s == 0.1
    assert config.nmpc_period_s == 0.2
    assert config.speed_ref_m_s == 1.3
    assert config.lookahead_m == 8.0


def test_short_run_records_callback_aligned_histories():
    # A short run must exercise LOS and NMPC without globals and record the
    # full callback-aligned history at every controller update.
    run = run_reference_scenario(SHORT)
    assert run.los.label == "LOS baseline"
    assert run.nmpc.label == "Nominal NMPC"
    assert run.disturbance_aware_nmpc.label == "Disturbance-aware NMPC"

    los_t = np.arange(20) * SHORT.los_period_s  # 0.0 .. 1.9 s
    nmpc_t = np.arange(10) * SHORT.nmpc_period_s  # 0.0 .. 1.8 s
    # The recorded times are integration-step multiples (k * dt), which differ
    # from k * period in the last ulp: compare with a tight tolerance.
    np.testing.assert_allclose(run.los.estimator.t, los_t, atol=1e-12)
    np.testing.assert_allclose(run.nmpc.estimator.t, nmpc_t, atol=1e-12)
    np.testing.assert_allclose(
        run.disturbance_aware_nmpc.estimator.t,
        nmpc_t,
        atol=1e-12,
    )

    controllers = (run.los, run.nmpc, run.disturbance_aware_nmpc)
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

    # Commands respect the physical actuator bounds (clamped per update).
    p = SHORT.nominal_params
    for controller in controllers:
        assert np.all(controller.command[:, 0] >= p.thrust_min - 1e-12)
        assert np.all(controller.command[:, 0] <= p.thrust_max + 1e-12)
        assert np.all(controller.command[:, 1] >= p.moment_min - 1e-12)
        assert np.all(controller.command[:, 1] <= p.moment_max + 1e-12)

    # NMPC-only records: one solve time, status and horizon per update.
    for controller in (run.nmpc, run.disturbance_aware_nmpc):
        assert controller.solve_time_s.shape == (10,)
        assert np.all(controller.solve_time_s >= 0.0)
        assert len(controller.solve_status) == 10
        assert all(controller.solve_status)  # every solve ran
        assert len(controller.horizon) == 10
        assert controller.horizon[0][0] == pytest.approx(0.0)
        assert controller.horizon[0][1].shape == (8, SHORT.nmpc.horizon + 1)

    # LOS carries no NMPC records (empty tuples, zero solve times).
    assert run.los.solve_status == ()
    assert run.los.horizon == ()
    np.testing.assert_array_equal(run.los.solve_time_s, np.zeros(20))

    # Deterministic metrics are schema-shaped with callback-aligned errors.
    metrics = reference_metrics(run)
    assert set(metrics["controllers"]) == {
        "los_pid_v1",
        "nominal_nmpc_v1",
        "disturbance_aware_nmpc_v1",
    }
    for controller_metrics in metrics["controllers"].values():
        assert set(controller_metrics) == set(CONTROLLER_METRIC_KEYS)
        assert all(np.isfinite(value) for value in controller_metrics.values())
    assert set(metrics["estimator"]) == set(ESTIMATOR_METRIC_KEYS)
    assert (
        metrics["estimator"]["current_error_transient_s"] == SHORT.estimator_transient_s
    )
    assert all(np.isfinite(value) for value in metrics["estimator"].values())


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

    for controller_name in ("los", "nmpc", "disturbance_aware_nmpc"):
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
    for controller_name in ("nmpc", "disturbance_aware_nmpc"):
        first_controller = getattr(first, controller_name)
        second_controller = getattr(second, controller_name)
        for k, (_, trajectory) in enumerate(first_controller.horizon):
            np.testing.assert_allclose(
                trajectory,
                second_controller.horizon[k][1],
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
