"""Cheap unit seams for the predictive-controller benchmark contract.

No test in this module runs a 60 s workload, the 1000 s simulation or the C++
kernel loop. Fake one-callback simulations verify workload wiring only.
"""

from types import SimpleNamespace

import numpy as np
import pytest
from vessel_gnc import _core

import benchmarks.benchmark_simulation as benchmark


def test_timing_record_reports_statistics_failures_histogram_and_budget():
    record = benchmark._timing_record(
        "synthetic_predictive_workload",
        [0.001, 0.002, 0.004],
        [
            "Solve_Succeeded",
            "Solved_To_Acceptable_Level",
            "Maximum_Iterations_Exceeded",
        ],
    )

    assert record["duration_s"] == 60.0
    assert record["control_period_s"] == 0.2
    assert record["control_budget_ms"] == 200.0
    assert record["samples"] == 3
    assert record["mean_ms"] == pytest.approx(7.0 / 3.0)
    assert record["median_ms"] == pytest.approx(2.0)
    assert record["p95_ms"] == pytest.approx(np.percentile([1.0, 2.0, 4.0], 95))
    assert record["max_ms"] == pytest.approx(4.0)
    assert record["failed_solves"] == 1
    assert record["final_status_histogram"] == {
        "Maximum_Iterations_Exceeded": 1,
        "Solve_Succeeded": 1,
        "Solved_To_Acceptable_Level": 1,
    }


@pytest.mark.parametrize(
    ("times", "statuses"),
    [([], []), ([0.001], []), ([0.0], ["Solve_Succeeded"])],
)
def test_timing_record_rejects_invalid_samples(times, statuses):
    with pytest.raises(ValueError):
        benchmark._timing_record("invalid", times, statuses)


def test_benchmark_assembly_is_v3_and_has_distinct_mpcc_workload():
    workloads = {
        "kernel": {"kind": "kernel"},
        "simulation": {"kind": "simulation"},
        "nominal_nmpc": {"kind": "nominal"},
        "disturbance_aware_nmpc": {"kind": "aware"},
        "disturbance_aware_mpcc": {"kind": "mpcc"},
    }
    document = benchmark._assemble_benchmark(**workloads)

    assert document["benchmark_id"] == "benchmark_v3"
    assert document["workloads"] == {
        "kernel": workloads["kernel"],
        "simulation": workloads["simulation"],
        "nmpc_nominal": workloads["nominal_nmpc"],
        "nmpc_disturbance_aware": workloads["disturbance_aware_nmpc"],
        "mpcc_disturbance_aware": workloads["disturbance_aware_mpcc"],
    }


def _one_callback_simulation(captured):
    def fake_simulate(
        duration,
        dt,
        *,
        control,
        environment,
        control_period,
        **_kwargs,
    ):
        captured["duration"] = duration
        captured["dt"] = dt
        captured["environment"] = environment
        captured["control_period"] = control_period
        control(0.0, _core.State())

    return fake_simulate


def test_mpcc_workload_uses_fixed_geometry_and_known_equivalent_estimate(monkeypatch):
    captured = {}
    path = object()

    class FakeMpcc:
        def __init__(self, *, params, path):
            captured["params"] = params
            captured["path"] = path
            self.last_solve_time = 0.003
            self.last_status = "Solve_Succeeded"

        def solve(self, **kwargs):
            captured["solve_kwargs"] = kwargs
            return _core.Control(thrust=1.0)

    monkeypatch.setattr(benchmark, "make_s_curve_geometry", lambda: path)
    monkeypatch.setattr(benchmark, "VesselMpcc", FakeMpcc)
    monkeypatch.setattr(benchmark, "simulate", _one_callback_simulation(captured))

    record = benchmark.bench_mpcc()

    assert captured["path"] is path
    assert captured["duration"] == 60.0
    assert captured["dt"] == 0.01
    assert captured["control_period"] == 0.2
    assert captured["environment"].current_east == pytest.approx(0.15)
    assert captured["environment"].wind_east == pytest.approx(3.0)
    assert set(captured["solve_kwargs"]) == {
        "state",
        "actuator",
        "u_prev",
        "disturbance_estimate",
    }
    estimate = captured["solve_kwargs"]["disturbance_estimate"]
    assert estimate.current_east == pytest.approx(0.15)
    assert estimate.wind_east == pytest.approx(0.0)
    assert record["name"] == "disturbance_aware_s_curve_mpcc_60s"
    assert record["samples"] == 1
    assert record["mean_ms"] == pytest.approx(3.0)


@pytest.mark.parametrize(
    ("disturbance_aware", "expected_estimate_east"),
    [(False, None), (True, 0.15)],
)
def test_nmpc_workloads_use_same_smooth_geometry_and_estimate_semantics(
    monkeypatch,
    disturbance_aware,
    expected_estimate_east,
):
    captured = {}

    class FakePath:
        def position(self, progress):
            captured["reference_progress"] = np.asarray(progress).copy()
            return np.column_stack([progress, np.zeros_like(progress)])

        def heading(self, progress):
            return np.zeros_like(progress)

    path = FakePath()

    class FakeNmpc:
        def __init__(self, params):
            captured["params"] = params
            self.config = SimpleNamespace(dt=0.4, horizon=2)
            self.last_solve_time = 0.002
            self.last_status = "Solve_Succeeded"

        def solve(self, **kwargs):
            captured["solve_kwargs"] = kwargs
            return _core.Control(thrust=1.0)

    monkeypatch.setattr(benchmark, "make_s_curve_geometry", lambda: path)
    monkeypatch.setattr(benchmark, "VesselNmpc", FakeNmpc)
    monkeypatch.setattr(benchmark, "simulate", _one_callback_simulation(captured))

    record = benchmark.bench_nmpc(disturbance_aware=disturbance_aware)

    np.testing.assert_allclose(captured["reference_progress"], [0.52, 1.04])
    estimate = captured["solve_kwargs"]["disturbance_estimate"]
    if expected_estimate_east is None:
        assert estimate is None
    else:
        assert estimate.current_east == pytest.approx(expected_estimate_east)
        assert estimate.wind_east == pytest.approx(0.0)
    assert record["samples"] == 1
