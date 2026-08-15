"""Reproducible performance numbers for the README (plan §19) and for the
committed reference benchmark artifact (results/reference/benchmark.json).

Run from the repository root:

    python benchmarks/benchmark_simulation.py

Reports the per-step cost of the C++ kernel through the binding, a 1000 s
simulation through the Python orchestration loop, and nominal versus
disturbance-aware NMPC solve-time statistics over matched 60 s S-curve
workloads. All values are machine-dependent; the benchmark is distinct from
the 120 s reference scenario and contains no deterministic tracking metric.
"""

from __future__ import annotations

import time

import numpy as np
from vessel_gnc import _core
from vessel_gnc.guidance import make_s_curve_path, path_reference
from vessel_gnc.nmpc import VesselNmpc
from vessel_gnc.simulation import simulate

DT = 0.01  # [s]
CONTROL_PERIOD = 0.2  # [s] 5 Hz NMPC


def bench_kernel() -> dict[str, object]:
    """Per-step cost of the C++ kernel through the pybind11 binding.

    Returns a JSON-serializable record with the timing [ns/step] and the
    number of propagated steps.
    """
    params = _core.default_params()
    env = _core.Environment(current_east=0.15)
    cmd = _core.Control(thrust=25.0, yaw_moment=1.0)
    state = _core.State()
    n = 200_000
    t0 = time.perf_counter()
    for _ in range(n):
        state = _core.rk4_step(state, cmd, env, params, DT)
    elapsed = time.perf_counter() - t0
    return {
        "name": "cpp_rk4_propagation",
        "ns_per_step": elapsed / n * 1e9,
        "steps": n,
    }


def bench_simulation() -> dict[str, object]:
    """Wall time of a 1000 s open-loop simulation through the Python loop."""
    t0 = time.perf_counter()
    simulate(1000.0, DT, control=_core.Control(thrust=25.0, yaw_moment=0.3))
    return {
        "name": "python_orchestration_1000s",
        "duration_s": 1000.0,
        "wall_time_ms": (time.perf_counter() - t0) * 1e3,
    }


def bench_nmpc(*, disturbance_aware: bool = False) -> dict[str, object]:
    """NMPC timing over a matched 60 s constant-current closed loop [ms].

    Args:
        disturbance_aware: pass the known constant current as the explicit
            prediction parameter; false uses zero-disturbance prediction.

    Returns:
        Sample count, timing statistics and final solver statuses. No tracking
        metric is included (timings are machine-dependent).
    """
    params = _core.default_params()
    path = make_s_curve_path()
    nmpc = VesselNmpc(params)
    prev = _core.Control()
    actuator = _core.ActuatorState()
    environment = _core.Environment(current_east=0.15, wind_east=3.0)
    current_estimate = _core.Environment(current_east=0.15)
    times: list[float] = []
    statuses: list[str] = []
    failed = 0

    def policy(t: float, state: _core.State) -> _core.Control:
        nonlocal prev, actuator, failed
        # Controller-side actuator state, stepped at the control rate with the
        # previously applied command (nominal actuator model).
        actuator = _core.actuator_step(actuator, prev, params, CONTROL_PERIOD)
        refs, psi_refs = path_reference(
            path, 1.3 * t, 1.3, nmpc.config.dt, nmpc.config.horizon
        )
        cmd = nmpc.solve(
            state,
            actuator,
            refs,
            psi_refs,
            prev,
            disturbance_estimate=(current_estimate if disturbance_aware else None),
        )
        times.append(nmpc.last_solve_time)
        statuses.append(nmpc.last_status)
        failed += 0 if nmpc.last_solve_succeeded else 1
        prev = _core.clamp_control(cmd, params)
        return prev

    simulate(
        60.0,
        DT,
        control=policy,
        environment=environment,
        control_period=CONTROL_PERIOD,
    )
    times_ms = np.asarray(times) * 1e3
    return {
        "name": (
            "disturbance_aware_s_curve_nmpc_60s"
            if disturbance_aware
            else "nominal_s_curve_nmpc_60s"
        ),
        "duration_s": 60.0,
        "control_period_s": CONTROL_PERIOD,
        "samples": len(times),
        "mean_ms": float(np.mean(times_ms)),
        "median_ms": float(np.median(times_ms)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "max_ms": float(np.max(times_ms)),
        "failed_solves": failed,
        "final_status_histogram": {
            status: statuses.count(status) for status in sorted(set(statuses))
        },
    }


def run_benchmarks() -> dict[str, object]:
    """Structured JSON-serializable benchmark record for the generation tool.

    All values are machine-dependent (host, load, BLAS); the record contains
    no deterministic tracking metric. The generation tool adds the artifact
    envelope (``$schema``, ``artifact_type``, ``schema_version``,
    ``git_commit``).
    """
    return {
        "benchmark_id": "benchmark_v2",
        "workloads": {
            "kernel": bench_kernel(),
            "simulation": bench_simulation(),
            "nmpc_nominal": bench_nmpc(disturbance_aware=False),
            "nmpc_disturbance_aware": bench_nmpc(disturbance_aware=True),
        },
    }


def main() -> None:
    kernel = bench_kernel()
    simulation = bench_simulation()
    controllers = {
        "nominal": bench_nmpc(disturbance_aware=False),
        "disturbance-aware": bench_nmpc(disturbance_aware=True),
    }

    print("vessel-gnc performance report (machine-dependent, plan §19)")
    print(
        f"3-DOF RK4 propagation (C++ via binding): {kernel['ns_per_step']:8.0f} ns/step"
    )
    print(
        f"1,000 s simulation (Python loop):        {simulation['wall_time_ms']:8.0f} ms"
    )
    for label, nmpc in controllers.items():
        print(
            f"{label:17s} NMPC [ms]: mean {nmpc['mean_ms']:6.0f}, "
            f"median {nmpc['median_ms']:6.0f}, p95 {nmpc['p95_ms']:6.0f}, "
            f"max {nmpc['max_ms']:6.0f}"
        )
        statuses = ", ".join(
            f"{status}: {count}"
            for status, count in nmpc["final_status_histogram"].items()
        )
        print(
            f"  {nmpc['samples']} samples, {nmpc['failed_solves']} failed; {statuses}"
        )


if __name__ == "__main__":
    main()
