"""Reproducible performance numbers for the README (plan §19) and for the
committed reference benchmark artifact (results/reference/benchmark.json).

Run from the repository root:

    python benchmarks/benchmark_simulation.py

Reports the per-step cost of the C++ kernel through the binding, a 1000 s
simulation through the Python orchestration loop, and solve-time statistics
for nominal NMPC, disturbance-aware NMPC and disturbance-aware MPCC over
matched 60 s smooth S-curve workloads. All values are machine-dependent; the
benchmark is distinct from the 120 s reference scenario and contains no
deterministic tracking metric.
"""

from __future__ import annotations

import time

import numpy as np
from vessel_gnc import _core
from vessel_gnc.mpcc import VesselMpcc
from vessel_gnc.nmpc import VesselNmpc
from vessel_gnc.path import make_s_curve_geometry
from vessel_gnc.prediction import ACCEPTED_IPOPT_STATUSES
from vessel_gnc.simulation import simulate

DT = 0.01  # [s]
CONTROL_PERIOD = 0.2  # [s] 5 Hz predictive-controller cadence
CONTROL_BUDGET_MS = 200.0  # [ms] wall-clock budget at 5 Hz
BENCHMARK_DURATION_S = 60.0  # [s]


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


def _timing_record(
    name: str,
    solve_times_s: list[float],
    statuses: list[str],
) -> dict[str, object]:
    """Build one schema-shaped predictive-controller timing record."""
    if not solve_times_s or len(solve_times_s) != len(statuses):
        raise ValueError("solve times and statuses must have the same non-zero length")
    times_ms = np.asarray(solve_times_s, dtype=float) * 1e3
    if not np.all(np.isfinite(times_ms)) or np.any(times_ms <= 0.0):
        raise ValueError("solve times must be finite and positive")
    histogram = {status: statuses.count(status) for status in sorted(set(statuses))}
    return {
        "name": name,
        "duration_s": BENCHMARK_DURATION_S,
        "control_period_s": CONTROL_PERIOD,
        "control_budget_ms": CONTROL_BUDGET_MS,
        "samples": len(solve_times_s),
        "mean_ms": float(np.mean(times_ms)),
        "median_ms": float(np.median(times_ms)),
        "p95_ms": float(np.percentile(times_ms, 95)),
        "max_ms": float(np.max(times_ms)),
        "failed_solves": sum(status not in ACCEPTED_IPOPT_STATUSES for status in statuses),
        "final_status_histogram": histogram,
    }


def bench_nmpc(*, disturbance_aware: bool = False) -> dict[str, object]:
    """NMPC timing over a matched 60 s constant-current closed loop [ms].

    Both NMPC variants evaluate their mission-clock references on the same
    fixed smooth path used by MPCC. The aware workload receives the fixed
    known equivalent-current estimate; the nominal workload receives ``None``.
    """
    params = _core.default_params()
    path = make_s_curve_geometry()
    nmpc = VesselNmpc(params)
    previous_command = _core.Control()
    actuator = _core.ActuatorState()
    environment = _core.Environment(current_east=0.15, wind_east=3.0)
    known_estimate = _core.Environment(current_east=0.15)
    solve_times_s: list[float] = []
    statuses: list[str] = []

    def policy(t: float, state: _core.State) -> _core.Control:
        nonlocal actuator, previous_command
        actuator = _core.actuator_step(
            actuator,
            previous_command,
            params,
            CONTROL_PERIOD,
        )
        reference_progress = 1.3 * (t + nmpc.config.dt * np.arange(1, nmpc.config.horizon + 1))
        command = nmpc.solve(
            state=state,
            actuator=actuator,
            refs=path.position(reference_progress),
            psi_refs=path.heading(reference_progress),
            u_prev=previous_command,
            disturbance_estimate=(known_estimate if disturbance_aware else None),
        )
        solve_times_s.append(nmpc.last_solve_time)
        statuses.append(nmpc.last_status)
        previous_command = _core.clamp_control(command, params)
        return previous_command

    simulate(
        BENCHMARK_DURATION_S,
        DT,
        control=policy,
        environment=environment,
        control_period=CONTROL_PERIOD,
    )
    name = "disturbance_aware_s_curve_nmpc_60s" if disturbance_aware else "nominal_s_curve_nmpc_60s"
    return _timing_record(name, solve_times_s, statuses)


def bench_mpcc() -> dict[str, object]:
    """MPCC timing on the matched fixed-path, known-estimate workload [ms].

    The physical plant and callback cadence match both NMPC workloads. MPCC
    receives the same fixed equivalent-current estimate as disturbance-aware
    NMPC and never receives mission time or a reference trajectory.
    """
    params = _core.default_params()
    path = make_s_curve_geometry()
    mpcc = VesselMpcc(params=params, path=path)
    previous_command = _core.Control()
    actuator = _core.ActuatorState()
    environment = _core.Environment(current_east=0.15, wind_east=3.0)
    known_estimate = _core.Environment(current_east=0.15)
    solve_times_s: list[float] = []
    statuses: list[str] = []

    def policy(_t: float, state: _core.State) -> _core.Control:
        nonlocal actuator, previous_command
        actuator = _core.actuator_step(
            actuator,
            previous_command,
            params,
            CONTROL_PERIOD,
        )
        command = mpcc.solve(
            state=state,
            actuator=actuator,
            u_prev=previous_command,
            disturbance_estimate=known_estimate,
        )
        solve_times_s.append(mpcc.last_solve_time)
        statuses.append(mpcc.last_status)
        previous_command = _core.clamp_control(command, params)
        return previous_command

    simulate(
        BENCHMARK_DURATION_S,
        DT,
        control=policy,
        environment=environment,
        control_period=CONTROL_PERIOD,
    )
    return _timing_record(
        "disturbance_aware_s_curve_mpcc_60s",
        solve_times_s,
        statuses,
    )


def _assemble_benchmark(
    *,
    kernel: dict[str, object],
    simulation: dict[str, object],
    nominal_nmpc: dict[str, object],
    disturbance_aware_nmpc: dict[str, object],
    disturbance_aware_mpcc: dict[str, object],
) -> dict[str, object]:
    """Assemble the benchmark contract from already measured workloads."""
    return {
        "benchmark_id": "benchmark_v3",
        "workloads": {
            "kernel": kernel,
            "simulation": simulation,
            "nmpc_nominal": nominal_nmpc,
            "nmpc_disturbance_aware": disturbance_aware_nmpc,
            "mpcc_disturbance_aware": disturbance_aware_mpcc,
        },
    }


def run_benchmarks() -> dict[str, object]:
    """Run and return all machine-dependent benchmark workloads."""
    return _assemble_benchmark(
        kernel=bench_kernel(),
        simulation=bench_simulation(),
        nominal_nmpc=bench_nmpc(disturbance_aware=False),
        disturbance_aware_nmpc=bench_nmpc(disturbance_aware=True),
        disturbance_aware_mpcc=bench_mpcc(),
    )


def main() -> None:
    report = run_benchmarks()
    kernel = report["workloads"]["kernel"]
    simulation = report["workloads"]["simulation"]
    controllers = {
        "nominal NMPC": report["workloads"]["nmpc_nominal"],
        "aware NMPC": report["workloads"]["nmpc_disturbance_aware"],
        "aware MPCC": report["workloads"]["mpcc_disturbance_aware"],
    }

    print("vessel-gnc performance report (machine-dependent, plan §19)")
    print(f"3-DOF RK4 propagation (C++ via binding): {kernel['ns_per_step']:8.0f} ns/step")
    print(f"1,000 s simulation (Python loop):        {simulation['wall_time_ms']:8.0f} ms")
    for label, controller in controllers.items():
        print(
            f"{label:17s} [ms]: mean {controller['mean_ms']:6.0f}, "
            f"median {controller['median_ms']:6.0f}, "
            f"p95 {controller['p95_ms']:6.0f}, max {controller['max_ms']:6.0f}"
        )
        statuses = ", ".join(
            f"{status}: {count}" for status, count in controller["final_status_histogram"].items()
        )
        print(
            f"  {controller['samples']} samples, "
            f"{controller['failed_solves']} failed, "
            f"{controller['control_budget_ms']:.0f} ms budget; {statuses}"
        )


if __name__ == "__main__":
    main()
