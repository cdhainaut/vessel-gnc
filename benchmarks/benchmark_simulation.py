"""Reproducible performance numbers for the README (plan §19).

Run from the repository root:

    python benchmarks/benchmark_simulation.py

Reports the per-step cost of the C++ kernel through the binding, a 1000 s
simulation through the Python orchestration loop, and NMPC solve times.
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


def bench_kernel() -> float:
    """Per-step cost of the C++ kernel through the pybind11 binding [ns]."""
    params = _core.default_params()
    env = _core.Environment(current_east=0.15)
    cmd = _core.Control(thrust=25.0, yaw_moment=1.0)
    state = _core.State()
    n = 200_000
    t0 = time.perf_counter()
    for _ in range(n):
        state = _core.rk4_step(state, cmd, env, params, DT)
    elapsed = time.perf_counter() - t0
    return elapsed / n * 1e9


def bench_simulation() -> float:
    """Wall time of a 1000 s open-loop simulation through the Python loop [ms]."""
    t0 = time.perf_counter()
    simulate(1000.0, DT, control=_core.Control(thrust=25.0, yaw_moment=0.3))
    return (time.perf_counter() - t0) * 1e3


def bench_nmpc() -> tuple[float, float]:
    """Mean and p95 NMPC solve times over a 60 s closed loop [ms]."""
    path = make_s_curve_path()
    nmpc = VesselNmpc(_core.default_params())
    prev = _core.Control()
    actuator = _core.ActuatorState()
    times = []

    def policy(t: float, state: _core.State) -> _core.Control:
        nonlocal prev, actuator
        actuator = _core.actuator_step(actuator, prev, _core.default_params(), 0.2)
        refs, psi_refs = path_reference(
            path, 1.3 * t, 1.3, nmpc.config.dt, nmpc.config.horizon
        )
        cmd = nmpc.solve(state, actuator, refs, psi_refs, prev)
        times.append(nmpc.last_solve_time)
        prev = _core.clamp_control(cmd, _core.default_params())
        return prev

    simulate(60.0, DT, control=policy, control_period=CONTROL_PERIOD)
    return np.mean(times) * 1e3, np.percentile(times, 95) * 1e3


def main() -> None:
    ns = bench_kernel()
    sim_ms = bench_simulation()
    nmpc_mean, nmpc_p95 = bench_nmpc()

    print("vessel-gnc performance report (plan §19)")
    print(f"3-DOF RK4 propagation (C++ via binding): {ns:8.0f} ns/step")
    print(f"1,000 s simulation (Python loop):        {sim_ms:8.0f} ms")
    print(f"NMPC mean solve time:                    {nmpc_mean:8.0f} ms")
    print(f"NMPC p95 solve time:                     {nmpc_p95:8.0f} ms")
    print(f"NMPC 5 Hz step budget usage (mean):      {nmpc_mean / 200 * 100:8.0f} %")


if __name__ == "__main__":
    main()
