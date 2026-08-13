# Validation record

This page is the **index** of every validation case in the repository. Each
case is automated (see the linked test files); the detailed formulations,
conventions and limitations live in the linked documentation — nothing is
duplicated here.

Status legend:

- **verified implementation** — the code does what the equations say
  (analytical, consistency and convergence checks);
- **validated physical model** — behaviour compared against known physics;
- **illustrative model** — the parameter values are order-of-magnitude, not
  identified from a real vessel (docs/model.md §6-§7).

## Kinematics and dynamics (3-DOF model)

| Case | Status | Location |
|---|---|---|
| Rotation matrix orthonormal, pure-surge kinematics | verified | `tests/test_dynamics.cpp` |
| Zero-force, zero-damping inertial motion (plan case A) | validated | `tests/test_dynamics.cpp`, `tests/test_integrator.cpp` |
| Damping sign, surge equilibrium (plan case B) | validated | `tests/test_dynamics.cpp`, `tests/test_integrator.cpp` |
| Coriolis conserves kinetic energy; steady turn with port sideslip (plan case C) | validated | `tests/test_dynamics.cpp`, `tests/test_integrator.cpp` |
| RK4 convergence O(dt⁴) vs analytical solution (plan case D) | verified | `tests/test_integrator.cpp` |
| Relative-velocity current model (drag along, co-moving equilibrium) | validated | `tests/test_dynamics.cpp` |
| Wind force through the rotation matrix | verified | `tests/test_dynamics.cpp` |
| Actuator saturation | verified | `tests/test_dynamics.cpp` |
| Determinism, finiteness over long runs | verified | `tests/test_integrator.cpp` |
| Parameter set is illustrative, not identified | illustrative | docs/model.md §6 |

## Baseline control (LOS + PID/PI)

| Case | Status | Location |
|---|---|---|
| `wrap_to_pi`, derivative-on-measurement, anti-windup, saturation | verified | `tests/test_controllers.cpp` |
| Closed-loop heading step and heading hold under current | validated | `tests/test_controllers.cpp` |
| LOS projection sign convention, clamping, lookahead bearing | verified | `tests/test_guidance.py` |
| Path-following regression (S-curve, current + wind) | validated | `tests/test_guidance.py` |

## Estimation (EKF)

| Case | Status | Location |
|---|---|---|
| Sensor schedules, noise statistics, compass wrapping | verified | `tests/test_ekf.py` |
| Covariance symmetry, positive definiteness, finiteness | verified | `tests/test_ekf.py` |
| No-noise consistency (exact model → estimate converges) | verified | `tests/test_ekf.py` |
| Closed-loop tracking on estimates with unknown current/wind | validated | `tests/test_ekf.py` |

## Optimal control (NMPC)

| Case | Status | Location |
|---|---|---|
| CasADi model vs C++ RK4 kernel (cross-validation of the duplication) | verified | `tests/test_nmpc.py` |
| Actuator bounds respected, straight-line tracking | verified | `tests/test_nmpc.py` |
| S-curve regression with unknown current/wind | validated | `tests/test_nmpc.py` |
| Solve-time budget, determinism, warm-start shift | verified | `tests/test_nmpc.py` |
| Flagship comparison (NMPC vs LOS, EKF in the loop) | validated | `examples/05_nmpc_demo.py` → `results/comparison_metrics.json` |

## Performance

| Measurement | Result | Location |
|---|---|---|
| C++ RK4 kernel | ~134 ns/step | `benchmarks/benchmark_core.cpp` |
| 1000 s simulation, Python loop | ~200 ms | `benchmarks/benchmark_simulation.py` |
| NMPC solve time | mean ~80 ms, p95 ~105 ms | `benchmarks/benchmark_simulation.py` |

Machine-dependent; regenerate with the benchmark scripts for fresh numbers.
