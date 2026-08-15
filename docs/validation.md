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
| Relative-current model (drag along, co-moving equilibrium, rotating-body transport) | validated | `tests/test_dynamics.cpp`, `tests/test_integrator.cpp` |
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
| No-noise consistency, 8 states (exact model → estimate converges) | verified | `tests/test_ekf.py` |
| Physical-current estimation (rotating current, exact model, no wind) | validated | `tests/test_ekf.py`, `examples/04_ekf.py` |
| Equivalent-current state under mismatch + unknown current/wind | demonstrated | `results/reference/metrics.json`, `assets/current_estimation.png` |
| Closed-loop tracking on estimates with unknown current/wind | validated | `tests/test_ekf.py` |

## Optimal control (NMPC)

| Case | Status | Location |
|---|---|---|
| CasADi 8-state model vs C++ with current/wind (actuator + vessel) | verified | `tests/test_nmpc.py` (max diff < 1e-8) |
| Actuator bounds respected, straight-line tracking | verified | `tests/test_nmpc.py` |
| S-curve regression with unknown current/wind | validated | `tests/test_nmpc.py` |
| Solve-time budget, determinism, warm-start shift | verified | `tests/test_nmpc.py` |
| Flagship LOS vs nominal/aware NMPC, EKF in the loop | validated | `results/reference/metrics.json` (scenario `scenario_v2_disturbance_aware`) |

## Performance

Wall-clock measurements are machine-dependent by construction and are
therefore never part of the deterministic validation record: they live only
in `results/reference/benchmark.json` and the generated table below. No
timing value from the flagship run is ever copied into the deterministic
metrics.

<!-- generated:reference-benchmark-v1:start -->
| Metric | Result |
|---|---:|
| C++ RK4 propagation (vessel + actuator) | **578.2 ns/step** |
| 1000 s simulation (Python loop) | **453 ms** |
| Nominal NMPC mean / p95 / max [ms] | **23.9 / 33.9 / 77.0** |
| Disturbance-aware NMPC mean / p95 / max [ms] | **22.5 / 31.0 / 40.8** |

Machine-dependent wall-clock measurements recorded in `results/reference/benchmark.json` (`benchmark_v2`, 600 samples, 0 failed solves). The 5 Hz NMPC control period corresponds to a 200 ms budget; these solve times make no real-time capability claim. Regenerate with `python tools/generate_reference_results.py`.

<!-- generated:reference-benchmark-v1:end -->

## Reference provenance

Every public number in this repository is generated from the committed
reference artifacts — nothing is hand-edited between the generated markers.
Scenario, seed, configuration, source revision and fingerprint:

<!-- generated:reference-provenance-v1:start -->
| Item | Value |
|---|---|
| Scenario | `scenario_v2_disturbance_aware` (revision 1) |
| Seed | 42 |
| Duration / integration step | 120.0 s / 0.01 s |
| Controllers | `los_pid_v1` · `nominal_nmpc_v1` · `disturbance_aware_nmpc_v1` |
| Estimator | `augmented_current_ekf_v1` |
| Schema | `results/reference/reference.schema.json` (version 2) |
| Deterministic metrics | `results/reference/metrics.json` |
| Machine-dependent benchmark | `results/reference/benchmark.json` |
| Generated at (UTC) | 2026-08-15T09:44:51+00:00 |
| Source commit | `53e0958039310f59f44ec2b7be4a840dd0e882d2` |
| Source fingerprint | dirty: true · `4f7198bfe08cf9b7118c779a77d1fab84dd61c8cba8506322cb17b34624aa414` |

`git_commit` and the `dirty` flag record the repository state at generation time; the source fingerprint is content-based and authoritative. After committing source changes, either regenerate the artifacts (`python tools/generate_reference_results.py`) or keep the source contents unchanged: `--check` compares only the content fingerprint, so a clean checkout at a new commit passes when the source contents are unchanged and fails when they changed. `--check` validates schema, scenario, source fingerprint, artifact hashes and marker bodies without any simulation; `--verify-determinism` runs one fresh 120 s reference and compares it with `results/reference/metrics.json`: the LOS baseline metrics exactly, and both NMPC variants plus estimator metrics within `rtol=1e-6, atol=1e-6` (IPOPT solves to `tol=1e-4`, so its full-precision iterates may differ in the last ulps), reporting the worst offending key and deviation on failure. Reproducibility is guaranteed within the software environment recorded in `metadata.json` (`software` block): regenerating in another environment requires a fresh `--verify-determinism` in that environment before the committed metrics can be trusted.

<!-- generated:reference-provenance-v1:end -->
