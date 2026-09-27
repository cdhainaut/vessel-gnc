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
| Flagship LOS vs nominal/aware NMPC and MPCC, EKF in the loop | validated | `results/reference/metrics.json` (scenario `scenario_v3_mpcc`) |

## Smooth path geometry and geometric MPCC (Milestone 5)

| Case | Status | Location |
|---|---|---|
| PCHIP monotone cubic Hermite on cumulative chord length: knot/interior interpolation, C1 tangent continuity at interior knots | verified | `tests/test_path.py` |
| Unit tangent norm, finite clock-from-North heading, port-positive contour sign | verified | `tests/test_path.py` |
| Deterministic projection, endpoint clamping, progress-hint validation | verified | `tests/test_path.py` |
| Duplicate/non-finite waypoint rejection, degenerate-tangent rejection | verified | `tests/test_path.py` |
| S-curve sampled geometry: no self-intersection, bounded deviation from the waypoint polyline | verified | `tests/test_path.py` |
| NumPy/CasADi path evaluators agree at knots and interior points (same stored coefficients) | verified | `tests/test_path.py` |
| Shared CasADi prediction step vs C++ kernel with non-zero current/wind | verified | `tests/test_prediction.py`, `tests/test_nmpc.py` (max diff < 1e-8) |
| `None` disturbance is exactly the zero-disturbance case | verified | `tests/test_prediction.py`, `tests/test_mpcc.py` |
| MPCC decision-vector layout/dims, bound consistency, no mission-time/reference argument | verified | `tests/test_mpcc.py` |
| Analytical contour/lag signs on straight North/East paths | verified | `tests/test_mpcc.py` |
| Disturbance sensitivity without truth leakage (explicit held estimate) | verified | `tests/test_mpcc.py`, `tests/test_reference.py` |
| Deterministic repeated solve, `reset()` clears warm start and progress | verified | `tests/test_mpcc.py` |
| Transactional failure handling: rejected/excepted solves keep accepted horizons at the original time, bounded `u_prev` fallback is applied | verified | `tests/test_mpcc.py`, `tests/test_reference.py` |
| Short calm straight progression, monotone bounded progress | validated | `tests/test_mpcc.py` |
| Short lateral-offset convergence, short non-zero-current case finite and accepted | validated | `tests/test_mpcc.py` |
| Tuned curved-path relative progress (≥ 98 % of aware-NMPC) without racing | validated | `tests/test_mpcc.py` |
| Stall diagnostic: virtual speed > achieved speed, terminal virtual-physical lead < 2 m | verified | `tests/test_mpcc.py` |
| Fairness/cadence: identical 10 Hz sensor sampling + EKF updates for every controller, predictive solves at 5 Hz, MPCC receives only EKF state + equivalent-current estimate | verified | `tests/test_reference.py` |
| Progress/completion metrics: final progress, 99 % completion rule, `null` completion time | verified | `tests/test_metrics.py` |

**Verified implementation versus physical-model validation.** The path,
projection, prediction-step, failure-handling and fairness rows are
*verified implementation*: they confirm the code reproduces its documented
equations, invariants and input/output contracts (analytical, cross-checked
or deterministic-contract tests). The short closed-loop rows are *validated*
on the illustrative vessel model: they confirm stable, physically
reasonable behaviour under the model's stated approximations, not fidelity
of the parameter set to a real hull (docs/model.md §6–§7). The tuned
`q_progress = 8.25` rationale and its two-seed short-case sweep are recorded
in `docs/control.md §5` (no truth/sensor/EKF change was involved); those
short cases are regression/tuning evidence and are not a substitute for the
deferred 120 s promotion run (plan §6.2).

## Performance

Wall-clock measurements are machine-dependent by construction and are
therefore never part of the deterministic validation record: they live only
in `results/reference/benchmark.json` and the generated table below. No
timing value from the flagship run is ever copied into the deterministic
metrics.

<!-- generated:reference-benchmark-v1:start -->
| Metric | Result |
|---|---:|
| C++ RK4 propagation (vessel + actuator) | **742.8 ns/step** |
| 1000 s simulation (Python loop) | **657 ms** |
| Nominal NMPC mean / median / p95 / max [ms] | **30.0 / 27.4 / 48.8 / 91.0** |
| Disturbance-aware NMPC mean / median / p95 / max [ms] | **30.6 / 27.6 / 44.6 / 81.7** |
| Disturbance-aware MPCC mean / median / p95 / max [ms] | **55.7 / 48.0 / 102.8 / 370.8** |

Machine-dependent wall-clock measurements recorded in `results/reference/benchmark.json` (`benchmark_v3`, 900 predictive solves, 0 failed). Per-workload status histograms: Nominal NMPC: 300 samples, 0 failed, Solve_Succeeded=300; Disturbance-aware NMPC: 300 samples, 0 failed, Solve_Succeeded=300; Disturbance-aware MPCC: 300 samples, 0 failed, Solve_Succeeded=300. The 5 Hz control period defines a 200 ms budget; these solve times make no real-time capability claim. Regenerate with `python tools/generate_reference_results.py`.

<!-- generated:reference-benchmark-v1:end -->

## Reference provenance

Every public number in this repository is generated from the committed
reference artifacts — nothing is hand-edited between the generated markers.
Scenario, seed, configuration, source revision and fingerprint:

<!-- generated:reference-provenance-v1:start -->
| Item | Value |
|---|---|
| Scenario | `scenario_v3_mpcc` (revision 1) |
| Seed | 42 |
| Duration / integration step | 120.0 s / 0.01 s |
| Controllers | `los_pid_v1` · `nominal_nmpc_v1` · `disturbance_aware_nmpc_v1` · `disturbance_aware_mpcc_v1` |
| Estimator | `augmented_current_ekf_v1` |
| Schema | `results/reference/reference.schema.json` (version 3) |
| Deterministic metrics | `results/reference/metrics.json` |
| Machine-dependent benchmark | `results/reference/benchmark.json` |
| Generated at (UTC) | 2026-09-27T19:42:39+00:00 |
| Source commit | `dec0802642417ed3b5f1eaf86f131926fdc591c5` |
| Source fingerprint | dirty: true · `3e710ef8ccd36660a31b388eb7f68ea2bc329af5e463893e1b18621c7f2ac367` |

Provenance rows are generation-time records; the content-based source fingerprint is the authoritative consistency check. The full reproducibility contract and the metric definitions live in the validation documentation.

<!-- generated:reference-provenance-v1:end -->

## Reproducibility contract

`git_commit` and the `dirty` flag record the repository state at generation
time; the authoritative consistency check is the content-based source
fingerprint (ordered source paths plus combined SHA-256). It changes if and
only if a source input changes: commit source changes first, then regenerate
the artifacts (`python tools/generate_reference_results.py`), or keep the
source contents unchanged. `--check` then passes in any clean checkout whose
source tree matches the fingerprint and fails whenever a scenario, parameter
or source change is not reflected in the committed artifacts; it validates
schema, scenario, source fingerprint, artifact hashes and marker bodies
without any simulation.

`--verify-determinism` runs one fresh 120 s reference and compares it with
`results/reference/metrics.json`: the LOS baseline metrics exactly, and the
NMPC, MPCC and estimator metrics within `rtol=1e-6, atol=1e-6` (IPOPT solves
to `tol=1e-4`, so its full-precision iterates may differ in the last ulps),
reporting the worst offending key and deviation on failure.

Solver termination is iteration- and tolerance-based (`max_iter`, `tol`,
`acceptable_tol`): the controllers deliberately carry no wall-clock
termination, which would make accepted iterates depend on machine load and
break this contract. Wall-clock solve times are properties of the workload
(measured in `results/reference/benchmark.json`), never of the solver's
stopping criterion. Reproducibility
is guaranteed within the software environment recorded in `metadata.json`
(`software` block): regenerating in another environment requires a fresh
`--verify-determinism` in that environment before the committed metrics can
be trusted.

## Metric definitions

Saturation duration (`metrics.json`): for each left-closed simulation
interval `[t_k, t_{k+1})` an actuator channel is saturated when its applied
value lies within `1 - saturation_threshold` (default 1 %) of that channel's
full `ModelParams` bound span from either bound, i.e. it reaches at least
99 % of the span from the nearer bound. The duration is the number of
flagged intervals multiplied by the integration step; the final sample
bounds no interval. Thrust, yaw moment and their union (no double counting)
are reported separately.
