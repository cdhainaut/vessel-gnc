# Vessel-GNC

[![CI](https://img.shields.io/github/actions/workflow/status/cdhainaut/vessel-gnc/ci.yml?label=build%20%26%20tests)](https://github.com/cdhainaut/vessel-gnc/actions)
[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)](pyproject.toml)
[![C++20](https://img.shields.io/badge/c%2B%2B-20-blue.svg)](CMakeLists.txt)

C++/Python simulation, estimation and nonlinear control for autonomous
surface vessels.

![Hero: NMPC path following with predicted horizon](assets/hero.gif)

*An autonomous surface vessel follows an S-curve reference path under a
rotating current and wind gusts — disturbances that are unknown to the
controllers and to the filter. The vessel is controlled from EKF estimates
of noisy sensors; the cyan lines are the NMPC's 10 s predicted horizons,
re-solved at 5 Hz.*

**3-DOF dynamics · EKF · LOS/PID · NMPC · C++20 · CasADi**

## Measured performance

<!-- generated:reference-benchmark-v1:start -->
| Metric | Result |
|---|---:|
| C++ RK4 propagation (vessel + actuator) | **472.1 ns/step** |
| 1000 s simulation (Python loop) | **342 ms** |
| NMPC mean / p95 / max solve time [ms] | **91.0 / 118.0 / 208.7** |

Machine-dependent wall-clock measurements recorded in `results/reference/benchmark.json` (`benchmark_v1`, 300 samples, 0 failed solves). The 5 Hz NMPC control period corresponds to a 200 ms budget; these solve times make no real-time capability claim. Regenerate with `python tools/generate_reference_results.py`.

<!-- generated:reference-benchmark-v1:end -->

Deterministic tracking and estimator metrics are kept strictly separate from
these wall-clock numbers: they live in `results/reference/metrics.json` and
are reproduced in the [Demo](#demo) table and in `docs/control.md`,
`docs/estimation.md` and `docs/validation.md`.

## Demo

The flagship scenario: an S-curve reference path under a slowly rotating
current and wind gusts — all unknown to the controllers and to the filter.
The vessel is controlled from EKF estimates only (GNSS at 5 Hz,
compass/speed/gyro at 10 Hz, all noisy), its true dynamics differ from the
controller model, and the commands reach the hull through a rate-limited
actuator with response lag (docs/model.md §5). The augmented EKF estimates
the ambient current online (example 04).

The plant runs the perturbed **truth parameters** (model mismatch) behind a
rate-limited actuator; the environment is time-varying (rotating current,
wind gusts) and the controllers and the filter use the nominal model — they
never see the true state or the true disturbance. The augmented EKF
estimates the ambient current online.

<!-- generated:reference-controller-comparison-v1:start -->
| Metric | LOS (PID/PI) | NMPC |
|---|---:|---:|
| RMS cross-track error [m] | 0.68 | 0.44 |
| P95 cross-track error [m] | 0.98 | 0.69 |
| Max cross-track error [m] | 1.37 | 0.80 |
| RMS wrapped heading error [deg] | 6.3 | 10.2 |
| Max wrapped heading error [deg] | 17.6 | 27.2 |
| RMS applied thrust [N] | 31.8 | 32.7 |
| Max applied thrust [N] | 38.1 | 58.9 |
| RMS applied yaw moment [N m] | 1.3 | 2.1 |
| Max applied yaw moment [N m] | 3.3 | 6.0 |
| Thrust saturation duration [s] | 0.0 | 0.0 |
| Yaw-moment saturation duration [s] | 0.0 | 1.8 |
| Either channel saturated [s] | 0.0 | 1.8 |

Deterministic flagship metrics formatted from `results/reference/metrics.json` (scenario `scenario_v1_mismatch_disturbance`, revision 1, seed 42, 120.0 s at 0.01 s integration). Saturation counts left-closed intervals whose applied value lies within 1% of a `ModelParams` bound span (docs/validation.md). No wall-clock timing appears here: NMPC solve times are machine-dependent and reported separately in the benchmark table.

<!-- generated:reference-controller-comparison-v1:end -->

NMPC tracks the path tighter at the price of more actuator activity; the LOS
baseline is simpler and cheaper. Every number above is formatted from the
committed reference artifacts — no cherry-picked values.

![Controller comparison](assets/controller_comparison.png)

## Architecture

![Control architecture](assets/architecture.svg)

The C++20 core owns the performance-critical kernel (3-DOF dynamics, RK4
integration, PID/PI baseline controllers); Python owns guidance, the EKF,
the NMPC, orchestration and visualization. The NMPC prediction model is an
independent CasADi implementation of the same equations, cross-validated
against the C++ kernel in the tests (max diff < 1e-8).

## Uncertainty-aware control

The reference scenario already runs the full chain — physical model,
simulation, estimation, optimal control — under disturbances that the
controllers do not see:

- actuator dynamics with saturation and rate limits;
- separate nominal (controller) and truth (plant) models with parameter
  mismatch;
- a time-varying rotating current with wind gusts, estimated online by an
  augmented EKF (with the documented limitation that the current estimate
  can absorb wind/model mismatch, docs/estimation.md §5);
- scenario-based robust NMPC propagating several model variants under one
  common control sequence (future work);
- Monte-Carlo evaluation of LOS vs nominal NMPC vs robust NMPC (future
  work).

No RL, PINNs or neural networks: the differentiator is quantitative
robustness analysis of physics-based control.

## Reference results

Committed, versioned artifacts — the sole numerical source of truth:

- `results/reference/config.json` — the full scenario configuration
  (parameter values, not digests, including nominal and truth plant sets);
- `results/reference/metrics.json` — deterministic tracking and estimator
  metrics, no timing;
- `results/reference/benchmark.json` — machine-dependent benchmark timing;
- `results/reference/metadata.json` — software/platform versions, source
  fingerprint and artifact hashes;
- `assets/hero.gif`, `assets/controller_comparison.png`,
  `assets/current_estimation.png` — flagship assets, regenerated from the
  same reference run.

The example scripts additionally write their own figures into `results/`
(open-loop animation, heading step, LOS path following, EKF estimation,
NMPC trajectories).

`results/reference/config.json` and `results/reference/benchmark.json`
record the `git_commit`, and `results/reference/metadata.json` records the
`dirty` flag, of the repository **at generation time** — honest provenance,
not a claim about the current checkout. The authoritative consistency check
is the content-based source fingerprint (source file list + combined
SHA-256): it changes if and only if a source input changes. Commit source
changes first, then regenerate the artifacts
(`python tools/generate_reference_results.py`), or keep the source contents
unchanged; `--check` then passes in any clean checkout whose source tree
matches the fingerprint and fails whenever a scenario/parameter/source
change is not reflected in the committed artifacts.

<!-- generated:reference-provenance-v1:start -->
| Item | Value |
|---|---|
| Scenario | `scenario_v1_mismatch_disturbance` (revision 1) |
| Seed | 42 |
| Duration / integration step | 120.0 s / 0.01 s |
| Controllers | `los_pid_v1` · `nominal_nmpc_v1` |
| Estimator | `augmented_current_ekf_v1` |
| Schema | `results/reference/reference.schema.json` (version 1) |
| Deterministic metrics | `results/reference/metrics.json` |
| Machine-dependent benchmark | `results/reference/benchmark.json` |
| Generated at (UTC) | 2026-08-15T08:44:43+00:00 |
| Source commit | `3add1f1086df420bc0e384dea9c9aba5b115a236` |
| Source fingerprint | dirty: true · `825516e58f37aab9840817b692bf2ecc7bd409d4a47ae17e27e6b75678041275` |

`git_commit` and the `dirty` flag record the repository state at generation time; the source fingerprint is content-based and authoritative. After committing source changes, either regenerate the artifacts (`python tools/generate_reference_results.py`) or keep the source contents unchanged: `--check` compares only the content fingerprint, so a clean checkout at a new commit passes when the source contents are unchanged and fails when they changed. `--check` validates schema, scenario, source fingerprint, artifact hashes and marker bodies without any simulation; `--verify-determinism` runs one fresh 120 s reference and compares it with `results/reference/metrics.json`: the LOS baseline metrics exactly, and the NMPC/estimator metrics within `rtol=1e-6, atol=1e-6` (IPOPT solves to `tol=1e-4`, so its full-precision iterates may differ in the last ulps), reporting the worst offending key and deviation on failure. Reproducibility is guaranteed within the software environment recorded in `metadata.json` (`software` block): regenerating in another environment requires a fresh `--verify-determinism` in that environment before the committed metrics can be trusted.

<!-- generated:reference-provenance-v1:end -->

## Model

3-DOF horizontal-plane manoeuvring model — a Fossen-inspired formulation,
docs/model.md:

```text
eta = [x, y, psi]^T          nu = [u, v, r]^T
eta_dot = R(psi) nu
M nu_dot + C(nu_rel) nu_rel + D(nu_rel) nu_rel = tau + tau_env
```

With diagonal mass + added mass, Coriolis/Munk coupling, linear + quadratic
damping, relative-velocity current and inertial wind force. SI units,
radians internally. Parameters are order-of-magnitude for a small USV —
explicitly **illustrative**, not identified from a specific hull. A detailed
mathematical audit of the combined relative-velocity formulation is left to
future work.

## Software

| Path | Responsibility |
|---|---|
| `include/vessel_gnc/`, `src/` | C++20 core: state, dynamics, integrator, controllers, pybind11 binding |
| `python/vessel_gnc/` | Simulation, guidance, metrics, sensors, EKF, NMPC, visualization, reference runner |
| `tools/` | Reference artifact generation/check tooling |
| `examples/` | Five runnable scenario scripts |
| `tests/` | C++ (GoogleTest) and Python (pytest) tests |
| `benchmarks/` | Reproducible performance benchmarks |
| `docs/` | Model, control, estimation and validation documentation |

## Running the demo

Requirements: CMake ≥ 3.20, a C++20 compiler, Python ≥ 3.12.

```bash
pip install -e .        # builds the C++ core via CMake (scikit-build-core)
python examples/05_nmpc_demo.py    # flagship: NMPC vs LOS + hero animation
```

Reference artifacts:

```bash
python tools/generate_reference_results.py            # full generation (120 s flagship + benchmark)
python tools/generate_reference_results.py --check    # cheap consistency validation (no simulation)
```

Other examples:

```bash
python examples/01_open_loop.py        # open-loop run + animated scene
python examples/02_heading_control.py  # heading step response
python examples/03_path_following.py   # LOS path following + metrics
python examples/04_ekf.py              # EKF estimation in the loop
```

C++-only build and tests:

```bash
cmake -B build && cmake --build build
ctest --test-dir build
```

Benchmarks:

```bash
python benchmarks/benchmark_simulation.py
cmake -B build -DVESSEL_GNC_BUILD_BENCHMARKS=ON && cmake --build build
./build/benchmark_core
```

## Validation

- the full per-case record lives in `docs/validation.md`; the C++
  (GoogleTest) and Python (pytest) suites run in CI;
- analytical and convergence validation (RK4 order, surge equilibrium, yaw
  balance, current equilibrium, EKF no-noise consistency);
- cross-validation of the CasADi NMPC model against the C++ kernel;
- deterministic simulations: seeded randomness everywhere, with an explicit
  one-run determinism check (`python tools/generate_reference_results.py
  --verify-determinism`).

## Documentation

- `docs/model.md` — reference frames, equations, environment model,
  parameters and approximations.
- `docs/control.md` — baseline controllers, LOS guidance and the NMPC
  formulation (weights, sub-stepping, warm start).
- `docs/estimation.md` — augmented EKF formulation (vessel + current),
  sensor model and the disturbance-estimation validation.
- `docs/validation.md` — the full validation record.

## Roadmap

- scenario-based robust NMPC and Monte-Carlo evaluation;
- coastal navigation environment for the flagship visual.

## References

- Fossen, *Handbook of Marine Craft Hydrodynamics and Motion Control* (2011).
- Rawlings, Mayne, Diehl, *Model Predictive Control: Theory, Computation,
  and Design* (2020).
