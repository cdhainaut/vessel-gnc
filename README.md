# Vessel-GNC

Compact C++/Python simulation and control stack for autonomous surface-vessel
experiments: 3-DOF marine dynamics, state estimation, baseline guidance and
nonlinear model predictive control.

> Skeleton milestone. The full README with the
> hero animation, architecture diagram and controller comparison lands with
> the feature milestones.

## Build

Requirements: CMake ≥ 3.20, a C++20 compiler, Python ≥ 3.12.

```bash
pip install -e .        # builds the C++ core via CMake (scikit-build-core)
cmake -B build && cmake --build build   # C++-only
ctest --test-dir build                  # C++ tests
pytest                                  # Python tests
```
