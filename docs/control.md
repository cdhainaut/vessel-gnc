# Baseline control — LOS guidance with PID heading and PI speed control

The controllers are implemented in C++ (`controllers.hpp`) and exposed
through the binding; guidance geometry lives in `python/vessel_gnc/guidance.py`.
All angles in radians, SI units.

## 1. Control architecture

```text
reference path ──► LOS guidance ──► psi_ref ──► heading PID ──┐
 (waypoints)       (lookahead)       u_ref ──► speed PI ──────┼──► [T, N] ──► vessel
                                                              │            (3-DOF)
                              ◄── psi, r, u ── measurements ──┘
```

The controllers run at a fixed rate (10 Hz in the examples); the command is
held constant between updates (zero-order hold). The simulation loop clamps
every command to the actuator saturation bounds of `ModelParams`.

## 2. Controllers

### Heading controller

```text
e = wrap_to_pi(psi_ref - psi)
N = sat( kp e - kd r + I )
I += ki e dt,  with anti-windup (see below)
```

Design choices:

- **Angle wrapping**: `wrap_to_pi` maps the heading error to `(-pi, pi]`, so a
  reference at `+179 deg` while the vessel sits at `-179 deg` produces a
  `2 deg` error, not `358 deg`.
- **Derivative on the measurement**: the D term uses the measured yaw rate
  `r`, not the derivative of the error — no derivative kick on reference
  steps, no noise amplification through wrapping.
- **Anti-windup**: the integrator is frozen whenever the output is saturated
  and the error pushes further into saturation (conditional integration), and
  the integrator state itself is clamped to `[-integrator_limit, +integrator_limit]`.

### Speed controller

```text
e = u_ref - u
T = sat( kp e + I )
I += ki e dt,  with the same anti-windup rule
```

No derivative term (the surge plant is well damped).

## 3. Gains and tuning

Default gains (`default_heading_gains`, `default_speed_gains`) are tuned
against the default vessel (docs/model.md §6):

| Controller | kp | ki | kd | Output limit | Integrator limit |
|---|---:|---:|---:|---:|---:|
| Heading | 12 | 0.1 | 0.5 | 6 N m | 1 N m |
| Speed | 25 | 15 | 0 | 40 N | 45 N |

Tuning rationale (linearized around cruise `u_eq = 1.36 m/s`):

- The yaw plant `r_dot ~ (N - N_r r)/m33` has a native damping rate
  `N_r/m33 ~ 5 s^-1`: the loop is inherently overdamped, so `kp` dominates.
  `kp = 12` gives `kp/N_r ~ 0.4 s^-1` heading-error decay and a `90 deg` step
  with actuator saturation as the rate-limiting element (~10 s turn).
- The surge plant time constant is
  `m11/(X_u + 2 X_|u|u u_eq) ~ 0.8 s`. The PI loop has two real poles; the
  slow one sits at `ki/(kp + d') ~ 0.21 s^-1` (tau ~ 5 s) and carries the
  steady thrust (`T_eq ~ 36 N` at `u_ref = 1.3 m/s`).

## 4. LOS guidance

Given the vessel position `p`, the projection onto the polyline path gives:

- **along-track** distance `s` from the start of the closest segment;
- **cross-track** error `e_ct`, signed: **positive when the vessel is to the
  left (port) of the path direction**;
- the **lookahead point** `p_los`, located `Delta = 8 m` further along the
  path (clamped to the path end);
- the desired heading `psi_ref = atan2(p_los.y - p.y, p_los.x - p.x)`
  (bearing from North, clockwise positive).

The fixed lookahead trades tracking sharpness against oscillation: a larger
`Delta` smooths the command but cuts corners more on curved paths. The
cross-track error is **not** fed back (pure geometric LOS, no drift
compensation).

## 5. Nonlinear model predictive control (CasADi)

Implementation in `python/vessel_gnc/nmpc.py`.

### Formulation

Discrete-time NMPC over a receding horizon of `N = 25` steps of
`dt = 0.4 s` (10 s horizon), solved at 5 Hz with IPOPT:

```text
min  sum_k [ q_p |p_k - p_ref,k|^2 + q_psi wrap(psi_k - psi_ref,k)^2
             + r_t T_cmd,k^2 + r_n N_cmd,k^2 + s_t dT_cmd,k^2 + s_n dN_cmd,k^2 ]
s.t. X_{k+1} = F(X_k, U_k, d_hat)   (RK4 model, sub-stepped, 8 states)
     T_min <= T_cmd,k <= T_max      (hard actuator bounds)
     N_min <= N_cmd,k <= N_max
     X_0 = (x_hat, actuator)        (pinned current state)
```

with `dT_cmd,k = T_cmd,k - T_cmd,k-1` (rate cost, `T_{-1}` = last command).
The reference is a **time-parametrized trajectory** along the path:
`p_ref,k = path(s_0 + k v_ref dt)` with `s_0 = v_ref t` (mission clock).
This is deliberate: re-anchoring the reference at the vessel's projection each
solve hides the along-track lag from the cost and lets the rate cost suppress
acceleration (the vessel would cruise behind schedule).

### Prediction model

`VesselNmpc.solve(..., disturbance_estimate=...)` exposes the disturbance
explicitly. Nominal NMPC passes `None`, which is exactly zero current/force.
The disturbance-aware variant passes the EKF equivalent-current state and
holds it constant over the 10 s finite horizon:

```text
d_hat(k + j) = d_hat(k),  j = 0 ... N
```

This zero-order-hold assumption is deliberate: the filter estimates a slowly
varying state, not a future disturbance trajectory. No truth environment is
available to the controller.

The model is an independent CasADi implementation of the same equations as
the C++ core — deliberately duplicated because CasADi needs symbolic
expressions. The state is 8-dimensional: the vessel `(x, y, psi, u, v, r)`
plus the actuator `(T, N)` (docs/model.md §5); the controls are the
commanded forces. The actuator block is stepped first, then the vessel block
with the applied forces held at their end-of-step values — the exact
composition of the C++ reference (`actuator_step` + `rk4_step`) — which
keeps the cross-validation test bit-tight (max diff < 1e-8 over random
states with non-zero inertial current and wind). Kinematics use absolute
body velocity, while Coriolis/damping use relative water velocity and retain
the rotating-body current transport term from docs/model.md §3.

Each model step integrates `substeps = 2` internal RK4 steps of 0.2 s. A
single 0.4 s step is outside RK4's stability margin for the yaw dynamics
once the Munk coupling is active (`m33/N_r ~ 0.2 s` time constant) and
produced exploding predictions; sub-stepping fixed it.

### Solver settings and initial guesses

IPOPT runs with a relaxed tolerance (`tol = 1e-4` plus acceptable-iteration
criteria and a 0.4 s wall-time cap): start-up and turn-transient optima are
flat regions (the vessel cannot catch the reference within the horizon), so
tight KKT tolerances are meaningless there and made IPOPT's dual iterates
diverge. The wall-time cap bounds the worst case; the best iterate of a
capped solve is used.

The NLP graph is expanded once by CasADi (`expand=True`), which removes the
runtime overhead of nested symbolic functions without changing the equations.

The initial guess is, in order: the previous solution shifted by one step
(`warm_start`, the default), the previously applied command rolled out
through the model, and a drag-balance cruise rollout. The solver falls back
down the list until one attempt converges (or the wall time is reached).

### Numerical reproducibility

IPOPT runs with `tol = 1e-4` (plus the acceptable-iteration criteria and
the 0.4 s wall-time cap above). Determinism is therefore a
*reproducibility contract*, not a promise of bit-identical iterates:
full-precision IPOPT solutions can legitimately differ in the last ulps
between runs or environments (threading/BLAS) even when every input is
identical. The committed reference metrics (docs/validation.md) enforce
the contract in `--verify-determinism`
(`python tools/generate_reference_results.py`): the LOS baseline metrics
must reproduce exactly (no iterative solver), while the NMPC and estimator
metrics must match within `rtol = 1e-6`, `atol = 1e-6`; a violation is
reported with the worst offending key and its deviation. The wall-time cap
bounds the worst-case solve duration (the 5 Hz control period is a 200 ms
budget) and is machine-dependent: solve times live only in
`results/reference/benchmark.json`, never in the deterministic metrics.
Reproducibility holds within the software environment recorded in
`results/reference/metadata.json` (the `software` block); regenerating in
another environment requires a fresh `--verify-determinism` run there.

### Weights

| Weight | Value | Role |
|---|---:|---|
| `q_position` | 8 | path tracking [1/m²] |
| `q_heading` | 1.5 | tangent alignment [1/rad²] |
| `r_thrust` / `r_moment` | 2e-3 / 1e-2 | actuation effort |
| `s_thrust` / `s_moment` | 5e-3 / 5e-2 | control-rate smoothing |

### Validation record

Automated in `tests/test_nmpc.py`:

| Case | Method | Result |
|---|---|---|
| Model consistency | CasADi vs C++ RK4, 20 random states, non-zero current/wind | max diff < 1e-8 |
| Constraints | Closed-loop run: all commands within actuator bounds | Pass |
| Straight-line tracking | Calm water: converges to cruise, no lateral drift | Pass |
| S-curve regression | Current + wind (unknown to the NMPC), 60 s | RMS cross-track < 1 m, max < 3 m |
| Solve time | 60 s closed loop | mean < 0.2 s, p95 < 0.4 s |
| Determinism | Same inputs, same warm start -> commands agree to abs=1e-5 (IPOPT tol=1e-4, see §5) | Pass |
| Warm start | Shifted guess = previous solution shifted one step | Pass |

## 6. Flagship reference metrics

The flagship scenario (`scenario_v2_disturbance_aware`, revision 1,
seed 42) runs LOS, nominal NMPC and disturbance-aware NMPC closed loop on
EKF estimates for 120.0 s at
0.01 s integration (controller periods 0.1 s / 0.2 s). The plant uses the
perturbed truth parameters behind the rate-limited actuator; the environment
is the rotating current with gusts. The metrics are computed from the
applied (post-actuator) histories with the saturation definition of
docs/validation.md: a channel is saturated on a left-closed interval when
the applied value lies within 1% of the `ModelParams` bound span. All values
below are deterministic and formatted from `results/reference/metrics.json`;
solve times are machine-dependent and reported only in the benchmark table.

<!-- generated:reference-controller-comparison-v1:start -->
| Metric | LOS (PID/PI) | Nominal NMPC | Aware NMPC |
|---|---:|---:|---:|
| RMS cross-track error [m] | 0.68 | 0.44 | 0.26 |
| P95 cross-track error [m] | 0.98 | 0.69 | 0.58 |
| Max cross-track error [m] | 1.37 | 0.80 | 0.76 |
| RMS wrapped heading error [deg] | 6.3 | 10.2 | 10.4 |
| Max wrapped heading error [deg] | 17.6 | 27.2 | 28.4 |
| RMS applied thrust [N] | 31.8 | 32.7 | 32.4 |
| Max applied thrust [N] | 38.1 | 58.9 | 58.8 |
| RMS applied yaw moment [N m] | 1.3 | 2.1 | 2.2 |
| Max applied yaw moment [N m] | 3.3 | 6.0 | 5.8 |
| Thrust saturation duration [s] | 0.0 | 0.0 | 0.0 |
| Yaw-moment saturation duration [s] | 0.0 | 1.8 | 0.0 |
| Either channel saturated [s] | 0.0 | 1.8 | 0.0 |

Deterministic flagship metrics formatted from `results/reference/metrics.json` (scenario `scenario_v2_disturbance_aware`, revision 1, seed 42, 120.0 s at 0.01 s integration). Saturation counts left-closed intervals whose applied value lies within 1% of a `ModelParams` bound span (docs/validation.md). No wall-clock timing appears here: NMPC solve times are machine-dependent and reported separately in the benchmark table.

<!-- generated:reference-controller-comparison-v1:end -->

## 7. Known limitations

- **No current compensation in LOS**: geometric LOS alone does not counteract
  mean current; it remains the simple baseline.
- Single speed reference along the whole path (no speed scheduling).
- The heading loop does not know the path curvature (no feed-forward yaw
  rate); corners are cut by roughly the lookahead distance.
- Nominal NMPC predicts with zero disturbance; disturbance-aware NMPC holds
  the EKF equivalent-current estimate constant over the horizon. Neither
  predicts future gust evolution.
- NMPC has no obstacle constraints and no terminal cost (10 s horizon is long
  relative to the vessel dynamics).
- Both NMPC variants retain mission-clock trajectory tracking; geometric
  predictive path following is deferred to the documented MPCC extension.

## 8. Validation record

Automated in `tests/test_controllers.cpp` and `tests/test_guidance.py`:

| Case | Method | Result |
|---|---|---|
| `wrap_to_pi` | Values across ±π boundaries, bounds check | Pass |
| Heading PID | Zero error → zero output; D acts on yaw rate | Pass |
| Anti-windup | Sustained large error: output at limit, integrator bounded, no transient on release | Pass |
| Saturation | Output never exceeds the moment/thrust limit | Pass |
| Closed loop | 90 deg step from rest with speed hold (30 s run) | Converged, no overshoot |
| Closed loop | Heading hold at `psi_ref = 0.4` under `V_c = 0.3 m/s` cross-current (60 s) | Steady-state error < 0.02 rad |
| LOS geometry | Projection sign convention (left = positive), clamping at path end, lookahead bearing | Pass |
| Path-following regression | S-curve, `V_c = 0.15 m/s`, `F_wind = 3 N`, 160 s | RMS `e_ct` < 2 m, max < 6 m, bounds respected |
| Heading step regression | 90 deg step, 30 s | Final error < 0.05 rad, max error < 0.15 rad |
