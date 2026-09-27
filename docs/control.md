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

### Geometric MPCC (CasADi)

Implementation in `python/vessel_gnc/mpcc.py`; every controller consumes the
same authoritative `PathGeometry` (`python/vessel_gnc/path.py`, the smooth
S-curve of the flagship scenario).

#### Smooth path representation

The path is a C1 piecewise-cubic Hermite curve per North/East coordinate on a
monotone chord-progress coordinate `s in [0, L]` [m], where the knots sit at
cumulative distances between the stored waypoints. Slopes use the
PCHIP-style Fritsch–Carlson monotone algorithm: each coordinate stays within
its adjacent knot range, so the curve cannot ring, self-intersect or
materially overshoot the waypoint polyline. The parameterization is
**cumulative chord length, not exact analytical arc length** (plan §4.2);
the tangent `t = dp/ds` is therefore not exactly unit and is normalized
before every error/heading use. The tangent heading is
`atan2(t_east, t_north)`, clockwise-positive from North (docs/model.md §1).
Projection is a deterministic global closest-point projection with the
hint (`progress_hint_m`) acting only as a validated local guide (earliest-
progress tie break).

#### Contouring and lag errors

At a candidate progress `S`, with vessel position `p_v`, path position
`p_path(S)` and the **unit** tangent `t`:

```text
delta  = p_v - p_path(S)
e_lag  = t_x * delta_x + t_y * delta_y
e_cont = t_y * delta_x - t_x * delta_y
```

- `e_lag` is positive when the vessel is ahead of the candidate path point
  (tangent-projected distance);
- `e_cont` is positive to the port/left of the path direction, matching
  `guidance.project_onto_path` (verified analytically on straight North and
  East paths in `tests/test_mpcc.py`).

#### Progress dynamics and bounds

```text
S[k+1] = S[k] + dt * V_s[k]
0 <= S[k] <= path_length
0 <= V_s[k] <= progress_speed_max
```

`S[0]` is pinned at every solve to the current geometric projection of the
EKF position; a monotone rule (`max(projection, last_accepted - 1e-9)`)
prevents noise-driven backward jumps, and the value is clamped to the path
domain. Future `S` are optimized through the virtual progress speed
`V_s in [0, progress_speed_max]` with defaults
`progress_speed_ref = 1.3 m/s`, `progress_speed_max = 2.0 m/s`. At the
reference rate each prediction step advances at most
`V_s * dt = 0.52 m`, which bounds the incremental virtual-physical lead.

#### Objective terms and units

One stage (evaluated at `X[:, k+1]`, `U[:, k]`, `S[k]`), all SI:

| Term | Expression | Unit of weight |
|---|---|---|
| Contouring | `q_contour * e_cont_k^2` | 1/m² |
| Lag | `q_lag * e_lag_k^2` | 1/m² |
| Heading | `q_heading * wrap(psi_k - psi_path(S_k))^2` | 1/rad² |
| Progress reward | `-q_progress * V_s,k` | s/m |
| Command effort | `r_thrust*T^2 + r_moment*N^2` | 1/N², 1/(N·m)² |
| Command rate | `s_thrust*dT^2 + s_moment*dN^2` | 1/N², 1/(N·m)² |
| Virtual-speed regularization | `r_vs*(V_s - V_s,ref)^2 + s_vs*(V_s,k+1 - V_s,k)^2` | s²/m² |

`dU_0 = U_0 - u_prev`, `dU_k = U_k - U_{k-1}` (the first increment anchors
on the previously applied command); the virtual-speed increment anchors on
the previously applied virtual speed (or `progress_speed_ref` after reset).
The reward is attached to the bounded virtual speed only — there is **no**
`speed_ref * t` position term anywhere, so MPCC imposes no absolute
mission-clock time-position schedule.

#### Solver settings and conditioning (M5-G remediation)

Two defaults changed during the benchmark remediation after deterministic
failed solves appeared in `bench_mpcc` (7/300 samples at the two S-curve
turns, all `Maximum_WallTime_Exceeded` under the 0.4 s per-attempt cap):

- **`r_vs` default `0.0 -> 0.5` [s²/m²]** — conditioning, not tracking. With
  `r_vs = 0.0` the virtual-speed Hessian block consists only of the
  rank-deficient `s_vs` increment penalty, so IPOPT's dual infeasibility
  stalls near `1e-2` at the first-turn instances (a genuine
  non-convergence: even a 3 s free budget left `inf_du ~ 9e-3`, and adaptive
  mu ended in `Error_In_Step_Computation`). The diagonal contribution
  `2*r_vs = 1.0` keeps that block positive definite and the same instances
  converge in 9-27 iterations under the combined config (`r_vs = 0.5` with
  `mu_strategy = "adaptive"`, next bullet). At cruise `V_s = V_s,ref = 1.3 m/s` the term
  is exactly zero; the progress reward `q_progress = 8.25` still dominates
  the bounded `V_s` choice (slope `-8.25 + 2*r_vs*(V_s - 1.3) < 0` at the
  `2.0 m/s` upper bound), so the optimum is not qualitatively changed — the
  regularization is a curvature term, not a speed schedule. Values up to
  `0.3` do not fix the stall; `0.5` is the smallest tested (with adaptive
  mu) that does.
- **IPOPT `mu_strategy = "adaptive"`** — the monotone barrier path needs
  hundreds of iterations on the turn-2 instances (wall-time-capped at
  ~120-150), while the adaptive barrier converges them in 10-25 iterations.
  This is solver tuning only: `tol`, the acceptable-iteration criteria and
  the 0.4 s wall-time cap are unchanged, and the adaptive strategy
  converges to the same optimum.

The regression is locked by
`tests/test_mpcc.py::test_hard_first_turn_instance_solves_within_budget`,
which replays the captured t = 33.8 s hard state with a fresh controller and
requires an accepted first-attempt solve. The tuned tracking weights
(`q_contour`/`q_lag`/`q_heading`/`q_progress`) are untouched by this
remediation.

#### Disturbance and schedule semantics

MPCC receives only the EKF state and the equivalent-current estimate, passed
as an explicit argument and held **constant over the prediction horizon**
(zero-order hold, exactly like the NMPC variants):
`d_hat(k + j) = d_hat(k)`, `j = 0 .. N`. No truth environment, wind forecast
or mission time enters the solver: `VesselMpcc.solve()` has no time or
reference-array argument.

#### Progression-versus-tracking trade-off

The contour cost alone can be minimized by stalling: `q_lag` couples the
vessel to the physical path point, the monotone progress record and the
progress/completion metrics catch stalls, and the bounded `V_s` limits how
far virtual progress can outrun the vessel. The trade-off is reported
honestly: the measured max virtual-physical horizon lead (virtual progress
minus projected physical progress at the horizon end) is **1.24–1.37 m**
across the `q_progress` sweep below, the tuned curved-path head-to-head
asserts `< 1.5 m`, and a deterministic stall regression keeps the terminal
lead below `MAX_PREDICTED_PROGRESS_LEAD_M = 2.0 m` even when the physical
actuator is intentionally stalled. The virtual progress is pinned to the
geometric projection at the solve origin, so the origin lead is zero by
construction; at the reference rate the per-step advance `V_s*dt = 0.52 m`
bounds the incremental lead. No hard lag bound is imposed by design
decision E; `q_lag` is the loophole guard.

### Geometric MPCC progress-weight tuning

`VesselMpcc` uses contour, lag and heading error plus physical effort/rate
terms, a bounded virtual progress speed and the linear reward
`-q_progress * V_s`. The final progress reward is `q_progress = 8.25 s/m`.
It was selected with a bounded 20 s sweep on the smooth S-curve after sensor
sampling and EKF updates were fixed at 10 Hz for every controller; predictive
solves remained at 5 Hz. The comparison denominator is the matched
sensor-seeded disturbance-aware time-NMPC final geometric progress.

| Seed | `q_progress` [s/m] | Relative progress [%] | MPCC RMS / max XTE [m] | Max virtual-physical horizon lead [m] | Accepted solves |
|---:|---:|---:|---:|---:|---:|
| 7 | 8.00 | 98.148 | 0.305 / 0.580 | 1.241 | 99 / 100 |
| 42 | 8.00 | 97.758 | 0.312 / 0.682 | 1.251 | 99 / 100 |
| 7 | **8.25** | **99.202** | **0.304 / 0.578** | **1.261** | **100 / 100** |
| 42 | **8.25** | **98.818** | **0.310 / 0.680** | **1.271** | **100 / 100** |
| 7 | 8.50 | 100.222 | 0.304 / 0.577 | 1.274 | 99 / 100 |
| 42 | 8.50 | 99.849 | 0.309 / 0.679 | 1.288 | 100 / 100 |
| 7 | 9.00 | 102.237 | 0.302 / 0.574 | 1.301 | 100 / 100 |
| 42 | 9.00 | 101.856 | 0.308 / 0.676 | 1.315 | 99 / 100 |
| 7 | 10.00 | 106.083 | 0.298 / 0.569 | 1.354 | 100 / 100 |
| 42 | 10.00 | 105.712 | 0.305 / 0.671 | 1.368 | 100 / 100 |

The 8.25 value is the smallest tested increment above the reviewer-provided
8.0 baseline that clears the 98% target for both seeds. Relative to 8.0 it did
not worsen curved-path error or saturation (all ten cases had zero applied
saturation), and increased maximum virtual lead by only 0.020 m. Higher rewards
offered no required acceptance benefit, increasingly overshot the matched
progress, and q = 8.5 / seed 7 and q = 9 / seed 42 each had one machine-time
solver rejection. Only the progress-reward weight changed: truth physics,
sensor configuration, EKF settings, progress bounds and every other weight
group were held fixed. These short cases are tuning/regression evidence, not a
substitute for the deferred 120 s promotion run.

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
| Metric | LOS (PID/PI) | Nominal NMPC | Aware NMPC | Aware MPCC |
|---|---:|---:|---:|---:|
| RMS cross-track error [m] | 0.69 | 0.41 | 0.23 | 0.25 |
| P95 cross-track error [m] | 0.98 | 0.65 | 0.53 | 0.61 |
| Max cross-track error [m] | 1.07 | 0.74 | 0.63 | 0.71 |
| RMS wrapped heading error [deg] | 7.1 | 10.5 | 11.2 | 9.8 |
| Max wrapped heading error [deg] | 20.4 | 27.7 | 31.7 | 29.1 |
| Final path progress [m] | 154.9 | 156.2 | 156.1 | 164.2 |
| Final path progress fraction [-] | 0.774 | 0.781 | 0.780 | 0.821 |
| Mean progress rate [m/s] | 1.29 | 1.30 | 1.30 | 1.37 |
| Route completion [s] | — | — | — | — |
| RMS applied thrust [N] | 31.8 | 33.7 | 33.5 | 36.6 |
| Max applied thrust [N] | 38.5 | 59.2 | 59.1 | 43.1 |
| RMS applied yaw moment [N m] | 1.5 | 2.2 | 2.5 | 2.3 |
| Max applied yaw moment [N m] | 3.9 | 6.0 | 6.0 | 6.0 |
| Thrust saturation duration [s] | 0.0 | 0.1 | 0.0 | 0.0 |
| Yaw-moment saturation duration [s] | 0.0 | 2.4 | 1.7 | 2.3 |
| Either channel saturated [s] | 0.0 | 2.5 | 1.7 | 2.3 |

Deterministic flagship metrics formatted from `results/reference/metrics.json` (scenario `scenario_v3_mpcc`, revision 1, seed 42, 120.0 s at 0.01 s integration). Route completion is the first sample at 99% of total chord progress; an incomplete route is shown as —. Saturation counts left-closed intervals whose applied value lies within 1% of a `ModelParams` bound span (docs/validation.md). No wall-clock timing appears here: predictive solve times are machine-dependent and reported separately in the benchmark table.

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
- Both legacy NMPC variants retain mission-clock trajectory tracking. The
  separate geometric MPCC removes that schedule but still uses an illustrative
  finite-horizon tuning that requires the full promotion-gate evaluation.
- MPCC's progress reward can transiently let virtual progress lead the
  physical position by up to ~1.2–1.4 m (bounded by the tests recorded in
  docs/validation.md); there is no hard lag bound (design decision E).

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
