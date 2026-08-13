# Vessel model — 3-DOF horizontal plane

The implementation is numerically verified (unit and convergence tests in
`tests/`); the parameter values are **illustrative** (order-of-magnitude for a
small ~1.5 m, 30 kg USV), not identified from a specific hull. Everything in
this document uses SI units and radians.

## 1. Reference frames and conventions

- **Inertial frame** (horizontal projection of NED): x → North, y → East, z → down
  (right-handed). Positions `(x, y)` in metres.
- **Body frame**: x → forward (surge), y → starboard (sway), z → down; origin at
  the centre of gravity (`x_G = y_G = 0`).
- **Heading** `psi` is measured from North, positive clockwise viewed from above
  (right-hand rule about the downward z axis). Yaw rate `r` is positive clockwise.
- All angles are radians internally; degree conversions happen only at I/O
  boundaries.

State vector and control vector:

```text
eta = [x, y, psi]^T      (inertial position/heading)
nu  = [u, v, r]^T        (body-frame velocities: surge, sway, yaw rate)
tau = [T, 0, N]^T        (actuator: surge thrust T, yaw moment N)
```

## 2. Kinematics

```text
eta_dot = R(psi) nu
```

with the body-to-inertial rotation matrix

```text
R(psi) = [cos psi  -sin psi]
         [sin psi   cos psi]

x_dot   = u cos psi - v sin psi
y_dot   = u sin psi + v cos psi
psi_dot = r
```

## 3. Dynamics

```text
M nu_dot + C(nu_rel) nu_rel + D(nu_rel) nu_rel = tau + tau_env
```

**Mass matrix** (rigid body + diagonal added mass):

```text
M = diag(m11, m22, m33)
m11 = m + X_udot      m22 = m + Y_vdot      m33 = I_z + N_rdot
```

**Coriolis/centripetal term** for a diagonal M (skew-symmetric, standard
3-DOF construction — see Fossen, *Handbook of Marine Craft Hydrodynamics and
Motion Control*, 2011, eq. 3.26 ff.):

```text
C(nu_rel) nu_rel = [-m22 v_rel r ,  m11 u_rel r ,  (m22 - m11) u_rel v_rel]^T
```

The yaw component `(m22 - m11) u_rel v_rel` is the (destabilizing) Munk
coupling: for a slender hull (`m22 > m11`), a starboard sideslip at forward
speed produces a clockwise yaw moment. It is kept in the model because it is
physically important for turning behaviour; the default parameters are chosen
so that the hull remains directionally stable at cruise. In a steady turn the
vessel develops a *port* sideslip (`v < 0`, bow into the turn), so the Munk
moment opposes the turn and reduces the effective yaw damping.

**Damping** (linear + quadratic, diagonal):

```text
d_u = X_u u_rel + X_|u|u |u_rel| u_rel
d_v = Y_v v_rel + Y_|v|v |v_rel| v_rel
d_r = N_r r_rel + N_|r|r |r_rel| r_rel
```

**Actuator force**: `tau = [T, 0, N]^T` — ideal surge force and yaw moment,
subject to saturation bounds (see §5).

**Environment force**: `tau_env = R(psi)^T [F_wind_N, F_wind_E]^T` — the wind
force is defined in the inertial frame and rotated into the body frame.

## 4. Environment model

- **Current** — ambient flow with inertial components `(V_cN, V_cE)`, assumed
  uniform and irrotational. Damping and Coriolis terms act on the *relative*
  velocity `nu_rel = nu - R(psi)^T [V_cN, V_cE]^T`, which is the standard
  manoeuvring formulation. Physical consequences: a vessel at rest in a current
  is dragged along with it; a vessel moving exactly with the current feels no
  hydrodynamic force. (Both are validated in `tests/`.)
- **Wind** — constant force in the inertial frame, applied at the hull centre.
  Wind-induced yaw moment is neglected in the current model.

## 5. Actuator model

Actuators are **not** ideal force sources: the commanded forces go through a
first-order response lag with a smooth rate limit and saturation. The two
channels are decoupled:

```text
dT/dt = T_dot_max tanh((T_cmd - T) / (tau_T T_dot_max))
dN/dt = N_dot_max tanh((N_cmd - N) / (tau_N N_dot_max))
```

with the state projected into `[T_min, T_max] x [N_min, N_max]` after each
step and the command clamped to the same bounds. The tanh keeps
`|dT/dt| <= T_dot_max` strictly and makes the response C1 (fast convergence
of the NMPC solver); the small-signal behaviour is the plain first-order
lag, full-scale steps are rate-limited ramps.

Parameters (illustrative): `tau_T = 1.0 s`, `tau_N = 0.6 s`,
`T_dot_max = 50 N/s`, `N_dot_max = 8 N m/s`. `clamp_control()` enforces the
command bounds; the simulation loop propagates the actuator state
(`actuator_step`) and applies its output to the vessel.

## 6. Parameters (illustrative)

Default values (`ModelParams`, `default_params()`), order-of-magnitude for a
small ~1.5 m, 30 kg USV:

| Symbol | Value | Unit | Meaning |
|---|---:|---|---|
| `m` | 30 | kg | vessel mass |
| `I_z` | 4 | kg m² | yaw inertia about CG |
| `X_udot` | 5 | kg | added mass, surge |
| `Y_vdot` | 20 | kg | added mass, sway |
| `N_rdot` | 2 | kg m² | added inertia, yaw |
| `X_u` | 2 | N s/m | linear surge damping |
| `Y_v` | 150 | N s/m | linear sway damping |
| `N_r` | 30 | N m s/rad | linear yaw damping |
| `X_|u|u` | 20 | N s²/m² | quadratic surge damping |
| `Y_|v|v` | 250 | N s²/m² | quadratic sway damping |
| `N_|r|r` | 60 | N m s²/rad² | quadratic yaw damping |
| `T_min`, `T_max` | −20, 60 | N | thrust saturation |
| `N_min`, `N_max` | −6, 6 | N m | moment saturation |
| `tau_T`, `tau_N` | 1.0, 0.6 | s | actuator response lag |
| `T_dot_max`, `N_dot_max` | 50, 8 | N/s, N m/s | actuator rate limits |

Consequences (used by the validation tests):

- surge equilibrium at `T = 40 N`: `u_eq ≈ 1.36 m/s` (drag balance
  `T = X_u u + X_|u|u u²`);
- steady turn at `T = 40 N`, `N = 2 N m`: `r ≈ 0.072 rad/s`, turn radius ≈ 19 m,
  port sideslip `v ≈ -0.022 m/s` (bow into the turn);
- the hull is directionally stable at cruise: yaw damping exceeds the Munk
  coupling (`N_r > (m22 - m11) u dv/dr`).

**Important**: these values are *not* identified from a real vessel and no
physical fidelity claim is made for a specific craft. They are chosen to give a
small USV plausible manoeuvring behaviour (speeds of 1–2 m/s, turning radii of
order 10–20 m).

## 7. Approximations and limitations

| Approximation | Why | Expected validity | Limitation |
|---|---|---|---|
| 3-DOF horizontal plane | Scope of v1 (plan §3) | Quasi-horizontal motions, calm water, low Froude number | No heave/roll/pitch, no wave forces |
| Diagonal added mass | Standard for symmetric hulls | Manoeuvring studies | No cross-coupling (`Y_rdot`, `N_vdot`) |
| Diagonal linear + quadratic damping | Standard surge/sway/yaw representation | Moderate speeds and drift angles | No damping cross-terms |
| Constant inertial wind force, no wind moment | Current simplification | Weak-to-moderate wind | No apparent-wind model, no wind moment |
| First-order actuator with rate limits | Credible lag/saturation/rate behaviour | Low-frequency control studies | No propeller/rudder detail; illustrative time constants |
| Zero-order hold control per RK4 step | Standard sampled control | `dt` small relative to dynamics | Inter-sample behaviour not modelled |
| Illustrative parameters | No vessel data available | Order-of-magnitude behaviour only | Not valid for a specific craft |

## 8. Validation performed

All cases below are automated in `tests/test_dynamics.cpp`,
`tests/test_integrator.cpp` and `tests/test_python.py`.

| Case | Method | Result |
|---|---|---|
| Kinematics | `R Rᵀ = I`, det `R = 1`; pure surge at ψ = 0 and π/2 | Pass |
| Zero force, zero damping (plan case A) | Derivative and 10 s RK4 run (pure surge, where the Coriolis term vanishes) | Velocity constant, straight inertial motion |
| Coriolis coupling | Kinetic energy conserved at derivative and integration level (mixed velocities, unforced, undamped) | Pass |
| Damping sign | Forward speed decays without actuation | Pass |
| Surge equilibrium (plan case B) | Derivative at drag balance; 60 s run reaches `u_eq` | Pass |
| Steady yaw (plan case C) | Yaw acceleration vanishes at drag balance; 60 s run settles into a steady turn with port sideslip (`v < 0`) | Pass |
| Current model | Resting vessel dragged along; co-moving vessel feels no force | Pass |
| Wind model | Inertial force acts through the rotation matrix | Pass |
| RK4 convergence (plan case D) | Error vs analytical surge solution at dt, dt/2, dt/4 | Ratio ≈ 16 (O(dt⁴)) |
| Determinism | Repeated runs bit-identical | Pass |
| Finiteness | 300 s forced run with current: no NaN/Inf | Pass |
| Actuator saturation | `clamp_control` respects bounds | Pass |
