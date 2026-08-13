"""Extended Kalman filter for the 3-DOF vessel (docs/estimation.md).

The prediction reuses the C++ RK4 kernel (no duplicated dynamics in Python);
the state-transition Jacobian is computed by central finite differences of
the discrete map. All measurements are linear observations of state
components, updated in Joseph form.
"""

from __future__ import annotations

import numpy as np

from vessel_gnc import _core

__all__ = ["VesselEKF", "STATE_NAMES", "SENSOR_INDICES"]

STATE_NAMES = ["x", "y", "psi", "u", "v", "r"]

# State components observed by each sensor.
SENSOR_INDICES = {"gnss": [0, 1], "compass": [2], "speed": [3], "gyro": [5]}
# Sensors whose innovation must be wrapped to (-pi, pi] (heading channels).
WRAP_INNOVATION = {"compass"}

_DEFAULT_PROCESS_NOISE = [
    1e-4,
    1e-4,
    1e-5,
    5e-4,
    2e-3,
    5e-5,  # vessel per-step variances
    4e-5,
    4e-5,  # current components (slowly varying random walk)
]
_DEFAULT_COV0 = [1.0, 1.0, 1e-2, 0.25, 0.25, 1e-2, 1e-2, 1e-2]


class VesselEKF:
    """Augmented EKF: the 6-D vessel state plus the ambient current.

    State vector ``[x, y, psi, u, v, r, V_cx, V_cy]``: the vessel state
    (docs/model.md §1) augmented with the inertial current components, which
    evolve as a slowly varying random walk. The prediction model is the C++
    RK4 kernel evaluated with the estimated current as the relative-velocity
    reference (docs/estimation.md §3).

    Args:
        params: vessel model parameters.
        process_noise: 8-vector of per-step variances for the diagonal Q
            (defaults: vessel channels plus a slowly varying current walk).
        dt: filter step [s] (one control period).
        state0: initial state vector (default: rest at the origin, calm).
        cov0: initial covariance diagonal (default: modest launch
            uncertainty, current uncertain to 0.1 m/s).

    Example:
        >>> import numpy as np
        >>> from vessel_gnc import _core
        >>> from vessel_gnc.ekf import VesselEKF
        >>> ekf = VesselEKF(_core.default_params(), dt=0.1)
        >>> ekf.predict(_core.Control(thrust=20.0))
        >>> ekf.observe(
        ...     {"gnss": np.array([0.5, -0.2])},
        ...     {"gnss": np.diag([0.25, 0.25])},
        ... )
    """

    def __init__(
        self,
        params: _core.ModelParams,
        process_noise: np.ndarray | None = None,
        dt: float = 0.1,
        state0: np.ndarray | None = None,
        cov0: np.ndarray | None = None,
    ):
        self.params = params
        self.dt = dt
        self.q = np.asarray(
            process_noise if process_noise is not None else _DEFAULT_PROCESS_NOISE,
            dtype=float,
        )
        self.x = np.asarray(state0 if state0 is not None else [0.0] * 8, dtype=float)
        self.P = np.diag(
            np.asarray(cov0 if cov0 is not None else _DEFAULT_COV0, dtype=float)
        )
        # The filter carries the nominal actuator state as a known quantity
        # (docs/model.md §5); the current is part of the estimated state.
        self.actuator = _core.ActuatorState()

    @property
    def estimate(self) -> _core.State:
        """The current vessel-state estimate as a ``_core.State``."""
        return _core.State(
            x=self.x[0],
            y=self.x[1],
            psi=self.x[2],
            u=self.x[3],
            v=self.x[4],
            r=self.x[5],
        )

    @property
    def current_estimate(self) -> _core.Environment:
        """The estimated ambient current as an Environment (zero wind)."""
        return _core.Environment(current_north=self.x[6], current_east=self.x[7])

    def predict(self, control: _core.Control) -> None:
        """Propagate the estimate by one step (RK4 kernel + linearized covariance).

        The command goes through the nominal actuator model; the resulting
        applied forces drive the vessel prediction.
        """
        self.actuator = _core.actuator_step(
            self.actuator, control, self.params, self.dt
        )
        applied = _core.Control(
            thrust=self.actuator.thrust, yaw_moment=self.actuator.yaw_moment
        )
        F = self._jacobian(applied)
        self.x = self._propagate(self.x, applied)
        self.P = F @ self.P @ F.T + np.diag(self.q)

    def observe(
        self, measurements: dict[str, np.ndarray], r: dict[str, np.ndarray]
    ) -> None:
        """Update with the available measurements (Joseph form).

        Args:
            measurements: ``{sensor_name: measurement vector}``.
            r: ``{sensor_name: measurement covariance matrix}``.
        """
        for name, z in measurements.items():
            if name not in SENSOR_INDICES:
                raise ValueError(f"unknown sensor '{name}'")
            self._update_linear(
                z, SENSOR_INDICES[name], r[name], name in WRAP_INNOVATION
            )

    # --- internals ---------------------------------------------------------

    def _propagate(self, x: np.ndarray, control: _core.Control) -> np.ndarray:
        """One filter step of the 8-state map: the vessel integrates with the
        estimated current as the relative-velocity reference; the current
        components follow a random walk (unchanged in the map)."""
        s = _core.State(x=x[0], y=x[1], psi=x[2], u=x[3], v=x[4], r=x[5])
        environment = _core.Environment(current_north=x[6], current_east=x[7])
        nxt = _core.rk4_step(s, control, environment, self.params, self.dt)
        return np.array([nxt.x, nxt.y, nxt.psi, nxt.u, nxt.v, nxt.r, x[6], x[7]])

    def _jacobian(self, control: _core.Control) -> np.ndarray:
        """Central finite differences of the 8-state discrete map."""
        finite_difference_step = 1e-6
        F = np.empty((8, 8))
        for j in range(8):
            xp = self.x.copy()
            xp[j] += finite_difference_step
            xm = self.x.copy()
            xm[j] -= finite_difference_step
            F[:, j] = (self._propagate(xp, control) - self._propagate(xm, control)) / (
                2.0 * finite_difference_step
            )
        return F

    def _update_linear(
        self, z: np.ndarray, indices: list[int], r: np.ndarray, wrap: bool
    ) -> None:
        H = np.zeros((len(indices), 8))
        H[np.arange(len(indices)), indices] = 1.0
        innovation = z - H @ self.x
        if wrap:
            innovation = np.arctan2(np.sin(innovation), np.cos(innovation))
        S = H @ self.P @ H.T + r
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        # Joseph form, then explicit symmetrization (numerical robustness).
        eye = np.eye(8)
        self.P = (eye - K @ H) @ self.P @ (eye - K @ H).T + K @ r @ K.T
        self.P = 0.5 * (self.P + self.P.T)
