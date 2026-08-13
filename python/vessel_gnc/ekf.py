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

_DEFAULT_PROCESS_NOISE = [1e-4, 1e-4, 1e-5, 5e-4, 2e-3, 5e-5]  # per-step variances
_DEFAULT_COV0 = [1.0, 1.0, 1e-2, 0.25, 0.25, 1e-2]


class VesselEKF:
    """EKF on the 6-D vessel state ``[x, y, psi, u, v, r]``.

    Args:
        params: vessel model parameters (the filter propagates the nominal,
            calm-water model; unmodelled disturbances are covered by the
            process noise).
        process_noise: 6-vector of per-step variances for the diagonal Q.
        dt: filter step [s] (one control period).
        state0: initial state vector (default: rest at the origin).
        cov0: initial covariance diagonal (default: modest launch uncertainty).
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
        self.x = np.asarray(state0 if state0 is not None else [0.0] * 6, dtype=float)
        self.P = np.diag(np.asarray(cov0 if cov0 is not None else _DEFAULT_COV0, dtype=float))
        # The filter predicts with the nominal (calm) environment: it does not
        # know the true current/wind (docs/estimation.md §3).
        self.environment = _core.Environment()

    @property
    def estimate(self) -> _core.State:
        """The current state estimate as a ``_core.State``."""
        return _core.State(
            x=self.x[0],
            y=self.x[1],
            psi=self.x[2],
            u=self.x[3],
            v=self.x[4],
            r=self.x[5],
        )

    def predict(self, control: _core.Control) -> None:
        """Propagate the estimate by one step (RK4 kernel + linearized covariance)."""
        F = self._jacobian(control)
        self.x = self._propagate(self.x, control)
        self.P = F @ self.P @ F.T + np.diag(self.q)

    def observe(self, measurements: dict[str, np.ndarray], r: dict[str, np.ndarray]) -> None:
        """Update with the available measurements (Joseph form).

        Args:
            measurements: ``{sensor_name: measurement vector}``.
            r: ``{sensor_name: measurement covariance matrix}``.
        """
        for name, z in measurements.items():
            if name not in SENSOR_INDICES:
                raise ValueError(f"unknown sensor '{name}'")
            self._update_linear(z, SENSOR_INDICES[name], r[name], name in WRAP_INNOVATION)

    # --- internals ---------------------------------------------------------

    def _propagate(self, x: np.ndarray, control: _core.Control) -> np.ndarray:
        s = _core.State(x=x[0], y=x[1], psi=x[2], u=x[3], v=x[4], r=x[5])
        nxt = _core.rk4_step(s, control, self.environment, self.params, self.dt)
        return np.array([nxt.x, nxt.y, nxt.psi, nxt.u, nxt.v, nxt.r])

    def _jacobian(self, control: _core.Control) -> np.ndarray:
        """Central finite differences of the discrete map around the current state."""
        h = 1e-6
        F = np.empty((6, 6))
        for j in range(6):
            xp = self.x.copy()
            xp[j] += h
            xm = self.x.copy()
            xm[j] -= h
            F[:, j] = (self._propagate(xp, control) - self._propagate(xm, control)) / (2.0 * h)
        return F

    def _update_linear(self, z: np.ndarray, indices: list[int], r: np.ndarray, wrap: bool) -> None:
        H = np.zeros((len(indices), 6))
        H[np.arange(len(indices)), indices] = 1.0
        innovation = z - H @ self.x
        if wrap:
            innovation = np.arctan2(np.sin(innovation), np.cos(innovation))
        S = H @ self.P @ H.T + r
        K = self.P @ H.T @ np.linalg.inv(S)
        self.x = self.x + K @ innovation
        # Joseph form, then explicit symmetrization (numerical robustness).
        eye = np.eye(6)
        self.P = (eye - K @ H) @ self.P @ (eye - K @ H).T + K @ r @ K.T
        self.P = 0.5 * (self.P + self.P.T)
