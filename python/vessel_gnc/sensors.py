"""Sensor models for the state-estimation demo (docs/estimation.md §2).

Sensors sample the true state on their own schedules and add zero-mean
Gaussian noise. The compass output is wrapped to (-pi, pi] like a real
heading sensor.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vessel_gnc import _core

__all__ = ["SensorConfig", "SensorSuite"]


@dataclass(frozen=True)
class SensorConfig:
    """Sampling periods and noise standard deviations.

    A period of ``None`` disables the sensor.
    """

    gnss_period: float | None = 0.2  # [s] GNSS fix interval (x, y)
    compass_period: float | None = 0.1  # [s] yaw measurement
    speed_period: float | None = 0.1  # [s] surge speed
    gyro_period: float | None = 0.1  # [s] yaw rate
    gnss_sigma: float = 0.5  # [m] per axis
    compass_sigma: float = 0.0175  # [rad] ~ 1 deg
    speed_sigma: float = 0.05  # [m/s]
    gyro_sigma: float = 0.01  # [rad/s]

    def covariance(self, name: str) -> np.ndarray:
        """Measurement covariance R for a sensor."""
        sigmas = {
            "gnss": [self.gnss_sigma] * 2,
            "compass": [self.compass_sigma],
            "speed": [self.speed_sigma],
            "gyro": [self.gyro_sigma],
        }
        return np.diag(np.square(sigmas[name]))


class SensorSuite:
    """Samples the configured sensors on their own schedules."""

    def __init__(self, config: SensorConfig, rng: np.random.Generator):
        self.config = config
        self.rng = rng
        self._periods = {
            "gnss": config.gnss_period,
            "compass": config.compass_period,
            "speed": config.speed_period,
            "gyro": config.gyro_period,
        }
        self._next = {name: 0.0 for name in self._periods}

    def sample(self, state: _core.State, t: float) -> dict[str, np.ndarray]:
        """Measurements due at time ``t`` (empty dict when none are due)."""
        out: dict[str, np.ndarray] = {}
        for name, period in self._periods.items():
            if period is None or t < self._next[name] - 1e-9:
                continue
            self._next[name] = t + period
            out[name] = self._measure(name, state)
        return out

    def _measure(self, name: str, state: _core.State) -> np.ndarray:
        if name == "gnss":
            noise = self.rng.normal(0.0, self.config.gnss_sigma, 2)
            return np.array([state.x, state.y]) + noise
        if name == "compass":
            psi = state.psi + self.rng.normal(0.0, self.config.compass_sigma)
            return np.array([np.arctan2(np.sin(psi), np.cos(psi))])
        if name == "speed":
            return np.array([state.u + self.rng.normal(0.0, self.config.speed_sigma)])
        if name == "gyro":
            return np.array([state.r + self.rng.normal(0.0, self.config.gyro_sigma)])
        raise ValueError(f"unknown sensor '{name}'")
