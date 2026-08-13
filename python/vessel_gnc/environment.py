"""Deterministic time-varying disturbance scenario (portfolio plan Phase D).

The environment is a reproducible function of time: the current rotates
slowly (base vector plus a sinusoid) and the wind adds smooth gust bumps on
top of a mean component. Everything is deterministic — no RNG — so every run
is exactly reproducible.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vessel_gnc import _core

__all__ = ["EnvironmentScenario"]


@dataclass(frozen=True)
class EnvironmentScenario:
    """Slowly varying current plus mean wind with gust events.

    Args:
        current_base_east: mean current component [m/s] (East).
        current_amplitude: rotation amplitude of the current vector [m/s].
        current_period: rotation period [s].
        current_phase: initial phase [rad].
        wind_mean_east: mean wind force [N] (East).
        gust_times: gust event times [s].
        gust_peak: peak gust force [N].
        gust_width: gust width [s] (Gaussian bump sigma).

    Example:
        >>> from vessel_gnc.environment import EnvironmentScenario
        >>> scenario = EnvironmentScenario()
        >>> scenario.sample(0.0).current_east  # doctest: +SKIP
    """

    current_base_east: float = 0.12  # [m/s]
    current_amplitude: float = 0.06  # [m/s]
    current_period: float = 80.0  # [s]
    current_phase: float = 0.0  # [rad]
    wind_mean_east: float = 3.0  # [N]
    gust_times: tuple[float, ...] = (40.0, 85.0)  # [s]
    gust_peak: float = 4.0  # [N]
    gust_width: float = 5.0  # [s]

    def sample(self, t: float) -> _core.Environment:
        """The environment at time ``t`` [s] (SI units).

        The current vector rotates slowly: its East component carries the
        base plus the cosine modulation, the North component the sine. The
        wind is a mean force plus Gaussian gust bumps at the configured times.
        """
        omega = 2.0 * np.pi / self.current_period
        current_north = self.current_amplitude * np.sin(omega * t + self.current_phase)
        current_east = self.current_base_east + self.current_amplitude * np.cos(
            omega * t + self.current_phase
        )

        wind_east = self.wind_mean_east
        for gust_time in self.gust_times:
            wind_east += self.gust_peak * np.exp(
                -0.5 * ((t - gust_time) / self.gust_width) ** 2
            )
        return _core.Environment(
            current_north=float(current_north),
            current_east=float(current_east),
            wind_north=0.0,
            wind_east=float(wind_east),
        )
