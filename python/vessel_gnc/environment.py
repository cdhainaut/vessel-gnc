"""Deterministic disturbance scenarios (portfolio plan Phase D).

The environment is a reproducible function of time and position: a base
current, a Rankine eddy (a mesoscale-style vortex with an irrotational
outer decay), and wind with smooth gust bumps. Everything is deterministic
(no RNG anywhere), so every run is exactly reproducible. The flow is
quasi-static: at each instant the water around the hull is treated as
locally uniform and irrotational, so the plant equations are unchanged and
the field is simply sampled at the vessel position.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from vessel_gnc import _core

__all__ = ["EnvironmentScenario"]


@dataclass(frozen=True)
class EnvironmentScenario:
    """Base current, Rankine eddy, and mean wind with gust events.

    The eddy is a two-part vortex: solid-body rotation inside
    ``eddy_radius_m`` and an irrotational ``1/r`` decay outside it, with a
    peak tangential speed of ``eddy_peak_m_s`` at the radius. It is steady
    in time, so it is what the vessel has to work against everywhere on the
    path; the base current adds a slow rotation and the wind adds gusts.

    Args:
        current_base_east: mean current component [m/s] (East).
        current_amplitude: rotation amplitude of the current vector [m/s].
        current_period: rotation period [s].
        current_phase: initial phase [rad].
        eddy_north_m: eddy centre, North coordinate [m].
        eddy_east_m: eddy centre, East coordinate [m].
        eddy_radius_m: eddy core radius [m].
        eddy_peak_m_s: tangential speed at the core radius [m/s],
            counter-clockwise when positive.
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
    eddy_north_m: float = 60.0  # [m]
    eddy_east_m: float = 25.0  # [m]
    eddy_radius_m: float = 35.0  # [m]
    eddy_peak_m_s: float = 0.25  # [m/s]
    wind_mean_east: float = 3.0  # [N]
    gust_times: tuple[float, ...] = (40.0, 85.0)  # [s]
    gust_peak: float = 4.0  # [N]
    gust_width: float = 5.0  # [s]

    def eddy_current(self, x: float, y: float) -> tuple[float, float]:
        """Eddy-induced current components ``(north, east)`` [m/s] at ``(x, y)``.

        Rankine vortex: tangential speed grows linearly inside the core
        radius and decays as ``1/r`` outside it, so the outer field is
        irrotational and the whole flow stays divergence-free.
        """
        dn = x - self.eddy_north_m
        de = y - self.eddy_east_m
        radius = float(np.hypot(dn, de))
        if radius == 0.0:
            return 0.0, 0.0
        if radius < self.eddy_radius_m:
            tangential = self.eddy_peak_m_s * radius / self.eddy_radius_m
        else:
            tangential = self.eddy_peak_m_s * self.eddy_radius_m / radius
        # Counter-clockwise in the (North, East) plane for a positive peak.
        return float(-tangential * de / radius), float(tangential * dn / radius)

    def sample(self, t: float, x: float = 0.0, y: float = 0.0) -> _core.Environment:
        """The environment at time ``t`` [s] and position ``(x, y)`` [m].

        The base current vector rotates slowly (East carries the base plus
        the cosine modulation, North the sine), the eddy is steady and
        spatial, and the wind is a mean force plus Gaussian gust bumps.
        Position arguments are optional so time-only callers keep working.
        """
        omega = 2.0 * np.pi / self.current_period
        eddy_north, eddy_east = self.eddy_current(x, y)
        current_north = self.current_amplitude * np.sin(omega * t + self.current_phase) + eddy_north
        current_east = (
            self.current_base_east
            + self.current_amplitude * np.cos(omega * t + self.current_phase)
            + eddy_east
        )

        wind_east = self.wind_mean_east
        for gust_time in self.gust_times:
            wind_east += self.gust_peak * np.exp(-0.5 * ((t - gust_time) / self.gust_width) ** 2)
        return _core.Environment(
            current_north=float(current_north),
            current_east=float(current_east),
            wind_north=0.0,
            wind_east=float(wind_east),
        )
