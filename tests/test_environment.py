"""Unit tests for the deterministic disturbance field (docs/model.md §4)."""

import numpy as np
import pytest
from vessel_gnc.environment import EnvironmentScenario


def test_eddy_peak_tangential_speed_at_core_radius():
    scenario = EnvironmentScenario(current_base_east=0.0, current_amplitude=0.0)
    # Tangential speed peaks at the core radius: 0.25 m/s due east of a
    # counter-clockwise eddy centred at (60, 25).
    north, east = scenario.eddy_current(60.0, 25.0 + scenario.eddy_radius_m)
    assert north == pytest.approx(-scenario.eddy_peak_m_s, abs=1e-12)
    assert east == pytest.approx(0.0, abs=1e-12)


def test_eddy_decays_outside_the_core():
    scenario = EnvironmentScenario()
    at_radius = np.hypot(*scenario.eddy_current(60.0, 25.0 + 35.0))
    far_away = np.hypot(*scenario.eddy_current(60.0, 25.0 + 140.0))
    assert far_away < 0.3 * at_radius


def test_eddy_is_irrotational_outside_the_core():
    # A 1/r velocity field has zero circulation outside the core: the speed
    # is tangential and scales strictly as 1/r.
    scenario = EnvironmentScenario()
    inner = np.hypot(*scenario.eddy_current(60.0, 25.0 + 70.0))
    outer = np.hypot(*scenario.eddy_current(60.0, 25.0 + 140.0))
    assert outer == pytest.approx(inner * 0.5, rel=1e-12)


def test_sample_is_deterministic_and_position_aware():
    scenario = EnvironmentScenario()
    first = scenario.sample(25.0, 60.0, 25.0)
    second = scenario.sample(25.0, 60.0, 25.0)
    assert first.current_north == second.current_north
    assert first.current_east == second.current_east
    elsewhere = scenario.sample(25.0, -200.0, -200.0)
    assert elsewhere.current_east != first.current_east


def test_sample_defaults_keep_time_only_callers_working():
    scenario = EnvironmentScenario()
    expected_east = (
        scenario.current_base_east
        + scenario.current_amplitude * np.cos(scenario.current_phase)
        + scenario.eddy_current(0.0, 0.0)[1]
    )
    assert scenario.sample(0.0).current_east == pytest.approx(expected_east)


def test_wind_gusts_are_position_independent():
    scenario = EnvironmentScenario()
    assert scenario.sample(40.0, 0.0, 0.0).wind_east == pytest.approx(
        scenario.sample(40.0, 500.0, -500.0).wind_east
    )
