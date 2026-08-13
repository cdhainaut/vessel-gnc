"""Tests for LOS guidance geometry and closed-loop path following."""

import json

import numpy as np
import pytest
import vessel_gnc
from vessel_gnc import _core
from vessel_gnc.guidance import los_heading, make_s_curve_path, project_onto_path
from vessel_gnc.metrics import path_following_metrics
from vessel_gnc.simulation import simulate

# --- Geometry -----------------------------------------------------------------


def test_projection_on_straight_north_path():
    path = np.array([[0.0, 0.0], [100.0, 0.0]])  # heading North (x = North)
    # East of the path = starboard = negative; West = port = positive.
    seg, along, cross = project_onto_path(
        np.array([[50.0, 5.0], [50.0, -5.0], [120.0, 0.0]]), path
    )
    assert np.all(seg == 0)
    assert along == pytest.approx([50.0, 50.0, 100.0])  # clamped at the end
    assert cross == pytest.approx([-5.0, 5.0, 0.0])


def test_projection_on_straight_east_path():
    path = np.array([[0.0, 0.0], [0.0, 100.0]])  # heading East (y = East)
    # North of the path = port = positive; South = starboard = negative.
    _, _, cross = project_onto_path(np.array([[1.0, 50.0], [-1.0, 50.0]]), path)
    assert cross == pytest.approx([1.0, -1.0])


def test_los_heading_values():
    path = np.array([[0.0, 0.0], [0.0, 100.0]])  # East
    # On the path: desired heading = East = +90 deg.
    psi = los_heading(np.array([[0.0, 0.0]]), path, 10.0)
    assert psi[0] == pytest.approx(np.pi / 2)
    # 10 m south of the path start: the projection clamps to the segment
    # start and the LOS point lies 10 m ahead, so the bearing is due East.
    psi = los_heading(np.array([[0.0, -10.0]]), path, 10.0)
    assert psi[0] == pytest.approx(np.arctan2(20.0, 0.0))
    # Beyond the path end, the lookahead clamps to the final waypoint.
    psi = los_heading(np.array([[-10.0, 95.0]]), path, 10.0)
    assert psi[0] == pytest.approx(np.arctan2(5.0, 10.0))


def test_los_rejects_bad_input():
    path = np.array([[0.0, 0.0], [100.0, 0.0]])
    with pytest.raises(ValueError):
        los_heading(np.array([[0.0, 0.0]]), path, 0.0)
    dup = np.array([[0.0, 0.0], [0.0, 0.0], [10.0, 0.0]])
    with pytest.raises(ValueError):
        project_onto_path(np.array([[1.0, 1.0]]), dup)


# --- Closed-loop regression ---------------------------------------------------


def test_path_following_regression():
    # Full LOS + PID/PI loop over the S-curve with current and wind: the
    # cross-track error stays bounded (baseline behaviour, no drift
    # compensation) and everything remains finite.
    path = make_s_curve_path()
    env = _core.Environment(current_east=0.15, wind_east=3.0)
    heading = _core.HeadingController(_core.default_heading_gains())
    speed = _core.SpeedController(_core.default_speed_gains())

    def policy(t: float, state: _core.State) -> _core.Control:
        (psi_los,) = los_heading(np.array([[state.x, state.y]]), path, 8.0)
        moment = heading.update(psi_los, state.psi, state.r, 0.1)
        thrust = speed.update(1.3, state.u, 0.1)
        return _core.Control(thrust=thrust, yaw_moment=moment)

    r = simulate(160.0, 0.01, control=policy, environment=env, control_period=0.1)

    assert np.all(np.isfinite(r.x))
    m = path_following_metrics(r, path, 8.0)
    assert m["cross_track_rms_m"] < 2.0
    assert m["cross_track_max_m"] < 6.0
    assert m["moment_max_Nm"] <= 6.0 + 1e-9  # actuator limits respected
    assert m["thrust_max_N"] <= 40.0 + 1e-9


def test_heading_step_regression():
    # 90 deg heading step from rest: converged within 20 s, no oscillation.
    heading = _core.HeadingController(_core.default_heading_gains())
    speed = _core.SpeedController(_core.default_speed_gains())

    def policy(t: float, state: _core.State) -> _core.Control:
        moment = heading.update(np.pi / 2, state.psi, state.r, 0.1)
        thrust = speed.update(1.0, state.u, 0.1)
        return _core.Control(thrust=thrust, yaw_moment=moment)

    r = simulate(30.0, 0.01, control=policy, control_period=0.1)
    err = np.arctan2(np.sin(r.psi - np.pi / 2), np.cos(r.psi - np.pi / 2))
    assert np.abs(err[-1]) < 0.05
    assert np.max(err) < 0.15  # no significant overshoot


def test_metrics_json_serializable():
    path = make_s_curve_path()
    r = vessel_gnc.simulate(30.0, 0.01, control=_core.Control(thrust=40.0))
    m = path_following_metrics(r, path, 8.0)
    json.dumps(m)  # must not raise
