"""Tests for the sensor models and the EKF (docs/estimation.md §4)."""

import numpy as np
import pytest
from vessel_gnc import _core
from vessel_gnc.ekf import VesselEKF
from vessel_gnc.guidance import los_heading, make_s_curve_path
from vessel_gnc.sensors import SensorConfig, SensorSuite
from vessel_gnc.simulation import simulate

# --- Sensor model ------------------------------------------------------------


def test_sensor_suite_firing_schedule():
    config = SensorConfig(
        gnss_period=0.2, compass_period=0.1, speed_period=0.1, gyro_period=0.1
    )
    suite = SensorSuite(config, np.random.default_rng(0))
    state = _core.State(x=1.0, y=2.0, psi=0.5, u=1.0, v=0.0, r=0.1)
    for t in (0.0, 0.1, 0.2, 0.3):
        meas = suite.sample(state, t)
        if np.isclose(t % 0.2, 0.0):
            expected = {"gnss", "compass", "speed", "gyro"}
        else:
            expected = {"compass", "speed", "gyro"}
        assert set(meas) == expected


def test_sensor_noise_statistics():
    config = SensorConfig(gnss_period=None, compass_period=None, speed_period=None)
    suite = SensorSuite(config, np.random.default_rng(1))
    samples = []
    t = 0.0
    for _ in range(2000):
        meas = suite.sample(_core.State(r=0.0), t)
        assert "gyro" in meas
        samples.append(meas["gyro"][0])
        t += 0.1
    assert np.std(samples) == pytest.approx(0.01, rel=0.2)


def test_compass_output_wrapped():
    config = SensorConfig(gnss_period=None, speed_period=None, gyro_period=None)
    suite = SensorSuite(config, np.random.default_rng(2))
    z = suite.sample(_core.State(psi=4.0), 0.0)["compass"][0]  # 4 rad = 229 deg
    assert -np.pi < z <= np.pi


# --- EKF ---------------------------------------------------------------------


def _run_ekf_scenario(env, sensors, seed, duration=60.0, use_estimate=False):
    """Closed-loop LOS run; returns (result, filter, estimate positions)."""
    path = make_s_curve_path()
    params = _core.default_params()
    rng = np.random.default_rng(seed)
    suite = SensorSuite(sensors, rng)
    r_cov = {
        name: sensors.covariance(name) for name in ("gnss", "compass", "speed", "gyro")
    }
    heading = _core.HeadingController(_core.default_heading_gains())
    speed = _core.SpeedController(_core.default_speed_gains())
    ekf = VesselEKF(params, dt=0.1)
    prev_cmd = _core.Control()
    est = []

    def policy(t: float, state: _core.State) -> _core.Control:
        nonlocal prev_cmd
        ekf.predict(prev_cmd)
        ekf.observe(suite.sample(state, t), r_cov)
        xhat = ekf.estimate
        s = xhat if use_estimate else state
        (psi_los,) = los_heading(np.array([[s.x, s.y]]), path, 8.0)
        moment = heading.update(psi_los, s.psi, s.r, 0.1)
        thrust = speed.update(1.3, s.u, 0.1)
        prev_cmd = _core.clamp_control(
            _core.Control(thrust=thrust, yaw_moment=moment), params
        )
        est.append((xhat.x, xhat.y))
        return prev_cmd

    result = simulate(
        duration, 0.01, control=policy, environment=env, control_period=0.1
    )
    return result, ekf, np.array(est)


def test_ekf_covariance_symmetric():
    params = _core.default_params()
    rng = np.random.default_rng(3)
    suite = SensorSuite(SensorConfig(), rng)
    r_cov = {
        name: SensorConfig().covariance(name)
        for name in ("gnss", "compass", "speed", "gyro")
    }
    ekf = VesselEKF(params, dt=0.1)
    for k in range(200):  # 20 s
        state = _core.State(x=10.0 + k * 0.1, y=5.0, psi=0.3, u=1.0, v=0.0, r=0.1)
        ekf.predict(_core.Control(thrust=20.0, yaw_moment=0.2))
        ekf.observe(suite.sample(state, k * 0.1), r_cov)
        assert np.allclose(ekf.P, ekf.P.T, atol=1e-12)
        assert np.all(np.isfinite(ekf.P))
        assert np.all(np.linalg.eigvalsh(ekf.P) > 0.0)  # positive definite


def test_ekf_no_noise_consistent():
    # Exact model (calm water), near-zero measurement noise: the estimate
    # converges to the true state (docs/estimation.md §4, case B).
    params = _core.default_params()
    ekf = VesselEKF(
        params,
        process_noise=np.zeros(6),
        dt=0.1,
        state0=np.array([1.0, -2.0, 0.5, 0.0, 0.0, 0.0]),
        cov0=np.ones(6),
    )
    r_noiseless = {
        name: np.eye(len(SensorConfig().covariance(name))) * 1e-8
        for name in ("gnss", "compass", "speed", "gyro")
    }
    state = _core.State(x=1.0, y=-2.0, psi=0.5)
    truth_actuator = _core.ActuatorState()
    for _ in range(600):  # 60 s
        cmd = _core.Control(thrust=25.0, yaw_moment=0.3)
        truth_actuator = _core.actuator_step(truth_actuator, cmd, params, 0.1)
        applied = _core.Control(
            thrust=truth_actuator.thrust, yaw_moment=truth_actuator.yaw_moment
        )
        state = _core.rk4_step(state, applied, _core.Environment(), params, 0.1)
        measurements = {
            "gnss": np.array([state.x, state.y]),
            "compass": np.array([state.psi]),
            "speed": np.array([state.u]),
            "gyro": np.array([state.r]),
        }
        ekf.predict(cmd)
        ekf.observe(measurements, r_noiseless)
    err = np.abs(
        ekf.x - np.array([state.x, state.y, state.psi, state.u, state.v, state.r])
    )
    assert np.max(err) < 1e-3


def test_ekf_tracks_under_noise_closed_loop():
    # Full loop on EKF estimates with current/wind unknown to the filter:
    # position error stays bounded and the run remains finite.
    env = _core.Environment(current_east=0.15, wind_east=3.0)
    result, ekf, est = _run_ekf_scenario(
        env, SensorConfig(), seed=42, duration=120.0, use_estimate=True
    )
    n = len(est)
    pos_err = np.hypot(result.x[::10][:n] - est[:, 0], result.y[::10][:n] - est[:, 1])
    assert np.all(np.isfinite(result.x))
    assert np.all(np.isfinite(ekf.P))
    assert np.sqrt(np.mean(pos_err**2)) < 3.0
    assert np.max(pos_err) < 6.0
