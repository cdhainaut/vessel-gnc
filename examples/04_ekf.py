"""EKF state estimation during closed-loop path following.

Run from the repository root:

    python examples/04_ekf.py

The vessel follows the S-curve path with LOS guidance, but the controller
sees only noisy, low-rate sensor measurements filtered by the EKF (GNSS,
compass, speed log and gyro). Writes ``results/estimator.png``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from vessel_gnc import _core
from vessel_gnc.ekf import VesselEKF
from vessel_gnc.guidance import los_heading, make_s_curve_path
from vessel_gnc.sensors import SensorConfig, SensorSuite
from vessel_gnc.simulation import simulate

# --- Scenario parameters ---------------------------------------------------
DURATION = 120.0  # [s]
DT = 0.01  # [s] integration step
CONTROL_PERIOD = 0.1  # [s] 10 Hz control / filter
SPEED_REF = 1.3  # [m/s]
LOOKAHEAD = 8.0  # [m]
CURRENT_EAST = 0.15  # [m/s] (unknown to the filter)
WIND_EAST = 3.0  # [N] (unknown to the filter)
SEED = 42  # reproducible sensor noise
OUTPUT = Path("results/estimator.png")

SENSORS = SensorConfig()  # default rates: GNSS 5 Hz, compass/speed/gyro 10 Hz


def main() -> None:
    path = make_s_curve_path()
    params = _core.default_params()
    environment = _core.Environment(current_east=CURRENT_EAST, wind_east=WIND_EAST)
    rng = np.random.default_rng(SEED)
    sensors = SensorSuite(SENSORS, rng)
    r_cov = {name: SENSORS.covariance(name) for name in ("gnss", "compass", "speed", "gyro")}

    heading = _core.HeadingController(_core.default_heading_gains())
    speed = _core.SpeedController(_core.default_speed_gains())
    ekf = VesselEKF(params, dt=CONTROL_PERIOD)

    # Recorded data for the figure (true state, estimate, raw measurements).
    t_rec, x_rec, y_rec, psi_rec = [], [], [], []
    xhat_rec, yhat_rec, rhat_rec = [], [], []
    gnss_x, gnss_y = [], []
    r_meas = []
    prev_cmd = _core.Control()

    def policy(t: float, state: _core.State) -> _core.Control:
        nonlocal prev_cmd
        measurements = sensors.sample(state, t)
        ekf.predict(prev_cmd)
        ekf.observe(measurements, r_cov)
        xhat = ekf.estimate

        (psi_los,) = los_heading(np.array([[xhat.x, xhat.y]]), path, LOOKAHEAD)
        moment = heading.update(psi_los, xhat.psi, xhat.r, CONTROL_PERIOD)
        thrust = speed.update(SPEED_REF, xhat.u, CONTROL_PERIOD)
        prev_cmd = _core.clamp_control(_core.Control(thrust=thrust, yaw_moment=moment), params)

        t_rec.append(t)
        x_rec.append(state.x)
        y_rec.append(state.y)
        psi_rec.append(state.psi)
        xhat_rec.append(xhat.x)
        yhat_rec.append(xhat.y)
        rhat_rec.append(xhat.r)
        if "gnss" in measurements:
            gnss_x.append(measurements["gnss"][0])
            gnss_y.append(measurements["gnss"][1])
        if "gyro" in measurements:
            r_meas.append((t, measurements["gyro"][0]))
        return prev_cmd

    result = simulate(
        DURATION,
        DT,
        params=params,
        control=policy,
        environment=environment,
        control_period=CONTROL_PERIOD,
    )

    # Estimation error statistics.
    x_true = np.array(x_rec)
    y_true = np.array(y_rec)
    x_hat = np.array(xhat_rec)
    y_hat = np.array(yhat_rec)
    pos_err = np.hypot(x_true - x_hat, y_true - y_hat)
    r_hat = np.array(rhat_rec)
    r_true = result.r[::10][: len(r_hat)]  # control-rate samples match the recording
    print(f"position error: rms {np.sqrt(np.mean(pos_err**2)):.2f} m, max {np.max(pos_err):.2f} m")
    print(f"yaw-rate error: rms {np.sqrt(np.mean((r_true - r_hat) ** 2)):.3f} rad/s")

    t_arr = np.array(t_rec)
    fig, (ax_traj, ax_r, ax_err) = plt.subplots(1, 3, figsize=(14.5, 4.8), constrained_layout=True)

    ax_traj.plot(path[:, 0], path[:, 1], "k--", lw=1.2, label="reference path")
    ax_traj.plot(x_true, y_true, color="0.6", lw=1.4, label="true")
    ax_traj.plot(x_hat, y_hat, color="tab:blue", lw=1.4, label="EKF estimate")
    ax_traj.scatter(gnss_x, gnss_y, s=4, color="tab:red", alpha=0.5, label="GNSS fixes")
    ax_traj.set_aspect("equal")
    ax_traj.set_xlabel("x [m] (North)")
    ax_traj.set_ylabel("y [m] (East)")
    ax_traj.set_title("Trajectory")
    ax_traj.legend(loc="best", framealpha=0.9)
    ax_traj.grid(alpha=0.3)

    t_gyro, r_meas_arr = zip(*r_meas, strict=True) if r_meas else ([], [])
    ax_r.plot(t_arr, r_true, color="0.6", lw=1.4, label="true")
    ax_r.plot(t_gyro, r_meas_arr, color="tab:red", lw=0.8, alpha=0.5, label="gyro (raw)")
    ax_r.plot(t_arr, r_hat, color="tab:blue", lw=1.4, label="EKF")
    ax_r.set_xlabel("t [s]")
    ax_r.set_ylabel("yaw rate r [rad/s]")
    ax_r.set_title("Yaw rate")
    ax_r.legend(loc="best", framealpha=0.9)
    ax_r.grid(alpha=0.3)

    ax_err.plot(t_arr, pos_err, color="tab:blue", lw=1.4)
    ax_err.set_xlabel("t [s]")
    ax_err.set_ylabel("position error [m]")
    ax_err.set_title("Estimation error")
    ax_err.grid(alpha=0.3)

    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
