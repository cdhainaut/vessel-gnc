"""EKF state and current estimation during closed-loop path following.

Run from the repository root:

    python examples/04_ekf.py

The vessel follows the S-curve path with LOS guidance, but the controller
sees only noisy, low-rate sensor measurements filtered by the augmented EKF
(GNSS, compass, speed log and gyro). The environment is time-varying: a
slowly rotating current and wind gusts (portfolio plan Phases D-E), all
unknown to the filter, which estimates the current alongside the vessel
state. Writes ``results/estimator.png`` and
``results/current_estimation.png``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from vessel_gnc import _core
from vessel_gnc.ekf import VesselEKF
from vessel_gnc.environment import EnvironmentScenario
from vessel_gnc.guidance import los_heading, make_s_curve_path
from vessel_gnc.sensors import SensorConfig, SensorSuite
from vessel_gnc.simulation import simulate

# --- Scenario parameters ---------------------------------------------------
DURATION = 120.0  # [s]
DT = 0.01  # [s] integration step
CONTROL_PERIOD = 0.1  # [s] 10 Hz control / filter
SPEED_REF = 1.3  # [m/s]
LOOKAHEAD = 8.0  # [m]
SEED = 42  # reproducible sensor noise
OUTPUT = Path("results/estimator.png")
CURRENT_OUTPUT = Path("results/current_estimation.png")

SENSORS = SensorConfig()  # default rates: GNSS 5 Hz, compass/speed/gyro 10 Hz
SCENARIO = EnvironmentScenario()  # deterministic time-varying disturbances


def main() -> None:
    path = make_s_curve_path()
    # Model mismatch: truth plant, nominal filter (portfolio plan Phase C).
    plant_params = _core.truth_params()
    params = _core.default_params()
    rng = np.random.default_rng(SEED)
    sensors = SensorSuite(SENSORS, rng)
    r_cov = {
        name: SENSORS.covariance(name) for name in ("gnss", "compass", "speed", "gyro")
    }

    heading = _core.HeadingController(_core.default_heading_gains())
    speed = _core.SpeedController(_core.default_speed_gains())
    ekf = VesselEKF(params, dt=CONTROL_PERIOD)

    # Recorded data for the figure (true state, estimate, raw measurements,
    # true and estimated current).
    t_rec, x_rec, y_rec = [], [], []
    xhat_rec, yhat_rec, rhat_rec = [], [], []
    gnss_x, gnss_y = [], []
    r_meas = []
    current_true = []
    current_est = []
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
        prev_cmd = _core.clamp_control(
            _core.Control(thrust=thrust, yaw_moment=moment), params
        )

        environment = SCENARIO.sample(t)
        t_rec.append(t)
        x_rec.append(state.x)
        y_rec.append(state.y)
        xhat_rec.append(xhat.x)
        yhat_rec.append(xhat.y)
        rhat_rec.append(xhat.r)
        current_true.append((environment.current_north, environment.current_east))
        current_est.append((ekf.x[6], ekf.x[7]))
        if "gnss" in measurements:
            gnss_x.append(measurements["gnss"][0])
            gnss_y.append(measurements["gnss"][1])
        if "gyro" in measurements:
            r_meas.append((t, measurements["gyro"][0]))
        return prev_cmd

    result = simulate(
        DURATION,
        DT,
        params=plant_params,
        control=policy,
        environment=SCENARIO.sample,
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
    current_true = np.array(current_true)
    current_est = np.array(current_est)
    current_err = np.hypot(
        current_est[:, 0] - current_true[:, 0],
        current_est[:, 1] - current_true[:, 1],
    )
    print(
        f"position error: rms {np.sqrt(np.mean(pos_err**2)):.2f} m, "
        f"max {np.max(pos_err):.2f} m"
    )
    print(f"yaw-rate error: rms {np.sqrt(np.mean((r_true - r_hat) ** 2)):.3f} rad/s")
    print(
        f"current error (after 20 s): "
        f"rms {np.sqrt(np.mean(current_err[200:] ** 2)) * 1e3:.0f} mm/s, "
        f"max {np.max(current_err[200:]) * 1e3:.0f} mm/s"
    )

    t_arr = np.array(t_rec)
    fig, axes = plt.subplots(2, 2, figsize=(12.0, 9.0), constrained_layout=True)
    (ax_traj, ax_r), (ax_err, ax_cur) = axes

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
    ax_r.plot(
        t_gyro, r_meas_arr, color="tab:red", lw=0.8, alpha=0.5, label="gyro (raw)"
    )
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

    ax_cur.plot(
        t_arr, current_true[:, 0], color="tab:orange", lw=1.2, label="V_cx true"
    )
    ax_cur.plot(
        t_arr, current_est[:, 0], color="tab:orange", lw=1.2, ls="--", label="V_cx est."
    )
    ax_cur.plot(t_arr, current_true[:, 1], color="tab:blue", lw=1.2, label="V_cy true")
    ax_cur.plot(
        t_arr, current_est[:, 1], color="tab:blue", lw=1.2, ls="--", label="V_cy est."
    )
    ax_cur.set_xlabel("t [s]")
    ax_cur.set_ylabel("current [m/s]")
    ax_cur.set_title("Ambient current: true vs estimated")
    ax_cur.legend(loc="best", framealpha=0.9, ncols=2)
    ax_cur.grid(alpha=0.3)

    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUTPUT}")

    # Dedicated current-estimation figure (portfolio plan §9).
    fig_cur, ax_c = plt.subplots(figsize=(7.2, 4.4), constrained_layout=True)
    ax_c.plot(
        t_arr, current_true[:, 0], color="tab:orange", lw=1.4, label="true (north)"
    )
    ax_c.plot(
        t_arr,
        current_est[:, 0],
        color="tab:orange",
        lw=1.2,
        ls="--",
        label="estimated (north)",
    )
    ax_c.plot(t_arr, current_true[:, 1], color="tab:blue", lw=1.4, label="true (east)")
    ax_c.plot(
        t_arr,
        current_est[:, 1],
        color="tab:blue",
        lw=1.2,
        ls="--",
        label="estimated (east)",
    )
    ax_c.set_xlabel("t [s]")
    ax_c.set_ylabel("current [m/s]")
    ax_c.set_title("Ambient current estimated by the augmented EKF")
    ax_c.legend(loc="best", framealpha=0.9)
    ax_c.grid(alpha=0.3)
    fig_cur.savefig(CURRENT_OUTPUT, dpi=150, bbox_inches="tight")
    print(f"wrote {CURRENT_OUTPUT}")


if __name__ == "__main__":
    main()
