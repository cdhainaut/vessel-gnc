"""Station-keeping in a spatially varying current with the disturbance-aware NMPC.

Run from the repository root:

    python examples/06_station_keeping.py

The vessel holds a watch circle on the rim of the Rankine eddy, where the
current is strongest (~0.25 m/s), while wind gusts push it around. This is
the dynamic-positioning family of problems: a point must be kept, not a
path, and the operational constraint is explicit (a watch radius, plus the
hard actuator bounds inside the OCP). The controller sees only noisy
low-rate sensor measurements filtered by the augmented EKF; its
current-equivalent estimate is held constant over each prediction horizon.
Writes ``results/station_keeping.png``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from vessel_gnc import _core
from vessel_gnc.ekf import VesselEKF
from vessel_gnc.environment import EnvironmentScenario
from vessel_gnc.nmpc import VesselNmpc
from vessel_gnc.plot_style import apply_style, save_figure
from vessel_gnc.sensors import SensorConfig, SensorSuite
from vessel_gnc.simulation import simulate

# --- Scenario parameters ---------------------------------------------------
DURATION = 90.0  # [s]
DT = 0.01  # [s] integration step
CONTROL_PERIOD = 0.2  # [s] 5 Hz NMPC and filter
STATION = np.array([95.0, 25.0])  # [m] on the eddy rim (35 m from its centre)
START_OFFSET = np.array([5.0, 3.0])  # [m] capture from ~6 m off the station
HEADING_REF = 0.0  # [rad] hold head to wind/current
WATCH_RADIUS = 0.5  # [m] operational watch circle
SEED = 7  # reproducible sensor noise
OUTPUT = Path("results/station_keeping.png")

SENSORS = SensorConfig()
SCENARIO = EnvironmentScenario()  # eddy + rotating base current + gusts


def main() -> None:
    params = _core.default_params()
    sensors = SensorSuite(SENSORS, np.random.default_rng(SEED))
    ekf = VesselEKF(params, dt=CONTROL_PERIOD)
    nmpc = VesselNmpc(params)
    horizon = nmpc.config.horizon
    station_refs = np.tile(STATION, (horizon, 1))
    heading_refs = np.full(horizon, HEADING_REF)

    command = _core.Control()
    actuator = _core.ActuatorState()
    r_cov = {name: SENSORS.covariance(name) for name in ("gnss", "compass", "speed", "gyro")}
    errors = []

    def pilot(t: float, state: _core.State) -> _core.Control:
        nonlocal command, actuator
        ekf.predict(command)
        ekf.observe(sensors.sample(state, t), r_cov)
        actuator = _core.actuator_step(actuator, command, params, CONTROL_PERIOD)
        command = nmpc.solve(
            ekf.estimate,
            actuator,
            station_refs,
            heading_refs,
            command,
            disturbance_estimate=ekf.equivalent_current_estimate,
        )
        # Logged against the truth state: the requirement is on the real
        # position, while the controller only ever sees ``ekf.estimate``.
        errors.append((t, np.hypot(state.x - STATION[0], state.y - STATION[1])))
        return command

    result = simulate(
        DURATION,
        DT,
        params=params,
        state0=_core.State(x=STATION[0] + START_OFFSET[0], y=STATION[1] + START_OFFSET[1]),
        control=pilot,
        environment=SCENARIO.sample,
        control_period=CONTROL_PERIOD,
    )

    error_t = np.array([sample[0] for sample in errors])
    errors = np.array([sample[1] for sample in errors])
    held = float(errors[error_t > 20.0].max()) < WATCH_RADIUS
    print(f"watch circle +/-{WATCH_RADIUS:.1f} m held after capture: {held}")
    print(f"position error: mean {errors.mean():.2f} m, max {errors.max():.2f} m")
    print(f"error after capture (t > 20 s): max {errors[error_t > 20.0].max():.2f} m")
    print(
        f"thrust span used: {np.ptp(result.thrust):.1f} N of "
        f"{params.thrust_max - params.thrust_min:.0f} N available"
    )

    _draw(result, error_t, errors, params)


def _draw(
    result,
    error_t: np.ndarray,
    errors: np.ndarray,
    params: _core.ModelParams,
) -> None:
    apply_style()
    fig, (ax_map, ax_error, ax_control) = plt.subplots(
        3, 1, figsize=(8.6, 9.2), height_ratios=[2.2, 1.0, 1.0], constrained_layout=True
    )

    # Map: the current field, the watch circle and the track actually flown.
    grid_n, grid_e = np.meshgrid(
        np.linspace(STATION[0] - 45, STATION[0] + 45, 16),
        np.linspace(STATION[1] - 45, STATION[1] + 45, 16),
    )
    field = np.empty(grid_n.shape + (2,))
    for row in range(grid_n.shape[0]):
        for col in range(grid_n.shape[1]):
            field[row, col] = SCENARIO.eddy_current(grid_n[row, col], grid_e[row, col])
    speed = np.hypot(field[..., 0], field[..., 1])
    ax_map.pcolormesh(
        grid_n, grid_e, speed, cmap="Blues", alpha=0.5, shading="gouraud", vmin=0.0, vmax=0.3
    )
    ax_map.quiver(
        grid_n,
        grid_e,
        field[..., 0],
        field[..., 1],
        color="0.35",
        alpha=0.5,
        units="xy",
        scale=0.06,
        width=0.3,
    )
    watch = plt.Circle(STATION, WATCH_RADIUS, fill=True, color="tab:red", alpha=0.15)
    ax_map.add_patch(watch)
    ax_map.plot(*STATION, "+", color="tab:red", ms=12, mew=2, label="station")
    ax_map.plot(result.x, result.y, color="tab:blue", lw=1.0, alpha=0.6, label="track flown")
    ax_map.plot(result.x[0], result.y[0], "o", color="tab:green", ms=7, label="start")
    ax_map.set_aspect("equal")
    ax_map.set_xlabel("x [m] (North)")
    ax_map.set_ylabel("y [m] (East)")
    ax_map.set_title("Station-keeping on the eddy rim (watch circle in red)")
    ax_map.legend(loc="best")

    # Constraint panel: distance to station against the watch radius.
    ax_error.plot(error_t, errors, color="tab:blue", lw=1.2)
    ax_error.axhline(WATCH_RADIUS, color="tab:red", ls="--", lw=1.0)
    ax_error.annotate(
        f"watch radius {WATCH_RADIUS:g} m",
        xy=(error_t[-1], WATCH_RADIUS),
        xytext=(-4, 4),
        textcoords="offset points",
        ha="right",
        va="bottom",
        color="tab:red",
        fontsize=8,
    )
    ax_error.set_xlabel("t [s]")
    ax_error.set_ylabel("distance to station [m]")
    ax_error.set_title("Watch-circle constraint")

    # Control panel: the hard actuator bounds of the OCP.
    ax_control.plot(result.t, result.thrust, color="tab:blue", lw=1.0, label="thrust [N]")
    ax_control.axhline(params.thrust_max, color="r", ls=":", lw=1)
    ax_control.axhline(params.thrust_min, color="r", ls=":", lw=1)
    ax_control.plot(
        result.t, result.yaw_moment, color="tab:orange", lw=1.0, label="yaw moment [N m]"
    )
    ax_control.axhline(params.moment_max, color="r", ls=":", lw=1)
    ax_control.axhline(params.moment_min, color="r", ls=":", lw=1)
    ax_control.set_xlabel("t [s]")
    ax_control.set_ylabel("applied command")
    ax_control.set_title("Actuator commands with the hard OCP bounds")
    ax_control.legend(loc="best")

    save_figure(fig, OUTPUT)
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
