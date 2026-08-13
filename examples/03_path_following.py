"""Path following with LOS guidance and the baseline C++ controllers.

Run from the repository root:

    python examples/03_path_following.py

The vessel follows an S-curve reference path (plan §10) under a constant
cross-current and a light beam wind, using line-of-sight guidance with the
PID heading and PI speed controllers. Writes:

- ``results/path_following.png`` (trajectory + cross-track error);
- ``results/path_following_metrics.json`` (plan §21).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from vessel_gnc import _core
from vessel_gnc.guidance import los_heading, make_s_curve_path, project_onto_path
from vessel_gnc.metrics import path_following_metrics
from vessel_gnc.simulation import simulate

# --- Scenario parameters ---------------------------------------------------
DURATION = 160.0  # [s]
DT = 0.01  # [s] integration step
CONTROL_PERIOD = 0.1  # [s] 10 Hz control
SPEED_REF = 1.3  # [m/s]
LOOKAHEAD = 8.0  # [m] LOS lookahead distance
CURRENT_EAST = 0.15  # [m/s] constant cross-current
WIND_EAST = 3.0  # [N] light beam wind
OUTPUT = Path("results/path_following.png")
METRICS_OUTPUT = Path("results/path_following_metrics.json")


def main() -> None:
    path = make_s_curve_path()
    # Model mismatch: the plant runs the perturbed truth parameters while the
    # controllers use the nominal set (portfolio plan Phase C).
    plant_params = _core.truth_params()
    environment = _core.Environment(current_east=CURRENT_EAST, wind_east=WIND_EAST)
    heading = _core.HeadingController(_core.default_heading_gains())
    speed = _core.SpeedController(_core.default_speed_gains())

    def policy(t: float, state: _core.State) -> _core.Control:
        (psi_los,) = los_heading(np.array([[state.x, state.y]]), path, LOOKAHEAD)
        moment = heading.update(psi_los, state.psi, state.r, CONTROL_PERIOD)
        thrust = speed.update(SPEED_REF, state.u, CONTROL_PERIOD)
        return _core.Control(thrust=thrust, yaw_moment=moment)

    result = simulate(
        DURATION,
        DT,
        params=plant_params,
        control=policy,
        environment=environment,
        control_period=CONTROL_PERIOD,
    )

    metrics = path_following_metrics(result, path, LOOKAHEAD)
    print("path-following metrics (LOS baseline):")
    for key, value in metrics.items():
        print(f"  {key:26s} {value:.3f}")

    _, _, cross = project_onto_path(np.column_stack([result.x, result.y]), path)

    fig, (ax_traj, ax_err) = plt.subplots(
        1, 2, figsize=(11.5, 5.2), constrained_layout=True
    )
    ax_traj.plot(path[:, 0], path[:, 1], "k--", lw=1.2, label="reference path")
    ax_traj.plot(result.x, result.y, lw=1.6, label="vessel")
    ax_traj.plot(result.x[0], result.y[0], "o", color="tab:green", ms=8, label="start")
    ax_traj.plot(
        result.x[-1], result.y[-1], "x", color="tab:red", ms=10, mew=2, label="end"
    )
    ax_traj.set_aspect("equal")
    ax_traj.set_xlabel("x [m] (North)")
    ax_traj.set_ylabel("y [m] (East)")
    ax_traj.set_title("LOS path following")
    ax_traj.legend(loc="best", framealpha=0.9)
    ax_traj.grid(alpha=0.3)

    ax_err.plot(result.t, cross, lw=1.4)
    ax_err.axhline(0.0, color="k", lw=0.8)
    ax_err.set_xlabel("t [s]")
    ax_err.set_ylabel("cross-track error [m]")
    ax_err.set_title("Cross-track error (positive = left of path)")
    ax_err.grid(alpha=0.3)

    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUTPUT}")

    METRICS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUTPUT.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"wrote {METRICS_OUTPUT}")


if __name__ == "__main__":
    main()
