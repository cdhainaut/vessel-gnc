"""Open-loop simulation: constant thrust and yaw moment, no feedback.

Run from the repository root:

    python examples/01_open_loop.py

Propagates the vessel for 30 s under constant actuation, a cross-current and a
beam wind, then writes ``results/trajectory_open_loop.png``.
"""

from pathlib import Path

import numpy as np
from vessel_gnc import _core
from vessel_gnc.simulation import simulate
from vessel_gnc.visualization import animate_trajectory, plot_trajectory

# --- Scenario parameters ---------------------------------------------------
DURATION = 30.0  # [s]
DT = 0.01  # [s] integration step
THRUST = 40.0  # [N] constant surge force
YAW_MOMENT = 1.5  # [N m] constant yaw moment (clockwise positive)
CURRENT_NORTH = 0.0  # [m/s]
CURRENT_EAST = 0.15  # [m/s] cross-current
WIND_NORTH = 0.0  # [N]
WIND_EAST = 5.0  # [N] beam wind
OUTPUT = Path("results/trajectory_open_loop.png")
WRITE_GIF = True  # also render the animated scene (plan's first visual milestone)
GIF_OUTPUT = Path("assets/open_loop.gif")


def main() -> None:
    params = _core.default_params()
    environment = _core.Environment(
        current_north=CURRENT_NORTH,
        current_east=CURRENT_EAST,
        wind_north=WIND_NORTH,
        wind_east=WIND_EAST,
    )
    control = _core.Control(thrust=THRUST, yaw_moment=YAW_MOMENT)

    result = simulate(
        DURATION, DT, params=params, control=control, environment=environment
    )

    distance = float(np.hypot(np.diff(result.x), np.diff(result.y)).sum())
    speed = np.hypot(result.u, result.v)
    print(f"simulated {DURATION:g} s in {result.n_steps} steps (dt = {DT:g} s)")
    print(f"final position: x = {result.x[-1]:.2f} m, y = {result.y[-1]:.2f} m")
    print(f"final speed:    {speed[-1]:.2f} m/s")
    print(f"heading:        {np.degrees(result.psi[-1]):.1f} deg")
    print(f"distance:       {distance:.1f} m")

    plot_trajectory(
        result,
        output_path=OUTPUT,
        environment=environment,
        title=(
            f"Open-loop trajectory — {DURATION:g} s, "
            f"T = {THRUST:g} N, N = {YAW_MOMENT:g} N m"
        ),
    )
    print(f"wrote {OUTPUT}")

    if WRITE_GIF:
        animate_trajectory(
            result,
            output_path=GIF_OUTPUT,
            environment=environment,
            title=f"Open-loop — T = {THRUST:g} N, N = {YAW_MOMENT:g} N m",
        )
        print(f"wrote {GIF_OUTPUT}")


if __name__ == "__main__":
    main()
