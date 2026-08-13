"""Heading step response with the baseline C++ controllers.

Run from the repository root:

    python examples/02_heading_control.py

Commands three heading steps (0 -> +90 -> -90 deg) while a PI speed
controller holds cruise speed. Writes ``results/heading_control.png``.
"""

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from vessel_gnc import _core
from vessel_gnc.simulation import simulate

# --- Scenario parameters ---------------------------------------------------
DURATION = 55.0  # [s]
DT = 0.01  # [s] integration step
CONTROL_PERIOD = 0.1  # [s] 10 Hz control
SPEED_REF = 1.0  # [m/s]
OUTPUT = Path("results/heading_control.png")


# Heading reference [rad]: 0 until 15 s, +90 deg until 30 s, then -90 deg.
def heading_ref(t: float) -> float:
    if t < 15.0:
        return 0.0
    if t < 30.0:
        return np.pi / 2
    return -np.pi / 2


def main() -> None:
    params = _core.default_params()
    heading = _core.HeadingController(_core.default_heading_gains())
    speed = _core.SpeedController(_core.default_speed_gains())

    def policy(t: float, state: _core.State) -> _core.Control:
        moment = heading.update(heading_ref(t), state.psi, state.r, CONTROL_PERIOD)
        thrust = speed.update(SPEED_REF, state.u, CONTROL_PERIOD)
        return _core.Control(thrust=thrust, yaw_moment=moment)

    result = simulate(
        DURATION,
        DT,
        params=params,
        control=policy,
        control_period=CONTROL_PERIOD,
    )

    # 90% rise time and overshoot of the first step (t = 15 s).
    window = result.t >= 15.0
    idx = int(np.argmax(result.psi[window] >= 0.9 * np.pi / 2))
    t90 = result.t[window][idx]
    overshoot = np.degrees(np.max(result.psi[window]) - np.pi / 2)
    print(f"90-deg step: rise time {t90 - 15.0:.2f} s, overshoot {overshoot:.2f} deg")
    print(
        f"final: heading {np.degrees(result.psi[-1]):.1f} deg, "
        f"speed {np.hypot(result.u[-1], result.v[-1]):.2f} m/s"
    )

    fig, (ax_psi, ax_m) = plt.subplots(
        2, 1, figsize=(7.2, 6.4), sharex=True, constrained_layout=True
    )
    ref = np.array([heading_ref(t) for t in result.t])
    ax_psi.plot(result.t, np.degrees(ref), "k--", lw=1.2, label="reference")
    ax_psi.plot(result.t, np.degrees(result.psi), lw=1.8, label="vessel")
    ax_psi.set_ylabel("heading [deg]")
    ax_psi.legend(loc="best")
    ax_psi.grid(alpha=0.3)

    ax_m.plot(result.t, result.yaw_moment, lw=1.8)
    ax_m.axhline(params.moment_max, color="r", ls=":", lw=1)
    ax_m.axhline(params.moment_min, color="r", ls=":", lw=1, label="actuator limits")
    ax_m.set_ylabel("yaw moment [N m]")
    ax_m.set_xlabel("t [s]")
    ax_m.legend(loc="best")
    ax_m.grid(alpha=0.3)

    fig.savefig(OUTPUT, dpi=150, bbox_inches="tight")
    print(f"wrote {OUTPUT}")


if __name__ == "__main__":
    main()
