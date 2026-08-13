"""Flagship demo: NMPC vs LOS baseline under current and wind, with EKF.

Run from the repository root:

    python examples/05_nmpc_demo.py

Both controllers run closed loop on EKF estimates from noisy sensors (the
plant state is never seen directly). Writes:

- ``results/nmpc_trajectory.png`` — trajectories with NMPC horizon snapshots;
- ``results/controller_comparison.png`` — cross-track error, controls, solve
  times and a metrics table;
- ``results/comparison_metrics.json`` (plan §13, §21).
"""

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from vessel_gnc import _core
from vessel_gnc.ekf import VesselEKF
from vessel_gnc.environment import EnvironmentScenario
from vessel_gnc.guidance import (
    los_heading,
    make_s_curve_path,
    path_reference,
    project_onto_path,
)
from vessel_gnc.metrics import path_following_metrics
from vessel_gnc.nmpc import VesselNmpc
from vessel_gnc.sensors import SensorConfig, SensorSuite
from vessel_gnc.simulation import simulate

# --- Scenario parameters ---------------------------------------------------
DURATION = 120.0  # [s]
DT = 0.01  # [s] integration step
NMPC_PERIOD = 0.2  # [s] 5 Hz NMPC / filter
LOS_PERIOD = 0.1  # [s] 10 Hz LOS controller
SPEED_REF = 1.3  # [m/s]
LOOKAHEAD = 8.0  # [m] LOS lookahead
SCENARIO = EnvironmentScenario()  # time-varying current + gusts, unknown
SEED = 42  # reproducible sensor noise
TRAJ_OUTPUT = Path("results/nmpc_trajectory.png")
COMPARISON_OUTPUT = Path("results/controller_comparison.png")
METRICS_OUTPUT = Path("results/comparison_metrics.json")
HORIZON_SHOT_TIMES = (20.0, 50.0, 80.0)  # [s] horizon snapshots on the figure
WRITE_HERO = True  # render the README hero animation (plan §14)
HERO_OUTPUT = Path("assets/hero.gif")


def run_controller(make_policy, period: float, label: str, nmpc=None):
    """Closed loop on EKF estimates; returns (result, metrics, extras)."""
    # Model mismatch: the plant runs the perturbed truth parameters while the
    # filter and the controllers use the nominal set (portfolio plan Phase C).
    plant_params = _core.truth_params()
    params = _core.default_params()
    sensors = SensorSuite(SENSORS, np.random.default_rng(SEED))
    r_cov = {
        name: SENSORS.covariance(name) for name in ("gnss", "compass", "speed", "gyro")
    }
    ekf = VesselEKF(params, dt=period)
    prev = _core.Control()
    horizon_shots = []
    solve_times = []
    current_est = []  # (t, V_cx, V_cy) for the hero animation

    def policy(t: float, state: _core.State) -> _core.Control:
        nonlocal prev
        ekf.predict(prev)
        ekf.observe(sensors.sample(state, t), r_cov)
        xhat = ekf.estimate
        cmd = make_policy(t, xhat, prev, horizon_shots, solve_times, ekf)
        current_est.append((t, ekf.x[6], ekf.x[7]))
        prev = _core.clamp_control(cmd, params)
        return prev

    result = simulate(
        DURATION,
        DT,
        params=plant_params,
        control=policy,
        environment=SCENARIO.sample,
        control_period=period,
    )
    path = make_s_curve_path()
    metrics = path_following_metrics(result, path, LOOKAHEAD)
    print(f"--- {label} ---")
    for key in (
        "cross_track_rms_m",
        "cross_track_max_m",
        "heading_error_rms_rad",
        "thrust_rms_N",
        "moment_rms_Nm",
    ):
        print(f"  {key:24s} {metrics[key]:8.3f}")
    return result, metrics, horizon_shots, solve_times, np.array(current_est)


def los_policy(t, xhat, prev, horizon_shots, solve_times, ekf):
    path = make_s_curve_path()
    (psi_los,) = los_heading(np.array([[xhat.x, xhat.y]]), path, LOOKAHEAD)
    moment = LOS_HEADING.update(psi_los, xhat.psi, xhat.r, LOS_PERIOD)
    thrust = LOS_SPEED.update(SPEED_REF, xhat.u, LOS_PERIOD)
    return _core.Control(thrust=thrust, yaw_moment=moment)


def nmpc_policy(t, xhat, prev, horizon_shots, solve_times, ekf):
    path = make_s_curve_path()
    refs, psi_refs = path_reference(
        path, SPEED_REF * t, SPEED_REF, NMPC.config.dt, NMPC.config.horizon
    )
    # The NMPC model includes the actuator states (docs/model.md §5): the
    # filter's nominal actuator state is the current initial condition.
    cmd = NMPC.solve(xhat, ekf.actuator, refs, psi_refs, prev)
    solve_times.append(NMPC.last_solve_time)
    horizon_shots.append((t, NMPC.last_trajectory.copy()))  # for the hero
    return cmd


def main() -> None:
    path = make_s_curve_path()
    los_result, los_metrics, _, _, los_current = run_controller(
        los_policy, LOS_PERIOD, "LOS baseline"
    )
    (
        nmpc_result,
        nmpc_metrics,
        horizon_shots,
        solve_times,
        nmpc_current,
    ) = run_controller(nmpc_policy, NMPC_PERIOD, "NMPC")

    print(
        f"  NMPC solve time: mean {np.mean(solve_times) * 1e3:.0f} ms, "
        f"p95 {np.percentile(solve_times, 95) * 1e3:.0f} ms, "
        f"max {np.max(solve_times) * 1e3:.0f} ms"
    )
    metrics = {
        "los": los_metrics,
        "nmpc": nmpc_metrics,
        "nmpc_solve_time_s": {
            "mean": float(np.mean(solve_times)),
            "p95": float(np.percentile(solve_times, 95)),
            "max": float(np.max(solve_times)),
        },
    }
    METRICS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUTPUT.write_text(json.dumps(metrics, indent=2) + "\n")
    print(f"wrote {METRICS_OUTPUT}")

    plot_trajectories(los_result, nmpc_result, horizon_shots)
    plot_comparison(los_result, los_metrics, nmpc_result, nmpc_metrics, solve_times)

    if WRITE_HERO:
        from vessel_gnc.visualization import animate_trajectory

        def estimated_environment(t: float) -> _core.Environment:
            # Nearest recorded filter estimate.
            idx = int(np.searchsorted(nmpc_current[:, 0], t, side="right"))
            idx = min(max(idx - 1, 0), len(nmpc_current) - 1)
            return _core.Environment(
                current_north=float(nmpc_current[idx, 1]),
                current_east=float(nmpc_current[idx, 2]),
            )

        animate_trajectory(
            nmpc_result,
            output_path=HERO_OUTPUT,
            environment=SCENARIO.sample,
            estimated_environment=estimated_environment,
            title="NMPC path following — predicted horizon",
            stride=40,  # 0.4 s per frame
            fps=12,
            wake_duration=12.0,
            reference_path=path,
            horizon=horizon_shots,
            horizon_label=(
                "NMPC prediction "
                f"({NMPC.config.horizon * NMPC.config.dt:.0f} s horizon)"
            ),
        )
        print(f"wrote {HERO_OUTPUT}")


def plot_trajectories(los_result, nmpc_result, horizon_shots):
    path = make_s_curve_path()
    fig, ax = plt.subplots(figsize=(8.0, 6.6))
    ax.plot(path[:, 0], path[:, 1], "k--", lw=1.2, label="reference path")
    ax.plot(los_result.x, los_result.y, color="0.6", lw=1.4, label="LOS baseline")
    ax.plot(nmpc_result.x, nmpc_result.y, color="tab:blue", lw=1.6, label="NMPC")
    for _, traj in horizon_shots:
        ax.plot(traj[0], traj[1], color="tab:cyan", lw=1.0, alpha=0.7)
        ax.plot(traj[0, 0], traj[1, 0], "o", color="tab:cyan", ms=4)
    ax.plot(
        nmpc_result.x[0], nmpc_result.y[0], "o", color="tab:green", ms=8, label="start"
    )
    ax.plot(
        nmpc_result.x[-1],
        nmpc_result.y[-1],
        "x",
        color="tab:red",
        ms=10,
        mew=2,
        label="end (NMPC)",
    )
    ax.annotate(
        "",
        xy=(30.0, 18.0),
        xytext=(0.0, 0.0),
        arrowprops=dict(arrowstyle="->", color="tab:blue", lw=2),
    )
    ax.text(30.5, 18.0, "current (time-varying)", fontsize=8, color="tab:blue")
    ax.set_aspect("equal")
    ax.set_xlabel("x [m] (North)")
    ax.set_ylabel("y [m] (East)")
    ax.set_title("NMPC vs LOS path following (EKF estimates, 120 s)")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.savefig(TRAJ_OUTPUT, dpi=150, bbox_inches="tight")
    print(f"wrote {TRAJ_OUTPUT}")


def plot_comparison(los_result, los_metrics, nmpc_result, nmpc_metrics, solve_times):
    path = make_s_curve_path()
    _, _, cross_los = project_onto_path(
        np.column_stack([los_result.x, los_result.y]), path
    )
    _, _, cross_nmpc = project_onto_path(
        np.column_stack([nmpc_result.x, nmpc_result.y]), path
    )

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(los_result.t, cross_los, color="0.6", lw=1.2, label="LOS")
    ax.plot(nmpc_result.t, cross_nmpc, color="tab:blue", lw=1.2, label="NMPC")
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("cross-track [m]")
    ax.set_title("Cross-track error (positive = left of path)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(los_result.t, los_result.thrust, color="0.6", lw=1.0)
    ax.plot(nmpc_result.t, nmpc_result.thrust, color="tab:blue", lw=1.0)
    ax.axhline(_core.default_params().thrust_max, color="r", ls=":", lw=1)
    ax.axhline(_core.default_params().thrust_min, color="r", ls=":", lw=1)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("thrust [N]")
    ax.set_title("Surge thrust (LOS gray, NMPC blue)")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.hist(np.array(solve_times) * 1e3, bins=20, color="tab:blue", alpha=0.8)
    ax.axvline(
        np.mean(solve_times) * 1e3,
        color="k",
        ls="--",
        lw=1,
        label=f"mean {np.mean(solve_times) * 1e3:.0f} ms",
    )
    ax.axvline(
        np.percentile(solve_times, 95) * 1e3,
        color="r",
        ls="--",
        lw=1,
        label=f"p95 {np.percentile(solve_times, 95) * 1e3:.0f} ms",
    )
    ax.set_xlabel("solve time [ms]")
    ax.set_ylabel("count")
    ax.set_title("NMPC solve time (600 solves)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.axis("off")
    rows = [
        ("", "LOS", "NMPC"),
        (
            "RMS cross-track [m]",
            f"{los_metrics['cross_track_rms_m']:.2f}",
            f"{nmpc_metrics['cross_track_rms_m']:.2f}",
        ),
        (
            "Max cross-track [m]",
            f"{los_metrics['cross_track_max_m']:.2f}",
            f"{nmpc_metrics['cross_track_max_m']:.2f}",
        ),
        (
            "RMS heading error [deg]",
            f"{np.degrees(los_metrics['heading_error_rms_rad']):.1f}",
            f"{np.degrees(nmpc_metrics['heading_error_rms_rad']):.1f}",
        ),
        (
            "RMS thrust [N]",
            f"{los_metrics['thrust_rms_N']:.1f}",
            f"{nmpc_metrics['thrust_rms_N']:.1f}",
        ),
        (
            "Max moment [N m]",
            f"{los_metrics['moment_max_Nm']:.1f}",
            f"{nmpc_metrics['moment_max_Nm']:.1f}",
        ),
        ("Mean solve time [ms]", "—", f"{np.mean(solve_times) * 1e3:.0f}"),
    ]
    table = ax.table(cellText=rows, loc="center", cellLoc="center")
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.0, 1.6)
    ax.set_title("Comparison (plan §13)")

    fig.savefig(COMPARISON_OUTPUT, dpi=150, bbox_inches="tight")
    print(f"wrote {COMPARISON_OUTPUT}")


SENSORS = SensorConfig()
LOS_HEADING = _core.HeadingController(_core.default_heading_gains())
LOS_SPEED = _core.SpeedController(_core.default_speed_gains())
NMPC = VesselNmpc(_core.default_params())


if __name__ == "__main__":
    main()
