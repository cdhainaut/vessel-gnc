"""Flagship: LOS vs nominal and disturbance-aware NMPC with EKF.

Run from the repository root:

    python examples/05_nmpc_demo.py

All controllers run closed loop on EKF estimates from noisy sensors (the
plant state is never seen directly). The reference scenario, the policies,
the estimator recording and the deterministic metrics live in
``vessel_gnc.reference``; this script is a thin entry point that runs the
shared runner and renders the recorded run through the shared visualization
helpers (``plot_reference_trajectories``, ``plot_controller_comparison``,
``animate_trajectory``) — it keeps no private plotting copies. Writes:

- ``results/nmpc_trajectory.png`` — trajectories with NMPC horizon snapshots;
- ``results/controller_comparison.png`` — deterministic cross-track error,
  controls and a metrics table (no timing: solve times live in
  ``results/reference/benchmark.json`` and the generated benchmark tables);
- ``results/comparison_metrics.json`` — deterministic controller metrics
  (plan §13, §21);
- ``assets/hero.gif`` — README hero animation (plan §14).
"""

import json
from pathlib import Path

import numpy as np
from vessel_gnc import _core
from vessel_gnc.reference import reference_metrics, run_reference_scenario
from vessel_gnc.visualization import (
    animate_trajectory,
    plot_controller_comparison,
    plot_reference_trajectories,
)

TRAJ_OUTPUT = Path("results/nmpc_trajectory.png")
COMPARISON_OUTPUT = Path("results/controller_comparison.png")
METRICS_OUTPUT = Path("results/comparison_metrics.json")
WRITE_HERO = True  # render the README hero animation (plan §14)
HERO_OUTPUT = Path("assets/hero.gif")

_METRIC_ROWS = (
    "cross_track_rms_m",
    "cross_track_max_m",
    "heading_error_rms_rad",
    "thrust_rms_N",
    "moment_rms_Nm",
)


def main() -> None:
    run = run_reference_scenario()
    metrics = reference_metrics(run)
    los_metrics = metrics["controllers"]["los_pid_v1"]
    nmpc_metrics = metrics["controllers"]["nominal_nmpc_v1"]
    aware_metrics = metrics["controllers"]["disturbance_aware_nmpc_v1"]

    for label, controller_metrics in (
        (run.los.label, los_metrics),
        (run.nmpc.label, nmpc_metrics),
        (run.disturbance_aware_nmpc.label, aware_metrics),
    ):
        print(f"--- {label} ---")
        for key in _METRIC_ROWS:
            print(f"  {key:24s} {controller_metrics[key]:8.3f}")

    METRICS_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    METRICS_OUTPUT.write_text(
        json.dumps(
            {
                "los": los_metrics,
                "nominal_nmpc": nmpc_metrics,
                "disturbance_aware_nmpc": aware_metrics,
            },
            indent=2,
        )
        + "\n"
    )
    print(f"wrote {METRICS_OUTPUT}")

    plot_reference_trajectories(run, TRAJ_OUTPUT)
    print(f"wrote {TRAJ_OUTPUT}")
    plot_controller_comparison(run, metrics, COMPARISON_OUTPUT)
    print(f"wrote {COMPARISON_OUTPUT}")

    if WRITE_HERO:
        controller = run.disturbance_aware_nmpc
        t_est = controller.estimator.t
        current_est = controller.estimator.current_estimate

        def estimated_environment(t: float) -> _core.Environment:
            # Nearest recorded filter estimate.
            idx = int(np.searchsorted(t_est, t, side="right"))
            idx = min(max(idx - 1, 0), len(t_est) - 1)
            return _core.Environment(
                current_north=float(current_est[idx, 0]),
                current_east=float(current_est[idx, 1]),
            )

        cfg = run.config
        animate_trajectory(
            controller.result,
            output_path=HERO_OUTPUT,
            environment=cfg.environment.sample,
            estimated_environment=estimated_environment,
            title="Disturbance-aware NMPC — predicted horizon",
            stride=cfg.render_hero_stride_frames,
            fps=cfg.render_fps,
            wake_duration=cfg.render_hero_wake_duration_s,
            reference_path=run.path,
            horizon=controller.horizon,
            horizon_label=(
                "disturbance-aware prediction "
                f"({cfg.nmpc.horizon * cfg.nmpc.dt:.0f} s horizon)"
            ),
        )
        print(f"wrote {HERO_OUTPUT}")


if __name__ == "__main__":
    main()
