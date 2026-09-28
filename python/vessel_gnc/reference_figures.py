"""Flagship comparison figures built from a :class:`~vessel_gnc.reference.ReferenceRun`.

All numbers come from the in-memory reference run and its deterministic
metrics; figures are exported with the shared package style
(``vessel_gnc.plot_style``). The trajectory and animation helpers for plain
simulation results live in ``vessel_gnc.visualization``.
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np

from vessel_gnc.guidance import project_onto_path
from vessel_gnc.plot_style import (
    CONTROLLER_STYLES,
    DOUBLE_SIZE,
    SINGLE_SIZE,
    apply_style,
    save_figure,
)
from vessel_gnc.visualization import environment_arrows

if TYPE_CHECKING:
    from vessel_gnc.reference import ReferenceRun

__all__ = [
    "plot_reference_trajectories",
    "plot_controller_comparison",
    "plot_current_estimation",
]


def plot_reference_trajectories(
    run: ReferenceRun,
    output_path: str | os.PathLike[str],
) -> plt.Figure:
    """Reference-run trajectory overview: four controllers with horizons.

    Draws the reference path, the four controller trajectories (LOS, nominal
    NMPC, disturbance-aware NMPC, geometric MPCC), the disturbance-aware
    NMPC prediction horizons, the MPCC predicted horizon and the true
    environment sampled at four times along the aware trajectory. Saved to
    ``output_path`` (the ignored results figure of the reference pipeline).

    Args:
        run: the in-memory reference run.
        output_path: PNG destination (parent directories are created).

    Returns:
        The figure.

    Example:
        >>> from vessel_gnc.reference import run_reference_scenario
        >>> from vessel_gnc.visualization import plot_reference_trajectories
        >>> run = run_reference_scenario()  # doctest: +SKIP  (120 s flagship)
        >>> figure = plot_reference_trajectories(
        ...     run, "results/reference/nmpc_trajectory.png"
        ... )  # doctest: +SKIP
    """
    path = run.path
    los_result = run.los.result
    nominal_result = run.nmpc.result
    aware_result = run.disturbance_aware_nmpc.result
    mpcc_result = (
        run.disturbance_aware_mpcc.result if run.disturbance_aware_mpcc is not None else None
    )

    apply_style()
    fig, ax = plt.subplots(figsize=SINGLE_SIZE)
    ax.plot(path[:, 0], path[:, 1], "k--", lw=1.2, label="reference path")
    ax.plot(los_result.x, los_result.y, color="0.6", lw=1.3, label="LOS baseline")
    ax.plot(
        nominal_result.x,
        nominal_result.y,
        color="tab:blue",
        lw=1.4,
        label="nominal NMPC",
    )
    ax.plot(
        aware_result.x,
        aware_result.y,
        color="tab:green",
        lw=1.7,
        label="disturbance-aware NMPC",
    )
    if mpcc_result is not None:
        ax.plot(
            mpcc_result.x,
            mpcc_result.y,
            color="tab:purple",
            lw=1.4,
            label="geometric MPCC",
        )
    for _, traj in run.disturbance_aware_nmpc.horizon:
        ax.plot(traj[0], traj[1], color="tab:cyan", lw=1.0, alpha=0.7)
    if run.disturbance_aware_mpcc is not None:
        for _, traj in run.disturbance_aware_mpcc.horizon:
            ax.plot(traj[0], traj[1], color="tab:purple", lw=1.0, alpha=0.35)
    ax.plot(
        aware_result.x[0],
        aware_result.y[0],
        "o",
        color="tab:green",
        ms=8,
        label="start",
    )
    ax.plot(
        aware_result.x[-1],
        aware_result.y[-1],
        "x",
        color="tab:red",
        ms=10,
        mew=2,
        label="end (aware NMPC)",
    )

    # True environment at four times, anchored on the aware-NMPC trajectory.
    # Use one legend entry per quantity instead of overlapping arrow labels.
    t = aware_result.t
    environment_handles: list[tuple] = []
    for index, shot_t in enumerate(np.linspace(0.0, t[-1], 4)):
        x0 = float(np.interp(shot_t, t, aware_result.x))
        y0 = float(np.interp(shot_t, t, aware_result.y))
        _, handles = environment_arrows(
            ax,
            run.config.environment.sample(shot_t, x0, y0),
            x0,
            y0,
            annotate=False,
        )
        if index == 0:
            environment_handles = handles

    ax.set_aspect("equal")
    ax.set_xlabel("x [m] (North)")
    ax.set_ylabel("y [m] (East)")
    ax.set_title("LOS vs nominal/aware NMPC and geometric MPCC")
    plot_handles, plot_labels = ax.get_legend_handles_labels()
    plot_handles.extend(handle for handle, _ in environment_handles)
    plot_labels.extend(label for _, label in environment_handles)
    ax.legend(plot_handles, plot_labels, loc="best", framealpha=0.9)
    fig.tight_layout()

    save_figure(fig, output_path)
    return fig


def plot_controller_comparison(
    run: ReferenceRun,
    metrics: dict[str, object],
    output_path: str | os.PathLike[str],
) -> plt.Figure:
    """Deterministic LOS/NMPC/MPCC comparison (tracking, controls, progress).

    Four panels: the signed cross-track error, the applied surge thrust and
    yaw moment with the truth-plant actuator bounds annotated on the bound
    lines, and a metrics table assembled from the deterministic reference
    metrics. One figure-level legend identifies the controllers. Timing is
    deliberately absent from this figure: wall-clock data lives exclusively
    in ``benchmark.json`` and the generated benchmark tables.

    Args:
        run: the in-memory reference run.
        metrics: the ``reference_metrics(run)`` document (controllers only).
        output_path: PNG destination (parent directories are created).

    Returns:
        The figure.

    Example:
        >>> from vessel_gnc.reference import reference_metrics, run_reference_scenario
        >>> from vessel_gnc.visualization import plot_controller_comparison
        >>> run = run_reference_scenario()  # doctest: +SKIP  (120 s flagship)
        >>> figure = plot_controller_comparison(
        ...     run, reference_metrics(run), "assets/controller_comparison.png"
        ... )  # doctest: +SKIP
    """
    apply_style()
    bounds = run.config.truth_params
    available = (
        ("los_pid_v1", "LOS", run.los.result),
        ("nominal_nmpc_v1", "Nominal", run.nmpc.result),
        ("disturbance_aware_nmpc_v1", "Aware", run.disturbance_aware_nmpc.result),
        (
            "disturbance_aware_mpcc_v1",
            "MPCC",
            run.disturbance_aware_mpcc.result if run.disturbance_aware_mpcc is not None else None,
        ),
    )
    controllers = [
        (CONTROLLER_STYLES[component_id], column, result)
        for component_id, column, result in available
        if result is not None
    ]
    cross_track = []
    for _, _, result in controllers:
        _, _, cross = project_onto_path(np.column_stack([result.x, result.y]), run.path)
        cross_track.append(cross)

    fig, axes = plt.subplots(2, 2, figsize=DOUBLE_SIZE, constrained_layout=True)
    panels = (
        (
            axes[0, 0],
            cross_track,
            "cross-track [m]",
            "Cross-track error (positive = left of path)",
        ),
        (
            axes[0, 1],
            [result.thrust for _, _, result in controllers],
            "thrust [N]",
            "Surge thrust with physical bounds",
        ),
        (
            axes[1, 0],
            [result.yaw_moment for _, _, result in controllers],
            "yaw moment [N m]",
            "Yaw moment with physical bounds",
        ),
    )
    for ax, series, ylabel, title in panels:
        for (style, _, result), values in zip(controllers, series, strict=True):
            ax.plot(result.t, values, color=style["color"], lw=1.1, label=style["label"])
        ax.set_xlabel("t [s]")
        ax.set_ylabel(ylabel)
        ax.set_title(title)

    # Reference zero and the physical truth-plant bounds, labelled in place.
    axes[0, 0].axhline(0.0, color="k", lw=0.8)
    bound_panels = (
        (axes[0, 1], bounds.thrust_min, bounds.thrust_max),
        (axes[1, 0], bounds.moment_min, bounds.moment_max),
    )
    for ax, lower, upper in bound_panels:
        for bound in (lower, upper):
            ax.axhline(bound, color="r", ls=":", lw=1)
            ax.annotate(
                f"{bound:g}",
                xy=(1.0, bound),
                xycoords=("axes fraction", "data"),
                ha="right",
                va="bottom",
                color="r",
                fontsize=8,
            )

    legend_handles = [
        plt.Line2D([], [], color=style["color"], lw=1.5) for style, _, _ in controllers
    ]
    fig.legend(
        legend_handles,
        [style["label"] for style, _, _ in controllers],
        loc="outside upper right",
    )

    ax = axes[1, 1]
    ax.axis("off")
    controller_metrics = [metrics["controllers"][component_id] for component_id, _, _ in available]

    def cell(value: object, spec: str) -> str:
        """Format a metric cell, keeping the em dash for incomplete routes."""
        return "—" if value is None else f"{float(value):{spec}}"

    metric_rows = (
        ("RMS cross-track [m]", ".2f", lambda m: m["cross_track_rms_m"]),
        ("Max cross-track [m]", ".2f", lambda m: m["cross_track_max_m"]),
        (
            "RMS heading error [deg]",
            ".1f",
            lambda m: np.degrees(m["heading_error_rms_rad"]),
        ),
        ("Final progress [m]", ".1f", lambda m: m["path_progress_final_m"]),
        ("Progress fraction [-]", ".3f", lambda m: m["path_progress_fraction"]),
        ("Mean progress rate [m/s]", ".2f", lambda m: m["mean_progress_rate_m_s"]),
        ("Route completion [s]", ".1f", lambda m: m["route_completion_s"]),
        ("RMS thrust [N]", ".1f", lambda m: m["thrust_rms_N"]),
        ("Max yaw moment [N m]", ".1f", lambda m: m["moment_max_Nm"]),
        ("Any saturation [s]", ".1f", lambda m: m["any_saturation_duration_s"]),
    )
    rows = [("", *(column for _, column, _ in controllers))]
    for label, spec, value_of in metric_rows:
        rows.append((label, *(cell(value_of(m), spec) for m in controller_metrics)))
    table = ax.table(
        cellText=rows,
        colWidths=[0.33, 0.14, 0.19, 0.17, 0.17],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(8.5)
    table.scale(1.0, 1.6)
    ax.set_title("Comparison (deterministic metrics)")

    save_figure(fig, output_path)
    return fig


def plot_current_estimation(
    run: ReferenceRun,
    output_path: str | os.PathLike[str],
) -> plt.Figure:
    """Equivalent-current figure from the combined-uncertainty flagship.

    Physical current (solid) and the EKF equivalent-current state (dashed)
    are shown with their difference after the discarded estimator transient.
    The difference includes wind/model-mismatch confounders; it is not a
    standalone current-sensor error. Rendered from the disturbance-aware
    controller's estimator history so
    it shares the exact reference scenario and seed of the other assets.

    Args:
        run: the in-memory reference run.
        output_path: PNG destination (parent directories are created).

    Returns:
        The figure.

    Example:
        >>> from vessel_gnc.reference import run_reference_scenario
        >>> from vessel_gnc.visualization import plot_current_estimation
        >>> run = run_reference_scenario()  # doctest: +SKIP  (120 s flagship)
        >>> figure = plot_current_estimation(
        ...     run, "assets/current_estimation.png"
        ... )  # doctest: +SKIP
    """
    history = run.disturbance_aware_nmpc.estimator
    t = history.t
    true = history.current_true
    estimate = history.current_estimate
    transient = run.config.estimator_transient_s
    error = np.hypot(estimate[:, 0] - true[:, 0], estimate[:, 1] - true[:, 1])

    apply_style()
    fig, axes = plt.subplots(3, 1, figsize=(7.2, 7.8), sharex=True, constrained_layout=True)

    ax = axes[0]
    ax.plot(
        t,
        true[:, 0],
        color="tab:orange",
        lw=1.4,
        label="physical current (north)",
    )
    ax.plot(
        t,
        estimate[:, 0],
        color="tab:orange",
        lw=1.2,
        ls="--",
        label="EKF equivalent current (north)",
    )
    ax.set_ylabel("north [m/s]")
    ax.set_title("Equivalent-current state under combined uncertainty")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(alpha=0.3)

    ax = axes[1]
    ax.plot(
        t,
        true[:, 1],
        color="tab:blue",
        lw=1.4,
        label="physical current (east)",
    )
    ax.plot(
        t,
        estimate[:, 1],
        color="tab:blue",
        lw=1.2,
        ls="--",
        label="EKF equivalent current (east)",
    )
    ax.set_ylabel("east [m/s]")
    ax.legend(loc="best", framealpha=0.9)
    ax.grid(alpha=0.3)

    ax = axes[2]
    ax.plot(t, error, color="0.3", lw=1.4)
    ax.axvspan(0.0, transient, color="0.85", zorder=0)
    ax.axvline(
        transient,
        color="k",
        ls=":",
        lw=1,
        label=f"transient {transient:.0f} s (excluded)",
    )
    ax.set_xlabel("t [s]")
    ax.set_ylabel("vector difference [m/s]")
    ax.set_title("Difference from physical current (includes confounders)")
    ax.legend(loc="best", framealpha=0.9)

    save_figure(fig, output_path)
    return fig
