"""Plotting helpers for simulation results.

Visualization only: all numerical logic lives in ``simulation.py`` and the
C++ kernel. Figures and animations are generated from scripts, never edited
by hand.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D

from vessel_gnc import _core
from vessel_gnc.guidance import project_onto_path
from vessel_gnc.simulation import EnvironmentPolicy, SimulationResult

if TYPE_CHECKING:
    from vessel_gnc.reference import ReferenceRun

__all__ = [
    "plot_trajectory",
    "animate_trajectory",
    "draw_vessel",
    "plot_reference_trajectories",
    "plot_controller_comparison",
    "plot_current_estimation",
]

# Hull outline in the body frame [m] (x forward, y starboard): a ~1.5 m long,
# ~0.5 m beam small-USV shape.
HULL = np.array(
    [
        [0.75, 0.0],
        [0.45, 0.22],
        [0.1, 0.25],
        [-0.7, 0.2],
        [-0.75, 0.0],
        [-0.7, -0.2],
        [0.1, -0.25],
        [0.45, -0.22],
    ]
)


def draw_vessel(ax, x: float, y: float, psi: float, scale: float = 1.0):
    """Draw the vessel hull and heading line at pose ``(x, y, psi)``.

    Coordinates are in the data frame (x North, y East, psi clockwise from
    North). Returns ``(hull, heading)`` for later per-frame updates.
    """
    hull, heading = _create_vessel_artists(ax, scale)
    _set_vessel_pose(hull, heading, x, y, psi)
    return hull, heading


def _create_vessel_artists(ax, scale: float = 1.0):
    """Create the hull polygon and heading line (updated via ``_set_vessel_pose``)."""
    hull = Polygon(
        scale * HULL,
        closed=True,
        facecolor="0.25",
        edgecolor="black",
        lw=1.0,
        zorder=5,
    )
    ax.add_patch(hull)
    (heading,) = ax.plot([], [], color="0.1", lw=1.2, zorder=4)
    return hull, heading


def _set_vessel_pose(hull, heading, x: float, y: float, psi: float) -> None:
    hull.set_transform(Affine2D().rotate(psi).translate(x, y) + hull.axes.transData)
    heading.set_data([x, x + 1.3 * np.cos(psi)], [y, y + 1.3 * np.sin(psi)])


def _environment_arrows(
    ax,
    environment: _core.Environment | None,
    x0: float,
    y0: float,
    annotate: bool = False,
    estimated: bool = False,
) -> list[tuple]:
    """Draw current/wind arrows anchored at ``(x0, y0)`` in data coordinates.

    Arrows are magnitude-scaled for visibility; the true values are reported
    either as legend handles (``annotate=False``) or as text next to each
    arrow (``annotate=True``). The wind arrow is anchored 1.5 m north of the
    current arrow so the two never overlap when they point in the same
    direction. With ``estimated=True`` the arrow is dashed and labelled as
    the EKF equivalent-current state used in the combined-uncertainty
    flagship. Zero components are skipped.
    """
    linestyle = "--" if estimated else "-"
    handles: list[tuple] = []
    if environment is None:
        return handles

    if environment.current_north != 0.0 or environment.current_east != 0.0:
        vn, ve = environment.current_north, environment.current_east
        scale = 10.0  # m of arrow per m/s of current
        tip = (x0 + scale * vn, y0 + scale * ve)
        ax.annotate(
            "",
            xy=tip,
            xytext=(x0, y0),
            arrowprops=dict(
                arrowstyle="->", color="tab:blue", lw=2, linestyle=linestyle
            ),
            zorder=5,
        )
        quantity = "equiv. current (EKF)" if estimated else "physical current"
        label = f"{quantity} ({np.hypot(vn, ve):.2f} m/s)"
        handles.append(
            (plt.Line2D([], [], color="tab:blue", lw=2, ls=linestyle), label)
        )
        if annotate:
            ax.text(tip[0] + 0.2, tip[1] + 0.2, label, fontsize=8, color="tab:blue")

    if environment.wind_north != 0.0 or environment.wind_east != 0.0:
        wn, we = environment.wind_north, environment.wind_east
        scale = 0.2  # m of arrow per N of wind force
        # Offset anchor so the wind arrow never overlaps the current arrow.
        y_wind = y0 + 1.5
        tip = (x0 + scale * wn, y_wind + scale * we)
        ax.annotate(
            "",
            xy=tip,
            xytext=(x0, y_wind),
            arrowprops=dict(arrowstyle="->", color="tab:orange", lw=2),
            zorder=5,
        )
        label = f"wind ({np.hypot(wn, we):.0f} N)"
        handles.append((plt.Line2D([], [], color="tab:orange", lw=2), label))
        if annotate:
            ax.text(tip[0] + 0.2, tip[1] + 0.2, label, fontsize=8, color="tab:orange")

    return handles


def plot_trajectory(
    result: SimulationResult,
    output_path: str | os.PathLike[str] | None = None,
    environment: _core.Environment | None = None,
    title: str = "Vessel trajectory",
) -> plt.Figure:
    """Plot the vessel path coloured by speed, with heading and environment arrows.

    Saves the figure to ``output_path`` when given (parent directories are
    created) and returns the figure.

    Example:
        >>> from vessel_gnc import _core, simulate
        >>> from vessel_gnc.visualization import plot_trajectory
        >>> result = simulate(30.0, 0.01, control=_core.Control(thrust=40.0))
        >>> figure = plot_trajectory(result, output_path="results/example.png")
    """
    speed = np.hypot(result.u, result.v)

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    # Path coloured by speed.
    points = ax.scatter(result.x, result.y, c=speed, s=6, cmap="viridis", zorder=3)
    fig.colorbar(points, ax=ax, label="speed [m/s]")

    # Heading arrows every ~25 samples (1.5 m long in data units).
    step = max(1, result.n_steps // 25)
    idx = slice(0, result.n_steps + 1, step)
    ax.quiver(
        result.x[idx],
        result.y[idx],
        1.5 * np.cos(result.psi[idx]),
        1.5 * np.sin(result.psi[idx]),
        units="xy",
        scale=1.0,
        width=0.012,
        color="0.25",
        alpha=0.85,
        headwidth=4,
        headlength=5,
        zorder=4,
    )

    # Start / end markers and the vessel at its final pose.
    (start_line,) = ax.plot(result.x[0], result.y[0], "o", color="tab:green", ms=8)
    (end_line,) = ax.plot(
        result.x[-1], result.y[-1], "x", color="tab:red", ms=10, mew=2
    )
    draw_vessel(ax, result.x[-1], result.y[-1], result.psi[-1])

    # Environment arrows (with legend entries carrying the true values).
    handles = [(start_line, "start"), (end_line, "end")]
    handles += _environment_arrows(ax, environment, result.x[0], result.y[0])
    if handles:
        artists, labels = zip(*handles, strict=True)
        ax.legend(list(artists), list(labels), loc="best", framealpha=0.9)

    ax.set_xlabel("x [m] (North)")
    ax.set_ylabel("y [m] (East)")
    ax.set_title(title)
    fig.tight_layout()

    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig


def animate_trajectory(
    result: SimulationResult,
    output_path: str | os.PathLike[str] | None = None,
    environment: EnvironmentPolicy | _core.Environment | None = None,
    estimated_environment: EnvironmentPolicy | None = None,
    title: str = "Vessel trajectory",
    stride: int = 10,
    fps: int = 10,
    wake_duration: float = 8.0,
    dpi: int = 90,
    reference_path: np.ndarray | None = None,
    horizon: list[tuple[float, np.ndarray]] | None = None,
    horizon_label: str = "NMPC prediction",
) -> animation.FuncAnimation:
    """Render a top-down animation of the vessel along the trajectory.

    Scene: hull outline with heading line, past trajectory (faint full path,
    brighter trail), a fading speed-coloured wake, environment arrows and a
    time/speed/heading overlay. With ``reference_path`` and ``horizon`` the
    scene becomes the flagship demo: the reference path is drawn dashed and
    the NMPC predicted trajectory is shown ahead of the vessel, taken from
    the nearest recorded prediction (``horizon``: list of ``(t, traj)`` with
    ``traj`` a (6, N+1) predicted trajectory).

    Args:
        result: simulation result to animate.
        output_path: save the animation as a GIF here when given.
        environment: ambient current and wind, either constant or sampled
            per frame (``t -> Environment``), drawn as corner arrows.
        estimated_environment: estimated current (e.g. from the EKF),
            sampled per frame and drawn dashed next to the true arrows.
        title: figure title.
        stride: display one sample every ``stride`` integration steps
            (frame period = ``dt * stride``).
        fps: GIF frame rate.
        wake_duration: length of the fading wake trail [s].
        dpi: GIF resolution.
        reference_path: (M, 2) waypoints drawn as the reference path.
        horizon: recorded NMPC predictions, shown per frame.
        horizon_label: legend entry for the prediction line (e.g.
            "NMPC prediction (10 s horizon)").
    """
    speed = np.hypot(result.u, result.v)

    fig, ax = plt.subplots(figsize=(7.2, 6.4))
    ax.set_aspect("equal")
    ax.grid(alpha=0.3)

    # Fixed view over the trajectory (and the reference path), with margin.
    margin = 3.0
    x_lo, x_hi = result.x.min(), result.x.max()
    y_lo, y_hi = result.y.min(), result.y.max()
    if reference_path is not None:
        rp = np.asarray(reference_path)
        x_lo = min(x_lo, rp[:, 0].min())
        x_hi = max(x_hi, rp[:, 0].max())
        y_lo = min(y_lo, rp[:, 1].min())
        y_hi = max(y_hi, rp[:, 1].max())
    ax.set_xlim(x_lo - margin, x_hi + margin)
    ax.set_ylim(y_lo - margin, y_hi + margin)

    # Static elements.
    (path_line,) = ax.plot(result.x, result.y, color="0.65", lw=0.8, zorder=1)
    legend_entries = []
    if reference_path is not None:
        ax.plot(rp[:, 0], rp[:, 1], "k--", lw=1.2, zorder=1)
        legend_entries.append(
            (plt.Line2D([], [], color="k", ls="--", lw=1.2), "reference")
        )
    if horizon:
        legend_entries.append(
            (
                plt.Line2D([], [], color="tab:cyan", lw=1.6, marker="o", ms=4),
                horizon_label,
            )
        )
    if legend_entries:
        artists, labels = zip(*legend_entries, strict=True)
        ax.legend(list(artists), list(labels), loc="upper right", framealpha=0.9)
    ax.plot(result.x[0], result.y[0], "o", color="tab:green", ms=8)
    hull, heading = _create_vessel_artists(ax)
    # Environment arrows are per-frame when the environment is time-varying.
    env_arrow_anchor = (x_lo - margin + 1.5, y_lo - margin + 1.5)
    env_artists = []
    if not callable(environment):
        env_artists += _environment_arrows(
            ax, environment, *env_arrow_anchor, annotate=True
        )
    if not callable(estimated_environment):
        env_artists += _environment_arrows(
            ax, estimated_environment, *env_arrow_anchor, annotate=True, estimated=True
        )
    info = ax.text(
        0.02,
        0.97,
        "",
        transform=ax.transAxes,
        va="top",
        family="monospace",
        fontsize=10,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white", alpha=0.85),
        zorder=6,
    )
    ax.set_xlabel("x [m] (North)")
    ax.set_ylabel("y [m] (East)")
    ax.set_title(title)

    # Per-frame artists.
    (trail_line,) = ax.plot([], [], color="0.35", lw=1.6, zorder=3)
    wake = ax.scatter([], [], s=14, zorder=2)
    (horizon_line,) = ax.plot([], [], color="tab:cyan", lw=1.6, zorder=4)
    (horizon_end,) = ax.plot([], [], "o", color="tab:cyan", ms=5, zorder=4)

    # Horizon shots sorted by time, for per-frame lookup.
    horizon_times = np.array([t for t, _ in horizon]) if horizon else np.empty(0)

    wake_steps = max(1, int(wake_duration / result.dt))
    indices = np.arange(0, result.n_steps + 1, stride)

    def update(frame: int) -> tuple:
        i = int(indices[frame])
        frame_env = env_artists
        if callable(environment) or callable(estimated_environment):
            for artist in env_artists:
                artist.remove()
            frame_env = []
            t_now = result.t[i]
            if callable(environment):
                frame_env += _environment_arrows(
                    ax, environment(t_now), *env_arrow_anchor, annotate=True
                )
            if callable(estimated_environment):
                frame_env += _environment_arrows(
                    ax,
                    estimated_environment(t_now),
                    *env_arrow_anchor,
                    annotate=True,
                    estimated=True,
                )
        trail_line.set_data(result.x[:i], result.y[:i])

        j0 = max(0, i - wake_steps)
        xw, yw = result.x[j0:i], result.y[j0:i]
        if len(xw) > 1:
            age = np.arange(len(xw), dtype=float) / (len(xw) - 1)
            colors = plt.cm.viridis(0.3 + 0.7 * age)  # old = dark, recent = bright
            colors[:, 3] = 0.15 + 0.6 * age
            wake.set_offsets(np.column_stack([xw, yw]))
            wake.set_facecolors(colors)
        else:
            wake.set_offsets(np.empty((0, 2)))

        if horizon:
            # Nearest recorded prediction not later than the frame time.
            idx = int(np.searchsorted(horizon_times, result.t[i], side="right")) - 1
            traj = horizon[max(0, idx)][1]
            horizon_line.set_data(traj[0], traj[1])
            horizon_end.set_data([traj[0, -1]], [traj[1, -1]])
        else:
            horizon_line.set_data([], [])
            horizon_end.set_data([], [])

        _set_vessel_pose(hull, heading, result.x[i], result.y[i], result.psi[i])
        info.set_text(
            f"t = {result.t[i]:5.1f} s\n"
            f"V = {speed[i]:4.2f} m/s\n"
            f"psi = {np.degrees(result.psi[i]):6.1f} deg"
        )
        return (trail_line, wake, hull, heading, info, horizon_line, horizon_end)

    anim = animation.FuncAnimation(
        fig, update, frames=len(indices), interval=1000 // fps, blit=False
    )
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        anim.save(out, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    return anim


def plot_reference_trajectories(
    run: ReferenceRun,
    output_path: str | os.PathLike[str],
) -> plt.Figure:
    """Reference-run trajectory overview: LOS vs NMPC with prediction horizons.

    Draws the reference path, all controller trajectories, the
    disturbance-aware prediction horizons and the true environment sampled
    at four times along the aware trajectory. Saved
    to ``output_path`` (the ignored results figure of the reference
    pipeline).

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

    fig, ax = plt.subplots(figsize=(8.0, 6.6))
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
    for _, traj in run.disturbance_aware_nmpc.horizon:
        ax.plot(traj[0], traj[1], color="tab:cyan", lw=1.0, alpha=0.7)
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
        handles = _environment_arrows(
            ax,
            run.config.environment.sample(shot_t),
            x0,
            y0,
            annotate=False,
        )
        if index == 0:
            environment_handles = handles

    ax.set_aspect("equal")
    ax.set_xlabel("x [m] (North)")
    ax.set_ylabel("y [m] (East)")
    ax.set_title("LOS vs nominal and disturbance-aware NMPC")
    plot_handles, plot_labels = ax.get_legend_handles_labels()
    plot_handles.extend(handle for handle, _ in environment_handles)
    plot_labels.extend(label for _, label in environment_handles)
    ax.legend(plot_handles, plot_labels, loc="best", framealpha=0.9)
    ax.grid(alpha=0.3)
    fig.tight_layout()

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig


def plot_controller_comparison(
    run: ReferenceRun,
    metrics: dict[str, object],
    output_path: str | os.PathLike[str],
) -> plt.Figure:
    """Deterministic LOS/nominal/aware comparison (tracking and controls).

    Timing is deliberately absent from this figure: wall-clock data lives
    exclusively in ``benchmark.json`` and the generated benchmark tables.
    The four panels show the signed cross-track error, the applied surge
    thrust and yaw moment (with the physical truth-plant actuator bounds)
    and a metrics table assembled from the deterministic reference metrics.

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
    path = run.path
    los_result = run.los.result
    nominal_result = run.nmpc.result
    aware_result = run.disturbance_aware_nmpc.result
    los_metrics = metrics["controllers"]["los_pid_v1"]
    nominal_metrics = metrics["controllers"]["nominal_nmpc_v1"]
    aware_metrics = metrics["controllers"]["disturbance_aware_nmpc_v1"]
    _, _, cross_los = project_onto_path(
        np.column_stack([los_result.x, los_result.y]), path
    )
    _, _, cross_nominal = project_onto_path(
        np.column_stack([nominal_result.x, nominal_result.y]), path
    )
    _, _, cross_aware = project_onto_path(
        np.column_stack([aware_result.x, aware_result.y]), path
    )
    bounds = run.config.truth_params

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), constrained_layout=True)

    ax = axes[0, 0]
    ax.plot(los_result.t, cross_los, color="0.6", lw=1.2, label="LOS")
    ax.plot(
        nominal_result.t,
        cross_nominal,
        color="tab:blue",
        lw=1.1,
        label="nominal NMPC",
    )
    ax.plot(
        aware_result.t,
        cross_aware,
        color="tab:green",
        lw=1.2,
        label="aware NMPC",
    )
    ax.axhline(0.0, color="k", lw=0.8)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("cross-track [m]")
    ax.set_title("Cross-track error (positive = left of path)")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    ax = axes[0, 1]
    ax.plot(los_result.t, los_result.thrust, color="0.6", lw=1.0, label="LOS")
    ax.plot(
        nominal_result.t,
        nominal_result.thrust,
        color="tab:blue",
        lw=1.0,
        label="nominal NMPC",
    )
    ax.plot(
        aware_result.t,
        aware_result.thrust,
        color="tab:green",
        lw=1.0,
        label="aware NMPC",
    )
    ax.axhline(bounds.thrust_max, color="r", ls=":", lw=1)
    ax.axhline(bounds.thrust_min, color="r", ls=":", lw=1)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("thrust [N]")
    ax.set_title("Surge thrust with physical bounds")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(
        los_result.t,
        los_result.yaw_moment,
        color="0.6",
        lw=1.0,
        label="LOS",
    )
    ax.plot(
        nominal_result.t,
        nominal_result.yaw_moment,
        color="tab:blue",
        lw=1.0,
        label="nominal NMPC",
    )
    ax.plot(
        aware_result.t,
        aware_result.yaw_moment,
        color="tab:green",
        lw=1.0,
        label="aware NMPC",
    )
    ax.axhline(bounds.moment_max, color="r", ls=":", lw=1)
    ax.axhline(bounds.moment_min, color="r", ls=":", lw=1)
    ax.set_xlabel("t [s]")
    ax.set_ylabel("yaw moment [N m]")
    ax.set_title("Yaw moment with physical bounds")
    ax.legend(loc="best")
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    ax.axis("off")
    rows = [
        ("", "LOS", "Nominal", "Aware"),
        (
            "RMS cross-track [m]",
            f"{los_metrics['cross_track_rms_m']:.2f}",
            f"{nominal_metrics['cross_track_rms_m']:.2f}",
            f"{aware_metrics['cross_track_rms_m']:.2f}",
        ),
        (
            "Max cross-track [m]",
            f"{los_metrics['cross_track_max_m']:.2f}",
            f"{nominal_metrics['cross_track_max_m']:.2f}",
            f"{aware_metrics['cross_track_max_m']:.2f}",
        ),
        (
            "RMS heading error [deg]",
            f"{np.degrees(los_metrics['heading_error_rms_rad']):.1f}",
            f"{np.degrees(nominal_metrics['heading_error_rms_rad']):.1f}",
            f"{np.degrees(aware_metrics['heading_error_rms_rad']):.1f}",
        ),
        (
            "RMS thrust [N]",
            f"{los_metrics['thrust_rms_N']:.1f}",
            f"{nominal_metrics['thrust_rms_N']:.1f}",
            f"{aware_metrics['thrust_rms_N']:.1f}",
        ),
        (
            "Max yaw moment [N m]",
            f"{los_metrics['moment_max_Nm']:.1f}",
            f"{nominal_metrics['moment_max_Nm']:.1f}",
            f"{aware_metrics['moment_max_Nm']:.1f}",
        ),
        (
            "Any saturation [s]",
            f"{los_metrics['any_saturation_duration_s']:.1f}",
            f"{nominal_metrics['any_saturation_duration_s']:.1f}",
            f"{aware_metrics['any_saturation_duration_s']:.1f}",
        ),
    ]
    table = ax.table(
        cellText=rows,
        colWidths=[0.43, 0.19, 0.19, 0.19],
        loc="center",
        cellLoc="center",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)
    ax.set_title("Comparison (deterministic metrics)")

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
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

    fig, axes = plt.subplots(
        3, 1, figsize=(7.2, 7.8), sharex=True, constrained_layout=True
    )

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
    ax.set_ylabel("current-equivalent velocity [m/s]")
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
    ax.set_ylabel("current-equivalent velocity [m/s]")
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
    ax.grid(alpha=0.3)

    out = Path(output_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    return fig
