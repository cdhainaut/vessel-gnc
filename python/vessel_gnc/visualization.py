"""Plotting helpers for simulation results.

Visualization only: all numerical logic lives in ``simulation.py`` and the
C++ kernel. Figures and animations are generated from scripts, never edited
by hand.
"""

from __future__ import annotations

import os
from collections.abc import Callable
from pathlib import Path
from typing import TYPE_CHECKING

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.collections import LineCollection
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D

from vessel_gnc import _core
from vessel_gnc.plot_style import (
    SINGLE_SIZE,
    apply_style,
    compress_gif,
    save_figure,
)
from vessel_gnc.simulation import EnvironmentPolicy, SimulationResult

if TYPE_CHECKING:
    pass

__all__ = [
    "plot_trajectory",
    "animate_trajectory",
    "draw_vessel",
    "environment_arrows",
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
        facecolor="white",
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


def environment_arrows(
    ax,
    environment: _core.Environment | None,
    x0: float,
    y0: float,
    annotate: bool = False,
    estimated: bool = False,
) -> tuple[list, list[tuple]]:
    """Draw current/wind arrows anchored at ``(x0, y0)`` in data coordinates.

    Arrows are magnitude-scaled for visibility; the true values are reported
    either as legend handles (``annotate=False``) or as text next to each
    arrow (``annotate=True``). The wind arrow is anchored 3.5 m north of the
    current arrow so the arrows and their labels never overlap. With
    ``estimated=True`` the arrow is dashed and labelled as the EKF
    equivalent-current state used in the combined-uncertainty flagship.
    Zero components are skipped.

    Returns:
        ``(artists, handles)``: the drawn artists (so per-frame callers can
        remove them again) and ``(line, label)`` legend handles.
    """
    linestyle = "--" if estimated else "-"
    artists: list = []
    handles: list[tuple] = []
    if environment is None:
        return artists, handles

    if environment.current_north != 0.0 or environment.current_east != 0.0:
        vn, ve = environment.current_north, environment.current_east
        scale = 10.0  # m of arrow per m/s of current
        tip = (x0 + scale * vn, y0 + scale * ve)
        artists.append(
            ax.annotate(
                "",
                xy=tip,
                xytext=(x0, y0),
                arrowprops=dict(arrowstyle="->", color="tab:blue", lw=2, linestyle=linestyle),
                zorder=5,
            )
        )
        quantity = "equiv. current (EKF)" if estimated else "physical current"
        label = f"{quantity} ({np.hypot(vn, ve):.2f} m/s)"
        handles.append((plt.Line2D([], [], color="tab:blue", lw=2, ls=linestyle), label))
        if annotate:
            artists.append(ax.text(tip[0] + 0.2, tip[1] + 0.2, label, fontsize=8, color="tab:blue"))

    if environment.wind_north != 0.0 or environment.wind_east != 0.0:
        wn, we = environment.wind_north, environment.wind_east
        scale = 0.2  # m of arrow per N of wind force
        # Offset anchor so the wind arrow never overlaps the current arrow.
        y_wind = y0 + 3.5
        tip = (x0 + scale * wn, y_wind + scale * we)
        artists.append(
            ax.annotate(
                "",
                xy=tip,
                xytext=(x0, y_wind),
                arrowprops=dict(arrowstyle="->", color="tab:orange", lw=2),
                zorder=5,
            )
        )
        quantity = "equiv. wind (EKF)" if estimated else "wind"
        label = f"{quantity} ({np.hypot(wn, we):.0f} N)"
        handles.append((plt.Line2D([], [], color="tab:orange", lw=2), label))
        if annotate:
            artists.append(
                ax.text(tip[0] + 0.2, tip[1] + 0.2, label, fontsize=8, color="tab:orange")
            )

    return artists, handles


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
    apply_style()
    speed = np.hypot(result.u, result.v)

    fig, ax = plt.subplots(figsize=SINGLE_SIZE)
    ax.set_aspect("equal")

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
    (end_line,) = ax.plot(result.x[-1], result.y[-1], "x", color="tab:red", ms=10, mew=2)
    draw_vessel(ax, result.x[-1], result.y[-1], result.psi[-1])

    # Environment arrows (with legend entries carrying the true values).
    handles = [(start_line, "start"), (end_line, "end")]
    _, arrow_handles = environment_arrows(ax, environment, result.x[0], result.y[0])
    handles += arrow_handles
    if handles:
        artists, labels = zip(*handles, strict=True)
        ax.legend(list(artists), list(labels), loc="best", framealpha=0.9)

    ax.set_xlabel("x [m] (North)")
    ax.set_ylabel("y [m] (East)")
    ax.set_title(title)
    fig.tight_layout()

    if output_path is not None:
        save_figure(fig, output_path)
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
    dpi: int = 72,
    reference_path: np.ndarray | None = None,
    corridor_half_width_m: float | None = None,
    station_circle: tuple[np.ndarray, float] | None = None,
    follow_view_width_m: float | None = None,
    horizon: list[tuple[float, np.ndarray]] | None = None,
    horizon_label: str = "NMPC prediction",
    extra_trajectories: list[tuple[str, np.ndarray, str]] | None = None,
    extra_horizon: list[tuple[float, np.ndarray]] | None = None,
    extra_horizon_label: str = "MPCC prediction",
    progress_text: Callable[[float, float, float], str | None] | None = None,
) -> animation.FuncAnimation:
    """Render a top-down animation of the vessel along the trajectory.

    Scene: white hull with heading line, the travelled trail with a
    speed-coloured wake ribbon, environment arrows and a
    time/speed/heading overlay. With ``reference_path`` and ``horizon`` the
    scene becomes the flagship demo: the reference path is drawn dashed and
    the predictive horizon is shown ahead of the vessel, taken from the
    nearest recorded prediction (``horizon``: list of ``(t, traj)`` with
    ``traj`` a (6, N+1) or (8, N+1) predicted trajectory).

    ``extra_trajectories`` draws additional static controller trajectories as
    comparison lines (each ``(label, (M, 2) positions, color)``);
    ``extra_horizon`` draws a second, distinct predicted horizon series
    (e.g. the MPCC recorded predictions) with its own label. ``progress_text``
    is an optional ``(t, x, y) -> str | None`` callback whose returned line is
    appended to the overlay (used for compact path-progress information).

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
        wake_duration: length of the speed-coloured wake ribbon [s].
        corridor_half_width_m: optional operational track tolerance [m],
            drawn as a corridor band around ``reference_path``.
        station_circle: optional ``(station, radius_m)`` watch circle for
            station-keeping scenes, drawn as a translucent disc.
        follow_view_width_m: optional camera window [m]; when set, the view
            follows the vessel with this window width so the corridor, the
            vessel and the disturbance arrows stay readable.
        dpi: GIF resolution.
        reference_path: (M, 2) waypoints drawn as the reference path.
        horizon: recorded predictive predictions, shown per frame.
        horizon_label: legend entry for the prediction line.
        extra_trajectories: static comparison trajectories
            (``(label, (M, 2) positions, color)``).
        extra_horizon: recorded second prediction series, shown per frame
            (same ``(t, traj)`` shape as ``horizon``).
        extra_horizon_label: legend entry for the second prediction line.
        progress_text: optional callback returning a path-progress line.
    """
    apply_style()
    speed = np.hypot(result.u, result.v)

    fig, ax = plt.subplots(figsize=(9.0, 6.4))
    ax.set_aspect("equal")
    # Room under the axes for the shared legend row; the title must fit.
    fig.subplots_adjust(left=0.06, right=0.99, top=0.95, bottom=0.22)

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
    if station_circle is not None:
        station, radius = station_circle
        x_lo = min(x_lo, float(station[0]) - radius)
        x_hi = max(x_hi, float(station[0]) + radius)
        y_lo = min(y_lo, float(station[1]) - radius)
        y_hi = max(y_hi, float(station[1]) + radius)
    if extra_trajectories:
        for _, positions, _color in extra_trajectories:
            pts = np.asarray(positions)
            x_lo = min(x_lo, pts[:, 0].min())
            x_hi = max(x_hi, pts[:, 0].max())
            y_lo = min(y_lo, pts[:, 1].min())
            y_hi = max(y_hi, pts[:, 1].max())
    ax.set_xlim(x_lo - margin, x_hi + margin)
    ax.set_ylim(y_lo - margin, y_hi + margin)

    # Environmental field (policies only): current speed shading with the
    # velocity vectors on top, evaluated at each frame's time and position.
    field_xx, field_yy = np.meshgrid(
        np.linspace(x_lo - margin, x_hi + margin, 15),
        np.linspace(y_lo - margin, y_hi + margin, 11),
    )
    field_north = np.zeros_like(field_xx)
    field_east = np.zeros_like(field_xx)
    field_speed = np.zeros_like(field_xx)
    # Banded filled contours rather than a smooth gradient: they read as a
    # technical map, and flat colour regions keep the GIF palette small.
    field_levels = np.linspace(0.0, 0.3, 7)
    field_contours = ax.contourf(
        field_xx,
        field_yy,
        field_speed,
        levels=field_levels,
        cmap="Blues",
        alpha=0.5,
        zorder=0,
    )
    field_arrows = ax.quiver(
        field_xx,
        field_yy,
        field_north,
        field_east,
        color="0.35",
        alpha=0.5,
        units="xy",
        scale=0.06,
        width=0.3,
        zorder=1,
    )

    # Static elements.
    legend_entries = []
    if station_circle is not None:
        station, radius = station_circle
        ax.add_patch(
            plt.Circle(
                (float(station[0]), float(station[1])),
                radius,
                facecolor="tab:red",
                alpha=0.15,
                zorder=0,
            )
        )
        ax.plot(station[0], station[1], "+", color="tab:red", ms=12, mew=2)
        legend_entries.append(
            (
                plt.Rectangle((0, 0), 1, 1, facecolor="tab:red", alpha=0.25),
                f"watch circle +/-{radius:g} m",
            )
        )
    if reference_path is not None:
        if corridor_half_width_m is not None:
            # Operational track tolerance: the corridor the mission has to
            # hold around the reference, drawn under every trajectory.
            tangent = np.gradient(rp, axis=0)
            norm = np.linalg.norm(tangent, axis=1)
            norm[norm == 0.0] = 1.0
            offset = (
                corridor_half_width_m
                * np.column_stack([tangent[:, 1], -tangent[:, 0]])
                / norm[:, None]
            )
            corridor = np.concatenate([rp + offset, (rp - offset)[::-1]])
            ax.fill(
                corridor[:, 0],
                corridor[:, 1],
                color="tab:green",
                alpha=0.25,
                zorder=0,
                linewidth=0,
            )
            legend_entries.append(
                (
                    plt.Rectangle((0, 0), 1, 1, facecolor="tab:green", alpha=0.25),
                    f"+/-{corridor_half_width_m:g} m track tolerance",
                )
            )
        ax.plot(rp[:, 0], rp[:, 1], "k--", lw=1.2, zorder=1)
        legend_entries.append((plt.Line2D([], [], color="k", ls="--", lw=1.2), "reference"))
    if extra_trajectories:
        for label, positions, color in extra_trajectories:
            pts = np.asarray(positions)
            ax.plot(pts[:, 0], pts[:, 1], color=color, lw=1.0, alpha=0.55, zorder=2)
            legend_entries.append(
                (
                    plt.Line2D(
                        [],
                        [],
                        color=color,
                        lw=1.6,
                        alpha=0.85,
                        marker="o",
                        ms=3,
                    ),
                    label,
                )
            )
    if horizon:
        legend_entries.append(
            (
                plt.Line2D([], [], color="tab:cyan", lw=1.6, marker="o", ms=4),
                horizon_label,
            )
        )
    if extra_horizon:
        legend_entries.append(
            (
                plt.Line2D([], [], color="tab:purple", lw=1.6, marker="o", ms=3),
                extra_horizon_label,
            )
        )
    if legend_entries:
        artists, labels = zip(*legend_entries, strict=True)
        # Below the axes: the legend never covers the trajectory or the
        # prediction horizons, whatever the scenario extent.
        ax.legend(
            list(artists),
            list(labels),
            loc="upper center",
            bbox_to_anchor=(0.5, -0.14),
            ncol=3,
            framealpha=0.9,
        )
    ax.plot(result.x[0], result.y[0], "o", color="tab:green", ms=8)
    hull, heading = _create_vessel_artists(ax, scale=2.5)
    # Disturbance arrows ride with the vessel: they show what the boat is
    # responding to at each instant, true (solid) versus the EKF estimate
    # (dashed). Constant environments are drawn once at the first position.
    env_artists = []
    if not callable(environment):
        static_artists, _ = environment_arrows(
            ax, environment, result.x[0], result.y[0], annotate=False
        )
        env_artists += static_artists
    if not callable(estimated_environment):
        static_artists, _ = environment_arrows(
            ax,
            estimated_environment,
            result.x[0],
            result.y[0],
            annotate=False,
            estimated=True,
        )
        env_artists += static_artists
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

    # Per-frame artists. The recent trail is a speed-coloured ribbon; the
    # full taken path is already drawn as the controller track, so no
    # separate trail line duplicates it.
    wake = LineCollection([], cmap="viridis", lw=1.8, alpha=0.85, zorder=3)
    wake.set_clim(speed.min(), speed.max())
    ax.add_collection(wake)
    (horizon_line,) = ax.plot([], [], color="tab:cyan", lw=1.6, zorder=4)
    (horizon_end,) = ax.plot([], [], "o", color="tab:cyan", ms=5, zorder=4)
    (extra_horizon_line,) = ax.plot([], [], color="tab:purple", lw=1.6, zorder=4)
    (extra_horizon_end,) = ax.plot([], [], "o", color="tab:purple", ms=5, zorder=4)

    # Horizon shots sorted by time, for per-frame lookup.
    horizon_times = np.array([t for t, _ in horizon]) if horizon else np.empty(0)
    extra_horizon_times = np.array([t for t, _ in extra_horizon]) if extra_horizon else np.empty(0)

    wake_steps = max(1, int(wake_duration / result.dt))
    indices = np.arange(0, result.n_steps + 1, stride)
    # Data-per-axis ratio of the drawing box, so an equal-aspect chase
    # window exactly fills the axes without resizing the layout.
    box = ax.get_position()
    view_height_over_width = (box.height * fig.get_figheight()) / (box.width * fig.get_figwidth())

    def update(frame: int) -> tuple:
        nonlocal env_artists, field_contours
        i = int(indices[frame])
        if follow_view_width_m is not None:
            # Chase camera: world-fixed field and paths, vessel-centred view.
            half_w = follow_view_width_m / 2.0
            half_h = half_w * view_height_over_width
            ax.set_xlim(result.x[i] - half_w, result.x[i] + half_w)
            ax.set_ylim(result.y[i] - half_h, result.y[i] + half_h)
        if callable(environment) or callable(estimated_environment):
            # Per-frame environment: replace the previous arrows in place.
            for artist in env_artists:
                artist.remove()
            env_artists = []
            t_now = result.t[i]
            if callable(environment):
                frame_artists, _ = environment_arrows(
                    ax,
                    environment(t_now, result.x[i], result.y[i]),
                    result.x[i],
                    result.y[i],
                    annotate=False,
                )
                env_artists += frame_artists
                # Field map at this instant: banded speed contours plus vectors.
                for row in range(field_xx.shape[0]):
                    for col in range(field_xx.shape[1]):
                        cell = environment(t_now, field_xx[row, col], field_yy[row, col])
                        field_north[row, col] = cell.current_north
                        field_east[row, col] = cell.current_east
                field_speed[:] = np.hypot(field_north, field_east)
                field_contours.remove()
                field_contours = ax.contourf(
                    field_xx,
                    field_yy,
                    field_speed,
                    levels=field_levels,
                    cmap="Blues",
                    alpha=0.5,
                    zorder=0,
                )
                field_arrows.set_UVC(field_north, field_east)
            if callable(estimated_environment):
                frame_artists, _ = environment_arrows(
                    ax,
                    estimated_environment(t_now, result.x[i], result.y[i]),
                    result.x[i],
                    result.y[i],
                    annotate=False,
                    estimated=True,
                )
                env_artists += frame_artists
        j0 = max(0, i - wake_steps)
        wake_points = np.column_stack([result.x[j0:i], result.y[j0:i]])
        if len(wake_points) > 1:
            wake.set_segments(np.stack([wake_points[:-1], wake_points[1:]], axis=1))
            wake.set_array(speed[j0 : i - 1])
        else:
            wake.set_segments([])

        if horizon:
            # Nearest recorded prediction not later than the frame time.
            idx = int(np.searchsorted(horizon_times, result.t[i], side="right")) - 1
            traj = horizon[max(0, idx)][1]
            horizon_line.set_data(traj[0], traj[1])
            horizon_end.set_data([traj[0, -1]], [traj[1, -1]])
        else:
            horizon_line.set_data([], [])
            horizon_end.set_data([], [])

        if extra_horizon:
            idx = int(np.searchsorted(extra_horizon_times, result.t[i], side="right"))
            idx = max(0, idx - 1)
            traj = extra_horizon[idx][1]
            extra_horizon_line.set_data(traj[0], traj[1])
            extra_horizon_end.set_data([traj[0, -1]], [traj[1, -1]])
        else:
            extra_horizon_line.set_data([], [])
            extra_horizon_end.set_data([], [])

        _set_vessel_pose(hull, heading, result.x[i], result.y[i], result.psi[i])
        progress_line = ""
        if progress_text is not None:
            line = progress_text(result.t[i], result.x[i], result.y[i])
            if line:
                progress_line = line + "\n"
        info.set_text(
            progress_line
            + f"t = {result.t[i]:5.1f} s\n"
            + f"V = {speed[i]:4.2f} m/s\n"
            + f"psi = {np.degrees(result.psi[i]):6.1f} deg"
        )
        return (
            wake,
            hull,
            heading,
            info,
            horizon_line,
            horizon_end,
            extra_horizon_line,
            extra_horizon_end,
        )

    anim = animation.FuncAnimation(
        fig, update, frames=len(indices), interval=1000 // fps, blit=False
    )
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        anim.save(out, writer=animation.PillowWriter(fps=fps), dpi=dpi)
        compress_gif(out, colors=80)
    return anim
