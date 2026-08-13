"""Plotting helpers for simulation results.

Visualization only: all numerical logic lives in ``simulation.py`` and the
C++ kernel. Figures and animations are generated from scripts, never edited
by hand.
"""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib import animation
from matplotlib.patches import Polygon
from matplotlib.transforms import Affine2D

from vessel_gnc import _core
from vessel_gnc.simulation import SimulationResult

__all__ = ["plot_trajectory", "animate_trajectory", "draw_vessel"]

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
) -> list[tuple]:
    """Draw current/wind arrows anchored at ``(x0, y0)`` in data coordinates.

    Arrows are magnitude-scaled for visibility; the true values are reported
    either as legend handles (``annotate=False``) or as text next to each
    arrow (``annotate=True``). The wind arrow is anchored 1.5 m north of the
    current arrow so the two never overlap when they point in the same
    direction. Zero components are skipped.
    """
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
            arrowprops=dict(arrowstyle="->", color="tab:blue", lw=2),
            zorder=5,
        )
        label = f"current ({np.hypot(vn, ve):.2f} m/s)"
        handles.append((plt.Line2D([], [], color="tab:blue", lw=2), label))
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
    (end_line,) = ax.plot(result.x[-1], result.y[-1], "x", color="tab:red", ms=10, mew=2)
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
    environment: _core.Environment | None = None,
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
        environment: ambient current and wind, drawn as corner arrows.
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
        legend_entries.append((plt.Line2D([], [], color="k", ls="--", lw=1.2), "reference"))
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
    _environment_arrows(
        ax,
        environment,
        x_lo - margin + 1.5,
        y_lo - margin + 1.5,
        annotate=True,
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
        return trail_line, wake, hull, heading, info, horizon_line, horizon_end

    anim = animation.FuncAnimation(
        fig, update, frames=len(indices), interval=1000 // fps, blit=False
    )
    if output_path is not None:
        out = Path(output_path)
        out.parent.mkdir(parents=True, exist_ok=True)
        anim.save(out, writer=animation.PillowWriter(fps=fps), dpi=dpi)
    return anim
