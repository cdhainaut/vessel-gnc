"""Line-of-sight path-following guidance.

Pure geometry: path projection (signed cross-track error) and the LOS
desired-heading law. Numerical control logic lives in the C++ core
(``vessel_gnc._core`` controllers); this module is vectorized over points.
See docs/control.md §4 for the conventions.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "project_onto_path",
    "los_heading",
    "make_s_curve_path",
    "path_arc_lengths",
    "path_points",
    "path_reference",
]


def project_onto_path(
    points: np.ndarray, path: np.ndarray
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Project points onto the polyline path.

    Args:
        points: (N, 2) array of (x, y) positions [m].
        path: (M, 2) waypoints [m]; waypoints must be distinct.

    Returns:
        ``(segment, along_track, cross_track)`` arrays of length N:

        - segment: index of the closest segment;
        - along_track: distance [m] from the segment start to the projection
          (clamped to the segment);
        - cross_track: signed distance [m], **positive when the point is to
          the left (port) of the path direction**.
    """
    points = np.atleast_2d(np.asarray(points, dtype=float))
    path = np.asarray(path, dtype=float)
    if path.ndim != 2 or path.shape[1] != 2 or path.shape[0] < 2:
        raise ValueError("path must be a (M, 2) array with M >= 2")

    segs = np.diff(path, axis=0)  # (M-1, 2)
    seg_len = np.hypot(segs[:, 0], segs[:, 1])
    if np.any(seg_len == 0.0):
        raise ValueError("path contains duplicate waypoints (zero-length segment)")
    unit = segs / seg_len[:, None]

    rel = points[:, None, :] - path[:-1][None, :, :]  # (N, M-1, 2)
    along = rel[:, :, 0] * unit[:, 0][None, :] + rel[:, :, 1] * unit[:, 1][None, :]
    along = np.clip(along, 0.0, seg_len[None, :])
    proj = path[:-1][None, :, :] + along[:, :, None] * unit[None, :, :]
    dist2 = np.sum((proj - points[:, None, :]) ** 2, axis=2)  # (N, M-1)
    segment = np.argmin(dist2, axis=1)

    seg = segment
    n = np.arange(len(points))
    along_track = along[n, seg]
    # Signed distance: positive = left (port) of the path direction.
    rel_seg = points[n] - path[seg]
    cross_track = rel_seg[:, 0] * unit[seg, 1] - rel_seg[:, 1] * unit[seg, 0]
    return segment, along_track, cross_track


def los_heading(points: np.ndarray, path: np.ndarray, lookahead: float) -> np.ndarray:
    """Desired LOS heading [rad] at each point (from North, clockwise positive).

    The lookahead point lies ``lookahead`` [m] along the path from the
    projection of the point (clamped to the path end). The desired heading is
    the bearing to that point.
    """
    if lookahead <= 0.0:
        raise ValueError("lookahead must be positive")
    points = np.atleast_2d(np.asarray(points, dtype=float))
    path = np.asarray(path, dtype=float)

    segment, along_track, _ = project_onto_path(points, path)

    # Absolute arc length of the lookahead point, then locate its segment.
    segs = np.diff(path, axis=0)
    seg_len = np.hypot(segs[:, 0], segs[:, 1])
    cumlen = np.concatenate([[0.0], np.cumsum(seg_len)])
    s_abs = cumlen[segment] + along_track + lookahead
    s_abs = np.clip(s_abs, 0.0, cumlen[-1])

    seg_los = np.searchsorted(cumlen, s_abs, side="right") - 1
    seg_los = np.clip(seg_los, 0, len(seg_len) - 1)
    frac = (s_abs - cumlen[seg_los]) / seg_len[seg_los]
    los = path[seg_los] + frac[:, None] * segs[seg_los]

    return np.arctan2(los[:, 1] - points[:, 1], los[:, 0] - points[:, 0])


def path_arc_lengths(path: np.ndarray) -> np.ndarray:
    """Cumulative arc length [m] at each waypoint (length M, starts at 0)."""
    path = np.asarray(path, dtype=float)
    return np.concatenate([[0.0], np.cumsum(np.hypot(*np.diff(path, axis=0).T))])


def path_points(path: np.ndarray, arc_lengths: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Positions and tangent headings at absolute arc lengths along the path.

    Arc lengths are clamped to the path ends. Returns ``(refs, psi_refs)`` of
    shape (n, 2) and (n,) with ``psi_refs`` the path tangent heading [rad].
    """
    path = np.asarray(path, dtype=float)
    arc_lengths = np.asarray(arc_lengths, dtype=float)
    cumlen = path_arc_lengths(path)
    s = np.clip(arc_lengths, 0.0, cumlen[-1])
    seg_len = np.diff(cumlen)
    seg_idx = np.clip(np.searchsorted(cumlen, s, side="right") - 1, 0, len(seg_len) - 1)
    frac = (s - cumlen[seg_idx]) / seg_len[seg_idx]
    segs = np.diff(path, axis=0)
    refs = path[seg_idx] + frac[:, None] * segs[seg_idx]
    psi_refs = np.arctan2(segs[seg_idx, 1], segs[seg_idx, 0])
    return refs, psi_refs


def path_reference(
    path: np.ndarray, s0: float, speed_ref: float, dt: float, n_steps: int
) -> tuple[np.ndarray, np.ndarray]:
    """Reference trajectory along the path from an absolute arc-length anchor.

    The references lie at arc lengths ``s0 + k * speed_ref * dt``
    (k = 1..n), clamped to the path end. ``s0`` is the mission-schedule
    position [m] (e.g. ``speed_ref * t`` for trajectory tracking, or the arc
    length of the vessel's projection for path following).

    Returns:
        ``(refs, psi_refs)``: (n, 2) positions and (n,) tangent headings [rad].
    """
    arc = s0 + speed_ref * dt * np.arange(1, n_steps + 1)
    return path_points(path, arc)


def make_s_curve_path() -> np.ndarray:
    """Reference path for the flagship scenario (plan §10): a straight start,
    two large turns, one tighter turn and a final straight. (M, 2) waypoints,
    x = North, y = East. Total length ~200 m.
    """
    return np.array(
        [
            [0.0, 0.0],
            [30.0, 0.0],
            [55.0, 15.0],
            [70.0, 40.0],
            [85.0, 42.0],
            [110.0, 42.0],
            [125.0, 22.0],
            [145.0, 18.0],
            [165.0, 35.0],
        ]
    )
