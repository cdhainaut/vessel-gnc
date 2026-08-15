"""Quantitative metrics for path-following controller comparison.

Used by the examples and by the README comparison table (plan §13).
"""

from __future__ import annotations

import numpy as np

from vessel_gnc import _core
from vessel_gnc.guidance import los_heading, project_onto_path
from vessel_gnc.simulation import SimulationResult

__all__ = ["path_following_metrics"]


def path_following_metrics(
    result: SimulationResult,
    path: np.ndarray,
    lookahead: float,
    *,
    params: _core.ModelParams | None = None,
    saturation_threshold: float = 0.99,
) -> dict[str, float]:
    """Path-following metrics over the whole run (SI units, angles in rad).

    Cross-track statistics use the absolute signed cross-track error
    (positive = left of the path direction, see guidance.project_onto_path).
    Heading errors are wrapped to (-pi, pi]. Actuator effort uses the applied
    (post-actuator) histories, not the raw commands.

    Saturation duration: for each left-closed simulation interval
    ``[t_k, t_{k+1})`` a channel is saturated when its applied actuator value
    is within ``1 - saturation_threshold`` of that channel's full physical
    bound span from either ``ModelParams`` bound, i.e. it reaches at least
    ``saturation_threshold`` (default 99 %) of the span from the nearer bound.
    Duration is the number of flagged intervals multiplied by the integration
    step ``dt``; the final sample (``t = duration``) bounds no interval and
    contributes no duration. Thrust, yaw moment and their union (no double
    counting) are reported separately.

    Args:
        result: simulation history (state and applied actuator values).
        path: (M, 2) reference waypoints [m].
        lookahead: LOS lookahead distance [m].
        params: model parameters providing the physical actuator bounds
            (default: ``_core.default_params()``).
        saturation_threshold: fraction of the full bound span required for a
            channel to count as saturated (0.99 = within 1 % of a bound).

    Returns:
        A JSON-serializable dict with cross-track error, LOS heading error,
        actuator effort and saturation-duration statistics.

    Example:
        >>> import json
        >>> from vessel_gnc import _core, simulate
        >>> from vessel_gnc.guidance import make_s_curve_path
        >>> from vessel_gnc.metrics import path_following_metrics
        >>> result = simulate(30.0, 0.01, control=_core.Control(thrust=40.0))
        >>> metrics = path_following_metrics(result, make_s_curve_path(), 8.0)
        >>> json.dumps(metrics)  # doctest: +SKIP
    """
    params = params if params is not None else _core.default_params()
    if not 0.5 <= saturation_threshold <= 1.0:
        raise ValueError("saturation_threshold must be in [0.5, 1.0]")

    points = np.column_stack([result.x, result.y])
    _, _, cross = project_onto_path(points, path)
    cross_abs = np.abs(cross)
    psi_los = los_heading(points, path, lookahead)
    heading_error = np.arctan2(
        np.sin(psi_los - result.psi), np.cos(psi_los - result.psi)
    )

    # Saturation over the left-closed intervals [t_k, t_{k+1}): only the
    # first n_steps samples bound an interval (the final sample contributes
    # no duration).
    n = result.n_steps
    thrust_sat = _saturated_intervals(
        result.thrust[:n], params.thrust_min, params.thrust_max, saturation_threshold
    )
    moment_sat = _saturated_intervals(
        result.yaw_moment[:n],
        params.moment_min,
        params.moment_max,
        saturation_threshold,
    )

    return {
        "cross_track_rms_m": float(np.sqrt(np.mean(cross**2))),
        "cross_track_p95_m": float(np.percentile(cross_abs, 95)),
        "cross_track_max_m": float(np.max(cross_abs)),
        "heading_error_rms_rad": float(np.sqrt(np.mean(heading_error**2))),
        "heading_error_max_rad": float(np.max(np.abs(heading_error))),
        "thrust_rms_N": float(np.sqrt(np.mean(result.thrust**2))),
        "thrust_max_N": float(np.max(np.abs(result.thrust))),
        "moment_rms_Nm": float(np.sqrt(np.mean(result.yaw_moment**2))),
        "moment_max_Nm": float(np.max(np.abs(result.yaw_moment))),
        "thrust_saturation_duration_s": float(np.count_nonzero(thrust_sat)) * result.dt,
        "moment_saturation_duration_s": float(np.count_nonzero(moment_sat)) * result.dt,
        "any_saturation_duration_s": float(np.count_nonzero(thrust_sat | moment_sat))
        * result.dt,
    }


def _saturated_intervals(
    values: np.ndarray, lower: float, upper: float, threshold: float
) -> np.ndarray:
    """Boolean per-interval flags: value k saturates the channel on
    ``[t_k, t_{k+1})``.

    A value saturates when it is within ``1 - threshold`` of the full bound
    span from either bound: ``value >= lower + threshold * span`` or
    ``value <= lower + (1 - threshold) * span``.
    """
    span = upper - lower
    if span <= 0.0:
        raise ValueError("actuator bound span must be positive")
    center = 0.5 * (lower + upper)
    return np.abs(values - center) >= (threshold - 0.5) * span
