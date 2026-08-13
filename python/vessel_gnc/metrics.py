"""Quantitative metrics for path-following controller comparison.

Used by the examples and by the README comparison table (plan §13).
"""

from __future__ import annotations

import numpy as np

from vessel_gnc.guidance import los_heading, project_onto_path
from vessel_gnc.simulation import SimulationResult

__all__ = ["path_following_metrics"]


def path_following_metrics(
    result: SimulationResult, path: np.ndarray, lookahead: float
) -> dict[str, float]:
    """Path-following metrics over the whole run (SI units, angles in rad).

    Returns a JSON-serializable dict with cross-track error, LOS heading
    error and actuator effort statistics.
    """
    points = np.column_stack([result.x, result.y])
    _, _, cross = project_onto_path(points, path)
    psi_los = los_heading(points, path, lookahead)
    heading_error = np.arctan2(np.sin(psi_los - result.psi), np.cos(psi_los - result.psi))

    return {
        "cross_track_rms_m": float(np.sqrt(np.mean(cross**2))),
        "cross_track_max_m": float(np.max(np.abs(cross))),
        "heading_error_rms_rad": float(np.sqrt(np.mean(heading_error**2))),
        "heading_error_max_rad": float(np.max(np.abs(heading_error))),
        "thrust_rms_N": float(np.sqrt(np.mean(result.thrust**2))),
        "thrust_max_N": float(np.max(np.abs(result.thrust))),
        "moment_rms_Nm": float(np.sqrt(np.mean(result.yaw_moment**2))),
        "moment_max_Nm": float(np.max(np.abs(result.yaw_moment))),
    }
