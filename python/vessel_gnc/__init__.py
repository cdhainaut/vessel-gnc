"""Vessel-GNC: compact C++/Python simulation and control stack
for autonomous surface vessels."""

from vessel_gnc import _core  # noqa: F401  (compiled module, required)
from vessel_gnc.path import PathGeometry, PathProjection, make_s_curve_geometry
from vessel_gnc.simulation import SimulationResult, simulate

__all__ = [
    "simulate",
    "SimulationResult",
    "PathGeometry",
    "PathProjection",
    "make_s_curve_geometry",
]
__version__ = "0.5.0"
