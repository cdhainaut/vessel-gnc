"""Vessel-GNC: compact C++/Python simulation and control stack
for autonomous surface vessels."""

from vessel_gnc import _core  # noqa: F401  (compiled module, required)
from vessel_gnc.simulation import SimulationResult, simulate

__all__ = ["simulate", "SimulationResult"]
__version__ = "0.4.0"
