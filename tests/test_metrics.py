"""Unit tests for path-following metrics: P95 and explicit saturation duration.

The saturation definition is fixed in metrics.py: on each left-closed
simulation interval ``[t_k, t_{k+1})`` a channel is saturated when its applied
value is within ``1 - saturation_threshold`` of the full physical bound span
from either ``ModelParams`` bound; the final sample bounds no interval and
contributes no duration.
"""

import json

import numpy as np
import pytest
from vessel_gnc import _core
from vessel_gnc.metrics import path_following_metrics
from vessel_gnc.simulation import SimulationResult

PATH = np.array([[0.0, 0.0], [100.0, 0.0]])  # straight North (x = North)
LOOKAHEAD = 8.0  # [m]


def make_result(
    x: np.ndarray,
    y: np.ndarray,
    thrust: np.ndarray,
    yaw_moment: np.ndarray,
    dt: float = 1.0,
) -> SimulationResult:
    """Hand-authored SimulationResult: uniform dt, rest velocities."""
    n = len(x)
    return SimulationResult(
        t=np.arange(n, dtype=float) * dt,
        x=np.asarray(x, dtype=float),
        y=np.asarray(y, dtype=float),
        psi=np.zeros(n),
        u=np.zeros(n),
        v=np.zeros(n),
        r=np.zeros(n),
        thrust=np.asarray(thrust, dtype=float),
        yaw_moment=np.asarray(yaw_moment, dtype=float),
    )


def test_cross_track_p95_and_max():
    # 21 samples on the straight North path with |cross_track| = 0..20 m:
    # for a linear ramp of length 21 the 95th percentile is element 19 (the
    # interpolation index 0.95 * 20 is an integer).
    k = np.arange(21)
    y = -k  # for a North-heading path the signed cross-track error is -y
    result = make_result(np.full(21, 50.0), y, np.zeros(21), np.zeros(21))
    metrics = path_following_metrics(result, PATH, LOOKAHEAD)
    assert metrics["cross_track_p95_m"] == pytest.approx(19.0)
    assert metrics["cross_track_max_m"] == pytest.approx(20.0)
    assert metrics["cross_track_rms_m"] == pytest.approx(np.sqrt(np.mean(k**2)))


def test_thrust_saturation_duration_with_asymmetric_bounds():
    # Asymmetric thrust bounds [-10, 50] N: with threshold 0.99 the span is
    # 60 N and a sample saturates when |thrust - 20| >= 29.4 N (within 0.6 N
    # of a bound). The final saturated sample (t = 3 s) bounds no interval.
    params = _core.ModelParams(
        thrust_min=-10.0, thrust_max=50.0, moment_min=-2.0, moment_max=8.0
    )
    result = make_result(
        np.full(4, 50.0),
        np.zeros(4),
        thrust=np.array([50.0, -10.0, 0.0, 50.0]),
        yaw_moment=np.zeros(4),
    )
    metrics = path_following_metrics(result, PATH, LOOKAHEAD, params=params)
    assert metrics["thrust_saturation_duration_s"] == pytest.approx(2.0)
    assert metrics["moment_saturation_duration_s"] == pytest.approx(0.0)
    assert metrics["any_saturation_duration_s"] == pytest.approx(2.0)


def test_union_saturation_duration_no_double_counting():
    # Thrust saturates on interval 0, moment on intervals 0 and 1: the union
    # counts interval 0 once (2 intervals total, not 3).
    params = _core.ModelParams(
        thrust_min=-10.0, thrust_max=50.0, moment_min=-2.0, moment_max=8.0
    )
    result = make_result(
        np.full(4, 50.0),
        np.zeros(4),
        thrust=np.array([50.0, 0.0, 0.0, 0.0]),
        yaw_moment=np.array([8.0, 8.0, 0.0, 0.0]),
    )
    metrics = path_following_metrics(result, PATH, LOOKAHEAD, params=params)
    assert metrics["thrust_saturation_duration_s"] == pytest.approx(1.0)
    assert metrics["moment_saturation_duration_s"] == pytest.approx(2.0)
    assert metrics["any_saturation_duration_s"] == pytest.approx(2.0)


def test_zero_saturation_duration():
    # All values strictly inside the default bounds: no saturation at all.
    result = make_result(
        np.full(4, 50.0),
        np.zeros(4),
        thrust=np.full(4, 30.0),
        yaw_moment=np.full(4, 0.0),
    )
    metrics = path_following_metrics(result, PATH, LOOKAHEAD)
    assert metrics["thrust_saturation_duration_s"] == pytest.approx(0.0)
    assert metrics["moment_saturation_duration_s"] == pytest.approx(0.0)
    assert metrics["any_saturation_duration_s"] == pytest.approx(0.0)


def test_default_params_used_for_positional_callers():
    # Existing positional callers (result, path, lookahead) keep working and
    # the default bounds apply: 60 N is exactly the default thrust maximum.
    result = make_result(
        np.full(4, 50.0),
        np.zeros(4),
        thrust=np.array([60.0, 0.0, 0.0, 0.0]),
        yaw_moment=np.zeros(4),
    )
    metrics = path_following_metrics(result, PATH, LOOKAHEAD)
    assert metrics["thrust_saturation_duration_s"] == pytest.approx(1.0)
    assert metrics["moment_saturation_duration_s"] == pytest.approx(0.0)


def test_existing_keys_retained():
    result = make_result(np.full(4, 50.0), np.zeros(4), np.zeros(4), np.zeros(4))
    metrics = path_following_metrics(result, PATH, LOOKAHEAD)
    for key in (
        "cross_track_rms_m",
        "cross_track_max_m",
        "heading_error_rms_rad",
        "heading_error_max_rad",
        "thrust_rms_N",
        "thrust_max_N",
        "moment_rms_Nm",
        "moment_max_Nm",
    ):
        assert key in metrics
    assert "cross_track_p95_m" in metrics
    assert "thrust_saturation_duration_s" in metrics
    assert "moment_saturation_duration_s" in metrics
    assert "any_saturation_duration_s" in metrics


def test_metrics_json_serializable():
    params = _core.ModelParams(thrust_min=-10.0, thrust_max=50.0)
    result = make_result(
        np.full(4, 50.0),
        np.zeros(4),
        thrust=np.array([50.0, 0.0, 0.0, 0.0]),
        yaw_moment=np.zeros(4),
    )
    metrics = path_following_metrics(result, PATH, LOOKAHEAD, params=params)
    text = json.dumps(metrics)  # must not raise
    assert json.loads(text) == metrics


def test_saturation_threshold_validation():
    result = make_result(np.full(4, 50.0), np.zeros(4), np.zeros(4), np.zeros(4))
    with pytest.raises(ValueError):
        path_following_metrics(result, PATH, LOOKAHEAD, saturation_threshold=0.4)
    with pytest.raises(ValueError):
        path_following_metrics(result, PATH, LOOKAHEAD, saturation_threshold=1.5)
