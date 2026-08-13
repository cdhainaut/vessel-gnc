"""Python-level tests exercising the C++ core through the pybind11 binding.

Requires the package to be installed with the compiled module, e.g.
``pip install -e .`` in the repository root.
"""

import numpy as np
import pytest
import vessel_gnc
from vessel_gnc import _core


def test_kinematics_through_binding():
    # Heading East with pure surge moves along +y.
    s = _core.State(u=2.0, psi=np.pi / 2)
    d = _core.derivative(s, _core.Control(), _core.Environment(), _core.default_params())
    assert d.x == pytest.approx(0.0, abs=1e-12)
    assert d.y == pytest.approx(2.0, abs=1e-12)


def test_rotation_matrix_orthonormal():
    R = _core.rotation_matrix(0.7)
    assert np.allclose(R @ R.T, np.eye(2), atol=1e-12)
    assert np.isclose(np.linalg.det(R), 1.0, atol=1e-12)


def test_clamp_control_respects_limits():
    p = _core.default_params()
    c = _core.clamp_control(_core.Control(thrust=1e6, yaw_moment=-1e6), p)
    assert c.thrust == p.thrust_max
    assert c.yaw_moment == p.moment_min


def test_simulate_rejects_non_finite_inputs():
    with pytest.raises(ValueError):
        vessel_gnc.simulate(10.0, 0.01, state0=_core.State(u=float("nan")))


def test_simulate_is_deterministic():
    r1 = vessel_gnc.simulate(10.0, 0.01, control=_core.Control(thrust=40.0))
    r2 = vessel_gnc.simulate(10.0, 0.01, control=_core.Control(thrust=40.0))
    assert np.array_equal(r1.x, r2.x)
    assert np.array_equal(r1.u, r2.u)


def test_open_loop_straight_line():
    # No yaw moment, no environment: motion stays exactly on the North axis.
    r = vessel_gnc.simulate(30.0, 0.01, control=_core.Control(thrust=25.0))
    assert np.all(np.isfinite(r.x))
    assert np.max(np.abs(r.y)) < 1e-6
    assert 25.0 < r.x[-1] < 40.0  # cruise ~1.07 m/s after the transient
    assert r.u[-1] == pytest.approx(1.07, abs=0.01)


def test_open_loop_turns_east():
    # Constant yaw moment: the vessel turns clockwise (toward East) and
    # develops a port sideslip (bow into the turn, v < 0).
    r = vessel_gnc.simulate(30.0, 0.01, control=_core.Control(thrust=25.0, yaw_moment=2.0))
    assert r.y[-1] > 15.0
    assert 1.0 < r.psi[-1] < 3.0
    assert r.v[-1] < 0.0


def test_simulate_with_control_policy():
    def policy(t: float, state: _core.State) -> _core.Control:
        return _core.Control(thrust=10.0 + t)

    r = vessel_gnc.simulate(2.0, 0.5, control=policy)
    assert len(r.t) == 5
    assert r.thrust[1] == pytest.approx(10.5)


def test_animation_writes_gif(tmp_path):
    from vessel_gnc.visualization import animate_trajectory

    r = vessel_gnc.simulate(2.0, 0.01, control=_core.Control(thrust=40.0))
    out = tmp_path / "anim.gif"
    animate_trajectory(r, output_path=out, stride=10, fps=5)
    assert out.exists()
    assert out.stat().st_size > 1000
    assert out.read_bytes()[:6] == b"GIF89a"


def test_animation_with_horizon(tmp_path):
    # Hero-mode rendering: reference path + recorded NMPC predictions.
    from vessel_gnc.visualization import animate_trajectory

    r = vessel_gnc.simulate(4.0, 0.01, control=_core.Control(thrust=40.0))
    path = np.array([[0.0, 0.0], [20.0, 0.0], [30.0, 8.0]])
    horizon = [
        (0.5, np.tile([1.0, 0.0, 0.0, 1.2, 0.0, 0.0], (26, 1)).T),
        (2.0, np.tile([2.0, 0.0, 0.0, 1.2, 0.0, 0.0], (26, 1)).T),
    ]
    out = tmp_path / "hero.gif"
    animate_trajectory(r, output_path=out, stride=10, fps=5, reference_path=path, horizon=horizon)
    assert out.exists()
    assert out.stat().st_size > 1000
