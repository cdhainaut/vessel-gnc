"""Tests for the immutable smooth path geometry (Milestone 5 path ticket).

Covers the required M5-B cases: straight North/East paths, knot/interior
interpolation, C1 tangent continuity, unit tangents and finite headings,
port-positive contour signs, deterministic projection with clamping and
progress hints, duplicate/non-finite rejection, S-curve sampled-geometry
properties (dx/ds > 0, no self-intersection, < 3 m polyline deviation), the
exact smooth-geometry LOS lookahead, and NumPy/CasADi agreement.
"""

import casadi as ca
import numpy as np
import pytest
from vessel_gnc import PathGeometry, make_s_curve_geometry
from vessel_gnc.guidance import los_heading, make_s_curve_path, project_onto_path
from vessel_gnc.path import _validate_regular_curve

NORTH = np.array([[0.0, 0.0], [100.0, 0.0]])  # heading North (x = North)
EAST = np.array([[0.0, 0.0], [0.0, 100.0]])  # heading East (y = East)


def _dense_brute_force_projection(
    geometry: PathGeometry,
    queries: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Independent dense-grid closest-point reference for projection tests."""
    progress = np.linspace(0.0, geometry.length, 5000 * geometry.n_segments + 1)
    path_points = geometry.position(progress)
    reference_progress = np.empty(len(queries))
    reference_distance_squared = np.empty(len(queries))
    for query_index, query in enumerate(queries):
        distance_squared = ((path_points - query) ** 2).sum(axis=1)
        closest_index = np.argmin(distance_squared)
        reference_progress[query_index] = progress[closest_index]
        reference_distance_squared[query_index] = distance_squared[closest_index]
    return reference_progress, reference_distance_squared, progress[1] - progress[0]


def _assert_matches_dense_brute_force(
    geometry: PathGeometry,
    queries: np.ndarray,
) -> None:
    projection = geometry.project(queries)
    reference_progress, reference_distance_squared, grid_spacing = _dense_brute_force_projection(
        geometry, queries
    )
    projected_distance_squared = ((projection.position - queries) ** 2).sum(axis=1)
    assert np.all(projected_distance_squared <= reference_distance_squared + 1e-10)
    np.testing.assert_allclose(
        projection.progress,
        reference_progress,
        atol=2.0 * grid_spacing,
        rtol=0.0,
    )


# --- straight paths ---------------------------------------------------------


def test_straight_north_path_geometry():
    g = PathGeometry(NORTH)
    assert g.length == pytest.approx(100.0)
    assert g.position(50.0) == pytest.approx([50.0, 0.0])
    assert g.unit_tangent(50.0) == pytest.approx([1.0, 0.0])
    assert g.heading(50.0) == pytest.approx(0.0)

    proj = g.project(np.array([[50.0, 5.0], [50.0, -5.0], [120.0, 0.0]]))
    assert proj.progress == pytest.approx([50.0, 50.0, 100.0])  # clamped at the end
    assert proj.cross_track == pytest.approx([-5.0, 5.0, 0.0])
    np.testing.assert_allclose(proj.position, [[50.0, 0.0], [50.0, 0.0], [100.0, 0.0]])


def test_straight_east_path_geometry():
    g = PathGeometry(EAST)
    assert g.heading(50.0) == pytest.approx(np.pi / 2)
    assert g.position(25.0) == pytest.approx([0.0, 25.0])
    # North of the path = port = positive; South = starboard = negative.
    proj = g.project(np.array([[1.0, 50.0], [-1.0, 50.0]]))
    assert proj.cross_track == pytest.approx([1.0, -1.0])


# --- interpolation ----------------------------------------------------------


def test_endpoint_and_knot_interpolation():
    g = make_s_curve_geometry()
    assert g.position(0.0) == pytest.approx([0.0, 0.0])
    assert g.position(g.length) == pytest.approx([165.0, 35.0])
    np.testing.assert_allclose(g.position(g.knots), g.waypoints, atol=1e-12)
    # interior points of the last segment
    interior = np.linspace(g.knots[-2], g.knots[-1], 5)
    assert np.all(np.isfinite(g.position(interior)))


def test_c1_tangent_continuity_at_interior_knots():
    g = make_s_curve_geometry()
    for s_knot in g.knots[1:-1]:
        left = g.derivative(s_knot - 1e-8)
        right = g.derivative(s_knot + 1e-8)
        assert np.allclose(left, right, rtol=1e-5, atol=1e-7)
        t_left = g.unit_tangent(s_knot - 1e-8)
        t_right = g.unit_tangent(s_knot + 1e-8)
        assert np.dot(t_left, t_right) == pytest.approx(1.0, abs=1e-7)


def test_unit_tangent_nonzero_and_heading_finite():
    g = make_s_curve_geometry()
    s = np.linspace(0.0, g.length, 501)
    tangents = g.unit_tangent(s)
    assert np.allclose(np.linalg.norm(tangents, axis=1), 1.0, atol=1e-12)
    headings = g.heading(s)
    assert np.all(np.isfinite(headings))
    assert np.all((headings >= -np.pi) & (headings <= np.pi))


# --- contour sign -----------------------------------------------------------


def test_contour_sign_port_positive():
    g = make_s_curve_geometry()
    for s0 in (0.0, 40.0, 95.0, 140.0, g.length):
        t = g.unit_tangent(s0)
        p = g.position(s0)
        port = p + np.array([t[1], -t[0]])  # 1 m to the port/left
        starboard = p - np.array([t[1], -t[0]])
        proj = g.project(np.array([port, starboard]))
        assert proj.cross_track[0] == pytest.approx(1.0, abs=1e-9)
        assert proj.cross_track[1] == pytest.approx(-1.0, abs=1e-9)
    # identical to the legacy polyline sign on the straight first segment
    s_legacy = g.project(np.array([[10.0, 2.0], [10.0, -2.0]]))
    legacy = project_onto_path(np.array([[10.0, 2.0], [10.0, -2.0]]), make_s_curve_path())
    assert s_legacy.cross_track == pytest.approx(legacy[2])


# --- projection -------------------------------------------------------------


def test_projection_deterministic_and_clamped():
    g = make_s_curve_geometry()
    pts = np.array(
        [
            [50.0, 10.0],
            [80.0, 45.0],
            [130.0, 30.0],
            [160.0, 25.0],
            [55.0, 15.0],
            [30.0, 0.0],
        ]
    )
    p1 = g.project(pts)
    p2 = g.project(pts)
    assert np.array_equal(p1.progress, p2.progress)
    assert np.array_equal(p1.cross_track, p2.cross_track)
    np.testing.assert_array_equal(p1.position, p2.position)

    proj = g.project(np.array([[-50.0, -50.0], [300.0, 100.0]]))
    assert proj.progress == pytest.approx([0.0, g.length])
    assert proj.cross_track[0] > 0.0  # far west of the North heading = port


def test_projection_accuracy_against_dense_grid():
    g = make_s_curve_geometry()
    s_dense = np.linspace(0.0, g.length, 400 * g.n_segments)
    dense_pos = g.position(s_dense)
    queries = np.array(
        [
            [55.0, 15.0],
            [70.0, 40.0],
            [110.0, 42.0],
            [125.0, 22.0],
            [145.0, 18.0],
            [50.0, 10.0],
            [80.0, 45.0],
            [100.0, 30.0],
            [120.0, 35.0],
            [150.0, 20.0],
            [-10.0, -5.0],
            [200.0, 50.0],
        ]
    )
    dense_min = ((dense_pos[None, :, :] - queries[:, None, :]) ** 2).sum(axis=2).min(axis=1)
    proj = g.project(queries)
    d_proj = ((proj.position - queries) ** 2).sum(axis=1)
    assert np.all(d_proj <= dense_min + 1e-9)
    # orthogonality of the residual at interior projections
    t = g.unit_tangent(proj.progress)
    residual = ((queries - proj.position) * t).sum(axis=1)
    interior = (proj.progress > 1e-6) & (proj.progress < g.length - 1e-6)
    assert np.all(np.abs(residual[interior]) < 1e-6)


def test_projection_global_minimum_for_deterministic_wide_box_queries():
    g = make_s_curve_geometry()
    rng = np.random.default_rng(20250305)
    queries = rng.uniform([-80.0, -80.0], [240.0, 120.0], size=(48, 2))
    # Regression for the former knot-pinning failure at s = 153.442... m.
    queries = np.vstack([queries, [102.63982229020311, -26.80168678873615]])
    _assert_matches_dense_brute_force(g, queries)


def test_projection_global_minimum_for_large_normal_offsets():
    g = make_s_curve_geometry()
    source_progress = np.linspace(5.0, g.length - 5.0, 12)
    tangents = g.unit_tangent(source_progress)
    port_normals = np.column_stack([tangents[:, 1], -tangents[:, 0]])
    path_points = g.position(source_progress)
    queries = np.vstack(
        [path_points + offset * port_normals for offset in (-40.0, -20.0, 20.0, 40.0)]
    )
    # Regressions around 20 m offset where the former multimodal refinement
    # pinned to a knot or selected a nearby local minimum.
    queries = np.vstack(
        [
            queries,
            [54.79032238483822, -12.542555984880696],
            [138.52091515877294, 38.150839378479205],
            [99.83486760398651, 19.120583363595312],
            [100.06605688061322, 19.166292908814164],
        ]
    )
    _assert_matches_dense_brute_force(g, queries)


def test_projection_of_on_path_points_is_fixed_point():
    g = make_s_curve_geometry()
    for s0 in np.linspace(0.0, g.length, 21):
        q = g.position(s0)
        s_proj = g.project(np.array([q])).progress[0]
        assert s_proj == pytest.approx(s0, abs=1e-6)


def test_projection_hint_never_worsens():
    g = make_s_curve_geometry()
    rng = np.random.default_rng(11)
    for _ in range(40):
        q = rng.uniform(-30.0, 200.0, size=2)
        hint = float(rng.uniform(0.0, g.length))
        d_none = np.linalg.norm(q - g.project(np.array([q])).position[0])
        d_hint = np.linalg.norm(q - g.project(np.array([q]), progress_hint_m=hint).position[0])
        assert d_hint <= d_none + 1e-12
    # a hint at the true projection does not move the result
    q = g.position(85.0) + np.array([2.0, -1.0])
    proj0 = g.project(np.array([q]))
    s_true = proj0.progress[0]
    proj_h = g.project(np.array([q]), progress_hint_m=s_true)
    assert proj_h.progress[0] == pytest.approx(s_true, abs=1e-9)
    # vector hints broadcast
    pts = np.array([[50.0, 10.0], [130.0, 30.0]])
    proj_v = g.project(pts, progress_hint_m=np.array([40.0, 120.0]))
    assert proj_v.progress.shape == (2,)


# --- input rejection --------------------------------------------------------


def test_rejects_consecutive_and_nonconsecutive_repeated_waypoints():
    with pytest.raises(ValueError, match="pairwise distinct"):
        PathGeometry([[0.0, 0.0], [0.0, 0.0], [10.0, 0.0]])
    with pytest.raises(ValueError, match="pairwise distinct"):
        PathGeometry([[0.0, 0.0], [10.0, 0.0], [0.0, 0.0]])


def test_rejects_zero_derivative_at_right_angle_knot_and_segment_interior():
    with pytest.raises(ValueError, match="derivative must never vanish"):
        PathGeometry([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])

    # Synthetic Hermite segment p(t) = (3t - 6t^2 + 4t^3, 0) has
    # p'(t) = (12(t - 1/2)^2, 0), an even-multiplicity interior zero that a
    # sign-change test or finite sampling can miss.
    with pytest.raises(ValueError, match="local progress 0.5 m"):
        _validate_regular_curve(
            coeff_north=np.array([[0.0, 3.0, -6.0, 4.0]]),
            coeff_east=np.zeros((1, 4)),
            chords=np.ones(1),
        )


def test_rejects_nonfinite_or_malformed_waypoints():
    with pytest.raises(ValueError):
        PathGeometry([[0.0, 0.0], [np.nan, 1.0]])
    with pytest.raises(ValueError):
        PathGeometry([[0.0, 0.0], [np.inf, 1.0]])
    with pytest.raises(ValueError):
        PathGeometry([[0.0, 0.0]])  # fewer than two waypoints
    with pytest.raises(ValueError):
        PathGeometry(np.zeros((3, 3)))  # wrong column count
    with pytest.raises(ValueError):
        PathGeometry(np.zeros(4))  # not 2-D
    g = make_s_curve_geometry()
    with pytest.raises(ValueError):
        g.project([[50.0, np.nan]])
    with pytest.raises(ValueError, match="non-empty"):
        g.project(np.empty((0, 2)))
    with pytest.raises(ValueError, match="non-empty"):
        g.project([])


# --- S-curve sampled geometry -----------------------------------------------


def test_s_curve_dx_ds_positive_and_no_self_intersection():
    g = make_s_curve_geometry()
    s = np.linspace(0.0, g.length, 4001)
    pos = g.position(s)
    d = g.derivative(s)
    assert np.all(d[:, 0] > 0.0)  # North strictly increases: no self-intersection
    assert np.all(np.diff(pos[:, 0]) > 0.0)

    # direct check: sampled segments do not cross non-adjacent segments
    sampled = pos[::20]  # 201 points, 200 segments
    n_seg = len(sampled) - 1

    def _orient(p, q, r):
        return (q[0] - p[0]) * (r[1] - p[1]) - (q[1] - p[1]) * (r[0] - p[0])

    for i in range(n_seg):
        a, b = sampled[i], sampled[i + 1]
        for j in range(i + 2, n_seg):
            if abs(i - j) <= 1:
                continue
            c, d = sampled[j], sampled[j + 1]
            o1 = _orient(a, b, c) * _orient(a, b, d)
            o2 = _orient(c, d, a) * _orient(c, d, b)
            assert not (o1 < 0.0 and o2 < 0.0), (i, j)


def test_s_curve_no_material_overshoot():
    g = make_s_curve_geometry()
    wps = g.waypoints
    s = np.linspace(0.0, g.length, 4001)
    pos = g.position(s)
    for i in range(g.n_segments):
        sel = (s >= g.knots[i] - 1e-12) & (s <= g.knots[i + 1] + 1e-12)
        seg_pts = pos[sel]
        # monotone per coordinate: stay inside the segment's coordinate box
        assert seg_pts[:, 0].min() >= min(wps[i, 0], wps[i + 1, 0]) - 1e-9
        assert seg_pts[:, 0].max() <= max(wps[i, 0], wps[i + 1, 0]) + 1e-9
        assert seg_pts[:, 1].min() >= min(wps[i, 1], wps[i + 1, 1]) - 1e-9
        assert seg_pts[:, 1].max() <= max(wps[i, 1], wps[i + 1, 1]) + 1e-9


def test_s_curve_polyline_deviation_below_3m():
    g = make_s_curve_geometry()
    polyline = make_s_curve_path()
    s = np.linspace(0.0, g.length, 4001)
    pos = g.position(s)
    seg, along, _ = project_onto_path(pos, polyline)
    seg_dir = np.diff(polyline, axis=0)
    unit = seg_dir / np.linalg.norm(seg_dir, axis=1)[:, None]
    proj = polyline[seg] + along[:, None] * unit[seg]
    deviation = np.linalg.norm(pos - proj, axis=1)
    assert deviation.max() < 3.0


def test_s_curve_geometry_passes_through_original_waypoints():
    g = make_s_curve_geometry()
    wp = make_s_curve_path()
    # one collinear midpoint per chord: 17 knots, original waypoints at even knots
    assert g.n_segments == 2 * (len(wp) - 1)
    assert g.waypoints.shape[0] == 2 * len(wp) - 1
    np.testing.assert_allclose(g.position(g.knots[::2]), wp, atol=1e-12)


# --- NumPy / CasADi agreement -----------------------------------------------


def test_numpy_casadi_agreement_at_boundaries_and_interior():
    g = make_s_curve_geometry()
    probe = np.unique(np.concatenate([np.linspace(0.0, g.length, 101), g.knots]))
    p_np = g.position(probe)
    p_ca = np.array(g.casadi_position(ca.DM(probe))).reshape(2, -1).T
    np.testing.assert_allclose(p_ca, p_np, atol=1e-12, rtol=0)

    d_np = g.derivative(probe)
    d_ca = np.array(g.casadi_derivative(ca.DM(probe))).reshape(2, -1).T
    np.testing.assert_allclose(d_ca, d_np, atol=1e-12, rtol=0)

    t_np = g.unit_tangent(probe)
    t_ca = np.array(g.casadi_unit_tangent(ca.DM(probe))).reshape(2, -1).T
    np.testing.assert_allclose(t_ca, t_np, atol=1e-12, rtol=0)

    h_np = g.heading(probe)
    h_ca = np.array(g.casadi_heading(ca.DM(probe))).reshape(-1)
    np.testing.assert_allclose(h_ca, h_np, atol=1e-12, rtol=0)

    # scalar evaluation, including outside-domain clamping
    for s0 in (0.0, g.length, probe[37], -10.0, g.length + 50.0):
        np.testing.assert_allclose(
            np.array(g.casadi_position(ca.DM(s0))).ravel(), g.position(s0), atol=1e-12
        )
    # vector evaluation via a (n, 1) symbolic column: build the mapped
    # function from the symbolic expression, then evaluate it numerically
    S = ca.MX.sym("S", probe.size)
    p_sym = g.casadi_position(S)
    assert p_sym.shape == (2, probe.size)
    p_func = ca.Function("p", [S], [p_sym])
    p_val = np.array(p_func(ca.DM(probe))).reshape(2, -1).T
    np.testing.assert_allclose(p_val, p_np, atol=1e-12, rtol=0)


# --- exact smooth-geometry LOS lookahead ------------------------------------


def test_los_lookahead_exact_on_geometry():
    g = PathGeometry(EAST)
    # on the path: desired heading = East = +90 deg
    assert g.los_heading(np.array([[0.0, 0.0]]), 10.0)[0] == pytest.approx(np.pi / 2)
    # 10 m south of the start: projection clamps to s = 0, LOS point 10 m ahead
    assert g.los_heading(np.array([[0.0, -10.0]]), 10.0)[0] == pytest.approx(np.arctan2(20.0, 0.0))
    # beyond the path end: lookahead clamps to the final point
    assert g.los_heading(np.array([[-10.0, 95.0]]), 10.0)[0] == pytest.approx(np.arctan2(5.0, 10.0))

    pos_los, s_los = g.lookahead_point(np.array([0.0, -10.0]), 10.0)
    assert s_los == pytest.approx(10.0)
    assert pos_los == pytest.approx([0.0, 10.0])

    with pytest.raises(ValueError):
        g.los_heading(np.array([[0.0, 0.0]]), 0.0)
    with pytest.raises(ValueError):
        g.lookahead_point(np.array([0.0, 0.0]), 0.0)


def test_los_matches_polyline_los_on_straight_paths():
    # LOS can consume the same path: exact smooth lookahead == legacy polyline
    # LOS on straight paths (they coincide there).
    for path in (NORTH, EAST):
        geom = PathGeometry(path)
        points = np.array([[20.0, 3.0], [50.0, -4.0], [120.0, 10.0]])
        assert np.allclose(geom.los_heading(points, 8.0), los_heading(points, path, 8.0))


# --- API, immutability, sampling --------------------------------------------


def test_geometry_is_immutable():
    g = make_s_curve_geometry()
    expected = g.position(50.0)
    with pytest.raises(AttributeError):
        g.length = 0.0
    with pytest.raises(AttributeError):
        g.some_new_attr = 1
    w = g.waypoints
    w[:] = 0.0
    k = g.knots
    k[:] = 0.0
    assert g.position(50.0) == pytest.approx(expected)


def test_sample_and_exports():
    g = make_s_curve_geometry()
    pos, headings = g.sample(101)
    assert pos.shape == (101, 2)
    assert headings.shape == (101,)
    assert pos[0] == pytest.approx(g.position(0.0))
    assert pos[-1] == pytest.approx(g.position(g.length))
    with pytest.raises(ValueError):
        g.sample(1)
    # public package-level exports
    assert isinstance(g, PathGeometry)
    assert PathGeometry(NORTH).length == pytest.approx(100.0)
