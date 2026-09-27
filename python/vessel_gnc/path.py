"""Immutable differentiable path geometry (Milestone 5, path ticket).

A ``PathGeometry`` owns a fixed piecewise-cubic representation of a smooth
path through waypoints and provides geometric queries shared by LOS, the
legacy NMPC and the future geometric MPCC:

- progress coordinate ``s`` [m]: cumulative chord length, monotone and C1
  (an honest chord-length parameter, not exact analytic arc length);
- position, first derivative, unit tangent and tangent heading;
- deterministic closest-point projection with signed cross-track error;
- a CasADi evaluator using the exact same stored coefficients as NumPy.

Interpolation: PCHIP-style Fritsch-Carlson monotone cubic Hermite per
North/East coordinate against cumulative chord length. Each coordinate is
monotone on every segment (no overshoot beyond the segment's coordinate
bounds) and the whole curve is C1 at the knots.

Conventions (docs/model.md §1): waypoints are (M, 2) arrays with North in
column 0 and East in column 1 [m]; tangent heading is ``atan2(t_east,
t_north)``, clockwise-positive from North; the signed cross-track error is
positive when the point is to the port/left of the path direction, matching
``guidance.project_onto_path``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import casadi as ca
import numpy as np

from vessel_gnc.guidance import make_s_curve_path

__all__ = ["PathGeometry", "PathProjection", "make_s_curve_geometry"]

_ROOT_BISECTION_ITERATIONS = 64
_ROOT_VALUE_TOLERANCE = 128.0 * np.finfo(float).eps
_MIN_CHORD_LENGTH = 1e-12  # [m] below this two waypoints count as duplicates
_MIN_TANGENT_NORM = 1e-12  # below this a tangent is treated as degenerate


def _validate_waypoints(waypoints: np.ndarray) -> np.ndarray:
    """Validate and copy waypoints: (M, 2) [North, East], M >= 2, finite."""
    wps = np.asarray(waypoints, dtype=float)
    if wps.ndim != 2 or wps.shape[1] != 2 or wps.shape[0] < 2:
        raise ValueError("waypoints must be a (M, 2) array with M >= 2 (columns: North, East) [m]")
    if not np.all(np.isfinite(wps)):
        raise ValueError("waypoints must be finite")
    for waypoint_index, waypoint in enumerate(wps[:-1]):
        pairwise_distances = np.linalg.norm(wps[waypoint_index + 1 :] - waypoint, axis=1)
        if np.any(pairwise_distances <= _MIN_CHORD_LENGTH):
            raise ValueError(
                "waypoints must be pairwise distinct "
                f"(separation must exceed {_MIN_CHORD_LENGTH} m)"
            )
    return np.array(wps, copy=True)


def _endpoint_slope(ha: float, hb: float, da: float, db: float) -> float:
    """One-sided three-point endpoint slope, clamped like PCHIP."""
    slope = ((2.0 * ha + hb) * da - ha * db) / (ha + hb)
    if np.sign(slope) != np.sign(da):
        return 0.0
    if np.sign(da) != np.sign(db) and abs(slope) > 3.0 * abs(da):
        return 3.0 * da
    return slope


def _fritsch_carlson_slopes(values: np.ndarray, chords: np.ndarray) -> np.ndarray:
    """PCHIP Fritsch-Carlson endpoint slopes dy/ds at each knot.

    Interior knots use the weighted harmonic mean of the neighbouring
    secants; a secant sign change forces a zero slope (monotone per
    coordinate). Endpoints use the one-sided three-point formula clamped to
    the neighbouring secant, as in PCHIP.
    """
    m = len(chords) + 1
    secants = np.diff(values) / chords
    slopes = np.zeros(m)
    if m == 2:
        slopes[:] = secants[0]
        return slopes
    w1 = 2.0 * chords[:-1] + chords[1:]
    w2 = chords[:-1] + 2.0 * chords[1:]
    same_sign = secants[:-1] * secants[1:] > 0.0
    with np.errstate(divide="ignore", invalid="ignore"):
        harmonic = (w1 + w2) / (w1 / secants[:-1] + w2 / secants[1:])
    slopes[1:-1] = np.where(same_sign, harmonic, 0.0)
    slopes[0] = _endpoint_slope(chords[0], chords[1], secants[0], secants[1])
    slopes[-1] = _endpoint_slope(chords[-1], chords[-2], secants[-1], secants[-2])
    return slopes


def _hermite_coefficients(values: np.ndarray, chords: np.ndarray, slopes: np.ndarray) -> np.ndarray:
    """Per-segment cubic coefficients ``p(u) = a + b u + c u^2 + d u^3``.

    With ``u = s - s_i`` the local progress inside segment ``i``; arrays are
    shaped (M-1, 4) with columns a, b, c, d.
    """
    y0 = values[:-1]
    y1 = values[1:]
    m0 = slopes[:-1]
    m1 = slopes[1:]
    a = y0
    b = m0
    c = (3.0 * (y1 - y0) / chords - 2.0 * m0 - m1) / chords
    d = (m0 + m1 - 2.0 * (y1 - y0) / chords) / chords**2
    return np.stack([a, b, c, d], axis=1)


def _refine_polyline(waypoints: np.ndarray, n_sub: int) -> np.ndarray:
    """Insert ``n_sub - 1`` collinear points on each chord (n_sub >= 1)."""
    if n_sub < 1:
        raise ValueError("n_sub must be >= 1")
    wps = np.asarray(waypoints, dtype=float)
    refined = [wps[0]]
    for i in range(len(wps) - 1):
        for k in range(1, n_sub):
            refined.append(wps[i] + (k / n_sub) * (wps[i + 1] - wps[i]))
        refined.append(wps[i + 1])
    return np.asarray(refined, dtype=float)


def _horner(value: float, coefficients: list[float]) -> float:
    """Evaluate ascending-order coefficients at one point with Horner's rule."""
    result = coefficients[-1]
    for coefficient in coefficients[-2::-1]:
        result = result * value + coefficient
    return result


def _real_roots_in_unit_interval(coefficients: np.ndarray) -> np.ndarray:
    """Find all real roots of a low-degree polynomial on ``[0, 1]``.

    Coefficients are in ascending power order. Recursive derivative-root
    isolation partitions the interval into monotone pieces, so sign-changing
    roots are bracketed and repeated roots are retained at derivative roots.
    The scalar arithmetic is plain float arithmetic (Horner evaluation): the
    solver is on the projection hot path and NumPy call overhead dominated the
    root search, while the algorithm and its tolerances are unchanged.
    """
    coeff = [float(value) for value in coefficients]
    while coeff and coeff[-1] == 0.0:
        coeff.pop()
    if not coeff:
        return np.empty(0)
    scale = max(abs(value) for value in coeff)
    coeff = [value / scale for value in coeff]
    degree = len(coeff) - 1
    if degree == 0:
        return np.empty(0)
    if degree == 1:
        root = -coeff[0] / coeff[1]
        if -_ROOT_VALUE_TOLERANCE <= root <= 1.0 + _ROOT_VALUE_TOLERANCE:
            return np.array([min(max(root, 0.0), 1.0)])
        return np.empty(0)

    derivative_coeff = [index * value for index, value in enumerate(coeff)][1:]
    critical_points = _real_roots_in_unit_interval(derivative_coeff)
    boundaries = sorted({0.0, 1.0, *(float(value) for value in critical_points)})
    values = [_horner(boundary, coeff) for boundary in boundaries]
    value_tolerance = _ROOT_VALUE_TOLERANCE * math.fsum(abs(value) for value in coeff)
    roots = [
        boundary
        for boundary, value in zip(boundaries, values, strict=True)
        if abs(value) <= value_tolerance
    ]

    for left, right, value_left, value_right in zip(
        boundaries[:-1], boundaries[1:], values[:-1], values[1:], strict=True
    ):
        if value_left == 0.0 or value_right == 0.0:
            continue
        if (value_left < 0.0) == (value_right < 0.0):
            continue
        for _ in range(_ROOT_BISECTION_ITERATIONS):
            midpoint = 0.5 * (left + right)
            if midpoint == left or midpoint == right:
                break
            value_midpoint = _horner(midpoint, coeff)
            if value_midpoint == 0.0:
                left = right = midpoint
                break
            if (value_midpoint < 0.0) == (value_left < 0.0):
                left, value_left = midpoint, value_midpoint
            else:
                right, value_right = midpoint, value_midpoint
        roots.append(0.5 * (left + right))

    return np.asarray(sorted({min(max(root, 0.0), 1.0) for root in roots}), dtype=float)


def _validate_regular_curve(
    coeff_north: np.ndarray,
    coeff_east: np.ndarray,
    chords: np.ndarray,
) -> None:
    """Reject a cubic whose derivative vector vanishes anywhere on a segment.

    Each coordinate derivative is a quadratic in normalized local progress
    ``t in [0, 1]``. A zero derivative vector must occur at an endpoint or at
    a real root of at least one coordinate derivative, so evaluating that
    finite coefficient-derived candidate set is exhaustive and deterministic.
    """
    for segment, chord in enumerate(chords):
        derivative_north = np.array(
            [
                coeff_north[segment, 1],
                2.0 * chord * coeff_north[segment, 2],
                3.0 * chord**2 * coeff_north[segment, 3],
            ]
        )
        derivative_east = np.array(
            [
                coeff_east[segment, 1],
                2.0 * chord * coeff_east[segment, 2],
                3.0 * chord**2 * coeff_east[segment, 3],
            ]
        )
        candidates = np.unique(
            np.concatenate(
                [
                    [0.0, 1.0],
                    _real_roots_in_unit_interval(derivative_north),
                    _real_roots_in_unit_interval(derivative_east),
                ]
            )
        )
        north_values = np.polynomial.polynomial.polyval(
            candidates,
            derivative_north,
        )
        east_values = np.polynomial.polynomial.polyval(
            candidates,
            derivative_east,
        )
        tangent_norm = np.hypot(north_values, east_values)
        if np.any(tangent_norm < _MIN_TANGENT_NORM):
            local_progress = float(candidates[np.argmin(tangent_norm)] * chord)
            raise ValueError(
                "path derivative must never vanish; degenerate tangent on "
                f"segment {segment} at local progress {local_progress:.16g} m"
            )


@dataclass(frozen=True)
class PathProjection:
    """Deterministic closest-point projection of points onto the path.

    Attributes:
        progress: (n,) chord-length progress [m] of the closest path point,
            clamped to ``[0, length]``.
        cross_track: (n,) signed distance [m], positive to port/left of the
            path direction (matches ``guidance.project_onto_path``).
        position: (n, 2) closest path position [m] (North, East).
    """

    progress: np.ndarray
    cross_track: np.ndarray
    position: np.ndarray


class PathGeometry:
    """Immutable, differentiable smooth path through waypoints.

    The path is fixed at construction: waypoint positions, knot progress and
    the per-segment cubic coefficients are stored read-only, and the CasADi
    evaluator is built once from those same coefficients. Do not mutate the
    arrays returned by the properties; they are copies.

    Args:
        waypoints: (M, 2) array of (North, East) positions [m]; M >= 2,
            finite and pairwise separated by more than 1e-12 m.
    """

    __slots__ = (
        "_waypoints",
        "_knots",
        "_chord",
        "_n_segments",
        "_coeff_n",
        "_coeff_e",
        "_segment_bounds",
        "_f_position",
        "_f_derivative",
        "_f_unit_tangent",
        "_f_heading",
    )

    def __init__(self, waypoints: np.ndarray):
        wps = _validate_waypoints(waypoints)
        chords = np.linalg.norm(np.diff(wps, axis=0), axis=1)
        knots = np.concatenate([[0.0], np.cumsum(chords)])
        slopes_n = _fritsch_carlson_slopes(wps[:, 0], chords)
        slopes_e = _fritsch_carlson_slopes(wps[:, 1], chords)
        coeff_n = _hermite_coefficients(wps[:, 0], chords, slopes_n)
        coeff_e = _hermite_coefficients(wps[:, 1], chords, slopes_e)
        _validate_regular_curve(coeff_n, coeff_e, chords)
        for arr in (wps, knots, chords, coeff_n, coeff_e):
            arr.flags.writeable = False
        self._waypoints = wps
        self._knots = knots
        self._chord = chords
        self._n_segments = len(chords)
        self._coeff_n = coeff_n
        self._coeff_e = coeff_e
        self._segment_bounds = self._compute_segment_bounds()
        self._build_casadi()

    def _compute_segment_bounds(self) -> np.ndarray:
        """Exact axis-aligned bounds of every segment (projection pruning).

        A cubic attains its extrema at the interval endpoints and at the real
        roots of its derivative, so the bounds are exact. The projection uses
        them to skip segments whose bounding box is strictly farther from a
        query point than the best candidate so far.
        """
        bounds = np.empty((self._n_segments, 2, 2))
        for segment in range(self._n_segments):
            powers = self._chord[segment] ** np.arange(4)
            for axis, coefficients in enumerate((self._coeff_n, self._coeff_e)):
                scaled = coefficients[segment] * powers
                derivative = np.arange(1, 4) * scaled[1:]
                critical = _real_roots_in_unit_interval(derivative)
                samples = np.concatenate(([0.0, 1.0], critical))
                values = scaled[0] + samples * (
                    scaled[1] + samples * (scaled[2] + samples * scaled[3])
                )
                bounds[segment, axis] = (values.min(), values.max())
        bounds.flags.writeable = False
        return bounds

    # --- introspection -----------------------------------------------------

    @property
    def length(self) -> float:
        """Total progress [m] (cumulative chord length from the first waypoint)."""
        return float(self._knots[-1])

    @property
    def n_segments(self) -> int:
        """Number of polynomial segments (one per chord)."""
        return self._n_segments

    @property
    def waypoints(self) -> np.ndarray:
        """(M, 2) knot positions [m] (North, East), as a copy."""
        return self._waypoints.copy()

    @property
    def knots(self) -> np.ndarray:
        """(M,) cumulative chord length [m] at each knot, as a copy."""
        return self._knots.copy()

    @property
    def chord_lengths(self) -> np.ndarray:
        """(M-1,) chord length [m] of each segment, as a copy."""
        return self._chord.copy()

    # --- NumPy evaluation --------------------------------------------------

    def _evaluate(self, coeff: np.ndarray, s: np.ndarray, deriv_order: int) -> np.ndarray:
        """Evaluate one coordinate's piecewise cubic; ``s`` (n,) is clamped."""
        seg = np.clip(
            np.searchsorted(self._knots, s, side="right") - 1,
            0,
            self._n_segments - 1,
        )
        u = s - self._knots[seg]
        a, b, c, d = coeff[seg, 0], coeff[seg, 1], coeff[seg, 2], coeff[seg, 3]
        if deriv_order == 0:
            return a + u * (b + u * (c + u * d))
        if deriv_order == 1:
            return b + u * (2.0 * c + 3.0 * d * u)
        if deriv_order == 2:
            return 2.0 * c + 6.0 * d * u
        raise ValueError(f"deriv_order must be 0, 1 or 2, got {deriv_order}")

    def position(self, s: float | np.ndarray) -> np.ndarray:
        """Path position [m] at progress value(s) ``s`` (North, East).

        Progress is cumulative chord length from the first waypoint; values
        outside ``[0, length]`` are clamped to the domain.

        Args:
            s: scalar or (n,) progress values [m].

        Returns:
            (2,) position for a scalar input, (n, 2) for an array input.
        """
        s_arr = np.asarray(s, dtype=float)
        s_flat = np.atleast_1d(s_arr).ravel()
        s_clamped = np.clip(s_flat, 0.0, self.length)
        north = self._evaluate(self._coeff_n, s_clamped, 0)
        east = self._evaluate(self._coeff_e, s_clamped, 0)
        out = np.column_stack([north, east])
        if s_arr.ndim == 0:
            return out[0]
        return out.reshape(s_arr.shape + (2,))

    def derivative(self, s: float | np.ndarray) -> np.ndarray:
        """First derivative d/ds of the path position (dimensionless).

        Shape conventions match :meth:`position`.
        """
        s_arr = np.asarray(s, dtype=float)
        s_flat = np.atleast_1d(s_arr).ravel()
        s_clamped = np.clip(s_flat, 0.0, self.length)
        north = self._evaluate(self._coeff_n, s_clamped, 1)
        east = self._evaluate(self._coeff_e, s_clamped, 1)
        out = np.column_stack([north, east])
        if s_arr.ndim == 0:
            return out[0]
        return out.reshape(s_arr.shape + (2,))

    def unit_tangent(self, s: float | np.ndarray) -> np.ndarray:
        """Unit tangent of the regular path.

        Construction rejects every coefficient-derived zero of the derivative
        vector. NumPy therefore uses the same direct derivative normalization
        as the CasADi evaluator, with no backend-specific fallback direction.

        Shape conventions match :meth:`position`.
        """
        derivative = self.derivative(s)
        scalar_input = derivative.ndim == 1
        derivative_array = np.atleast_2d(derivative)
        norm = np.linalg.norm(derivative_array, axis=1)
        if np.any(~np.isfinite(norm)):
            raise ValueError("progress values must be finite")
        if np.any(norm < _MIN_TANGENT_NORM):
            raise RuntimeError("regular-path invariant violated: zero derivative")
        tangent = derivative_array / norm[:, None]
        return tangent[0] if scalar_input else tangent

    def heading(self, s: float | np.ndarray) -> np.ndarray:
        """Tangent heading [rad], clockwise-positive from North.

        ``atan2(t_east, t_north)`` of the unit tangent; shape conventions
        match :meth:`position`.
        """
        tangent = self.unit_tangent(s)
        return np.arctan2(tangent[..., 1], tangent[..., 0])

    # --- projection --------------------------------------------------------

    def _stationary_progress(self, point: np.ndarray, segment: int) -> np.ndarray:
        """All stationary squared-distance progress values on one segment."""
        chord = self._chord[segment]
        powers = np.array([1.0, chord, chord**2, chord**3])
        delta_north = self._coeff_n[segment] * powers
        delta_east = self._coeff_e[segment] * powers
        delta_north[0] -= point[0]
        delta_east[0] -= point[1]
        stationary_coeff = np.polynomial.polynomial.polyadd(
            np.polynomial.polynomial.polymul(
                delta_north,
                np.polynomial.polynomial.polyder(delta_north),
            ),
            np.polynomial.polynomial.polymul(
                delta_east,
                np.polynomial.polynomial.polyder(delta_east),
            ),
        )
        local_roots = _real_roots_in_unit_interval(stationary_coeff)
        return self._knots[segment] + chord * local_roots

    def project(
        self,
        points: np.ndarray,
        progress_hint_m: float | np.ndarray | None = None,
    ) -> PathProjection:
        """Deterministic global closest-point projection onto the path.

        For every cubic segment, the method forms the degree-at-most-five
        polynomial ``(p(t) - q) . p'(t)`` in normalized local progress. It
        evaluates every segment endpoint and every real stationary root in
        ``[0, 1]``. These candidates contain the global minimum of squared
        distance on the compact piecewise-cubic path, including multimodal
        distance functions and flat stationary valleys.

        ``progress_hint_m`` is retained for the MPCC-facing API and validated,
        but cannot change the global result or its deterministic earliest-
        progress tie break.

        Args:
            points: (n, 2) or (2,) positions [m] (North, East); must be
                finite.
            progress_hint_m: optional progress [m] near which the projection
                is expected, scalar or (n,) array.

        Returns:
            :class:`PathProjection` with progress [m], signed cross-track
            error [m] (positive = port/left) and closest path positions [m].
        """
        points_array = np.asarray(points, dtype=float)
        if points_array.ndim == 1 and points_array.shape == (2,):
            pts = points_array[None, :]
        elif points_array.ndim == 2 and points_array.shape[1] == 2 and points_array.shape[0] > 0:
            pts = points_array
        else:
            raise ValueError("points must be a non-empty (n, 2) array or one (2,) point")
        if not np.all(np.isfinite(pts)):
            raise ValueError("points must be finite")
        n_points = pts.shape[0]

        if progress_hint_m is not None:
            try:
                hints = np.broadcast_to(
                    np.asarray(progress_hint_m, dtype=float),
                    (n_points,),
                )
            except ValueError as error:
                raise ValueError(
                    "progress_hint_m must be a finite scalar or an (n,) array"
                ) from error
            if not np.all(np.isfinite(hints)):
                raise ValueError("progress_hint_m must be a finite scalar or an (n,) array")

        progress = np.empty(n_points)
        knot_positions = self.position(self._knots)
        for point_index, point in enumerate(pts):
            best_squared = float(np.min(((knot_positions - point) ** 2).sum(axis=1)))
            candidate_progress = [self._knots]
            for segment in range(self._n_segments):
                bounds = self._segment_bounds[segment]  # axis -> (min, max)
                dn = max(bounds[0, 0] - point[0], 0.0, point[0] - bounds[0, 1])
                de = max(bounds[1, 0] - point[1], 0.0, point[1] - bounds[1, 1])
                if dn * dn + de * de > best_squared:
                    # Strictly farther than the best candidate: this segment
                    # can neither improve the minimum nor tie it, so its
                    # stationary roots cannot affect the deterministic
                    # earliest-progress tie break.
                    continue
                stationary = self._stationary_progress(point, segment)
                if stationary.size == 0:
                    continue
                candidate_progress.append(stationary)
                root_positions = self.position(stationary)
                best_squared = min(
                    best_squared,
                    float(np.min(((root_positions - point) ** 2).sum(axis=1))),
                )
            candidates = np.unique(np.concatenate(candidate_progress))
            candidate_positions = self.position(candidates)
            distance_squared = ((candidate_positions - point) ** 2).sum(axis=1)
            progress[point_index] = candidates[np.argmin(distance_squared)]

        proj_pos = self.position(progress)
        tangent = self.unit_tangent(progress)
        delta = pts - proj_pos
        cross_track = tangent[:, 1] * delta[:, 0] - tangent[:, 0] * delta[:, 1]
        return PathProjection(progress=progress, cross_track=cross_track, position=proj_pos)

    # --- sampling and LOS --------------------------------------------------

    def sample(self, n_points: int) -> tuple[np.ndarray, np.ndarray]:
        """Uniform-progress sampling for plots and legacy-compatible consumers.

        Args:
            n_points: number of samples (>= 2).

        Returns:
            ``(positions, headings)``: (n, 2) positions [m] and (n,) tangent
            headings [rad] at uniformly spaced progress from 0 to length.
        """
        if n_points < 2:
            raise ValueError("n_points must be >= 2")
        s = np.linspace(0.0, self.length, n_points)
        return self.position(s), self.heading(s)

    def lookahead_point(self, point: np.ndarray, lookahead: float) -> tuple[np.ndarray, float]:
        """LOS lookahead point and its progress [m] for one position.

        The lookahead lies ``lookahead`` [m] further in chord-progress from
        the projection of ``point``, clamped to the path end. It is not an
        exact curve arc-length offset.

        Args:
            point: (2,) position [m] (North, East).
            lookahead: lookahead distance [m], must be positive.

        Returns:
            ``(position, progress)`` of the lookahead point [m].
        """
        if lookahead <= 0.0:
            raise ValueError("lookahead must be positive")
        (s_proj,) = self.project(np.atleast_2d(np.asarray(point, dtype=float))).progress
        s_los = float(np.clip(s_proj + lookahead, 0.0, self.length))
        return self.position(s_los), s_los

    def los_heading(
        self,
        points: np.ndarray,
        lookahead: float,
        *,
        progress: np.ndarray | None = None,
    ) -> np.ndarray:
        """Exact smooth-geometry LOS desired heading [rad] at each point.

        Exact counterpart of ``guidance.los_heading`` on the polyline: the
        lookahead point lies ``lookahead`` [m] further in chord-progress from
        the projection (clamped to the path end), not by exact curve arc
        length, and the desired heading is its clockwise-positive bearing
        from North.

        Args:
            points: (n, 2) or (2,) positions [m] (North, East).
            lookahead: lookahead distance [m], must be positive.
            progress: optional (n,) projection progress [m] from
                :meth:`project`, reused to avoid projecting twice.

        Returns:
            (n,) desired headings [rad].
        """
        if lookahead <= 0.0:
            raise ValueError("lookahead must be positive")
        pts = np.atleast_2d(np.asarray(points, dtype=float))
        if progress is None:
            projected = self.project(pts).progress
        else:
            projected = np.asarray(progress, dtype=float)
            if projected.shape != (pts.shape[0],) or not np.all(np.isfinite(projected)):
                raise ValueError("progress must be a finite (n,) array")
        s_los = np.clip(projected + lookahead, 0.0, self.length)
        los = self.position(s_los)
        return np.arctan2(los[:, 1] - pts[:, 1], los[:, 0] - pts[:, 0])

    # --- CasADi evaluation -------------------------------------------------

    def _build_casadi(self) -> None:
        """Build scalar CasADi functions from the stored coefficients."""
        s = ca.MX.sym("s")
        s_c = ca.fmin(ca.fmax(s, 0.0), self.length)
        p_expr = ca.MX.zeros(2, 1)
        d_expr = ca.MX.zeros(2, 1)
        for i in range(self._n_segments - 1, -1, -1):
            u = s_c - self._knots[i]
            a_n, b_n, c_n, d_n = self._coeff_n[i]
            a_e, b_e, c_e, d_e = self._coeff_e[i]
            poly_n = a_n + u * (b_n + u * (c_n + u * d_n))
            poly_e = a_e + u * (b_e + u * (c_e + u * d_e))
            der_n = b_n + u * (2.0 * c_n + 3.0 * d_n * u)
            der_e = b_e + u * (2.0 * c_e + 3.0 * d_e * u)
            if i == self._n_segments - 1:
                p_expr = ca.vertcat(poly_n, poly_e)
                d_expr = ca.vertcat(der_n, der_e)
            else:
                p_expr = ca.if_else(s_c < self._knots[i + 1], ca.vertcat(poly_n, poly_e), p_expr)
                d_expr = ca.if_else(s_c < self._knots[i + 1], ca.vertcat(der_n, der_e), d_expr)
        t_expr = d_expr / ca.sqrt(ca.dot(d_expr, d_expr))
        h_expr = ca.atan2(t_expr[1], t_expr[0])
        self._f_position = ca.Function("path_position", [s], [p_expr])
        self._f_derivative = ca.Function("path_derivative", [s], [d_expr])
        self._f_unit_tangent = ca.Function("path_unit_tangent", [s], [t_expr])
        self._f_heading = ca.Function("path_heading", [s], [h_expr])

    @staticmethod
    def _as_mx(s: ca.MX | ca.DM | np.ndarray) -> ca.MX | ca.DM:
        return s if isinstance(s, (ca.MX, ca.DM)) else ca.MX(s)

    def _casadi_map(self, function: ca.Function, s: ca.MX) -> ca.MX:
        """Evaluate a scalar-s CasADi function at scalar or (n, 1) input."""
        s_mx = self._as_mx(s)
        if s_mx.numel() == 1:
            return function(s_mx)
        return function.map(s_mx.numel())(ca.reshape(s_mx, 1, -1))

    def casadi_position(self, s: ca.MX | np.ndarray) -> ca.MX:
        """Symbolic path position [m] using the same coefficients as NumPy.

        Args:
            s: scalar or (n, 1) progress [m] (clamped to the domain like the
                NumPy evaluator).

        Returns:
            (2, 1) MX for scalar input, (2, n) MX for n progress values.
        """
        return self._casadi_map(self._f_position, s)

    def casadi_derivative(self, s: ca.MX | np.ndarray) -> ca.MX:
        """Symbolic first derivative d/ds of the path position."""
        return self._casadi_map(self._f_derivative, s)

    def casadi_unit_tangent(self, s: ca.MX | np.ndarray) -> ca.MX:
        """Symbolic unit tangent (requires a nonzero derivative)."""
        return self._casadi_map(self._f_unit_tangent, s)

    def casadi_heading(self, s: ca.MX | np.ndarray) -> ca.MX:
        """Symbolic tangent heading [rad], clockwise-positive from North."""
        return self._casadi_map(self._f_heading, s)


def make_s_curve_geometry() -> PathGeometry:
    """Authoritative smooth S-curve geometry for the flagship scenario.

    Uses the legacy S-curve waypoints (:func:`guidance.make_s_curve_path`)
    refined with one collinear midpoint per chord before interpolation, so
    the monotone cubic stays within ~1.2 m of the polyline (the required
    < 3 m deviation bound) while still passing exactly through every
    original waypoint. Total length ~200 m.

    Returns:
        The smooth flagship path geometry.
    """
    return PathGeometry(_refine_polyline(make_s_curve_path(), 2))
