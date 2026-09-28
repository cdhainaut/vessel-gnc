"""Committed reference artifacts: JSON, provenance, consistency, determinism.

Owns the four schema-valid ``results/reference/`` JSON documents, the flagship
asset rendering, the generated Markdown marker bodies and the cheap
``check_reference_consistency`` validation. The reproducibility and provenance
contract (content source fingerprint, ``--check``, ``--verify-determinism``
tolerances, generation environment) is documented once in ``docs/validation.md``.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from vessel_gnc import _core
from vessel_gnc.mpcc import MPCC_COMPONENT_ID
from vessel_gnc.path import PathGeometry, make_s_curve_geometry
from vessel_gnc.reference import (
    DISTURBANCE_AWARE_NMPC_COMPONENT_ID,
    LOS_COMPONENT_ID,
    NMPC_COMPONENT_ID,
    ReferenceRun,
    ReferenceScenarioConfig,
    default_reference_config,
    reference_metrics,
    run_reference_scenario,
)
from vessel_gnc.reference_markdown import ARTIFACT_FILENAMES, marker_body_problems

__all__ = [
    "write_reference_json",
    "render_reference_assets",
    "check_reference_consistency",
    "verify_reference_determinism",
]

SCHEMA_RELPATH = "results/reference/reference.schema.json"
SCHEMA_VERSION = 3
SCENARIO_ID = "scenario_v3_mpcc"
BENCHMARK_ID = "benchmark_v3"

# Reproducibility contract of ``verify_reference_determinism``: the LOS
# baseline has no iterative solver and must reproduce exactly, while NMPC and
# MPCC solve with IPOPT at ``tol=1e-4`` (docs/control.md §5), whose
# full-precision iterates may legitimately differ in the last ulps between
# runs. Predictive-controller and estimator metrics are therefore compared
# with these relative/absolute tolerances, and the worst offending
# key/deviation is reported on failure.
DETERMINISM_RTOL = 1e-6
DETERMINISM_ATOL = 1e-6

COMMITTED_ASSET_RELPATHS = (
    "assets/hero.gif",
    "assets/controller_comparison.png",
    "assets/current_estimation.png",
)
HORIZON_SHOT_TIMES_S = (20.0, 50.0, 80.0)  # [s] hero horizon snapshots

# Source inputs whose content defines the fingerprint: the C++ core, the
# Python layer, the generator and the benchmark source. Untracked files are
# hashed as well as tracked ones (an uncommitted milestone must still be
# reproduced faithfully).
SOURCE_GLOBS = (
    "include/vessel_gnc/*",
    "src/*",
    "python/vessel_gnc/*.py",
    "tools/generate_reference_results.py",
    "benchmarks/benchmark_simulation.py",
    "results/reference/reference.schema.json",
    "pyproject.toml",
)

_SCENARIO_DESCRIPTION = (
    "Flagship reference scenario: one smooth S-curve geometry with LOS, nominal "
    "NMPC, disturbance-aware NMPC and disturbance-aware geometric MPCC. All "
    "run on matched EKF estimates under a rotating current, gusts and a "
    "perturbed truth plant; aware predictors hold the EKF equivalent-current "
    "estimate constant over their horizon."
)
_TIMING_TOKENS = ("solve", "wall", "elapsed", "_ms", "time_")


# --- public API --------------------------------------------------------------


def write_reference_json(
    run: ReferenceRun,
    benchmark: dict[str, object],
    reference_dir: Path,
) -> None:
    """Write the four committed reference artifacts (metadata hashes last).

    Writes ``config.json``, ``metrics.json`` and ``benchmark.json``, then
    ``metadata.json`` last so its ``artifacts`` hashes cover the three
    non-metadata JSON files just written and every committed generated asset
    currently on disk. All files use sorted keys, two-space indentation,
    finite native numbers and a trailing newline; the deterministic metrics
    artifact contains no timing values (timing lives in ``benchmark.json``).

    Args:
        run: the in-memory reference run (shared path plus LOS and NMPC
            controller runs).
        benchmark: the structured record returned by
            ``benchmarks.benchmark_simulation.run_benchmarks()``.
        reference_dir: the ``results/reference`` directory of the repository
            (created if needed); the repository root is derived from it.

    Example:
        >>> from vessel_gnc.reference_artifacts import write_reference_json
        >>> write_reference_json(run, benchmark, "results/reference")  # doctest: +SKIP
    """
    reference_dir = Path(reference_dir)
    repo_root = _repo_root_from(reference_dir)
    reference_dir.mkdir(parents=True, exist_ok=True)
    if run.path_geometry is None or run.disturbance_aware_mpcc is None:
        raise ValueError("schema v3 reference runs require a PathGeometry and MPCC controller run")
    _write_json(
        reference_dir / "config.json",
        _config_document(repo_root, run.config, run.path_geometry),
    )
    _write_json(reference_dir / "metrics.json", _metrics_document(run))
    _write_json(reference_dir / "benchmark.json", _benchmark_document(benchmark, repo_root))
    _write_json(reference_dir / "metadata.json", _metadata_document(repo_root))


def render_reference_assets(run: ReferenceRun, repo_root: Path) -> list[Path]:
    """Render the flagship assets from an in-memory reference run.

    Regenerates the three committed public assets (the hero animation
    ``assets/hero.gif``, the deterministic controller-comparison figure
    ``assets/controller_comparison.png`` and the current-estimation
    figure ``assets/current_estimation.png``) plus the ignored
    trajectory figure (``results/reference/nmpc_trajectory.png``), all from
    the same in-memory ``run`` so every asset shares the exact reference
    scenario and seed. Wall-clock timing never enters these figures:
    timing lives exclusively in ``benchmark.json`` and the generated
    benchmark tables.

    Args:
        run: the in-memory reference run the assets are derived from.
        repo_root: repository root containing ``assets/`` and
            ``results/reference/`` (both are created if needed).

    Returns:
        The list of written asset paths (hero, comparison, estimation,
        trajectory).

    Example:
        >>> from vessel_gnc.reference import run_reference_scenario
        >>> from vessel_gnc.reference_artifacts import render_reference_assets
        >>> render_reference_assets(
        ...     run_reference_scenario(), "."
        ... )  # doctest: +SKIP  (120 s flagship)
    """
    from vessel_gnc.reference_figures import (
        plot_controller_comparison,
        plot_current_estimation,
        plot_reference_trajectories,
    )

    repo_root = Path(repo_root)
    assets_dir = repo_root / "assets"
    reference_dir = repo_root / "results" / "reference"
    assets_dir.mkdir(parents=True, exist_ok=True)
    reference_dir.mkdir(parents=True, exist_ok=True)

    hero_path = assets_dir / "hero.gif"
    _render_hero(run, hero_path)
    comparison_path = assets_dir / "controller_comparison.png"
    plot_controller_comparison(run, reference_metrics(run), comparison_path)
    estimation_path = assets_dir / "current_estimation.png"
    plot_current_estimation(run, estimation_path)
    trajectory_path = reference_dir / "nmpc_trajectory.png"
    plot_reference_trajectories(run, trajectory_path)

    return [hero_path, comparison_path, estimation_path, trajectory_path]


def _render_hero(run: ReferenceRun, output_path: Path) -> None:
    """The four-controller hero animation with reference and horizons.

    Reuses the shared animation helper: the scene shows the truth-plant
    trajectories of all four controllers (LOS baseline, nominal NMPC,
    disturbance-aware NMPC and geometric MPCC), the recorded NMPC and MPCC
    prediction horizons, a compact path-progress overlay and the true
    (sampled) versus EKF-estimated current arrows, at the render settings
    recorded in ``run.config``. The animated vessel is the
    disturbance-aware NMPC run (the legacy hero); the other three
    trajectories are drawn as static comparison lines.
    """
    from vessel_gnc.visualization import animate_trajectory

    config = run.config
    controller = run.disturbance_aware_nmpc
    t_est = controller.estimator.t
    current_est = controller.estimator.current_estimate

    def estimated_environment(t: float) -> _core.Environment:
        # Nearest recorded filter estimate.
        idx = int(np.searchsorted(t_est, t, side="right"))
        idx = min(max(idx - 1, 0), len(t_est) - 1)
        return _core.Environment(
            current_north=float(current_est[idx, 0]),
            current_east=float(current_est[idx, 1]),
        )

    extra_trajectories = [
        ("LOS baseline", np.column_stack([run.los.result.x, run.los.result.y]), "0.6"),
        (
            "nominal NMPC",
            np.column_stack([run.nmpc.result.x, run.nmpc.result.y]),
            "tab:blue",
        ),
        (
            "disturbance-aware NMPC",
            np.column_stack(
                [
                    run.disturbance_aware_nmpc.result.x,
                    run.disturbance_aware_nmpc.result.y,
                ]
            ),
            "tab:green",
        ),
    ]
    extra_horizon = None
    extra_horizon_label = ""
    if run.disturbance_aware_mpcc is not None:
        extra_trajectories.append(
            (
                "geometric MPCC",
                np.column_stack(
                    [
                        run.disturbance_aware_mpcc.result.x,
                        run.disturbance_aware_mpcc.result.y,
                    ]
                ),
                "tab:purple",
            )
        )
        extra_horizon = run.disturbance_aware_mpcc.horizon
        extra_horizon_label = (
            f"geometric MPCC prediction ({config.mpcc.horizon * config.mpcc.dt:.0f} s horizon)"
        )

    def progress_text(t: float, x: float, y: float) -> str | None:
        """Compact deterministic path-progress line for the overlay."""
        if run.path_geometry is None:
            return None
        (progress,) = run.path_geometry.project(np.array([[x, y]])).progress
        return (
            f"path progress {progress:5.1f} m ({100.0 * progress / run.path_geometry.length:3.1f}%)"
        )

    animate_trajectory(
        controller.result,
        output_path=output_path,
        environment=config.environment.sample,
        estimated_environment=estimated_environment,
        title=(
            "Four controllers on the smooth S-curve: "
            "disturbance-aware NMPC with prediction horizons"
        ),
        stride=config.render_hero_stride_frames,
        fps=config.render_fps,
        wake_duration=config.render_hero_wake_duration_s,
        reference_path=run.path,
        horizon=controller.horizon,
        horizon_label=(
            f"disturbance-aware prediction ({config.nmpc.horizon * config.nmpc.dt:.0f} s horizon)"
        ),
        extra_trajectories=extra_trajectories,
        extra_horizon=extra_horizon,
        extra_horizon_label=extra_horizon_label,
        progress_text=progress_text,
    )


def check_reference_consistency(repo_root: Path) -> list[str]:
    """Cheap consistency validation of the committed reference artifacts.

    Validates the schema itself and every artifact against it, checks the
    scenario document against what the current code would generate (no
    simulation), verifies that the deterministic metrics contain no
    timing-like keys and only finite numbers, recomputes the content-based
    source fingerprint and artifact hashes, and compares every generated
    Markdown marker body exactly with what the committed JSON would
    produce. Never runs ``run_reference_scenario()`` or the benchmark
    workload, so it is safe for CI.

    The check compares only the *content* part of the source fingerprint
    (``files`` and ``sha256``) and ignores the volatile ``git_commit``
    field of ``config.json``: both record the generation-time repository
    state, not the current checkout. Any scenario, parameter or source
    change still fails the check through the content fingerprint, the
    scenario document comparison and the artifact hashes.

    Args:
        repo_root: repository root containing ``results/reference/``.

    Returns:
        The list of problems found (empty when the artifacts are consistent).

    Example:
        >>> from vessel_gnc.reference_artifacts import check_reference_consistency
        >>> problems = check_reference_consistency(".")  # doctest: +SKIP
    """
    try:
        from jsonschema import Draft202012Validator, SchemaError
    except ImportError as exc:
        return [f"jsonschema is required for --check (install the dev extra): {exc}"]

    repo_root = Path(repo_root)
    reference_dir = repo_root / "results" / "reference"
    schema_path = reference_dir / "reference.schema.json"
    problems: list[str] = []

    if not schema_path.is_file():
        return [f"missing reference schema: {schema_path}"]
    try:
        schema = json.loads(schema_path.read_text())
    except json.JSONDecodeError as exc:
        return [f"invalid reference schema JSON: {exc}"]
    try:
        Draft202012Validator.check_schema(schema)
    except SchemaError as exc:
        return [f"invalid reference schema: {exc}"]
    validator = Draft202012Validator(schema)

    documents: dict[str, dict[str, object]] = {}
    for name in ARTIFACT_FILENAMES:
        path = reference_dir / name
        if not path.is_file():
            problems.append(f"missing reference artifact: {path}")
            continue
        try:
            document = json.loads(path.read_text())
        except json.JSONDecodeError as exc:
            problems.append(f"invalid JSON in {path}: {exc}")
            continue
        documents[name] = document
        for error in validator.iter_errors(document):
            problems.append(f"{name} violates reference schema: {error.message}")

    config_document = documents.get("config.json")
    if config_document is not None:
        expected = _config_document(
            repo_root,
            default_reference_config(),
            make_s_curve_geometry(),
        )
        # git_commit is generation-time provenance, not a property of the
        # current checkout: it must not fail the check after the source is
        # committed at a different HEAD.
        difference = _first_difference(
            config_document, expected, ignored_keys=frozenset({"git_commit"})
        )
        if difference is not None:
            problems.append(f"config.json no longer matches the current code ({difference})")

    metrics_document = documents.get("metrics.json")
    if metrics_document is not None:
        timing_keys = _timing_keys(metrics_document)
        if timing_keys:
            problems.append(f"metrics.json contains timing-like keys: {sorted(timing_keys)}")
        if not _numbers_finite(metrics_document):
            problems.append("metrics.json contains non-finite numbers")

    metadata_document = documents.get("metadata.json")
    if metadata_document is not None:
        recorded = metadata_document.get("source_fingerprint", {})
        current = source_fingerprint(repo_root)
        if not _source_fingerprint_matches(recorded, current):
            problems.append("metadata.json source_fingerprint does not match the current tree")
        if metadata_document.get("artifacts") != _artifact_hashes(repo_root):
            problems.append("metadata.json artifact hashes do not match the committed files")

    problems.extend(marker_body_problems(repo_root))
    return problems


def verify_reference_determinism(repo_root: Path) -> None:
    """One fresh reference run compared with the committed metrics.

    Performs exactly one ``run_reference_scenario()`` (the flagship) and
    never runs a benchmark or writes anything. Raises ``AssertionError``
    when the fresh deterministic metrics violate the reproducibility
    contract: the LOS baseline metrics must match
    ``results/reference/metrics.json`` exactly (no iterative solver), while
    both NMPC variants, MPCC and estimator metrics must match within
    ``rtol=1e-6``, ``atol=1e-6``; IPOPT solves to ``tol=1e-4``
    (docs/control.md §5) and its full-precision iterates may legitimately
    differ in the last ulps between runs. On failure the message reports the
    worst offending key and its absolute/relative deviation. This is the
    explicit determinism validation command and must not be part of normal
    pytest.

    Args:
        repo_root: repository root containing ``results/reference/``.

    Raises:
        FileNotFoundError: when no committed ``metrics.json`` exists.
        AssertionError: when a fresh run violates the reproducibility
            contract.
    """
    repo_root = Path(repo_root)
    metrics_path = repo_root / "results" / "reference" / "metrics.json"
    if not metrics_path.is_file():
        raise FileNotFoundError(f"missing committed metrics artifact: {metrics_path}")
    committed = json.loads(metrics_path.read_text())
    fresh = reference_metrics(run_reference_scenario())

    los_dotted = f"controllers.{LOS_COMPONENT_ID}"
    if committed["controllers"][LOS_COMPONENT_ID] != fresh["controllers"][LOS_COMPONENT_ID]:
        difference = _first_difference(
            committed["controllers"][LOS_COMPONENT_ID],
            fresh["controllers"][LOS_COMPONENT_ID],
            los_dotted,
        )
        raise AssertionError(
            f"determinism check failed: metrics.json '{los_dotted}' differs "
            f"exactly ({difference}); the LOS baseline has no iterative "
            "solver, so its metrics must reproduce bit-for-bit"
        )

    for dotted, committed_section, fresh_section in (
        (
            f"controllers.{NMPC_COMPONENT_ID}",
            committed["controllers"][NMPC_COMPONENT_ID],
            fresh["controllers"][NMPC_COMPONENT_ID],
        ),
        (
            f"controllers.{DISTURBANCE_AWARE_NMPC_COMPONENT_ID}",
            committed["controllers"][DISTURBANCE_AWARE_NMPC_COMPONENT_ID],
            fresh["controllers"][DISTURBANCE_AWARE_NMPC_COMPONENT_ID],
        ),
        (
            f"controllers.{MPCC_COMPONENT_ID}",
            committed["controllers"][MPCC_COMPONENT_ID],
            fresh["controllers"][MPCC_COMPONENT_ID],
        ),
        ("estimator", committed["estimator"], fresh["estimator"]),
    ):
        violation = _worst_metric_violation(committed_section, fresh_section, f"{dotted}.")
        if violation is not None:
            path, abs_dev, rel_dev, _ = violation
            raise AssertionError(
                "determinism check failed: "
                f"metrics.json '{dotted}' exceeds the reproducibility "
                f"tolerance (rtol={DETERMINISM_RTOL:g}, "
                f"atol={DETERMINISM_ATOL:g}); worst key '{path}': "
                f"absolute deviation {abs_dev:.3e}, relative deviation "
                f"{rel_dev:.3e}"
            )


# --- generated Markdown markers ---------------------------------------------


def _config_document(
    repo_root: Path,
    config: ReferenceScenarioConfig,
    path: PathGeometry,
) -> dict[str, object]:
    """The versioned config.json document for a scenario and its path."""
    return {
        "$schema": SCHEMA_RELPATH,
        "artifact_type": "config",
        "schema_version": SCHEMA_VERSION,
        "git_commit": _git_commit(repo_root),
        "scenario": _scenario_document(config, path),
    }


def _scenario_document(
    config: ReferenceScenarioConfig,
    path: PathGeometry,
) -> dict[str, object]:
    """The scenario record with parameter values, not digests.

    The key names and units mirror ``results/reference/reference.schema.json``
    (``#/$defs/scenario``); every number is a native Python scalar so the
    document round-trips through JSON exactly.
    """
    env = config.environment
    sensors = config.sensors
    nmpc = config.nmpc
    return {
        "id": SCENARIO_ID,
        "components": {
            "path": "smooth_s_curve_v1",
            "environment": "rotating_current_gusts_v1",
            "controller_los": LOS_COMPONENT_ID,
            "controller_nmpc": NMPC_COMPONENT_ID,
            "controller_disturbance_aware_nmpc": (DISTURBANCE_AWARE_NMPC_COMPONENT_ID),
            "controller_disturbance_aware_mpcc": MPCC_COMPONENT_ID,
            "estimator": "augmented_current_ekf_v1",
        },
        "description": _SCENARIO_DESCRIPTION,
        "revision": 1,
        "seed": config.seed,
        "duration_s": config.duration_s,
        "integration_dt_s": config.integration_dt_s,
        "control_periods": {
            "estimator_s": config.estimator_period_s,
            "los_s": config.los_period_s,
            "nmpc_s": config.nmpc_period_s,
            "mpcc_s": config.mpcc_period_s,
        },
        "path": {
            "name": "smooth_s_curve",
            "interpolation": "piecewise_cubic_hermite_pchip",
            "parameterization": "cumulative_waypoint_chord_length_m",
            "parameterization_is_exact_arc_length": False,
            "out_of_domain_progress": "clamped",
            "projection": "global_closest_point_earliest_progress_tie_break",
            "progress_domain_start_m": 0.0,
            "total_progress_m": path.length,
            "speed_ref_m_s": config.speed_ref_m_s,
            "lookahead_m": config.lookahead_m,
            "waypoints": [[float(north), float(east)] for north, east in path.waypoints],
        },
        "environment": {
            "current_base_east_m_s": env.current_base_east,
            "current_amplitude_m_s": env.current_amplitude,
            "current_period_s": env.current_period,
            "current_phase_rad": env.current_phase,
            "wind_mean_east_N": env.wind_mean_east,
            "gust_times_s": [float(t) for t in env.gust_times],
            "gust_peak_N": env.gust_peak,
            "gust_width_s": env.gust_width,
        },
        "sensors": {
            "gnss_period_s": sensors.gnss_period,
            "compass_period_s": sensors.compass_period,
            "speed_period_s": sensors.speed_period,
            "gyro_period_s": sensors.gyro_period,
            "gnss_sigma_m": sensors.gnss_sigma,
            "compass_sigma_rad": sensors.compass_sigma,
            "speed_sigma_m_s": sensors.speed_sigma,
            "gyro_sigma_rad_s": sensors.gyro_sigma,
        },
        "nmpc": {
            "horizon": nmpc.horizon,
            "dt_s": nmpc.dt,
            "substeps": nmpc.substeps,
            "q_position": nmpc.q_position,
            "q_heading": nmpc.q_heading,
            "r_thrust": nmpc.r_thrust,
            "r_moment": nmpc.r_moment,
            "s_thrust": nmpc.s_thrust,
            "s_moment": nmpc.s_moment,
            "warm_start": nmpc.warm_start,
        },
        "mpcc": {
            "horizon": config.mpcc.horizon,
            "dt_s": config.mpcc.dt,
            "substeps": config.mpcc.substeps,
            "q_contour": config.mpcc.q_contour,
            "q_lag": config.mpcc.q_lag,
            "q_heading": config.mpcc.q_heading,
            "q_progress": config.mpcc.q_progress,
            "r_thrust": config.mpcc.r_thrust,
            "r_moment": config.mpcc.r_moment,
            "s_thrust": config.mpcc.s_thrust,
            "s_moment": config.mpcc.s_moment,
            "r_vs": config.mpcc.r_vs,
            "s_vs": config.mpcc.s_vs,
            "progress_speed_ref_m_s": config.mpcc.progress_speed_ref,
            "progress_speed_max_m_s": config.mpcc.progress_speed_max,
            "warm_start": config.mpcc.warm_start,
        },
        "los": {
            "heading_gains": _pid_gains(config.los_heading_gains),
            "speed_gains": _pid_gains(config.los_speed_gains),
            "heading_moment_limit_Nm": config.los_heading_moment_limit_Nm,
            "speed_thrust_limit_N": config.los_speed_thrust_limit_N,
            "period_s": config.los_period_s,
        },
        "nominal_params": _model_params(config.nominal_params),
        "truth_params": _model_params(config.truth_params),
        "render": {
            "fps": config.render_fps,
            "hero_stride_frames": config.render_hero_stride_frames,
            "hero_wake_duration_s": config.render_hero_wake_duration_s,
            "horizon_shot_times_s": list(HORIZON_SHOT_TIMES_S),
            "path_render_samples": config.path_render_samples,
        },
    }


def _metrics_document(run: ReferenceRun) -> dict[str, object]:
    """The deterministic metrics.json artifact (no timing or provenance)."""
    metrics = reference_metrics(run)
    return {
        "$schema": SCHEMA_RELPATH,
        "artifact_type": "metrics",
        "schema_version": SCHEMA_VERSION,
        "scenario_id": SCENARIO_ID,
        "controllers": metrics["controllers"],
        "estimator": metrics["estimator"],
    }


def _benchmark_document(benchmark: dict[str, object], repo_root: Path) -> dict[str, object]:
    """The benchmark.json artifact: machine-dependent timing only."""
    if benchmark.get("benchmark_id") != BENCHMARK_ID or "workloads" not in benchmark:
        raise ValueError(
            f"benchmark must contain benchmark_id={BENCHMARK_ID!r} and 'workloads' "
            "(see benchmarks/benchmark_simulation.py run_benchmarks)"
        )
    return {
        "$schema": SCHEMA_RELPATH,
        "artifact_type": "benchmark",
        "schema_version": SCHEMA_VERSION,
        "benchmark_id": benchmark["benchmark_id"],
        "git_commit": _git_commit(repo_root),
        "workloads": benchmark["workloads"],
    }


def _metadata_document(repo_root: Path) -> dict[str, object]:
    """The metadata.json artifact: environment, fingerprint and hashes."""
    return {
        "$schema": SCHEMA_RELPATH,
        "artifact_type": "metadata",
        "schema_version": SCHEMA_VERSION,
        "generated_at_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "software": _software_versions(),
        "platform": {"os": platform.system(), "machine": platform.machine()},
        "source_fingerprint": source_fingerprint(repo_root),
        "artifacts": _artifact_hashes(repo_root),
    }


def _pid_gains(gains: _core.PidGains) -> dict[str, float]:
    """Serializable PID gains (kp/ki/kd)."""
    return {"kp": gains.kp, "ki": gains.ki, "kd": gains.kd}


def _model_params(params: _core.ModelParams) -> dict[str, float]:
    """Serializable ModelParams record with SI-unit key names."""
    return {
        "mass_kg": params.mass,
        "inertia_z_kg_m2": params.inertia_z,
        "added_mass_x_kg": params.added_mass_x,
        "added_mass_y_kg": params.added_mass_y,
        "added_inertia_z_kg_m2": params.added_inertia_z,
        "lin_damping_u_N_s_m": params.lin_damping_u,
        "lin_damping_v_N_s_m": params.lin_damping_v,
        "lin_damping_r_N_m_s_rad": params.lin_damping_r,
        "quad_damping_u_N_s2_m2": params.quad_damping_u,
        "quad_damping_v_N_s2_m2": params.quad_damping_v,
        "quad_damping_r_N_m_s2_rad2": params.quad_damping_r,
        "thrust_min_N": params.thrust_min,
        "thrust_max_N": params.thrust_max,
        "moment_min_Nm": params.moment_min,
        "moment_max_Nm": params.moment_max,
        "thrust_time_constant_s": params.thrust_time_constant,
        "moment_time_constant_s": params.moment_time_constant,
        "thrust_rate_limit_N_s": params.thrust_rate_limit,
        "moment_rate_limit_N_m_s": params.moment_rate_limit,
    }


# --- provenance ---------------------------------------------------------------


def source_fingerprint(repo_root: Path) -> dict[str, object]:
    """Deterministic provenance of the source inputs at generation time.

    Returns the ordered relative source paths (sorted, forward slashes), the
    combined SHA-256 of path + content per file (a missing file contributes
    empty content, so the digest changes when any source input appears,
    disappears or changes) and a dirty flag recording whether tracked files
    carry uncommitted modifications. The content part (``files`` +
    ``sha256``) is authoritative for consistency checks; ``dirty`` is an
    honest generation-time record and is never compared against the current
    checkout.

    Example:
        >>> from vessel_gnc.reference_artifacts import source_fingerprint
        >>> fp = source_fingerprint(".")  # doctest: +SKIP
        >>> len(fp["sha256"])
        64
    """
    repo_root = Path(repo_root)
    relpaths = sorted(
        {
            relpath.relative_to(repo_root).as_posix()
            for pattern in SOURCE_GLOBS
            for relpath in repo_root.glob(pattern)
        }
    )
    digest = hashlib.sha256()
    for relpath in relpaths:
        digest.update(relpath.encode())
        digest.update(b"\0")
        digest.update(_read_optional(repo_root / relpath))
    return {
        "dirty": _git_dirty(repo_root),
        "files": relpaths,
        "sha256": digest.hexdigest(),
    }


def _artifact_hashes(repo_root: Path) -> dict[str, str]:
    """SHA-256 of the committed artifacts, keyed by repository-relative path.

    Covers the three non-metadata JSON artifacts and every committed
    generated asset; ``metadata.json`` is excluded from its own hash to avoid
    a cycle. Files absent from disk are simply not hashed.
    """
    relpaths = (
        *(f"results/reference/{name}" for name in ARTIFACT_FILENAMES[:3]),
        *COMMITTED_ASSET_RELPATHS,
    )
    hashes: dict[str, str] = {}
    for relpath in relpaths:
        path = repo_root / relpath
        if path.is_file():
            hashes[relpath] = hashlib.sha256(path.read_bytes()).hexdigest()
    return hashes


def _software_versions() -> dict[str, str]:
    """Python and library versions recorded in metadata (generation only)."""
    import casadi
    import matplotlib
    import numpy
    import PIL

    import vessel_gnc

    return {
        "python": platform.python_version(),
        "numpy": numpy.__version__,
        "matplotlib": matplotlib.__version__,
        "pillow": PIL.__version__,
        "casadi": casadi.__version__,
        "vessel_gnc": vessel_gnc.__version__,
    }


# --- small helpers -------------------------------------------------------------


def _write_json(path: Path, document: dict[str, object]) -> None:
    """Canonical artifact form: sorted keys, 2-space indent, finite, LF."""
    text = json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
    path.write_text(text)


def _repo_root_from(reference_dir: Path) -> Path:
    """The repository root containing ``reference_dir`` (results/reference)."""
    reference_dir = reference_dir.resolve()
    for candidate in (reference_dir, *reference_dir.parents):
        if candidate / "results" / "reference" == reference_dir:
            return candidate
    raise ValueError(f"{reference_dir} is not a results/reference directory")


def _git_commit(repo_root: Path) -> str:
    """HEAD commit of the repository ("" outside a git work tree)."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.stdout.strip() if result.returncode == 0 else ""


def _git_dirty(repo_root: Path) -> bool:
    """Whether tracked files carry uncommitted modifications."""
    result = subprocess.run(
        ["git", "-C", str(repo_root), "status", "--porcelain", "--untracked-files=no"],
        capture_output=True,
        text=True,
        check=False,
    )
    return result.returncode == 0 and bool(result.stdout.strip())


def _read_optional(path: Path) -> bytes:
    try:
        return path.read_bytes()
    except FileNotFoundError:
        return b""


def _source_fingerprint_matches(recorded: dict[str, object], current: dict[str, object]) -> bool:
    """Content-based fingerprint equality: source list and combined digest.

    ``dirty`` is a generation-time provenance record (as are the
    ``git_commit`` fields of config/benchmark) and is deliberately not
    compared: after the source is committed, a clean checkout reports
    ``dirty: false`` and a new HEAD while the content digest is unchanged.
    The content fingerprint remains authoritative: any scenario/parameter/
    source change alters ``files`` or ``sha256`` and fails the check.
    """
    return recorded.get("files") == current["files"] and recorded.get("sha256") == current["sha256"]


def _first_difference(
    expected: object,
    actual: object,
    prefix: str = "",
    ignored_keys: frozenset[str] = frozenset(),
) -> str | None:
    """The first differing location between two JSON-compatible objects.

    Keys named by ``ignored_keys`` hold volatile generation-time provenance
    (``git_commit``) and are not compared.
    """
    if type(expected) is not type(actual):
        return f"{prefix} type {type(expected).__name__} != {type(actual).__name__}"
    if isinstance(expected, dict):
        expected_keys = set(expected) - ignored_keys
        actual_keys = set(actual) - ignored_keys
        if expected_keys != actual_keys:
            return f"{prefix} keys differ: {sorted(expected_keys ^ actual_keys)}"
        for key in expected:
            if key in ignored_keys:
                continue
            difference = _first_difference(
                expected[key], actual[key], f"{prefix}.{key}", ignored_keys
            )
            if difference is not None:
                return difference
        return None
    if isinstance(expected, list):
        if len(expected) != len(actual):
            return f"{prefix} length {len(expected)} != {len(actual)}"
        for index, (item, other) in enumerate(zip(expected, actual, strict=True)):
            difference = _first_difference(item, other, f"{prefix}[{index}]")
            if difference is not None:
                return difference
        return None
    if expected != actual:
        return f"{prefix}: {expected!r} != {actual!r}"
    return None


def _timing_keys(document: dict[str, object]) -> list[str]:
    """Dotted keys that look like wall-clock timing (metrics must have none)."""

    def walk(value: object, prefix: str) -> list[str]:
        hits: list[str] = []
        if isinstance(value, dict):
            for key, item in value.items():
                path = f"{prefix}.{key}" if prefix else key
                if any(token in key for token in _TIMING_TOKENS):
                    hits.append(path)
                hits.extend(walk(item, path))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                hits.extend(walk(item, f"{prefix}[{index}]"))
        return hits

    return walk(document, "")


def _numbers_finite(document: object) -> bool:
    """Whether every JSON number in the document is finite."""
    if isinstance(document, dict):
        return all(_numbers_finite(value) for value in document.values())
    if isinstance(document, list):
        return all(_numbers_finite(value) for value in document)
    if isinstance(document, (int, float)) and not isinstance(document, bool):
        import math

        return math.isfinite(document)
    return True


def _worst_metric_violation(
    committed: object,
    fresh: object,
    prefix: str = "",
) -> tuple[str, float, float, float] | None:
    """The leaf violating the reproducibility tolerance most severely.

    Recursive comparison of JSON-compatible metric documents: numeric
    leaves must satisfy the reproducibility contract (``math.isclose`` with
    ``rel_tol=DETERMINISM_RTOL``, ``abs_tol=DETERMINISM_ATOL``), while
    structure (dict keys, list lengths) and non-numeric leaves must match
    exactly. Returns ``(dotted path, absolute deviation, relative
    deviation, normalized margin)`` of the worst violating leaf, where the
    margin is ``abs_dev / (atol + rtol * max(|a|, |b|))`` (1.0 means
    exactly at the tolerance budget; structural mismatches carry an
    infinite margin). Returns ``None`` when every leaf matches within
    tolerance.
    """
    if isinstance(committed, dict) and isinstance(fresh, dict):
        worst: tuple[str, float, float, float] | None = None
        for key in sorted(set(committed) | set(fresh)):
            if key not in committed or key not in fresh:
                violation = (f"{prefix}{key}", math.inf, math.inf, math.inf)
            else:
                violation = _worst_metric_violation(committed[key], fresh[key], f"{prefix}{key}.")
            worst = _pick_worst(worst, violation)
        return worst
    if isinstance(committed, list) and isinstance(fresh, list):
        if len(committed) != len(fresh):
            return (prefix.rstrip("."), math.inf, math.inf, math.inf)
        worst = None
        for index, (item, other) in enumerate(zip(committed, fresh, strict=True)):
            violation = _worst_metric_violation(item, other, f"{prefix}{index}.")
            worst = _pick_worst(worst, violation)
        return worst
    if isinstance(committed, (int, float)) and isinstance(fresh, (int, float)):
        a, b = float(committed), float(fresh)
        if math.isclose(b, a, rel_tol=DETERMINISM_RTOL, abs_tol=DETERMINISM_ATOL):
            return None
        abs_dev = abs(b - a)
        scale = max(abs(a), abs(b), np.finfo(float).tiny)
        margin = abs_dev / (DETERMINISM_ATOL + DETERMINISM_RTOL * max(abs(a), abs(b)))
        return (prefix.rstrip("."), abs_dev, abs_dev / scale, margin)
    if committed == fresh:
        return None
    return (prefix.rstrip("."), math.inf, math.inf, math.inf)


def _pick_worst(
    worst: tuple[str, float, float, float] | None,
    violation: tuple[str, float, float, float] | None,
) -> tuple[str, float, float, float] | None:
    """The violation deviating most from the tolerance budget.

    Ranked by the normalized margin (a structural mismatch always wins);
    ties are broken by the absolute deviation.
    """
    if violation is None:
        return worst
    if worst is None or violation[3] > worst[3]:
        return violation
    if violation[3] == worst[3] and violation[1] > worst[1]:
        return violation
    return worst
