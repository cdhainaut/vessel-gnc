"""Committed reference artifacts: JSON, provenance, consistency, determinism.

This module owns the four schema-valid reference artifacts under
``results/reference/`` (plan M1): ``config.json`` (full scenario
configuration with parameter values, not digests), ``metrics.json``
(deterministic path-following and estimator metrics, no timing),
``benchmark.json`` (machine-dependent timing only) and ``metadata.json``
(software/platform versions, source fingerprint, artifact hashes, written
last so its hashes cover the other three files and every committed asset).
``render_reference_assets`` regenerates the flagship assets (hero animation,
controller comparison, current estimation) and the ignored trajectory
figure from a shared in-memory reference run.

Canonical JSON form is fixed in the plan: sorted keys, two-space
indentation, finite native numbers (``allow_nan=False``) and a trailing
newline. ``check_reference_consistency`` is deliberately cheap — it never
runs the reference scenario or the benchmark workload, so CI can validate
the committed artifacts on every push. ``verify_reference_determinism`` is
the only entry point that performs a fresh reference run, and it writes
nothing. Its comparison follows the reproducibility contract: the LOS
baseline metrics must reproduce exactly (no iterative solver), while the
NMPC and estimator metrics must match within ``rtol=1e-6, atol=1e-6``
because IPOPT (``tol=1e-4``) can legitimately differ in the last ulps
between runs; a violation is reported with the worst offending key and its
deviation (docs/control.md §5).

Provenance semantics: ``config.json``/``benchmark.json`` record the
``git_commit`` and ``metadata.json`` records the ``dirty`` flag of the
repository *at generation time*. These are honest historical records and
are not compared against the current checkout — after the source is
committed, a clean checkout reports a new HEAD and ``dirty: false`` while
the artifact contents stay valid. The authoritative consistency check is
the *content* source fingerprint (ordered source paths plus combined
SHA-256 over ``SOURCE_GLOBS``): it changes if and only if a source file
appears, disappears or changes content. The workflow is therefore either
commit the source first and then regenerate the artifacts, or keep the
source contents unchanged; ``--check`` then passes in any clean checkout
whose tree has identical source contents, and fails whenever a source
change is not reflected in the committed fingerprint.

``update_generated_markdown`` owns the public Markdown numbers: every
``<!-- generated:<marker-id>:start -->`` / ``<!-- generated:<marker-id>:end -->``
pair in README.md and the documentation files gets its body regenerated
from the committed reference JSON, so the only place public numbers can
appear is between the markers — never hand-edited.
"""

from __future__ import annotations

import hashlib
import json
import math
import platform
import re
import subprocess
from datetime import UTC, datetime
from pathlib import Path

import numpy as np

from vessel_gnc import _core
from vessel_gnc.guidance import make_s_curve_path
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

__all__ = [
    "write_reference_json",
    "render_reference_assets",
    "update_generated_markdown",
    "check_reference_consistency",
    "verify_reference_determinism",
]

SCHEMA_RELPATH = "results/reference/reference.schema.json"
SCHEMA_VERSION = 2
SCENARIO_ID = "scenario_v2_disturbance_aware"

# Reproducibility contract of ``verify_reference_determinism``: the LOS
# baseline has no iterative solver and must reproduce exactly, while the
# NMPC solves with IPOPT at ``tol=1e-4`` (docs/control.md §5), whose
# full-precision iterates may legitimately differ in the last ulps between
# runs. The NMPC and estimator metrics are therefore compared with these
# relative/absolute tolerances, and the worst offending key/deviation is
# reported on failure.
DETERMINISM_RTOL = 1e-6
DETERMINISM_ATOL = 1e-6

ARTIFACT_FILENAMES = ("config.json", "metrics.json", "benchmark.json", "metadata.json")
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
)

_SCENARIO_DESCRIPTION = (
    "Flagship reference scenario: S-curve path (s_curve_v1) with LOS, nominal "
    "NMPC and disturbance-aware NMPC. All run on EKF state estimates under a "
    "rotating current, gusts and a perturbed truth plant; the aware predictor "
    "holds the EKF equivalent-current estimate constant over its horizon."
)
_TIMING_TOKENS = ("solve", "wall", "elapsed", "_ms", "time_")

# Generated Markdown markers: each marker ID maps to the documentation files
# that must contain exactly one ``<!-- generated:<id>:start -->`` /
# ``<!-- generated:<id>:end -->`` pair. The bodies between the pairs are the
# only place where public numbers are displayed; ``update_generated_markdown``
# regenerates them from the committed reference JSON and never appends a
# second table, so missing/duplicated/misplaced pairs are a hard error.
MARKDOWN_MARKERS: dict[str, tuple[str, ...]] = {
    "reference-benchmark-v1": ("README.md", "docs/validation.md"),
    "reference-controller-comparison-v1": ("README.md", "docs/control.md"),
    "reference-estimator-v1": ("docs/estimation.md",),
    "reference-provenance-v1": ("README.md", "docs/validation.md"),
}

_MARKER_RE = re.compile(r"<!-- generated:([a-z0-9-]+):(start|end) -->")

# Deterministic controller rows of the comparison marker, in display order:
# (label, metrics.json key, format spec, convert-to-degrees flag).
_COMPARISON_ROWS = (
    ("RMS cross-track error [m]", "cross_track_rms_m", ".2f", False),
    ("P95 cross-track error [m]", "cross_track_p95_m", ".2f", False),
    ("Max cross-track error [m]", "cross_track_max_m", ".2f", False),
    ("RMS wrapped heading error [deg]", "heading_error_rms_rad", ".1f", True),
    ("Max wrapped heading error [deg]", "heading_error_max_rad", ".1f", True),
    ("RMS applied thrust [N]", "thrust_rms_N", ".1f", False),
    ("Max applied thrust [N]", "thrust_max_N", ".1f", False),
    ("RMS applied yaw moment [N m]", "moment_rms_Nm", ".1f", False),
    ("Max applied yaw moment [N m]", "moment_max_Nm", ".1f", False),
    ("Thrust saturation duration [s]", "thrust_saturation_duration_s", ".1f", False),
    (
        "Yaw-moment saturation duration [s]",
        "moment_saturation_duration_s",
        ".1f",
        False,
    ),
    ("Either channel saturated [s]", "any_saturation_duration_s", ".1f", False),
)


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
    _write_json(
        reference_dir / "config.json",
        _config_document(repo_root, run.config, run.path),
    )
    _write_json(reference_dir / "metrics.json", _metrics_document(run))
    _write_json(
        reference_dir / "benchmark.json", _benchmark_document(benchmark, repo_root)
    )
    _write_json(reference_dir / "metadata.json", _metadata_document(repo_root))


def render_reference_assets(run: ReferenceRun, repo_root: Path) -> list[Path]:
    """Render the flagship assets from an in-memory reference run.

    Regenerates the three committed public assets — the hero animation
    (``assets/hero.gif``), the deterministic controller-comparison figure
    (``assets/controller_comparison.png``) and the current-estimation
    figure (``assets/current_estimation.png``) — plus the ignored
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
    from vessel_gnc.visualization import (
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
    """The aware-NMPC hero animation with reference and prediction horizons.

    Reuses the shared animation helper: the scene shows the truth-plant
    trajectory, the reference path, the recorded NMPC predictions and the
    true (sampled) versus EKF-estimated current arrows, at the render
    settings recorded in ``run.config``.
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

    animate_trajectory(
        controller.result,
        output_path=output_path,
        environment=config.environment.sample,
        estimated_environment=estimated_environment,
        title="Disturbance-aware NMPC — predicted horizon",
        stride=config.render_hero_stride_frames,
        fps=config.render_fps,
        wake_duration=config.render_hero_wake_duration_s,
        reference_path=run.path,
        horizon=controller.horizon,
        horizon_label=(
            "disturbance-aware prediction "
            f"({config.nmpc.horizon * config.nmpc.dt:.0f} s horizon)"
        ),
    )


def update_generated_markdown(repo_root: Path) -> None:
    """Regenerate the numbers between the reference marker pairs.

    Reads the four committed reference artifacts and rewrites the Markdown
    bodies between every ``<!-- generated:<marker-id>:start -->`` /
    ``<!-- generated:<marker-id>:end -->`` pair in README.md and the
    documentation files. All displayed numbers are formatted from the JSON
    artifacts — nothing is hand-entered — so a regeneration cannot drift
    from the committed data. Missing, duplicated, malformed or misplaced
    marker pairs are a hard error: the generator never appends a second
    table. Documentation files are left untouched when the reference
    artifacts are absent (the default pipeline writes them first).

    Args:
        repo_root: repository root containing ``results/reference/`` and
            the Markdown files.

    Example:
        >>> from vessel_gnc.reference_artifacts import update_generated_markdown
        >>> update_generated_markdown(".")  # doctest: +SKIP
    """
    repo_root = Path(repo_root)
    reference_dir = repo_root / "results" / "reference"
    if any(not (reference_dir / name).is_file() for name in ARTIFACT_FILENAMES):
        return None
    bodies = _marker_bodies(repo_root)
    for relpath in _marker_relpaths():
        path = repo_root / relpath
        if not path.is_file():
            continue
        original = path.read_text()
        text = original
        pairs = _marker_pairs(text, relpath)
        _validate_marker_placement(relpath, pairs)
        for marker_id, body_start, body_end in sorted(
            pairs, key=lambda pair: pair[1], reverse=True
        ):
            text = text[:body_start] + bodies[marker_id] + text[body_end:]
        if text != original:
            path.write_text(text)
    return None


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
            repo_root, default_reference_config(), make_s_curve_path()
        )
        # git_commit is generation-time provenance, not a property of the
        # current checkout: it must not fail the check after the source is
        # committed at a different HEAD.
        difference = _first_difference(
            config_document, expected, ignored_keys=frozenset({"git_commit"})
        )
        if difference is not None:
            problems.append(
                f"config.json no longer matches the current code ({difference})"
            )

    metrics_document = documents.get("metrics.json")
    if metrics_document is not None:
        timing_keys = _timing_keys(metrics_document)
        if timing_keys:
            problems.append(
                f"metrics.json contains timing-like keys: {sorted(timing_keys)}"
            )
        if not _numbers_finite(metrics_document):
            problems.append("metrics.json contains non-finite numbers")

    metadata_document = documents.get("metadata.json")
    if metadata_document is not None:
        recorded = metadata_document.get("source_fingerprint", {})
        current = source_fingerprint(repo_root)
        if not _source_fingerprint_matches(recorded, current):
            problems.append(
                "metadata.json source_fingerprint does not match the current tree"
            )
        if metadata_document.get("artifacts") != _artifact_hashes(repo_root):
            problems.append(
                "metadata.json artifact hashes do not match the committed files"
            )

    problems.extend(_marker_body_problems(repo_root))
    return problems


def verify_reference_determinism(repo_root: Path) -> None:
    """One fresh reference run compared with the committed metrics.

    Performs exactly one ``run_reference_scenario()`` (the flagship) and
    never runs a benchmark or writes anything. Raises ``AssertionError``
    when the fresh deterministic metrics violate the reproducibility
    contract: the LOS baseline metrics must match
    ``results/reference/metrics.json`` exactly (no iterative solver), while
    both NMPC variants and estimator metrics must match within ``rtol=1e-6``,
    ``atol=1e-6`` — IPOPT solves to ``tol=1e-4`` (docs/control.md §5) and
    its full-precision iterates may legitimately differ in the last ulps
    between runs. On failure the message reports the worst offending key
    and its absolute/relative deviation. This is the explicit determinism
    validation command and must not be part of normal pytest.

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
    if (
        committed["controllers"][LOS_COMPONENT_ID]
        != fresh["controllers"][LOS_COMPONENT_ID]
    ):
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
        ("estimator", committed["estimator"], fresh["estimator"]),
    ):
        violation = _worst_metric_violation(
            committed_section, fresh_section, f"{dotted}."
        )
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


def _marker_relpaths() -> tuple[str, ...]:
    """The documentation files that may contain generated markers, sorted."""
    files = {relpath for relpaths in MARKDOWN_MARKERS.values() for relpath in relpaths}
    return tuple(sorted(files))


def _marker_pairs(text: str, relpath: str) -> list[tuple[str, int, int]]:
    """``(marker_id, body_start, body_end)`` for each complete marker pair.

    ``body_start``/``body_end`` are the character offsets of the body between
    the start and end comment. Raises ``ValueError`` on unknown marker IDs,
    malformed pairs (start without end or vice versa) and duplicated pairs.
    """
    starts: dict[str, list[int]] = {}
    ends: dict[str, list[int]] = {}
    for match in _MARKER_RE.finditer(text):
        marker_id, kind = match.group(1), match.group(2)
        if kind == "start":
            starts.setdefault(marker_id, []).append(match.end())
        else:
            ends.setdefault(marker_id, []).append(match.start())
    problems: list[str] = []
    for marker_id in starts:
        if marker_id not in MARKDOWN_MARKERS:
            problems.append(f"{relpath}: unknown generated marker '{marker_id}'")
        elif len(starts[marker_id]) != 1 or len(ends.get(marker_id, [])) != 1:
            problems.append(
                f"{relpath}: generated marker '{marker_id}' must appear exactly "
                "once as a start/end pair"
            )
    for marker_id in ends:
        if marker_id not in starts:
            problems.append(
                f"{relpath}: generated marker '{marker_id}' end without start"
            )
    if problems:
        raise ValueError("; ".join(problems))
    return [
        (marker_id, starts[marker_id][0], ends[marker_id][0]) for marker_id in starts
    ]


def _validate_marker_placement(relpath: str, pairs: list[tuple[str, int, int]]) -> None:
    """The marker set of a file must equal its expected set exactly."""
    expected = {
        marker_id
        for marker_id, relpaths in MARKDOWN_MARKERS.items()
        if relpath in relpaths
    }
    present = {marker_id for marker_id, _, _ in pairs}
    if present != expected:
        details: list[str] = []
        missing = sorted(expected - present)
        extra = sorted(present - expected)
        if missing:
            details.append(f"missing marker(s): {', '.join(missing)}")
        if extra:
            details.append(f"unexpected marker(s): {', '.join(extra)}")
        raise ValueError(f"{relpath}: {'; '.join(details)}")


def _marker_bodies(repo_root: Path) -> dict[str, str]:
    """Generated Markdown body per marker ID, formatted from the reference JSON.

    Every displayed number is formatted from
    ``results/reference/{config,metrics,benchmark,metadata}.json``; the bodies
    are byte-identical for a given artifact state, so the consistency check
    compares them exactly with the committed documentation.
    """
    reference_dir = repo_root / "results" / "reference"
    config = json.loads((reference_dir / "config.json").read_text())
    metrics = json.loads((reference_dir / "metrics.json").read_text())
    benchmark = json.loads((reference_dir / "benchmark.json").read_text())
    metadata = json.loads((reference_dir / "metadata.json").read_text())

    scenario = config["scenario"]
    components = scenario["components"]
    controllers = metrics["controllers"]
    workloads = benchmark["workloads"]

    return {
        "reference-benchmark-v1": _benchmark_body(benchmark, workloads),
        "reference-controller-comparison-v1": _comparison_body(
            scenario,
            controllers[LOS_COMPONENT_ID],
            controllers[NMPC_COMPONENT_ID],
            controllers[DISTURBANCE_AWARE_NMPC_COMPONENT_ID],
        ),
        "reference-estimator-v1": _estimator_body(scenario, metrics["estimator"]),
        "reference-provenance-v1": _provenance_body(
            config, metadata, scenario, components
        ),
    }


def _benchmark_body(benchmark: dict, workloads: dict) -> str:
    """The machine-dependent benchmark table (README/validation)."""
    kernel = workloads["kernel"]
    simulation = workloads["simulation"]
    nominal = workloads["nmpc_nominal"]
    aware = workloads["nmpc_disturbance_aware"]
    budget_ms = 1000.0 * nominal["control_period_s"]
    sample_count = nominal["samples"] + aware["samples"]
    failed_count = nominal["failed_solves"] + aware["failed_solves"]
    return (
        "\n"
        "| Metric | Result |\n"
        "|---|---:|\n"
        f"| C++ RK4 propagation (vessel + actuator) | "
        f"**{kernel['ns_per_step']:.1f} ns/step** |\n"
        f"| 1000 s simulation (Python loop) | "
        f"**{simulation['wall_time_ms']:.0f} ms** |\n"
        f"| Nominal NMPC mean / p95 / max [ms] | "
        f"**{nominal['mean_ms']:.1f} / {nominal['p95_ms']:.1f} / "
        f"{nominal['max_ms']:.1f}** |\n"
        f"| Disturbance-aware NMPC mean / p95 / max [ms] | "
        f"**{aware['mean_ms']:.1f} / {aware['p95_ms']:.1f} / "
        f"{aware['max_ms']:.1f}** |\n"
        "\n"
        f"Machine-dependent wall-clock measurements recorded in "
        f"`results/reference/benchmark.json` (`{benchmark['benchmark_id']}`, "
        f"{sample_count} samples, {failed_count} failed solves). "
        f"The 5 Hz NMPC control period corresponds to a {budget_ms:.0f} ms "
        f"budget; these solve times make no real-time capability claim. "
        f"Regenerate with `python tools/generate_reference_results.py`.\n"
        "\n"
    )


def _comparison_body(
    scenario: dict,
    los: dict,
    nominal: dict,
    disturbance_aware: dict,
) -> str:
    """The deterministic LOS/nominal/aware controller table."""
    lines = [
        "| Metric | LOS (PID/PI) | Nominal NMPC | Aware NMPC |",
        "|---|---:|---:|---:|",
    ]
    for label, key, spec, degrees in _COMPARISON_ROWS:
        lines.append(
            f"| {label} | {_fmt(los[key], spec, degrees)} | "
            f"{_fmt(nominal[key], spec, degrees)} | "
            f"{_fmt(disturbance_aware[key], spec, degrees)} |"
        )
    return (
        "\n"
        + "\n".join(lines)
        + "\n\n"
        + "Deterministic flagship metrics formatted from "
        "`results/reference/metrics.json` "
        f"(scenario `{SCENARIO_ID}`, revision {scenario['revision']}, "
        f"seed {scenario['seed']}, {scenario['duration_s']:.1f} s at "
        f"{scenario['integration_dt_s']:.2f} s integration). Saturation counts "
        "left-closed intervals whose applied value lies within 1% of a "
        "`ModelParams` bound span (docs/validation.md). No wall-clock timing "
        "appears here: NMPC solve times are machine-dependent and reported "
        "separately in the benchmark table.\n"
        "\n"
    )


def _estimator_body(scenario: dict, estimator: dict) -> str:
    """The deterministic estimator-metrics table (docs/estimation.md)."""
    transient = estimator["current_error_transient_s"]
    return (
        "\n"
        "| Metric | Value |\n"
        "|---|---:|\n"
        f"| Position error RMS [m] | {estimator['position_error_rms_m']:.2f} |\n"
        f"| Position error max [m] | {estimator['position_error_max_m']:.2f} |\n"
        f"| Yaw-rate error RMS [rad/s] | "
        f"{estimator['yaw_rate_error_rms_rad_s']:.3f} |\n"
        f"| Equivalent-current difference RMS [m/s] "
        f"(after {transient:.1f} s transient) | "
        f"{estimator['current_error_rms_m_s']:.3f} |\n"
        f"| Equivalent-current difference max [m/s] "
        f"(after {transient:.1f} s transient) | "
        f"{estimator['current_error_max_m_s']:.3f} |\n"
        "\n"
        "Estimator errors of the NMPC reference run, computed from the "
        "callback-aligned true/estimated records and formatted from "
        f"`results/reference/metrics.json` (scenario `{SCENARIO_ID}`, "
        f"seed {scenario['seed']}). In this combined-uncertainty run the "
        "augmented state is an equivalent-current proxy: wind gusts and model "
        "mismatch can shift it away from the physical current. The difference "
        "reported here quantifies that confounding (docs/estimation.md §5); "
        "the isolated current-only validation is reported separately.\n"
        "\n"
    )


def _provenance_body(
    config: dict, metadata: dict, scenario: dict, components: dict
) -> str:
    """The scenario/provenance table (README/validation)."""
    fingerprint = metadata["source_fingerprint"]
    return (
        "\n"
        "| Item | Value |\n"
        "|---|---|\n"
        f"| Scenario | `{scenario['id']}` (revision {scenario['revision']}) |\n"
        f"| Seed | {scenario['seed']} |\n"
        f"| Duration / integration step | {scenario['duration_s']:.1f} s / "
        f"{scenario['integration_dt_s']:.2f} s |\n"
        f"| Controllers | `{components['controller_los']}` · "
        f"`{components['controller_nmpc']}` · "
        f"`{components['controller_disturbance_aware_nmpc']}` |\n"
        f"| Estimator | `{components['estimator']}` |\n"
        "| Schema | `results/reference/reference.schema.json` "
        f"(version {config['schema_version']}) |\n"
        "| Deterministic metrics | `results/reference/metrics.json` |\n"
        "| Machine-dependent benchmark | `results/reference/benchmark.json` |\n"
        f"| Generated at (UTC) | {metadata['generated_at_utc']} |\n"
        f"| Source commit | `{config['git_commit']}` |\n"
        f"| Source fingerprint | dirty: {str(fingerprint['dirty']).lower()} · "
        f"`{fingerprint['sha256']}` |\n"
        "\n"
        "`git_commit` and the `dirty` flag record the repository state at "
        "generation time; the source fingerprint is content-based and "
        "authoritative. After committing source changes, either regenerate "
        "the artifacts (`python tools/generate_reference_results.py`) or "
        "keep the source contents unchanged: `--check` compares only the "
        "content fingerprint, so a clean checkout at a new commit passes "
        "when the source contents are unchanged and fails when they "
        "changed. `--check` validates schema, scenario, source fingerprint, "
        "artifact hashes and marker bodies without any simulation; "
        "`--verify-determinism` runs one fresh 120 s reference and compares "
        "it with `results/reference/metrics.json`: the LOS baseline metrics "
        "exactly, and both NMPC variants plus estimator metrics within "
        "`rtol=1e-6, atol=1e-6` (IPOPT solves to `tol=1e-4`, so its "
        "full-precision iterates may differ in the last ulps), reporting "
        "the worst offending key and deviation on failure. Reproducibility "
        "is guaranteed within the software environment recorded in "
        "`metadata.json` (`software` block): regenerating in another "
        "environment requires a fresh `--verify-determinism` in that "
        "environment before the committed metrics can be trusted.\n"
        "\n"
    )


def _fmt(value: object, spec: str, degrees: bool = False) -> str:
    """Format a JSON number, optionally converted to degrees, for a cell."""
    number = float(value)
    if degrees:
        number = math.degrees(number)
    return f"{number:{spec}}"


def _marker_body_problems(repo_root: Path) -> list[str]:
    """Marker structure and body-equality problems of the documentation files.

    Body equality is only meaningful once the reference artifacts exist;
    missing artifacts are already reported by the main consistency check.
    """
    reference_dir = repo_root / "results" / "reference"
    if any(not (reference_dir / name).is_file() for name in ARTIFACT_FILENAMES):
        return []
    try:
        bodies = _marker_bodies(repo_root)
    except (json.JSONDecodeError, KeyError, OSError, ValueError) as exc:
        return [f"cannot regenerate marker bodies from the reference JSON: {exc}"]
    problems: list[str] = []
    for relpath in _marker_relpaths():
        path = repo_root / relpath
        if not path.is_file():
            continue
        text = path.read_text()
        try:
            pairs = _marker_pairs(text, relpath)
            _validate_marker_placement(relpath, pairs)
        except ValueError as exc:
            problems.append(str(exc))
            continue
        for marker_id, body_start, body_end in pairs:
            if text[body_start:body_end] != bodies[marker_id]:
                problems.append(
                    f"{relpath}: marker '{marker_id}' body does not match the "
                    "committed reference JSON (regenerate with "
                    "python tools/generate_reference_results.py)"
                )
    return problems


# --- document assembly --------------------------------------------------------


def _config_document(
    repo_root: Path,
    config: ReferenceScenarioConfig,
    path: np.ndarray,
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
    config: ReferenceScenarioConfig, path: np.ndarray
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
            "path": "s_curve_v1",
            "environment": "rotating_current_gusts_v1",
            "controller_los": LOS_COMPONENT_ID,
            "controller_nmpc": NMPC_COMPONENT_ID,
            "controller_disturbance_aware_nmpc": (DISTURBANCE_AWARE_NMPC_COMPONENT_ID),
            "estimator": "augmented_current_ekf_v1",
        },
        "description": _SCENARIO_DESCRIPTION,
        "revision": 1,
        "seed": config.seed,
        "duration_s": config.duration_s,
        "integration_dt_s": config.integration_dt_s,
        "control_periods": {
            "los_s": config.los_period_s,
            "nmpc_s": config.nmpc_period_s,
        },
        "path": {
            "name": "s_curve",
            "speed_ref_m_s": config.speed_ref_m_s,
            "lookahead_m": config.lookahead_m,
            "waypoints": [[float(x), float(y)] for x, y in path],
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


def _benchmark_document(
    benchmark: dict[str, object], repo_root: Path
) -> dict[str, object]:
    """The benchmark.json artifact: machine-dependent timing only."""
    if "benchmark_id" not in benchmark or "workloads" not in benchmark:
        raise ValueError(
            "benchmark must contain 'benchmark_id' and 'workloads' "
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


def _source_fingerprint_matches(
    recorded: dict[str, object], current: dict[str, object]
) -> bool:
    """Content-based fingerprint equality: source list and combined digest.

    ``dirty`` is a generation-time provenance record (as are the
    ``git_commit`` fields of config/benchmark) and is deliberately not
    compared: after the source is committed, a clean checkout reports
    ``dirty: false`` and a new HEAD while the content digest is unchanged.
    The content fingerprint remains authoritative — any scenario/parameter/
    source change alters ``files`` or ``sha256`` and fails the check.
    """
    return (
        recorded.get("files") == current["files"]
        and recorded.get("sha256") == current["sha256"]
    )


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
                violation = _worst_metric_violation(
                    committed[key], fresh[key], f"{prefix}{key}."
                )
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
