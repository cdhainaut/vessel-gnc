"""Generated Markdown contract of the committed reference artifacts.

Owns the ``<!-- generated:<marker-id>:start/end -->`` marker map, the marker
body validation and the table bodies themselves: every public number is
formatted here from the committed ``results/reference/*.json`` documents and
appears only between the marker pairs, never hand-edited. The
reproducibility contract behind this design is documented in
``docs/validation.md``.
"""

from __future__ import annotations

import json
import math
import re
from pathlib import Path

from vessel_gnc.mpcc import MPCC_COMPONENT_ID
from vessel_gnc.reference import (
    DISTURBANCE_AWARE_NMPC_COMPONENT_ID,
    LOS_COMPONENT_ID,
    NMPC_COMPONENT_ID,
)

__all__ = [
    "ARTIFACT_FILENAMES",
    "MARKDOWN_MARKERS",
    "marker_body_problems",
    "update_generated_markdown",
]

# Files that constitute the committed reference set: the first three carry
# content, ``metadata.json`` is written last and hashes the others.
ARTIFACT_FILENAMES = ("config.json", "metrics.json", "benchmark.json", "metadata.json")

MARKDOWN_MARKERS: dict[str, tuple[str, ...]] = {
    "reference-benchmark-v1": ("README.md", "docs/validation.md"),
    "reference-controller-comparison-v1": ("README.md", "docs/control.md"),
    "reference-estimator-v1": ("docs/estimation.md",),
    "reference-provenance-v1": ("README.md", "docs/validation.md"),
}

_MARKER_RE = re.compile(r"<!-- generated:([a-z0-9-]+):(start|end) -->")


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
            problems.append(f"{relpath}: generated marker '{marker_id}' end without start")
    if problems:
        raise ValueError("; ".join(problems))
    return [(marker_id, starts[marker_id][0], ends[marker_id][0]) for marker_id in starts]


def _validate_marker_placement(relpath: str, pairs: list[tuple[str, int, int]]) -> None:
    """The marker set of a file must equal its expected set exactly."""
    expected = {
        marker_id for marker_id, relpaths in MARKDOWN_MARKERS.items() if relpath in relpaths
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
            controllers[MPCC_COMPONENT_ID],
        ),
        "reference-estimator-v1": _estimator_body(scenario, metrics["estimator"]),
        "reference-provenance-v1": _provenance_body(config, metadata, scenario, components),
    }


def _benchmark_body(benchmark: dict, workloads: dict) -> str:
    """The machine-dependent benchmark table (README/validation)."""
    kernel = workloads["kernel"]
    simulation = workloads["simulation"]
    controllers = (
        ("Nominal NMPC", workloads["nmpc_nominal"]),
        ("Disturbance-aware NMPC", workloads["nmpc_disturbance_aware"]),
        ("Disturbance-aware MPCC", workloads["mpcc_disturbance_aware"]),
    )
    sample_count = sum(workload["samples"] for _, workload in controllers)
    failed_count = sum(workload["failed_solves"] for _, workload in controllers)
    budget_ms = controllers[0][1]["control_budget_ms"]
    controller_rows = "".join(
        f"| {label} mean / median / p95 / max [ms] | "
        f"**{workload['mean_ms']:.1f} / {workload['median_ms']:.1f} / "
        f"{workload['p95_ms']:.1f} / {workload['max_ms']:.1f}** |\n"
        for label, workload in controllers
    )
    status_summary = "; ".join(
        f"{label}: {workload['samples']} samples, "
        f"{workload['failed_solves']} failed, "
        + ", ".join(
            f"{status}={count}" for status, count in workload["final_status_histogram"].items()
        )
        for label, workload in controllers
    )
    return (
        "\n"
        "| Metric | Result |\n"
        "|---|---:|\n"
        f"| C++ RK4 propagation (vessel + actuator) | "
        f"**{kernel['ns_per_step']:.1f} ns/step** |\n"
        f"| 1000 s simulation (Python loop) | "
        f"**{simulation['wall_time_ms']:.0f} ms** |\n"
        f"{controller_rows}"
        "\n"
        f"Machine-dependent wall-clock measurements recorded in "
        f"`results/reference/benchmark.json` (`{benchmark['benchmark_id']}`, "
        f"{sample_count} predictive solves, {failed_count} failed). "
        f"Per-workload status histograms: {status_summary}. The 5 Hz control "
        f"period defines a {budget_ms:.0f} ms budget; these solve times make "
        f"no real-time capability claim. Regenerate with "
        f"`python tools/generate_reference_results.py`.\n"
        "\n"
    )


# Deterministic controller rows of the comparison marker, in display order:
# (label, metrics.json key, format spec, convert-to-degrees flag).
_COMPARISON_ROWS = (
    ("RMS cross-track error [m]", "cross_track_rms_m", ".2f", False),
    ("P95 cross-track error [m]", "cross_track_p95_m", ".2f", False),
    ("Max cross-track error [m]", "cross_track_max_m", ".2f", False),
    ("RMS wrapped heading error [deg]", "heading_error_rms_rad", ".1f", True),
    ("Max wrapped heading error [deg]", "heading_error_max_rad", ".1f", True),
    ("Final path progress [m]", "path_progress_final_m", ".1f", False),
    ("Final path progress fraction [-]", "path_progress_fraction", ".3f", False),
    ("Mean progress rate [m/s]", "mean_progress_rate_m_s", ".2f", False),
    ("Route completion [s]", "route_completion_s", ".1f", False),
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


def _comparison_body(
    scenario: dict,
    los: dict,
    nominal: dict,
    disturbance_aware: dict,
    mpcc: dict,
) -> str:
    """The deterministic four-controller comparison table."""
    lines = [
        "| Metric | LOS (PID/PI) | Nominal NMPC | Aware NMPC | Aware MPCC |",
        "|---|---:|---:|---:|---:|",
    ]
    for label, key, spec, degrees in _COMPARISON_ROWS:
        lines.append(
            f"| {label} | {_fmt(los[key], spec, degrees)} | "
            f"{_fmt(nominal[key], spec, degrees)} | "
            f"{_fmt(disturbance_aware[key], spec, degrees)} | "
            f"{_fmt(mpcc[key], spec, degrees)} |"
        )
    return (
        "\n" + "\n".join(lines) + "\n\n" + "Deterministic flagship metrics formatted from "
        "`results/reference/metrics.json` "
        f"(scenario `{scenario['id']}`, revision {scenario['revision']}, "
        f"seed {scenario['seed']}, {scenario['duration_s']:.1f} s at "
        f"{scenario['integration_dt_s']:.2f} s integration). Route completion "
        "is the first sample at 99% of total chord progress; an incomplete "
        "route is shown as —. Saturation counts left-closed intervals whose "
        "applied value lies within 1% of a `ModelParams` bound span "
        "(docs/validation.md). No wall-clock timing appears here: predictive "
        "solve times are machine-dependent and reported separately in the "
        "benchmark table.\n"
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
        f"`results/reference/metrics.json` (scenario `{scenario['id']}`, "
        f"seed {scenario['seed']}). In this combined-uncertainty run the "
        "augmented state is an equivalent-current proxy: wind gusts and model "
        "mismatch can shift it away from the physical current. The difference "
        "reported here quantifies that confounding (docs/estimation.md §5); "
        "the isolated current-only validation is reported separately.\n"
        "\n"
    )


def _provenance_body(config: dict, metadata: dict, scenario: dict, components: dict) -> str:
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
        f"`{components['controller_disturbance_aware_nmpc']}` · "
        f"`{components['controller_disturbance_aware_mpcc']}` |\n"
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
        "Provenance rows are generation-time records; the content-based "
        "source fingerprint is the authoritative consistency check. "
        "The full reproducibility contract and the metric definitions live "
        "in the validation documentation.\n"
        "\n"
    )


def _fmt(value: object, spec: str, degrees: bool = False) -> str:
    """Format a JSON number, optionally converted to degrees, for a cell."""
    if value is None:
        return "—"
    number = float(value)
    if degrees:
        number = math.degrees(number)
    return f"{number:{spec}}"


def marker_body_problems(repo_root: Path) -> list[str]:
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
