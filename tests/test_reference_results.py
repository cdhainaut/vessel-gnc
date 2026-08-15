"""Cheap consistency tests for the committed reference artifacts (plan M1-G).

These tests validate the *committed* artifacts under
``results/reference/`` against the current repository state: the schema
itself, the four schema-valid JSON documents, the canonical scenario
(120.0 s / 0.01 s / seed 42), finite deterministic metrics with no
timing-like keys, the required benchmark statistics (without asserting
machine-dependent timing values), the current content-based source
fingerprint, the recorded artifact/asset hashes and the exact generated
Markdown marker bodies. A monkeypatch guard proves that
``check_reference_consistency`` invokes neither the expensive flagship
runner nor the benchmark workload, and a static scan proves that normal
pytest cannot launch the 120 s flagship generation (determinism stays the
explicit ``--verify-determinism`` command).

Provenance is treated honestly: ``git_commit`` and the ``dirty`` flag
record the generation-time repository state and are *not* asserted against
the current checkout; only the content-based fingerprint (source paths +
combined SHA-256) must match the current tree.

No 120 s simulation and no benchmark workload is executed here; the only
files read are the committed artifacts, the source tree and the
documentation files.
"""

from __future__ import annotations

import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path

from jsonschema import Draft202012Validator
from vessel_gnc.reference import default_reference_config
from vessel_gnc.reference_artifacts import (
    MARKDOWN_MARKERS,
    SCENARIO_ID,
    SCHEMA_VERSION,
    _artifact_hashes,
    _marker_bodies,
    _marker_pairs,
    _marker_relpaths,
    _validate_marker_placement,
    check_reference_consistency,
    source_fingerprint,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = REPO_ROOT / "results" / "reference"
SCHEMA_PATH = REFERENCE_DIR / "reference.schema.json"

ARTIFACT_NAMES = ("config.json", "metrics.json", "benchmark.json", "metadata.json")
EXPECTED_ARTIFACT_TYPES = {
    "config.json": "config",
    "metrics.json": "metrics",
    "benchmark.json": "benchmark",
    "metadata.json": "metadata",
}

# Deterministic metric key contract (mirrors reference.schema.json
# #/$defs/controllerMetrics and #/$defs/estimatorMetrics).
CONTROLLER_METRIC_KEYS = (
    "cross_track_rms_m",
    "cross_track_p95_m",
    "cross_track_max_m",
    "heading_error_rms_rad",
    "heading_error_max_rad",
    "thrust_rms_N",
    "thrust_max_N",
    "moment_rms_Nm",
    "moment_max_Nm",
    "thrust_saturation_duration_s",
    "moment_saturation_duration_s",
    "any_saturation_duration_s",
)
ESTIMATOR_METRIC_KEYS = (
    "position_error_rms_m",
    "position_error_max_m",
    "yaw_rate_error_rms_rad_s",
    "current_error_rms_m_s",
    "current_error_max_m_s",
    "current_error_transient_s",
)

# Tokens that would betray a wall-clock timing value in the deterministic
# metrics artifact (independent of reference_artifacts._timing_keys).
_TIMING_TOKENS = ("solve", "wall", "elapsed", "_ms", "time_")

# IPOPT final statuses treated as accepted solves (python/vessel_gnc/nmpc.py).
_ACCEPTED_STATUSES = ("Solve_Succeeded", "Solved_To_Acceptable_Level")

# The three non-metadata JSON artifacts and the three committed generated
# assets whose hashes must be recorded in metadata.json["artifacts"].
HASHED_RELPATHS = (
    "results/reference/config.json",
    "results/reference/metrics.json",
    "results/reference/benchmark.json",
    "assets/hero.gif",
    "assets/controller_comparison.png",
    "assets/current_estimation.png",
)


def _load(name: str) -> dict[str, object]:
    """Load one committed artifact document."""
    return json.loads((REFERENCE_DIR / name).read_text())


def _numbers_finite(value: object) -> bool:
    """Whether every JSON number in a document is finite."""
    if isinstance(value, dict):
        return all(_numbers_finite(item) for item in value.values())
    if isinstance(value, list):
        return all(_numbers_finite(item) for item in value)
    if isinstance(value, bool):
        return True
    if isinstance(value, (int, float)):
        return math.isfinite(value)
    return True


def _timing_like_keys(document: object, prefix: str = "") -> list[str]:
    """Dotted keys whose name looks like wall-clock timing (independent check)."""

    def walk(value: object, path: str) -> list[str]:
        hits: list[str] = []
        if isinstance(value, dict):
            for key, item in value.items():
                full = f"{path}.{key}" if path else key
                if any(token in key for token in _TIMING_TOKENS):
                    hits.append(full)
                hits.extend(walk(item, full))
        elif isinstance(value, list):
            for index, item in enumerate(value):
                hits.extend(walk(item, f"{path}[{index}]"))
        return hits

    return walk(document, "")


# --- schema -------------------------------------------------------------------


def test_reference_schema_is_valid_draft_2020_12():
    schema = json.loads(SCHEMA_PATH.read_text())
    Draft202012Validator.check_schema(schema)  # raises on an invalid schema
    assert schema["$schema"].endswith("draft/2020-12/schema")


def test_committed_artifacts_validate_against_schema():
    schema = json.loads(SCHEMA_PATH.read_text())
    validator = Draft202012Validator(schema)
    for name in ARTIFACT_NAMES:
        errors = list(validator.iter_errors(_load(name)))
        assert errors == [], f"{name} violates the schema: {errors}"


def test_committed_artifacts_are_canonical_json():
    # Canonical contract: sorted keys, two-space indentation, trailing LF.
    for name in ARTIFACT_NAMES:
        text = (REFERENCE_DIR / name).read_text()
        document = json.loads(text)
        assert text.endswith("\n")
        canonical = json.dumps(document, indent=2, sort_keys=True, allow_nan=False)
        assert canonical + "\n" == text


# --- versions and scenario ------------------------------------------------------


def test_artifact_types_and_schema_versions_match():
    assert SCHEMA_VERSION == 2
    for name in ARTIFACT_NAMES:
        document = _load(name)
        assert document["artifact_type"] == EXPECTED_ARTIFACT_TYPES[name]
        assert document["schema_version"] == SCHEMA_VERSION
        assert document["$schema"] == "results/reference/reference.schema.json"


def test_config_documents_canonical_scenario():
    scenario = _load("config.json")["scenario"]
    assert scenario["id"] == SCENARIO_ID
    assert scenario["revision"] == 1
    assert scenario["seed"] == 42
    assert scenario["duration_s"] == 120.0
    assert scenario["integration_dt_s"] == 0.01
    assert scenario["control_periods"] == {"los_s": 0.1, "nmpc_s": 0.2}
    assert scenario["path"]["speed_ref_m_s"] == 1.3
    assert scenario["path"]["lookahead_m"] == 8.0
    assert scenario["components"] == {
        "path": "s_curve_v1",
        "environment": "rotating_current_gusts_v1",
        "controller_los": "los_pid_v1",
        "controller_nmpc": "nominal_nmpc_v1",
        "controller_disturbance_aware_nmpc": "disturbance_aware_nmpc_v1",
        "estimator": "augmented_current_ekf_v1",
    }
    # The committed scenario must match the current code defaults exactly.
    current = default_reference_config()
    assert scenario["seed"] == current.seed
    assert scenario["duration_s"] == current.duration_s
    assert scenario["integration_dt_s"] == current.integration_dt_s
    assert scenario["control_periods"] == {
        "los_s": current.los_period_s,
        "nmpc_s": current.nmpc_period_s,
    }


# --- deterministic metrics --------------------------------------------------------


def test_metrics_are_finite_deterministic_and_timing_free():
    metrics = _load("metrics.json")
    assert metrics["scenario_id"] == SCENARIO_ID
    assert _numbers_finite(metrics), "metrics.json must contain only finite numbers"
    timing_keys = _timing_like_keys(metrics)
    assert timing_keys == [], f"metrics.json contains timing-like keys: {timing_keys}"
    assert set(metrics["controllers"]) == {
        "los_pid_v1",
        "nominal_nmpc_v1",
        "disturbance_aware_nmpc_v1",
    }
    for controller_metrics in metrics["controllers"].values():
        assert set(controller_metrics) == set(CONTROLLER_METRIC_KEYS)
        for value in controller_metrics.values():
            assert value >= 0.0  # RMS/P95/max/saturation durations are non-negative
    assert set(metrics["estimator"]) == set(ESTIMATOR_METRIC_KEYS)
    assert metrics["estimator"]["current_error_transient_s"] == 20.0


# --- machine-dependent benchmark ----------------------------------------------------


def test_benchmark_has_required_statistics_without_timing_assertions():
    # Presence, structure and invariants only: never assert the timing values
    # themselves (they are machine-dependent and must not be part of
    # deterministic checks).
    benchmark = _load("benchmark.json")
    workloads = benchmark["workloads"]
    assert set(workloads) == {
        "kernel",
        "simulation",
        "nmpc_nominal",
        "nmpc_disturbance_aware",
    }

    kernel = workloads["kernel"]
    assert kernel["name"] == "cpp_rk4_propagation"
    assert isinstance(kernel["steps"], int) and kernel["steps"] > 0
    assert kernel["ns_per_step"] > 0.0

    simulation = workloads["simulation"]
    assert simulation["name"] == "python_orchestration_1000s"
    assert simulation["duration_s"] == 1000.0
    assert simulation["wall_time_ms"] > 0.0

    expected_names = {
        "nmpc_nominal": "nominal_s_curve_nmpc_60s",
        "nmpc_disturbance_aware": "disturbance_aware_s_curve_nmpc_60s",
    }
    for workload_key, expected_name in expected_names.items():
        nmpc = workloads[workload_key]
        assert nmpc["name"] == expected_name
        assert nmpc["duration_s"] == 60.0
        assert nmpc["control_period_s"] == 0.2
        assert isinstance(nmpc["samples"], int) and nmpc["samples"] > 0
        for key in ("mean_ms", "median_ms", "p95_ms", "max_ms"):
            assert nmpc[key] > 0.0, f"{workload_key}.{key} must be positive"
        assert nmpc["median_ms"] <= nmpc["p95_ms"] <= nmpc["max_ms"]
        assert nmpc["mean_ms"] <= nmpc["max_ms"]
        assert isinstance(nmpc["failed_solves"], int)
        assert nmpc["failed_solves"] >= 0
        histogram = nmpc["final_status_histogram"]
        assert isinstance(histogram, dict) and len(histogram) >= 1
        assert all(
            isinstance(count, int) and count >= 0 for count in histogram.values()
        )
        assert sum(histogram.values()) == nmpc["samples"]
        assert nmpc["failed_solves"] == sum(
            count
            for status, count in histogram.items()
            if status not in _ACCEPTED_STATUSES
        )
    # The benchmark artifact carries timing only: no deterministic tracking
    # metric key may appear anywhere in it.
    text = json.dumps(benchmark)
    for key in (*CONTROLLER_METRIC_KEYS, *ESTIMATOR_METRIC_KEYS):
        assert key not in text


# --- provenance ---------------------------------------------------------------------


def test_metadata_source_fingerprint_matches_current_tree():
    metadata = _load("metadata.json")
    fingerprint = metadata["source_fingerprint"]
    current = source_fingerprint(REPO_ROOT)
    # The content fingerprint is authoritative: the source list and the
    # combined digest must match the current tree. ``dirty`` is a
    # generation-time provenance record and is deliberately not asserted
    # against the current tree state (after the source commit, a clean
    # checkout reports dirty: false while the content digest is unchanged).
    assert fingerprint["files"] == current["files"]
    assert fingerprint["sha256"] == current["sha256"]
    assert isinstance(fingerprint["dirty"], bool)
    assert re.fullmatch(r"[0-9a-f]{64}", fingerprint["sha256"])
    assert "python/vessel_gnc/reference.py" in fingerprint["files"]
    assert "tools/generate_reference_results.py" in fingerprint["files"]


def test_metadata_artifact_hashes_match_committed_files():
    metadata = _load("metadata.json")
    recorded = metadata["artifacts"]
    assert set(recorded) == set(HASHED_RELPATHS)
    assert recorded == _artifact_hashes(REPO_ROOT)
    for relpath, digest in recorded.items():
        assert re.fullmatch(r"[0-9a-f]{64}", digest)
        assert (REPO_ROOT / relpath).is_file()
    # metadata.json is excluded from its own hash to avoid a cycle.
    assert "results/reference/metadata.json" not in recorded


def test_metadata_provenance_fields_are_sane():
    metadata = _load("metadata.json")
    assert datetime.fromisoformat(metadata["generated_at_utc"]).tzinfo is not None
    assert set(metadata["software"]) == {
        "python",
        "numpy",
        "matplotlib",
        "pillow",
        "casadi",
        "vessel_gnc",
    }
    assert metadata["platform"]["os"]
    assert metadata["platform"]["machine"]


# --- generated Markdown markers -----------------------------------------------------


def test_marker_bodies_match_committed_json_exactly():
    bodies = _marker_bodies(REPO_ROOT)
    assert set(bodies) == set(MARKDOWN_MARKERS)
    for relpath in _marker_relpaths():
        text = (REPO_ROOT / relpath).read_text()
        pairs = _marker_pairs(text, relpath)
        _validate_marker_placement(relpath, pairs)
        for marker_id, body_start, body_end in pairs:
            assert text[body_start:body_end] == bodies[marker_id], (
                f"{relpath}: marker '{marker_id}' body differs from the "
                "committed reference JSON (regenerate with "
                "python tools/generate_reference_results.py)"
            )


# --- guards: no simulation, no benchmark in normal pytest ----------------------------


def test_check_consistency_invokes_neither_runner_nor_benchmark(monkeypatch):
    import vessel_gnc.reference_artifacts as artifacts

    def forbid(*_args, **_kwargs):
        raise AssertionError(
            "check_reference_consistency must not invoke the expensive runner"
        )

    monkeypatch.setattr(artifacts, "run_reference_scenario", forbid)
    monkeypatch.setattr(artifacts, "reference_metrics", forbid)
    # Any import of the benchmark module during the check would raise here.
    monkeypatch.setitem(sys.modules, "benchmarks", None)
    monkeypatch.setitem(sys.modules, "benchmarks.benchmark_simulation", None)
    assert check_reference_consistency(REPO_ROOT) == []


def test_normal_pytest_cannot_launch_the_flagship():
    # Static guard: the test suite must never call the flagship runner
    # without an explicit short configuration, and never invoke the
    # determinism command that performs a fresh 120 s run. The flagship
    # stays behind the explicit --verify-determinism tool command.
    # The labels intentionally avoid the exact call text so the scan never
    # matches its own pattern descriptions.
    forbidden_calls = (
        (
            re.compile(r"run_reference_scenario\s*\(\s*\)"),
            "the runner with no arguments",
        ),
        (
            re.compile(
                r"run_reference_scenario\s*\(\s*default_reference_config\s*\(\s*\)\s*\)"
            ),
            "the runner with the default configuration",
        ),
        (
            re.compile(r"verify_reference_determinism\s*\(\s*REPO_ROOT\s*\)"),
            "determinism verification on the repository root",
        ),
    )
    for path in sorted((REPO_ROOT / "tests").glob("test_*.py")):
        text = path.read_text()
        for pattern, label in forbidden_calls:
            match = pattern.search(text)
            assert match is None, (
                f"{path.name}:{text.count(chr(10), 0, match.start()) + 1} must not "
                f"call {label} (normal pytest must never launch the 120 s flagship)"
            )
