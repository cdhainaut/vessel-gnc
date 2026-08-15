"""Synthetic, temp-directory tests for the reference artifact pipeline.

These tests never run the flagship scenario or the benchmark workload: the
reference run used here is a hand-authored ``ReferenceRun`` with the default
configuration (so the serialized scenario matches what the current code
would generate), and ``run_reference_scenario`` / the benchmark module are
monkeypatched or never imported. Schema validation uses the committed
``results/reference/reference.schema.json``.
"""

from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

import jsonschema
import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:  # noqa: E402  (repo-local tool import)
    sys.path.insert(0, str(REPO_ROOT))

from vessel_gnc.guidance import make_s_curve_path  # noqa: E402
from vessel_gnc.reference import (  # noqa: E402
    ControllerReferenceRun,
    EstimatorHistory,
    ReferenceRun,
    default_reference_config,
)
from vessel_gnc.reference_artifacts import (  # noqa: E402
    MARKDOWN_MARKERS,
    SCENARIO_ID,
    SCHEMA_VERSION,
    _scenario_document,
    check_reference_consistency,
    render_reference_assets,
    source_fingerprint,
    update_generated_markdown,
    verify_reference_determinism,
    write_reference_json,
)
from vessel_gnc.simulation import SimulationResult  # noqa: E402

import tools.generate_reference_results as tool  # noqa: E402

SCHEMA_PATH = REPO_ROOT / "results" / "reference" / "reference.schema.json"

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


def _synthetic_run() -> ReferenceRun:
    """A hand-authored default-configuration run (no simulation at all).

    The estimator records include a post-transient sample (t >= 20 s) so
    ``reference_metrics`` has data for the current-error statistics.
    """
    config = default_reference_config()
    path = make_s_curve_path()
    n = 3
    t = np.array([0.0, 10.0, 21.0])

    def result() -> SimulationResult:
        return SimulationResult(
            t=t,
            x=np.zeros(n),
            y=np.zeros(n),
            psi=np.zeros(n),
            u=np.zeros(n),
            v=np.zeros(n),
            r=np.zeros(n),
            thrust=np.zeros(n),
            yaw_moment=np.zeros(n),
        )

    def estimator() -> EstimatorHistory:
        return EstimatorHistory(
            t=t,
            state_true=np.arange(n * 6, dtype=float).reshape(n, 6),
            state_estimate=np.arange(n * 6, dtype=float).reshape(n, 6),
            current_true=np.zeros((n, 2)),
            current_estimate=np.zeros((n, 2)),
        )

    def controller(label: str, period: float) -> ControllerReferenceRun:
        return ControllerReferenceRun(
            label=label,
            period_s=period,
            result=result(),
            estimator=estimator(),
            command=np.zeros((n, 2)),
            solve_time_s=np.zeros(n),
            solve_status=(),
            horizon=(),
        )

    return ReferenceRun(
        config=config,
        path=path,
        los=controller("LOS baseline", config.los_period_s),
        nmpc=controller("Nominal NMPC", config.nmpc_period_s),
        disturbance_aware_nmpc=controller(
            "Disturbance-aware NMPC",
            config.nmpc_period_s,
        ),
    )


def _synthetic_benchmark() -> dict[str, object]:
    """A schema-shaped benchmark record with synthetic machine numbers."""
    return {
        "benchmark_id": "benchmark_v2",
        "workloads": {
            "kernel": {
                "name": "cpp_rk4_propagation",
                "ns_per_step": 123.0,
                "steps": 1000,
            },
            "simulation": {
                "name": "python_orchestration_1000s",
                "duration_s": 1000.0,
                "wall_time_ms": 42.0,
            },
            "nmpc_nominal": {
                "name": "nominal_s_curve_nmpc_60s",
                "duration_s": 60.0,
                "control_period_s": 0.2,
                "samples": 10,
                "mean_ms": 1.0,
                "median_ms": 0.9,
                "p95_ms": 1.5,
                "max_ms": 2.0,
                "failed_solves": 0,
                "final_status_histogram": {"Solve_Succeeded": 10},
            },
            "nmpc_disturbance_aware": {
                "name": "disturbance_aware_s_curve_nmpc_60s",
                "duration_s": 60.0,
                "control_period_s": 0.2,
                "samples": 10,
                "mean_ms": 1.1,
                "median_ms": 1.0,
                "p95_ms": 1.6,
                "max_ms": 2.1,
                "failed_solves": 0,
                "final_status_histogram": {"Solve_Succeeded": 10},
            },
        },
    }


def _write(reference_dir: Path) -> None:
    """Write the four artifacts and the documentation markers.

    Also copies the committed schema beside them, mirrors the repository
    layout (results/reference holds the schema plus the four artifacts), and
    creates the README/docs marker files with placeholder bodies before
    ``update_generated_markdown`` fills them from the synthetic JSON — the
    same order the default generation tool uses.
    """
    reference_dir.mkdir(parents=True, exist_ok=True)
    (reference_dir / "reference.schema.json").write_text(SCHEMA_PATH.read_text())
    write_reference_json(_synthetic_run(), _synthetic_benchmark(), reference_dir)
    _write_markdown(reference_dir.parents[1])
    update_generated_markdown(reference_dir.parents[1])


def _write_markdown(repo_root: Path) -> None:
    """Placeholder marker files for every documentation file in the mapping."""
    for relpath in sorted(
        {file for files in MARKDOWN_MARKERS.values() for file in files}
    ):
        path = repo_root / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        marker_ids = [
            marker_id
            for marker_id, files in MARKDOWN_MARKERS.items()
            if relpath in files
        ]
        parts = [f"# {relpath}\n"]
        for marker_id in marker_ids:
            parts.append(
                f"<!-- generated:{marker_id}:start -->\n"
                "\n"
                "_placeholder body_\n"
                "\n"
                f"<!-- generated:{marker_id}:end -->\n"
            )
        path.write_text("".join(parts))


def _validator() -> jsonschema.Draft202012Validator:
    schema = json.loads(SCHEMA_PATH.read_text())
    jsonschema.Draft202012Validator.check_schema(schema)
    return jsonschema.Draft202012Validator(schema)


# --- write path ----------------------------------------------------------------


def test_write_writes_canonical_sorted_json(tmp_path):
    reference_dir = tmp_path / "results" / "reference"
    _write(reference_dir)
    assert sorted(p.name for p in reference_dir.iterdir()) == [
        "benchmark.json",
        "config.json",
        "metadata.json",
        "metrics.json",
        "reference.schema.json",
    ]
    for name in ("config.json", "metrics.json", "benchmark.json", "metadata.json"):
        text = (reference_dir / name).read_text()
        assert text.endswith("\n")
        document = json.loads(text)
        # Canonical form: re-dumping with sorted keys reproduces the bytes.
        assert (
            json.dumps(document, indent=2, sort_keys=True, allow_nan=False) + "\n"
            == text
        )


def test_write_artifacts_validate_against_schema(tmp_path):
    reference_dir = tmp_path / "results" / "reference"
    _write(reference_dir)
    validator = _validator()
    for name in ("config.json", "metrics.json", "benchmark.json", "metadata.json"):
        document = json.loads((reference_dir / name).read_text())
        errors = list(validator.iter_errors(document))
        assert errors == [], f"{name} violates the schema: {errors}"


def test_metrics_artifact_has_no_timing_keys_and_exact_key_sets(tmp_path):
    reference_dir = tmp_path / "results" / "reference"
    _write(reference_dir)
    metrics = json.loads((reference_dir / "metrics.json").read_text())
    assert metrics["artifact_type"] == "metrics"
    assert metrics["schema_version"] == SCHEMA_VERSION
    assert metrics["scenario_id"] == SCENARIO_ID
    assert set(metrics["controllers"]) == {
        "los_pid_v1",
        "nominal_nmpc_v1",
        "disturbance_aware_nmpc_v1",
    }
    for controller in metrics["controllers"].values():
        assert set(controller) == set(CONTROLLER_METRIC_KEYS)
    assert set(metrics["estimator"]) == set(ESTIMATOR_METRIC_KEYS)
    for key in metrics:
        assert not any(
            token in key for token in ("solve", "wall", "elapsed", "_ms", "time_")
        )


def test_metadata_hashes_cover_three_json_artifacts_only(tmp_path):
    reference_dir = tmp_path / "results" / "reference"
    _write(reference_dir)
    metadata = json.loads((reference_dir / "metadata.json").read_text())
    expected = {
        f"results/reference/{name}": hashlib.sha256(
            (reference_dir / name).read_bytes()
        ).hexdigest()
        for name in ("config.json", "metrics.json", "benchmark.json")
    }
    assert metadata["artifacts"] == expected
    assert "results/reference/metadata.json" not in metadata["artifacts"]


def test_config_document_matches_default_scenario(tmp_path):
    reference_dir = tmp_path / "results" / "reference"
    _write(reference_dir)
    config = json.loads((reference_dir / "config.json").read_text())
    assert config["artifact_type"] == "config"
    assert config["schema_version"] == SCHEMA_VERSION
    assert config["scenario"]["id"] == SCENARIO_ID
    assert config["scenario"]["revision"] == 1
    assert config["scenario"]["seed"] == 42
    assert config["scenario"]["duration_s"] == 120.0
    assert config["scenario"]["integration_dt_s"] == 0.01
    assert config["scenario"]["control_periods"] == {"los_s": 0.1, "nmpc_s": 0.2}
    expected_scenario = _scenario_document(
        default_reference_config(), make_s_curve_path()
    )
    assert config["scenario"] == expected_scenario


def test_generation_is_stable_sorted_json(tmp_path):
    first = tmp_path / "a" / "results" / "reference"
    second = tmp_path / "b" / "results" / "reference"
    _write(first)
    _write(second)
    for name in ("config.json", "metrics.json", "benchmark.json"):
        assert (first / name).read_bytes() == (second / name).read_bytes()
    first_metadata = json.loads((first / "metadata.json").read_text())
    second_metadata = json.loads((second / "metadata.json").read_text())
    first_metadata.pop("generated_at_utc")
    second_metadata.pop("generated_at_utc")
    assert first_metadata == second_metadata


def test_write_rejects_non_finite_numbers(tmp_path):
    benchmark = _synthetic_benchmark()
    benchmark["workloads"]["kernel"]["ns_per_step"] = float("nan")
    with pytest.raises(ValueError):
        write_reference_json(
            _synthetic_run(), benchmark, tmp_path / "results" / "reference"
        )


# --- consistency check ---------------------------------------------------------


def test_check_consistency_clean(tmp_path):
    _write(tmp_path / "results" / "reference")
    assert check_reference_consistency(tmp_path) == []


def test_check_consistency_reports_missing_schema(tmp_path):
    schema_path = tmp_path / "results" / "reference" / "reference.schema.json"
    problems = check_reference_consistency(tmp_path)
    assert problems == [f"missing reference schema: {schema_path}"]


def test_check_consistency_reports_missing_artifacts(tmp_path):
    (tmp_path / "results" / "reference").mkdir(parents=True)
    (tmp_path / "results" / "reference" / "reference.schema.json").write_text(
        SCHEMA_PATH.read_text()
    )
    problems = check_reference_consistency(tmp_path)
    assert any("missing reference artifact" in problem for problem in problems)
    assert len([p for p in problems if "missing reference artifact" in p]) == 4


def test_check_consistency_detects_metrics_tampering(tmp_path):
    reference_dir = tmp_path / "results" / "reference"
    _write(reference_dir)
    metrics_path = reference_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["controllers"]["los_pid_v1"]["cross_track_rms_m"] += 1.0
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    problems = check_reference_consistency(tmp_path)
    assert any("artifact hashes" in problem for problem in problems)


def test_check_consistency_detects_config_drift(tmp_path):
    reference_dir = tmp_path / "results" / "reference"
    _write(reference_dir)
    config_path = reference_dir / "config.json"
    config = json.loads(config_path.read_text())
    config["scenario"]["duration_s"] = 121.0
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    problems = check_reference_consistency(tmp_path)
    assert any(
        "config.json no longer matches the current code" in problem
        for problem in problems
    )


def test_check_consistency_detects_scenario_id_violation(tmp_path):
    reference_dir = tmp_path / "results" / "reference"
    _write(reference_dir)
    config_path = reference_dir / "config.json"
    config = json.loads(config_path.read_text())
    config["scenario"]["id"] = "scenario_v2_other"
    config_path.write_text(json.dumps(config, indent=2, sort_keys=True) + "\n")
    problems = check_reference_consistency(tmp_path)
    assert any("violates reference schema" in problem for problem in problems)


def test_check_consistency_detects_fingerprint_drift(tmp_path):
    reference_dir = tmp_path / "results" / "reference"
    _write(reference_dir)
    assert check_reference_consistency(tmp_path) == []
    source = tmp_path / "python" / "vessel_gnc" / "extra_source.py"
    source.parent.mkdir(parents=True)
    source.write_text("x = 1\n")
    problems = check_reference_consistency(tmp_path)
    assert any("source_fingerprint" in problem for problem in problems)


def test_check_consistency_survives_commit_transition_when_contents_unchanged(
    tmp_path, monkeypatch
):
    """A simulated HEAD/dirty transition does not fail on unchanged content.

    The artifacts are generated in a "dirty" tree at an old HEAD, then the
    source is committed: the checkout is clean at a new HEAD with identical
    source contents. ``git_commit`` and the ``dirty`` flag are honest
    generation-time provenance records, so the check must still pass while
    the recorded provenance values stay untouched.
    """
    import vessel_gnc.reference_artifacts as artifacts

    monkeypatch.setattr(artifacts, "_git_commit", lambda _repo_root: "a" * 40)
    monkeypatch.setattr(artifacts, "_git_dirty", lambda _repo_root: True)
    reference_dir = tmp_path / "results" / "reference"
    _write(reference_dir)

    monkeypatch.setattr(artifacts, "_git_commit", lambda _repo_root: "b" * 40)
    monkeypatch.setattr(artifacts, "_git_dirty", lambda _repo_root: False)
    assert check_reference_consistency(tmp_path) == []

    # Provenance stays an honest record of the generation-time state.
    metadata = json.loads((reference_dir / "metadata.json").read_text())
    assert metadata["source_fingerprint"]["dirty"] is True
    config = json.loads((reference_dir / "config.json").read_text())
    assert config["git_commit"] == "a" * 40


def test_check_consistency_still_detects_source_change_across_commit_transition(
    tmp_path, monkeypatch
):
    """Content changes still fail the check even when HEAD/dirty moved."""
    import vessel_gnc.reference_artifacts as artifacts

    monkeypatch.setattr(artifacts, "_git_commit", lambda _repo_root: "a" * 40)
    monkeypatch.setattr(artifacts, "_git_dirty", lambda _repo_root: True)
    reference_dir = tmp_path / "results" / "reference"
    _write(reference_dir)

    monkeypatch.setattr(artifacts, "_git_commit", lambda _repo_root: "b" * 40)
    monkeypatch.setattr(artifacts, "_git_dirty", lambda _repo_root: False)
    source = tmp_path / "python" / "vessel_gnc" / "extra_source.py"
    source.parent.mkdir(parents=True)
    source.write_text("x = 1\n")
    problems = check_reference_consistency(tmp_path)
    assert any("source_fingerprint" in problem for problem in problems)


def test_check_consistency_detects_metadata_hash_drift_of_assets(tmp_path):
    reference_dir = tmp_path / "results" / "reference"
    _write(reference_dir)
    assert check_reference_consistency(tmp_path) == []
    # A committed generated asset appears after generation: its hash must be
    # recorded for the check to pass again.
    asset = tmp_path / "assets" / "hero.gif"
    asset.parent.mkdir(parents=True)
    asset.write_bytes(b"GIF89a-fake-hero")
    problems = check_reference_consistency(tmp_path)
    assert any("artifact hashes" in problem for problem in problems)


def test_check_consistency_never_invokes_runner_or_metrics(tmp_path, monkeypatch):
    reference_dir = tmp_path / "results" / "reference"
    _write(reference_dir)
    import vessel_gnc.reference_artifacts as artifacts

    def forbid(*_args, **_kwargs):
        raise AssertionError(
            "reference runner or metrics must not be invoked by --check"
        )

    monkeypatch.setattr(artifacts, "run_reference_scenario", forbid)
    monkeypatch.setattr(artifacts, "reference_metrics", forbid)
    assert check_reference_consistency(tmp_path) == []


# --- determinism verification ---------------------------------------------------


def test_verify_determinism_passes_on_identical_run(tmp_path, monkeypatch):
    reference_dir = tmp_path / "results" / "reference"
    _write(reference_dir)
    import vessel_gnc.reference_artifacts as artifacts

    monkeypatch.setattr(artifacts, "run_reference_scenario", lambda: _synthetic_run())
    verify_reference_determinism(tmp_path)  # must not raise


def test_verify_determinism_raises_on_different_metrics(tmp_path, monkeypatch):
    reference_dir = tmp_path / "results" / "reference"
    _write(reference_dir)
    metrics_path = reference_dir / "metrics.json"
    metrics = json.loads(metrics_path.read_text())
    metrics["estimator"]["position_error_rms_m"] = 0.5
    metrics_path.write_text(json.dumps(metrics, indent=2, sort_keys=True) + "\n")
    import vessel_gnc.reference_artifacts as artifacts

    monkeypatch.setattr(artifacts, "run_reference_scenario", lambda: _synthetic_run())
    with pytest.raises(AssertionError, match="determinism check failed"):
        verify_reference_determinism(tmp_path)


def test_verify_determinism_requires_committed_metrics(tmp_path):
    with pytest.raises(FileNotFoundError, match="missing committed metrics artifact"):
        verify_reference_determinism(tmp_path)


def test_verify_determinism_accepts_1e8_nmpc_and_estimator_deviation(
    tmp_path, monkeypatch
):
    """A synthetic 1e-8 deviation passes: far inside rtol/atol = 1e-6."""
    reference_dir = tmp_path / "results" / "reference"
    _write(reference_dir)
    metrics_path = reference_dir / "metrics.json"
    import vessel_gnc.reference_artifacts as artifacts

    def fresh_metrics(_run):
        # The synthetic run's metrics equal the committed document; nudge two
        # NMPC/estimator keys by 1e-8, well inside the reproducibility
        # tolerance (rtol=1e-6, atol=1e-6).
        fresh = json.loads(metrics_path.read_text())
        fresh["controllers"]["nominal_nmpc_v1"]["cross_track_rms_m"] += 1e-8
        fresh["controllers"]["disturbance_aware_nmpc_v1"]["cross_track_rms_m"] += 1e-8
        fresh["estimator"]["position_error_rms_m"] += 1e-8
        return fresh

    monkeypatch.setattr(artifacts, "run_reference_scenario", lambda: _synthetic_run())
    monkeypatch.setattr(artifacts, "reference_metrics", fresh_metrics)
    verify_reference_determinism(tmp_path)  # must not raise


def test_verify_determinism_rejects_1e3_nmpc_deviation_with_worst_key(
    tmp_path, monkeypatch
):
    """A synthetic 1e-3 deviation fails, reporting the worst key and deviation."""
    reference_dir = tmp_path / "results" / "reference"
    _write(reference_dir)
    metrics_path = reference_dir / "metrics.json"
    import vessel_gnc.reference_artifacts as artifacts

    def fresh_metrics(_run):
        fresh = json.loads(metrics_path.read_text())
        fresh["controllers"]["nominal_nmpc_v1"]["cross_track_rms_m"] += 1e-3
        return fresh

    monkeypatch.setattr(artifacts, "run_reference_scenario", lambda: _synthetic_run())
    monkeypatch.setattr(artifacts, "reference_metrics", fresh_metrics)
    with pytest.raises(AssertionError) as exc_info:
        verify_reference_determinism(tmp_path)
    message = str(exc_info.value)
    assert "determinism check failed" in message
    assert "controllers.nominal_nmpc_v1.cross_track_rms_m" in message
    assert "e-03" in message  # the reported absolute/relative deviation


def test_verify_determinism_keeps_los_metrics_exact(tmp_path, monkeypatch):
    """Even a 1e-9 LOS deviation fails: the LOS baseline is exact-only."""
    reference_dir = tmp_path / "results" / "reference"
    _write(reference_dir)
    metrics_path = reference_dir / "metrics.json"
    import vessel_gnc.reference_artifacts as artifacts

    def fresh_metrics(_run):
        fresh = json.loads(metrics_path.read_text())
        fresh["controllers"]["los_pid_v1"]["cross_track_rms_m"] += 1e-9
        return fresh

    monkeypatch.setattr(artifacts, "run_reference_scenario", lambda: _synthetic_run())
    monkeypatch.setattr(artifacts, "reference_metrics", fresh_metrics)
    with pytest.raises(AssertionError, match="must reproduce bit-for-bit"):
        verify_reference_determinism(tmp_path)


# --- provenance -----------------------------------------------------------------


def test_source_fingerprint_is_deterministic_and_content_sensitive(tmp_path):
    source = tmp_path / "python" / "vessel_gnc" / "fingerprint_target.py"
    source.parent.mkdir(parents=True)
    source.write_text("value = 1\n")
    first = source_fingerprint(tmp_path)
    assert first["files"] == ["python/vessel_gnc/fingerprint_target.py"]
    assert len(first["sha256"]) == 64
    assert first["dirty"] is False  # temp dir is not a git work tree
    assert source_fingerprint(tmp_path) == first  # byte-identical inputs
    source.write_text("value = 2\n")
    second = source_fingerprint(tmp_path)
    assert second["sha256"] != first["sha256"]
    assert second["files"] == first["files"]


def test_render_writes_all_assets_from_tiny_synthetic_run(tmp_path):
    # The tiny hand-authored run must render all four pipeline outputs into
    # the given repo root (no simulation, no committed file touched).
    run = _synthetic_run()
    written = render_reference_assets(run, tmp_path)
    assert [path.relative_to(tmp_path).as_posix() for path in written] == [
        "assets/hero.gif",
        "assets/controller_comparison.png",
        "assets/current_estimation.png",
        "results/reference/nmpc_trajectory.png",
    ]
    for path in written:
        assert path.is_file() and path.stat().st_size > 0
    assert update_generated_markdown(tmp_path) is None


# --- generated Markdown markers ----------------------------------------------


def test_update_generated_markdown_replaces_placeholder_bodies(tmp_path):
    reference_dir = tmp_path / "results" / "reference"
    _write(reference_dir)
    readme = (tmp_path / "README.md").read_text()
    # The placeholder is gone and numbers come from the synthetic JSON only.
    assert "_placeholder body_" not in readme
    assert "123.0 ns/step" in readme
    assert "42 ms" in readme
    assert "1.0 / 1.5 / 2.0" in readme
    # Exactly one start/end pair per marker survives.
    for marker_id in ("reference-benchmark-v1", "reference-provenance-v1"):
        assert readme.count(f"<!-- generated:{marker_id}:start -->") == 1
        assert readme.count(f"<!-- generated:{marker_id}:end -->") == 1
    control = (tmp_path / "docs" / "control.md").read_text()
    assert "P95 cross-track error [m]" in control  # deterministic P95 row present
    estimation = (tmp_path / "docs" / "estimation.md").read_text()
    assert "Position error RMS [m]" in estimation
    validation = (tmp_path / "docs" / "validation.md").read_text()
    assert "Source fingerprint" in validation


def test_check_consistency_detects_edited_number_between_markers(tmp_path):
    reference_dir = tmp_path / "results" / "reference"
    _write(reference_dir)
    assert check_reference_consistency(tmp_path) == []
    readme = tmp_path / "README.md"
    readme.write_text(readme.read_text().replace("123.0 ns/step", "999.0 ns/step"))
    problems = check_reference_consistency(tmp_path)
    assert any(
        "reference-benchmark-v1" in problem and "does not match" in problem
        for problem in problems
    )


def test_check_consistency_detects_duplicated_marker_pair(tmp_path):
    reference_dir = tmp_path / "results" / "reference"
    _write(reference_dir)
    readme = tmp_path / "README.md"
    extra = (
        "<!-- generated:reference-estimator-v1:start -->\n"
        "\n"
        "_extra table_\n"
        "\n"
        "<!-- generated:reference-estimator-v1:end -->\n"
    )
    readme.write_text(readme.read_text() + "\n" + extra)
    problems = check_reference_consistency(tmp_path)
    assert any("unexpected marker" in problem for problem in problems)


def test_update_generated_markdown_raises_on_missing_marker(tmp_path):
    reference_dir = tmp_path / "results" / "reference"
    _write(reference_dir)
    readme = tmp_path / "README.md"
    pair = re.compile(
        r"<!-- generated:reference-provenance-v1:start -->.*?"
        r"<!-- generated:reference-provenance-v1:end -->\n",
        re.DOTALL,
    )
    readme.write_text(pair.sub("", readme.read_text()))
    with pytest.raises(ValueError, match="missing marker"):
        update_generated_markdown(tmp_path)


def test_check_consistency_detects_start_without_end(tmp_path):
    reference_dir = tmp_path / "results" / "reference"
    _write(reference_dir)
    estimation = tmp_path / "docs" / "estimation.md"
    text = estimation.read_text()
    estimation.write_text(
        text.replace("<!-- generated:reference-estimator-v1:end -->", "")
    )
    problems = check_reference_consistency(tmp_path)
    assert any("must appear exactly once" in problem for problem in problems)


# --- CLI ------------------------------------------------------------------------


def test_cli_check_returns_zero_on_consistent_artifacts(tmp_path, monkeypatch, capsys):
    _write(tmp_path / "results" / "reference")
    monkeypatch.setattr(tool, "REPO_ROOT", tmp_path)
    assert tool.main(["--check"]) == 0
    assert "check passed" in capsys.readouterr().out


def test_cli_check_returns_nonzero_on_problems(tmp_path, monkeypatch, capsys):
    monkeypatch.setattr(tool, "REPO_ROOT", tmp_path)
    assert tool.main(["--check"]) == 1
    assert "check failed" in capsys.readouterr().out


def test_cli_verify_determinism_dispatches_without_workloads(tmp_path, monkeypatch):
    monkeypatch.setattr(tool, "REPO_ROOT", tmp_path)
    calls = []

    def fake_verify(repo_root):
        calls.append(repo_root)

    monkeypatch.setattr(tool, "verify_reference_determinism", fake_verify)
    assert tool.main(["--verify-determinism"]) == 0
    assert calls == [tmp_path]


def test_cli_default_mode_wiring_order(tmp_path, monkeypatch):
    monkeypatch.setattr(tool, "REPO_ROOT", tmp_path)
    order: list[str] = []

    def record(name: str):
        def fake(*_args, **_kwargs):
            order.append(name)
            return []

        return fake

    monkeypatch.setattr(tool, "run_reference_scenario", record("run"))
    monkeypatch.setattr(tool, "_run_benchmarks", record("benchmark"))
    monkeypatch.setattr(tool, "render_reference_assets", record("render"))
    monkeypatch.setattr(tool, "write_reference_json", record("write_json"))
    monkeypatch.setattr(tool, "update_generated_markdown", record("update_markdown"))
    assert tool.main([]) == 0
    assert order == ["run", "benchmark", "render", "write_json", "update_markdown"]
