"""Generate, check and verify the committed reference artifacts.

Default mode (no flags) runs the flagship reference scenario exactly once and
the separate benchmark workload once, then renders the flagship assets,
writes the four schema-valid JSON artifacts (``metadata.json`` last so its
hashes cover the final files and assets) and updates the generated Markdown
markers. ``--check`` validates schema, config, scenario, source fingerprint
and artifact hashes without any simulation or benchmark;
``--verify-determinism`` performs one fresh reference run and compares its
deterministic metrics with the committed ``metrics.json`` under the
reproducibility contract: LOS baseline metrics exactly, NMPC and estimator
metrics within ``rtol=1e-6, atol=1e-6`` (IPOPT solves to ``tol=1e-4`` and
its full-precision iterates may differ in the last ulps), reporting the
worst offending key and deviation on failure.

Provenance semantics: ``config.json``/``benchmark.json`` record the
``git_commit`` and ``metadata.json`` records the ``dirty`` flag of the
repository *at generation time*. These honest provenance records are not
compared against the current checkout (after committing the source, a clean
checkout reports a new HEAD and ``dirty: false``). The authoritative
consistency check is the *content* source fingerprint — ordered source paths
plus combined SHA-256 — which changes if and only if a source file appears,
disappears or changes content. Workflow: commit the source first, then
regenerate the artifacts; or keep the source contents unchanged. ``--check``
then passes in any clean checkout whose source tree matches the fingerprint,
and fails whenever a scenario/parameter/source change is not reflected in
the committed artifacts.

Exact determinism is guaranteed only within the software environment
recorded in ``results/reference/metadata.json`` (``software`` block): the
committed artifacts must be generated with a supported Python (``>=3.12``,
pyproject.toml) and a fresh ``--verify-determinism`` in that same
environment. Regenerating in any other environment invalidates the exact
comparison until a new determinism run confirms it.

Run from the repository root:

    python tools/generate_reference_results.py
    python tools/generate_reference_results.py --check
    python tools/generate_reference_results.py --verify-determinism
"""

from __future__ import annotations

import argparse
from pathlib import Path

from vessel_gnc.reference import run_reference_scenario
from vessel_gnc.reference_artifacts import (
    check_reference_consistency,
    render_reference_assets,
    update_generated_markdown,
    verify_reference_determinism,
    write_reference_json,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
REFERENCE_DIR = REPO_ROOT / "results" / "reference"


def main(argv: list[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    parser = argparse.ArgumentParser(
        description=(
            "Generate, check or verify the committed reference artifacts. "
            "Default mode runs the flagship reference scenario once and the "
            "separate benchmark workload once."
        )
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--check",
        action="store_true",
        help=(
            "validate schema, config scenario, content-based source "
            "fingerprint, artifact hashes and marker bodies without any "
            "simulation or benchmark (git_commit and dirty provenance are "
            "recorded, not asserted against the current checkout)"
        ),
    )
    mode.add_argument(
        "--verify-determinism",
        action="store_true",
        help=(
            "run one fresh reference and compare its deterministic metrics "
            "with the committed metrics.json (LOS exact, NMPC/estimator "
            "within rtol=1e-6, atol=1e-6)"
        ),
    )
    args = parser.parse_args(argv)

    if args.check:
        problems = check_reference_consistency(REPO_ROOT)
        for problem in problems:
            print(f"check failed: {problem}")
        if problems:
            return 1
        print(
            "check passed: schema, config, scenario, source fingerprint and "
            "artifact hashes are consistent"
        )
        return 0

    if args.verify_determinism:
        try:
            verify_reference_determinism(REPO_ROOT)
        except (AssertionError, FileNotFoundError) as exc:
            print(f"verify-determinism failed: {exc}")
            return 1
        print(
            "determinism verified: fresh reference metrics match "
            "results/reference/metrics.json within the reproducibility "
            "contract (LOS exact, NMPC/estimator rtol=1e-6, atol=1e-6)"
        )
        return 0

    _generate_default()
    return 0


def _generate_default() -> None:
    """One shared reference run, one benchmark run, assets, JSON, markers.

    The assets are regenerated from the same in-memory run before the JSON is
    written, so ``metadata.json`` records the hashes of the final artifacts.
    """
    print("running the flagship reference scenario (120 s) ...")
    run = run_reference_scenario()
    print("running the separate benchmark workload (60 s NMPC) ...")
    benchmark = _run_benchmarks()
    rendered = render_reference_assets(run, REPO_ROOT)
    write_reference_json(run, benchmark, REFERENCE_DIR)
    update_generated_markdown(REPO_ROOT)
    for path in rendered:
        print(f"wrote {path}")
    print(
        "wrote results/reference/{config,metrics,benchmark,metadata}.json "
        "(metadata hashes last)"
    )


def _run_benchmarks() -> dict[str, object]:
    """The structured benchmark record from benchmarks/benchmark_simulation.py.

    Imported lazily so ``--check`` never loads the benchmark module (and thus
    never triggers CasADi solver construction).
    """
    import sys

    if str(REPO_ROOT) not in sys.path:
        sys.path.insert(0, str(REPO_ROOT))
    from benchmarks.benchmark_simulation import run_benchmarks

    return run_benchmarks()


if __name__ == "__main__":
    raise SystemExit(main())
