"""Generate, check and verify the committed reference artifacts.

Default mode (no flags) runs the flagship reference scenario exactly once and
the separate benchmark workload once, then renders the flagship assets,
writes the four schema-valid JSON artifacts (``metadata.json`` last so its
hashes cover the final files and assets) and updates the generated Markdown
markers. ``--check`` validates the committed artifacts without any
simulation or benchmark; ``--verify-determinism`` performs one fresh
reference run and compares its deterministic metrics with the committed
``metrics.json``. The reproducibility and provenance contract (fingerprint
workflow, comparison tolerances, generation environment) is documented in
``docs/validation.md``.

Canonical generation runs **single-threaded BLAS**: before any NumPy/CasADi
import this module pins the OpenBLAS, OMP, MKL and NumExpr thread counts to
one via ``os.environ.setdefault``. IPOPT factorizations (MUMPS) and NumPy
reductions are deterministic only when the linear-algebra backends do not
schedule work across threads; multithreaded BLAS can flip the last-ulp
IPOPT iterate path and with it the accepted status of borderline solves,
breaking the deterministic metric contract. The pinning is a tool-level
contract for the canonical artifacts and is intentionally *not* applied
inside the library package; ``setdefault`` keeps an explicit caller
environment intact.

Run from the repository root:

    python tools/generate_reference_results.py
    python tools/generate_reference_results.py --check
    python tools/generate_reference_results.py --verify-determinism
"""

from __future__ import annotations

import os

# Single-threaded BLAS for run-to-run determinism of the canonical
# artifacts (see the reproducibility contract in the module docstring).
# Must run before any NumPy/CasADi import below, which is why this sits
# above the argparse/vessel_gnc imports. ``setdefault`` keeps an explicit
# caller environment intact.
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import argparse
from pathlib import Path

from vessel_gnc.reference import run_reference_scenario
from vessel_gnc.reference_artifacts import (
    check_reference_consistency,
    render_reference_assets,
    verify_reference_determinism,
    write_reference_json,
)
from vessel_gnc.reference_markdown import update_generated_markdown

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
            "with the committed metrics.json (LOS exact, NMPC/MPCC/estimator "
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
            "contract (LOS exact, NMPC/MPCC/estimator rtol=1e-6, atol=1e-6)"
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
    print("running the separate matched benchmark workloads (60 s each) ...")
    benchmark = _run_benchmarks()
    rendered = render_reference_assets(run, REPO_ROOT)
    write_reference_json(run, benchmark, REFERENCE_DIR)
    update_generated_markdown(REPO_ROOT)
    for path in rendered:
        print(f"wrote {path}")
    print("wrote results/reference/{config,metrics,benchmark,metadata}.json (metadata hashes last)")


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
