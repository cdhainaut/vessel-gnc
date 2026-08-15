"""Focused rendering tests for the flagship reference assets.

Renders the three committed assets plus the ignored trajectory figure from
one short (2 s) in-memory reference run into temporary directories. No
120 s flagship, no benchmark workload and no repository file is touched:
``render_reference_assets`` receives ``tmp_path`` as the repository root.
Rendering uses the headless Agg backend so the tests run on CI without a
display.
"""

from __future__ import annotations

import matplotlib
import pytest

matplotlib.use("Agg")  # headless backend; must precede any pyplot import

from vessel_gnc.reference import ReferenceScenarioConfig, run_reference_scenario
from vessel_gnc.reference_artifacts import render_reference_assets

# Short deterministic scenario shared by every render test (2 s, LOS 0.1 s,
# NMPC 0.2 s), with a reduced estimator transient so metrics have data.
SHORT = ReferenceScenarioConfig(duration_s=2.0, estimator_transient_s=0.5)

EXPECTED_RELPATHS = (
    "assets/hero.gif",
    "assets/controller_comparison.png",
    "assets/current_estimation.png",
    "results/reference/nmpc_trajectory.png",
)


@pytest.fixture(scope="module")
def short_run():
    """One real short reference run, shared by all rendering tests."""
    return run_reference_scenario(SHORT)


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    import matplotlib.pyplot as plt

    plt.close("all")


def test_render_writes_all_four_assets(tmp_path, short_run):
    written = render_reference_assets(short_run, tmp_path)
    assert [path.relative_to(tmp_path).as_posix() for path in written] == list(
        EXPECTED_RELPATHS
    )
    for path in written:
        assert path.is_file()
        assert path.stat().st_size > 0
    # Magic bytes: GIF for the animation, PNG for the three figures.
    assert (tmp_path / "assets" / "hero.gif").read_bytes().startswith(b"GIF8")
    for name in (
        "assets/controller_comparison.png",
        "assets/current_estimation.png",
        "results/reference/nmpc_trajectory.png",
    ):
        assert (tmp_path / name).read_bytes().startswith(b"\x89PNG")


def test_render_is_deterministic_across_identical_runs(tmp_path, short_run):
    first_paths = render_reference_assets(short_run, tmp_path / "first")
    second_paths = render_reference_assets(short_run, tmp_path / "second")
    for first, second in zip(first_paths, second_paths, strict=True):
        assert first.read_bytes() == second.read_bytes(), (
            f"{first.relative_to(tmp_path)} is not byte-identical"
        )


def test_render_figures_are_sized_and_nonblank(tmp_path, short_run):
    import numpy as np
    from PIL import Image

    render_reference_assets(short_run, tmp_path)
    for name in ("controller_comparison.png", "current_estimation.png"):
        image = Image.open(tmp_path / "assets" / name).convert("L")
        assert image.width > 600
        assert image.height > 400
        assert float(np.std(np.asarray(image))) > 5.0  # not a blank canvas
    gif = Image.open(tmp_path / "assets" / "hero.gif")
    assert gif.n_frames >= 1


def test_render_consumes_the_given_run_object(tmp_path, short_run, monkeypatch):
    # render_reference_assets must render from the shared in-memory run (the
    # one produced by the tool's single run_reference_scenario call), never
    # from a fresh simulation of its own.
    import vessel_gnc.reference_artifacts as artifacts

    received: list[object] = []
    real_metrics = artifacts.reference_metrics

    def spy(run):
        received.append(run)
        return real_metrics(run)

    monkeypatch.setattr(artifacts, "reference_metrics", spy)
    render_reference_assets(short_run, tmp_path)
    assert received == [short_run]
