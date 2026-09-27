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


def test_comparison_figure_shows_all_four_controllers_and_progress(tmp_path, short_run):
    # The deterministic comparison figure must display the four controllers
    # (LOS, nominal NMPC, aware NMPC, geometric MPCC) in every panel and
    # compact path-progress rows in the metrics table.
    from vessel_gnc.reference import reference_metrics
    from vessel_gnc.visualization import plot_controller_comparison

    out = tmp_path / "comparison.png"
    figure = plot_controller_comparison(short_run, reference_metrics(short_run), out)
    assert len(figure.axes) == 4

    # Each tracking/control panel carries the four controller lines.
    for ax in (figure.axes[0], figure.axes[1], figure.axes[2]):
        labels = [line.get_label() for line in ax.get_lines() if line.get_label()]
        assert "LOS" in labels
        assert "nominal NMPC" in labels
        assert "aware NMPC" in labels
        assert "geometric MPCC" in labels

    # The metrics table has five columns and the path-progress rows.
    table = next(ax.tables[0] for ax in figure.axes if ax.tables)
    cells = table.get_celld()
    row_count = max(row for row, _ in cells) + 1
    column_count = max(column for _, column in cells) + 1
    header = [
        cells[(0, column)].get_text().get_text() for column in range(column_count)
    ]
    assert header[0] == ""
    assert header[1:] == ["LOS", "Nominal", "Aware", "MPCC"]
    first_column = [cells[(row, 0)].get_text().get_text() for row in range(row_count)]
    assert "Final progress [m]" in first_column
    assert "Progress fraction [-]".replace(" ", "") in [
        label.replace(" ", "") for label in first_column
    ]
    assert "Mean progress rate [m/s]" in first_column


def test_hero_rendering_includes_four_controllers_and_mpcc_horizon(
    tmp_path, short_run, monkeypatch
):
    # The hero animation must receive the four controller trajectories, the
    # MPCC recorded prediction horizons and a compact path-progress overlay
    # (captured at the animate_trajectory call; the GIF itself is rendered
    # by the shared helper covered by test_render_writes_all_four_assets).
    from vessel_gnc import visualization

    captured: dict[str, object] = {}

    def spy(*args, **kwargs):
        captured["kwargs"] = kwargs

    monkeypatch.setattr(visualization, "animate_trajectory", spy)
    render_reference_assets(short_run, tmp_path)
    kwargs = captured["kwargs"]

    labels = [label for label, _, _ in kwargs["extra_trajectories"]]
    assert set(labels) == {
        "LOS baseline",
        "nominal NMPC",
        "disturbance-aware NMPC",
        "geometric MPCC",
    }
    horizon = kwargs["extra_horizon"]
    assert horizon, "MPCC prediction horizon must be drawn in the hero"
    assert kwargs["extra_horizon_label"].startswith("geometric MPCC prediction")
    progress = kwargs["progress_text"]
    line = progress(
        0.5,
        float(short_run.disturbance_aware_nmpc.result.x[0]),
        float(short_run.disturbance_aware_nmpc.result.y[0]),
    )
    assert line is not None
    assert "path progress" in line and "%" in line
