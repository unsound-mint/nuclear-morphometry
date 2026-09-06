"""End-to-end test for additional-channel measurements (spec section 25,
Phase 8): H3K9Ac (nuclear), Lamin A/C (shell/core), and MitoTracker
(perinuclear rings), all reusing the Hoechst-derived FixtureSegmenter mask.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import tifffile
from skimage.draw import disk

from dayana_nuclei.io.manifest import validate_manifest, write_manifest_csv
from dayana_nuclei.pipeline.analyze import run_pipeline


def _build_multichannel_run(tmp_path: Path) -> Path:
    """A single 4-channel (CYX) TIFF, matching a real multi-channel confocal
    acquisition. Plain (non-OME) multi-channel TIFFs get bioio's positional
    default channel names "Channel:0:{c}" -- there is no custom-name
    metadata path available without full OME-XML, so the manifest/config
    below reference channels by that positional name (see
    tests/integration/test_pipeline_e2e.py's identical convention for the
    single-channel case)."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shape = (128, 128)

    stack = np.zeros((4, *shape), dtype=np.uint16)
    rr, cc = disk((64, 64), 20)
    stack[0][rr, cc] = 5000  # Hoechst
    stack[1][rr, cc] = 1000  # H3K9Ac
    stack[2][rr, cc] = 2000  # LaminAC
    stack[3] = 100  # MitoTracker background
    outer_rr, outer_cc = disk((64, 64), 25)
    stack[3][outer_rr, outer_cc] = 800
    stack[3][rr, cc] = 100

    stack_path = data_dir / "field.tif"
    tifffile.imwrite(
        stack_path,
        stack,
        resolution=(50000.0, 50000.0),
        resolutionunit="CENTIMETER",
        metadata={"axes": "CYX"},
        photometric="minisblack",
    )

    image_id = "SW620_Sort01_low_48h_Field001"
    manifest_df = pl.DataFrame(
        {
            "image_id": [image_id] * 4,
            "cell_line": ["SW620"] * 4,
            "sort_id": ["Sort01"] * 4,
            "condition": ["low"] * 4,
            "timepoint": ["48h"] * 4,
            "field": ["001"] * 4,
            "channel": ["Channel:0:0", "Channel:0:1", "Channel:0:2", "Channel:0:3"],
            "path": [str(stack_path)] * 4,
            "scene": [None] * 4,
            "acquisition_batch": [None] * 4,
        }
    )
    result = validate_manifest(manifest_df)
    assert result.is_valid, result.errors
    manifest_path = tmp_path / "manifest.csv"
    write_manifest_csv(manifest_df, manifest_path)

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""
        [experiment]
        name = "multichannel_test"
        manifest = "{manifest_path.as_posix()}"
        output_root = "{(tmp_path / "results").as_posix()}"

        [analysis]
        mode = "2d"
        projection = "none"

        [input]
        hoechst_channel = "Channel:0:0"
        additional_channels = ["Channel:0:1", "Channel:0:2", "Channel:0:3"]

        [segmentation]
        backend = "fixture"
        normalize_for_segmentation = false

        [[measurements.additional_channels]]
        channel = "Channel:0:1"
        prefix = "h3k9ac"
        kind = "nuclear"

        [[measurements.additional_channels]]
        channel = "Channel:0:2"
        prefix = "laminac"
        kind = "lamin_shell_core"
        shell_width_um = 1.0

        [[measurements.additional_channels]]
        channel = "Channel:0:3"
        prefix = "mito"
        kind = "mitotracker_rings"
        near_ring_um = [0.0, 1.0]
        far_ring_um = [3.0, 4.0]

        [output]
        save_masks = false
        write_csv = false
        """
    )
    return run_pipeline(config_path)


def test_multichannel_run_populates_prefixed_columns(tmp_path: Path) -> None:
    run_dir = _build_multichannel_run(tmp_path)
    nuclei_df = pl.read_parquet(run_dir / "nuclei.parquet")

    assert nuclei_df.height == 1
    row = nuclei_df.row(0, named=True)

    # H3K9Ac: plain prefixed nuclear intensity.
    assert row["h3k9ac_mean_intensity"] == pytest.approx(1000.0, rel=1e-3)

    # Hoechst stays unprefixed.
    assert row["mean_intensity"] == pytest.approx(5000.0, rel=1e-3)

    # Lamin A/C: shell/core columns populated (uniform intensity -> equal).
    assert row["laminac_total_mean_intensity"] == pytest.approx(2000.0, rel=1e-3)
    assert row["laminac_shell_mean_intensity"] == pytest.approx(2000.0, rel=1e-3)

    # MitoTracker: near ring (bright) vs far ring (dim), enrichment > 1.
    assert row["mito_near_ring_mean_intensity"] == pytest.approx(800.0, rel=1e-3)
    assert row["mito_far_ring_mean_intensity"] == pytest.approx(100.0, rel=1e-3)
    assert row["mito_perinuclear_enrichment_ratio"] == pytest.approx(8.0, rel=1e-3)


def test_missing_additional_channel_leaves_columns_null_not_broken(tmp_path: Path) -> None:
    """A field missing one of the configured additional channels (spec
    25.2: "if present") must not break the run -- its columns for that
    channel are simply null, and the shared schema across fields still
    concatenates cleanly."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    shape = (64, 64)
    hoechst = np.zeros(shape, dtype=np.uint16)
    rr, cc = disk((32, 32), 10)
    hoechst[rr, cc] = 5000
    hoechst_path = data_dir / "hoechst.tif"
    tifffile.imwrite(
        hoechst_path,
        hoechst,
        resolution=(50000.0, 50000.0),
        resolutionunit="CENTIMETER",
        metadata={"axes": "YX"},
    )

    image_id = "SW620_Sort01_low_48h_Field002"
    manifest_df = pl.DataFrame(
        {
            "image_id": [image_id],
            "cell_line": ["SW620"],
            "sort_id": ["Sort01"],
            "condition": ["low"],
            "timepoint": ["48h"],
            "field": ["002"],
            "channel": ["Channel:0:0"],
            "path": [str(hoechst_path)],
            "scene": [None],
            "acquisition_batch": [None],
        }
    )
    result = validate_manifest(manifest_df)
    assert result.is_valid, result.errors
    manifest_path = tmp_path / "manifest.csv"
    write_manifest_csv(manifest_df, manifest_path)

    config_path = tmp_path / "config.toml"
    config_path.write_text(
        f"""
        [experiment]
        name = "missing_channel_test"
        manifest = "{manifest_path.as_posix()}"
        output_root = "{(tmp_path / "results").as_posix()}"

        [analysis]
        mode = "2d"
        projection = "none"

        [input]
        hoechst_channel = "Channel:0:0"
        additional_channels = ["H3K9Ac"]

        [segmentation]
        backend = "fixture"
        normalize_for_segmentation = false

        [[measurements.additional_channels]]
        channel = "H3K9Ac"
        prefix = "h3k9ac"
        kind = "nuclear"

        [output]
        save_masks = false
        write_csv = false
        """
    )
    run_dir = run_pipeline(config_path)

    from dayana_nuclei.pipeline.run_state import load_run_state

    run_state = load_run_state(run_dir / "run_state.json")
    assert run_state is not None
    assert all(f.status == "complete" for f in run_state.fields.values()), run_state.fields

    nuclei_df = pl.read_parquet(run_dir / "nuclei.parquet")
    assert nuclei_df.height == 1
    assert "h3k9ac_mean_intensity" in nuclei_df.columns
    assert nuclei_df["h3k9ac_mean_intensity"].null_count() == 1
