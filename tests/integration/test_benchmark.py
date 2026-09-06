"""Integration test for pipeline.benchmark (spec section 32): real fixture-
backend pipeline run through run_benchmark, asserting real per-stage timings
and metadata are reported, and that a scratch run leaves no results/ output.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import tifffile
from skimage.draw import disk

from dayana_nuclei.config import load_config
from dayana_nuclei.io.manifest import validate_manifest, write_manifest_csv
from dayana_nuclei.pipeline.benchmark import run_benchmark


def _write_synthetic_field(path: Path) -> None:
    image = np.zeros((128, 128), dtype=np.uint16)
    rr, cc = disk((40, 40), 20)
    image[rr, cc] = 5000
    rr, cc = disk((90, 90), 15)
    image[rr, cc] = 5000
    tifffile.imwrite(
        path,
        image,
        resolution=(50000.0, 50000.0),
        resolutionunit="CENTIMETER",
        metadata={"axes": "YX"},
    )


def _write_blank_field(path: Path) -> None:
    """A field with no signal above threshold: FixtureSegmenter finds zero
    objects, exercising the loop-over-objects stages with an empty loop."""
    image = np.zeros((128, 128), dtype=np.uint16)
    tifffile.imwrite(
        path,
        image,
        resolution=(50000.0, 50000.0),
        resolutionunit="CENTIMETER",
        metadata={"axes": "YX"},
    )


def _build_config_and_manifest(tmp_path: Path, *, include_blank_field: bool = False) -> Path:
    """Same shape as tests/integration/test_pipeline_e2e.py's helper
    (fixture backend, two synthetic 2D fields) -- kept self-contained per
    this project's convention of no cross-test-module imports."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    field_a = data_dir / "field_a.tif"
    field_b = data_dir / "field_b.tif"
    _write_synthetic_field(field_a)
    _write_synthetic_field(field_b)

    image_ids = ["SW620_Sort01_low_48h_Field001", "SW620_Sort01_low_48h_Field002"]
    fields = ["001", "002"]
    paths = [str(field_a), str(field_b)]

    if include_blank_field:
        field_c = data_dir / "field_c.tif"
        _write_blank_field(field_c)
        image_ids.append("SW620_Sort01_low_48h_Field003")
        fields.append("003")
        paths.append(str(field_c))

    n = len(image_ids)
    manifest_df = pl.DataFrame(
        {
            "image_id": image_ids,
            "cell_line": ["SW620"] * n,
            "sort_id": ["Sort01"] * n,
            "condition": ["low"] * n,
            "timepoint": ["48h"] * n,
            "field": fields,
            "channel": ["Channel:0:0"] * n,
            "path": paths,
            "scene": [None] * n,
            "acquisition_batch": [None] * n,
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
        name = "benchmark_test"
        manifest = "{manifest_path.as_posix()}"
        output_root = "{(tmp_path / "results").as_posix()}"

        [analysis]
        mode = "2d"
        projection = "none"

        [input]
        hoechst_channel = "Channel:0:0"

        [segmentation]
        backend = "fixture"
        normalize_for_segmentation = false

        [output]
        save_masks = true
        write_csv = true
        """
    )
    config, _ = load_config(config_path)
    assert config.segmentation.backend == "fixture"
    return config_path


def test_benchmark_reports_stage_timings_and_metadata(tmp_path: Path) -> None:
    config_path = _build_config_and_manifest(tmp_path)

    report = run_benchmark(config_path)

    assert len(report.fields) == 2
    assert report.segmentation_backend == "fixture"
    for field in report.fields:
        assert field.object_count >= 1
        assert len(field.shape) == 2
        assert field.stage_timings.total_s > 0.0
        assert field.stage_timings.io_s > 0.0
        assert field.stage_timings.segmentation_s > 0.0
        assert field.stage_timings.row_assembly_s > 0.0
        # mask_serialization_s is always present (real, never missing) even
        # though save_masks = true actually ran it for this config.
        assert field.stage_timings.mask_serialization_s >= 0.0
        # total_s covers strictly more work than any single stage.
        assert field.stage_timings.total_s >= field.stage_timings.segmentation_s

    # A benchmark run is a timing measurement, not a kept analysis run:
    # run_benchmark must never create the config's output_root at all.
    config, _ = load_config(config_path)
    assert not config.experiment.output_root.exists()


def test_benchmark_handles_zero_object_field(tmp_path: Path) -> None:
    """Regression: qc_s/row_assembly_s are only set inside the per-object
    loop in _process_field, so a field with zero objects must not leave
    those stage-timing keys missing (StageTimings(**stage_timings) would
    otherwise raise a pydantic validation error for a real, reachable
    empty-field input, not just a hypothetical one)."""
    config_path = _build_config_and_manifest(tmp_path, include_blank_field=True)

    report = run_benchmark(config_path)

    assert len(report.fields) == 3
    blank = next(f for f in report.fields if f.object_count == 0)
    assert blank.stage_timings.qc_s >= 0.0
    assert blank.stage_timings.row_assembly_s >= 0.0
    assert blank.stage_timings.total_s > 0.0


def test_benchmark_respects_limit(tmp_path: Path) -> None:
    config_path = _build_config_and_manifest(tmp_path)

    report = run_benchmark(config_path, limit=1)

    assert len(report.fields) == 1


def test_benchmark_raises_on_empty_manifest_after_limit_zero(tmp_path: Path) -> None:
    config_path = _build_config_and_manifest(tmp_path)

    with pytest.raises(ValueError, match="No fields to benchmark"):
        run_benchmark(config_path, limit=0)
