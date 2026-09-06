"""End-to-end architecture proof (spec sections 36.3, 53.4): synthetic
manifest -> validate -> deterministic FixtureSegmenter -> masks + Parquet ->
resume without duplicate rows -> prepare-analysis, all on CPU, no Cellpose.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import tifffile
from skimage.draw import disk, ellipse

from dayana_nuclei.config import load_config
from dayana_nuclei.io.manifest import validate_manifest, write_manifest_csv
from dayana_nuclei.pipeline.analyze import run_pipeline
from dayana_nuclei.pipeline.run_state import load_run_state


def _write_synthetic_field(path: Path, *, elongated: bool, border_object: bool) -> None:
    image = np.zeros((256, 256), dtype=np.uint16)
    rr, cc = disk((80, 80), 30)
    image[rr, cc] = 5000
    if elongated:
        rr, cc = ellipse(180, 180, 8, 60)
        image[rr, cc] = 5000
    else:
        rr, cc = disk((180, 180), 25)
        image[rr, cc] = 5000
    if border_object:
        rr, cc = disk((5, 128), 20)
        rr = np.clip(rr, 0, 255)
        image[rr, cc] = 5000

    tifffile.imwrite(
        path,
        image,
        resolution=(1.0 / 0.2, 1.0 / 0.2),
        resolutionunit="CENTIMETER",
        metadata={"axes": "YX"},
    )


def _write_blank_field(path: Path) -> None:
    """A field with no signal above threshold: FixtureSegmenter finds zero objects."""
    image = np.zeros((256, 256), dtype=np.uint16)
    tifffile.imwrite(
        path,
        image,
        resolution=(1.0 / 0.2, 1.0 / 0.2),
        resolutionunit="CENTIMETER",
        metadata={"axes": "YX"},
    )


def _build_config_and_manifest(tmp_path: Path, *, include_blank_field: bool = False) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    field_a = data_dir / "field_a.tif"
    field_b = data_dir / "field_b.tif"
    _write_synthetic_field(field_a, elongated=False, border_object=True)
    _write_synthetic_field(field_b, elongated=True, border_object=False)

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
            # Plain (non-OME) synthetic TIFFs have no real channel-name metadata;
            # bioio assigns the positional default "Channel:0:0" to a single-channel
            # file. Real Hoechst channel resolution is exercised in
            # tests/unit/test_io_images.py against BioIO's channel_names directly.
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
        name = "e2e_test"
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


def test_end_to_end_synthetic_run(tmp_path: Path) -> None:
    config_path = _build_config_and_manifest(tmp_path)

    run_dir = run_pipeline(config_path)

    run_state = load_run_state(run_dir / "run_state.json")
    assert run_state is not None
    assert all(f.status == "complete" for f in run_state.fields.values())

    nuclei_path = run_dir / "nuclei.parquet"
    fields_path = run_dir / "fields.parquet"
    assert nuclei_path.exists()
    assert fields_path.exists()
    assert (run_dir / "nuclei.csv").exists()
    assert (run_dir / "provenance.json").exists()

    nuclei_df = pl.read_parquet(nuclei_path)
    fields_df = pl.read_parquet(fields_path)
    assert fields_df.height == 2
    assert nuclei_df.height >= 3  # at least 2 real objects per field + the border object

    masks = list((run_dir / "masks").glob("*_labels.tif"))
    assert len(masks) == 2

    # 53.6: the elongated object must NOT be excluded merely for being elongated/low-circularity.
    elongated_rows = nuclei_df.filter(pl.col("eccentricity") > 0.9)
    assert elongated_rows.height >= 1
    assert not elongated_rows["qc_excluded_default"].any()

    # 53.6: the border object must be kept but excluded by default, with reason "border".
    border_rows = nuclei_df.filter(pl.col("qc_border"))
    assert border_rows.height >= 1
    assert border_rows["qc_excluded_default"].to_list() == [True] * border_rows.height
    assert set(border_rows["qc_exclusion_reason"].to_list()) == {"border"}


def test_resume_does_not_duplicate_rows(tmp_path: Path) -> None:
    config_path = _build_config_and_manifest(tmp_path)
    run_dir = run_pipeline(config_path)
    nuclei_before = pl.read_parquet(run_dir / "nuclei.parquet")

    resumed_dir = run_pipeline(config_path, resume_run_dir=run_dir)
    nuclei_after = pl.read_parquet(resumed_dir / "nuclei.parquet")

    assert nuclei_after.height == nuclei_before.height


def test_field_with_zero_objects_does_not_break_the_run(tmp_path: Path) -> None:
    """A field where segmentation finds nothing (blank/focus-failure/etc.) must not
    poison nuclei.parquet for the rest of the run (spec 33: one field's outcome
    is isolated from the others)."""
    config_path = _build_config_and_manifest(tmp_path, include_blank_field=True)

    run_dir = run_pipeline(config_path)

    run_state = load_run_state(run_dir / "run_state.json")
    assert run_state is not None
    assert all(f.status == "complete" for f in run_state.fields.values()), run_state.fields

    fields_df = pl.read_parquet(run_dir / "fields.parquet")
    assert fields_df.height == 3
    blank_field_count = fields_df.filter(pl.col("image_id").str.ends_with("Field003"))[
        "object_count"
    ].to_list()
    assert blank_field_count == [0]

    # The other two fields' rows must survive the concat with the zero-row field.
    nuclei_df = pl.read_parquet(run_dir / "nuclei.parquet")
    assert nuclei_df.height >= 3
    assert set(nuclei_df.columns) >= {"qc_excluded_default", "qc_exclusion_reason"}

    from dayana_nuclei.export import prepare_analysis

    out_path = prepare_analysis(run_dir)
    assert pl.read_parquet(out_path).height == nuclei_df.height


def test_resume_retries_an_interrupted_running_field(tmp_path: Path) -> None:
    """spec 33: resume must retry a field that was mid-processing when the
    process died, not just pending/failed ones. Simulates a kill by hand-editing
    run_state.json to "running" and deleting that field's partial output."""
    from dayana_nuclei.export import partial_nuclei_path
    from dayana_nuclei.pipeline.run_state import save_run_state

    config_path = _build_config_and_manifest(tmp_path)
    run_dir = run_pipeline(config_path)

    run_state = load_run_state(run_dir / "run_state.json")
    assert run_state is not None
    interrupted_image_id = next(iter(run_state.fields))
    run_state.fields[interrupted_image_id].status = "running"
    run_state.fields[interrupted_image_id].completed_at = None
    save_run_state(run_dir / "run_state.json", run_state)
    partial_nuclei_path(run_dir, interrupted_image_id).unlink()

    resumed_dir = run_pipeline(config_path, resume_run_dir=run_dir)

    resumed_state = load_run_state(resumed_dir / "run_state.json")
    assert resumed_state is not None
    assert resumed_state.fields[interrupted_image_id].status == "complete"
    assert partial_nuclei_path(resumed_dir, interrupted_image_id).exists()


def test_prepare_analysis_after_run(tmp_path: Path) -> None:
    from dayana_nuclei.export import prepare_analysis

    config_path = _build_config_and_manifest(tmp_path)
    run_dir = run_pipeline(config_path)

    out_path = prepare_analysis(run_dir)
    analysis_df = pl.read_parquet(out_path)
    nuclei_df = pl.read_parquet(run_dir / "nuclei.parquet")

    assert analysis_df.height == nuclei_df.height
    assert "include_default" in analysis_df.columns
    assert set(analysis_df["sort_id"].unique().to_list()) == {"Sort01"}
