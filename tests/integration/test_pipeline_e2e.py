"""End-to-end architecture proof (spec sections 36.3, 53.4): synthetic
manifest -> validate -> deterministic FixtureSegmenter -> masks + Parquet ->
resume without duplicate rows -> prepare-analysis, all on CPU, no Cellpose.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import pytest
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

    # spec 21: image-level QC metrics land on fields.parquet, measurement-only.
    assert fields_df["image_max_intensity"].to_list() == [5000.0, 5000.0]
    assert (fields_df["image_occupied_fraction"] > 0).all()
    assert fields_df["image_saturation_fraction"].null_count() == 0

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


def _build_texture_config_and_manifest(tmp_path: Path) -> Path:
    """Same fields as _build_config_and_manifest but with texture_2d enabled and a
    non-default distance list, to prove nuclei_table_schema() is genuinely derived
    from config rather than accidentally matching the (3, 5, 10, 20) default."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    field_a = data_dir / "field_a.tif"
    _write_synthetic_field(field_a, elongated=False, border_object=False)

    manifest_df = pl.DataFrame(
        {
            "image_id": ["SW620_Sort01_low_48h_Field001"],
            "cell_line": ["SW620"],
            "sort_id": ["Sort01"],
            "condition": ["low"],
            "timepoint": ["48h"],
            "field": ["001"],
            "channel": ["Channel:0:0"],
            "path": [str(field_a)],
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
        name = "e2e_texture_test"
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

        [measurements]
        texture_2d = true
        texture_distances_px = [3, 5]

        [output]
        save_masks = false
        write_csv = false
        """
    )
    return config_path


def _write_field_with_pixel_size(
    path: Path, *, x_um: float, y_um: float, elongated: bool = False
) -> None:
    image = np.zeros((256, 256), dtype=np.uint16)
    rr, cc = disk((80, 80), 30)
    image[rr, cc] = 5000
    if elongated:
        rr, cc = ellipse(180, 180, 8, 60)
        image[rr, cc] = 5000
    else:
        rr, cc = disk((180, 180), 25)
        image[rr, cc] = 5000
    tifffile.imwrite(
        path,
        image,
        # bioio-tifffile resolves physical_pixel_sizes as 1e4 / (resolution
        # value in pixels/cm) um/pixel for RESUNIT.CENTIMETER -- verified
        # directly against BioImage.physical_pixel_sizes, since this file's
        # own pre-existing helper's "0.2" resolution values actually resolve
        # to 2000 um/px (harmless there: no existing test asserts the literal
        # calibration value), which this test's assertions do depend on.
        resolution=(10_000.0 / x_um, 10_000.0 / y_um),
        resolutionunit="CENTIMETER",
        metadata={"axes": "YX"},
    )


def _build_texture_um_config_and_manifest(
    tmp_path: Path, *, second_field_x_um: float = 0.2, second_field_y_um: float = 0.2
) -> Path:
    """Two fields with potentially *different* X/Y calibration, both requesting
    the same measurements.texture_distances_um -- proves docs/decisions/0012's
    fix: columns are named by the configured um value (identical across
    fields) even when the two fields resolve that distance to different pixel
    counts, so finalize_tables' concat across fields does not break."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    field_a = data_dir / "field_a.tif"
    field_b = data_dir / "field_b.tif"
    _write_field_with_pixel_size(field_a, x_um=0.2, y_um=0.2)
    _write_field_with_pixel_size(field_b, x_um=second_field_x_um, y_um=second_field_y_um)

    manifest_df = pl.DataFrame(
        {
            "image_id": ["SW620_Sort01_low_48h_Field001", "SW620_Sort01_low_48h_Field002"],
            "cell_line": ["SW620", "SW620"],
            "sort_id": ["Sort01", "Sort01"],
            "condition": ["low", "low"],
            "timepoint": ["48h", "48h"],
            "field": ["001", "002"],
            "channel": ["Channel:0:0", "Channel:0:0"],
            "path": [str(field_a), str(field_b)],
            "scene": [None, None],
            "acquisition_batch": [None, None],
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
        name = "e2e_texture_um_test"
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

        [measurements]
        texture_2d = true
        texture_distances_px = []
        texture_distances_um = [2.0]

        [output]
        save_masks = false
        write_csv = false
        """
    )
    return config_path


def test_end_to_end_run_with_texture_distances_um(tmp_path: Path) -> None:
    """Fields with different X/Y calibration (0.2 vs 0.25 um/px, resolving 2.0
    um to 10px and 8px respectively) must still produce one shared, correctly
    named ``*_d2um`` column set (docs/decisions/0012), not diverge and break
    the run's single nuclei.parquet schema."""
    config_path = _build_texture_um_config_and_manifest(
        tmp_path, second_field_x_um=0.25, second_field_y_um=0.25
    )

    run_dir = run_pipeline(config_path)

    nuclei_df = pl.read_parquet(run_dir / "nuclei.parquet")
    assert nuclei_df.height >= 2  # at least one object per field

    expected_columns = {
        f"{prop}_d2um" for prop in ("contrast", "homogeneity", "correlation", "energy", "entropy")
    }
    assert expected_columns <= set(nuclei_df.columns)
    assert "contrast_d10" not in nuclei_df.columns
    assert "contrast_d8" not in nuclei_df.columns

    for column in expected_columns:
        assert nuclei_df[column].null_count() == 0, column

    from dayana_nuclei.export import prepare_analysis

    out_path = prepare_analysis(run_dir)
    assert pl.read_parquet(out_path).height == nuclei_df.height


def test_texture_distances_um_rejects_anisotropic_field(tmp_path: Path) -> None:
    """Per-field failure isolation (spec 33) means an anisotropic-pixel field's
    ValueError never propagates out of run_pipeline -- it marks that one field
    failed and continues, so the regression check reads run_state.json rather
    than expecting run_pipeline itself to raise."""
    config_path = _build_texture_um_config_and_manifest(
        tmp_path, second_field_x_um=0.2, second_field_y_um=0.3
    )

    run_dir = run_pipeline(config_path)

    run_state = load_run_state(run_dir / "run_state.json")
    field_b = run_state.fields["SW620_Sort01_low_48h_Field002"]
    assert field_b.status == "failed"
    assert field_b.error is not None and "square X/Y pixels" in field_b.error
    # The isotropic field must still have completed and been measured normally.
    field_a = run_state.fields["SW620_Sort01_low_48h_Field001"]
    assert field_a.status == "complete"


def _build_radial_config_and_manifest(tmp_path: Path) -> Path:
    """Same fields as _build_config_and_manifest but with radial_distribution_2d
    enabled and a non-default bin count, to prove nuclei_table_schema() is
    genuinely derived from config rather than accidentally matching the default."""
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    field_a = data_dir / "field_a.tif"
    _write_synthetic_field(field_a, elongated=False, border_object=False)

    manifest_df = pl.DataFrame(
        {
            "image_id": ["SW620_Sort01_low_48h_Field001"],
            "cell_line": ["SW620"],
            "sort_id": ["Sort01"],
            "condition": ["low"],
            "timepoint": ["48h"],
            "field": ["001"],
            "channel": ["Channel:0:0"],
            "path": [str(field_a)],
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
        name = "e2e_radial_test"
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

        [measurements]
        radial_distribution_2d = true
        radial_bins = 3

        [output]
        save_masks = false
        write_csv = false
        """
    )
    return config_path


def test_end_to_end_run_with_radial_distribution_columns(tmp_path: Path) -> None:
    """Same schema-mismatch class of bug as the texture e2e test: the pipeline's
    row dict must match nuclei_table_schema()'s dynamically-derived radial-bin
    column set for the configured (non-default) bin count."""
    config_path = _build_radial_config_and_manifest(tmp_path)

    run_dir = run_pipeline(config_path)

    nuclei_df = pl.read_parquet(run_dir / "nuclei.parquet")
    assert nuclei_df.height >= 1

    expected_columns = {
        f"radial_bin{b}_{prop}"
        for b in range(3)
        for prop in ("mean_intensity", "frac_intensity", "frac_pixels")
    }
    assert expected_columns <= set(nuclei_df.columns)
    # A 4th bin (not requested) must NOT appear (proves the schema is config-derived).
    assert "radial_bin3_mean_intensity" not in nuclei_df.columns

    for column in expected_columns:
        assert nuclei_df[column].null_count() == 0, column

    from dayana_nuclei.export import prepare_analysis

    out_path = prepare_analysis(run_dir)
    assert pl.read_parquet(out_path).height == nuclei_df.height


def test_end_to_end_run_with_texture_columns(tmp_path: Path) -> None:
    """Reproduces the class of bug caught in schema.NUCLEI_TABLE_SCHEMA (spec/decision
    0004): the pipeline's actual per-object row dict must match nuclei_table_schema()'s
    dynamically-derived texture column set, not just the 4-distance default."""
    config_path = _build_texture_config_and_manifest(tmp_path)

    run_dir = run_pipeline(config_path)

    nuclei_df = pl.read_parquet(run_dir / "nuclei.parquet")
    assert nuclei_df.height >= 1

    expected_texture_columns = {
        f"{prop}_d{distance}"
        for distance in (3, 5)
        for prop in ("contrast", "homogeneity", "correlation", "energy", "entropy")
    }
    assert expected_texture_columns <= set(nuclei_df.columns)
    # A distance not requested must NOT appear (proves the schema is config-derived).
    assert "contrast_d10" not in nuclei_df.columns

    for column in expected_texture_columns:
        assert nuclei_df[column].null_count() == 0, column

    # finalize_tables' concat must succeed across fields with this schema too.
    from dayana_nuclei.export import prepare_analysis

    out_path = prepare_analysis(run_dir)
    assert pl.read_parquet(out_path).height == nuclei_df.height


def _write_synthetic_3d_field(path: Path) -> None:
    shape = (30, 100, 100)
    zz, yy, xx = np.indices(shape)
    center = np.array(shape) / 2
    sphere = ((zz - center[0]) ** 2 + (yy - center[1]) ** 2 + (xx - center[2]) ** 2) <= 8**2
    image = np.where(sphere, 5000, 0).astype(np.uint16)
    tifffile.imwrite(
        path,
        image,
        imagej=True,
        resolution=(1.0 / 0.2, 1.0 / 0.2),
        metadata={"axes": "ZYX", "spacing": 0.5, "unit": "um"},
    )


def _build_3d_config_and_manifest(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    field_a = data_dir / "field_a.tif"
    _write_synthetic_3d_field(field_a)

    manifest_df = pl.DataFrame(
        {
            "image_id": ["SW620_Sort01_low_48h_Field001"],
            "cell_line": ["SW620"],
            "sort_id": ["Sort01"],
            "condition": ["low"],
            "timepoint": ["48h"],
            "field": ["001"],
            "channel": ["Channel:0:0"],
            "path": [str(field_a)],
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
        name = "e2e_3d_test"
        manifest = "{manifest_path.as_posix()}"
        output_root = "{(tmp_path / "results").as_posix()}"

        [analysis]
        mode = "3d"
        projection = "none"

        [input]
        hoechst_channel = "Channel:0:0"

        [segmentation]
        backend = "fixture"
        normalize_for_segmentation = false

        [output]
        save_masks = false
        write_csv = false
        """
    )
    return config_path


def test_end_to_end_3d_run_via_fixture_backend(tmp_path: Path) -> None:
    """Exercises the 3D pipeline wiring (analysis.mode='3d' -> measure_3d_morphology
    -> nuclei_table_schema()) without depending on Cellpose-SAM's 3D segmentation
    quality (see docs/decisions/0008): FixtureSegmenter handles 3D via
    connectivity=image.ndim, isolating this test to pipeline architecture."""
    config_path = _build_3d_config_and_manifest(tmp_path)

    run_dir = run_pipeline(config_path)

    run_state = load_run_state(run_dir / "run_state.json")
    assert run_state is not None
    assert all(f.status == "complete" for f in run_state.fields.values()), run_state.fields

    nuclei_df = pl.read_parquet(run_dir / "nuclei.parquet")
    assert nuclei_df.height >= 1

    # 3D-only columns must be populated.
    assert nuclei_df["volume_um3"].null_count() == 0
    assert (nuclei_df["volume_um3"] > 0).all()
    assert nuclei_df["axis_major_um"].null_count() == 0
    # sphericity can be None for border/too-small objects, but this synthetic
    # interior sphere must produce a real value.
    assert nuclei_df["sphericity"].null_count() < nuclei_df.height

    # 2D-only columns must be present (schema is unconditional) but null in 3D mode.
    assert "circularity" in nuclei_df.columns
    assert nuclei_df["circularity"].null_count() == nuclei_df.height

    fields_df = pl.read_parquet(run_dir / "fields.parquet")
    assert fields_df["axes"].to_list() == ["ZYX"]
    assert fields_df["spacing_z_um"].to_list() == pytest.approx([0.5])


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
