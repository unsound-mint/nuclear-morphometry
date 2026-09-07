"""Integration test for qc/viewer.py (spec section 23, acceptance 53.9):
builds a real napari Viewer -- offscreen Qt, no real display needed -- and
exercises field switching, object selection, and tag persistence against a
real FixtureSegmenter pipeline run. Never calls napari.run() (the blocking
event loop); build_qc_viewer stops short of that on purpose so it can be
exercised headlessly here.

QT_QPA_PLATFORM must be set before napari (and therefore Qt) is first
imported anywhere in the process, so it is set at module level, before any
import below -- verified directly against a real napari.Viewer(show=False)
call in this environment (no DISPLAY required).
"""

from __future__ import annotations

import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from pathlib import Path

import numpy as np
import polars as pl
import pytest
import tifffile
from skimage.draw import disk

pytest.importorskip("napari")

from dayana_nuclei.io.manifest import validate_manifest, write_manifest_csv
from dayana_nuclei.io.masks import mask_path_for
from dayana_nuclei.pipeline.analyze import run_pipeline
from dayana_nuclei.qc.annotations import load_annotations
from dayana_nuclei.qc.viewer import (
    _TAG_KEYS,
    _format_measurement_text,
    _selected_object_number,
    build_qc_viewer,
)

pytestmark = pytest.mark.gui


def _write_synthetic_field(path: Path) -> None:
    image = np.zeros((128, 128), dtype=np.uint16)
    rr, cc = disk((40, 40), 20)
    image[rr, cc] = 5000
    rr, cc = disk((90, 90), 15)
    image[rr, cc] = 5000
    tifffile.imwrite(
        path,
        image,
        resolution=(50_000.0, 50_000.0),
        resolutionunit="CENTIMETER",
        metadata={"axes": "YX"},
    )


def _build_run(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()
    field_a = data_dir / "field_a.tif"
    field_b = data_dir / "field_b.tif"
    _write_synthetic_field(field_a)
    _write_synthetic_field(field_b)

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
        name = "qc_viewer_test"
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
    return run_pipeline(config_path)


def test_selected_object_number_treats_background_as_none() -> None:
    assert _selected_object_number(None) is None
    assert _selected_object_number(0) is None
    assert _selected_object_number(np.int32(0)) is None
    assert _selected_object_number(7) == 7
    assert _selected_object_number(np.int32(3)) == 3


def test_format_measurement_text_reports_no_selection() -> None:
    nuclei_df = pl.DataFrame({"image_id": ["a"], "object_number": [1], "area_um2": [12.5]})
    assert "No object selected" in _format_measurement_text(nuclei_df, "a", None, None)


def test_format_measurement_text_includes_present_columns_only() -> None:
    nuclei_df = pl.DataFrame(
        {
            "image_id": ["a"],
            "object_number": [1],
            "area_um2": [12.5],
            "qc_border": [False],
            "volume_um3": [None],
        }
    )
    text = _format_measurement_text(nuclei_df, "a", 1, "debris")
    assert "Object 1" in text
    assert "tag: debris" in text
    assert "area_um2: 12.5" in text
    assert "volume_um3" not in text  # null on this (2D) row, must not appear


def test_build_qc_viewer_loads_hoechst_and_labels(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path)
    qc_viewer = build_qc_viewer(run_dir)
    try:
        assert qc_viewer.image_id == "SW620_Sort01_low_48h_Field001"
        assert "hoechst" in qc_viewer.viewer.layers
        labels_layer = qc_viewer.viewer.layers["labels"]
        assert labels_layer.editable is False
        assert "qc annotations" in qc_viewer.viewer.layers
    finally:
        qc_viewer.viewer.close()


def test_apply_tag_saves_annotation_without_touching_mask_or_source(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path)
    qc_viewer = build_qc_viewer(run_dir)
    try:
        image_id = qc_viewer.image_id
        mask_path = mask_path_for(run_dir, image_id)
        mask_bytes_before = mask_path.read_bytes()
        assert qc_viewer._labels is not None
        object_number = int(qc_viewer._labels.max())
        assert object_number >= 1

        qc_viewer.select_object(object_number)
        assert qc_viewer.info_label is not None
        assert f"Object {object_number}" in qc_viewer.info_label.value

        qc_viewer.apply_tag("debris")

        annotations = load_annotations(run_dir)
        assert annotations[(image_id, object_number)].tag == "debris"
        assert mask_path.read_bytes() == mask_bytes_before
        assert "tag: debris" in qc_viewer.info_label.value
    finally:
        qc_viewer.viewer.close()


def test_apply_tag_with_nothing_selected_does_not_raise_or_save(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path)
    qc_viewer = build_qc_viewer(run_dir)
    try:
        qc_viewer.apply_tag("good")
        assert load_annotations(run_dir) == {}
        assert qc_viewer.info_label is not None
        assert "Select an object" in qc_viewer.info_label.value
    finally:
        qc_viewer.viewer.close()


def test_reopening_viewer_reloads_existing_annotations(tmp_path: Path) -> None:
    """spec 23: "existing annotations must reload" -- a fresh build_qc_viewer
    on the same run_dir must show a manual tag saved by a prior session."""
    run_dir = _build_run(tmp_path)

    first = build_qc_viewer(run_dir)
    try:
        image_id = first.image_id
        assert first._labels is not None
        object_number = int(first._labels.max())
        first.select_object(object_number)
        first.apply_tag("merge")
    finally:
        first.viewer.close()

    second = build_qc_viewer(run_dir, image_id=image_id)
    try:
        points_layer = second.viewer.layers["qc annotations"]
        assert points_layer.data.shape[0] == 1
        assert list(points_layer.properties["tag"]) == ["merge"]
        # Confirms viewer.bind_key is instance-scoped (not written onto a
        # shared class-level keymap): binding g/d/m/s/o on this *second*
        # Viewer instance, after already having bound them on `first`
        # (closed above), must not raise or silently no-op -- the build
        # above completing at all is the main evidence; this pins that the
        # binding is actually present too.
        assert len(second.viewer.keymap) >= len(_TAG_KEYS)
    finally:
        second.viewer.close()


def test_switch_field_replaces_layers_and_resets_selection(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path)
    qc_viewer = build_qc_viewer(run_dir)
    try:
        image_ids = ("SW620_Sort01_low_48h_Field001", "SW620_Sort01_low_48h_Field002")
        first_image_id = qc_viewer.image_id
        other_image_id = next(i for i in image_ids if i != first_image_id)
        assert qc_viewer._labels is not None
        qc_viewer.select_object(int(qc_viewer._labels.max()))

        qc_viewer.switch_field(other_image_id)

        assert qc_viewer.image_id == other_image_id
        assert qc_viewer.object_number is None
        # Exactly one hoechst/labels layer each -- switching must replace,
        # not accumulate, layers.
        assert sum(1 for layer in qc_viewer.viewer.layers if layer.name == "hoechst") == 1
        assert sum(1 for layer in qc_viewer.viewer.layers if layer.name == "labels") == 1
    finally:
        qc_viewer.viewer.close()


def test_switch_field_rejects_unknown_image_id(tmp_path: Path) -> None:
    run_dir = _build_run(tmp_path)
    qc_viewer = build_qc_viewer(run_dir)
    try:
        with pytest.raises(ValueError, match="not one of this run's analyzed fields"):
            qc_viewer.switch_field("nonexistent_field")
    finally:
        qc_viewer.viewer.close()


def test_build_qc_viewer_raises_on_missing_run(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        build_qc_viewer(tmp_path / "no_such_run")


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


def _build_3d_run(tmp_path: Path) -> Path:
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
        name = "qc_viewer_3d_test"
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
        save_masks = true
        write_csv = false
        """
    )
    return run_pipeline(config_path)


def test_build_qc_viewer_and_tagging_work_for_a_3d_field(tmp_path: Path) -> None:
    """The three 3D-specific code paths in viewer.py (empty-points ndim,
    add_points(ndim=...), and the annotation-marker size computed from the
    YX axes only, not Z) are otherwise never exercised by the 2D-only tests
    above."""
    run_dir = _build_3d_run(tmp_path)
    qc_viewer = build_qc_viewer(run_dir)
    try:
        assert qc_viewer._labels is not None
        assert qc_viewer._labels.ndim == 3
        object_number = int(qc_viewer._labels.max())
        assert object_number >= 1

        qc_viewer.select_object(object_number)
        qc_viewer.apply_tag("split")

        annotations = load_annotations(run_dir)
        assert annotations[(qc_viewer.image_id, object_number)].tag == "split"

        points_layer = qc_viewer.viewer.layers["qc annotations"]
        assert points_layer.data.shape == (1, 3)  # (z, row, col)
        assert points_layer.size.min() >= 2  # not collapsed by the Z depth
    finally:
        qc_viewer.viewer.close()
