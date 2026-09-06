"""Integration tests for qc/report.py (spec section 24), run through a real
FixtureSegmenter pipeline run so the report is exercised against genuine
nuclei.parquet/fields.parquet/masks, not hand-built fixtures.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import polars as pl
import tifffile
from skimage.draw import disk

from dayana_nuclei.io.manifest import validate_manifest, write_manifest_csv
from dayana_nuclei.pipeline.analyze import run_pipeline
from dayana_nuclei.qc.annotations import save_annotation
from dayana_nuclei.qc.report import generate_qc_report


def _write_field(path: Path, *, seed: int) -> None:
    rng = np.random.default_rng(seed)
    image = np.zeros((128, 128), dtype=np.uint16)
    for _ in range(2):
        row = int(rng.integers(20, 108))
        col = int(rng.integers(20, 108))
        rr, cc = disk((row, col), 12)
        image[rr, cc] = 5000
    tifffile.imwrite(
        path,
        image,
        resolution=(1.0 / 0.2, 1.0 / 0.2),
        resolutionunit="CENTIMETER",
        metadata={"axes": "YX"},
    )


def _build_multi_stratum_run(tmp_path: Path) -> Path:
    data_dir = tmp_path / "data"
    data_dir.mkdir()

    strata = [
        ("SW480", "low", "Sort01"),
        ("SW480", "high", "Sort01"),
        ("SW620", "low", "Sort02"),
        ("SW620", "bulk", "Sort02"),
    ]
    image_ids, cell_lines, sort_ids, conditions, fields, paths = [], [], [], [], [], []
    for i, (cell_line, condition, sort_id) in enumerate(strata):
        path = data_dir / f"field_{i}.tif"
        _write_field(path, seed=i)
        image_ids.append(f"{cell_line}_{sort_id}_{condition}_48h_Field{i:03d}")
        cell_lines.append(cell_line)
        sort_ids.append(sort_id)
        conditions.append(condition)
        fields.append(f"{i:03d}")
        paths.append(str(path))

    n = len(image_ids)
    manifest_df = pl.DataFrame(
        {
            "image_id": image_ids,
            "cell_line": cell_lines,
            "sort_id": sort_ids,
            "condition": conditions,
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
        name = "qc_report_test"
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
        write_csv = false
        """
    )
    return run_pipeline(config_path)


def test_qc_report_generates_html_and_overlays(tmp_path: Path) -> None:
    run_dir = _build_multi_stratum_run(tmp_path)

    report_path = generate_qc_report(run_dir, seed=0, n_overlays=4)

    assert report_path == run_dir / "qc" / "report.html"
    text = report_path.read_text()
    assert "QC report" in text
    assert "SW480" in text and "SW620" in text

    overlays = list((run_dir / "qc" / "overlays").glob("*.png"))
    assert len(overlays) == 4  # one field per stratum, all masks saved


def test_qc_report_overlay_selection_is_stratified() -> None:
    fields_df = pl.DataFrame(
        {
            "image_id": [f"f{i}" for i in range(8)],
            "cell_line": ["SW480"] * 4 + ["SW620"] * 4,
            "condition": (["low"] * 2 + ["high"] * 2) * 2,
            "sort_id": ["Sort01"] * 8,
        }
    )
    from dayana_nuclei.qc.report import _select_stratified_overlays

    selected = _select_stratified_overlays(fields_df, n=4, seed=0)

    assert len(selected) == 4
    selected_rows = fields_df.filter(pl.col("image_id").is_in(selected))
    # All 4 strata represented, not just the first n rows by file order.
    assert selected_rows.select(["cell_line", "condition"]).unique().height == 4


def test_qc_report_selection_is_deterministic_for_a_given_seed() -> None:
    fields_df = pl.DataFrame(
        {
            "image_id": [f"f{i}" for i in range(10)],
            "cell_line": ["SW480"] * 5 + ["SW620"] * 5,
            "condition": ["low"] * 10,
            "sort_id": ["Sort01"] * 10,
        }
    )
    from dayana_nuclei.qc.report import _select_stratified_overlays

    first = _select_stratified_overlays(fields_df, n=3, seed=42)
    second = _select_stratified_overlays(fields_df, n=3, seed=42)

    assert first == second


def test_qc_report_reflects_manual_annotations(tmp_path: Path) -> None:
    run_dir = _build_multi_stratum_run(tmp_path)
    nuclei_df = pl.read_parquet(run_dir / "nuclei.parquet")
    first_row = nuclei_df.row(0, named=True)
    save_annotation(
        run_dir,
        image_id=first_row["image_id"],
        object_number=first_row["object_number"],
        tag="debris",
    )

    report_path = generate_qc_report(run_dir, seed=0, n_overlays=0)

    text = report_path.read_text()
    assert "debris: 1" in text


def test_qc_report_raises_clearly_without_finalized_tables(tmp_path: Path) -> None:
    empty_dir = tmp_path / "not_a_run"
    empty_dir.mkdir()

    try:
        generate_qc_report(empty_dir)
    except FileNotFoundError as exc:
        assert "nuclei.parquet" in str(exc)
    else:
        raise AssertionError("expected FileNotFoundError")
