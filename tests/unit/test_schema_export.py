from pathlib import Path

import polars as pl

from nuclear_morphometry.export import (
    finalize_tables,
    partial_fields_path,
    partial_nuclei_path,
    prepare_analysis,
    write_partial_table,
)
from nuclear_morphometry.qc.annotations import save_annotation
from nuclear_morphometry.schema import compute_include_default, include_default_expr


def test_compute_include_default_inverts_exclusion() -> None:
    assert compute_include_default(True) is False
    assert compute_include_default(False) is True


def test_include_default_expr_matches_scalar_helper() -> None:
    df = pl.DataFrame({"qc_excluded_default": [True, False]})
    out = df.with_columns(include_default_expr())
    assert out["include_default"].to_list() == [False, True]


def test_finalize_tables_concatenates_partial_files_and_is_idempotent(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    nuclei_a = pl.DataFrame({"image_id": ["a"], "object_number": [1]})
    nuclei_b = pl.DataFrame({"image_id": ["b"], "object_number": [1]})
    fields_a = pl.DataFrame({"image_id": ["a"], "object_count": [1]})
    write_partial_table(nuclei_a, partial_nuclei_path(run_dir, "a"))
    write_partial_table(nuclei_b, partial_nuclei_path(run_dir, "b"))
    write_partial_table(fields_a, partial_fields_path(run_dir, "a"))

    nuclei_path, fields_path = finalize_tables(run_dir, write_csv=True)
    assert pl.read_parquet(nuclei_path).height == 2
    assert pl.read_parquet(fields_path).height == 1
    assert (run_dir / "nuclei.csv").exists()

    # Idempotent: calling again (e.g. after a resume) must not duplicate rows.
    nuclei_path, _ = finalize_tables(run_dir, write_csv=True)
    assert pl.read_parquet(nuclei_path).height == 2


def test_retried_field_overwrites_its_own_partial_file_not_append(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    first_attempt = pl.DataFrame({"image_id": ["a"], "object_number": [1]})
    retry_attempt = pl.DataFrame({"image_id": ["a", "a"], "object_number": [1, 2]})
    write_partial_table(first_attempt, partial_nuclei_path(run_dir, "a"))
    write_partial_table(retry_attempt, partial_nuclei_path(run_dir, "a"))

    nuclei_path, _ = finalize_tables(run_dir, write_csv=False)
    assert pl.read_parquet(nuclei_path).height == 2


def test_prepare_analysis_adds_include_default_without_dropping_rows(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    pl.DataFrame(
        {
            "image_id": ["a", "a"],
            "object_number": [1, 2],
            "qc_excluded_default": [True, False],
        }
    ).write_parquet(run_dir / "nuclei.parquet")

    out_path = prepare_analysis(run_dir)
    out = pl.read_parquet(out_path)
    assert out.height == 2
    assert out["include_default"].to_list() == [False, True]


def test_prepare_analysis_applies_manual_qc_without_mutating_raw_table(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    raw = pl.DataFrame(
        {
            "image_id": ["a", "a"],
            "object_number": [1, 2],
            "qc_manual_debris": [False, False],
            "qc_manual_merge": [False, False],
            "qc_manual_split": [False, False],
            "qc_manual_other": [False, False],
            "qc_excluded_default": [False, False],
            "qc_exclusion_reason": [None, None],
        },
        schema_overrides={"qc_exclusion_reason": pl.Utf8},
    )
    raw.write_parquet(run_dir / "nuclei.parquet")
    save_annotation(run_dir, image_id="a", object_number=2, tag="merge")

    out = pl.read_parquet(prepare_analysis(run_dir))

    raw_after = pl.read_parquet(run_dir / "nuclei.parquet")
    assert raw_after.equals(raw)
    annotated = out.filter(pl.col("object_number") == 2).row(0, named=True)
    assert annotated["qc_manual_merge"] is True
    assert annotated["qc_exclusion_reason"] == "manual_merge"
    assert annotated["qc_excluded_default"] is True
    assert annotated["include_default"] is False
