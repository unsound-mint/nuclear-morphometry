from pathlib import Path

import polars as pl

from dayana_nuclei.export import (
    finalize_tables,
    partial_fields_path,
    partial_nuclei_path,
    prepare_analysis,
    write_partial_table,
)
from dayana_nuclei.schema import compute_include_default, include_default_expr


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
