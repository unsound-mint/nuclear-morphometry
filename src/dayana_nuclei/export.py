"""Canonical Parquet/CSV outputs and per-field atomic writers.

Spec sections 26 (output data model), 30.1 (atomic per-field commits), 33
(resume must not duplicate rows). Each field owns exactly one partial-table
file, keyed by image_id; a retried field simply overwrites its own file, so
resume can never produce duplicate rows in the finalized tables.
"""

from __future__ import annotations

from pathlib import Path

import polars as pl

from dayana_nuclei.qc.annotations import apply_annotations_to_nuclei
from dayana_nuclei.schema import include_default_expr


def partial_nuclei_path(run_dir: Path, image_id: str) -> Path:
    return run_dir / "partial" / "nuclei" / f"{image_id}.parquet"


def partial_fields_path(run_dir: Path, image_id: str) -> Path:
    return run_dir / "partial" / "fields" / f"{image_id}.parquet"


def write_partial_table(df: pl.DataFrame, path: Path) -> None:
    """Atomically (temp file + rename) write one field's rows."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    df.write_parquet(tmp_path)
    tmp_path.replace(path)


def finalize_tables(run_dir: Path, *, write_csv: bool) -> tuple[Path, Path]:
    """Rebuild nuclei.parquet / fields.parquet from all committed partial files.

    Idempotent and safe to call repeatedly, including after a resume.
    """
    nuclei_parts = sorted((run_dir / "partial" / "nuclei").glob("*.parquet"))
    fields_parts = sorted((run_dir / "partial" / "fields").glob("*.parquet"))

    nuclei_df = (
        pl.concat([pl.read_parquet(p) for p in nuclei_parts]) if nuclei_parts else pl.DataFrame()
    )
    fields_df = (
        pl.concat([pl.read_parquet(p) for p in fields_parts]) if fields_parts else pl.DataFrame()
    )

    nuclei_path = run_dir / "nuclei.parquet"
    fields_path = run_dir / "fields.parquet"
    nuclei_df.write_parquet(nuclei_path)
    fields_df.write_parquet(fields_path)

    if write_csv:
        nuclei_df.write_csv(run_dir / "nuclei.csv")

    return nuclei_path, fields_path


def prepare_analysis(run_dir: Path) -> Path:
    """Build the analysis-ready table (spec section 26.3).

    Does not aggregate or drop any nucleus row -- it only adds
    include_default, derived once from qc_excluded_default (schema.py).
    """
    nuclei_path = run_dir / "nuclei.parquet"
    if not nuclei_path.exists():
        raise FileNotFoundError(
            f"{nuclei_path} does not exist.\n\n"
            f"Run `dayana-nuclei run <config>` (or `resume` an in-progress run) "
            f"to completion before `prepare-analysis`."
        )
    raw_df = pl.read_parquet(nuclei_path)
    df = apply_annotations_to_nuclei(raw_df, run_dir).with_columns(include_default_expr())
    out_path = run_dir / "analysis_ready.parquet"
    df.write_parquet(out_path)
    return out_path
