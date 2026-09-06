"""Legacy CellProfiler measurement-parity comparison (spec sections 16.1, 28).

The key scientific technique this backs (spec 16.1): compare measurements on
the **same masks** before comparing end-to-end segmentation pipelines, so
segmentation differences never contaminate measurement-definition parity
testing. This module assumes that precondition already holds -- it joins
"ours" (this pipeline's ``nuclei.parquet``) against a legacy CellProfiler
export by exact join-key match (default ``image_id``/``object_number``) and
does not attempt fuzzy or nearest-object matching. If the reference export's
own identifiers don't already correspond 1:1 (e.g. CellProfiler's numeric
``ImageNumber``), the mapping file's ``[join]`` table can rename them, but
producing a compatible ``image_id`` value in the reference export is the
user's responsibility -- this is a deliberate scope limit, not an oversight
(spec 16.1: "if exact CellProfiler semantics are not known ... document the
difference", which applies equally to a difference in what counts as "the
same object"). Also does no unit/scale conversion between mapped columns.
See docs/decisions/0011 for both decisions and their consequences.
"""

from __future__ import annotations

import tomllib
from pathlib import Path
from typing import Any

import numpy as np
import polars as pl
from pydantic import BaseModel, ConfigDict, model_validator


class MeasurementMapping(BaseModel):
    """Reference-column -> ours-column measurement mapping, plus the join
    keys used to match objects across the two tables (spec 16.1's
    ``--mapping`` TOML)."""

    model_config = ConfigDict(frozen=True)

    reference_join_keys: tuple[str, ...] = ("image_id", "object_number")
    ours_join_keys: tuple[str, ...] = ("image_id", "object_number")
    columns: dict[str, str]

    @model_validator(mode="after")
    def _check_shape(self) -> MeasurementMapping:
        if len(self.reference_join_keys) != len(self.ours_join_keys):
            raise ValueError(
                f"mapping.join.reference ({self.reference_join_keys}) and "
                f"mapping.join.ours ({self.ours_join_keys}) must name the same "
                f"number of columns, paired positionally."
            )
        if not self.reference_join_keys:
            raise ValueError("mapping.join must name at least one join key.")
        if not self.columns:
            raise ValueError("mapping.columns must map at least one measurement.")
        return self


def load_measurement_mapping(path: Path) -> MeasurementMapping:
    """Load a ``[join]``/``[columns]`` TOML mapping file (see
    ``configs/cellprofiler_mapping.toml`` for the expected shape)."""
    raw = path.read_bytes()
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Failed to parse mapping {path}: {exc}") from exc

    join = data.get("join", {})
    kwargs: dict[str, Any] = {"columns": data.get("columns", {})}
    if "reference" in join:
        kwargs["reference_join_keys"] = tuple(join["reference"])
    if "ours" in join:
        kwargs["ours_join_keys"] = tuple(join["ours"])
    return MeasurementMapping.model_validate(kwargs)


class ColumnComparison(BaseModel):
    """Per-measurement agreement between reference and ours, over the
    matched objects that have a non-null, finite value on both sides for
    this specific column (spec 16.1) -- a NaN or inf value (e.g. from a
    legacy divide-by-zero) is excluded rather than corrupting every
    statistic and producing invalid JSON. ``None`` fields follow this
    project's undefined-not-fabricated-zero convention (see
    docs/decisions/0008 and measurements/lamin.py for the same rule applied
    elsewhere)."""

    model_config = ConfigDict(frozen=True)

    reference_column: str
    ours_column: str
    n_both_finite: int
    mean_absolute_difference: float | None
    median_absolute_difference: float | None
    max_absolute_difference: float | None
    mean_relative_difference: float | None
    pearson_r: float | None


class MeasurementComparisonReport(BaseModel):
    """Full comparison result; JSON-serializable (spec 16.1's
    ``compare-measurements`` output)."""

    model_config = ConfigDict(frozen=True)

    ours_path: str
    reference_path: str
    mapping_path: str
    n_ours_objects: int
    n_reference_objects: int
    n_matched_objects: int
    n_only_in_ours: int
    n_only_in_reference: int
    columns: tuple[ColumnComparison, ...]


def _read_table(path: Path) -> pl.DataFrame:
    if path.suffix == ".parquet":
        return pl.read_parquet(path)
    if path.suffix == ".csv":
        return pl.read_csv(path)
    raise ValueError(
        f"Unsupported table format {path.suffix!r} for {path}; expected .parquet or .csv."
    )


def _compare_column(
    joined: pl.DataFrame, reference_select_column: str, ours_column: str, *, reference_column: str
) -> ColumnComparison:
    pair = (
        joined.select(
            pl.col(reference_select_column).cast(pl.Float64).alias("reference"),
            pl.col(ours_column).cast(pl.Float64).alias("ours"),
        )
        .drop_nulls()
        .filter(pl.col("reference").is_finite() & pl.col("ours").is_finite())
    )

    n = pair.height
    if n == 0:
        return ColumnComparison(
            reference_column=reference_column,
            ours_column=ours_column,
            n_both_finite=0,
            mean_absolute_difference=None,
            median_absolute_difference=None,
            max_absolute_difference=None,
            mean_relative_difference=None,
            pearson_r=None,
        )

    reference = pair["reference"].to_numpy()
    ours = pair["ours"].to_numpy()
    abs_diff = np.abs(ours - reference)

    nonzero = reference != 0.0
    mean_relative_difference = (
        float(np.mean(abs_diff[nonzero] / np.abs(reference[nonzero]))) if nonzero.any() else None
    )

    pearson_r: float | None = None
    if n >= 2 and np.std(reference) > 0.0 and np.std(ours) > 0.0:
        pearson_r = float(np.corrcoef(reference, ours)[0, 1])

    return ColumnComparison(
        reference_column=reference_column,
        ours_column=ours_column,
        n_both_finite=n,
        mean_absolute_difference=float(np.mean(abs_diff)),
        median_absolute_difference=float(np.median(abs_diff)),
        max_absolute_difference=float(np.max(abs_diff)),
        mean_relative_difference=mean_relative_difference,
        pearson_r=pearson_r,
    )


def compare_measurements(
    ours: pl.DataFrame,
    reference: pl.DataFrame,
    mapping: MeasurementMapping,
    *,
    ours_path: str = "<in-memory>",
    reference_path: str = "<in-memory>",
    mapping_path: str = "<in-memory>",
) -> MeasurementComparisonReport:
    """Compare ``ours`` against ``reference`` for every column pair in
    ``mapping.columns``, joining on ``mapping.*_join_keys``. Both inputs
    must already refer to the same objects on the same masks (spec 16.1)."""
    missing_reference_keys = [c for c in mapping.reference_join_keys if c not in reference.columns]
    if missing_reference_keys:
        raise ValueError(
            f"mapping.join.reference names column(s) {missing_reference_keys} not present in "
            f"{reference_path} (columns: {reference.columns})."
        )
    missing_ours_keys = [c for c in mapping.ours_join_keys if c not in ours.columns]
    if missing_ours_keys:
        raise ValueError(
            f"mapping.join.ours names column(s) {missing_ours_keys} not present in "
            f"{ours_path} (columns: {ours.columns})."
        )
    missing_reference = [c for c in mapping.columns if c not in reference.columns]
    if missing_reference:
        raise ValueError(
            f"mapping.columns names reference column(s) {missing_reference} not present in "
            f"{reference_path} (columns: {reference.columns})."
        )
    missing_ours = [c for c in mapping.columns.values() if c not in ours.columns]
    if missing_ours:
        raise ValueError(
            f"mapping.columns names ours column(s) {missing_ours} not present in "
            f"{ours_path} (columns: {ours.columns})."
        )

    reference_renamed = reference.rename(
        dict(zip(mapping.reference_join_keys, mapping.ours_join_keys, strict=True))
    )

    ours_keys = ours.select(mapping.ours_join_keys).unique()
    reference_keys = reference_renamed.select(mapping.ours_join_keys).unique()
    if ours.height != ours_keys.height:
        raise ValueError(
            f"--ours ({ours_path}) has duplicate values for join key(s) "
            f"{mapping.ours_join_keys}: {ours.height} rows but {ours_keys.height} distinct "
            f"keys. Each row must identify one object on one mask; deduplicate before "
            f"comparing (e.g. filter to a single run_id if this is a concatenated "
            f"nuclei.parquet)."
        )
    if reference.height != reference_keys.height:
        raise ValueError(
            f"--reference ({reference_path}) has duplicate values for join key(s) "
            f"{mapping.reference_join_keys}: {reference.height} rows but "
            f"{reference_keys.height} distinct keys."
        )
    matched_keys = ours_keys.join(reference_keys, on=list(mapping.ours_join_keys), how="inner")

    # Rename mapped reference columns under a unique prefix before joining,
    # so a reference column that happens to share ours' name (e.g. both
    # exported "eccentricity") can never collide with -- or worse, silently
    # shadow -- the real ours column of that name.
    reference_prepped = reference_renamed.rename(
        {col: f"__reference__{col}" for col in mapping.columns}
    )
    joined = ours.join(reference_prepped, on=list(mapping.ours_join_keys), how="inner")

    columns = tuple(
        _compare_column(
            joined,
            f"__reference__{reference_column}",
            ours_column,
            reference_column=reference_column,
        )
        for reference_column, ours_column in mapping.columns.items()
    )

    return MeasurementComparisonReport(
        ours_path=ours_path,
        reference_path=reference_path,
        mapping_path=mapping_path,
        n_ours_objects=ours_keys.height,
        n_reference_objects=reference_keys.height,
        n_matched_objects=matched_keys.height,
        n_only_in_ours=ours_keys.height - matched_keys.height,
        n_only_in_reference=reference_keys.height - matched_keys.height,
        columns=columns,
    )


def compare_measurements_from_files(
    ours_path: Path, reference_path: Path, mapping_path: Path
) -> MeasurementComparisonReport:
    """Load ``ours``/``reference`` tables (``.parquet`` or ``.csv``) and a
    mapping TOML, then compare (spec 16.1/28's ``compare-measurements`` CLI)."""
    ours = _read_table(ours_path)
    reference = _read_table(reference_path)
    mapping = load_measurement_mapping(mapping_path)
    return compare_measurements(
        ours,
        reference,
        mapping,
        ours_path=str(ours_path),
        reference_path=str(reference_path),
        mapping_path=str(mapping_path),
    )
