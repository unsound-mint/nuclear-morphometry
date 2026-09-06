"""Canonical output column names and default-inclusion semantics.

Single source of truth for the per-nucleus and per-field table schemas
(spec sections 22, 26, 45). Every module that writes or reads these tables
must import the names from here rather than re-typing string literals, so
the schema cannot silently drift between the pipeline, QC, and export code.

Design decision (see docs/decisions/0005-include-default-semantics.md):
``qc_excluded_default`` is computed once, authoritatively, during the main
pipeline run (qc/flags.py) as each object is measured. ``include_default``
is derived from it -- ``include_default = not qc_excluded_default`` -- and
is only materialized later, on the analysis-ready table produced by
``prepare-analysis``. The raw ``nuclei.parquet`` therefore carries the
exclusion reason and flag but not the inverted boolean; the analysis-ready
table carries the boolean researchers actually filter on.
"""

from __future__ import annotations

import polars as pl

# --- Identity / metadata columns (present on nuclei.parquet and fields.parquet) ---
IDENTITY_COLUMNS: tuple[str, ...] = (
    "run_id",
    "image_id",
    "cell_line",
    "sort_id",
    "condition",
    "timepoint",
    "field",
    "acquisition_batch",
)

# Object-level identity, in addition to IDENTITY_COLUMNS.
OBJECT_IDENTITY_COLUMNS: tuple[str, ...] = ("object_number",)

# --- Provenance / linkage columns ---
LINKAGE_COLUMNS: tuple[str, ...] = (
    "source_path",
    "scene",
    "mask_path",
)

# --- Object-level QC columns (spec section 22) ---
QC_COLUMNS: tuple[str, ...] = (
    "qc_border",
    "qc_manual_debris",
    "qc_manual_merge",
    "qc_manual_split",
    "qc_manual_other",
    "qc_excluded_default",
    "qc_exclusion_reason",
)

# Exclusion reasons recognized by qc_exclusion_reason. Shape-based measurements
# (eccentricity, solidity, circularity, elongation, irregularity) must never
# appear here -- see spec sections 3.1, 22, 45 and AGENTS.md.
EXCLUSION_REASON_BORDER = "border"
EXCLUSION_REASON_MANUAL_DEBRIS = "manual_debris"
EXCLUSION_REASON_MANUAL_MERGE = "manual_merge"
EXCLUSION_REASON_MANUAL_SPLIT = "manual_split"
EXCLUSION_REASON_MANUAL_OTHER = "manual_other"
EXCLUSION_REASON_NONE = None

VALID_EXCLUSION_REASONS: frozenset[str] = frozenset(
    {
        EXCLUSION_REASON_BORDER,
        EXCLUSION_REASON_MANUAL_DEBRIS,
        EXCLUSION_REASON_MANUAL_MERGE,
        EXCLUSION_REASON_MANUAL_SPLIT,
        EXCLUSION_REASON_MANUAL_OTHER,
    }
)

# Nucleus global key (spec section 15): uniquely identifies an object across
# the whole experiment. object_number alone is only unique within an image.
NUCLEUS_KEY_COLUMNS: tuple[str, ...] = ("image_id", "object_number")

# Canonical dtype schema for one field's nuclei rows (spec section 26.1).
#
# Every writer of a per-field nuclei table (pipeline/analyze.py, and any
# future segmentation backend or measurement module) MUST construct its
# Polars DataFrame with this schema explicitly, including when there are
# zero objects in the field. Two failure modes this prevents:
#
# 1. A field with zero segmented objects produces an untyped/empty
#    pl.DataFrame(), which has zero columns. Concatenating that with a
#    normal field's ~35-column frame in export.finalize_tables raises a
#    schema error and breaks nuclei.parquet for the *entire run*, not just
#    the empty field.
# 2. Polars infers a column's dtype from its data. A field where every
#    object happens to share one qc_exclusion_reason (e.g. all None, or all
#    "border") can infer a different dtype (Null vs Utf8) than another
#    field with mixed values, again breaking the cross-field concat.
NUCLEI_TABLE_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
    "run_id": pl.Utf8,
    "image_id": pl.Utf8,
    "object_number": pl.Int64,
    "cell_line": pl.Utf8,
    "sort_id": pl.Utf8,
    "condition": pl.Utf8,
    "timepoint": pl.Utf8,
    "field": pl.Utf8,
    "acquisition_batch": pl.Utf8,
    "source_path": pl.Utf8,
    "scene": pl.Utf8,
    "mask_path": pl.Utf8,
    "area_px": pl.Float64,
    "area_um2": pl.Float64,
    "perimeter_px": pl.Float64,
    "perimeter_um": pl.Float64,
    "circularity": pl.Float64,
    "form_factor": pl.Float64,
    "solidity": pl.Float64,
    "eccentricity": pl.Float64,
    "major_axis_um": pl.Float64,
    "minor_axis_um": pl.Float64,
    "aspect_ratio": pl.Float64,
    "extent": pl.Float64,
    "centroid_row_px": pl.Float64,
    "centroid_col_px": pl.Float64,
    "centroid_y_um": pl.Float64,
    "centroid_x_um": pl.Float64,
    "touches_border": pl.Boolean,
    "qc_border": pl.Boolean,
    "qc_manual_debris": pl.Boolean,
    "qc_manual_merge": pl.Boolean,
    "qc_manual_split": pl.Boolean,
    "qc_manual_other": pl.Boolean,
    "qc_excluded_default": pl.Boolean,
    "qc_exclusion_reason": pl.Utf8,
}

# --- Analysis-ready table (prepare-analysis output, spec section 26.3) ---
ANALYSIS_READY_REQUIRED_COLUMNS: tuple[str, ...] = (
    "sort_id",
    "field",
    "image_id",
    "object_number",
    "include_default",
)


def compute_include_default(qc_excluded_default: bool) -> bool:
    """The single definition of include_default, used only by prepare-analysis.

    Raw nuclei.parquet never stores this column -- it stores
    qc_excluded_default and qc_exclusion_reason. prepare-analysis derives
    include_default from those at read time so there is exactly one place
    the inversion happens.
    """
    return not qc_excluded_default


def include_default_expr() -> pl.Expr:
    """Vectorized equivalent of compute_include_default, for use in a Polars pipeline."""
    return (~pl.col("qc_excluded_default")).alias("include_default")
