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
