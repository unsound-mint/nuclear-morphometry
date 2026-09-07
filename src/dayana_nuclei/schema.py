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

from typing import Literal

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
BASE_NUCLEI_TABLE_SCHEMA: dict[str, pl.DataType | type[pl.DataType]] = {
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
    # 3D-only morphology (spec section 17). Null on every row of a 2D-mode run;
    # "extent" above is shared between 2D and 3D (same concept, mode-exclusive).
    "volume_voxels": pl.Int64,
    "volume_um3": pl.Float64,
    "surface_area_um2": pl.Float64,
    "z_depth_slices": pl.Int64,
    "z_depth_um": pl.Float64,
    "axis_major_um": pl.Float64,
    "axis_intermediate_um": pl.Float64,
    "axis_minor_um": pl.Float64,
    "sphericity": pl.Float64,
    "touches_border": pl.Boolean,
    "qc_border": pl.Boolean,
    "qc_manual_debris": pl.Boolean,
    "qc_manual_merge": pl.Boolean,
    "qc_manual_split": pl.Boolean,
    "qc_manual_other": pl.Boolean,
    "qc_excluded_default": pl.Boolean,
    "qc_exclusion_reason": pl.Utf8,
}

# --- Intensity (spec section 18). Flat, unprefixed column names -- this is
# always the Hoechst channel; additional channels (spec 25, Phase 8) use the
# prefixed scheme below instead, so these names never collide with them. ---
INTENSITY_COLUMNS: tuple[str, ...] = (
    "mean_intensity",
    "median_intensity",
    "integrated_intensity",
    "min_intensity",
    "max_intensity",
    "std_intensity",
)

# --- Texture (spec section 19.4): column names are {property}_d{distance_px}
# in legacy pixel-distance mode, or {property}_d{distance_um formatted}um in
# physical-scale mode (see measurements/texture.py::format_um_distance_label
# and docs/decisions/0012), dynamic on the run's configured distances -- see
# nuclei_table_schema(). ---
_TEXTURE_PROPERTIES: tuple[str, ...] = (
    "contrast",
    "homogeneity",
    "correlation",
    "energy",
    "entropy",
)

# --- Additional channels (spec section 25, Phase 8). Every column is
# {prefix}_{suffix}; "hoechst" is a reserved prefix (config.py rejects it)
# since Hoechst's own columns above are deliberately unprefixed. ---
ChannelKind = Literal["nuclear", "lamin_shell_core", "mitotracker_rings"]

# H3K9Ac / H3K9me3 (spec 25.1/25.2): plain nuclear intensity, same shape as
# INTENSITY_COLUMNS, reusing measurements.intensity.measure_intensity.
NUCLEAR_CHANNEL_SUFFIXES: tuple[str, ...] = INTENSITY_COLUMNS

# Lamin A/C (spec 25.3): total/shell/core intensity + shell:core ratio, where
# shell/core are defined by a physically-calibrated distance-from-boundary
# erosion (measurements/lamin.py). Note: {prefix}_total_mean_intensity is
# definitionally the same quantity as the unprefixed mean_intensity would be
# for this channel -- not an independent measurement.
LAMIN_SHELL_CORE_SUFFIXES: tuple[str, ...] = (
    "total_mean_intensity",
    "shell_mean_intensity",
    "core_mean_intensity",
    "shell_core_ratio",
)

# MitoTracker (spec 25.4): mean intensity in a near and a far perinuclear
# ring plus their ratio, with pixels assigned to their nearest nucleus to
# avoid double-counting overlapping perinuclear regions (measurements/spatial.py).
# The two *_touches_border columns flag a ring clipped by the field of view
# (analogous to qc_border for the nucleus itself) -- a clipped ring's mean is
# still reported, never dropped, but the flag lets analysis exclude or
# stratify by it rather than silently averaging a truncated sample.
MITOTRACKER_RING_SUFFIXES: tuple[str, ...] = (
    "near_ring_mean_intensity",
    "far_ring_mean_intensity",
    "perinuclear_enrichment_ratio",
)
MITOTRACKER_RING_BOOLEAN_SUFFIXES: tuple[str, ...] = (
    "near_ring_touches_border",
    "far_ring_touches_border",
)

_CHANNEL_KIND_SUFFIXES: dict[ChannelKind, tuple[str, ...]] = {
    "nuclear": NUCLEAR_CHANNEL_SUFFIXES,
    "lamin_shell_core": LAMIN_SHELL_CORE_SUFFIXES,
    "mitotracker_rings": MITOTRACKER_RING_SUFFIXES,
}
_CHANNEL_KIND_BOOLEAN_SUFFIXES: dict[ChannelKind, tuple[str, ...]] = {
    "nuclear": (),
    "lamin_shell_core": (),
    "mitotracker_rings": MITOTRACKER_RING_BOOLEAN_SUFFIXES,
}

# --- 2D radial intensity distribution (spec section 20). Column names are
# radial_bin{i}_{mean_intensity,frac_intensity,frac_pixels}, dynamic on the
# run's configured bin count -- see nuclei_table_schema() and
# measurements/radial.py::radial_bin_columns (kept in sync by
# tests/unit/test_radial.py's schema-consistency check, matching the
# existing texture-property duplication convention in this module). ---
_RADIAL_BIN_PROPERTIES: tuple[str, ...] = ("mean_intensity", "frac_intensity", "frac_pixels")


def nuclei_table_schema(
    *,
    include_intensity: bool,
    include_texture: bool,
    texture_distances_px: tuple[int, ...] = (),
    texture_distance_labels_um: tuple[str, ...] = (),
    additional_channels: tuple[tuple[str, ChannelKind], ...] = (),
    radial_bins: int = 0,
) -> dict[str, pl.DataType | type[pl.DataType]]:
    """The full per-field nuclei schema for one run, resolved once from its config.

    Texture column names depend on the run's configured pixel distances
    (spec 19.4), additional-channel column names depend on the run's
    configured channels/prefixes (spec 25), and radial-distribution column
    names depend on the run's configured bin count (spec 20) -- none of
    these can be part of the static BASE_NUCLEI_TABLE_SCHEMA. Call this once
    per run (config is fixed for the run's duration) and reuse the same
    dict for every field -- that is what guarantees every field's partial
    table shares one schema and can be concatenated safely (see
    docs/decisions/0004-parquet-as-canonical-table.md), including a field
    where a given additional channel was not present.

    ``additional_channels`` is a sequence of (prefix, kind) pairs, e.g.
    ``[("h3k9ac", "nuclear"), ("laminac", "lamin_shell_core")]``.
    ``radial_bins`` is 0 when radial distribution is disabled (the default);
    a positive value adds that many bins' worth of columns.

    ``texture_distance_labels_um``, when non-empty, selects physical-scale
    texture mode (spec 19.4) and is used in place of ``texture_distances_px``
    for column naming -- see
    ``measurements/texture.py::format_um_distance_label`` and
    ``docs/decisions/0012-texture-um-distance-column-naming.md`` for why
    physical-scale columns must be named by the configured um value, not the
    per-field resolved pixel distance. The two are mutually exclusive,
    matching ``config.MeasurementsConfig``'s own validation.
    """
    result = dict(BASE_NUCLEI_TABLE_SCHEMA)
    if include_intensity:
        for column in INTENSITY_COLUMNS:
            result[column] = pl.Float64
    if include_texture:
        distance_labels = (
            texture_distance_labels_um
            if texture_distance_labels_um
            else tuple(str(d) for d in texture_distances_px)
        )
        for label in distance_labels:
            for prop in _TEXTURE_PROPERTIES:
                result[f"{prop}_d{label}"] = pl.Float64
    for prefix, kind in additional_channels:
        for suffix in _CHANNEL_KIND_SUFFIXES[kind]:
            result[f"{prefix}_{suffix}"] = pl.Float64
        for suffix in _CHANNEL_KIND_BOOLEAN_SUFFIXES[kind]:
            result[f"{prefix}_{suffix}"] = pl.Boolean
    for radial_bin in range(radial_bins):
        for prop in _RADIAL_BIN_PROPERTIES:
            result[f"radial_bin{radial_bin}_{prop}"] = pl.Float64
    return result


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
