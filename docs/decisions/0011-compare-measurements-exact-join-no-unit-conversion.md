# 0011: compare-measurements joins by exact key, does no unit conversion

## Status

Accepted.

## Context

Spec section 16.1 requires a tool to compare this pipeline's measurements against a legacy
CellProfiler export "on the same masks," to isolate measurement-definition differences from
segmentation differences: `compare_measurements.py` + the `compare-measurements` CLI (spec
28).

Two scope questions came up while implementing it that spec 16.1 does not answer directly,
and spec 16.1 itself says what to do when that happens: "If exact CellProfiler semantics are
not known for a feature, do not claim exact parity. Document the difference."

## Decisions

**1. Object matching is exact join-key equality, never fuzzy or nearest-object matching.**

`compare_measurements()` joins `ours` and `reference` on `mapping.ours_join_keys` /
`mapping.reference_join_keys` (default `image_id`/`object_number` on both sides) with an
exact-equality inner join. It does not attempt to match objects by mask overlap, centroid
proximity, or any other heuristic.

This follows directly from spec 16.1's own precondition: the two tables must already
describe the *same masks*. If the masks are genuinely the same, `object_number` (this
pipeline's `skimage.measure.regionprops` label numbering) and CellProfiler's `ObjectNumber`
must already agree row-for-row, once CellProfiler's own image identifier is translated into
this pipeline's `image_id` (a translation spec 16.1 does not specify how to do generically,
since CellProfiler's per-project `ImageNumber` scheme is pipeline-configuration-dependent).
The mapping file's `[join]` table lets the two sides use differently-named key columns, but
producing a `reference` table whose join-key values already correspond 1:1 to `ours` is the
user's responsibility — this tool cannot discover that correspondence on its own without
either mask access (which would make it a segmentation-validation tool, not a
measurement-comparison one) or an unreliable heuristic that would then need its own
validation. Duplicate join-key values on either side are rejected outright (a concatenated
multi-run `nuclei.parquet` must be filtered to one `run_id` first) rather than silently
producing a cartesian-product join that would corrupt every statistic while still reporting
a plausible-looking object count.

**2. No unit or scale conversion between mapped columns.**

`compare_measurements()` compares the two mapped columns' raw numeric values directly (both
cast to `Float64`). It has no concept of a per-column scale factor or unit.

CellProfiler's `AreaShape` module reports several length measurements
(`MajorAxisLength`, `MinorAxisLength`, and others) in pixels by default; this pipeline
exports only the physically-calibrated micron form (`major_axis_um`, `minor_axis_um` —
`Nucleus2DMorphology` has no `*_axis_px` column). Mapping `AreaShape_MajorAxisLength` to
`major_axis_um` directly would silently compare a pixel count against a micron value and
report a large, meaningless "difference" as if it were a real measurement discrepancy.
`configs/cellprofiler_mapping.toml` therefore does not map these two columns, with a comment
explaining why, rather than mapping them incorrectly or adding unit-conversion machinery
before there is a concrete second unit pair that needs it.

## Consequences

- A `--reference` export whose join-key column doesn't already correspond to this pipeline's
  `image_id` will produce a misleadingly low `n_matched_objects` (every row falls into
  `n_only_in_ours`/`n_only_in_reference`) rather than an error — this is expected: it is the
  intended signal that the "same masks" precondition has not actually been met yet, not a
  detected failure the tool can name specifically.
- If area/perimeter/eccentricity/solidity/extent parity looks acceptable but axis-length
  parity is later needed, the fix is a pixel-unit column added to `Nucleus2DMorphology`
  (documented in `docs/measurement-dictionary.md` per this project's definition of done),
  not a unit-conversion parameter bolted onto the comparison tool.
- Duplicate-key rejection means `compare-measurements` cannot be pointed at a run's full
  concatenated `nuclei.parquet` across multiple `run_id`s without pre-filtering; this is
  intentional (there is no meaningful "the same masks" claim across multiple runs).
