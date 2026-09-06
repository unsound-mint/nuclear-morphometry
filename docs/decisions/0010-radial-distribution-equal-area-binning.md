# 0010: radial distribution bins by equal pixel count, not equal normalized-distance width

## Status

Accepted.

## Context

`measurements/radial.py` (spec section 20) buckets each nucleus's pixels into `radial_bins`
concentric groups by distance from the object's boundary, without CellProfiler parity (spec
20 explicitly requires a from-scratch, documented definition).

The first implementation defined a per-pixel `normalized_distance_from_edge = 1 -
distance_to_edge / max_distance_to_edge` and split `[0, 1]` into `radial_bins` equal-width
ranges. This looked reasonable but was wrong: `distance_to_edge`'s maximum is attained by
only the one or few pixels on an object's medial axis, and for a convex shape the pixel
count at a given depth grows roughly linearly with distance from the boundary (the classic
"most of a disk's area is near its edge" fact). Equal-width bins on a nonlinear-density axis
are therefore radically unequal in pixel count.

Measured on a 20 px-radius disk with 5 bins: bin 0 (innermost) held ~4.6% of pixels, bin 4
(outermost) held ~34%. A "radial distribution across 5 bins" that puts 7x more pixels in the
last bin than the first does not mean what a reader (or a comparison to the legacy 5-bin
CellProfiler output) would expect, and would report numbers dominated by whichever bin
happens to be largest rather than genuine spatial structure.

Caught by `advisor()` review before this shipped past its first commit, then confirmed
empirically by directly measuring the bin populations on the disk fixture already used in
`tests/unit/test_radial.py`.

## Decision

Bin by **equal pixel count** (rank of `distance_to_edge`, split into `radial_bins` equal-size
groups), not equal-width ranges of a normalized distance:

1. Take each object's per-pixel `distance_to_edge` (same padded `distance_transform_edt` as
   before, unchanged).
2. Rank pixels ascending by distance (rank 0 = smallest distance = boundary-most; stable sort
   for deterministic tie-breaking).
3. Split the ascending ranks into `radial_bins` equal-size chunks; chunk 0 (smallest
   distances) maps to the *last* bin (touches the boundary), chunk `radial_bins - 1` (largest
   distances) maps to bin 0 (deepest interior) — preserving the original "bin 0 is innermost"
   convention.

This keeps every bin within a small tolerance of `1 / radial_bins` of the object's total
pixel count, for any shape, by construction — not just for a disk.

## Consequences

- Every number previously produced by `measure_radial_distribution_2d` changes (this was
  caught before the definition had any downstream consumer beyond its own tests and the
  pipeline wiring landed in the same commit, so there is no external data to migrate).
- A very small or shallow object can still leave the *innermost* bin(s) empty (fewer distinct
  depth ranks than `radial_bins`) — this is the expected direction now, not the outermost
  bin as an equal-width scheme would have implied. See
  `test_small_object_may_have_empty_inner_bins_reported_as_none`.
- `max_distance_to_edge` is no longer computed or divided by, which also removes a
  theoretical (untested, never actually reachable for a non-empty mask) division-by-zero
  path from the module.
