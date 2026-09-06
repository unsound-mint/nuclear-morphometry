# 0009 - MitoTracker perinuclear measurement: exactly two rings, nearest-nucleus disambiguation

## Status

Accepted.

## Context

Spec section 25.4 requires a perinuclear spatial measurement for
MitoTracker based on distance from the nuclear boundary, explicitly
permits "a clean v1 implementation" with "configured outside ring(s)
around each nucleus," and requires that overlapping perinuclear regions
from neighboring nuclei not double-count pixels, "assigning each pixel to
its nearest nucleus or ... another explicitly documented disambiguation
strategy."

Two scope decisions were made when implementing `measurements/spatial.py`
that narrow the spec's "ring(s)" (plural, unspecified count) to something
concrete and testable.

## Decision

1. **Exactly two configured bands**: a near ring and a far reference ring,
   each an explicit `(start, end)` distance pair in micrometers
   (`near_ring_um`, `far_ring_um`), ordered and non-overlapping. This is
   enough to compute spec 25.4's required "perinuclear/farther-region
   enrichment ratio" (near mean / far mean) without introducing an
   arbitrary-length ring-list configuration surface, per-ring naming
   scheme, or N-way ratio ambiguity. If a future analysis genuinely needs
   more than two bands, extending `AdditionalChannelConfig` and
   `measure_perinuclear_rings` to accept a list is a compatible addition,
   not a redesign.

2. **Nearest-nucleus disambiguation via a labeled Euclidean distance
   transform**: `scipy.ndimage.distance_transform_edt(background,
   sampling=spacing, return_indices=True)` gives, for every background
   pixel, both its physical distance to the nearest labeled pixel and that
   pixel's coordinates -- looking up `labels` at those coordinates assigns
   every background pixel to exactly one nucleus (a Voronoi tessellation
   by nearest labeled object). This was verified empirically (not assumed
   from documentation) against small synthetic label arrays before use.
   Every ring pixel is counted for exactly one nucleus; overlapping
   perinuclear regions from neighboring nuclei never double-count a pixel.

3. **Ring truncation is flagged, not hidden**: a ring that touches the
   image's edge is likely clipped by the field of view, biasing its mean
   toward whatever fraction of the ring happened to be captured. Rather
   than silently averaging a partial sample, `near_ring_touches_border`/
   `far_ring_touches_border` boolean columns are reported alongside the
   mean (never dropping the row), analogous to `qc_border` for the
   nucleus itself (spec 22.1). The mean is still computed and kept --
   downstream analysis decides whether to exclude or stratify by it.

## Consequences

- `NucleusPerinuclearIntensity` has a fixed 5-field shape (near/far mean,
  ratio, near/far border-touch flags), not a dynamic per-ring list --
  simpler schema (`schema.MITOTRACKER_RING_SUFFIXES` /
  `MITOTRACKER_RING_BOOLEAN_SUFFIXES`), simpler tests, and no ambiguity
  about which two rings a ratio compares.
- Cost scales with `n_objects x image_size` (a full-image boolean mask is
  recomputed per object) -- acceptable at validation/thesis field scale;
  not benchmarked against a much larger dataset. Revisit only with
  profiling evidence (AGENTS.md: profile before optimizing), not
  speculatively.
- A nucleus whose ring is entirely absent (e.g. crowded out completely by
  a closer neighbor) reports `None` for that ring's mean and the ratio,
  not a fabricated value -- consistent with this project's
  undefined-is-`None` convention elsewhere (segmentation validation's
  precision/recall, Lamin A/C's core-mean).
