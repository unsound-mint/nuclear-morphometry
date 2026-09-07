# 0013 - Require matching physical spacing across image channels

## Status

Accepted.

## Context

Additional-channel measurements reuse the Hoechst-derived nuclear labels. Matching array
shape alone is insufficient evidence that the arrays share one physical coordinate system:
two TIFF/CZI sources can have identical pixel dimensions but different X/Y/Z calibration.
Using Hoechst spacing to construct a Lamin A/C shell or MitoTracker ring for such a channel
would silently change the requested physical width.

## Decision

Before measuring any additional channel, require its axes, array shape, and X/Y/Z physical
spacing to match the Hoechst volume. Spacing comparisons use `rel_tol=1e-6` and
`abs_tol=1e-9` micrometers to allow insignificant metadata serialization noise. A mismatch
fails that field with an actionable error; the pipeline never resamples or registers it.

## Consequences

- Hoechst masks are reused only when physical calibration is consistent.
- A mismatch identifies either incorrect acquisition metadata or a registration/resampling
  requirement that must be resolved and documented outside v1.
- No measurement definition or accepted Hoechst segmentation is changed.
