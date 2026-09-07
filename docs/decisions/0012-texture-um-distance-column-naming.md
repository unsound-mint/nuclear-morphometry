# 0012: physical-scale texture columns are named by the configured um value

## Status

Accepted.

## Context

`measurements.texture_distances_um` (spec 19.4's physical-scale texture mode) was config-valid
but deliberately not wired into `run_pipeline`: converting a um distance to a pixel distance
requires a per-field X/Y calibration, and doing that independently per field risked each field
resolving a different pixel distance for the same configured um value -- and `texture.py`
named its output columns directly from the pixel distance (`{property}_d{distance_px}`), so
two fields disagreeing on the resolved pixel distance would disagree on column names, which
would break the run's single `nuclei.parquet` schema (see `docs/decisions/0004`).

## Decision

Physical-scale texture columns are named by the *configured* um value (e.g. `contrast_d2um`,
`entropy_d1p5um`), never by the per-field resolved pixel distance. `measure_texture_2d` gained
an optional `distance_labels` parameter that overrides the `d{distance_px}` suffix; when unset
(legacy pixel-distance mode) behavior is unchanged. `format_um_distance_label` (in
`measurements/texture.py`) formats a um value into an identifier-safe suffix, and
`pipeline.analyze.texture_distance_labels_um` derives the run-wide label tuple from config,
shared by `nuclei_table_schema`'s `texture_distance_labels_um` parameter so every field's
partial table is built against the same schema regardless of that field's own calibration.

Per-field pixel conversion still uses the existing `um_distances_to_pixels`, which already
fails loudly (spec 7.3) if a configured um distance rounds to less than 1 pixel at that
field's calibration. Additionally, since a single scalar pixel distance is inherently
isotropic, `_process_field` now requires `x_um == y_um` for the field being measured when
physical-scale mode is active, and raises a named error naming the field otherwise --
converting to "the" pixel distance is ambiguous for anisotropic pixels, and this pipeline
does not guess which axis to use.

## Consequences

- Two fields with slightly different X/Y calibration can end up sampling the GLCM at a
  slightly different actual pixel distance for the same configured um value (e.g. 2.0 um
  rounds to 10px at one field's calibration and 11px at another's), while their output columns
  carry the same name (`contrast_d2um`) and represent the same *requested* physical distance.
  This is an accepted, documented approximation of the same kind as the marching-cubes surface
  smoothing tradeoff in `docs/decisions/0007` -- the alternative (naming by resolved pixel
  distance) is not merely cosmetically different, it silently breaks cross-field concatenation.
- A field whose calibration has non-square X/Y pixels cannot use
  `measurements.texture_distances_um` at all; `measurements.texture_distances_px` remains
  available for anisotropic-pixel data (it already carries the analogous, pre-existing
  implicit assumption that "distance in pixels" means the same thing in X and Y).
- `measurements.texture_distances_px` and `measurements.texture_distances_um` remain mutually
  exclusive per run (`config.MeasurementsConfig`'s existing validator); a run cannot mix both
  naming schemes.
