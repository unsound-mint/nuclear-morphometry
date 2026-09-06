# 0003 - Physical units, never assumed isotropic spacing

## Status

Accepted. The invariant is implemented for the parts of the pipeline that
exist today (I/O, 2D morphology); 3D-specific measurements (volume, surface
area, principal axes, sphericity -- spec section 17) are implemented in a
later phase and will be built directly on top of this same
`PhysicalSpacing` model, not a separate one.

## Context

The legacy CellProfiler project ran with `Process as 3D = No` and an
effective `1/1/1` relative X/Y/Z spacing (spec section 3). Confocal Z-stacks
were therefore never represented as physically calibrated volumes. Spec
section 7.3 makes this a hard requirement to fix: "Never silently assume
`1/1/1` voxel spacing," and 3D analysis on a source with unknown Z spacing
must fail loudly rather than proceed with a fabricated value.

## Decision

- `models.PhysicalSpacing` has `z_um: float | None` -- `None` is a valid,
  explicit state (a single 2D plane), never silently coerced to `1.0`.
  `PhysicalSpacing.require_z()` is the single place that turns "Z spacing
  missing" into the actionable error text used everywhere 3D analysis needs
  it (spec section 49 style: what's wrong, how to inspect the file, what to
  fix).
- `io.images.load_channel_volume` enforces this at load time: 3D mode
  raises if X, Y, or Z spacing is unavailable; 2D mode raises if X/Y
  spacing is unavailable for calibrated output. Neither path ever
  substitutes a default.
- A real, non-obvious risk was found and closed at the reader-library level,
  not just our own code: `bioio-tifffile` can report a fabricated `1.0
  um/pixel` for TIFFs with no real resolution tag, indistinguishable from a
  genuine 1.0 um/pixel calibration unless the underlying `ResolutionUnit`
  tag is checked directly. See
  `docs/decisions/0006-tiff-uncalibrated-resolution-detection.md`. Without
  that check, "never assume 1/1/1" would have been satisfied in our code
  while still being silently violated by the library underneath it.
- All final morphology output columns are physical-unit-first
  (`area_um2`, `perimeter_um`, ...) with the pixel-space values retained
  alongside them for audit/debugging, per spec section 16 -- never the
  reverse.
- Physical unit naming uses ASCII (`um`, `um2`, `um3`), never Unicode `µm`,
  in column names (spec section 47); human-facing docs may use `µm`.

## Consequences

- A dataset with genuinely unknown or unrecorded calibration cannot be
  analyzed calibrated -- this is intentional friction, not a bug to route
  around. The fix is to record real calibration, not to add a fallback.
- When 3D morphology (spec section 17) is implemented, `voxel_volume_um3`,
  marching-cubes surface area, and the principal-axis computation must all
  consume `PhysicalSpacing` the same way -- through `require_z()`-style
  hard failure, never a bare `spacing.z_um or 1.0`-style default.
