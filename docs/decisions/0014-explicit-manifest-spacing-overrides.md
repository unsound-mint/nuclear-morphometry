# 0014 - Explicit manifest spacing overrides for metadata-stripped TIFFs

## Status

Accepted.

## Context

Some TIFF exports can genuinely lose physical calibration. A manifest fallback is needed for
those files. The current thesis MetaMorph TIFFs are not in that category: their proprietary
page metadata is now read directly as documented in decision 0015.

## Decision

The manifest may include optional `spacing_x_um`, `spacing_y_um`, and `spacing_z_um` columns.
These values must be transcribed from an authoritative microscope acquisition record; they
are never inferred. X and Y must be supplied together and every supplied value must be
positive. When embedded calibration is also present, the two sources must agree within
`rel_tol=1e-6` and `abs_tol=1e-9` micrometers or the field fails rather than choosing one.

The copied manifest is already preserved and hashed with each run, so the override becomes
part of the reproducibility record without altering the raw microscopy file.

## Consequences

- Metadata-stripped TIFFs can be analyzed once their real acquisition calibration is known.
- A 3D row still requires all three values; a 2D row requires X and Y.
- Conflicting source and manifest metadata is an actionable error, not a silent override.
- Raw image files remain immutable.
