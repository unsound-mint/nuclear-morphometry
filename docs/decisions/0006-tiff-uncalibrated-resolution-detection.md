# 0006 - Detecting fabricated 1.0 um/pixel calibration from bioio-tifffile

## Status

Accepted.

## Context

Spec section 7.3/12/49 requires the pipeline to never silently assume `1/1/1`
voxel spacing. During implementation of `io/metadata.py` and `io/images.py`
we found a case where the *reader library itself*, not our code, introduces
exactly that risk.

`bioio-tifffile` computes `BioImage.physical_pixel_sizes` from the TIFF's
standard `XResolution`/`YResolution`/`ResolutionUnit` tags. `tifffile`'s
writer (and many third-party TIFF writers) default `resolution=(1, 1)` with
`ResolutionUnit = NONE` when no real calibration is supplied. `bioio-tifffile`
converts this default into `x_um = y_um = 1.0` -- a value that is bit-for-bit
identical to a genuine "1 micron per pixel" acquisition and carries no
distinguishing marker at the `physical_pixel_sizes` level.

This is a real risk for this project: plain (non-CZI, non-ImageJ) TIFF
exports from ad hoc conversion tools, or synthetic/test TIFFs, will silently
report calibrated 1.0 um/pixel spacing that is actually "no calibration was
ever recorded."

ImageJ-format TIFFs are not affected the same way: `bioio-tifffile` reads
their calibration from the `imagej_metadata['unit']`/`['spacing']` keys
rather than from `ResolutionUnit`, and ImageJ's own writer sets
`ResolutionUnit = NONE` unconditionally regardless of whether real
calibration was provided via those metadata keys -- so checking
`ResolutionUnit` for ImageJ files would produce false positives.

## Decision

`io/metadata.read_physical_spacing()` (shared by `inspect_image` and
`io.images.load_channel_volume`) additionally inspects the TIFF's own tags
via `tifffile.TiffFile(path).series[i].keyframe.tags["ResolutionUnit"]` when
the active reader is `bioio_tifffile` and a physical size was returned:

- If the file `is_imagej`, trust `physical_pixel_sizes` as-is (ImageJ's own
  metadata keys are the calibration source, not `ResolutionUnit`).
- Otherwise, if `ResolutionUnit` is absent or equals `NONE` (tifffile's
  `RESUNIT.NONE = 1`), treat `x_um`/`y_um` as unknown (`None`) rather than
  trusting the fabricated `1.0`, and record a warning.

CZI files are not affected (bioio-czi reads calibration from CZI's own
metadata schema, which has no equivalent ambiguous default) and are excluded
from this check by gating on the reader module name.

## Consequences

- A plain TIFF with no real resolution tag will now correctly fail 3D
  analysis (`z_um` requirement) and calibrated-2D analysis (`x_um`/`y_um`
  requirement) with an actionable error, instead of silently proceeding
  with a fabricated 1.0 um/pixel calibration.
- `dayana-nuclei inspect` surfaces this as an explicit warning
  ("TIFF file has no resolution tag...") so a researcher reviewing a file
  before analysis sees the ambiguity immediately.
- Synthetic test fixtures that need calibrated spacing must write a real
  `resolution=` + `resolutionunit=` (or use the ImageJ metadata path) via
  `tifffile.imwrite`, not rely on the default.
