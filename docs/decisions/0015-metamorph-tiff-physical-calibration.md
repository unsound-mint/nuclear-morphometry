# 0015 - Read physical calibration from MetaMorph TIFF page metadata

## Status

Accepted.

## Context

The thesis TIFFs were written by MetaMorph/MetaSeries. They have no standard TIFF
`XResolution`, `YResolution`, or `ResolutionUnit` tags, so BioIO reports physical spacing as
unknown. The calibration is nevertheless present in each page's XML `ImageDescription`:

- `spatial-calibration-state = on`
- `spatial-calibration-x = 0.183`
- `spatial-calibration-y = 0.183`
- `spatial-calibration-units = um`
- an absolute `z-position` for every plane

The similarly named `pixel-size-x` and `pixel-size-y` values are array dimensions (1200),
not physical calibration.

## Decision

For TIFFs read through `bioio-tifffile`, inspect the active tifffile series' page
descriptions. Accept positive X/Y calibration only when the state is on, units are
micrometers, and values agree across pages. Derive Z step from the median absolute difference
of a complete, monotonic position sequence and require individual steps to agree within the
stored decimal precision. Repeated, non-monotonic, missing, or non-uniform positions produce
unknown Z spacing and an explicit warning.

Standard TIFF and ImageJ calibration remain supported. Explicit manifest calibration remains
a fallback for genuinely metadata-stripped files and must agree with embedded values when
both exist.

## Consequences

- The current multi-plane thesis TIFFs resolve to X=0.183 um, Y=0.183 um, Z=0.1 um directly
  from their own metadata.
- Raw TIFFs remain unchanged.
- Physical calibration is included in field outputs and run provenance through the existing
  pipeline path.
