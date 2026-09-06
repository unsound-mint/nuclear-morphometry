# 0007 - Gaussian pre-smoothing for marching-cubes surface area

## Status

Accepted.

## Context

Spec section 17.2 requires per-object surface area via marching cubes on each object's own
cropped binary mask, and section 17.6 requires sphericity computed from that surface area
and the exact voxel volume, validated against a synthetic sphere.

Empirical testing (see `tests/unit/test_morphology_3d.py`) found that running
`skimage.measure.marching_cubes` directly on a *hard* (0/1, unsmoothed) rasterized sphere
mask has a systematic surface-area overestimation of roughly 8-9% at typical nuclear-scale
voxel resolutions. Because volume is exact (a voxel count times physical voxel volume, no
reconstruction involved), this bias flows entirely into sphericity: a perfect analytical
sphere -- which must have `sphericity == 1.0` -- computed to roughly 0.92 with unsmoothed
marching cubes. That is large enough to be mistaken for a real morphological signal rather
than a reconstruction artifact, which would be scientifically dangerous for a study whose
outcome measures include sphericity.

The cause is the well-known "staircase" surface area inflation of marching cubes on a hard
binary boundary: the reconstructed isosurface hugs the voxel-grid steps rather than the true
smooth boundary, adding real (not illusory) extra surface area to the mesh.

## Decision

Before marching cubes, each object's cropped, zero-padded binary mask is smoothed with an
isotropic Gaussian filter (`sigma = 1.0` voxel, applied in voxel-index space, i.e. before
the physical `spacing` argument is used by `marching_cubes`) and the isosurface is
extracted at level `0.5` of the smoothed field. This is verified empirically, not assumed:
with this smoothing, the analytical-sphere sphericity test passes at `1.0 ± 0.05`; without
it, the same test fails at ~0.92.

This smoothing is applied **only** to the surface-area/sphericity reconstruction. It never
touches `volume_um3` (exact voxel count × physical voxel volume) or the principal-axis
computation (exact voxel-coordinate covariance) -- both remain unsmoothed and unbiased.

## Consequences

- **Tradeoff for angular shapes**: the same smoothing that fixes the sphere bias rounds
  sharp corners, which *understates* surface area (and therefore *overstates* sphericity)
  for a cuboid or any other angular object.
  `tests/unit/test_morphology_3d.py::test_cuboid_exact_volume_and_extent` only asserts
  `sphericity < 1.0` for a cuboid, not a tight analytical value, for exactly this reason.
  This tradeoff is accepted deliberately: real nuclei are much closer to smooth ellipsoids
  than to sharp-edged polyhedra, so optimizing the correction for the sphere case is the
  scientifically relevant choice.
- **Objects too small to reconstruct**: if the padded, smoothed mask's isosurface level
  (0.5) doesn't fall strictly between the smoothed field's min and max -- empirically,
  smaller than roughly a 3×3×3-voxel object at `sigma=1` -- `marching_cubes` cannot produce
  a meaningful surface at all. Rather than let its `ValueError` propagate or fabricate a
  surface area, `measure_3d_morphology` reports `surface_area_um2 = None` and
  `sphericity = None` for that object, the same "don't emit misleading sphericity" treatment
  spec 17.6 requires for border-truncated objects (which also get `None` here, since a
  truncated object's true surface is unknown by definition).
- Any future change to the smoothing parameter (`sigma`), padding width, or isosurface level
  must be re-validated against the analytical sphere/ellipsoid tests before being accepted --
  these constants were tuned empirically, not derived analytically, and are not guaranteed
  optimal for a different voxel size/anisotropy regime without re-checking.
