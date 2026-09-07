"""3D nucleus morphology (spec section 17).

Volume (17.1) and the principal axes (17.4) are computed directly from the
exact voxel mask -- no smoothing, no approximation beyond the voxel grid
itself.

Surface area (17.2), and therefore sphericity (17.6) which depends on it,
are different: each object is cropped to its own bounding box plus a small
zero-padded margin and reconstructed independently -- never one
marching-cubes pass over the full field (spec 17.2 is explicit about this).
The cropped binary mask is smoothed with a small Gaussian (sigma=1 voxel,
isotropic in voxel-index space, applied before the physical `spacing` is
used) before marching cubes. This is not a stylistic default -- verified
empirically against an analytical sphere (see
``tests/unit/test_morphology_3d.py``): marching cubes on a *hard* 0/1
rasterized sphere has a systematic ~8-9% surface-area overestimation bias
at these resolutions, which alone drags computed sphericity for a perfect
sphere down to roughly 0.92, not the analytically-correct 1.0. Gaussian
smoothing (sigma=1) before marching cubes brings that back to ~1.00-1.01.
Smoothing is applied only for this surface reconstruction step and never
touches the volume or axis computations.

The same smoothing that corrects the sphere bias has an opposite-direction
cost for sharp-edged objects: it rounds corners, which understates surface
area (and therefore overstates sphericity) for a cuboid or any other
angular shape. Real nuclei are much closer to smooth than to angular, so
this tradeoff is accepted deliberately -- but a cuboid is a poor case for
validating an exact sphericity number for this reason (see the cuboid test,
which only asserts sphericity < 1, not a tight analytical value).

An object whose padded mask is too small for the smoothed values to
straddle the isosurface level (empirically, smaller than roughly a 3x3x3
voxel object at sigma=1 -- see the single-voxel test case) cannot yield a
meaningful surface at all; ``skimage.measure.marching_cubes`` itself raises
``ValueError`` in that case. Rather than letting that propagate, or
returning a fabricated surface area, this module checks for it explicitly
and reports ``surface_area_um2 = None`` and ``sphericity = None`` -- the
same "don't emit misleading sphericity" treatment spec 17.6 requires for a
border-truncated object, applied here to a reconstruction-too-small object.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from scipy.ndimage import gaussian_filter
from skimage.measure import marching_cubes, mesh_surface_area, regionprops

from nuclear_morphometry.models import PhysicalSpacing

_SURFACE_SMOOTH_SIGMA = 1.0
_SURFACE_PAD_VOXELS = 3
_ISOSURFACE_LEVEL = 0.5


class Nucleus3DMorphology(BaseModel):
    model_config = ConfigDict(frozen=True)

    object_number: int
    volume_voxels: int
    volume_um3: float
    surface_area_um2: float | None
    z_depth_slices: int
    z_depth_um: float
    axis_major_um: float
    axis_intermediate_um: float
    axis_minor_um: float
    extent: float
    sphericity: float | None
    touches_border: bool


def _reconstruct_surface_area_um2(
    object_mask: NDArray[np.bool_], spacing: PhysicalSpacing, z_um: float
) -> float | None:
    """Surface area of one already-cropped object mask, or None if unreconstructable."""
    padded = np.pad(object_mask, _SURFACE_PAD_VOXELS, mode="constant", constant_values=False)
    smoothed = gaussian_filter(padded.astype(np.float64), sigma=_SURFACE_SMOOTH_SIGMA)
    if not (smoothed.min() < _ISOSURFACE_LEVEL < smoothed.max()):
        return None
    verts, faces, _normals, _values = marching_cubes(
        smoothed,
        level=_ISOSURFACE_LEVEL,
        spacing=(z_um, spacing.y_um, spacing.x_um),
    )
    return float(mesh_surface_area(verts, faces))


def measure_3d_morphology(
    labels: NDArray[np.integer[Any]],
    spacing: PhysicalSpacing,
) -> list[Nucleus3DMorphology]:
    if labels.ndim != 3:
        raise ValueError(f"measure_3d_morphology requires a 3D label image, got ndim={labels.ndim}")

    z_um = spacing.require_z(context="3D morphology measurement")
    voxel_volume_um3 = spacing.voxel_volume_um3
    depth, height, width = labels.shape

    results: list[Nucleus3DMorphology] = []
    for prop in regionprops(labels):
        min_z, min_y, min_x, max_z, max_y, max_x = prop.bbox
        touches_border = (
            min_z == 0
            or min_y == 0
            or min_x == 0
            or max_z == depth
            or max_y == height
            or max_x == width
        )

        volume_voxels = int(prop.area)
        volume_um3 = volume_voxels * voxel_volume_um3

        # z_depth: occupied-plane count, not a continuous physical span -- the
        # number of distinct Z indices any voxel of this object occupies. Uses
        # unique z-indices among the object's own voxels rather than the
        # bounding-box span so a (rare) non-contiguous object in Z is not
        # over-counted.
        z_depth_slices = int(np.unique(prop.coords[:, 0]).shape[0])
        z_depth_um = z_depth_slices * z_um

        # Principal axes (spec 17.4): covariance of voxel-center coordinates in
        # physical units, eigenvalues -> equivalent uniform-solid-ellipsoid axis
        # lengths via L = 2*sqrt(5*lambda). ddof=0 matches the uniform-solid
        # derivation the spec gives (covariance along an axis of a uniform solid
        # ellipsoid is exactly a^2/5, a population moment, not a sample estimate).
        physical_coords = prop.coords.astype(np.float64) * np.array(
            [z_um, spacing.y_um, spacing.x_um]
        )
        physical_coords -= physical_coords.mean(axis=0)
        cov = (physical_coords.T @ physical_coords) / physical_coords.shape[0]
        eigenvalues = np.clip(np.linalg.eigvalsh(cov), 0.0, None)
        axis_major_um, axis_intermediate_um, axis_minor_um = sorted(
            (2.0 * math.sqrt(5.0 * lam) for lam in eigenvalues), reverse=True
        )

        bbox_volume_um3 = (
            (max_z - min_z) * z_um * (max_y - min_y) * spacing.y_um * (max_x - min_x) * spacing.x_um
        )
        extent = volume_um3 / bbox_volume_um3 if bbox_volume_um3 > 0 else 0.0

        surface_area_um2 = (
            None if touches_border else _reconstruct_surface_area_um2(prop.image, spacing, z_um)
        )
        sphericity = None
        if surface_area_um2 is not None and surface_area_um2 > 0:
            sphericity = (
                (math.pi ** (1.0 / 3.0)) * (6.0 * volume_um3) ** (2.0 / 3.0) / surface_area_um2
            )

        results.append(
            Nucleus3DMorphology(
                object_number=int(prop.label),
                volume_voxels=volume_voxels,
                volume_um3=volume_um3,
                surface_area_um2=surface_area_um2,
                z_depth_slices=z_depth_slices,
                z_depth_um=z_depth_um,
                axis_major_um=axis_major_um,
                axis_intermediate_um=axis_intermediate_um,
                axis_minor_um=axis_minor_um,
                extent=extent,
                sphericity=sphericity,
                touches_border=touches_border,
            )
        )
    return results
