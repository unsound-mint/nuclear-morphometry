"""Synthetic-shape tests with analytically-known expected values (spec 17.7)."""

import numpy as np
import pytest

from nuclear_morphometry.measurements.morphology_3d import measure_3d_morphology
from nuclear_morphometry.models import PhysicalSpacing


def _sphere_labels(radius_vox: int, margin: int = 5) -> np.ndarray:
    pad = radius_vox + margin
    zz, yy, xx = np.ogrid[-pad : pad + 1, -pad : pad + 1, -pad : pad + 1]
    mask = zz**2 + yy**2 + xx**2 <= radius_vox**2
    labels = np.zeros(mask.shape, dtype=np.int32)
    labels[mask] = 1
    return labels


def test_sphere_volume_axes_and_sphericity() -> None:
    # r=20 voxels is large enough that rasterization discretization error is
    # small; marching-cubes-on-a-hard-mask has a documented ~8-9% systematic
    # surface-area bias at any radius (see module docstring) which the
    # Gaussian pre-smoothing step corrects for -- these tolerances reflect
    # residual rasterization noise, not that known bias.
    radius_vox = 20
    vox = 0.2
    labels = _sphere_labels(radius_vox)
    spacing = PhysicalSpacing(x_um=vox, y_um=vox, z_um=vox)

    results = measure_3d_morphology(labels, spacing)
    assert len(results) == 1
    r = results[0]

    analytical_volume = 4.0 / 3.0 * np.pi * (radius_vox * vox) ** 3
    assert r.volume_um3 == pytest.approx(analytical_volume, rel=0.02)

    analytical_axis = 2 * radius_vox * vox
    assert r.axis_major_um == pytest.approx(analytical_axis, rel=0.02)
    assert r.axis_intermediate_um == pytest.approx(analytical_axis, rel=0.02)
    assert r.axis_minor_um == pytest.approx(analytical_axis, rel=0.02)

    assert r.surface_area_um2 is not None
    assert r.sphericity is not None
    # The load-bearing assertion: sphericity=1 for a sphere is what proves
    # volume, surface area, and the formula are mutually consistent, not
    # merely individually plausible.
    assert r.sphericity == pytest.approx(1.0, abs=0.05)

    assert not r.touches_border
    assert r.z_depth_slices == 2 * radius_vox + 1
    assert r.z_depth_um == pytest.approx((2 * radius_vox + 1) * vox)


def test_ellipsoid_volume_and_ordered_axes() -> None:
    a, b, c = 25, 15, 10  # semi-axes in voxels, z/y/x
    margin = 5
    pad_z, pad_y, pad_x = a + margin, b + margin, c + margin
    zz, yy, xx = np.ogrid[-pad_z : pad_z + 1, -pad_y : pad_y + 1, -pad_x : pad_x + 1]
    mask = (zz / a) ** 2 + (yy / b) ** 2 + (xx / c) ** 2 <= 1
    labels = np.zeros(mask.shape, dtype=np.int32)
    labels[mask] = 1

    vox = 0.2
    spacing = PhysicalSpacing(x_um=vox, y_um=vox, z_um=vox)
    r = measure_3d_morphology(labels, spacing)[0]

    analytical_volume = 4.0 / 3.0 * np.pi * a * b * c * vox**3
    assert r.volume_um3 == pytest.approx(analytical_volume, rel=0.02)

    # Recovered axes validate the L = 2*sqrt(5*lambda) formula against ground
    # truth, not just that the code executes.
    assert r.axis_major_um == pytest.approx(2 * a * vox, rel=0.02)
    assert r.axis_intermediate_um == pytest.approx(2 * b * vox, rel=0.02)
    assert r.axis_minor_um == pytest.approx(2 * c * vox, rel=0.02)
    assert r.axis_major_um > r.axis_intermediate_um > r.axis_minor_um


def test_cuboid_exact_volume_and_extent() -> None:
    labels = np.zeros((30, 30, 30), dtype=np.int32)
    labels[5:15, 8:20, 10:22] = 1  # 10 x 12 x 12 voxel box, away from all edges
    vox = 0.25
    spacing = PhysicalSpacing(x_um=vox, y_um=vox, z_um=vox)

    r = measure_3d_morphology(labels, spacing)[0]

    # An axis-aligned box has zero rasterization ambiguity: volume and extent
    # should match analytically, essentially exactly.
    assert r.volume_um3 == pytest.approx(10 * 12 * 12 * vox**3, rel=1e-9)
    assert r.extent == pytest.approx(1.0, rel=1e-9)

    # A cuboid is a poor sphericity test case: Gaussian pre-smoothing (needed
    # to correct the sphere bias) rounds sharp corners, which understates
    # surface area and so *overstates* sphericity for angular shapes -- see
    # the module docstring. Only assert the qualitative fact that a box is
    # less sphere-like than a sphere, not a tight analytical value.
    assert r.sphericity is not None
    assert r.sphericity < 1.0


def test_single_voxel_object_has_no_surface_or_sphericity() -> None:
    labels = np.zeros((10, 10, 10), dtype=np.int32)
    labels[5, 5, 5] = 1
    spacing = PhysicalSpacing(x_um=0.25, y_um=0.25, z_um=0.25)

    r = measure_3d_morphology(labels, spacing)[0]

    assert r.volume_voxels == 1
    assert r.volume_um3 == pytest.approx(0.25**3)
    # Too small for the smoothed mask to straddle the isosurface level --
    # marching_cubes cannot extract a surface, so both are None rather than
    # a fabricated or degenerate (zero/inf) value.
    assert r.surface_area_um2 is None
    assert r.sphericity is None


def test_anisotropic_spacing_volume_and_z_depth() -> None:
    radius_vox = 15
    vox_xy = 0.2
    vox_z = 0.5
    labels = _sphere_labels(radius_vox)
    spacing = PhysicalSpacing(x_um=vox_xy, y_um=vox_xy, z_um=vox_z)

    r = measure_3d_morphology(labels, spacing)[0]

    analytical_volume = 4.0 / 3.0 * np.pi * radius_vox**3 * (vox_xy * vox_xy * vox_z)
    assert r.volume_um3 == pytest.approx(analytical_volume, rel=0.02)
    assert r.z_depth_slices == 2 * radius_vox + 1
    assert r.z_depth_um == pytest.approx((2 * radius_vox + 1) * vox_z)


def test_border_touching_object_has_no_sphericity() -> None:
    labels = np.zeros((10, 10, 10), dtype=np.int32)
    labels[0:3, 3:6, 3:6] = 1  # clipped at the Z=0 face
    spacing = PhysicalSpacing(x_um=0.25, y_um=0.25, z_um=0.25)

    r = measure_3d_morphology(labels, spacing)[0]

    assert r.touches_border is True
    assert r.surface_area_um2 is None
    assert r.sphericity is None


def test_rejects_non_3d_input() -> None:
    spacing = PhysicalSpacing(x_um=1.0, y_um=1.0, z_um=1.0)
    with pytest.raises(ValueError, match="requires a 3D label image"):
        measure_3d_morphology(np.zeros((10, 10), dtype=np.int32), spacing)


def test_missing_z_spacing_raises_actionable_error() -> None:
    labels = np.zeros((5, 5, 5), dtype=np.int32)
    labels[1:3, 1:3, 1:3] = 1
    spacing = PhysicalSpacing(x_um=1.0, y_um=1.0, z_um=None)
    with pytest.raises(ValueError, match="3D analysis requires physical Z spacing"):
        measure_3d_morphology(labels, spacing)
