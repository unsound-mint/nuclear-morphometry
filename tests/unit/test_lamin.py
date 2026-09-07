import numpy as np
import pytest
from skimage.draw import disk

from nuclear_morphometry.measurements.lamin import measure_lamin_shell_core
from nuclear_morphometry.models import PhysicalSpacing


def _disk_label(radius: int, *, shape: tuple[int, int] = (60, 60)) -> np.ndarray:
    labels = np.zeros(shape, dtype=np.int32)
    rr, cc = disk((shape[0] // 2, shape[1] // 2), radius)
    labels[rr, cc] = 1
    return labels


def test_large_disk_splits_into_shell_and_core() -> None:
    labels = _disk_label(radius=20)
    intensity = np.zeros_like(labels, dtype=np.float64)
    intensity[labels == 1] = 100.0
    spacing = PhysicalSpacing(x_um=1.0, y_um=1.0)

    results = measure_lamin_shell_core(labels, intensity, spacing, shell_width_um=5.0)

    assert len(results) == 1
    r = results[0]
    assert r.total_mean_intensity == pytest.approx(100.0)
    assert r.shell_mean_intensity == pytest.approx(100.0)
    assert r.core_mean_intensity == pytest.approx(100.0)
    assert r.shell_core_ratio == pytest.approx(1.0)


def test_shell_intensity_differs_from_core_when_signal_is_peripheral() -> None:
    labels = _disk_label(radius=20)
    intensity = np.zeros_like(labels, dtype=np.float64)
    # Bright rim only: bright everywhere within the disk, then zero out a
    # smaller inner disk to leave signal only near the boundary.
    intensity[labels == 1] = 200.0
    inner_rr, inner_cc = disk((30, 30), 12)
    intensity[inner_rr, inner_cc] = 0.0

    spacing = PhysicalSpacing(x_um=1.0, y_um=1.0)
    results = measure_lamin_shell_core(labels, intensity, spacing, shell_width_um=5.0)

    r = results[0]
    assert r.shell_mean_intensity is not None
    assert r.core_mean_intensity is not None
    assert r.shell_mean_intensity > r.core_mean_intensity
    assert r.shell_core_ratio == pytest.approx(r.shell_mean_intensity / r.core_mean_intensity)


def test_small_object_has_no_core() -> None:
    """An object smaller than shell_width_um everywhere is reported entirely
    as shell -- core_mean/ratio are None, not fabricated."""
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[9:12, 9:12] = 1  # 3x3 square: max interior distance is 2.0
    intensity = np.zeros_like(labels, dtype=np.float64)
    intensity[labels == 1] = 50.0
    spacing = PhysicalSpacing(x_um=1.0, y_um=1.0)

    results = measure_lamin_shell_core(labels, intensity, spacing, shell_width_um=5.0)

    r = results[0]
    assert r.shell_mean_intensity == pytest.approx(50.0)
    assert r.core_mean_intensity is None
    assert r.shell_core_ratio is None


def test_rectangular_object_filling_its_own_bounding_box() -> None:
    """Regression: an axis-aligned object with True on every border pixel of
    its own bbox has no zero pixel inside the crop for distance_transform_edt
    to measure against unless the mask is padded first -- verified empirically
    to otherwise produce meaningless distances. A 21x21 square with a 5um
    shell at 1um/px must still find a real core in the middle."""
    labels = np.zeros((30, 30), dtype=np.int32)
    labels[4:25, 4:25] = 1  # 21x21 solid square, fills its own bbox exactly
    intensity = np.zeros_like(labels, dtype=np.float64)
    intensity[labels == 1] = 75.0
    spacing = PhysicalSpacing(x_um=1.0, y_um=1.0)

    results = measure_lamin_shell_core(labels, intensity, spacing, shell_width_um=5.0)

    r = results[0]
    assert r.core_mean_intensity is not None
    assert r.core_mean_intensity == pytest.approx(75.0)
    assert r.shell_mean_intensity == pytest.approx(75.0)


def test_shell_width_is_physical_not_pixel_count() -> None:
    """The same disk radius in pixels, at two different pixel sizes, must
    produce different shell/core splits for the same shell_width_um --
    proving the erosion is calibrated in real distance, not a fixed pixel
    count. Uses a radial intensity gradient (not uniform intensity) so a
    bigger/smaller core actually changes the measured mean -- a naive
    pixel-count erosion would produce identical core means at both
    spacings, since the disk radius in pixels is the same either way."""
    shape = (60, 60)
    labels = _disk_label(radius=20, shape=shape)
    yy, xx = np.indices(shape)
    radial_distance = np.sqrt((yy - shape[0] // 2) ** 2 + (xx - shape[1] // 2) ** 2)
    intensity = np.where(labels == 1, radial_distance, 0.0)

    fine_spacing = PhysicalSpacing(x_um=0.5, y_um=0.5)
    coarse_spacing = PhysicalSpacing(x_um=2.0, y_um=2.0)

    fine = measure_lamin_shell_core(labels, intensity, fine_spacing, shell_width_um=5.0)[0]
    coarse = measure_lamin_shell_core(labels, intensity, coarse_spacing, shell_width_um=5.0)[0]

    # 5 um at 0.5 um/px = 10 px erosion depth -> a small, deep-interior core
    # (low mean radial distance). 5 um at 2 um/px = 2.5 px erosion depth ->
    # a much bigger core reaching closer to the boundary (higher mean
    # radial distance). A pixel-count-based (non-physical) implementation
    # would give the same erosion depth, and therefore the same core mean,
    # at both spacings.
    assert fine.core_mean_intensity is not None
    assert coarse.core_mean_intensity is not None
    assert fine.core_mean_intensity < coarse.core_mean_intensity


def test_requires_z_for_3d() -> None:
    labels = np.zeros((3, 10, 10), dtype=np.int32)
    labels[1, 3:7, 3:7] = 1
    intensity = np.zeros_like(labels, dtype=np.float64)
    spacing = PhysicalSpacing(x_um=1.0, y_um=1.0, z_um=None)

    with pytest.raises(ValueError, match="3D analysis requires"):
        measure_lamin_shell_core(labels, intensity, spacing, shell_width_um=1.0)


def test_shape_mismatch_raises() -> None:
    labels = np.zeros((10, 10), dtype=np.int32)
    intensity = np.zeros((8, 8), dtype=np.float64)
    spacing = PhysicalSpacing(x_um=1.0, y_um=1.0)

    with pytest.raises(ValueError, match="does not match"):
        measure_lamin_shell_core(labels, intensity, spacing, shell_width_um=1.0)


def test_non_positive_shell_width_raises() -> None:
    labels = _disk_label(radius=5)
    intensity = np.zeros_like(labels, dtype=np.float64)
    spacing = PhysicalSpacing(x_um=1.0, y_um=1.0)

    with pytest.raises(ValueError, match="shell_width_um"):
        measure_lamin_shell_core(labels, intensity, spacing, shell_width_um=0.0)
