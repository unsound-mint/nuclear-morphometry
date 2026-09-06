import numpy as np
import pytest
from skimage.draw import disk

from dayana_nuclei.measurements.spatial import measure_perinuclear_rings
from dayana_nuclei.models import PhysicalSpacing


def test_near_and_far_ring_intensities_are_measured() -> None:
    labels = np.zeros((60, 60), dtype=np.int32)
    rr, cc = disk((30, 30), 8)
    labels[rr, cc] = 1
    intensity = np.zeros_like(labels, dtype=np.float64)
    # Bright band just outside the nucleus (near ring), dim far away.
    outer_rr, outer_cc = disk((30, 30), 11)
    intensity[outer_rr, outer_cc] = 500.0
    intensity[labels == 1] = 0.0  # inside the nucleus itself is irrelevant here
    spacing = PhysicalSpacing(x_um=1.0, y_um=1.0)

    results = measure_perinuclear_rings(
        labels, intensity, spacing, near_ring_um=(0.0, 3.0), far_ring_um=(10.0, 15.0)
    )

    assert len(results) == 1
    r = results[0]
    assert r.near_ring_mean_intensity == pytest.approx(500.0)
    assert r.far_ring_mean_intensity == pytest.approx(0.0)
    assert r.perinuclear_enrichment_ratio is None  # far mean is 0 -> ratio undefined


def test_enrichment_ratio_when_far_is_nonzero() -> None:
    labels = np.zeros((60, 60), dtype=np.int32)
    rr, cc = disk((30, 30), 8)
    labels[rr, cc] = 1
    intensity = np.full(labels.shape, 10.0, dtype=np.float64)
    near_rr, near_cc = disk((30, 30), 11)
    intensity[near_rr, near_cc] = 100.0
    intensity[labels == 1] = 0.0
    spacing = PhysicalSpacing(x_um=1.0, y_um=1.0)

    results = measure_perinuclear_rings(
        labels, intensity, spacing, near_ring_um=(0.0, 3.0), far_ring_um=(20.0, 25.0)
    )

    r = results[0]
    assert r.perinuclear_enrichment_ratio == pytest.approx(
        r.near_ring_mean_intensity / r.far_ring_mean_intensity  # type: ignore[operator]
    )
    assert r.perinuclear_enrichment_ratio > 1.0


def test_overlapping_perinuclear_regions_do_not_double_count() -> None:
    """Two nuclei close enough that naive per-object rings would overlap:
    each background pixel must be counted for exactly one nucleus (the
    nearest), never both."""
    labels = np.zeros((60, 60), dtype=np.int32)
    rr1, cc1 = disk((30, 20), 6)
    labels[rr1, cc1] = 1
    rr2, cc2 = disk((30, 40), 6)
    labels[rr2, cc2] = 2
    intensity = np.full(labels.shape, 50.0, dtype=np.float64)
    spacing = PhysicalSpacing(x_um=1.0, y_um=1.0)

    results = measure_perinuclear_rings(
        labels, intensity, spacing, near_ring_um=(0.0, 20.0), far_ring_um=(20.0, 25.0)
    )

    # Reconstruct near-ring pixel sets per object directly and assert they
    # are disjoint -- the real regression this test guards against.
    from scipy.ndimage import distance_transform_edt

    background = labels == 0
    distance, nearest_index = distance_transform_edt(background, return_indices=True)
    nearest_label = labels[tuple(nearest_index)]
    near1 = background & (nearest_label == 1) & (distance >= 0.0) & (distance < 20.0)
    near2 = background & (nearest_label == 2) & (distance >= 0.0) & (distance < 20.0)
    assert not np.any(near1 & near2)
    assert len(results) == 2


def test_invalid_ring_ordering_raises() -> None:
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[8:12, 8:12] = 1
    intensity = np.zeros_like(labels, dtype=np.float64)
    spacing = PhysicalSpacing(x_um=1.0, y_um=1.0)

    with pytest.raises(ValueError, match="ordered, non-overlapping"):
        measure_perinuclear_rings(
            labels, intensity, spacing, near_ring_um=(0.0, 10.0), far_ring_um=(5.0, 15.0)
        )


def test_no_objects_returns_empty_list() -> None:
    labels = np.zeros((10, 10), dtype=np.int32)
    intensity = np.zeros_like(labels, dtype=np.float64)
    spacing = PhysicalSpacing(x_um=1.0, y_um=1.0)

    results = measure_perinuclear_rings(
        labels, intensity, spacing, near_ring_um=(0.0, 1.0), far_ring_um=(1.0, 2.0)
    )

    assert results == []


def test_ring_touching_field_edge_is_flagged() -> None:
    """A nucleus close enough to the image boundary that its far ring runs
    off the edge must be flagged -- the mean is still reported (never
    dropped), but downstream analysis needs to know it's a partial sample."""
    labels = np.zeros((30, 30), dtype=np.int32)
    rr, cc = disk((3, 15), 3)  # near the top edge
    labels[rr, cc] = 1
    intensity = np.full(labels.shape, 100.0)
    spacing = PhysicalSpacing(x_um=1.0, y_um=1.0)

    results = measure_perinuclear_rings(
        labels, intensity, spacing, near_ring_um=(0.0, 2.0), far_ring_um=(5.0, 20.0)
    )

    r = results[0]
    assert r.far_ring_touches_border is True


def test_ring_far_from_field_edge_is_not_flagged() -> None:
    labels = np.zeros((100, 100), dtype=np.int32)
    rr, cc = disk((50, 50), 5)
    labels[rr, cc] = 1
    intensity = np.full(labels.shape, 100.0)
    spacing = PhysicalSpacing(x_um=1.0, y_um=1.0)

    results = measure_perinuclear_rings(
        labels, intensity, spacing, near_ring_um=(0.0, 2.0), far_ring_um=(5.0, 10.0)
    )

    r = results[0]
    assert r.near_ring_touches_border is False
    assert r.far_ring_touches_border is False


def test_shape_mismatch_raises() -> None:
    labels = np.zeros((10, 10), dtype=np.int32)
    intensity = np.zeros((8, 8), dtype=np.float64)
    spacing = PhysicalSpacing(x_um=1.0, y_um=1.0)

    with pytest.raises(ValueError, match="does not match"):
        measure_perinuclear_rings(
            labels, intensity, spacing, near_ring_um=(0.0, 1.0), far_ring_um=(1.0, 2.0)
        )
