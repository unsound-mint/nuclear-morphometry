import numpy as np
import pytest
from skimage.draw import disk, ellipse

from dayana_nuclei.measurements.morphology_2d import measure_2d_morphology
from dayana_nuclei.models import PhysicalSpacing


def test_circle_has_high_circularity_and_known_area() -> None:
    labels = np.zeros((200, 200), dtype=np.int32)
    rr, cc = disk((100, 100), 40)
    labels[rr, cc] = 1
    spacing = PhysicalSpacing(x_um=0.5, y_um=0.5)

    results = measure_2d_morphology(labels, spacing)
    assert len(results) == 1
    r = results[0]
    # The pixel-boundary perimeter estimator (see module docstring) has a
    # known ~5-10% bias on rasterized circles at this radius; 0.85 is "high
    # circularity, no measurement bug" without asserting sub-pixel accuracy
    # the estimator doesn't provide.
    assert 0.85 < r.circularity <= 1.0
    assert r.area_um2 == pytest.approx(r.area_px * 0.25, rel=1e-9)
    assert not r.touches_border


def test_elongated_ellipse_is_not_excluded_by_shape() -> None:
    """Spec 45/53.6: eccentricity/solidity/circularity alone must never exclude an object.

    This module only computes the morphology values -- it asserts that a
    highly elongated, low-circularity object still produces a normal,
    complete row (no QC decision is made here; qc/flags.py is what must
    never look at these fields for exclusion, which is tested separately).
    """
    labels = np.zeros((100, 300), dtype=np.int32)
    rr, cc = ellipse(50, 150, 10, 140)
    labels[rr, cc] = 1
    spacing = PhysicalSpacing(x_um=0.5, y_um=0.5)

    results = measure_2d_morphology(labels, spacing)
    assert len(results) == 1
    r = results[0]
    assert r.eccentricity > 0.95
    assert r.circularity < 0.5
    assert r.area_um2 > 0


def test_border_touching_object_is_flagged() -> None:
    labels = np.zeros((50, 50), dtype=np.int32)
    labels[0:10, 0:10] = 1
    spacing = PhysicalSpacing(x_um=1.0, y_um=1.0)

    results = measure_2d_morphology(labels, spacing)
    assert results[0].touches_border is True


def test_rejects_non_2d_input() -> None:
    spacing = PhysicalSpacing(x_um=1.0, y_um=1.0)
    with pytest.raises(ValueError, match="requires a 2D label image"):
        measure_2d_morphology(np.zeros((2, 3, 4), dtype=np.int32), spacing)
