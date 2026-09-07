import numpy as np
import pytest
from skimage.draw import disk

from nuclear_morphometry.models import PhysicalSpacing
from nuclear_morphometry.segmentation.fixture import FixtureSegmenter
from nuclear_morphometry.segmentation.normalize import normalize_percentile


def test_fixture_segmenter_is_deterministic() -> None:
    image = np.zeros((100, 100), dtype=np.float32)
    rr, cc = disk((50, 50), 20)
    image[rr, cc] = 1.0
    spacing = PhysicalSpacing(x_um=0.2, y_um=0.2)

    segmenter = FixtureSegmenter()
    result_a = segmenter.segment(image, spacing)
    result_b = segmenter.segment(image, spacing)

    np.testing.assert_array_equal(result_a.labels, result_b.labels)
    assert result_a.labels.max() == 1
    assert result_a.backend == "fixture"


def test_fixture_segmenter_two_objects() -> None:
    image = np.zeros((100, 200), dtype=np.float32)
    rr, cc = disk((50, 50), 20)
    image[rr, cc] = 1.0
    rr, cc = disk((50, 150), 20)
    image[rr, cc] = 1.0
    spacing = PhysicalSpacing(x_um=1.0, y_um=1.0)

    result = FixtureSegmenter().segment(image, spacing)
    assert result.labels.max() == 2


def test_normalize_percentile_clips_to_unit_range() -> None:
    image = np.linspace(0, 1000, 100).reshape(10, 10).astype(np.uint16)
    normalized = normalize_percentile(image, percentile_low=1.0, percentile_high=99.0)
    assert normalized.dtype == np.float32
    assert normalized.min() >= 0.0
    assert normalized.max() <= 1.0


def test_normalize_percentile_rejects_degenerate_range() -> None:
    image = np.zeros((10, 10), dtype=np.uint16)
    with pytest.raises(ValueError, match="degenerate range"):
        normalize_percentile(image, percentile_low=1.0, percentile_high=99.0)
