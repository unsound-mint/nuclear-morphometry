import numpy as np
import pytest

from nuclear_morphometry.measurements.intensity import measure_intensity


def test_2d_constant_intensity_objects() -> None:
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[2:6, 2:6] = 1  # 16 px, value 100
    labels[10:13, 10:13] = 2  # 9 px, value 200

    image = np.zeros((20, 20), dtype=np.float64)
    image[labels == 1] = 100.0
    image[labels == 2] = 200.0

    results = {r.object_number: r for r in measure_intensity(labels, image)}
    assert set(results) == {1, 2}

    r1 = results[1]
    assert r1.mean_intensity == pytest.approx(100.0)
    assert r1.median_intensity == pytest.approx(100.0)
    assert r1.min_intensity == pytest.approx(100.0)
    assert r1.max_intensity == pytest.approx(100.0)
    assert r1.std_intensity == pytest.approx(0.0)
    assert r1.integrated_intensity == pytest.approx(16 * 100.0)

    r2 = results[2]
    assert r2.integrated_intensity == pytest.approx(9 * 200.0)


def test_3d_is_dimension_agnostic() -> None:
    labels = np.zeros((6, 10, 10), dtype=np.int32)
    labels[1:4, 2:5, 2:5] = 1  # 3*3*3 = 27 voxels

    image = np.zeros((6, 10, 10), dtype=np.float64)
    image[labels == 1] = 50.0

    results = measure_intensity(labels, image)
    assert len(results) == 1
    r = results[0]
    assert r.mean_intensity == pytest.approx(50.0)
    assert r.integrated_intensity == pytest.approx(27 * 50.0)


def test_shape_mismatch_raises() -> None:
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[2:5, 2:5] = 1
    image = np.zeros((10, 11), dtype=np.float64)

    with pytest.raises(ValueError, match="does not match"):
        measure_intensity(labels, image)


def test_non_constant_gradient_integrated_intensity_matches_analytical_sum() -> None:
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[3:7, 3:7] = 1  # 4x4 = 16 px

    image = np.arange(100, dtype=np.float64).reshape(10, 10)
    expected_sum = image[3:7, 3:7].sum()

    results = measure_intensity(labels, image)
    assert len(results) == 1
    assert results[0].integrated_intensity == pytest.approx(expected_sum)
    assert results[0].mean_intensity == pytest.approx(expected_sum / 16)
