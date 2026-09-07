import numpy as np
import pytest

from nuclear_morphometry.measurements.texture import (
    format_um_distance_label,
    measure_texture_2d,
    quantize_for_texture,
    um_distances_to_pixels,
)
from nuclear_morphometry.schema import nuclei_table_schema


def test_uniform_object_has_near_zero_contrast_and_entropy() -> None:
    labels = np.zeros((30, 30), dtype=np.int32)
    labels[5:25, 5:25] = 1
    image = np.full((30, 30), 128.0)

    results = measure_texture_2d(labels, image, distances_px=[1, 3])
    assert len(results) == 1
    r = results[0].model_dump()

    for d in (1, 3):
        assert r[f"contrast_d{d}"] == pytest.approx(0.0, abs=1e-9)
        assert r[f"entropy_d{d}"] == pytest.approx(0.0, abs=1e-9)
        assert r[f"energy_d{d}"] == pytest.approx(1.0, abs=1e-6)


def test_checkerboard_has_higher_contrast_and_entropy_than_uniform() -> None:
    labels = np.zeros((30, 30), dtype=np.int32)
    labels[5:25, 5:25] = 1

    uniform_image = np.full((30, 30), 128.0)
    checkerboard = np.zeros((30, 30))
    checkerboard[::2, ::2] = 255.0
    checkerboard[1::2, 1::2] = 255.0

    uniform_result = measure_texture_2d(labels, uniform_image, distances_px=[1])[0].model_dump()
    checker_result = measure_texture_2d(labels, checkerboard, distances_px=[1])[0].model_dump()

    # Texture isn't as analytically exact as morphology; assert the expected
    # direction of the effect, not exact values.
    assert checker_result["contrast_d1"] > uniform_result["contrast_d1"]
    assert checker_result["entropy_d1"] > uniform_result["entropy_d1"]


def test_um_distances_to_pixels_normal_conversion() -> None:
    pixels = um_distances_to_pixels([0.6, 1.0, 2.0], pixel_size_um=0.2)
    assert pixels == [3, 5, 10]


def test_um_distances_to_pixels_rejects_sub_pixel_distance() -> None:
    with pytest.raises(ValueError, match="less than 1 pixel"):
        um_distances_to_pixels([0.05], pixel_size_um=0.2)


def test_format_um_distance_label() -> None:
    assert format_um_distance_label(2.0) == "2um"
    assert format_um_distance_label(1.5) == "1p5um"
    assert format_um_distance_label(0.6) == "0p6um"


def test_measure_texture_2d_with_distance_labels_names_columns_by_label() -> None:
    labels = np.zeros((30, 30), dtype=np.int32)
    labels[5:25, 5:25] = 1
    image = np.full((30, 30), 128.0)

    results = measure_texture_2d(
        labels, image, distances_px=[5, 10], distance_labels=["1um", "2um"]
    )
    r = results[0].model_dump()

    assert "contrast_d1um" in r
    assert "entropy_d2um" in r
    assert "contrast_d5" not in r


def test_measure_texture_2d_rejects_mismatched_distance_labels_length() -> None:
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[2:8, 2:8] = 1
    with pytest.raises(ValueError, match="paired 1:1"):
        measure_texture_2d(
            labels, labels.astype(np.float64), distances_px=[1, 2], distance_labels=["1um"]
        )


def test_format_um_distance_label_matches_nuclei_table_schema() -> None:
    """Kept in sync with schema.py's nuclei_table_schema by this test, the
    same convention test_radial.py uses for radial-bin column names."""
    labels = tuple(format_um_distance_label(d) for d in (0.5, 1.0, 2.0))
    schema = nuclei_table_schema(
        include_intensity=False, include_texture=True, texture_distance_labels_um=labels
    )
    for label in labels:
        assert f"contrast_d{label}" in schema
        assert f"entropy_d{label}" in schema


def test_measure_texture_2d_rejects_3d_labels() -> None:
    labels_3d = np.zeros((3, 10, 10), dtype=np.int32)
    with pytest.raises(ValueError, match="requires a 2D label image"):
        measure_texture_2d(labels_3d, labels_3d.astype(np.float64), distances_px=[1])


def test_quantize_for_texture_output_range() -> None:
    rng = np.random.default_rng(0)
    image = rng.uniform(0, 1000, size=(20, 20))
    mask = np.zeros((20, 20), dtype=np.bool_)
    mask[5:15, 5:15] = True

    quantized = quantize_for_texture(image, mask, gray_levels=64)
    assert quantized.dtype == np.uint16
    assert quantized[mask].min() >= 0
    assert quantized[mask].max() <= 63


def test_quantize_for_texture_constant_region_maps_to_zero() -> None:
    image = np.full((10, 10), 42.0)
    mask = np.ones((10, 10), dtype=np.bool_)
    quantized = quantize_for_texture(image, mask, gray_levels=256)
    assert np.all(quantized == 0)


def test_quantize_for_texture_empty_mask_raises() -> None:
    image = np.zeros((10, 10))
    mask = np.zeros((10, 10), dtype=np.bool_)
    with pytest.raises(ValueError, match="no True pixels"):
        quantize_for_texture(image, mask, gray_levels=256)
