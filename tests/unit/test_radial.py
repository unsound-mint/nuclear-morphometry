import numpy as np
import pytest
from skimage.draw import disk

from dayana_nuclei.measurements.radial import (
    measure_radial_distribution_2d,
    radial_bin_columns,
)


def test_radial_bin_columns_naming() -> None:
    columns = radial_bin_columns(3)
    assert columns == (
        "radial_bin0_mean_intensity",
        "radial_bin0_frac_intensity",
        "radial_bin0_frac_pixels",
        "radial_bin1_mean_intensity",
        "radial_bin1_frac_intensity",
        "radial_bin1_frac_pixels",
        "radial_bin2_mean_intensity",
        "radial_bin2_frac_intensity",
        "radial_bin2_frac_pixels",
    )


def test_frac_pixels_sums_to_one() -> None:
    labels = np.zeros((60, 60), dtype=np.int32)
    rr, cc = disk((30, 30), 20)
    labels[rr, cc] = 1
    intensity = np.full(labels.shape, 100.0)

    results = measure_radial_distribution_2d(labels, intensity, radial_bins=5)

    r = results[0]
    values = r.model_dump(exclude={"object_number"})
    frac_pixels_total = sum(values[f"radial_bin{b}_frac_pixels"] for b in range(5))
    assert frac_pixels_total == pytest.approx(1.0)


def test_uniform_intensity_gives_equal_mean_per_bin() -> None:
    labels = np.zeros((60, 60), dtype=np.int32)
    rr, cc = disk((30, 30), 20)
    labels[rr, cc] = 1
    intensity = np.full(labels.shape, 100.0)

    results = measure_radial_distribution_2d(labels, intensity, radial_bins=5)
    values = results[0].model_dump(exclude={"object_number"})

    for b in range(5):
        assert values[f"radial_bin{b}_mean_intensity"] == pytest.approx(100.0)


def test_center_bright_edge_dim_increases_toward_center() -> None:
    """A radial intensity gradient (bright center, dim edge) must produce a
    monotonically decreasing mean intensity from the innermost to the
    outermost bin -- the real discriminating test that the bins are
    ordered center-to-edge as documented, not reversed or scrambled."""
    shape = (80, 80)
    labels = np.zeros(shape, dtype=np.int32)
    rr, cc = disk((40, 40), 30)
    labels[rr, cc] = 1
    yy, xx = np.indices(shape)
    radial_distance = np.sqrt((yy - 40) ** 2 + (xx - 40) ** 2)
    intensity = np.where(labels == 1, 1000.0 - radial_distance * 10.0, 0.0)

    results = measure_radial_distribution_2d(labels, intensity, radial_bins=5)
    values = results[0].model_dump(exclude={"object_number"})
    means = [values[f"radial_bin{b}_mean_intensity"] for b in range(5)]

    assert all(m is not None for m in means)
    assert means == sorted(means, reverse=True)


def test_frac_intensity_sums_to_one_when_intensity_nonzero() -> None:
    shape = (60, 60)
    labels = np.zeros(shape, dtype=np.int32)
    rr, cc = disk((30, 30), 20)
    labels[rr, cc] = 1
    yy, xx = np.indices(shape)
    radial_distance = np.sqrt((yy - 30) ** 2 + (xx - 30) ** 2)
    intensity = np.where(labels == 1, 500.0 - radial_distance * 5.0, 0.0)

    results = measure_radial_distribution_2d(labels, intensity, radial_bins=4)
    values = results[0].model_dump(exclude={"object_number"})
    frac_total = sum(values[f"radial_bin{b}_frac_intensity"] for b in range(4))
    assert frac_total == pytest.approx(1.0, rel=1e-6)


def test_small_object_may_have_empty_inner_bins_reported_as_none() -> None:
    """A 4-pixel object split into 5 equal-count bins cannot fill every bin:
    with equal-pixel-count binning the innermost (deepest-interior) bins are
    the ones left empty, since a shallow object has few/no pixels at that
    rank -- not the outermost/boundary bin, which is always populated first."""
    labels = np.zeros((20, 20), dtype=np.int32)
    labels[9:11, 9:11] = 1  # 2x2 object, very shallow depth
    intensity = np.full(labels.shape, 50.0)

    results = measure_radial_distribution_2d(labels, intensity, radial_bins=5)
    values = results[0].model_dump(exclude={"object_number"})

    assert values["radial_bin0_frac_pixels"] == 0.0
    assert values["radial_bin0_mean_intensity"] is None
    assert values["radial_bin0_frac_intensity"] is None
    assert values["radial_bin4_frac_pixels"] > 0.0
    assert values["radial_bin4_mean_intensity"] == pytest.approx(50.0)


def test_rectangular_object_filling_its_own_bounding_box() -> None:
    """Same padding regression as measurements/lamin.py: an object with no
    zero pixel inside its own bbox must still produce sane distances."""
    labels = np.zeros((30, 30), dtype=np.int32)
    labels[4:25, 4:25] = 1
    intensity = np.full(labels.shape, 75.0)

    results = measure_radial_distribution_2d(labels, intensity, radial_bins=3)
    values = results[0].model_dump(exclude={"object_number"})

    assert values["radial_bin0_mean_intensity"] == pytest.approx(75.0)
    assert sum(values[f"radial_bin{b}_frac_pixels"] for b in range(3)) == pytest.approx(1.0)


def test_bins_are_approximately_equal_area_not_equal_normalized_width() -> None:
    """Regression for the equal-width-normalized-distance definition, where a
    20 px-radius disk put ~4.6% of pixels in bin 0 and ~34% in the last bin
    for 5 bins. Equal-pixel-count binning must keep every bin within a small
    tolerance of 1/radial_bins, regardless of object shape."""
    labels = np.zeros((60, 60), dtype=np.int32)
    rr, cc = disk((30, 30), 20)
    labels[rr, cc] = 1
    intensity = np.full(labels.shape, 100.0)

    results = measure_radial_distribution_2d(labels, intensity, radial_bins=5)
    values = results[0].model_dump(exclude={"object_number"})
    fracs = [values[f"radial_bin{b}_frac_pixels"] for b in range(5)]

    for frac in fracs:
        assert frac == pytest.approx(0.2, abs=0.03)


def test_single_pixel_object_does_not_divide_by_zero() -> None:
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[5, 5] = 1
    intensity = np.full(labels.shape, 42.0)

    results = measure_radial_distribution_2d(labels, intensity, radial_bins=5)
    values = results[0].model_dump(exclude={"object_number"})

    assert sum(values[f"radial_bin{b}_frac_pixels"] for b in range(5)) == pytest.approx(1.0)
    populated = [b for b in range(5) if values[f"radial_bin{b}_frac_pixels"] > 0.0]
    assert populated == [4]
    assert values["radial_bin4_mean_intensity"] == pytest.approx(42.0)


def test_3d_labels_raise() -> None:
    labels = np.zeros((3, 10, 10), dtype=np.int32)
    labels[1, 3:7, 3:7] = 1
    intensity = np.zeros_like(labels, dtype=np.float64)

    with pytest.raises(ValueError, match="2D-only"):
        measure_radial_distribution_2d(labels, intensity)


def test_shape_mismatch_raises() -> None:
    labels = np.zeros((10, 10), dtype=np.int32)
    intensity = np.zeros((8, 8), dtype=np.float64)

    with pytest.raises(ValueError, match="does not match"):
        measure_radial_distribution_2d(labels, intensity)


def test_invalid_radial_bins_raises() -> None:
    labels = np.zeros((10, 10), dtype=np.int32)
    intensity = np.zeros((10, 10), dtype=np.float64)

    with pytest.raises(ValueError, match="radial_bins"):
        measure_radial_distribution_2d(labels, intensity, radial_bins=0)


def test_radial_bin_columns_matches_nuclei_table_schema() -> None:
    """schema.py duplicates this module's column names (matching the
    project's existing texture-property duplication convention) rather than
    importing measurements/ from schema.py -- this test is what keeps the
    two definitions from drifting apart."""
    from dayana_nuclei.schema import nuclei_table_schema

    for n in (1, 3, 5):
        expected = set(radial_bin_columns(n))
        schema_columns = set(
            nuclei_table_schema(include_intensity=False, include_texture=False, radial_bins=n)
        )
        assert expected <= schema_columns
