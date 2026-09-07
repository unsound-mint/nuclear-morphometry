import numpy as np
import pytest

from nuclear_morphometry.qc.image_metrics import compute_image_qc_metrics


def test_basic_metrics_on_uint16_image() -> None:
    image = np.zeros((10, 10), dtype=np.uint16)
    image[2:5, 2:5] = 1000
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[2:5, 2:5] = 1

    metrics = compute_image_qc_metrics(image, labels)

    assert metrics["image_min_intensity"] == 0.0
    assert metrics["image_max_intensity"] == 1000.0
    assert metrics["image_mean_intensity"] == pytest.approx(image.mean())
    assert metrics["image_occupied_fraction"] == pytest.approx(9 / 100)
    assert metrics["image_saturation_fraction"] == 0.0


def test_saturation_fraction_uses_dtype_ceiling() -> None:
    image = np.zeros((4, 4), dtype=np.uint8)
    image[0, 0] = 255
    image[0, 1] = 255
    labels = np.zeros((4, 4), dtype=np.int32)

    metrics = compute_image_qc_metrics(image, labels)

    assert metrics["image_saturation_fraction"] == pytest.approx(2 / 16)


def test_float_image_saturation_is_undefined_not_fabricated() -> None:
    image = np.linspace(0.0, 1.0, 16, dtype=np.float32).reshape(4, 4)
    labels = np.zeros((4, 4), dtype=np.int32)

    metrics = compute_image_qc_metrics(image, labels)

    assert np.isnan(metrics["image_saturation_fraction"])


def test_focus_metric_is_higher_for_sharper_image() -> None:
    labels = np.zeros((20, 20), dtype=np.int32)

    sharp = np.zeros((20, 20), dtype=np.float32)
    sharp[10:, :] = 1000.0  # hard step edge -> high-frequency content

    blurry = np.linspace(0.0, 1000.0, 20, dtype=np.float32)
    blurry = np.tile(blurry.reshape(20, 1), (1, 20))  # smooth gradient

    sharp_metrics = compute_image_qc_metrics(sharp, labels)
    blurry_metrics = compute_image_qc_metrics(blurry, labels)

    assert sharp_metrics["image_focus_metric"] > blurry_metrics["image_focus_metric"]


def test_3d_input_is_supported() -> None:
    image = np.zeros((5, 8, 8), dtype=np.uint16)
    image[2, 3:5, 3:5] = 500
    labels = np.zeros((5, 8, 8), dtype=np.int32)
    labels[2, 3:5, 3:5] = 1

    metrics = compute_image_qc_metrics(image, labels)

    assert metrics["image_max_intensity"] == 500.0
    assert metrics["image_occupied_fraction"] == pytest.approx(4 / (5 * 8 * 8))


def test_shape_mismatch_raises() -> None:
    image = np.zeros((10, 10), dtype=np.uint16)
    labels = np.zeros((8, 8), dtype=np.int32)

    with pytest.raises(ValueError, match="does not match"):
        compute_image_qc_metrics(image, labels)
