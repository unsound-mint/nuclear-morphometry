from pathlib import Path

import numpy as np
import pytest

from dayana_nuclei.io.masks import save_label_mask
from dayana_nuclei.models import PhysicalSpacing
from dayana_nuclei.segmentation.validation import (
    validate_segmentation,
    validate_segmentation_batch,
    validate_segmentation_from_files,
)


def test_perfect_match_gives_iou_one_and_perfect_scores() -> None:
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[1:4, 1:4] = 1
    labels[6:9, 6:9] = 2

    report = validate_segmentation(labels, labels.copy(), iou_threshold=0.5)

    assert report.metrics.predicted_object_count == 2
    assert report.metrics.reference_object_count == 2
    assert report.metrics.matched_count == 2
    assert report.metrics.mean_matched_iou == pytest.approx(1.0)
    assert report.metrics.median_matched_iou == pytest.approx(1.0)
    assert report.metrics.precision == pytest.approx(1.0)
    assert report.metrics.recall == pytest.approx(1.0)
    assert report.metrics.f1 == pytest.approx(1.0)
    assert report.metrics.unmatched_prediction_count == 0
    assert report.metrics.unmatched_reference_count == 0


def test_partial_overlap_below_threshold_counts_as_unmatched() -> None:
    # Reference object: 10x10 = 100 px. Prediction: overlapping 4x10 = 40 px
    # subset -> intersection 40, union 100 + 40 - 40 = 100 -> IoU = 0.4.
    reference = np.zeros((10, 10), dtype=np.int32)
    reference[0:10, 0:10] = 1
    prediction = np.zeros((10, 10), dtype=np.int32)
    prediction[0:4, 0:10] = 1

    report = validate_segmentation(prediction, reference, iou_threshold=0.5)

    assert report.metrics.matched_count == 0
    assert report.metrics.unmatched_prediction_count == 1
    assert report.metrics.unmatched_reference_count == 1
    assert report.metrics.precision == pytest.approx(0.0)
    assert report.metrics.recall == pytest.approx(0.0)
    assert report.metrics.f1 == pytest.approx(0.0)
    # The candidate assignment (below threshold) is still reported, with its
    # true IoU, for visual/manual review -- it is not silently dropped.
    candidate = next(m for m in report.matches if m.prediction_label == 1)
    assert candidate.reference_label == 1
    assert candidate.iou == pytest.approx(0.4)
    assert candidate.is_true_positive is False


def test_zero_predicted_objects_precision_is_undefined_not_zero() -> None:
    reference = np.zeros((5, 5), dtype=np.int32)
    reference[1:3, 1:3] = 1
    prediction = np.zeros((5, 5), dtype=np.int32)

    report = validate_segmentation(prediction, reference)

    assert report.metrics.predicted_object_count == 0
    assert report.metrics.precision is None
    assert report.metrics.recall == pytest.approx(0.0)
    assert report.metrics.f1 is None
    assert report.metrics.unmatched_reference_count == 1


def test_zero_reference_objects_recall_is_undefined_not_zero() -> None:
    prediction = np.zeros((5, 5), dtype=np.int32)
    prediction[1:3, 1:3] = 1
    reference = np.zeros((5, 5), dtype=np.int32)

    report = validate_segmentation(prediction, reference)

    assert report.metrics.reference_object_count == 0
    assert report.metrics.recall is None
    assert report.metrics.precision == pytest.approx(0.0)
    assert report.metrics.f1 is None


def test_both_empty_precision_and_recall_are_undefined() -> None:
    empty = np.zeros((5, 5), dtype=np.int32)

    report = validate_segmentation(empty, empty)

    assert report.metrics.predicted_object_count == 0
    assert report.metrics.reference_object_count == 0
    assert report.metrics.precision is None
    assert report.metrics.recall is None
    assert report.metrics.f1 is None


def test_split_merge_disabled_by_default() -> None:
    labels = np.zeros((5, 5), dtype=np.int32)
    labels[1:3, 1:3] = 1

    report = validate_segmentation(labels, labels.copy())

    assert report.split_merge is None


def test_split_is_detected_when_one_reference_matches_two_predictions() -> None:
    reference = np.zeros((10, 10), dtype=np.int32)
    reference[0:10, 0:10] = 1
    prediction = np.zeros((10, 10), dtype=np.int32)
    prediction[0:5, 0:10] = 1
    prediction[5:10, 0:10] = 2

    report = validate_segmentation(
        prediction,
        reference,
        estimate_split_merge=True,
        split_merge_containment_threshold=0.1,
    )

    assert report.split_merge is not None
    assert report.split_merge.probable_split_reference_labels == (1,)
    assert report.split_merge.probable_merge_prediction_labels == ()


def test_merge_is_detected_when_one_prediction_covers_two_references() -> None:
    reference = np.zeros((10, 10), dtype=np.int32)
    reference[0:5, 0:10] = 1
    reference[5:10, 0:10] = 2
    prediction = np.zeros((10, 10), dtype=np.int32)
    prediction[0:10, 0:10] = 1

    report = validate_segmentation(
        prediction,
        reference,
        estimate_split_merge=True,
        split_merge_containment_threshold=0.1,
    )

    assert report.split_merge is not None
    assert report.split_merge.probable_merge_prediction_labels == (1,)
    assert report.split_merge.probable_split_reference_labels == ()


def test_split_is_detected_for_heavy_fragmentation_where_iou_would_miss_it() -> None:
    """Regression for docs/decisions/0008: Cellpose-SAM 3D fragmented one sphere
    into 25-434 pieces. Each fragment's IoU with the reference is ~1/n_fragments,
    which drops below any fixed IoU threshold as fragmentation worsens -- IoU
    cannot detect this. Containment (intersection / min(pred_area, ref_area))
    stays ~1.0 per fragment regardless of fragment count, so it must catch this
    even with the default threshold."""
    reference = np.zeros((10, 100), dtype=np.int32)
    reference[:, :] = 1
    prediction = np.zeros((10, 100), dtype=np.int32)
    for i in range(10):
        prediction[:, i * 10 : (i + 1) * 10] = i + 1  # 10 equal-sized fragments

    report = validate_segmentation(prediction, reference, estimate_split_merge=True)

    assert report.split_merge is not None
    assert report.split_merge.probable_split_reference_labels == (1,)


def test_3d_labels_are_supported() -> None:
    labels = np.zeros((4, 8, 8), dtype=np.int32)
    labels[1:3, 1:4, 1:4] = 1

    report = validate_segmentation(labels, labels.copy())

    assert report.metrics.matched_count == 1
    assert report.metrics.mean_matched_iou == pytest.approx(1.0)


def test_shape_mismatch_raises() -> None:
    prediction = np.zeros((5, 5), dtype=np.int32)
    reference = np.zeros((6, 6), dtype=np.int32)

    with pytest.raises(ValueError, match="does not match"):
        validate_segmentation(prediction, reference)


def test_validate_from_files_round_trips_through_tiff(tmp_path: Path) -> None:
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[1:4, 1:4] = 1
    spacing = PhysicalSpacing(x_um=0.2, y_um=0.2)
    prediction_path = tmp_path / "prediction.tif"
    reference_path = tmp_path / "reference.tif"
    save_label_mask(labels, prediction_path, spacing, "YX")
    save_label_mask(labels, reference_path, spacing, "YX")

    report = validate_segmentation_from_files(prediction_path, reference_path)

    assert report.case_id == "prediction"
    assert report.metrics.matched_count == 1


def test_batch_validates_multiple_cases(tmp_path: Path) -> None:
    labels = np.zeros((10, 10), dtype=np.int32)
    labels[1:4, 1:4] = 1
    spacing = PhysicalSpacing(x_um=0.2, y_um=0.2)
    cases = []
    for i in range(2):
        prediction_path = tmp_path / f"pred_{i}.tif"
        reference_path = tmp_path / f"ref_{i}.tif"
        save_label_mask(labels, prediction_path, spacing, "YX")
        save_label_mask(labels, reference_path, spacing, "YX")
        cases.append((f"case_{i}", prediction_path, reference_path))

    reports = validate_segmentation_batch(cases)

    assert [r.case_id for r in reports] == ["case_0", "case_1"]
    assert all(r.metrics.matched_count == 1 for r in reports)
