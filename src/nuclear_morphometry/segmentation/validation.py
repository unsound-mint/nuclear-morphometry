"""Segmentation validation against a reference mask (spec section 14).

This is a scientific gate, not optional tooling: it exists to catch
segmentation-quality problems like the Cellpose-SAM 3D over-segmentation
finding in docs/decisions/0008 with real numbers instead of eyeballed
object counts.

Matching is a transparent IoU matrix + Hungarian (linear sum) assignment
(spec 14.2's explicit preference over a black-box metric package), computed
identically for 2D (YX) or 3D (ZYX) label arrays -- overlap counting has no
2D/3D semantic split.

No universal IoU threshold is hard-coded as scientific truth (spec 14.3):
``iou_threshold`` is always an explicit, documented parameter, and this
module only computes metrics. Deciding whether a model is "accepted" still
requires a human to visually review representative overlays and record the
decision in a decision record (spec 14.3) -- that human step is
deliberately not automated here.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from scipy.optimize import linear_sum_assignment


class MatchResult(BaseModel):
    model_config = ConfigDict(frozen=True)

    prediction_label: int | None
    reference_label: int | None
    iou: float | None
    is_true_positive: bool


class ValidationMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    predicted_object_count: int
    reference_object_count: int
    iou_threshold: float
    matched_count: int
    mean_matched_iou: float | None
    median_matched_iou: float | None
    # Precision is undefined (None), not 0.0, when there are zero predictions
    # to score (0/0) -- same for recall with zero reference objects. Treating
    # "no objects to evaluate" as a false 0.0 would misreport a model that
    # correctly found nothing in an empty field as having failed.
    precision: float | None
    recall: float | None
    f1: float | None
    unmatched_prediction_count: int
    unmatched_reference_count: int


class SplitMergeEstimate(BaseModel):
    """Heuristic over-/under-segmentation estimate (spec 22.2).

    Independent of the one-to-one Hungarian matching above: this looks at
    *all* overlaps above ``containment_threshold`` (not just the best
    assignment), so a reference object claimed by several prediction
    objects reports as a probable split, and a prediction object covering
    several reference objects reports as a probable merge. Disabled by
    default (``validate_segmentation(estimate_split_merge=False)``) and
    always reported as labels to preserve, not as a filter -- spec 22.2
    requires this stay a documented heuristic that never deletes rows.

    Uses containment (intersection / min(pred_area, ref_area)), not IoU:
    for a reference object fragmented into n roughly-equal prediction
    pieces, each fragment's IoU with the reference is ~1/n and drops below
    any fixed IoU threshold as fragmentation gets worse -- exactly the
    Cellpose-SAM 3D failure mode in docs/decisions/0008 (25-434 fragments
    of one sphere). Containment stays ~1.0 for a fragment fully inside the
    reference regardless of fragment count, so it is the metric that
    actually detects this pathology.
    """

    model_config = ConfigDict(frozen=True)

    containment_threshold: float
    probable_split_reference_labels: tuple[int, ...]
    probable_merge_prediction_labels: tuple[int, ...]


class SegmentationValidationReport(BaseModel):
    model_config = ConfigDict(frozen=True)

    case_id: str
    metrics: ValidationMetrics
    matches: tuple[MatchResult, ...]
    split_merge: SplitMergeEstimate | None


def _object_areas(labels: NDArray[np.integer[Any]]) -> dict[int, int]:
    values, counts = np.unique(labels, return_counts=True)
    return {int(v): int(c) for v, c in zip(values, counts, strict=True) if v != 0}


def _overlap_matrices(
    prediction: NDArray[np.integer[Any]],
    reference: NDArray[np.integer[Any]],
    pred_labels: list[int],
    ref_labels: list[int],
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return (iou_matrix, containment_matrix) from one pass over the overlap.

    IoU (intersection / union) is what Hungarian matching uses -- it
    penalizes a partial or oversized prediction the way detection metrics
    should. Containment (intersection / min(pred_area, ref_area)) is what
    the split/merge heuristic uses -- see SplitMergeEstimate's docstring
    for why IoU cannot detect fragmentation.
    """
    pred_areas = _object_areas(prediction)
    ref_areas = _object_areas(reference)
    pred_index = {label: i for i, label in enumerate(pred_labels)}
    ref_index = {label: i for i, label in enumerate(ref_labels)}
    iou = np.zeros((len(pred_labels), len(ref_labels)), dtype=np.float64)
    containment = np.zeros_like(iou)

    overlap = (prediction != 0) & (reference != 0)
    if not np.any(overlap):
        return iou, containment

    pairs = np.stack(
        [prediction[overlap].astype(np.int64), reference[overlap].astype(np.int64)], axis=1
    )
    unique_pairs, counts = np.unique(pairs, axis=0, return_counts=True)
    for (pred_label, ref_label), intersection in zip(
        unique_pairs.tolist(), counts.tolist(), strict=True
    ):
        pred_area = pred_areas[pred_label]
        ref_area = ref_areas[ref_label]
        i, j = pred_index[pred_label], ref_index[ref_label]
        union = pred_area + ref_area - intersection
        if union > 0:
            iou[i, j] = intersection / union
        min_area = min(pred_area, ref_area)
        if min_area > 0:
            containment[i, j] = intersection / min_area
    return iou, containment


def _precision_recall_f1(
    true_positives: int, predicted_count: int, reference_count: int
) -> tuple[float | None, float | None, float | None]:
    precision = true_positives / predicted_count if predicted_count > 0 else None
    recall = true_positives / reference_count if reference_count > 0 else None
    if precision is None or recall is None:
        return precision, recall, None
    f1 = 0.0 if (precision + recall) == 0 else 2 * precision * recall / (precision + recall)
    return precision, recall, f1


def _estimate_split_merge(
    containment_matrix: NDArray[np.float64],
    pred_labels: list[int],
    ref_labels: list[int],
    *,
    containment_threshold: float,
) -> SplitMergeEstimate:
    overlaps = containment_matrix > containment_threshold
    probable_splits = tuple(
        sorted(ref_labels[j] for j in range(len(ref_labels)) if overlaps[:, j].sum() > 1)
    )
    probable_merges = tuple(
        sorted(pred_labels[i] for i in range(len(pred_labels)) if overlaps[i, :].sum() > 1)
    )
    return SplitMergeEstimate(
        containment_threshold=containment_threshold,
        probable_split_reference_labels=probable_splits,
        probable_merge_prediction_labels=probable_merges,
    )


def validate_segmentation(
    prediction: NDArray[np.integer[Any]],
    reference: NDArray[np.integer[Any]],
    *,
    case_id: str = "case",
    iou_threshold: float = 0.5,
    estimate_split_merge: bool = False,
    split_merge_containment_threshold: float = 0.5,
) -> SegmentationValidationReport:
    """Compare a predicted label image against a manually-reviewed reference.

    ``iou_threshold`` decides which Hungarian-assigned pairs count as a true
    positive for precision/recall/F1 -- it is a required, explicit argument
    (default 0.5 is scikit-image/COCO convention, not a validated scientific
    claim for this dataset; spec 14.3).
    """
    if prediction.shape != reference.shape:
        raise ValueError(
            f"validate_segmentation: prediction shape {prediction.shape} does not match "
            f"reference shape {reference.shape}."
        )

    pred_labels = sorted(_object_areas(prediction))
    ref_labels = sorted(_object_areas(reference))
    iou, containment = _overlap_matrices(prediction, reference, pred_labels, ref_labels)

    matches: list[MatchResult] = []
    matched_pred_indices: set[int] = set()
    matched_ref_indices: set[int] = set()
    true_positive_ious: list[float] = []

    if pred_labels and ref_labels:
        row_ind, col_ind = linear_sum_assignment(-iou)
        for i, j in zip(row_ind.tolist(), col_ind.tolist(), strict=True):
            pair_iou = float(iou[i, j])
            if pair_iou <= 0.0:
                # Hungarian must return one entry per row even with zero
                # overlap available; a zero-IoU "assignment" is not a real
                # candidate match and both sides remain unmatched.
                continue
            is_true_positive = pair_iou >= iou_threshold
            matches.append(
                MatchResult(
                    prediction_label=pred_labels[i],
                    reference_label=ref_labels[j],
                    iou=pair_iou,
                    is_true_positive=is_true_positive,
                )
            )
            matched_pred_indices.add(i)
            matched_ref_indices.add(j)
            if is_true_positive:
                true_positive_ious.append(pair_iou)

    for i, label in enumerate(pred_labels):
        if i not in matched_pred_indices:
            matches.append(
                MatchResult(
                    prediction_label=label, reference_label=None, iou=None, is_true_positive=False
                )
            )
    for j, label in enumerate(ref_labels):
        if j not in matched_ref_indices:
            matches.append(
                MatchResult(
                    prediction_label=None, reference_label=label, iou=None, is_true_positive=False
                )
            )

    true_positives = len(true_positive_ious)
    precision, recall, f1 = _precision_recall_f1(true_positives, len(pred_labels), len(ref_labels))

    metrics = ValidationMetrics(
        predicted_object_count=len(pred_labels),
        reference_object_count=len(ref_labels),
        iou_threshold=iou_threshold,
        matched_count=true_positives,
        mean_matched_iou=float(np.mean(true_positive_ious)) if true_positive_ious else None,
        median_matched_iou=float(np.median(true_positive_ious)) if true_positive_ious else None,
        precision=precision,
        recall=recall,
        f1=f1,
        unmatched_prediction_count=len(pred_labels) - true_positives,
        unmatched_reference_count=len(ref_labels) - true_positives,
    )

    split_merge = (
        _estimate_split_merge(
            containment,
            pred_labels,
            ref_labels,
            containment_threshold=split_merge_containment_threshold,
        )
        if estimate_split_merge
        else None
    )

    return SegmentationValidationReport(
        case_id=case_id,
        metrics=metrics,
        matches=tuple(matches),
        split_merge=split_merge,
    )


def validate_segmentation_from_files(
    prediction_path: Path,
    reference_path: Path,
    *,
    case_id: str | None = None,
    iou_threshold: float = 0.5,
    estimate_split_merge: bool = False,
    split_merge_containment_threshold: float = 0.5,
) -> SegmentationValidationReport:
    """Load two label-mask TIFFs (as written by ``io.masks.save_label_mask``,
    or any integer-labeled TIFF) and validate one against the other."""
    from nuclear_morphometry.io.masks import load_label_mask

    prediction, _ = load_label_mask(prediction_path)
    reference, _ = load_label_mask(reference_path)
    return validate_segmentation(
        prediction,
        reference,
        case_id=case_id or prediction_path.stem,
        iou_threshold=iou_threshold,
        estimate_split_merge=estimate_split_merge,
        split_merge_containment_threshold=split_merge_containment_threshold,
    )


def validate_segmentation_batch(
    cases: list[tuple[str, Path, Path]],
    *,
    iou_threshold: float = 0.5,
    estimate_split_merge: bool = False,
    split_merge_containment_threshold: float = 0.5,
) -> list[SegmentationValidationReport]:
    """Validate a list of (case_id, prediction_path, reference_path) triples.

    Backs the batch CLI form (spec 14: "a batch form that consumes a
    validation manifest"). A failure loading or validating any one case
    raises immediately and aborts the whole batch -- unlike the main
    pipeline's per-field failure isolation (spec 33), this is a
    pre-production scientific gate, not a long unattended run, so a
    partial/silently-incomplete validation report is a worse outcome than
    stopping and surfacing the error.
    """
    return [
        validate_segmentation_from_files(
            prediction_path,
            reference_path,
            case_id=case_id,
            iou_threshold=iou_threshold,
            estimate_split_merge=estimate_split_merge,
            split_merge_containment_threshold=split_merge_containment_threshold,
        )
        for case_id, prediction_path, reference_path in cases
    ]
