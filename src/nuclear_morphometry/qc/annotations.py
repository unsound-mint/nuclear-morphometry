"""Manual QC annotations (spec sections 22.2, 23, 24).

Annotations are a napari-independent artifact: a small JSON store under
``results/<run-id>/qc/annotations.json``, keyed by ``(image_id,
object_number)`` per spec section 23. ``qc/viewer.py`` (not yet
implemented) is expected to read/write this store; this module has no
napari or Qt dependency so it can be exercised headlessly (spec section 6)
and consumed by ``qc-report``.

Per spec section 48 (scientific result immutability) and
``docs/decisions/0004``, annotations never mutate ``nuclei.parquet`` in
place. ``apply_annotations_to_nuclei`` returns a new, derived DataFrame,
exactly parallel to how ``export.prepare_analysis`` derives
``analysis_ready.parquet`` without touching the canonical table.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import polars as pl
from pydantic import BaseModel, ConfigDict

from nuclear_morphometry.schema import (
    EXCLUSION_REASON_MANUAL_DEBRIS,
    EXCLUSION_REASON_MANUAL_MERGE,
    EXCLUSION_REASON_MANUAL_OTHER,
    EXCLUSION_REASON_MANUAL_SPLIT,
)

ManualTag = Literal["good", "debris", "merge", "split", "other"]

_TAG_TO_REASON: dict[ManualTag, str | None] = {
    "good": None,
    "debris": EXCLUSION_REASON_MANUAL_DEBRIS,
    "merge": EXCLUSION_REASON_MANUAL_MERGE,
    "split": EXCLUSION_REASON_MANUAL_SPLIT,
    "other": EXCLUSION_REASON_MANUAL_OTHER,
}


class ManualAnnotation(BaseModel):
    model_config = ConfigDict(frozen=True)

    image_id: str
    object_number: int
    tag: ManualTag
    note: str | None = None
    annotated_at: datetime


def annotations_path(run_dir: Path) -> Path:
    return run_dir / "qc" / "annotations.json"


def load_annotations(run_dir: Path) -> dict[tuple[str, int], ManualAnnotation]:
    """Load the annotation store, keyed by (image_id, object_number).

    Returns an empty dict if no annotations have been saved yet -- this is
    the normal state for a run that has not been through manual QC.
    """
    path = annotations_path(run_dir)
    if not path.exists():
        return {}
    raw = json.loads(path.read_text())
    annotations = [ManualAnnotation.model_validate(entry) for entry in raw]
    return {(a.image_id, a.object_number): a for a in annotations}


def save_annotation(
    run_dir: Path,
    *,
    image_id: str,
    object_number: int,
    tag: ManualTag,
    note: str | None = None,
) -> ManualAnnotation:
    """Upsert one annotation and persist the whole store atomically.

    Re-tagging an already-annotated object (including back to "good")
    overwrites its prior entry -- there is exactly one current tag per
    object, matching spec section 23's dock-widget model ("keyboard actions
    ... for manual tags"), not an append-only history.
    """
    annotation = ManualAnnotation(
        image_id=image_id,
        object_number=object_number,
        tag=tag,
        note=note,
        annotated_at=datetime.now(UTC),
    )
    store = load_annotations(run_dir)
    store[(image_id, object_number)] = annotation
    _write_store(run_dir, store)
    return annotation


def _write_store(run_dir: Path, store: dict[tuple[str, int], ManualAnnotation]) -> None:
    path = annotations_path(run_dir)
    path.parent.mkdir(parents=True, exist_ok=True)
    ordered = sorted(store.values(), key=lambda a: (a.image_id, a.object_number))
    payload = json.dumps([a.model_dump(mode="json") for a in ordered], indent=2)
    tmp_path = path.with_suffix(".json.tmp")
    tmp_path.write_text(payload)
    tmp_path.replace(path)


def apply_annotations_to_nuclei(nuclei_df: pl.DataFrame, run_dir: Path) -> pl.DataFrame:
    """Return a copy of ``nuclei_df`` with manual annotations folded in.

    Does not mutate ``nuclei_df`` or any file on disk. For an annotated
    object, ``qc_manual_{debris,merge,split,other}`` are set from its
    current tag (mutually exclusive; "good" clears all four) and
    ``qc_excluded_default``/``qc_exclusion_reason`` are updated -- except
    that an existing *border* exclusion is never overridden by a "good"
    manual tag: border truncation is a technical defect independent of
    manual review, so the two exclusion axes combine with border taking
    precedence when both would otherwise apply.
    """
    annotations = load_annotations(run_dir)
    if not annotations:
        return nuclei_df

    annotation_rows = [
        {"image_id": image_id, "object_number": object_number, "manual_tag": a.tag}
        for (image_id, object_number), a in annotations.items()
    ]
    annotation_df = pl.DataFrame(
        annotation_rows,
        schema={"image_id": pl.Utf8, "object_number": pl.Int64, "manual_tag": pl.Utf8},
    )

    joined = nuclei_df.join(annotation_df, on=["image_id", "object_number"], how="left")

    manual_reason = pl.col("manual_tag").replace_strict(_TAG_TO_REASON, default=None)
    was_border_excluded = pl.col("qc_exclusion_reason") == "border"
    manual_excludes = manual_reason.is_not_null()

    return (
        joined.with_columns(
            qc_manual_debris=(pl.col("manual_tag") == "debris").fill_null(False),
            qc_manual_merge=(pl.col("manual_tag") == "merge").fill_null(False),
            qc_manual_split=(pl.col("manual_tag") == "split").fill_null(False),
            qc_manual_other=(pl.col("manual_tag") == "other").fill_null(False),
        )
        .with_columns(
            qc_exclusion_reason=pl.when(pl.col("manual_tag").is_null())
            .then(pl.col("qc_exclusion_reason"))
            .when(was_border_excluded)
            .then(pl.col("qc_exclusion_reason"))
            .otherwise(manual_reason),
        )
        .with_columns(
            qc_excluded_default=pl.col("qc_excluded_default")
            | (manual_excludes.fill_null(False) & ~was_border_excluded),
        )
        .drop("manual_tag")
    )
