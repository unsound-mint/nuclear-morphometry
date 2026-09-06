from pathlib import Path

import polars as pl
import pytest

from dayana_nuclei.qc.annotations import (
    apply_annotations_to_nuclei,
    load_annotations,
    save_annotation,
)


def test_load_annotations_empty_when_no_store(tmp_path: Path) -> None:
    assert load_annotations(tmp_path) == {}


def test_save_and_reload_annotation(tmp_path: Path) -> None:
    save_annotation(tmp_path, image_id="field_a", object_number=3, tag="debris", note="fuzzy")

    reloaded = load_annotations(tmp_path)

    assert set(reloaded) == {("field_a", 3)}
    annotation = reloaded[("field_a", 3)]
    assert annotation.tag == "debris"
    assert annotation.note == "fuzzy"


def test_re_annotating_overwrites_prior_tag(tmp_path: Path) -> None:
    save_annotation(tmp_path, image_id="field_a", object_number=3, tag="debris")
    save_annotation(tmp_path, image_id="field_a", object_number=3, tag="good")

    reloaded = load_annotations(tmp_path)

    assert len(reloaded) == 1
    assert reloaded[("field_a", 3)].tag == "good"


def _base_nuclei_df() -> pl.DataFrame:
    return pl.DataFrame(
        {
            "image_id": ["field_a", "field_a", "field_a"],
            "object_number": [1, 2, 3],
            "qc_border": [False, True, False],
            "qc_manual_debris": [False, False, False],
            "qc_manual_merge": [False, False, False],
            "qc_manual_split": [False, False, False],
            "qc_manual_other": [False, False, False],
            "qc_excluded_default": [False, True, False],
            "qc_exclusion_reason": [None, "border", None],
        }
    )


def test_apply_annotations_is_noop_without_a_store(tmp_path: Path) -> None:
    nuclei_df = _base_nuclei_df()

    result = apply_annotations_to_nuclei(nuclei_df, tmp_path)

    assert result.equals(nuclei_df)


def test_apply_annotations_excludes_manually_flagged_object(tmp_path: Path) -> None:
    save_annotation(tmp_path, image_id="field_a", object_number=1, tag="split")
    nuclei_df = _base_nuclei_df()

    result = apply_annotations_to_nuclei(nuclei_df, tmp_path)
    row = result.filter(pl.col("object_number") == 1).row(0, named=True)

    assert row["qc_manual_split"] is True
    assert row["qc_manual_debris"] is False
    assert row["qc_excluded_default"] is True
    assert row["qc_exclusion_reason"] == "manual_split"


def test_apply_annotations_never_mutates_input(tmp_path: Path) -> None:
    save_annotation(tmp_path, image_id="field_a", object_number=1, tag="debris")
    nuclei_df = _base_nuclei_df()
    original = nuclei_df.clone()

    apply_annotations_to_nuclei(nuclei_df, tmp_path)

    assert nuclei_df.equals(original)


def test_border_exclusion_reason_takes_precedence_over_a_good_tag(tmp_path: Path) -> None:
    """A border-truncated object stays excluded with reason 'border' even after
    being manually reviewed as 'good' -- border is a technical defect independent
    of manual review, per spec section 22.1."""
    save_annotation(tmp_path, image_id="field_a", object_number=2, tag="good")
    nuclei_df = _base_nuclei_df()

    result = apply_annotations_to_nuclei(nuclei_df, tmp_path)
    row = result.filter(pl.col("object_number") == 2).row(0, named=True)

    assert row["qc_excluded_default"] is True
    assert row["qc_exclusion_reason"] == "border"
    assert all(
        row[c] is False
        for c in ("qc_manual_debris", "qc_manual_merge", "qc_manual_split", "qc_manual_other")
    )


def test_good_tag_leaves_a_non_border_object_included(tmp_path: Path) -> None:
    save_annotation(tmp_path, image_id="field_a", object_number=1, tag="good")
    nuclei_df = _base_nuclei_df()

    result = apply_annotations_to_nuclei(nuclei_df, tmp_path)
    row = result.filter(pl.col("object_number") == 1).row(0, named=True)

    assert row["qc_excluded_default"] is False
    assert row["qc_exclusion_reason"] is None


def test_invalid_tag_is_rejected(tmp_path: Path) -> None:
    with pytest.raises(ValueError):
        save_annotation(tmp_path, image_id="field_a", object_number=1, tag="not_a_real_tag")  # type: ignore[arg-type]
