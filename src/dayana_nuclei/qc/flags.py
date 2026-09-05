"""Object-level QC flags (spec sections 3.1, 22, 45).

Border truncation is the only automatic exclusion computed here. Shape-based
properties (eccentricity, solidity, circularity, elongation, irregularity)
must NEVER drive automatic exclusion -- see AGENTS.md and
docs/decisions/0002-no-phenotype-based-qc-filtering.md. This is the software
encoding of the thesis's central QC requirement.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from dayana_nuclei.schema import EXCLUSION_REASON_BORDER


class ObjectQCFlags(BaseModel):
    model_config = ConfigDict(frozen=True)

    qc_border: bool
    qc_manual_debris: bool = False
    qc_manual_merge: bool = False
    qc_manual_split: bool = False
    qc_manual_other: bool = False
    qc_excluded_default: bool
    qc_exclusion_reason: str | None


def compute_object_qc(
    *,
    touches_border: bool,
    flag_border_objects: bool,
    exclude_border_from_default: bool,
) -> ObjectQCFlags:
    """Compute automatic QC flags for one object.

    No shape statistic (eccentricity, solidity, circularity, ...) is an
    input to this function by design -- it only ever sees border-touch
    status. Manual annotations (debris/merge/split/other) are applied later
    by qc/annotations.py after napari review and are not computed here.
    """
    border = touches_border and flag_border_objects
    excluded = border and exclude_border_from_default
    reason = EXCLUSION_REASON_BORDER if excluded else None
    return ObjectQCFlags(
        qc_border=border,
        qc_excluded_default=excluded,
        qc_exclusion_reason=reason,
    )
