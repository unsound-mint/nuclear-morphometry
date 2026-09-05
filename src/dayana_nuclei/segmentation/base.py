"""Segmentation backend contract.

All segmentation backends (Cellpose, the deterministic fixture backend used
in tests) implement :class:`Segmenter`. Downstream measurement and QC code
must depend only on this protocol and on :class:`~dayana_nuclei.models.SegmentationResult`
-- never on a specific backend's internals.
"""

from __future__ import annotations

from typing import Any, Protocol, runtime_checkable

import numpy as np
from numpy.typing import NDArray

from dayana_nuclei.models import PhysicalSpacing, SegmentationResult


@runtime_checkable
class Segmenter(Protocol):
    """A nucleus segmentation backend.

    ``image`` is a single-channel ``YX`` or ``ZYX`` array already selected
    for the Hoechst/DNA channel. Implementations must not silently project
    a 3D volume to 2D or vice versa -- the caller decides the mode.
    """

    def segment(
        self,
        image: NDArray[Any],
        spacing: PhysicalSpacing,
    ) -> SegmentationResult: ...


class SegmenterUnavailableError(RuntimeError):
    """Raised when a configured segmentation backend cannot run as requested.

    Examples: CUDA requested but unavailable, or ``model = "auto"`` with no
    validated default yet selected. Backends must raise this rather than
    silently falling back to a different device or model.
    """


def validate_label_image(labels: NDArray[np.integer[Any]], *, expected_ndim: int) -> None:
    """Boundary check applied to every SegmentationResult before it leaves a backend."""
    if labels.ndim != expected_ndim:
        raise ValueError(
            f"Segmentation backend returned labels with ndim={labels.ndim}, "
            f"expected {expected_ndim} for this analysis mode."
        )
    if labels.min() < 0:
        raise ValueError("Segmentation backend returned negative label values.")
