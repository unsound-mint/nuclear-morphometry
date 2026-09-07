"""Deterministic segmentation backend for architecture/integration tests.

Not scientifically valid for real data -- no learned model, and it cannot
separate touching nuclei. Selected via ``segmentation.backend = "fixture"``.
Exists so the pipeline architecture (spec section 36.3) can be exercised
end-to-end without Cellpose or a GPU.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from skimage.filters import threshold_otsu
from skimage.measure import label

from nuclear_morphometry.models import PhysicalSpacing, SegmentationResult
from nuclear_morphometry.segmentation.base import validate_label_image


class FixtureSegmenter:
    """Otsu threshold + connected-component labeling. Deterministic for a given image."""

    def __init__(self, *, threshold: float | None = None) -> None:
        self._fixed_threshold = threshold

    def segment(
        self,
        image: NDArray[Any],
        spacing: PhysicalSpacing,
    ) -> SegmentationResult:
        threshold = (
            self._fixed_threshold if self._fixed_threshold is not None else threshold_otsu(image)
        )
        binary = image > threshold
        # skimage ships no type stubs; pyright infers label()'s return type from its
        # body across all `return_num` branches, producing a bogus tuple|int union
        # even with return_num=False passed explicitly. The runtime return here is
        # unambiguous (an ndarray), so this cast documents a real stub gap, not a
        # correctness escape hatch.
        raw_labels = cast(
            "NDArray[np.integer[Any]]",
            label(binary, return_num=False, connectivity=image.ndim),
        )
        labels = raw_labels.astype(np.int32)
        validate_label_image(labels, expected_ndim=image.ndim)
        return SegmentationResult(
            labels=labels,
            backend="fixture",
            model_id="connected-components-otsu-v1",
            backend_metadata={"threshold": float(threshold)},
        )
