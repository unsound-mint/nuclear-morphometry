"""Per-nucleus intensity measurements (spec section 18).

``intensity_image`` must always be the original source channel data, never
the segmentation-normalized copy from ``segmentation/normalize.py`` (spec
13.3, 18: "Always measure on the original source intensity values"). The
parameter name is deliberately explicit about this rather than a generic
``image`` to make a caller passing the wrong array visually obvious at the
call site.

Dimension-agnostic by design: unlike 2D morphology, intensity statistics
have no 2D/3D semantic split (a mean is a mean), so this module accepts
either a 2D (YX) or 3D (ZYX) label/intensity pair as long as their shapes
agree.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from skimage.measure import regionprops


class NucleusIntensity(BaseModel):
    model_config = ConfigDict(frozen=True)

    object_number: int
    mean_intensity: float
    median_intensity: float
    integrated_intensity: float
    min_intensity: float
    max_intensity: float
    std_intensity: float


def measure_intensity(
    labels: NDArray[np.integer[Any]],
    intensity_image: NDArray[Any],
) -> list[NucleusIntensity]:
    """Measure per-nucleus intensity statistics on the original channel data.

    ``labels`` and ``intensity_image`` must have identical shape -- this is
    checked explicitly because passing a segmentation-normalized array by
    mistake would silently produce scientifically wrong (but
    dimensionally valid) numbers.

    Multi-channel callers (spec section 25: H3K9Ac, Lamin A/C, MitoTracker)
    call this once per channel and are responsible for labeling/prefixing
    the resulting columns with the channel name -- this function only
    knows about one intensity image and stays that way deliberately.
    """
    if labels.shape != intensity_image.shape:
        raise ValueError(
            f"measure_intensity: labels shape {labels.shape} does not match "
            f"intensity_image shape {intensity_image.shape}. Check that you are "
            f"passing the original (non-segmentation-normalized) channel image."
        )

    results: list[NucleusIntensity] = []
    for prop in regionprops(labels, intensity_image=intensity_image):
        # image_intensity is the bbox-cropped intensity image with
        # non-object pixels already zeroed by regionprops, so summing it
        # directly gives the correct masked integrated intensity.
        integrated = float(prop.image_intensity.sum())
        results.append(
            NucleusIntensity(
                object_number=int(prop.label),
                mean_intensity=float(prop.intensity_mean),
                median_intensity=float(prop.intensity_median),
                integrated_intensity=integrated,
                min_intensity=float(prop.intensity_min),
                max_intensity=float(prop.intensity_max),
                std_intensity=float(prop.intensity_std),
            )
        )
    return results
