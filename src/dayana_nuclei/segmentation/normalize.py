"""Segmentation-only image normalization (spec section 13.3).

The output of this module must never be used for intensity measurement --
only for feeding a segmentation model. Primary intensity measurements read
the original image data (see measurements/intensity.py).
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray


def normalize_percentile(
    image: NDArray[Any],
    *,
    percentile_low: float,
    percentile_high: float,
) -> NDArray[np.float32]:
    """Per-field percentile clip-and-rescale to [0, 1] for segmentation input.

    Percentiles are computed over this image only (per-field strategy,
    spec 13.3's default), not across the experiment.
    """
    lo, hi = (float(v) for v in np.percentile(image, [percentile_low, percentile_high]))
    if hi <= lo:
        raise ValueError(
            f"Segmentation normalization percentiles ({percentile_low}, "
            f"{percentile_high}) produced a degenerate range (lo={lo}, hi={hi}); "
            f"the image may be blank or saturated. Inspect the source field "
            f"before proceeding."
        )
    normalized = (image.astype(np.float32) - np.float32(lo)) / np.float32(hi - lo)
    return np.clip(normalized, 0.0, 1.0, out=normalized)
