"""Image-level QC metrics (spec section 21).

Measurement-only, by design: this module computes numeric metrics for one
field and invents no pass/fail thresholds. Spec 21 is explicit that any
future flag derived from these numbers "should be threshold-driven from
configuration and default to measurement-only unless a threshold has been
scientifically/technically validated" -- there is deliberately no
``is_saturated``/``is_blurry``-style boolean here.

Dimension-agnostic (2D YX or 3D ZYX), like ``measurements/intensity.py``:
a min/mean/variance has no 2D/3D semantic split.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from numpy.typing import NDArray
from scipy.ndimage import laplace


def compute_image_qc_metrics(
    image: NDArray[Any],
    labels: NDArray[np.integer[Any]],
) -> dict[str, float]:
    """Compute per-field image-level QC metrics from the original channel image.

    ``image`` must be the original (non-segmentation-normalized) channel
    data, matching ``measurements/intensity.py``'s convention -- saturation
    is only meaningful against the sensor's real intensity range.
    """
    if image.shape != labels.shape:
        raise ValueError(
            f"compute_image_qc_metrics: image shape {image.shape} does not match "
            f"labels shape {labels.shape}."
        )

    image_float = image.astype(np.float64)

    if np.issubdtype(image.dtype, np.integer):
        dtype_max = np.iinfo(image.dtype).max
        saturation_fraction = float(np.mean(image == dtype_max))
    else:
        # Floating-point channel data (e.g. an already-normalized array) has
        # no fixed sensor ceiling to compare against, so saturation is
        # undefined rather than measured against a fabricated threshold.
        saturation_fraction = float("nan")

    return {
        "image_min_intensity": float(image_float.min()),
        "image_max_intensity": float(image_float.max()),
        "image_mean_intensity": float(image_float.mean()),
        "image_saturation_fraction": saturation_fraction,
        # Variance of the Laplacian: a standard, simple focus/blur proxy
        # (spec 21) -- higher variance means more high-frequency detail
        # (in focus), lower means blur. No documented "in focus" threshold.
        "image_focus_metric": float(laplace(image_float).var()),
        "image_occupied_fraction": float(np.mean(labels > 0)),
    }
