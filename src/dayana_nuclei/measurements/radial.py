"""2D radial intensity distribution (spec section 20).

Deliberately does NOT attempt to replicate CellProfiler's internal radial-
distribution algorithm -- spec 20 explicitly requires that this measurement
have "its own documented definition and validation" rather than blind
parity with an unclear legacy method (this project's whole premise per
spec section 3 is fixing problems with that legacy workflow, not
reproducing its opacity).

Definition: for each object, a per-pixel **normalized distance from the
boundary** is computed as ``1 - distance_to_edge / max_distance_to_edge``,
where ``distance_to_edge`` is a Euclidean distance transform of the
object's own mask (distance from each interior pixel to the nearest
background pixel). This is 0 at the object's deepest interior point(s) and
1 at the boundary. It requires no geometric center and is well-defined for
irregular/non-convex shapes -- consistent with this project's invariant
that a nucleus is never assumed to be a circle or ellipse (AGENTS.md,
docs/decisions/0002).

``[0, 1]`` is divided into ``radial_bins`` equal-width bins (default 5,
matching spec 20's stated legacy parity target); bin 0 is the innermost
bin, the last bin touches the boundary.

2D only, per spec 20's explicit title and its closing sentence ("Do not
enable a 3D radial-distribution equivalent by default... it must have its
own documented definition"). This module has no ndim==3 code path at all.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops

_BIN_PROPERTIES: tuple[str, ...] = ("mean_intensity", "frac_intensity", "frac_pixels")


class NucleusRadialDistribution(BaseModel):
    """One object's radial-bin columns. Extra fields hold
    ``radial_bin{i}_{mean_intensity,frac_intensity,frac_pixels}``."""

    model_config = ConfigDict(frozen=True, extra="allow")

    object_number: int


def radial_bin_columns(radial_bins: int) -> tuple[str, ...]:
    """The full ordered list of radial-bin column names for a given bin count."""
    return tuple(f"radial_bin{b}_{prop}" for b in range(radial_bins) for prop in _BIN_PROPERTIES)


def measure_radial_distribution_2d(
    labels: NDArray[np.integer[Any]],
    intensity_image: NDArray[Any],
    *,
    radial_bins: int = 5,
) -> list[NucleusRadialDistribution]:
    """Measure per-bin mean intensity, intensity fraction, and pixel-count
    fraction for each object, using the original (non-segmentation-
    normalized) channel image, matching ``measurements/intensity.py``'s
    convention.

    A bin with zero pixels for a given object (possible for a very small or
    irregularly-shaped object with fewer interior "depth levels" than
    ``radial_bins``) reports ``None`` for its mean and intensity fraction
    rather than a fabricated value; ``frac_pixels`` is always a real number
    (0.0 in that case).
    """
    if labels.ndim != 2:
        raise ValueError(
            f"measure_radial_distribution_2d is 2D-only (spec section 20); got a "
            f"{labels.ndim}D label array. A 3D equivalent requires its own documented "
            f"definition (spec 20's explicit closing requirement) and does not exist here."
        )
    if labels.shape != intensity_image.shape:
        raise ValueError(
            f"measure_radial_distribution_2d: labels shape {labels.shape} does not match "
            f"intensity_image shape {intensity_image.shape}."
        )
    if radial_bins < 1:
        raise ValueError(f"radial_bins must be >= 1, got {radial_bins!r}.")

    results: list[NucleusRadialDistribution] = []
    for prop in regionprops(labels, intensity_image=intensity_image):
        mask = prop.image
        crop_intensity = prop.image_intensity

        # See measurements/lamin.py's identical padding rationale: an
        # object filling its own bounding box has no zero pixel inside the
        # crop for distance_transform_edt to measure against otherwise.
        padded_mask = np.pad(mask, pad_width=1, constant_values=False)
        padded_distance = cast("NDArray[np.float64]", distance_transform_edt(padded_mask))
        crop = tuple(slice(1, -1) for _ in range(mask.ndim))
        distance_to_edge = padded_distance[crop]

        max_distance = float(distance_to_edge[mask].max())
        normalized_distance_from_edge = 1.0 - (distance_to_edge / max_distance)
        bin_index = np.clip(
            (normalized_distance_from_edge * radial_bins).astype(np.int64), 0, radial_bins - 1
        )

        total_intensity = float(crop_intensity[mask].sum())
        total_pixels = int(mask.sum())

        values: dict[str, Any] = {}
        for b in range(radial_bins):
            bin_mask = mask & (bin_index == b)
            bin_pixel_count = int(bin_mask.sum())
            values[f"radial_bin{b}_frac_pixels"] = bin_pixel_count / total_pixels
            if bin_pixel_count > 0:
                bin_intensity_sum = float(crop_intensity[bin_mask].sum())
                values[f"radial_bin{b}_mean_intensity"] = bin_intensity_sum / bin_pixel_count
                values[f"radial_bin{b}_frac_intensity"] = (
                    bin_intensity_sum / total_intensity if total_intensity != 0 else None
                )
            else:
                values[f"radial_bin{b}_mean_intensity"] = None
                values[f"radial_bin{b}_frac_intensity"] = None

        results.append(NucleusRadialDistribution(object_number=int(prop.label), **values))
    return results
