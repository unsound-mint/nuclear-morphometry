"""Per-nucleus 2D texture measurements (spec section 19).

2D only, by contract, not just by convention (spec 19.1): 3D texture has no
validated biological interpretation yet and this module has no ndim==3 code
path at all -- ``measure_texture_2d`` rejects a 3D label array outright
rather than silently squeezing or projecting it.

Column naming: because the set of configured distances is user-controlled
(spec 19.4: legacy pixel mode or physical-scale mode, resolved to a plain
pixel-distance list by the caller), :class:`NucleusTexture2D` cannot have a
fixed field for every distance ahead of time. It allows extra fields and
each result carries dynamically-named columns ``{property}_d{distance_px}``,
e.g. ``contrast_d3``, ``entropy_d10``.

Masked GLCM: ``skimage.feature.graycomatrix`` has no notion of an object
mask -- it treats every pixel pair in the array as valid. Computing texture
naively over an object's bounding-box crop would therefore mix in
background/neighboring-object pixels at the boundary. This module instead
quantizes the masked region into ``gray_levels`` bins and assigns
everything outside the mask a dedicated sentinel level (index
``gray_levels``, i.e. one level beyond the valid quantized range), computes
the GLCM over ``gray_levels + 1`` levels, then discards the sentinel's row
and column and renormalizes before computing any property. This yields
texture computed only from foreground-foreground pixel pairs.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import regionprops

_DEFAULT_ANGLES: tuple[float, ...] = (0.0, math.pi / 4, math.pi / 2, 3 * math.pi / 4)
_GRAYCOPROPS_PROPERTIES: tuple[str, ...] = ("contrast", "homogeneity", "correlation", "energy")


class NucleusTexture2D(BaseModel):
    """One object's texture columns. Extra fields hold ``{property}_d{distance}``."""

    model_config = ConfigDict(frozen=True, extra="allow")

    object_number: int


def quantize_for_texture(
    intensity_image: NDArray[Any],
    mask: NDArray[np.bool_],
    *,
    gray_levels: int,
) -> NDArray[np.uint16]:
    """Quantize an intensity crop into ``[0, gray_levels - 1]`` using the masked
    region's own min/max range.

    This is **per-object** min-max quantization: each nucleus is rescaled to
    its own intensity range, not the field's or experiment's range. This
    destroys between-object intensity comparability by construction (a dim
    and a bright nucleus with the same internal texture pattern will
    quantize identically) -- spec 19.3 requires this tradeoff be stated
    explicitly rather than silently chosen. It is still the right default
    for *within-object* co-occurrence texture (contrast, entropy, ...),
    where the pattern of local intensity variation, not the absolute
    brightness, is what these properties are designed to capture; the
    absolute-intensity comparison across objects is already covered by
    ``measurements/intensity.py``.

    Values outside ``mask`` are computed but not meaningful -- callers that
    need per-pixel validity must consult ``mask`` separately.
    """
    masked_values = intensity_image[mask]
    if masked_values.size == 0:
        raise ValueError("quantize_for_texture: mask has no True pixels.")

    lo = float(masked_values.min())
    hi = float(masked_values.max())
    if hi <= lo:
        return np.zeros(intensity_image.shape, dtype=np.uint16)

    scaled = (intensity_image.astype(np.float64) - lo) / (hi - lo)
    quantized = np.clip(np.round(scaled * (gray_levels - 1)), 0, gray_levels - 1)
    return quantized.astype(np.uint16)


def um_distances_to_pixels(distances_um: list[float], pixel_size_um: float) -> list[int]:
    """Convert physical-scale texture distances (spec 19.4) to pixel offsets.

    Raises if a configured distance rounds to less than 1 pixel (spec 7.3:
    "A configured texture scale converts to <1 pixel" is a named fail-loud
    boundary) -- silently rounding up to 1 would misrepresent the
    researcher's intended physical scale.
    """
    pixel_distances: list[int] = []
    for distance_um in distances_um:
        distance_px = distance_um / pixel_size_um
        rounded = round(distance_px)
        if rounded < 1:
            raise ValueError(
                f"Configured texture distance {distance_um} um converts to "
                f"{distance_px:.3f} pixels at {pixel_size_um} um/pixel, which "
                f"rounds to less than 1 pixel.\n\n"
                f"Increase measurements.texture_distances_um or acquire at "
                f"higher spatial resolution; the pipeline will not silently "
                f"substitute a 1-pixel distance for an unmeasurable scale."
            )
        pixel_distances.append(int(rounded))
    return pixel_distances


def _entropy(probabilities: NDArray[np.float64]) -> float:
    """Shannon entropy of a normalized 2D probability matrix, base 2, 0 for a
    degenerate (all-zero, e.g. no valid pixel pairs) matrix."""
    nonzero = probabilities[probabilities > 0]
    if nonzero.size == 0:
        return 0.0
    return float(-np.sum(nonzero * np.log2(nonzero)))


def measure_texture_2d(
    labels: NDArray[np.integer[Any]],
    intensity_image: NDArray[Any],
    *,
    distances_px: list[int],
    gray_levels: int = 256,
    angles: list[float] | None = None,
) -> list[NucleusTexture2D]:
    """Measure GLCM-based 2D texture per nucleus, averaged over ``angles``.

    ``angles`` defaults to the four standard GLCM directions (0, 45, 90,
    135 degrees); each property is averaged across angles per distance, a
    standard rotation-invariance convention for this kind of texture
    measure. Entropy is not one of skimage's ``graycoprops`` properties in
    this version, so it is computed directly here as
    ``-sum(p * log2(p))`` over the same (masked, renormalized) GLCM used for
    the other properties, at base 2.
    """
    if labels.ndim != 2:
        raise ValueError(f"measure_texture_2d requires a 2D label image, got ndim={labels.ndim}")

    active_angles = list(angles) if angles is not None else list(_DEFAULT_ANGLES)
    results: list[NucleusTexture2D] = []

    for prop in regionprops(labels):
        min_row, min_col, max_row, max_col = prop.bbox
        crop_intensity = intensity_image[min_row:max_row, min_col:max_col]
        mask = prop.image

        quantized = quantize_for_texture(crop_intensity, mask, gray_levels=gray_levels)
        sentinel = gray_levels
        glcm_input = np.where(mask, quantized, sentinel).astype(np.uint16)

        glcm = graycomatrix(
            glcm_input,
            distances=distances_px,
            angles=active_angles,
            levels=gray_levels + 1,
            symmetric=True,
            normed=True,
        )
        foreground_glcm = glcm[:gray_levels, :gray_levels, :, :].astype(np.float64)
        sums = foreground_glcm.sum(axis=(0, 1), keepdims=True)
        normalized_glcm = np.divide(
            foreground_glcm, sums, out=np.zeros_like(foreground_glcm), where=sums > 0
        )

        columns: dict[str, float] = {}
        for prop_name in _GRAYCOPROPS_PROPERTIES:
            # (n_distances, n_angles) -> average over angles per distance.
            per_distance_angle = np.nan_to_num(graycoprops(normalized_glcm, prop_name), nan=0.0)
            per_distance = per_distance_angle.mean(axis=1)
            for distance_px, value in zip(distances_px, per_distance, strict=True):
                columns[f"{prop_name}_d{distance_px}"] = float(value)

        for distance_index, distance_px in enumerate(distances_px):
            entropies = [
                _entropy(normalized_glcm[:, :, distance_index, angle_index])
                for angle_index in range(len(active_angles))
            ]
            columns[f"entropy_d{distance_px}"] = float(np.mean(entropies))

        results.append(NucleusTexture2D(object_number=int(prop.label), **columns))

    return results
