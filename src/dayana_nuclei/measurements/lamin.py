"""Lamin A/C shell/core intensity (spec section 25.3).

The nuclear mask is reused from validated Hoechst segmentation, never
re-segmented (spec 25's opening requirement) -- callers pass the same
``labels`` used for morphology/Hoechst intensity.

The shell is defined in **physical units**, per spec 25.3's explicit
requirement, not a fixed pixel count: for each object, a Euclidean distance
transform from the object's boundary (sampled with the run's real pixel/
voxel spacing) gives a physically-correct distance-to-boundary even under
anisotropic 3D spacing, where a fixed-pixel erosion would bite unevenly
into Z versus X/Y.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from scipy.ndimage import distance_transform_edt
from skimage.measure import regionprops

from dayana_nuclei.models import PhysicalSpacing


class NucleusLaminIntensity(BaseModel):
    model_config = ConfigDict(frozen=True)

    object_number: int
    total_mean_intensity: float
    shell_mean_intensity: float | None
    core_mean_intensity: float | None
    shell_core_ratio: float | None


def _spacing_sampling(spacing: PhysicalSpacing, ndim: int) -> tuple[float, ...]:
    if ndim == 3:
        return (
            spacing.require_z(context="Lamin A/C shell/core measurement in 3D"),
            spacing.y_um,
            spacing.x_um,
        )
    return (spacing.y_um, spacing.x_um)


def measure_lamin_shell_core(
    labels: NDArray[np.integer[Any]],
    intensity_image: NDArray[Any],
    spacing: PhysicalSpacing,
    *,
    shell_width_um: float,
) -> list[NucleusLaminIntensity]:
    """Measure total/shell/core Lamin A/C intensity per nucleus.

    ``intensity_image`` must be the original (non-segmentation-normalized)
    Lamin A/C channel, matching ``measurements/intensity.py``'s convention.
    An object smaller than ``shell_width_um`` everywhere has no pixel deep
    enough to count as core -- its entire mask is reported as shell, and
    ``core_mean_intensity``/``shell_core_ratio`` are ``None`` rather than a
    fabricated value.
    """
    if labels.shape != intensity_image.shape:
        raise ValueError(
            f"measure_lamin_shell_core: labels shape {labels.shape} does not match "
            f"intensity_image shape {intensity_image.shape}."
        )
    if shell_width_um <= 0:
        raise ValueError(f"shell_width_um must be > 0, got {shell_width_um!r}.")

    sampling = _spacing_sampling(spacing, labels.ndim)

    results: list[NucleusLaminIntensity] = []
    for prop in regionprops(labels, intensity_image=intensity_image):
        mask = prop.image
        crop_intensity = prop.image_intensity

        # regionprops' bounding box is the tightest box containing the
        # object, so an axis-aligned object (e.g. rectangular debris, or a
        # small rasterized blob that happens to fill its own bbox) can have
        # True on every border pixel of `mask`, leaving no zero pixel for
        # distance_transform_edt to measure against -- it silently produces
        # meaningless distances in that case (confirmed empirically). Pad
        # with one False voxel on every side so a real object boundary
        # always exists just outside the object, then crop the padding back
        # off before using the result.
        padded_mask = np.pad(mask, pad_width=1, constant_values=False)
        # scipy ships no type stubs; distance_transform_edt's return type is
        # inferred as a union that pyright can't index with a slice tuple,
        # even though it unambiguously returns a float ndarray here
        # (return_distances defaults to True, return_indices defaults to
        # False) -- a documented stub gap, not a correctness escape.
        padded_distance = cast(
            "NDArray[np.float64]", distance_transform_edt(padded_mask, sampling=sampling)
        )
        crop = tuple(slice(1, -1) for _ in range(mask.ndim))
        distance_inside = padded_distance[crop]
        core_mask = mask & (distance_inside > shell_width_um)
        shell_mask = mask & ~core_mask

        shell_mean = float(crop_intensity[shell_mask].mean()) if shell_mask.any() else None
        core_mean = float(crop_intensity[core_mask].mean()) if core_mask.any() else None
        ratio = (
            shell_mean / core_mean
            if shell_mean is not None and core_mean is not None and core_mean != 0
            else None
        )

        results.append(
            NucleusLaminIntensity(
                object_number=int(prop.label),
                total_mean_intensity=float(prop.intensity_mean),
                shell_mean_intensity=shell_mean,
                core_mean_intensity=core_mean,
                shell_core_ratio=ratio,
            )
        )
    return results
