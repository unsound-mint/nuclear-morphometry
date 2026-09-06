"""MitoTracker perinuclear spatial measurement (spec section 25.4).

Isolated and optional, per spec: "Keep this module optional and isolated
under measurements/spatial.py." MitoTracker intensity is explicitly NOT
mtDNA copy number -- this module only reports distance-banded intensity
around each nucleus and makes no claim beyond that.

Ring membership is resolved via a nearest-nucleus assignment: a
physically-calibrated Euclidean distance transform with
``return_indices=True`` gives, for every background pixel, both its
distance to the nearest nucleus and that nucleus's label. This is spec
25.4's explicit disambiguation strategy -- when perinuclear regions from
neighboring nuclei would otherwise overlap, each background pixel is
counted for exactly one nucleus (the nearest one), never double-counted.

This is a "clean v1" per spec 25.4: exactly two configured bands (a near
ring and a farther reference ring) rather than an arbitrary number of
rings, which is enough to compute the required perinuclear/farther-region
enrichment ratio.
"""

from __future__ import annotations

from typing import Any, cast

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from scipy.ndimage import distance_transform_edt

from dayana_nuclei.models import PhysicalSpacing


class NucleusPerinuclearIntensity(BaseModel):
    model_config = ConfigDict(frozen=True)

    object_number: int
    near_ring_mean_intensity: float | None
    far_ring_mean_intensity: float | None
    perinuclear_enrichment_ratio: float | None


def _spacing_sampling(spacing: PhysicalSpacing, ndim: int) -> tuple[float, ...]:
    if ndim == 3:
        return (
            spacing.require_z(context="MitoTracker perinuclear ring measurement in 3D"),
            spacing.y_um,
            spacing.x_um,
        )
    return (spacing.y_um, spacing.x_um)


def measure_perinuclear_rings(
    labels: NDArray[np.integer[Any]],
    intensity_image: NDArray[Any],
    spacing: PhysicalSpacing,
    *,
    near_ring_um: tuple[float, float],
    far_ring_um: tuple[float, float],
) -> list[NucleusPerinuclearIntensity]:
    """Measure near/far perinuclear ring intensity per nucleus.

    ``near_ring_um`` and ``far_ring_um`` are (start, end) distance bands in
    micrometers from the nuclear boundary, and must be ordered and
    non-overlapping, increasing outward (``0 <= near_start < near_end <=
    far_start < far_end``). A nucleus with no background pixel in a band
    (e.g. crowded out entirely by a closer neighbor) reports ``None`` for
    that band's mean, and the ratio, rather than a fabricated value.

    Note: this recomputes a full-image boolean mask per object, so cost
    scales with ``n_objects * image_size``. Acceptable for the validation-
    and thesis-scale fields this targets; revisit if profiling on a much
    larger dataset shows otherwise (see AGENTS.md: profile before
    optimizing).
    """
    if labels.shape != intensity_image.shape:
        raise ValueError(
            f"measure_perinuclear_rings: labels shape {labels.shape} does not match "
            f"intensity_image shape {intensity_image.shape}."
        )
    near_start, near_end = near_ring_um
    far_start, far_end = far_ring_um
    if not (0 <= near_start < near_end <= far_start < far_end):
        raise ValueError(
            "near_ring_um and far_ring_um must be ordered, non-overlapping bands "
            f"increasing outward from the nuclear boundary: got near={near_ring_um}, "
            f"far={far_ring_um}."
        )

    object_labels = sorted(int(v) for v in np.unique(labels) if v != 0)
    if not object_labels:
        return []

    background = labels == 0
    sampling = _spacing_sampling(spacing, labels.ndim)
    # scipy ships no type stubs; distance_transform_edt's return type is
    # inferred as a union pyright can't unpack, even though passing
    # return_indices=True unambiguously makes this call return a
    # (distances, indices) tuple -- a documented stub gap, not a
    # correctness escape (see the same pattern in measurements/lamin.py).
    distance, nearest_index = cast(
        "tuple[NDArray[np.float64], NDArray[np.intp]]",
        distance_transform_edt(background, sampling=sampling, return_indices=True),
    )
    nearest_label = labels[tuple(nearest_index)]

    results: list[NucleusPerinuclearIntensity] = []
    for label in object_labels:
        owned = background & (nearest_label == label)
        near_mask = owned & (distance >= near_start) & (distance < near_end)
        far_mask = owned & (distance >= far_start) & (distance < far_end)

        near_mean = float(intensity_image[near_mask].mean()) if near_mask.any() else None
        far_mean = float(intensity_image[far_mask].mean()) if far_mask.any() else None
        ratio = (
            near_mean / far_mean
            if near_mean is not None and far_mean is not None and far_mean != 0
            else None
        )
        results.append(
            NucleusPerinuclearIntensity(
                object_number=label,
                near_ring_mean_intensity=near_mean,
                far_ring_mean_intensity=far_mean,
                perinuclear_enrichment_ratio=ratio,
            )
        )
    return results
