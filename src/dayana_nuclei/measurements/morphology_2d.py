"""2D nucleus morphology (spec section 16).

Perimeter uses ``skimage.measure.regionprops`` ``perimeter``, which
approximates contour length via the Vossepoel & Smeulders pixel-boundary
estimator (a weighted count of boundary pixel transitions), not sub-pixel
contour tracing. This matches the style of perimeter estimate used by the
legacy CellProfiler workflow (also pixel-count based), which is why it is
the default here -- see spec section 2's continuity requirement.

Major/minor axis length and perimeter are computed from image moments in
pixel space, which implicitly assumes an isotropic pixel grid. When
converting to physical units we therefore use the geometric mean of the X/Y
pixel sizes (``sqrt(x_um * y_um)``) rather than either axis alone, which is
exact when pixels are square and a documented approximation otherwise.
``area_um2`` and physical centroid coordinates do not need this
approximation and use the true per-axis calibration directly.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict
from skimage.measure import regionprops

from dayana_nuclei.models import PhysicalSpacing


class Nucleus2DMorphology(BaseModel):
    model_config = ConfigDict(frozen=True)

    object_number: int
    area_px: float
    area_um2: float
    perimeter_px: float
    perimeter_um: float
    circularity: float
    form_factor: float
    solidity: float
    eccentricity: float
    major_axis_um: float
    minor_axis_um: float
    aspect_ratio: float
    extent: float
    centroid_row_px: float
    centroid_col_px: float
    centroid_y_um: float
    centroid_x_um: float
    touches_border: bool


def measure_2d_morphology(
    labels: NDArray[np.integer[Any]],
    spacing: PhysicalSpacing,
) -> list[Nucleus2DMorphology]:
    if labels.ndim != 2:
        raise ValueError(f"measure_2d_morphology requires a 2D label image, got ndim={labels.ndim}")

    isotropic_px_um = math.sqrt(spacing.x_um * spacing.y_um)
    height, width = labels.shape
    results: list[Nucleus2DMorphology] = []

    for prop in regionprops(labels):
        area_px = float(prop.area)
        perimeter_px = float(prop.perimeter)
        circularity = (4.0 * math.pi * area_px / (perimeter_px**2)) if perimeter_px > 0 else 0.0
        major_px = float(prop.axis_major_length)
        minor_px = float(prop.axis_minor_length)
        min_row, min_col, max_row, max_col = prop.bbox
        touches_border = min_row == 0 or min_col == 0 or max_row == height or max_col == width

        results.append(
            Nucleus2DMorphology(
                object_number=int(prop.label),
                area_px=area_px,
                area_um2=area_px * spacing.x_um * spacing.y_um,
                perimeter_px=perimeter_px,
                perimeter_um=perimeter_px * isotropic_px_um,
                circularity=circularity,
                form_factor=circularity,
                solidity=float(prop.solidity),
                eccentricity=float(prop.eccentricity),
                major_axis_um=major_px * isotropic_px_um,
                minor_axis_um=minor_px * isotropic_px_um,
                aspect_ratio=(major_px / minor_px) if minor_px > 0 else float("inf"),
                extent=float(prop.extent),
                centroid_row_px=float(prop.centroid[0]),
                centroid_col_px=float(prop.centroid[1]),
                centroid_y_um=float(prop.centroid[0]) * spacing.y_um,
                centroid_x_um=float(prop.centroid[1]) * spacing.x_um,
                touches_border=touches_border,
            )
        )
    return results
