"""Label-mask persistence (spec section 15).

Masks are written atomically (temp file + rename) since a long pipeline run
must never leave a half-written mask on disk after a crash (spec section 50).
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Literal

import numpy as np
import tifffile
from numpy.typing import NDArray

from dayana_nuclei.models import PhysicalSpacing

Axes = Literal["YX", "ZYX"]


def mask_path_for(run_dir: Path, image_id: str) -> Path:
    """Deterministic mask path for one image_id within a run directory."""
    return run_dir / "masks" / f"{image_id}_labels.tif"


def save_label_mask(
    labels: NDArray[np.integer[Any]],
    path: Path,
    spacing: PhysicalSpacing,
    axes: Axes,
    *,
    compress: bool = False,
) -> None:
    """Write an integer label image, preserving axes and physical spacing.

    Spacing and axes are stored as custom keys in tifffile's shaped-format
    ImageDescription JSON (not the ImageJ convention), since we control both
    the writer and reader here and this avoids relying on ImageJ-specific
    unit semantics. See docs/decisions/0006-tiff-uncalibrated-resolution-detection.md
    for why we do not rely on the ResolutionUnit/XResolution tags for masks.
    """
    expected_ndim = 2 if axes == "YX" else 3
    if labels.ndim != expected_ndim:
        raise ValueError(
            f"save_label_mask: labels has ndim={labels.ndim} but axes={axes!r} "
            f"requires ndim={expected_ndim}."
        )
    if axes == "ZYX" and spacing.z_um is None:
        raise ValueError(
            "save_label_mask: axes='ZYX' but spacing.z_um is None; 3D masks must "
            "carry a real Z step."
        )

    path.parent.mkdir(parents=True, exist_ok=True)
    metadata: dict[str, Any] = {
        "axes": axes,
        "x_um": spacing.x_um,
        "y_um": spacing.y_um,
        "z_um": spacing.z_um,
    }

    tmp_path = path.with_name(f".{path.name}.tmp-{os.getpid()}")
    try:
        tifffile.imwrite(
            tmp_path,
            labels,
            metadata=metadata,
            compression="zlib" if compress else None,
            photometric="minisblack",
        )
        tmp_path.replace(path)
    except BaseException:
        tmp_path.unlink(missing_ok=True)
        raise


def load_label_mask(path: Path) -> tuple[NDArray[np.integer[Any]], Axes]:
    """Round-trip a mask written by save_label_mask, returning (labels, axes)."""
    with tifffile.TiffFile(path) as tf:
        meta = tf.shaped_metadata
        if not meta or "axes" not in meta[0]:
            raise ValueError(
                f"{path} does not contain the axes metadata written by "
                f"save_label_mask; it may not be a dayana-nuclei mask file."
            )
        axes = meta[0]["axes"]
        if axes not in ("YX", "ZYX"):
            raise ValueError(f"{path} has unrecognized axes metadata {axes!r}.")
        labels = tf.asarray()

    if not np.issubdtype(labels.dtype, np.integer):
        raise ValueError(f"{path} does not contain an integer label array (dtype={labels.dtype}).")

    return labels, axes
