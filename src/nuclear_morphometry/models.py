"""Core domain model.

These types encode the scientific invariants of the pipeline (see
``Nuclear_Morphometry_Complete_Build_Spec.md`` sections 1.3 and 8): physical
calibration is never assumed, and every nucleus stays traceable to its
biological replicate (SortID) and source image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray
from pydantic import BaseModel, ConfigDict, Field, model_validator

Projection = Literal["none", "max", "mean", "specific_plane"]
AnalysisMode = Literal["2d", "3d"]


class PhysicalSpacing(BaseModel):
    """Voxel/pixel physical calibration in micrometers.

    ``z_um`` is ``None`` for a single-plane 2D image. It must never be
    fabricated as 1.0 to stand in for an unknown Z step.
    """

    model_config = ConfigDict(frozen=True)

    x_um: float = Field(gt=0)
    y_um: float = Field(gt=0)
    z_um: float | None = Field(default=None, gt=0)

    def require_z(self, *, context: str) -> float:
        """Return z_um or raise an actionable error if it is unavailable."""
        if self.z_um is None:
            raise ValueError(
                f"3D analysis requires physical Z spacing, but no Z step was found "
                f"for {context}.\n\n"
                f"Inspect the source with:\n"
                f"  nuclear-morphometry inspect <path>\n\n"
                f"Then either fix the source/manifest metadata or supply an "
                f"explicitly calibrated value from the acquisition record. "
                f"The pipeline will not assume Z=1."
            )
        return self.z_um

    @property
    def voxel_volume_um3(self) -> float:
        """Volume of one voxel; requires z_um to be set."""
        z = self.require_z(context="voxel volume computation")
        return self.x_um * self.y_um * z


class ExperimentalMetadata(BaseModel):
    """Biological/experimental identity of one image.

    ``sort_id`` is the biological replicate (spec section 1.3). It must be
    preserved on every downstream row so nested nuclei are never mistaken
    for independent replicates.
    """

    model_config = ConfigDict(frozen=True)

    image_id: str = Field(min_length=1)
    cell_line: str = Field(min_length=1)
    sort_id: str = Field(min_length=1)
    condition: str = Field(min_length=1)
    timepoint: str = Field(min_length=1)
    field: str = Field(min_length=1)
    acquisition_batch: str | None = None


class ImageSource(BaseModel):
    """A reference to one channel of one image on disk, prior to loading."""

    model_config = ConfigDict(frozen=True, arbitrary_types_allowed=True)

    path: Path
    scene: str | int | None
    channel: str
    metadata: ExperimentalMetadata
    # Optional values transcribed from the authoritative acquisition record
    # when an exported TIFF has lost calibration metadata. They are never
    # inferred from image dimensions or filenames.
    spacing_override: PhysicalSpacing | None = None


class ImageVolume(BaseModel):
    """A loaded, axis-normalized image ready for segmentation/measurement.

    ``axes`` is always exactly ``"YX"`` (2D) or ``"ZYX"`` (3D); channel and
    time dimensions must already have been selected before construction.
    """

    model_config = ConfigDict(arbitrary_types_allowed=True)

    data: NDArray[Any]
    spacing: PhysicalSpacing
    axes: Literal["YX", "ZYX"]
    dtype: str

    @model_validator(mode="after")
    def _check_rank(self) -> ImageVolume:
        expected_ndim = 2 if self.axes == "YX" else 3
        if self.data.ndim != expected_ndim:
            raise ValueError(
                f"ImageVolume with axes={self.axes!r} requires a "
                f"{expected_ndim}-D array, got ndim={self.data.ndim}"
            )
        if self.axes == "ZYX" and self.spacing.z_um is None:
            raise ValueError(
                "ImageVolume constructed with axes='ZYX' but spacing.z_um is None; "
                "3D volumes must carry a real Z step."
            )
        return self


class SegmentationResult(BaseModel):
    """Output of a Segmenter: an integer label image plus provenance."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    labels: NDArray[np.integer[Any]]
    backend: str
    model_id: str
    backend_metadata: dict[str, Any] = Field(default_factory=dict)


class RunIdentity(BaseModel):
    """Identity of one pipeline run, used to key the results directory."""

    model_config = ConfigDict(frozen=True)

    run_id: str
    config_hash: str
    git_commit: str | None
    git_dirty: bool = False
