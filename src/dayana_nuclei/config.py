"""TOML configuration schema (spec section 11).

Config is validated once at load time; downstream code consumes the typed
``Config`` object and must not re-parse or re-validate TOML.
"""

from __future__ import annotations

import hashlib
import tomllib
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

Projection = Literal["none", "max", "mean", "specific_plane"]
AnalysisMode = Literal["2d", "3d"]
Device = Literal["cuda", "cpu"]


class ExperimentConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str = Field(min_length=1)
    manifest: Path
    output_root: Path = Path("results")


class AnalysisConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    mode: AnalysisMode
    projection: Projection = "none"
    specific_plane: int | None = None

    @model_validator(mode="after")
    def _check_projection(self) -> AnalysisConfig:
        if self.mode == "3d" and self.projection != "none":
            raise ValueError(
                "3D analysis requested but a projection strategy "
                f"({self.projection!r}) is configured. 3D analysis must operate "
                'on the intact ZYX volume; set analysis.projection = "none" '
                '(the default) or switch analysis.mode to "2d".'
            )
        if self.projection == "specific_plane" and self.specific_plane is None:
            raise ValueError(
                'analysis.projection = "specific_plane" requires '
                "analysis.specific_plane to be set to a Z index."
            )
        return self


class InputConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    hoechst_channel: str = Field(min_length=1)
    strict_physical_spacing: bool = True
    additional_channels: tuple[str, ...] = ()


class Segmentation3DConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    use_anisotropy: bool = True
    flow3d_smooth: float = 0.0


class SegmentationConfig(BaseModel):
    model_config = ConfigDict(frozen=True, populate_by_name=True)

    backend: Literal["cellpose", "fixture"] = "cellpose"
    device: Device = "cuda"
    model: str = "auto"
    normalize_for_segmentation: bool = True
    percentile_low: float = Field(default=1.0, ge=0.0, lt=100.0)
    percentile_high: float = Field(default=99.8, gt=0.0, le=100.0)
    diameter_um: float = 0.0
    tile: bool = True
    batch_size: int = 0
    three_d: Segmentation3DConfig = Field(default_factory=Segmentation3DConfig, alias="3d")

    @model_validator(mode="after")
    def _check_percentiles(self) -> SegmentationConfig:
        if self.percentile_low >= self.percentile_high:
            raise ValueError(
                f"segmentation.percentile_low ({self.percentile_low}) must be "
                f"less than segmentation.percentile_high ({self.percentile_high})."
            )
        return self


class MeasurementsConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    morphology: bool = True
    intensity: bool = True
    texture_2d: bool = False
    radial_distribution_2d: bool = False
    gray_levels: int = Field(default=256, gt=1)
    texture_distances_px: tuple[int, ...] = (3, 5, 10, 20)
    texture_distances_um: tuple[float, ...] = ()
    radial_bins: int = Field(default=5, gt=0)

    @model_validator(mode="after")
    def _check_texture_scale_mode(self) -> MeasurementsConfig:
        if self.texture_distances_px and self.texture_distances_um:
            raise ValueError(
                "measurements.texture_distances_px and "
                "measurements.texture_distances_um are mutually exclusive; "
                "configure exactly one texture scale mode."
            )
        return self


class QCConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    flag_border_objects: bool = True
    exclude_border_from_default_analysis: bool = True
    save_overlays: bool = True
    overlay_samples_per_group: int = Field(default=5, ge=0)
    random_seed: int = 0


class PerformanceConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    prefetch_fields: int = Field(default=1, ge=0)


class OutputConfig(BaseModel):
    model_config = ConfigDict(frozen=True)

    save_masks: bool = True
    write_csv: bool = True
    compress_masks: bool = False


class Config(BaseModel):
    """Top-level validated configuration for one pipeline run."""

    model_config = ConfigDict(frozen=True)

    experiment: ExperimentConfig
    analysis: AnalysisConfig
    input: InputConfig
    segmentation: SegmentationConfig = Field(default_factory=SegmentationConfig)
    measurements: MeasurementsConfig = Field(default_factory=MeasurementsConfig)
    qc: QCConfig = Field(default_factory=QCConfig)
    performance: PerformanceConfig = Field(default_factory=PerformanceConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)


def load_config(path: Path) -> tuple[Config, str]:
    """Load and validate a TOML config file.

    Returns the validated Config and the SHA-256 hex digest of the raw file
    bytes, for provenance recording.
    """
    raw = path.read_bytes()
    config_hash = hashlib.sha256(raw).hexdigest()
    try:
        data = tomllib.loads(raw.decode("utf-8"))
    except tomllib.TOMLDecodeError as exc:
        raise ValueError(f"Failed to parse config {path}: {exc}") from exc
    config = Config.model_validate(data)
    return config, config_hash
