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

from dayana_nuclei.schema import ChannelKind

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


class AdditionalChannelConfig(BaseModel):
    """One additional (non-Hoechst) channel's measurement configuration
    (spec section 25). The Hoechst-derived nuclear mask is always reused --
    an additional channel is never re-segmented."""

    model_config = ConfigDict(frozen=True)

    channel: str = Field(min_length=1)
    prefix: str = Field(min_length=1)
    kind: ChannelKind = "nuclear"
    # Required for kind="lamin_shell_core" (spec 25.3).
    shell_width_um: float | None = None
    # Required for kind="mitotracker_rings" (spec 25.4): (start, end) bands
    # in um from the nuclear boundary, ordered near then far.
    near_ring_um: tuple[float, float] | None = None
    far_ring_um: tuple[float, float] | None = None

    @model_validator(mode="after")
    def _check_kind_parameters(self) -> AdditionalChannelConfig:
        if self.kind == "lamin_shell_core":
            if self.shell_width_um is None or self.shell_width_um <= 0:
                raise ValueError(
                    f"measurements.additional_channels prefix={self.prefix!r} has "
                    f'kind="lamin_shell_core" and requires shell_width_um > 0 '
                    f"(got {self.shell_width_um!r})."
                )
        elif self.kind == "mitotracker_rings":
            if self.near_ring_um is None or self.far_ring_um is None:
                raise ValueError(
                    f"measurements.additional_channels prefix={self.prefix!r} has "
                    f'kind="mitotracker_rings" and requires both near_ring_um and '
                    f"far_ring_um to be set."
                )
            near_start, near_end = self.near_ring_um
            far_start, far_end = self.far_ring_um
            if not (0 <= near_start < near_end <= far_start < far_end):
                raise ValueError(
                    f"measurements.additional_channels prefix={self.prefix!r}: "
                    f"near_ring_um and far_ring_um must be ordered, non-overlapping "
                    f"bands increasing outward from the nuclear boundary -- got "
                    f"near={self.near_ring_um}, far={self.far_ring_um}."
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
    additional_channels: tuple[AdditionalChannelConfig, ...] = ()

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

    @model_validator(mode="after")
    def _check_texture_requires_2d(self) -> Config:
        if self.measurements.texture_2d and self.analysis.mode != "2d":
            raise ValueError(
                'measurements.texture_2d = true requires analysis.mode = "2d" '
                "(spec section 19.1: 3D texture is not implemented and must remain "
                "disabled). Set measurements.texture_2d = false for a 3D run."
            )
        return self

    @model_validator(mode="after")
    def _check_cellpose_requires_normalization(self) -> Config:
        if (
            self.segmentation.backend == "cellpose"
            and not self.segmentation.normalize_for_segmentation
        ):
            raise ValueError(
                'segmentation.backend = "cellpose" requires '
                "segmentation.normalize_for_segmentation = true. CellposeSegmenter "
                "always calls Cellpose's eval() with normalize=False, trusting "
                "segmentation.normalize.normalize_percentile (invoked by the pipeline) "
                "to have already rescaled the image -- it has no code path that "
                "normalizes raw sensor-range data itself. Disabling pipeline "
                "normalization for this backend would silently feed raw, "
                "un-normalized intensities to the model (empirically confirmed this "
                "produces severe segmentation errors in 3D; see "
                "docs/decisions/0008-cellpose-3d-oversegmentation.md). Set "
                "segmentation.normalize_for_segmentation = true, or switch "
                'segmentation.backend to "fixture" if disabling normalization is '
                "genuinely intended (Otsu thresholding is scale-invariant)."
            )
        return self

    @model_validator(mode="after")
    def _check_additional_channels(self) -> Config:
        configured_channels = set(self.input.additional_channels)
        prefixes_seen: set[str] = set()
        for entry in self.measurements.additional_channels:
            if entry.channel not in configured_channels:
                raise ValueError(
                    f"measurements.additional_channels references channel "
                    f"{entry.channel!r}, which is not listed in "
                    f"input.additional_channels {sorted(configured_channels)!r}. "
                    f"Add it there so the pipeline knows to load it for each field."
                )
            if entry.prefix == "hoechst":
                raise ValueError(
                    'measurements.additional_channels prefix cannot be "hoechst" -- '
                    "Hoechst's own intensity columns are deliberately unprefixed "
                    "(mean_intensity, etc.); reusing that name as an additional-"
                    "channel prefix would create an ambiguous column name."
                )
            if entry.prefix in prefixes_seen:
                raise ValueError(
                    f"measurements.additional_channels has a duplicate prefix "
                    f"{entry.prefix!r}; every additional channel needs a unique "
                    f"column prefix."
                )
            prefixes_seen.add(entry.prefix)
        return self


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
