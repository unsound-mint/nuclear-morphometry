"""Per-stage performance benchmarking (spec section 32).

Reuses ``pipeline.analyze._process_field`` directly rather than duplicating
its segmentation/measurement logic -- the only new code here is field
selection, a scratch run directory, and the peak-memory/report bookkeeping
spec 32 asks for on top of the per-stage timings ``_process_field`` already
returns.

Runs against a temporary scratch directory: a benchmark is a timing
measurement, not an analysis run to keep, so its masks and partial tables
are discarded once each field's timings and object count are captured.
"""

from __future__ import annotations

import ast
import resource
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from nuclear_morphometry.config import load_config
from nuclear_morphometry.io.manifest import (
    read_manifest_csv,
    resolve_image_sources,
    validate_manifest,
)
from nuclear_morphometry.pipeline.analyze import (
    _process_field,
    build_segmenter,
    texture_distance_labels_um,
)
from nuclear_morphometry.provenance import package_version
from nuclear_morphometry.schema import nuclei_table_schema


class StageTimings(BaseModel):
    """Wall-clock seconds spent in each pipeline stage for one field."""

    model_config = ConfigDict(frozen=True)

    io_s: float
    segmentation_normalization_s: float
    segmentation_s: float
    mask_serialization_s: float
    morphology_s: float
    intensity_s: float
    texture_s: float
    radial_distribution_s: float
    additional_channels_s: float
    qc_s: float
    row_assembly_s: float
    output_s: float
    total_s: float


class FieldBenchmark(BaseModel):
    model_config = ConfigDict(frozen=True)

    image_id: str
    axes: str
    shape: tuple[int, ...]
    object_count: int
    segmentation_backend: str
    segmentation_model_id: str
    stage_timings: StageTimings


class BenchmarkReport(BaseModel):
    """One benchmark run's results; JSON-serializable for before/after diffing."""

    model_config = ConfigDict(frozen=True)

    config_path: str
    package_version: str | None
    generated_at: datetime
    segmentation_backend: str
    segmentation_device: str
    peak_rss_kb: int
    peak_cuda_memory_mb: float | None
    fields: tuple[FieldBenchmark, ...]


def _peak_cuda_memory_mb() -> float | None:
    try:
        import torch
    except ImportError:
        return None
    if not torch.cuda.is_available():
        return None
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def run_benchmark(
    config_path: Path,
    *,
    limit: int | None = None,
    allow_unvalidated_model: bool = False,
) -> BenchmarkReport:
    """Run the first ``limit`` fields (all fields if ``None``) of ``config_path``
    through the real pipeline, reporting per-stage timing, image dimensions,
    object counts, and peak process RSS / CUDA memory (spec section 32).
    """
    config, _config_hash = load_config(config_path)

    manifest_df = read_manifest_csv(config.experiment.manifest)
    validation = validate_manifest(manifest_df)
    if not validation.is_valid:
        raise ValueError(
            f"Manifest {config.experiment.manifest} failed validation:\n"
            + "\n".join(f"  - {e}" for e in validation.errors)
        )

    sources = resolve_image_sources(
        manifest_df,
        hoechst_channel=config.input.hoechst_channel,
        additional_channels=config.input.additional_channels,
    )
    image_ids = sorted(sources.keys())
    if limit is not None:
        image_ids = image_ids[:limit]
    if not image_ids:
        raise ValueError(f"No fields to benchmark for {config_path} (manifest resolved to 0).")

    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()
    except ImportError:
        pass

    # Load the model once (spec 32.1), same as a real run -- never per field.
    segmenter = build_segmenter(config, allow_unvalidated_model=allow_unvalidated_model)
    nuclei_schema = nuclei_table_schema(
        include_intensity=config.measurements.intensity,
        include_texture=config.measurements.texture_2d,
        texture_distances_px=config.measurements.texture_distances_px,
        texture_distance_labels_um=texture_distance_labels_um(config),
        additional_channels=tuple(
            (entry.prefix, entry.kind) for entry in config.measurements.additional_channels
        ),
        radial_bins=(
            config.measurements.radial_bins if config.measurements.radial_distribution_2d else 0
        ),
    )

    fields: list[FieldBenchmark] = []
    with tempfile.TemporaryDirectory(prefix="nuclear-morphometry-benchmark-") as scratch:
        run_dir = Path(scratch)
        for sub in ("masks", "partial/nuclei", "partial/fields"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)

        for image_id in image_ids:
            field_row, stage_timings = _process_field(
                config=config,
                run_dir=run_dir,
                run_id="benchmark",
                image_id=image_id,
                channels=sources[image_id],
                segmenter=segmenter,
                nuclei_schema=nuclei_schema,
            )
            fields.append(
                FieldBenchmark(
                    image_id=image_id,
                    axes=field_row["axes"],
                    shape=ast.literal_eval(field_row["shape"]),
                    object_count=field_row["object_count"],
                    segmentation_backend=field_row["segmentation_backend"],
                    segmentation_model_id=field_row["segmentation_model_id"] or "",
                    stage_timings=StageTimings(**stage_timings),
                )
            )

    return BenchmarkReport(
        config_path=str(config_path),
        package_version=package_version("nuclear-morphometry"),
        generated_at=datetime.now(UTC),
        segmentation_backend=config.segmentation.backend,
        segmentation_device=config.segmentation.device,
        peak_rss_kb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
        peak_cuda_memory_mb=_peak_cuda_memory_mb(),
        fields=tuple(fields),
    )
