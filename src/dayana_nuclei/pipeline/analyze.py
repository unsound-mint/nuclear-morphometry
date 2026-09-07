"""Pipeline orchestration (spec section 30): manifest -> per-field segmentation
and measurement -> atomic per-field commits -> finalized Parquet tables.

Failure isolation (spec 33): one field's exception marks it failed and moves
on; it never aborts the whole run. A field only reaches ``complete`` after
its partial tables (and mask, if configured) are committed, so resume can
safely retry any pending, failed, or interrupted ("running") field.
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import time
from collections.abc import Iterator
from contextlib import contextmanager
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from dayana_nuclei.config import Config, load_config
from dayana_nuclei.export import (
    finalize_tables,
    partial_fields_path,
    partial_nuclei_path,
    write_partial_table,
)
from dayana_nuclei.io.images import load_channel_volume
from dayana_nuclei.io.manifest import read_manifest_csv, resolve_image_sources, validate_manifest
from dayana_nuclei.io.masks import mask_path_for, save_label_mask
from dayana_nuclei.logging_utils import setup_logging
from dayana_nuclei.measurements.intensity import measure_intensity
from dayana_nuclei.measurements.lamin import measure_lamin_shell_core
from dayana_nuclei.measurements.morphology_2d import Nucleus2DMorphology, measure_2d_morphology
from dayana_nuclei.measurements.morphology_3d import Nucleus3DMorphology, measure_3d_morphology
from dayana_nuclei.measurements.radial import measure_radial_distribution_2d
from dayana_nuclei.measurements.spatial import measure_perinuclear_rings
from dayana_nuclei.measurements.texture import (
    format_um_distance_label,
    measure_texture_2d,
    um_distances_to_pixels,
)
from dayana_nuclei.models import ImageSource
from dayana_nuclei.pipeline.run_state import (
    FieldState,
    incomplete_image_ids,
    init_run_state,
    load_run_state,
    mark_complete,
    mark_failed,
    mark_running,
    save_run_state,
)
from dayana_nuclei.provenance import build_provenance, current_git_commit, finalize_provenance
from dayana_nuclei.qc.flags import compute_object_qc
from dayana_nuclei.qc.image_metrics import compute_image_qc_metrics
from dayana_nuclei.schema import nuclei_table_schema
from dayana_nuclei.segmentation.base import Segmenter
from dayana_nuclei.segmentation.fixture import FixtureSegmenter
from dayana_nuclei.segmentation.normalize import normalize_percentile

logger = logging.getLogger(__name__)


def generate_run_id(*, git_commit: str | None) -> str:
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    suffix = git_commit[:7] if git_commit else secrets.token_hex(3)
    return f"{timestamp}_{suffix}"


def build_segmenter(config: Config, *, allow_unvalidated_model: bool = False) -> Segmenter:
    if config.segmentation.backend == "fixture":
        return FixtureSegmenter()

    from dayana_nuclei.segmentation.cellpose_backend import CellposeSegmenter

    seg = config.segmentation
    return CellposeSegmenter(
        model=seg.model,
        device=seg.device,
        diameter_um=seg.diameter_um,
        use_anisotropy=seg.three_d.use_anisotropy,
        flow3d_smooth=seg.three_d.flow3d_smooth,
        batch_size=seg.batch_size,
        allow_unvalidated_model=allow_unvalidated_model,
    )


def _run_dir_for(config: Config, run_id: str) -> Path:
    return config.experiment.output_root / run_id


def texture_distance_labels_um(config: Config) -> tuple[str, ...]:
    """The run's texture column-name labels for physical-scale mode (spec
    19.4), or ``()`` when the run uses pixel-distance mode. Shared by
    ``nuclei_table_schema`` callers (``run_pipeline`` and
    ``pipeline.benchmark``) so both agree with what ``_process_field``
    actually names its columns -- see docs/decisions/0012."""
    return tuple(format_um_distance_label(d) for d in config.measurements.texture_distances_um)


@contextmanager
def _timed_stage(stage_timings: dict[str, float], key: str) -> Iterator[None]:
    """Accumulate wall time spent in the block into ``stage_timings[key]``.

    When wrapped unconditionally around a stage (spec section 32's per-stage
    benchmark) -- including one a config disables, e.g. mask serialization --
    the key is always present with a real value, 0.0 for a skipped stage,
    never missing. That guarantee does NOT hold for a key only set *inside*
    a loop over per-object work (``qc_s``, ``row_assembly_s``): a field with
    zero objects never enters the loop, so those two keys are seeded to 0.0
    explicitly before the loop rather than relying on this helper alone.
    """
    t = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - t
        stage_timings[key] = stage_timings.get(key, 0.0) + elapsed


def _process_field(
    *,
    config: Config,
    run_dir: Path,
    run_id: str,
    image_id: str,
    channels: dict[str, ImageSource],
    segmenter: Segmenter,
    nuclei_schema: dict[str, Any],
) -> tuple[dict[str, Any], dict[str, float]]:
    """Process one field end-to-end; returns ``(field_row, stage_timings)``
    so `pipeline.benchmark` can report per-stage wall-clock timings (spec
    section 32) and per-field metadata without duplicating this function's
    logic or re-reading its own Parquet output back. Production callers
    (`run_pipeline`) discard the return value; the timer overhead is
    negligible next to the I/O and segmentation work it wraps.
    """
    hoechst_source = channels[config.input.hoechst_channel]
    t0 = time.perf_counter()
    stage_timings: dict[str, float] = {}

    with _timed_stage(stage_timings, "io_s"):
        volume = load_channel_volume(
            hoechst_source,
            mode=config.analysis.mode,
            projection=config.analysis.projection,
            specific_plane=config.analysis.specific_plane,
        )

    with _timed_stage(stage_timings, "segmentation_normalization_s"):
        seg_input = (
            normalize_percentile(
                volume.data,
                percentile_low=config.segmentation.percentile_low,
                percentile_high=config.segmentation.percentile_high,
            )
            if config.segmentation.normalize_for_segmentation
            else volume.data
        )

    t_seg_start = time.perf_counter()
    result = segmenter.segment(seg_input, volume.spacing)
    segmentation_runtime_s = time.perf_counter() - t_seg_start
    stage_timings["segmentation_s"] = segmentation_runtime_s

    mask_path: Path | None = None
    with _timed_stage(stage_timings, "mask_serialization_s"):
        if config.output.save_masks:
            mask_path = mask_path_for(run_dir, image_id)
            save_label_mask(
                result.labels,
                mask_path,
                volume.spacing,
                volume.axes,
                compress=config.output.compress_masks,
            )

    morphologies: list[Nucleus2DMorphology] | list[Nucleus3DMorphology]
    with _timed_stage(stage_timings, "morphology_s"):
        if volume.axes == "YX":
            morphologies = measure_2d_morphology(result.labels, volume.spacing)
        else:
            morphologies = measure_3d_morphology(result.labels, volume.spacing)

    # Always the original source channel, never seg_input (spec 13.3, 18): the
    # segmentation-normalized copy exists only to feed the segmenter.
    intensity_by_object: dict[int, dict[str, Any]] = {}
    with _timed_stage(stage_timings, "intensity_s"):
        if config.measurements.intensity:
            intensity_by_object = {
                r.object_number: r.model_dump(exclude={"object_number"})
                for r in measure_intensity(result.labels, volume.data)
            }

    texture_by_object: dict[int, dict[str, Any]] = {}
    with _timed_stage(stage_timings, "texture_s"):
        if config.measurements.texture_2d and volume.axes == "YX":
            if config.measurements.texture_distances_um:
                # Physical-scale mode (spec 19.4): converting um to pixels is
                # inherently per-field (it depends on that field's own X/Y
                # calibration), but the resulting columns are named by the
                # configured um value, not the resolved pixel distance, so
                # every field agrees on column names regardless of small
                # calibration differences -- see docs/decisions/0012 and
                # schema.py::nuclei_table_schema's texture_distance_labels_um.
                if volume.spacing.x_um != volume.spacing.y_um:
                    raise ValueError(
                        f"{image_id!r}: measurements.texture_distances_um requires "
                        f"square X/Y pixels (converting a physical distance to an "
                        f"isotropic pixel count is ambiguous otherwise), but this "
                        f"field has x_um={volume.spacing.x_um}, "
                        f"y_um={volume.spacing.y_um}. Use "
                        f"measurements.texture_distances_px for anisotropic-pixel data."
                    )
                distances_px = um_distances_to_pixels(
                    list(config.measurements.texture_distances_um), volume.spacing.x_um
                )
                distance_labels: list[str] | None = list(texture_distance_labels_um(config))
            else:
                distances_px = list(config.measurements.texture_distances_px)
                distance_labels = None
            texture_by_object = {
                r.object_number: r.model_dump(exclude={"object_number"})
                for r in measure_texture_2d(
                    result.labels,
                    volume.data,
                    distances_px=distances_px,
                    distance_labels=distance_labels,
                    gray_levels=config.measurements.gray_levels,
                )
            }

    radial_by_object: dict[int, dict[str, Any]] = {}
    with _timed_stage(stage_timings, "radial_distribution_s"):
        if config.measurements.radial_distribution_2d and volume.axes == "YX":
            radial_by_object = {
                r.object_number: r.model_dump(exclude={"object_number"})
                for r in measure_radial_distribution_2d(
                    result.labels,
                    volume.data,
                    radial_bins=config.measurements.radial_bins,
                )
            }

    # Additional channels (spec 25): always reuse result.labels, the
    # Hoechst-derived mask -- never re-segmented. A channel missing for this
    # particular field (spec 25.2: "if present") simply contributes no
    # entries here; nuclei_schema still declares its columns, so those rows
    # get null for them rather than breaking the run's shared schema.
    additional_by_object: dict[int, dict[str, Any]] = {}
    with _timed_stage(stage_timings, "additional_channels_s"):
        for entry in config.measurements.additional_channels:
            channel_source = channels.get(entry.channel)
            if channel_source is None:
                continue
            channel_volume = load_channel_volume(
                channel_source,
                mode=config.analysis.mode,
                projection=config.analysis.projection,
                specific_plane=config.analysis.specific_plane,
            )
            if channel_volume.data.shape != result.labels.shape:
                raise ValueError(
                    f"Additional channel {entry.channel!r} for {image_id!r} has shape "
                    f"{channel_volume.data.shape}, which does not match the Hoechst-"
                    f"derived segmentation shape {result.labels.shape}. The nuclear mask "
                    f"cannot be reused for a differently-shaped channel (spec 25)."
                )
            if entry.kind == "nuclear":
                channel_results: list[Any] = measure_intensity(result.labels, channel_volume.data)
            elif entry.kind == "lamin_shell_core":
                assert entry.shell_width_um is not None  # enforced by config validation
                channel_results = measure_lamin_shell_core(
                    result.labels,
                    channel_volume.data,
                    volume.spacing,
                    shell_width_um=entry.shell_width_um,
                )
            else:  # "mitotracker_rings"
                assert entry.near_ring_um is not None and entry.far_ring_um is not None
                channel_results = measure_perinuclear_rings(
                    result.labels,
                    channel_volume.data,
                    volume.spacing,
                    near_ring_um=entry.near_ring_um,
                    far_ring_um=entry.far_ring_um,
                )
            for r in channel_results:
                values = {
                    f"{entry.prefix}_{k}": v
                    for k, v in r.model_dump(exclude={"object_number"}).items()
                }
                additional_by_object.setdefault(r.object_number, {}).update(values)

    metadata = hoechst_source.metadata
    nuclei_rows: list[dict[str, Any]] = []
    # Seeded here, not just accumulated by _timed_stage, because a
    # zero-object field never enters the loop below at all -- without this,
    # StageTimings(**stage_timings) would raise "field required" for both
    # keys on exactly the fixture-backend "blank field, zero objects" case
    # tests/integration/test_pipeline_e2e.py already exercises.
    stage_timings["qc_s"] = 0.0
    stage_timings["row_assembly_s"] = 0.0
    for morph in morphologies:
        with _timed_stage(stage_timings, "qc_s"):
            qc = compute_object_qc(
                touches_border=morph.touches_border,
                flag_border_objects=config.qc.flag_border_objects,
                exclude_border_from_default=config.qc.exclude_border_from_default_analysis,
            )
        with _timed_stage(stage_timings, "row_assembly_s"):
            nuclei_rows.append(
                {
                    "run_id": run_id,
                    "image_id": image_id,
                    "object_number": morph.object_number,
                    "cell_line": metadata.cell_line,
                    "sort_id": metadata.sort_id,
                    "condition": metadata.condition,
                    "timepoint": metadata.timepoint,
                    "field": metadata.field,
                    "acquisition_batch": metadata.acquisition_batch,
                    "source_path": str(hoechst_source.path),
                    "scene": (
                        str(hoechst_source.scene) if hoechst_source.scene is not None else None
                    ),
                    "mask_path": str(mask_path) if mask_path is not None else None,
                    **morph.model_dump(exclude={"object_number"}),
                    **intensity_by_object.get(morph.object_number, {}),
                    **texture_by_object.get(morph.object_number, {}),
                    **radial_by_object.get(morph.object_number, {}),
                    **additional_by_object.get(morph.object_number, {}),
                    **qc.model_dump(),
                }
            )

    with _timed_stage(stage_timings, "output_s"):
        nuclei_df = pl.DataFrame(nuclei_rows, schema=nuclei_schema)
        write_partial_table(nuclei_df, partial_nuclei_path(run_dir, image_id))

    # Always the original source channel (spec 21), never seg_input.
    with _timed_stage(stage_timings, "qc_s"):
        image_qc_metrics = compute_image_qc_metrics(volume.data, result.labels)

    total_runtime_s = time.perf_counter() - t0
    field_row = {
        "run_id": run_id,
        "image_id": image_id,
        "cell_line": metadata.cell_line,
        "sort_id": metadata.sort_id,
        "condition": metadata.condition,
        "timepoint": metadata.timepoint,
        "field": metadata.field,
        "acquisition_batch": metadata.acquisition_batch,
        "source_path": str(hoechst_source.path),
        "scene": str(hoechst_source.scene) if hoechst_source.scene is not None else None,
        "spacing_x_um": volume.spacing.x_um,
        "spacing_y_um": volume.spacing.y_um,
        "spacing_z_um": volume.spacing.z_um,
        "axes": volume.axes,
        "shape": str(volume.data.shape),
        "object_count": len(morphologies),
        "segmentation_backend": result.backend,
        "segmentation_model_id": result.model_id,
        "segmentation_runtime_s": segmentation_runtime_s,
        "total_runtime_s": total_runtime_s,
        **image_qc_metrics,
    }
    with _timed_stage(stage_timings, "output_s"):
        write_partial_table(pl.DataFrame([field_row]), partial_fields_path(run_dir, image_id))

    # Computed last so it covers the whole function, including the final
    # write above -- deliberately not the same value as field_row's
    # "total_runtime_s" column (a pre-existing, unchanged column defined at
    # the point just before that write); this one is benchmark-only.
    stage_timings["total_s"] = time.perf_counter() - t0
    return field_row, stage_timings


def run_pipeline(
    config_path: Path,
    *,
    resume_run_dir: Path | None = None,
    force: bool = False,
    allow_unvalidated_model: bool = False,
) -> Path:
    config, config_hash = load_config(config_path)

    manifest_df = read_manifest_csv(config.experiment.manifest)
    validation = validate_manifest(manifest_df)
    if not validation.is_valid:
        raise ValueError(
            f"Manifest {config.experiment.manifest} failed validation:\n"
            + "\n".join(f"  - {e}" for e in validation.errors)
        )
    manifest_hash = hashlib.sha256(config.experiment.manifest.read_bytes()).hexdigest()

    sources = resolve_image_sources(
        manifest_df,
        hoechst_channel=config.input.hoechst_channel,
        additional_channels=config.input.additional_channels,
    )
    image_ids = sorted(sources.keys())

    if resume_run_dir is None:
        run_id = generate_run_id(git_commit=current_git_commit())
        run_dir = _run_dir_for(config, run_id)
        if run_dir.exists():
            raise FileExistsError(
                f"Run directory {run_dir} already exists. Refusing to overwrite an "
                f"existing run (spec section 50). Use `dayana-nuclei resume {run_dir}` "
                f"if you intend to continue it."
            )
        for sub in ("masks", "qc", "logs", "partial/nuclei", "partial/fields"):
            (run_dir / sub).mkdir(parents=True, exist_ok=True)

        (run_dir / "config.toml").write_bytes(config_path.read_bytes())
        manifest_df.write_parquet(run_dir / "manifest.parquet")

        provenance = build_provenance(
            config=config,
            config_path=config_path,
            config_hash=config_hash,
            manifest_hash=manifest_hash,
            run_id=run_id,
        )
        (run_dir / "provenance.json").write_text(_dump_json(provenance))

        run_state = init_run_state(run_id=run_id, config_hash=config_hash, image_ids=image_ids)
    else:
        run_dir = resume_run_dir
        run_id = run_dir.name
        run_state = load_run_state(run_dir / "run_state.json")
        if run_state is None:
            raise FileNotFoundError(f"No run_state.json found in {run_dir}; cannot resume.")
        if run_state.config_hash != config_hash and not force:
            raise ValueError(
                f"Config hash mismatch on resume: run_state has "
                f"{run_state.config_hash}, current config is {config_hash}. "
                f"The config used for this run appears to have changed. Pass "
                f"force=True to resume anyway (spec 33)."
            )
        missing = [i for i in image_ids if i not in run_state.fields]
        for image_id in missing:
            run_state.fields[image_id] = FieldState(image_id=image_id, status="pending")

    setup_logging(run_dir)
    save_run_state(run_dir / "run_state.json", run_state)

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

    for image_id in incomplete_image_ids(run_state):
        mark_running(run_state, image_id)
        save_run_state(run_dir / "run_state.json", run_state)
        logger.info("Starting field %s", image_id)
        try:
            _process_field(
                config=config,
                run_dir=run_dir,
                run_id=run_id,
                image_id=image_id,
                channels=sources[image_id],
                segmenter=segmenter,
                nuclei_schema=nuclei_schema,
            )
        except Exception as exc:
            logger.exception("Field %s failed", image_id)
            mark_failed(run_state, image_id, str(exc))
            save_run_state(run_dir / "run_state.json", run_state)
            continue
        mark_complete(run_state, image_id)
        save_run_state(run_dir / "run_state.json", run_state)
        logger.info("Completed field %s", image_id)

    finalize_tables(run_dir, write_csv=config.output.write_csv)

    provenance_path = run_dir / "provenance.json"
    if provenance_path.exists():
        provenance = json.loads(provenance_path.read_text())
        provenance = finalize_provenance(provenance)
        provenance_path.write_text(_dump_json(provenance))

    return run_dir


def _dump_json(data: dict[str, Any]) -> str:
    return json.dumps(data, indent=2, default=str)
