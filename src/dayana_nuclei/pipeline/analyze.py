"""Pipeline orchestration (spec section 30): manifest -> per-field segmentation
and measurement -> atomic per-field commits -> finalized Parquet tables.

Failure isolation (spec 33): one field's exception marks it failed and moves
on; it never aborts the whole run. A field only reaches ``complete`` after
its partial tables (and mask, if configured) are committed, so resume can
safely retry only pending/failed fields.
"""

from __future__ import annotations

import hashlib
import json
import logging
import secrets
import time
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
from dayana_nuclei.measurements.morphology_2d import measure_2d_morphology
from dayana_nuclei.models import ImageSource
from dayana_nuclei.pipeline.run_state import (
    FieldState,
    init_run_state,
    load_run_state,
    mark_complete,
    mark_failed,
    mark_running,
    pending_or_failed_image_ids,
    save_run_state,
)
from dayana_nuclei.provenance import build_provenance, current_git_commit, finalize_provenance
from dayana_nuclei.qc.flags import compute_object_qc
from dayana_nuclei.segmentation.base import Segmenter, SegmenterUnavailableError
from dayana_nuclei.segmentation.fixture import FixtureSegmenter
from dayana_nuclei.segmentation.normalize import normalize_percentile

logger = logging.getLogger(__name__)


def generate_run_id(*, git_commit: str | None) -> str:
    timestamp = datetime.now(UTC).strftime("%Y-%m-%dT%H%M%SZ")
    suffix = git_commit[:7] if git_commit else secrets.token_hex(3)
    return f"{timestamp}_{suffix}"


def build_segmenter(config: Config) -> Segmenter:
    if config.segmentation.backend == "fixture":
        return FixtureSegmenter()
    raise SegmenterUnavailableError(
        'segmentation.backend = "cellpose" is not yet implemented in this build '
        "(see docs/decisions/0001-cellpose-segmentation-backend.md). Use "
        'segmentation.backend = "fixture" to exercise the pipeline architecture '
        "without a learned segmentation model."
    )


def _run_dir_for(config: Config, run_id: str) -> Path:
    return config.experiment.output_root / run_id


def _process_field(
    *,
    config: Config,
    run_dir: Path,
    run_id: str,
    image_id: str,
    channels: dict[str, ImageSource],
    segmenter: Segmenter,
) -> None:
    hoechst_source = channels[config.input.hoechst_channel]
    t0 = time.perf_counter()

    volume = load_channel_volume(
        hoechst_source,
        mode=config.analysis.mode,
        projection=config.analysis.projection,
        specific_plane=config.analysis.specific_plane,
    )

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

    mask_path: Path | None = None
    if config.output.save_masks:
        mask_path = mask_path_for(run_dir, image_id)
        save_label_mask(
            result.labels,
            mask_path,
            volume.spacing,
            volume.axes,
            compress=config.output.compress_masks,
        )

    if volume.axes != "YX":
        raise NotImplementedError(
            f"3D measurement is not yet implemented in this build (image_id={image_id}); "
            f'set analysis.mode = "2d" or wait for the 3D measurement phase.'
        )
    morphologies = measure_2d_morphology(result.labels, volume.spacing)

    metadata = hoechst_source.metadata
    nuclei_rows: list[dict[str, Any]] = []
    for morph in morphologies:
        qc = compute_object_qc(
            touches_border=morph.touches_border,
            flag_border_objects=config.qc.flag_border_objects,
            exclude_border_from_default=config.qc.exclude_border_from_default_analysis,
        )
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
                "scene": str(hoechst_source.scene) if hoechst_source.scene is not None else None,
                "mask_path": str(mask_path) if mask_path is not None else None,
                **morph.model_dump(exclude={"object_number"}),
                **qc.model_dump(),
            }
        )
    nuclei_df = pl.DataFrame(nuclei_rows) if nuclei_rows else pl.DataFrame()
    write_partial_table(nuclei_df, partial_nuclei_path(run_dir, image_id))

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
    }
    write_partial_table(pl.DataFrame([field_row]), partial_fields_path(run_dir, image_id))


def run_pipeline(
    config_path: Path, *, resume_run_dir: Path | None = None, force: bool = False
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

    segmenter = build_segmenter(config)

    for image_id in pending_or_failed_image_ids(run_state):
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
