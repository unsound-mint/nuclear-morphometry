"""nuclear-morphometry CLI (spec section 28)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import polars as pl
import typer

from nuclear_morphometry import provenance as prov
from nuclear_morphometry.compare_measurements import compare_measurements_from_files
from nuclear_morphometry.config import load_config
from nuclear_morphometry.export import prepare_analysis as _prepare_analysis
from nuclear_morphometry.io.manifest import (
    build_manifest,
    read_manifest_csv,
    validate_manifest,
    write_manifest_csv,
)
from nuclear_morphometry.pipeline.analyze import run_pipeline
from nuclear_morphometry.pipeline.benchmark import run_benchmark
from nuclear_morphometry.segmentation import validation as seg_validation

app = typer.Typer(no_args_is_help=True, add_completion=False)
manifest_app = typer.Typer(
    no_args_is_help=True, help="Build and validate the experimental manifest."
)
app.add_typer(manifest_app, name="manifest")


def _not_yet_implemented(command: str, phase_hint: str) -> None:
    typer.echo(
        f"`{command}` is not yet implemented in this build.\n{phase_hint}",
        err=True,
    )
    raise typer.Exit(code=2)


@app.command()
def doctor(json_output: Annotated[bool, typer.Option("--json")] = False) -> None:
    """Report environment capability: Python/package versions, CUDA, GUI, BioIO."""
    torch_info = prov.torch_provenance()
    cellpose_info = prov.cellpose_provenance()

    try:
        import bioio_czi  # noqa: F401

        bioio_czi_ok = True
    except ImportError:
        bioio_czi_ok = False

    try:
        import napari  # noqa: F401

        napari_ok = True
    except ImportError:
        napari_ok = False

    capable_cpu_dev = bioio_czi_ok
    capable_cuda_production = torch_info.get("cuda_available", False) and cellpose_info["available"]
    capable_gui_qc = napari_ok

    info = {
        "python_version": prov.platform.python_version(),
        "package_version": prov.package_version("nuclear-morphometry"),
        "torch": torch_info,
        "cellpose": cellpose_info,
        "bioio_czi_importable": bioio_czi_ok,
        "napari_importable": napari_ok,
        "capable": {
            "cpu_development": capable_cpu_dev,
            "cuda_production": capable_cuda_production,
            "gui_qc": capable_gui_qc,
        },
    }

    if json_output:
        typer.echo(json.dumps(info, indent=2))
    else:
        typer.echo(f"Python:          {info['python_version']}")
        typer.echo(f"nuclear-morphometry:   {info['package_version']}")
        typer.echo(
            f"Torch:           {torch_info.get('version', 'not installed')} "
            f"(CUDA available: {torch_info.get('cuda_available', False)})"
        )
        if torch_info.get("cuda_available"):
            typer.echo(f"GPU:             {torch_info.get('gpu_name')}")
        typer.echo(f"Cellpose:        {cellpose_info.get('version', 'not installed')}")
        typer.echo(f"BioIO CZI:       {'ok' if bioio_czi_ok else 'MISSING'}")
        typer.echo(f"napari:          {'ok' if napari_ok else 'MISSING'}")
        typer.echo()
        typer.echo(f"CPU development: {'yes' if capable_cpu_dev else 'no'}")
        typer.echo(f"CUDA production: {'yes' if capable_cuda_production else 'no'}")
        typer.echo(f"GUI QC:          {'yes' if capable_gui_qc else 'no'}")

    if not capable_cpu_dev:
        raise typer.Exit(code=1)


@app.command()
def inspect(
    path: Annotated[Path, typer.Argument(exists=True)],
    scene: Annotated[str | None, typer.Option("--scene")] = None,
    json_output: Annotated[bool, typer.Option("--json")] = False,
) -> None:
    """Print a structured summary of a CZI/TIFF file (spec section 10.5)."""
    from nuclear_morphometry.io.metadata import inspect_image

    result = inspect_image(path, scene=scene)
    if json_output:
        typer.echo(result.model_dump_json(indent=2))
    else:
        typer.echo(result.model_dump_json(indent=2))


@manifest_app.command("build")
def manifest_build(
    input_root: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    output: Annotated[Path, typer.Option("--output")] = Path("manifest.csv"),
) -> None:
    """Scan input_root for the thesis filename convention and write a draft manifest."""
    df = build_manifest(input_root)
    write_manifest_csv(df, output)
    typer.echo(
        f"Wrote {df.height} row(s) to {output}. Run `manifest validate {output}` before use."
    )


@manifest_app.command("validate")
def manifest_validate(manifest: Annotated[Path, typer.Argument(exists=True)]) -> None:
    """Validate a manifest CSV before it is used for analysis."""
    df = read_manifest_csv(manifest)
    result = validate_manifest(df)
    for warning in result.warnings:
        typer.echo(f"WARNING: {warning}", err=True)
    for error in result.errors:
        typer.echo(f"ERROR: {error}", err=True)
    if not result.is_valid:
        raise typer.Exit(code=1)
    typer.echo(f"{manifest} is valid ({df.height} row(s)).")


_ALLOW_UNVALIDATED_MODEL_HELP = (
    'Allow segmentation.model = "auto" to resolve to the unvalidated Cellpose-SAM '
    "default (spec 11.1). For architecture/demo testing only -- never for results "
    "you intend to report."
)


@app.command()
def run(
    config: Annotated[Path, typer.Argument(exists=True)],
    allow_unvalidated_model: Annotated[
        bool, typer.Option("--allow-unvalidated-model", help=_ALLOW_UNVALIDATED_MODEL_HELP)
    ] = False,
) -> None:
    """Run the full pipeline for a config (spec section 30)."""
    run_dir = run_pipeline(config, allow_unvalidated_model=allow_unvalidated_model)
    typer.echo(f"Run complete: {run_dir}")


@app.command()
def resume(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    force: Annotated[bool, typer.Option("--force")] = False,
    allow_unvalidated_model: Annotated[
        bool, typer.Option("--allow-unvalidated-model", help=_ALLOW_UNVALIDATED_MODEL_HELP)
    ] = False,
) -> None:
    """Resume an interrupted run, skipping already-completed fields (spec section 33)."""
    config_path = run_dir / "config.toml"
    if not config_path.exists():
        typer.echo(f"{config_path} not found; cannot resume {run_dir}.", err=True)
        raise typer.Exit(code=1)
    resumed_dir = run_pipeline(
        config_path,
        resume_run_dir=run_dir,
        force=force,
        allow_unvalidated_model=allow_unvalidated_model,
    )
    typer.echo(f"Resume complete: {resumed_dir}")


@app.command("prepare-analysis")
def prepare_analysis(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
) -> None:
    """Build the analysis-ready table with include_default (spec section 26.3)."""
    out_path = _prepare_analysis(run_dir)
    typer.echo(f"Wrote {out_path}")


@app.command("export-csv")
def export_csv(run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)]) -> None:
    """Re-export nuclei.parquet as CSV for interoperability (spec section 26.1)."""
    import polars as pl

    nuclei_path = run_dir / "nuclei.parquet"
    if not nuclei_path.exists():
        typer.echo(f"{nuclei_path} does not exist.", err=True)
        raise typer.Exit(code=1)
    out_path = run_dir / "nuclei.csv"
    pl.read_parquet(nuclei_path).write_csv(out_path)
    typer.echo(f"Wrote {out_path}")


@app.command("finalize-run")
def finalize_run(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    hash_inputs: Annotated[bool, typer.Option("--hash-inputs")] = False,
) -> None:
    """Finalize provenance for archival (spec section 27.1); optionally hash raw inputs."""
    import hashlib

    import polars as pl

    provenance_path = run_dir / "provenance.json"
    if not provenance_path.exists():
        typer.echo(f"{provenance_path} does not exist.", err=True)
        raise typer.Exit(code=1)
    provenance = json.loads(provenance_path.read_text())

    if hash_inputs:
        manifest_path = run_dir / "manifest.parquet"
        manifest_df = pl.read_parquet(manifest_path)
        unique_paths = sorted({p for p in manifest_df["path"].to_list() if p})
        hashes: dict[str, str] = {}
        for path_str in unique_paths:
            source_path = Path(path_str)
            if not source_path.exists():
                typer.echo(f"WARNING: {source_path} not found; skipping hash.", err=True)
                continue
            digest = hashlib.sha256()
            with source_path.open("rb") as f:
                for chunk in iter(lambda: f.read(1 << 20), b""):
                    digest.update(chunk)
            hashes[path_str] = digest.hexdigest()
        provenance["input_file_sha256"] = hashes

    provenance_path.write_text(json.dumps(provenance, indent=2, default=str))
    typer.echo(f"Finalized {provenance_path}")


@app.command()
def benchmark(
    config: Annotated[Path, typer.Argument(exists=True)],
    limit: Annotated[int | None, typer.Option("--limit")] = None,
    output: Annotated[Path | None, typer.Option("--output")] = None,
    allow_unvalidated_model: Annotated[
        bool, typer.Option("--allow-unvalidated-model", help=_ALLOW_UNVALIDATED_MODEL_HELP)
    ] = False,
) -> None:
    """Per-stage performance benchmark: I/O, segmentation, morphology,
    intensity, texture, QC, output timing plus peak RSS/CUDA memory, object
    counts, and image dimensions (spec section 32). Runs the real pipeline
    against a scratch directory -- masks/tables are not kept, only timings.
    """
    report = run_benchmark(
        config,
        limit=limit,
        allow_unvalidated_model=allow_unvalidated_model,
    )

    if output is None:
        loaded_config, _ = load_config(config)
        output_dir = loaded_config.experiment.output_root / "benchmarks"
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = report.generated_at.strftime("%Y-%m-%dT%H%M%SZ")
        output = output_dir / f"{stamp}_{config.stem}.json"
    else:
        output.parent.mkdir(parents=True, exist_ok=True)

    output.write_text(report.model_dump_json(indent=2))

    parquet_path = output.with_suffix(".parquet")
    rows = [
        {
            "image_id": f.image_id,
            "axes": f.axes,
            "shape": str(f.shape),
            "object_count": f.object_count,
            "segmentation_backend": f.segmentation_backend,
            "segmentation_model_id": f.segmentation_model_id,
            **f.stage_timings.model_dump(),
        }
        for f in report.fields
    ]
    pl.DataFrame(rows).write_parquet(parquet_path)

    typer.echo(f"Benchmark report: {output}")
    typer.echo(f"Benchmark table: {parquet_path}")
    for f in report.fields:
        typer.echo(
            f"  {f.image_id}: {f.object_count} objects, shape {f.shape}, "
            f"total {f.stage_timings.total_s:.3f}s "
            f"(segmentation {f.stage_timings.segmentation_s:.3f}s)"
        )
    typer.echo(f"Peak RSS: {report.peak_rss_kb / 1024:.1f} MB")
    if report.peak_cuda_memory_mb is not None:
        typer.echo(f"Peak CUDA memory: {report.peak_cuda_memory_mb:.1f} MB")


@app.command("validate-segmentation")
def validate_segmentation_command(
    prediction: Annotated[Path | None, typer.Option("--prediction", exists=True)] = None,
    reference: Annotated[Path | None, typer.Option("--reference", exists=True)] = None,
    manifest: Annotated[Path | None, typer.Option("--manifest", exists=True)] = None,
    iou_threshold: Annotated[float, typer.Option("--iou-threshold", min=0.0, max=1.0)] = 0.5,
    estimate_split_merge: Annotated[bool, typer.Option("--estimate-split-merge")] = False,
    split_merge_containment_threshold: Annotated[
        float, typer.Option("--split-merge-containment-threshold", min=0.0, max=1.0)
    ] = 0.5,
    output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    """Compare predicted vs. reference label masks (spec section 14).

    Pass either --prediction/--reference for a single pair, or --manifest
    for a batch (a CSV with columns case_id, prediction_path,
    reference_path). iou_threshold is never a hard-coded scientific claim
    (spec 14.3) -- it is always this explicit, documented parameter.

    This command only computes metrics. Accepting a segmentation model for
    real thesis analysis still requires manually reviewing representative
    overlays and recording the decision in a decision record under
    docs/decisions/ (spec 14.3); that judgment is not automated here.
    """
    if manifest is not None:
        if prediction is not None or reference is not None:
            typer.echo("Pass either --manifest or --prediction/--reference, not both.", err=True)
            raise typer.Exit(code=2)
        manifest_df = pl.read_csv(manifest)
        required_columns = {"case_id", "prediction_path", "reference_path"}
        missing_columns = required_columns - set(manifest_df.columns)
        if missing_columns:
            typer.echo(
                f"Validation manifest {manifest} is missing columns: {sorted(missing_columns)}. "
                f"Required columns: {sorted(required_columns)}.",
                err=True,
            )
            raise typer.Exit(code=2)
        cases = [
            (row["case_id"], Path(row["prediction_path"]), Path(row["reference_path"]))
            for row in manifest_df.iter_rows(named=True)
        ]
        reports = seg_validation.validate_segmentation_batch(
            cases,
            iou_threshold=iou_threshold,
            estimate_split_merge=estimate_split_merge,
            split_merge_containment_threshold=split_merge_containment_threshold,
        )
    elif prediction is not None and reference is not None:
        reports = [
            seg_validation.validate_segmentation_from_files(
                prediction,
                reference,
                iou_threshold=iou_threshold,
                estimate_split_merge=estimate_split_merge,
                split_merge_containment_threshold=split_merge_containment_threshold,
            )
        ]
    else:
        typer.echo(
            "Pass --prediction and --reference for one pair, or --manifest for a batch.",
            err=True,
        )
        raise typer.Exit(code=2)

    for report in reports:
        m = report.metrics
        typer.echo(
            f"{report.case_id}: predicted={m.predicted_object_count} "
            f"reference={m.reference_object_count} matched={m.matched_count} "
            f"precision={m.precision} recall={m.recall} f1={m.f1} "
            f"unmatched_prediction={m.unmatched_prediction_count} "
            f"unmatched_reference={m.unmatched_reference_count}"
        )

    if output is not None:
        payload = [report.model_dump(mode="json") for report in reports]
        output.write_text(json.dumps(payload, indent=2))
        typer.echo(f"Wrote full validation report for {len(reports)} case(s) to {output}")


@app.command("compare-measurements")
def compare_measurements_command(
    ours: Annotated[Path, typer.Option("--ours", exists=True)],
    reference: Annotated[Path, typer.Option("--reference", exists=True)],
    mapping: Annotated[Path, typer.Option("--mapping", exists=True)],
    output: Annotated[Path | None, typer.Option("--output")] = None,
) -> None:
    """Compare our measurements to legacy CellProfiler output on the same masks (spec 16.1).

    Both --ours and --reference must already measure the SAME masks (spec
    16.1): this isolates measurement-definition differences from
    segmentation differences. --mapping is a TOML file naming the join keys
    and the reference-column -> our-column measurement pairs to compare
    (see configs/cellprofiler_mapping.toml for the expected shape).
    """
    report = compare_measurements_from_files(ours, reference, mapping)

    typer.echo(
        f"Matched {report.n_matched_objects} objects "
        f"({report.n_only_in_ours} only in ours, {report.n_only_in_reference} only in reference)."
    )
    for col in report.columns:
        if col.n_both_finite == 0:
            typer.echo(f"  {col.reference_column} -> {col.ours_column}: no comparable rows")
            continue
        relative = (
            f"{col.mean_relative_difference:.4f}"
            if col.mean_relative_difference is not None
            else "undefined (all references 0)"
        )
        correlation = f"{col.pearson_r:.4f}" if col.pearson_r is not None else "undefined"
        typer.echo(
            f"  {col.reference_column} -> {col.ours_column}: n={col.n_both_finite}, "
            f"mean_abs_diff={col.mean_absolute_difference:.4g}, "
            f"mean_rel_diff={relative}, pearson_r={correlation}"
        )

    if output is not None:
        output.write_text(report.model_dump_json(indent=2))
        typer.echo(f"Wrote full comparison report to {output}")


@app.command()
def qc(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    image_id: Annotated[
        str | None, typer.Option("--image-id", help="Field to open initially (default: first).")
    ] = None,
) -> None:
    """Launch the interactive napari QC viewer for a run (spec section 23).

    Requires the optional 'gui' dependency group (napari + Qt) -- imported
    here, not at module load, so every other `nuclear-morphometry` command keeps
    working without it (AGENTS.md: "the computational core must not depend
    on napari or Qt").
    """
    try:
        from nuclear_morphometry.qc.viewer import launch_qc_viewer
    except ImportError as exc:
        typer.echo(
            f"The QC viewer requires the optional 'gui' dependency group (napari + Qt), "
            f"which is not installed ({exc}).\n\nInstall it with:\n  uv sync --extra gui",
            err=True,
        )
        raise typer.Exit(code=2) from exc
    launch_qc_viewer(run_dir, image_id=image_id)


@app.command("qc-report")
def qc_report(
    run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)],
    seed: Annotated[int, typer.Option("--seed")] = 0,
    n_overlays: Annotated[int, typer.Option("--n-overlays", min=0)] = 6,
) -> None:
    """Generate a static QC report for a run (spec section 24).

    Overlay field selection is stratified across cell line / condition /
    SortID and seeded by --seed for reproducibility.
    """
    from nuclear_morphometry.qc.report import generate_qc_report

    report_path = generate_qc_report(run_dir, seed=seed, n_overlays=n_overlays)
    typer.echo(f"Wrote QC report to {report_path}")


def main() -> None:
    app()


if __name__ == "__main__":
    sys.exit(main())
