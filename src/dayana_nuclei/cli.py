"""dayana-nuclei CLI (spec section 28)."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Annotated

import typer

from dayana_nuclei import provenance as prov
from dayana_nuclei.export import prepare_analysis as _prepare_analysis
from dayana_nuclei.io.manifest import (
    build_manifest,
    read_manifest_csv,
    validate_manifest,
    write_manifest_csv,
)
from dayana_nuclei.pipeline.analyze import run_pipeline

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
        "package_version": prov.package_version("dayana-nuclei"),
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
        typer.echo(f"dayana-nuclei:   {info['package_version']}")
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
    from dayana_nuclei.io.metadata import inspect_image

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
) -> None:
    """Per-stage performance benchmark (spec section 32)."""
    _not_yet_implemented(
        "benchmark", "Planned for the production-hardening phase (spec section 32)."
    )


@app.command("validate-segmentation")
def validate_segmentation(
    prediction: Annotated[Path, typer.Option("--prediction", exists=True)],
    reference: Annotated[Path, typer.Option("--reference", exists=True)],
) -> None:
    """Compare predicted vs. reference label masks (spec section 14)."""
    _not_yet_implemented(
        "validate-segmentation", "Planned for the segmentation-validation phase (spec section 14)."
    )


@app.command("compare-measurements")
def compare_measurements(
    ours: Annotated[Path, typer.Option("--ours", exists=True)],
    reference: Annotated[Path, typer.Option("--reference", exists=True)],
    mapping: Annotated[Path, typer.Option("--mapping", exists=True)],
) -> None:
    """Compare our measurements to legacy CellProfiler output on the same masks (spec 16.1)."""
    _not_yet_implemented(
        "compare-measurements", "Planned for the 2D-measurements phase (spec section 16.1)."
    )


@app.command()
def qc(run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)]) -> None:
    """Launch the napari QC viewer for a run (spec section 23)."""
    _not_yet_implemented("qc", "Planned for the QC phase (spec section 23).")


@app.command("qc-report")
def qc_report(run_dir: Annotated[Path, typer.Argument(exists=True, file_okay=False)]) -> None:
    """Generate a static QC report for a run (spec section 24)."""
    _not_yet_implemented("qc-report", "Planned for the QC phase (spec section 24).")


def main() -> None:
    app()


if __name__ == "__main__":
    sys.exit(main())
