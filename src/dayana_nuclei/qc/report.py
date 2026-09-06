"""Static QC report (spec section 24).

A self-contained HTML file plus PNG overlays under
``results/<run-id>/qc/`` -- no heavy report framework (spec 24 explicitly
asks to avoid one) and no napari/Qt dependency (spec section 6). Overlay
field selection is stratified across cell line / condition / SortID with a
fixed, configurable random seed, specifically so the report does not
always show the easiest or first fields (spec 24).
"""

from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import polars as pl
from numpy.typing import NDArray
from skimage.io import imsave
from skimage.segmentation import mark_boundaries

from dayana_nuclei.config import load_config
from dayana_nuclei.io.images import Mode, Projection, load_channel_volume
from dayana_nuclei.io.masks import load_label_mask
from dayana_nuclei.models import ExperimentalMetadata, ImageSource
from dayana_nuclei.qc.annotations import load_annotations
from dayana_nuclei.segmentation.normalize import normalize_percentile

_IMAGE_METRIC_COLUMNS = (
    "image_min_intensity",
    "image_max_intensity",
    "image_mean_intensity",
    "image_saturation_fraction",
    "image_focus_metric",
    "image_occupied_fraction",
)
_RUNTIME_COLUMNS = ("segmentation_runtime_s", "total_runtime_s")


def _select_stratified_overlays(fields_df: pl.DataFrame, *, n: int, seed: int) -> list[str]:
    """Deterministically pick up to ``n`` image_ids for one seed.

    Cycles across (cell_line, condition, sort_id) strata -- one field from
    every stratum before a second field from any stratum -- rather than
    taking the first ``n`` rows, so a run dominated by one condition does
    not crowd out the others. The seed controls both the within-stratum
    pick order and the stratum visiting order, so the same seed on the
    same run always reproduces the same selection.
    """
    if fields_df.height == 0 or n <= 0:
        return []
    rng = np.random.default_rng(seed)
    groups = fields_df.group_by(["cell_line", "condition", "sort_id"], maintain_order=True).agg(
        pl.col("image_id")
    )
    group_lists = [rng.permutation(ids.to_list()).tolist() for ids in groups["image_id"]]
    group_order = rng.permutation(len(group_lists)).tolist()

    selected: list[str] = []
    round_idx = 0
    while len(selected) < n and any(round_idx < len(g) for g in group_lists):
        for group_index in group_order:
            if round_idx < len(group_lists[group_index]):
                selected.append(group_lists[group_index][round_idx])
                if len(selected) >= n:
                    break
        round_idx += 1
    return selected


def _render_overlay(
    run_dir: Path,
    field_row: dict[str, Any],
    *,
    hoechst_channel: str,
    mode: Mode,
    projection: Projection,
    specific_plane: int | None,
    out_path: Path,
) -> bool:
    """Render one raw-image + segmentation-boundary overlay PNG.

    Returns False without raising if the mask or source image cannot be
    loaded (e.g. ``output.save_masks = false`` for this run, or a source
    file has moved) -- a missing overlay must not abort the rest of the
    report. 3D fields are max-projected for display only; this never
    touches scientific measurements, which operate on the intact volume.
    """
    mask_path = run_dir / "masks" / f"{field_row['image_id']}_labels.tif"
    if not mask_path.exists():
        return False
    labels, mask_axes = load_label_mask(mask_path)
    if mask_axes == "ZYX":
        labels = labels.max(axis=0)

    metadata = ExperimentalMetadata(
        image_id=field_row["image_id"],
        cell_line=field_row["cell_line"],
        sort_id=field_row["sort_id"],
        condition=field_row["condition"],
        timepoint=field_row["timepoint"],
        field=field_row["field"],
        acquisition_batch=field_row["acquisition_batch"],
    )
    source = ImageSource(
        path=Path(field_row["source_path"]),
        scene=field_row["scene"],
        channel=hoechst_channel,
        metadata=metadata,
    )
    try:
        volume = load_channel_volume(
            source, mode=mode, projection=projection, specific_plane=specific_plane
        )
    except (OSError, ValueError):
        return False

    image = volume.data.max(axis=0) if volume.axes == "ZYX" else volume.data
    if image.shape != labels.shape:
        return False

    display = normalize_percentile(image, percentile_low=1.0, percentile_high=99.5)
    # skimage ships no type stubs; mark_boundaries' return type is inferred as an
    # unresolvable union even though it unambiguously returns a float RGB ndarray
    # here (a documented stub gap, not a correctness escape -- see the same
    # pattern in segmentation/fixture.py).
    overlay = cast("NDArray[np.float64]", mark_boundaries(display, labels, color=(1, 0, 0)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    imsave(out_path, (np.clip(overlay, 0.0, 1.0) * 255).astype(np.uint8))
    return True


def _format_cell(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4g}"
    return str(value)


def _df_to_html_table(df: pl.DataFrame) -> str:
    header = "".join(f"<th>{html.escape(str(column))}</th>" for column in df.columns)
    body_rows = []
    for row in df.iter_rows():
        cells = "".join(f"<td>{html.escape(_format_cell(value))}</td>" for value in row)
        body_rows.append(f"<tr>{cells}</tr>")
    return f"<table><thead><tr>{header}</tr></thead><tbody>{''.join(body_rows)}</tbody></table>"


def _render_html(
    *,
    run_id: str,
    provenance: dict[str, Any],
    stratum_counts: pl.DataFrame,
    total_objects: int,
    border_count: int,
    border_fraction: float | None,
    manual_tag_counts: dict[str, int],
    image_metric_summary: pl.DataFrame | None,
    runtime_summary: pl.DataFrame | None,
    overlay_records: list[dict[str, str]],
    seed: int,
) -> str:
    border_fraction_text = f"{border_fraction:.3f}" if border_fraction is not None else "n/a"
    manual_items = (
        "".join(
            f"<li>{html.escape(tag)}: {count}</li>"
            for tag, count in sorted(manual_tag_counts.items())
        )
        or "<li>No manual annotations recorded.</li>"
    )

    image_metrics_html = (
        _df_to_html_table(image_metric_summary)
        if image_metric_summary is not None
        else "<p>No fields available.</p>"
    )
    runtime_html = (
        _df_to_html_table(runtime_summary)
        if runtime_summary is not None
        else "<p>No fields available.</p>"
    )
    overlay_html = (
        "".join(
            f'<figure><img src="{html.escape(record["png_path"])}" '
            f'alt="{html.escape(record["image_id"])}">'
            f"<figcaption>{html.escape(record['image_id'])} "
            f"({html.escape(record['cell_line'])}, {html.escape(record['condition'])}, "
            f"{html.escape(record['sort_id'])})</figcaption></figure>"
            for record in overlay_records
        )
        or "<p>No overlays could be rendered (masks not saved for this run?).</p>"
    )

    return f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<title>QC report: {html.escape(run_id)}</title>
<style>
body {{ font-family: sans-serif; margin: 2rem; color: #111; }}
table {{ border-collapse: collapse; margin: 0.5rem 0; }}
th, td {{ border: 1px solid #ccc; padding: 4px 8px; text-align: right; }}
th {{ background: #eee; }}
figure {{ display: inline-block; margin: 0.5rem; }}
figure img {{ max-width: 320px; display: block; }}
figcaption {{ font-size: 0.85em; text-align: center; }}
section {{ margin-bottom: 2rem; }}
</style>
</head>
<body>
<h1>QC report: {html.escape(run_id)}</h1>
<section>
<h2>Run identity</h2>
<ul>
<li>run_id: {html.escape(str(run_id))}</li>
<li>git_commit: {html.escape(str(provenance.get("git_commit")))}</li>
<li>start_time_utc: {html.escape(str(provenance.get("start_time_utc")))}</li>
<li>end_time_utc: {html.escape(str(provenance.get("end_time_utc")))}</li>
</ul>
</section>
<section>
<h2>Counts by cell line / SortID / condition</h2>
{_df_to_html_table(stratum_counts)}
</section>
<section>
<h2>Border objects</h2>
<p>{border_count} / {total_objects} objects touch the field-of-view border
({border_fraction_text}). Kept in nuclei.parquet, excluded from
qc_excluded_default only per the run's config.</p>
</section>
<section>
<h2>Manual QC annotations</h2>
<ul>{manual_items}</ul>
</section>
<section>
<h2>Image saturation / focus summary (fields.parquet, spec 21)</h2>
{image_metrics_html}
</section>
<section>
<h2>Segmentation / analysis runtime distribution</h2>
{runtime_html}
</section>
<section>
<h2>Representative segmentation overlays (seed={seed})</h2>
{overlay_html}
</section>
</body>
</html>
"""


def generate_qc_report(run_dir: Path, *, seed: int = 0, n_overlays: int = 6) -> Path:
    """Build a self-contained HTML QC report + PNG overlays under ``run_dir/qc/``.

    Backs ``dayana-nuclei qc-report`` (spec section 24). ``seed`` controls
    which fields are chosen for the representative overlays -- rerunning
    with the same seed on the same run reproduces the same selection.
    """
    nuclei_path = run_dir / "nuclei.parquet"
    fields_path = run_dir / "fields.parquet"
    if not nuclei_path.exists() or not fields_path.exists():
        raise FileNotFoundError(
            f"{run_dir} has no nuclei.parquet/fields.parquet. Run `dayana-nuclei run` "
            f"(or resume/finalize an interrupted run) before generating a QC report."
        )
    nuclei_df = pl.read_parquet(nuclei_path)
    fields_df = pl.read_parquet(fields_path)

    provenance_path = run_dir / "provenance.json"
    provenance = json.loads(provenance_path.read_text()) if provenance_path.exists() else {}
    run_id = str(provenance.get("run_id", run_dir.name))

    config, _ = load_config(run_dir / "config.toml")

    qc_dir = run_dir / "qc"
    overlays_dir = qc_dir / "overlays"

    stratum_counts = (
        nuclei_df.group_by(["cell_line", "sort_id", "condition"], maintain_order=True)
        .agg(pl.len().alias("nucleus_count"))
        .sort(["cell_line", "sort_id", "condition"])
    )

    total_objects = nuclei_df.height
    border_count = int(nuclei_df["qc_border"].sum()) if total_objects else 0
    border_fraction = border_count / total_objects if total_objects else None

    manual_tag_counts: dict[str, int] = {}
    for annotation in load_annotations(run_dir).values():
        manual_tag_counts[annotation.tag] = manual_tag_counts.get(annotation.tag, 0) + 1

    has_image_metrics = fields_df.height > 0 and all(
        column in fields_df.columns for column in _IMAGE_METRIC_COLUMNS
    )
    image_metric_summary = (
        fields_df.select(list(_IMAGE_METRIC_COLUMNS)).describe() if has_image_metrics else None
    )
    runtime_summary = (
        fields_df.select(list(_RUNTIME_COLUMNS)).describe() if fields_df.height else None
    )

    selected_image_ids = _select_stratified_overlays(fields_df, n=n_overlays, seed=seed)
    fields_by_id = {row["image_id"]: row for row in fields_df.iter_rows(named=True)}
    overlay_records: list[dict[str, str]] = []
    for image_id in selected_image_ids:
        field_row = fields_by_id[image_id]
        out_path = overlays_dir / f"{image_id}.png"
        rendered = _render_overlay(
            run_dir,
            field_row,
            hoechst_channel=config.input.hoechst_channel,
            mode=config.analysis.mode,
            projection=config.analysis.projection,
            specific_plane=config.analysis.specific_plane,
            out_path=out_path,
        )
        if rendered:
            overlay_records.append(
                {
                    "image_id": image_id,
                    "cell_line": field_row["cell_line"],
                    "condition": field_row["condition"],
                    "sort_id": field_row["sort_id"],
                    "png_path": f"overlays/{image_id}.png",
                }
            )

    html_text = _render_html(
        run_id=run_id,
        provenance=provenance,
        stratum_counts=stratum_counts,
        total_objects=total_objects,
        border_count=border_count,
        border_fraction=border_fraction,
        manual_tag_counts=manual_tag_counts,
        image_metric_summary=image_metric_summary,
        runtime_summary=runtime_summary,
        overlay_records=overlay_records,
        seed=seed,
    )
    qc_dir.mkdir(parents=True, exist_ok=True)
    report_path = qc_dir / "report.html"
    report_path.write_text(html_text)
    return report_path
