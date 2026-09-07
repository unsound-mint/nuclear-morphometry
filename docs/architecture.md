# Architecture

This describes the repository as it actually exists. See `Dayana_Nuclei_Complete_Build_Spec.md`
section 52 for the full phase plan and `AGENTS.md` for the current implementation status
summary.

## Module boundaries

```
src/dayana_nuclei/
├── models.py            domain types: PhysicalSpacing, ExperimentalMetadata,
│                         ImageSource, ImageVolume, SegmentationResult, RunIdentity
├── config.py             Pydantic TOML config schema (Config, *Config sub-models)
├── schema.py              canonical output column names + nuclei_table_schema() +
│                          include_default derivation (single source of truth)
├── provenance.py          per-run provenance.json assembly
├── logging_utils.py       console + per-run file logging setup
├── export.py              atomic per-field Parquet writers, finalize_tables,
│                          prepare_analysis
├── compare_measurements.py legacy CellProfiler measurement-parity comparison
│                          on the same masks (spec 16.1/28; backs
│                          `compare-measurements` CLI)
├── cli.py                 Typer app; all `dayana-nuclei` subcommands
│
├── io/
│   ├── metadata.py        ImageInspection + inspect_image (backs `inspect`)
│   ├── images.py          load_channel_volume: BioIO load -> axis-normalized,
│   │                      calibrated ImageVolume; enforces 2D/3D semantics
│   ├── masks.py            label mask persistence (tifffile, atomic writes)
│   └── manifest.py         manifest build/validate/read/write,
│                           resolve_image_sources (manifest row -> ImageSource)
│
├── segmentation/
│   ├── base.py             Segmenter protocol, SegmenterUnavailableError,
│   │                      validate_label_image
│   ├── normalize.py        segmentation-only percentile normalization
│   ├── fixture.py          FixtureSegmenter (deterministic, Otsu + connected
│   │                       components; architecture/test backend only)
│   ├── cellpose_backend.py Cellpose-SAM (4.x) production backend
│   └── validation.py       validate_segmentation: IoU matrix + Hungarian matching,
│                           precision/recall/F1, disabled-by-default split/merge
│                           heuristic (spec 14; backs `validate-segmentation` CLI)
│
├── measurements/
│   ├── morphology_2d.py    area/perimeter/circularity/solidity/eccentricity/...
│   ├── morphology_3d.py    volume/surface area/z-depth/principal axes/sphericity
│   ├── intensity.py        mean/median/integrated/min/max/std, dimension-agnostic
│   ├── texture.py          2D-only masked-GLCM contrast/entropy/homogeneity/
│   │                       correlation/energy
│   ├── lamin.py            shell/core intensity via a physically-calibrated
│   │                       distance-from-boundary erosion (spec 25.3)
│   ├── spatial.py          MitoTracker near/far perinuclear ring intensity via
│   │                       nearest-nucleus disambiguation (spec 25.4; see
│   │                       docs/decisions/0009)
│   └── radial.py           2D-only radial intensity distribution: concentric bins
│                           by normalized distance from the object's boundary
│                           (spec 20; a from-scratch definition, not CellProfiler
│                           parity)
│
├── qc/
│   ├── flags.py            compute_object_qc: border-only automatic exclusion
│   │                       (no shape-statistic inputs, by construction)
│   ├── image_metrics.py    compute_image_qc_metrics: per-field min/max/mean
│   │                       intensity, saturation fraction, focus (variance of
│   │                       Laplacian), occupied fraction (spec 21, measurement-only)
│   ├── annotations.py      manual QC tag store keyed by (image_id, object_number),
│   │                       napari-independent (spec 22.2/23); folds into a
│   │                       *derived* nuclei table, never mutates nuclei.parquet
│   ├── report.py           generate_qc_report: self-contained HTML + PNG overlays
│   │                       under results/<run-id>/qc/ (spec 24); seeded, stratified
│   │                       overlay sampling across cell line/condition/SortID
│   └── viewer.py           build_qc_viewer/launch_qc_viewer: interactive napari
│                           viewer (spec 23/53.9) -- click-to-select object + key
│                           measurements, keyboard/dock-widget manual tags, a
│                           Points-layer annotation overlay that reloads on open;
│                           the only module allowed to import napari/Qt, imported
│                           lazily by cli.py's `qc` command
│
└── pipeline/
    ├── analyze.py          run_pipeline: manifest -> per-field segment+measure ->
    │                       atomic commits -> finalized tables; _process_field
    │                       returns (field_row, stage_timings) so benchmark.py
    │                       can reuse it without duplicating its logic
    ├── run_state.py        per-field status tracking + resume logic
    └── benchmark.py        run_benchmark: per-stage wall-clock timing, peak
                             RSS/CUDA memory, image dims, object counts (spec
                             32; backs `benchmark` CLI) -- runs the real
                             pipeline against a scratch directory, discards
                             masks/tables, keeps only the timing report
```

## Data flow (2D or 3D, FixtureSegmenter or Cellpose)

```
manifest.csv (validated)
        |
resolve_image_sources -> {image_id: {channel: ImageSource}}
        |
io.images.load_channel_volume (Hoechst channel)
   - BioIO read, axis normalize to YX (2D) / ZYX (3D)
   - enforce real physical calibration or fail loudly
   - enforce explicit projection strategy for 2D+Z>1
        |
segmentation.normalize.normalize_percentile (segmentation-only copy)
        |
Segmenter.segment(normalized_image, spacing) -> SegmentationResult
   (FixtureSegmenter: Otsu + connected components, deterministic, test-only;
    CellposeSegmenter: Cellpose-SAM on GPU, model loaded once per run --
    2D verified working on real hardware; 3D verified correct on one
    realistic-signal synthetic case after fixing a normalization-bypass bug,
    real-data validation via `validate-segmentation` still required before
    production use, see docs/decisions/0008)
        |
   +----+---------------------------------------------------+
   |                                                         |
io.masks.save_label_mask                    measurements.morphology_2d / morphology_3d
   (atomic, on original                       (branches on volume.axes; on the *raw*
    label array)                               label image)
                                                          |
                                              measurements.intensity.measure_intensity
                                              (if config.measurements.intensity; on the
                                               *original*, non-normalized channel image)
                                                          |
                                              measurements.texture.measure_texture_2d
                                              (if config.measurements.texture_2d and
                                               axes == YX; masked GLCM, never 3D)
                                                          |
                                              measurements.radial.measure_radial_distribution_2d
                                              (if config.measurements.radial_distribution_2d
                                               and axes == YX; concentric bins by normalized
                                               boundary distance, never 3D, spec 20)
                                                          |
                                              per config.measurements.additional_channels:
                                              load_channel_volume(that channel) then
                                              measure_intensity / lamin.measure_lamin_
                                              shell_core / spatial.measure_perinuclear_rings
                                              on the SAME result.labels (spec 25 -- never
                                              re-segmented); {prefix}_-prefixed columns
                                                          |
                                              qc.flags.compute_object_qc
                                              (border-only; no shape inputs)
                                                          |
                                              export.write_partial_table (-> nuclei row)
                                              (schema.nuclei_table_schema(), resolved once
                                               per run from config.measurements and reused
                                               for every field; one file per image_id --
                                               resume-safe)
        |
qc.image_metrics.compute_image_qc_metrics (on the *original* channel image + labels;
   measurement-only, spec 21) -> merged into the same field's export.write_partial_table
   (-> fields row)
        |
export.finalize_tables -> nuclei.parquet, fields.parquet, nuclei.csv
        |
export.prepare_analysis -> analysis_ready.parquet (adds include_default)
        |
(optional, post-hoc) qc.annotations.apply_annotations_to_nuclei -> a *derived* copy of
   nuclei.parquet with manual tags folded in; never mutates nuclei.parquet itself (spec 48)
```

Per-field commits are atomic and idempotent (temp-file-plus-rename, one file per
`image_id`), so a field killed mid-processing or one producing zero objects cannot corrupt
or duplicate rows in the finalized tables — see
`docs/decisions/0004-parquet-as-canonical-table.md` for the concat-safety details this
depends on, and `pipeline/run_state.py`'s `incomplete_image_ids` for why an interrupted
("running") field is retried, not skipped, on resume.

## Invariants enforced by construction, not convention

- `PhysicalSpacing.z_um` is `None` for a genuinely uncalibrated/2D source and is never
  coerced to `1.0` — every 3D-requiring code path calls `require_z()`, which raises an
  actionable error rather than defaulting.
- `qc.flags.compute_object_qc` takes only `touches_border`, `flag_border_objects`, and
  `exclude_border_from_default` — it has no parameter through which a shape statistic
  (eccentricity, solidity, circularity, ...) could ever drive exclusion. See
  `docs/decisions/0002-no-phenotype-based-qc-filtering.md`.
- `schema.nuclei_table_schema()` is the one place per-field nuclei tables get their dtypes
  (base columns plus intensity/texture columns resolved once from the run's config, since
  texture's column names depend on configured pixel distances), including for a field with
  zero objects — see
  `docs/decisions/0004-parquet-as-canonical-table.md` for the concat-corruption bug this
  prevents.
- The computational core (everything above) has no napari or Qt import; GUI code is confined
  to `qc/viewer.py`, per spec section 6 -- `cli.py` imports it lazily inside the `qc` command
  body so every other command works without the optional `gui` dependency group installed.

## Segmentation backend selection

`segmentation.backend` in config selects `FixtureSegmenter` (deterministic,
architecture/test use only — no learned model, cannot separate touching nuclei) or the
Cellpose backend. See `docs/decisions/0001-cellpose-segmentation-backend.md` for the
installed Cellpose version's API and why `model = "auto"` requires an explicit opt-in
rather than resolving silently.

2D Cellpose-SAM segmentation is verified working end-to-end on real GPU hardware. 3D mode
initially showed severe over-segmentation on synthetic test volumes; this was traced to a
real normalization-bypass bug (now rejected at config validation, see
`Config._check_cellpose_requires_normalization`) rather than a genuine 3D limitation for
realistic signal — see `docs/decisions/0008-cellpose-3d-oversegmentation.md`. Do not trust a
3D run's object count until it has been checked with `validate-segmentation` against real
reference masks.
