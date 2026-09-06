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
├── schema.py              canonical output column names + NUCLEI_TABLE_SCHEMA +
│                          include_default derivation (single source of truth)
├── provenance.py          per-run provenance.json assembly
├── logging_utils.py       console + per-run file logging setup
├── export.py              atomic per-field Parquet writers, finalize_tables,
│                          prepare_analysis
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
│   └── cellpose_backend.py Cellpose-SAM (4.x) production backend
│
├── measurements/
│   ├── morphology_2d.py    area/perimeter/circularity/solidity/eccentricity/...
│   ├── morphology_3d.py    volume/surface area/z-depth/principal axes/sphericity
│   ├── intensity.py        mean/median/integrated/min/max/std, dimension-agnostic
│   └── texture.py          2D-only masked-GLCM contrast/entropy/homogeneity/
│                           correlation/energy
│
├── qc/
│   └── flags.py            compute_object_qc: border-only automatic exclusion
│                           (no shape-statistic inputs, by construction)
│
└── pipeline/
    ├── analyze.py          run_pipeline: manifest -> per-field segment+measure ->
    │                       atomic commits -> finalized tables
    └── run_state.py        per-field status tracking + resume logic
```

Not yet implemented: `segmentation/validation.py` (Phase 4), `measurements/radial.py`
(spec 20), `qc/annotations.py` / `qc/overlays.py` / `qc/report.py` / `qc/viewer.py`
(Phase 7), multi-channel measurement wiring (spec 25, Phase 8), `pipeline/benchmark.py`
(spec 32).

## Data flow (current: 2D, FixtureSegmenter)

```
manifest.csv (validated)
        |
resolve_image_sources -> {image_id: {channel: ImageSource}}
        |
io.images.load_channel_volume (Hoechst channel)
   - BioIO read, axis normalize to YX/ZYX
   - enforce real physical calibration or fail loudly
   - enforce explicit projection strategy for 2D+Z>1
        |
segmentation.normalize.normalize_percentile (segmentation-only copy)
        |
Segmenter.segment(normalized_image, spacing) -> SegmentationResult
        |
   +----+----------------------------+
   |                                 |
io.masks.save_label_mask      measurements.morphology_2d.measure_2d_morphology
   (atomic, on original             (on the *raw* label image)
    label array)                          |
                                    qc.flags.compute_object_qc
                                    (border-only; no shape inputs)
                                          |
                                export.write_partial_table
                                    (schema.NUCLEI_TABLE_SCHEMA,
                                     one file per image_id -- resume-safe)
        |
export.finalize_tables -> nuclei.parquet, fields.parquet, nuclei.csv
        |
export.prepare_analysis -> analysis_ready.parquet (adds include_default)
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
- `schema.NUCLEI_TABLE_SCHEMA` is the one place per-field nuclei tables get their dtypes,
  including for a field with zero objects — see
  `docs/decisions/0004-parquet-as-canonical-table.md` for the concat-corruption bug this
  prevents.
- The computational core (everything above) has no napari or Qt import; GUI code is
  confined to `qc/viewer.py` (not yet implemented), per spec section 6.

## Segmentation backend selection

`segmentation.backend` in config selects `FixtureSegmenter` (deterministic,
architecture/test use only — no learned model, cannot separate touching nuclei) or the
Cellpose backend. See `docs/decisions/0001-cellpose-segmentation-backend.md` for the
installed Cellpose version's API and why `model = "auto"` requires an explicit opt-in
rather than resolving silently.
