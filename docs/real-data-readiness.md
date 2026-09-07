# Real-data readiness checklist

The software implementation and the scientific acceptance of a thesis run are separate
milestones. Do not cite measurements from a run until every applicable item below is closed.

## 1. Recover authoritative acquisition metadata

For every acquisition batch, obtain from the Zeiss source metadata, microscope export, or
lab acquisition record:

- X pixel size in um/pixel;
- Y pixel size in um/pixel;
- Z step in um/slice for 3D stacks;
- objective and numerical aperture;
- detector bit depth and channel identities;
- exposure, laser power, gain, and any other intensity-setting values needed to establish
  comparability across low/high/bulk.

Prefer original CZI files with intact metadata. If only metadata-stripped TIFFs are
available, transcribe the verified calibration into manifest columns `spacing_x_um`,
`spacing_y_um`, and `spacing_z_um`. Never estimate spacing from array dimensions or nuclear
appearance.

## 2. Build the experimental manifest

Each image must have real values for `cell_line`, `sort_id`, `condition`, `timepoint`,
`field`, and `channel`. `sort_id` identifies the independent biological replicate.

The final experiment needs SW480 and SW620; low, high, and bulk; and at least three
independent sorts. A manifest can include preliminary or incomplete data, but it must not be
described as the final thesis dataset.

```bash
uv run dayana-nuclei manifest validate manifest.csv
uv run dayana-nuclei inspect /path/to/representative-source.czi
```

## 3. Select and validate segmentation on real nuclei

Create manually corrected reference masks stratified across:

- both cell lines;
- low, high, and bulk;
- independent sorts;
- sparse, dense/touching, irregular, elongated, and dim nuclei.

Run candidate Cellpose models/configurations on a CUDA machine, compare them against the
fixed references, and manually review overlays through multiple Z planes. Record the
accepted model, parameters, reference-set identity, metrics, and visual decision in a new
decision record. Do not use `--allow-unvalidated-model` for reportable results.

```bash
uv run dayana-nuclei validate-segmentation \
  --manifest validation_cases.csv \
  --estimate-split-merge \
  --output validation_report.json
```

## 4. Establish legacy 2D continuity

Obtain the original `hoechst_cellprofiler.cppproj`, its per-object export, and label masks.
Run both measurement implementations on the same masks before comparing them. Resolve and
document any mathematical-definition difference; do not change expected outputs merely to
force agreement.

```bash
uv run dayana-nuclei compare-measurements \
  --ours results/<run-id>/nuclei.parquet \
  --reference legacy_cellprofiler.csv \
  --mapping configs/cellprofiler_mapping.toml \
  --output comparison.json
```

Use `configs/example_2d.toml` as a starting point, not an accepted final config. Its explicit
max projection, 3/5/10/20-pixel texture distances, and five radial bins reproduce the
preliminary feature family but still require parity and physical-scale review.

## 5. Review a pilot before the full run

On a CUDA-capable machine with the selected model weights:

```bash
uv run dayana-nuclei doctor
uv run dayana-nuclei benchmark configs/final_3d.toml --limit 5
uv run dayana-nuclei run configs/final_3d.toml
uv run dayana-nuclei qc results/<run-id>
uv run dayana-nuclei qc-report results/<run-id>
uv run dayana-nuclei prepare-analysis results/<run-id>
```

Confirm complete Z coverage, saturation/focus acceptability, object counts, split/merge
rates, border exclusions, and segmentation quality in every biological group. Manually tag
technical failures. `prepare-analysis` folds those tags into the derived table without
modifying `nuclei.parquet`.

## 6. Validate additional channels before interpreting them

Confirm channel alignment and identical dimensions/calibration. Select Lamin A/C shell width
and MitoTracker near/far ring widths from acquisition resolution and biological rationale,
then document them. Do not equate MitoTracker intensity with mtDNA copy number or Hoechst
texture with chromatin accessibility.

## 7. Close the biological prerequisites

- Confirm low/high separation through matched mtDNA:nDNA qPCR using mitochondrial and
  nuclear loci.
- Keep conclusions at SYBR-low/SYBR-high until that validation supports stronger language.
- Analyze nuclei as observations nested within fields and independent SortIDs; do not treat
  every nucleus as an independent biological replicate.
- Link imaging and AFM only through the supported matched experimental structure, without
  claiming causality from association.

## Final-run acceptance artifacts

Preserve together:

- reviewed manifest and acquisition record;
- accepted 2D and 3D configs;
- segmentation reference masks and validation report;
- CellProfiler parity report;
- run provenance and input hashes;
- masks, raw and analysis-ready Parquet tables;
- manual QC annotations, overlays, and QC report;
- the decision record accepting the segmentation/configuration.
