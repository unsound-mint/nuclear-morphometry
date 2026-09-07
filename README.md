# dayana-nuclei

GPU-first, reproducible pipeline for 2D and true-3D analysis of colorectal-cancer nuclei
from confocal microscopy, supporting an undergraduate BIOL 490 Honors Thesis.

## What this does

Given Zeiss CZI or TIFF microscopy files and an experimental manifest, `dayana-nuclei`:

1. Reads images with BioIO, preserving physical pixel/voxel spacing and never fabricating
   it when the source file doesn't declare it.
2. Segments Hoechst-defined nuclei with Cellpose-SAM on the GPU.
3. Computes a small, predefined set of per-nucleus 2D and calibrated-3D measurements.
4. Flags technical QC problems (border truncation; more to come) without ever excluding a
   nucleus for being biologically extreme in shape.
5. Writes tidy per-nucleus and per-field Parquet tables, plus full run provenance.

See `Dayana_Nuclei_Complete_Build_Spec.md` for the complete specification and `AGENTS.md`
for engineering rules and what's implemented vs. still pending.

## Scientific scope and non-goals

This is thesis-specific software, not a general microscopy platform (see spec section 51
for the explicit out-of-scope list: no multi-GPU scheduling, no image registration, no
automatic biological conclusions, no qPCR/AFM analysis).

**Read before analyzing real data:**

- **SortID is the biological replicate. Individual nuclei are nested observations, not
  independent replicates** (spec section 1.3). Final biological comparisons require at
  least 3 independent sorts.
- **Nuclei are never excluded for being eccentric, elongated, low-solidity, or
  low-circularity** — those may be the phenotype under study. QC only flags technical
  failures (border truncation so far; debris/merge/split annotation is not yet
  implemented). See `docs/decisions/0002-no-phenotype-based-qc-filtering.md`.
- Terminology stays **SYBR-low / SYBR-high**, not mtDNA-low/high, until matched molecular
  validation (mtDNA:nDNA qPCR) supports the stronger claim.

## Installation (Linux)

```bash
uv sync                                   # core: I/O, config, CLI, 2D/3D measurements
uv sync --extra gpu                       # + torch, cellpose (CUDA segmentation)
uv sync --extra gui                       # + napari (interactive QC viewer)
uv sync --extra dev                       # + pytest, ruff, pyright
uv sync --extra gpu --extra gui --extra dev   # everything
```

### CUDA prerequisites

A CUDA-capable GPU and driver are required for production segmentation. Check with:

```bash
uv run dayana-nuclei doctor
```

This reports Python/package versions, CUDA availability and GPU name/VRAM, Cellpose
version, and whether BioIO CZI support and napari import correctly. It exits non-zero only
if the environment cannot support ordinary CPU development (e.g. BioIO's CZI plugin is
missing) — a CUDA-less machine is a valid `doctor` pass for development/testing with
`segmentation.backend = "fixture"`.

## Inspecting a source file

```bash
uv run dayana-nuclei inspect /path/to/field.czi
uv run dayana-nuclei inspect /path/to/field.czi --scene 1 --json
```

Prints format, scenes, dimensions, axis order, channel names, dtype, and X/Y/Z physical
spacing (with a warning if calibration looks fabricated or is missing — see
`docs/decisions/0006-tiff-uncalibrated-resolution-detection.md`).

## Manifest

Analysis never parses filenames at run time — it always reads a long-form manifest (one
row per image channel). Build a draft from the thesis filename convention and validate it:

```bash
uv run dayana-nuclei manifest build /data/dayana --output manifest.csv
uv run dayana-nuclei manifest validate manifest.csv
```

Always run `validate` before using a manifest for analysis, whether or not it came from
`manifest build`.

## Running an analysis

```toml
# configs/example_2d.toml / configs/example_3d.toml are worked examples.
```

```bash
# 2D (legacy-compatible; fails loudly if the source has multiple Z planes and
# analysis.projection = "none", rather than silently max-projecting):
uv run dayana-nuclei run configs/example_2d.toml

# True 3D (requires real, non-fabricated X/Y/Z physical spacing; never assumes 1/1/1):
uv run dayana-nuclei run configs/example_3d.toml
```

`segmentation.model = "auto"` fails loudly with instructions to run segmentation
validation and pin a model explicitly (spec section 11.1) — Cellpose 4.x has no
tissue-specific "nuclei" model to silently fall back to. For architecture/demo testing
only, pass `--allow-unvalidated-model`; never use that flag for a result you intend to
report.

## Resuming an interrupted run

```bash
uv run dayana-nuclei resume results/<run-id>
```

Skips fields already marked `complete`; retries `pending`, `failed`, and `running`
(interrupted mid-processing) fields. Never duplicates rows — each field owns exactly one
partial output file that a retry simply overwrites.

## Segmentation validation

Segmentation quality is a scientific gate, not optional tooling (spec section 14) — do not
trust a segmentation backend/model for real thesis analysis without running this first.

### Building the reference set

Manually correct (in napari, or any label-editing tool) a fixed set of reference masks,
stratified across:

- SW480 and SW620
- low, high, and bulk conditions
- independent sorts (SortID)
- sparse fields (few, well-separated nuclei)
- dense/touching nuclei
- irregular nuclei
- elongated nuclei
- dim nuclei
- any other case that was visibly difficult during a manual look at raw data

The point of stratification is that a model can look excellent on easy sparse fields and
fail badly on dense or dim ones — the reference set must contain the hard cases on purpose.

### Running the check

Single pair:

```bash
uv run dayana-nuclei validate-segmentation \
    --prediction results/<run-id>/masks/<image_id>_labels.tif \
    --reference path/to/manually_corrected_labels.tif \
    --output validation_report.json
```

Batch, via a CSV with columns `case_id, prediction_path, reference_path`:

```bash
uv run dayana-nuclei validate-segmentation --manifest validation_cases.csv --output report.json
```

Metrics computed (spec 14.2): predicted/reference object counts, one-to-one IoU-matrix +
Hungarian matching, mean/median matched IoU, precision/recall/F1 at `--iou-threshold`
(default 0.5 — a convention, not a validated scientific claim for this dataset; never
hard-coded, always pass what you intend), and unmatched counts on both sides. Precision is
reported as undefined rather than a misleading 0.0 when there are zero predicted objects to
score (recall, symmetrically, for zero reference objects). `--estimate-split-merge` adds an
optional, disabled-by-default heuristic (spec 22.2) flagging probable over-/under-
segmentation — it never filters or deletes anything, only reports labels to look at.

### Acceptance (spec 14.3)

This command only computes numbers. **Do not treat a passing metric alone as acceptance.**
Manually review representative overlays (prediction vs. reference, and prediction vs. raw
image) before trusting a model, and record the accepted model/config and the reference set
used in a new file under `docs/decisions/` — see `docs/decisions/0008-cellpose-3d-
oversegmentation.md` for the kind of finding this process exists to catch.

## QC

Manual QC annotations (`good`/`debris`/`merge`/`split`/`other`, keyed by
`(image_id, object_number)`) are stored under `results/<run-id>/qc/annotations.json` and
never mutate `nuclei.parquet` — see `docs/measurement-dictionary.md`. Image-level QC metrics
(intensity range, saturation fraction, a focus/blur proxy, occupied fraction) are already
computed into every run's `fields.parquet` (spec section 21).

A static QC report (spec section 24) is generated with:

```bash
uv run dayana-nuclei qc-report results/<run-id> [--seed 0] [--n-overlays 6]
```

This writes a self-contained `results/<run-id>/qc/report.html` plus PNG overlays under
`results/<run-id>/qc/overlays/` — run identity, counts by cell line/SortID/condition, border
object count and fraction, manual QC tag counts (if any), the `fields.parquet` image
saturation/focus/runtime summaries, and representative segmentation-boundary overlays.
Overlay fields are picked by a seeded, stratified sample across cell line × condition ×
SortID (not just the first or easiest fields), and only rendered when `output.save_masks =
true` was set for the run.

### Interactive viewer (spec section 23)

```bash
uv sync --extra gui   # if not already installed
uv run dayana-nuclei qc results/<run-id> [--image-id SW620_Sort01_low_48h_Field001]
```

Requires `output.save_masks = true` for the run. Opens napari with the raw Hoechst image,
the segmentation labels, and any configured additional channels (toggle visibility per
layer) for one field at a time (switch fields from the dock widget's dropdown). Z scrolling
and 2D/3D label rendering for 3D stacks are napari's own built-in controls — nothing
field-specific is needed to enable them.

Click a nucleus in the labels layer to select it; its object number and key measurements
(area/perimeter/circularity/eccentricity for 2D, volume/surface area/sphericity for 3D, plus
intensity where measured) appear in the dock widget. Tag it `good`/`debris`/`merge`/`split`/
`other` either via the dock's buttons or the `g`/`d`/`m`/`s`/`o` keyboard shortcuts.
Annotations are written to `results/<run-id>/qc/annotations.json`, keyed by
`(image_id, object_number)`, and already-tagged objects are shown as a colored points overlay
that reloads automatically the next time the viewer opens — reopening never loses prior
review. Neither the raw source file nor the saved mask is ever modified: the labels layer
shown to napari is a plain in-memory copy and is not editable from the viewer.

## Performance benchmarking

```bash
uv run dayana-nuclei benchmark configs/example_3d.toml [--limit 5] [--output path.json]
```

Runs the real pipeline (same code path as `run`, model loaded once) against the config's
manifest, but writes to a scratch directory instead of `results/` — a benchmark measures
timing, it does not produce an analysis run to keep. Reports, per field: wall-clock time for
I/O, segmentation normalization, GPU segmentation, mask serialization, morphology,
intensity, texture, radial distribution, additional channels, QC, row assembly, and
Parquet output, plus
image dimensions, object count, and total time; and for the whole run: peak process RSS and
peak CUDA memory (0.0 if CUDA is available but unused, `null` if no CUDA device is present).
Saves a full JSON report and a flat per-field Parquet table (same base name) under
`<output_root>/benchmarks/` by default, for before/after comparison across code or config
changes (spec section 32).

## Legacy CellProfiler measurement comparison

```bash
uv run dayana-nuclei compare-measurements \
    --ours results/<run-id>/nuclei.parquet \
    --reference legacy_cellprofiler.csv \
    --mapping configs/cellprofiler_mapping.toml \
    [--output comparison.json]
```

The key scientific precondition (spec section 16.1): compare measurements on the **same
masks** before comparing end-to-end segmentation pipelines, so segmentation differences
never contaminate measurement-definition parity testing — run this pipeline's measurement
stage on CellProfiler's own exported label masks, or export CellProfiler measurements
against this pipeline's masks, before running this command. `--reference` accepts `.csv` or
`.parquet`. `--mapping` is a TOML file naming the join keys (default
`image_id`/`object_number` on both sides) and the reference-column → our-column measurement
pairs to compare — see `configs/cellprofiler_mapping.toml` for the expected shape and its
caveats. The report gives, per mapped measurement over matched objects: mean/median/max
absolute difference, mean relative difference (`None` when every matched reference value is
exactly 0 — never a fabricated number), and the Pearson correlation. Objects present on only
one side are counted, never silently dropped. If exact CellProfiler semantics for a feature
are unknown, this tool cannot tell you that — document the difference in `docs/decisions/`
once you know it (spec 16.1).

## Exporting tables

```bash
uv run dayana-nuclei prepare-analysis results/<run-id>   # adds include_default
uv run dayana-nuclei export-csv results/<run-id>          # re-export nuclei.csv
uv run dayana-nuclei finalize-run results/<run-id> --hash-inputs   # archival provenance
```

## Output layout

```text
results/<run-id>/
    config.toml           exact config used
    provenance.json        code/config/manifest/package/GPU/model identity
    manifest.parquet
    fields.parquet         one row per analyzed field
    nuclei.parquet         one row per nucleus (canonical; nuclei.csv for interop)
    run_state.json          per-field pending/running/complete/failed state
    masks/                  one label TIFF per field
    qc/                     (QC report/overlays, once implemented)
    logs/
```

`nuclei.parquet` contains every successfully measured object, including ones excluded by
default (e.g. border-truncated), with the exclusion reason preserved — nothing is deleted.

## Reproducibility

Every run's `provenance.json` records the git commit and dirty-tree status, config and
manifest SHA-256 hashes, Python/package/Torch/Cellpose versions, GPU identity, and the
resolved segmentation parameters. `finalize-run --hash-inputs` additionally hashes the raw
source files for archival/final-thesis runs (skipped by default since it can be slow for
large files).

## Further reading

- `docs/architecture.md` — module boundaries and data flow.
- `docs/measurement-dictionary.md` — every implemented column: units, definition, source
  channel, primary/secondary status, caveats.
- `docs/decisions/` — ADRs for every scientifically meaningful assumption or third-party
  API adaptation made during implementation.
- `AGENTS.md` — engineering rules and current implementation status.
