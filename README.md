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
uv sync --extra gui                       # + napari (QC viewer, not yet implemented)
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

## QC

Not yet implemented in this build: the interactive napari viewer (`qc`) and static QC
report (`qc-report`). Both currently exit with an explicit "not yet implemented" message
naming the spec section that will implement them, rather than doing nothing silently.

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
