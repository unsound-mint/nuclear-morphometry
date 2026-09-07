# Project

GPU-first microscopy analysis pipeline for an undergraduate BIOL 490 thesis.

This is thesis-specific scientific software, not a general image-analysis framework.

## Priorities

1. Scientific correctness and reproducibility.
2. Segmentation validity across all biological conditions.
3. Performance.
4. Simplicity and maintainability.
5. Convenience.

Do not trade scientific correctness for performance.

## Before changing code

- Read the nearest relevant implementation and tests first.
- Do not introduce a dependency unless existing dependencies cannot solve the problem cleanly.
- For non-trivial changes, state the intended approach and invariants before implementation.
- Profile before performance optimization.
- Do not change validated scientific semantics as part of refactoring.

## Scientific invariants

- Raw microscopy files are immutable.
- True-3D analysis requires real X/Y/Z physical spacing. Never silently assume isotropic voxels.
  See `docs/decisions/0003-physical-units-for-3d.md`.
- Preserve CellLine, SortID, Condition, Timepoint, Field, and ObjectNumber for every nucleus.
- SortID is the biological replicate. Nuclei are nested observations.
- Never exclude nuclei because they are biologically irregular, elongated, eccentric,
  low-solidity, or low-circularity. See `docs/decisions/0002-no-phenotype-based-qc-filtering.md`.
  `qc/flags.compute_object_qc` has no shape-statistic parameters at all -- keep it that way.
- QC may remove/flag technical failures such as border truncation, debris, segmentation
  failure, saturation, or focus failure.
- Preserve rejected objects and the reason for rejection (`qc_exclusion_reason`); never delete
  a row.
- Do not change validated Hoechst segmentation when adding other channels without explicit
  revalidation.
- Do not silently change a measurement's mathematical definition. Document any change (or any
  adaptation forced by a third-party API/version change) in `docs/decisions/`.

## Architecture

Keep separate:
- microscopy I/O and metadata (`io/`)
- segmentation (`segmentation/`)
- measurements (`measurements/`)
- QC (`qc/`)
- pipeline execution (`pipeline/`)
- export/provenance (`export.py`, `provenance.py`)
- GUI (`qc/viewer.py`, not yet implemented)

The computational core must not depend on napari or Qt.

Every per-field nuclei DataFrame must be constructed against
`schema.NUCLEI_TABLE_SCHEMA` (explicit dtypes, including for zero-row frames) -- see
`docs/decisions/0004-parquet-as-canonical-table.md` for why an implicit schema broke a whole
run over a single empty field.

## GPU/performance

- Load the model once per run.
- Use CUDA when requested.
- Use inference mode.
- Avoid unnecessary CPU<->GPU transfers.
- Do not add CuPy, cuCIM, Dask, multiprocessing, or distributed execution without profiling
  evidence.
- Bound memory use.
- Do not optimize by silently downsampling or changing scientific semantics.
- Performance-sensitive changes require benchmark evidence.
- Cellpose in this environment resolved to 4.2.1.1 (Cellpose-SAM), not the tissue-specific-model
  generation the original spec assumed. See
  `docs/decisions/0001-cellpose-segmentation-backend.md` before touching
  `segmentation/cellpose_backend.py` (not yet implemented) -- `model_type`/`diam_mean` are dead
  parameters in this version, and `model = "auto"` has no "nuclei" model to fall back to.

## Testing

Every scientific measurement requires analytical or independently verified tests.
Maintain unit, integration, regression, and benchmark coverage.
A segmentation backend/model change requires rerunning the fixed validation set.

Never change expected scientific outputs merely to make a failing test pass.

The negative case matters as much as the positive one: any change touching QC must keep (or add)
a test asserting a shape-extreme synthetic nucleus is *not* excluded, not just that a border
nucleus *is*.

## Outputs

Canonical tables are Parquet. Each run preserves config, manifest, masks, measurements, QC, and
provenance under `results/<run-id>/`.

`qc_excluded_default` (raw table) and `include_default` (analysis-ready table) are the same
decision computed once and derived once -- see
`docs/decisions/0005-include-default-semantics.md` before adding a new exclusion path.

## Errors

Validate at boundaries and fail early with actionable errors (what's wrong, how to inspect, what
to fix -- see spec section 49's style, followed throughout `io/`, `models.py`, `config.py`).
Do not silently fall back from GPU to CPU in production.

## Definition of done

A change is complete only when:
- relevant tests pass
- Ruff and Pyright pass
- scientific invariants remain satisfied
- error paths are handled
- provenance remains complete
- performance has not materially regressed without justification
- behavior/measurement documentation is updated

## Current state

Implemented: project scaffold, domain model, config, manifest build/validate, BioIO CZI/TIFF
I/O with physical calibration, mask persistence, the deterministic `FixtureSegmenter`, the
Cellpose-SAM backend (`segmentation/cellpose_backend.py`), 2D and 3D morphology, intensity,
2D texture, border-only QC flags, image-level QC metrics (`qc/image_metrics.py`, spec 21,
measurement-only), manual QC annotation storage (`qc/annotations.py`, spec 22.2/23 --
napari-independent JSON store; the interactive viewer itself is not yet implemented),
segmentation validation tooling (`segmentation/validation.py` + `validate-segmentation` CLI,
spec 14 -- IoU matrix + Hungarian matching, precision/recall/F1 with undefined-not-zero
semantics when a count is 0, a containment-based (not IoU-based -- see
`docs/decisions/0008`) disabled-by-default split/merge heuristic), a static QC report
(`qc/report.py` + `qc-report` CLI, spec 24 -- seeded stratified overlay sampling across
cell line/condition/SortID, self-contained HTML, no report framework), multi-channel
measurements (spec 25, Phase 8 -- `measurements/lamin.py` shell/core via
physically-calibrated erosion, `measurements/spatial.py` MitoTracker perinuclear rings via
nearest-nucleus disambiguation, see `docs/decisions/0009`), 2D radial intensity
distribution (`measurements/radial.py`, spec 20 -- a from-scratch, equal-pixel-count
boundary-distance-ranked binning, not CellProfiler parity, see `docs/decisions/0010`),
per-stage performance benchmarking (`pipeline/benchmark.py` + `benchmark` CLI, spec 32 --
I/O, segmentation normalization, GPU segmentation, mask serialization, morphology,
intensity, texture, radial distribution, additional channels, QC, row assembly, and output
timing per field, plus peak RSS/CUDA memory, image dimensions, and object counts; saves a
JSON report
and a Parquet table for before/after comparison), legacy CellProfiler measurement-parity
comparison (`compare_measurements.py` + `compare-measurements` CLI, spec 16.1/28 -- joins on
exact object-key match across two tables measuring the *same masks*, reports per-mapped-
column absolute/relative difference and Pearson correlation with undefined-not-fabricated-
zero semantics; does not attempt fuzzy object matching), atomic per-field pipeline execution
with resume, Parquet export, `prepare-analysis`, provenance, CI, and the full CLI surface
(unimplemented commands fail loudly rather than silently stubbing).

**2D Cellpose segmentation is verified working end-to-end on real GPU hardware in this
session** (`tests/integration/test_cellpose_gpu.py`). **3D Cellpose segmentation initially
showed severe over-segmentation (25-434 spurious objects for a single synthetic sphere);
this was traced to a real bug** (un-normalized input reaching Cellpose, since
`CellposeSegmenter` always calls `eval(normalize=False)` trusting the pipeline to have
normalized first) **and is now fixed at two layers**: `Config` rejects
`backend = "cellpose"` with `normalize_for_segmentation = false` outright, and the
realistic-signal synthetic case (smoothly-varying intensity + noise, not a hard binary
edge) segments correctly once normalized. Two pathological binary-edge synthetic shapes
still fragment unexplained, but do not resemble real fluorescence signal — see
`docs/decisions/0008-cellpose-3d-oversegmentation.md`. Confidence in 3D is higher than
before, but `validate-segmentation` against a real reference mask set (spec section 14)
is still required before trusting a 3D analysis run's object count or masks.

Not yet implemented (see `Dayana_Nuclei_Complete_Build_Spec.md` section 52 for the full phase
plan): the interactive napari QC viewer (`qc/viewer.py` -- the intended way to actually
produce manual annotations). This CLI command currently exits with an explicit "not yet
implemented" message rather than a bare stub.
`measurements.texture_distances_um` (physical-scale texture mode, spec 19.4) is now wired
into the pipeline: `_process_field` converts per-field, names columns by the configured um
value (not the per-field resolved pixel distance), and requires square X/Y pixels for the
field being measured. See `docs/decisions/0012-texture-um-distance-column-naming.md`.
