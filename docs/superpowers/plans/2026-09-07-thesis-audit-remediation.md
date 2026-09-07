# Thesis Audit Remediation Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct every repository-side issue found in the thesis brief audit and leave a precise, executable list of real-data blockers.

**Architecture:** Preserve raw run outputs as immutable evidence. Apply manual annotations only while deriving `analysis_ready.parquet`, keep 2D and 3D configurations separate, and make automated gates accurately reflect optional GPU and GUI environments.

**Tech Stack:** Python 3.12, Polars, pytest, Pyright, Ruff, GitHub Actions, napari/Qt, Cellpose-SAM.

## Global Constraints

- Never modify raw microscopy files.
- Never infer X/Y/Z physical spacing or segmentation acceptance.
- Never exclude an object because of eccentricity, solidity, circularity, elongation, or irregularity.
- Preserve every object and its exclusion reason.
- Keep the computational core independent of napari and Qt.
- Do not change a scientific measurement definition in this remediation.

---

### Task 1: Fold manual QC into the derived analysis table

**Files:**
- Modify: `src/dayana_nuclei/export.py`
- Modify: `tests/unit/test_schema_export.py`

**Interfaces:**
- Consumes: `apply_annotations_to_nuclei(nuclei_df, run_dir) -> pl.DataFrame`
- Produces: `prepare_analysis(run_dir) -> Path` with manual technical exclusions reflected in `include_default`

- [x] Add a regression test that saves a manual merge annotation and proves the raw table stays unchanged while the derived table sets `qc_manual_merge=true`, `qc_exclusion_reason="manual_merge"`, and `include_default=false`.
- [x] Run the focused test and confirm it fails because `prepare_analysis` does not consume annotations.
- [x] Apply annotations before deriving `include_default`.
- [x] Run annotation and export unit tests.

### Task 2: Restore static-analysis and CI gates

**Files:**
- Modify: `tests/integration/test_pipeline_e2e.py`
- Modify: `tests/integration/test_qc_viewer.py`
- Modify: `tests/unit/test_compare_measurements.py`
- Modify: `tests/unit/test_spatial.py`
- Modify: `.github/workflows/ci.yml`

**Interfaces:**
- Produces: zero-error `uv run pyright`
- Produces: CPU CI that excludes the optional GUI tests and a separate GUI test job with the GUI dependency

- [x] Add explicit assertions or typed layer checks where tests currently dereference optional or generic values.
- [x] Update CI comments and dependency/test selection to match the implemented viewer.
- [x] Run Pyright, Ruff, formatting, and CPU tests; isolate the GUI test in CI because this sandbox cannot start Xvfb.

### Task 3: Supply a thesis-ready 2D replication example and truthful documentation

**Files:**
- Modify: `configs/example_2d.toml`
- Modify: `README.md`

**Interfaces:**
- Produces: an explicit max-projection 2D example with predefined Hoechst texture and five radial bins

- [x] Enable the required 2D texture/radial measurements and make projection explicit.
- [x] Correct stale README statements about QC, output layout, spec location, and manual annotation derivation.
- [x] Document that example values are starting points requiring dataset validation, not accepted thesis parameters.

### Task 4: Verify without weakening scientific gates

**Files:**
- No scientific implementation changes.

**Interfaces:**
- Produces: an evidence-backed list separating repository completion from real-data scientific acceptance

- [x] Run focused regression tests.
- [x] Run Ruff check and format check.
- [x] Run Pyright.
- [x] Run the full CPU suite and report GPU/GUI limitations separately.
- [x] Re-inspect representative TIFF calibration and enumerate missing experimental inputs and acceptance artifacts.

### Task 5: Enforce multi-channel physical consistency

**Files:**
- Modify: `src/dayana_nuclei/pipeline/analyze.py`
- Modify: `tests/integration/test_multichannel_pipeline.py`
- Create: `docs/decisions/0013-multichannel-spacing-consistency.md`

**Interfaces:**
- Produces: `_validate_channel_compatibility(...) -> None`

- [x] Add a regression test for equal-shape channels with mismatched calibration.
- [x] Confirm the test fails because the compatibility boundary is absent.
- [x] Validate axes, shape, and X/Y/Z spacing before reusing Hoechst labels.
- [x] Document the fail-loud scientific decision.

### Task 6: Make acquisition-record calibration usable

**Files:**
- Modify: `src/dayana_nuclei/models.py`
- Modify: `src/dayana_nuclei/io/manifest.py`
- Modify: `src/dayana_nuclei/io/images.py`
- Modify: `tests/unit/test_manifest.py`
- Modify: `tests/unit/test_io_images.py`
- Create: `docs/decisions/0014-explicit-manifest-spacing-overrides.md`

**Interfaces:**
- Consumes: optional manifest columns `spacing_x_um`, `spacing_y_um`, `spacing_z_um`
- Produces: `ImageSource.spacing_override: PhysicalSpacing | None`

- [x] Add failing tests for metadata-stripped TIFFs and conflicting calibration.
- [x] Validate positive, complete manifest overrides.
- [x] Use explicit overrides only when embedded values are absent and reject conflicts.
- [x] Preserve the calibration in the copied/hashed run manifest and document its provenance.
