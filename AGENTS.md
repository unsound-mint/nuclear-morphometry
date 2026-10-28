# Project

GPU-first nuclear morphometry pipeline for confocal microscopy (2D and true 3D).

This is focused scientific software, not a general image-analysis framework. The design is
in `docs/design-spec.md`; read `docs/measurement-dictionary.md` and the relevant files under
`docs/decisions/` before changing measurements or QC.

## Development commands

```bash
uv sync --extra dev --extra gpu --extra gui
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest tests/unit tests/integration -q -m "not gpu and not cellpose_model and not gui"
```

`gpu` and `cellpose_model` tests need a CUDA device and downloaded weights; `gui` tests need
Qt (CI runs them under `xvfb-run`). Tests skip cleanly when their requirement is missing.

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
- GUI (`qc/viewer.py`)

The computational core must not depend on napari or Qt: `cli.py` imports `qc/viewer.py` (and
therefore napari) lazily, inside the `qc` command body, so every other command works without
the optional `gui` dependency group installed.

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
- The backend targets Cellpose 4.x (Cellpose-SAM), which has no tissue-specific model
  generation. See
  `docs/decisions/0001-cellpose-segmentation-backend.md` before touching
  `segmentation/cellpose_backend.py` -- `model_type`/`diam_mean` are dead
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
