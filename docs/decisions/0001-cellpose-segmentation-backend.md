# 0001 - Cellpose segmentation backend and API version adaptation

## Status

Accepted. Implementation of `segmentation/cellpose_backend.py` itself is deferred to
a later work session (see AGENTS.md phase plan); this record captures the API
investigation so that implementation does not have to re-derive it.

## Context

The build spec (section 5.1, 13.2) requires a CUDA-capable Cellpose backend with
per-tissue model selection (`model = "auto"` resolving to a validated default,
spec 11.1) and explicit physical anisotropy support for 3D.

The environment resolved **Cellpose 4.2.1.1** (`uv sync --extra gpu`), which is a
materially different API from the Cellpose 2.x/3.x generation the spec's phrasing
implicitly assumes:

- `cellpose.models.CellposeModel.__init__` **ignores** `model_type` and `diam_mean`
  as of v4.0.1+ (logs a deprecation warning, does not raise). There is no longer a
  menu of tissue-specific pretrained models (no `"nuclei"`, `"cyto"`, `"cyto2"`,
  etc.). `MODEL_NAMES = ["cpsam_v2", "cpdino", "cpdino-vitb", "cpsam"]` — these are
  variants of a single foundation model family ("Cellpose-SAM"), not tissue-specific
  models. `pretrained_model` defaults to `"cpsam_v2"`.
- `CellposeModel.eval(...)` signature confirmed:
  `(x, batch_size=8, resample=True, channels=None, channel_axis=None, z_axis=None,
  normalize=True, rescale=None, diameter=None, flow_threshold=0.4,
  cellprob_threshold=0.0, do_3D=False, anisotropy=None, flow3D_smooth=0,
  stitch_threshold=0.0, min_size=15, max_size_fraction=0.4, niter=None,
  augment=False, tile_overlap=0.1, bsize=None, compute_masks=True, progress=None)`.
  **`anisotropy` is a supported keyword** — spec section 13.2's requirement
  ("physical anisotropy passed to the model when the current Cellpose API supports
  it") is satisfiable: pass `anisotropy = z_um / xy_um` when `do_3D=True`.
- `device` accepts a `torch.device` directly (`torch.device("cuda")` /
  `torch.device("cuda:N")`), which is what the backend should use for explicit
  device selection (spec 13.2) rather than the boolean `gpu=` flag alone.
- Verified in this environment: `torch==2.14.0+cu130`, `cuda.is_available() == True`,
  device = NVIDIA GeForce RTX 4070 Laptop GPU, 8.2 GB VRAM.

## Decision

1. `segmentation/cellpose_backend.py` will construct `CellposeModel` with
   `pretrained_model=<resolved model id>` and an explicit `device=torch.device(...)`,
   never the bare `gpu=True` shortcut, so device selection is auditable in
   provenance.
2. `model_type` and `diam_mean` will not be exposed in `configs/*.toml` — they are
   dead parameters in the installed API and configuring them would misleadingly
   imply per-tissue model selection still exists.
3. **`model = "auto"` resolution**: per spec 11.1, this must fail loudly with a
   message instructing the user to run `validate-segmentation`, unless the run is
   explicitly in a documented demo mode. Because Cellpose 4.x no longer offers a
   "nuclei" model to fall back to, the demo-mode default (when the user
   opts in) is `"cpsam_v2"` — the library's own current foundation-model default —
   used unvalidated and logged as such. This will be implemented as an explicit
   `--allow-unvalidated-model` CLI flag on `run`/`benchmark` (not a silent config
   default), recorded verbatim in `provenance.json`.
4. `diameter_um` in `configs/*.toml` maps to Cellpose's `diameter` eval kwarg,
   converted from microns using the field's X/Y pixel size; `diameter_um = 0.0`
   maps to `diameter=None` (Cellpose's own auto-diameter behavior for this model
   family), not to a hard-coded pixel value.
5. `anisotropy` is computed as `spacing.z_um / spacing.x_um` (Cellpose assumes
   square XY pixels for this parameter) and passed only when `do_3D=True` and
   `segmentation.3d.use_anisotropy = true`; if XY pixels are non-square, this will
   be flagged as a warning in provenance rather than silently averaged.

## Consequences

- The measurement-dictionary and README language should describe the backend as
  "Cellpose-SAM (v4.x)" rather than implying tissue-specific model selection.
- Segmentation validation (spec section 14) becomes more, not less, important:
  with a single foundation model rather than a curated "nuclei" model, empirical
  validation against the reference mask set is the only source of confidence that
  `cpsam_v2` (or a fine-tuned variant) is acceptable for Hoechst-stained SW480/SW620
  nuclei.
- If a project-specific fine-tuned model is later trained, its path is passed
  directly as `pretrained_model` (CellposeModel accepts a filesystem path), which
  the existing config's `model` field already supports as a non-`"auto"` string.
