# Nuclear Morphometry — Complete Build Specification

**Target implementer:** Claude Sonnet 5 (or another strong coding agent)  
**Project type:** Thesis-specific scientific image-analysis application  
**Primary objective:** Replace the current CellProfiler-based workflow with a reproducible, GPU-first Python pipeline for 2D and true-3D analysis of colorectal-cancer nuclei from confocal microscopy.

---

## 0. Instruction to the implementation agent

You are the primary implementation agent for this repository.

**Build the complete application described in this document end-to-end.** Do not stop after scaffolding, pseudo-code, or a proof of concept. Implement the CLI, data model, microscopy I/O, segmentation backend, measurements, QC workflow, exports, tests, documentation, and performance/provenance infrastructure described below.

Work in small, reviewable increments internally, but continue until the full Definition of Done is satisfied.

### Operating rules

1. Read this entire specification before changing files.
2. Scientific correctness and reproducibility are more important than code aesthetics or marginal speed.
3. Performance is a first-class requirement where it does not alter scientific semantics.
4. Do not silently invent missing experimental facts.
5. If a feature depends on data that are not yet available, implement the feature/interface and tests with synthetic fixtures, fail clearly when required real metadata are absent, and document what must be validated on the real data.
6. Do not ask the user to choose ordinary engineering details that this specification already resolves.
7. Do not broaden this into a generic microscopy platform. This is a thesis-specific pipeline.
8. Avoid unnecessary dependencies and abstractions.
9. Run tests, Ruff, and Pyright before considering the build complete.
10. Report any scientifically meaningful assumption or API/version adaptation in `docs/decisions/`.
11. Do not silently fall back from CUDA to CPU in a production configuration.
12. Do not change segmentation resolution, physical calibration, measurement definitions, or exclusion semantics as a performance shortcut.

The application may be developed on a machine without the final microscopy data or CUDA GPU. CPU-only unit tests and synthetic fixtures must still work. GPU-dependent integration tests must be cleanly marked/skipped when CUDA is unavailable.

---

# 1. Scientific context and source requirements

This application supports an undergraduate BIOL 490 Honor Thesis research on **mtDNA-associated cellular states, nuclear architecture, chromatin organization, and nuclear mechanics in colorectal cancer cells**.

The biological study uses:

- **Cell lines:** SW480 and SW620.
  - SW480 is primary-tumor-derived.
  - SW620 is lymph-node-metastasis-derived from the same patient.
- **Core sorted populations:** SYBR-low, SYBR-high, and bulk-sorted control.
  - SYBR-medium may exist in preliminary data but is not a required core group.
- **Biological replicate:** one independent sorting experiment.
  - Final target: at least 3 independent sorts.
- **Main imaging endpoint:** currently 48 h post-sort, subject to matched molecular validation.
- **Primary imaging channel:** Hoechst/DAPI-like nuclear DNA channel.
- **Additional channels/markers:** H3K9Ac, Lamin A/C, MitoTracker; H3K9me3 may also be used.
- **Mechanical endpoint outside this software:** isolated-nucleus AFM, including Young's modulus and nuclear height.
- **Validation outside this software:** post-sort flow re-analysis and mtDNA:nDNA qPCR.

The image-analysis objective is to produce robust, reproducible **per-nucleus** measurements in both 2D and true 3D **without filtering away abnormal morphology that may itself be the phenotype**.

## 1.1 Biological questions the image analysis must support

The pipeline must support testing whether SYBR-defined cellular states differ in:

- 2D nuclear size.
- Circularity/form factor.
- Elongation/eccentricity.
- Solidity/irregularity.
- True 3D volume.
- Surface area/boundary complexity.
- Nuclear Z-depth / height proxy.
- 3D axis lengths / elongation.
- Sphericity where mathematically valid.
- Hoechst intensity.
- Hoechst texture / image heterogeneity.
- H3K9Ac and optionally H3K9me3 nuclear signal.
- Lamin A/C organization relative to nuclear shape.
- MitoTracker/perinuclear mitochondrial organization.

## 1.2 Interpretation boundaries

The software and generated documentation must preserve these scientific boundaries:

- Until matched mtDNA:nDNA validation supports stronger terminology, refer to groups as **SYBR-low** and **SYBR-high**, not definitively mtDNA-low/high.
- Hoechst texture differences alone do not prove chromatin accessibility, transcriptional activity, or condensation state.
- Image associations do not establish that mtDNA state caused nuclear remodeling.
- Raw MitoTracker signal must not be interpreted as mtDNA copy number.
- Negative results must remain visible; do not feature-select alternatives merely because a primary measurement is non-significant.

## 1.3 Statistical unit

This is non-negotiable:

> **SortID / independent sort is the biological replicate. Individual nuclei are nested observations, not independent biological replicates.**

Every output nucleus must retain its biological hierarchy so downstream analysis cannot accidentally lose this structure.

---

# 2. Preliminary analysis that must remain reproducible

Exploratory SW620 data at 48 h reported approximately:

| Metric | SYBR-high | SYBR-medium | SYBR-low |
|---|---:|---:|---:|
| Nuclear area | ~2264 px² | ~2253 px² | ~2510 px² |
| Circularity | 0.77 | 0.74 | 0.70 |
| Entropy | 2.12 | 1.89 | 2.18 |
| Contrast | 5.05 | 3.55 | 5.73 |

The preliminary interpretation was:

- SYBR-low nuclei appeared about 10–15% larger in 2D.
- SYBR-low nuclei appeared less circular / more irregular.
- Entropy and contrast suggested altered image texture.

These are **preliminary continuity targets**, not hard-coded expected outcomes.

The new software must provide a way to compare its 2D measurements against legacy CellProfiler outputs on the **same masks**, so differences in segmentation do not contaminate measurement-parity testing.

---

# 3. Problems in the existing CellProfiler workflow that this project must fix

The current CellProfiler project is effectively a 2D pipeline:

- `Process as 3D = No`.
- Relative X/Y/Z spacing is effectively `1/1/1`.
- Confocal Z-stacks are therefore not currently represented as physically calibrated volumetric nuclei.

The existing pipeline also used morphology filters such as:

- Area.
- Solidity.
- Eccentricity.
- FormFactor/circularity.

This is dangerous because solidity, eccentricity, circularity, and irregularity are themselves thesis phenotypes.

## 3.1 Required correction

QC must remove or flag **technical failures**, not plausible biological extremes.

Allowed technical QC concepts include:

- Border truncation.
- Obvious debris.
- Clearly failed segmentation.
- Saturated acquisitions.
- Focus failure.
- Manually identified merge/split artifacts.
- Explicitly justified extreme technical size thresholds.

The software must **not** automatically remove a nucleus merely because it is:

- Highly eccentric.
- Low solidity.
- Low circularity.
- Elongated.
- Irregular.

All raw measurements and QC/exclusion reasons must remain auditable.

---

# 4. Product goal

Build a command-line scientific application named:

```text
nuclear-morphometry
```

It must:

1. Read Zeiss CZI and TIFF microscopy data.
2. Preserve physical pixel/voxel spacing and experimental metadata.
3. Segment Hoechst-defined nuclei using a CUDA-capable Cellpose backend.
4. Support legacy-compatible 2D analysis.
5. Support calibrated true-3D analysis from intact Z-stacks.
6. Compute a small, predefined, biologically defensible feature set.
7. Preserve all nuclei and represent QC as flags/annotations rather than phenotype-biased filtering.
8. Export tidy per-nucleus Parquet data and optional CSV.
9. Save segmentation masks and QC artifacts.
10. Provide an interactive napari QC viewer.
11. Provide reproducibility/provenance metadata for every run.
12. Provide validation tooling for segmentation and legacy measurement parity.
13. Be usable both interactively and headlessly.
14. Use the GPU where it materially helps.
15. Avoid CellProfiler as a runtime dependency.

CellProfiler may be used only as a **reference implementation** during 2D measurement validation.

---

# 5. Technology stack

Use the smallest stable stack that satisfies the requirements.

## 5.1 Required

- **Python 3.12**
- **uv** for environment/package management
- **PyTorch** for CUDA device/runtime integration
- **Cellpose** as the primary segmentation backend
- **BioIO**
- **bioio-czi**
- **bioio-tifffile** or an equivalent BioIO TIFF reader if required by the current BioIO ecosystem
- **NumPy**
- **SciPy**
- **scikit-image**
- **Polars**
- **PyArrow**
- **Pydantic v2**
- **Typer**
- **napari**
- **PyQt6** for the napari GUI
- **matplotlib**
- **tifffile** if needed for label-mask export
- **pytest**
- **pytest-benchmark**
- **Ruff**
- **Pyright**

Use current mutually compatible stable versions and lock them in `uv.lock`.

## 5.2 Do not add initially

Do **not** add these unless profiling or a concrete correctness requirement justifies them:

- CuPy
- cuCIM
- Dask
- Ray
- Spark
- Kubernetes
- a database server
- OME-Zarr
- a custom CUDA watershed
- a custom napari plugin package
- a generic workflow engine
- a web frontend
- Jupyter as part of the production pipeline

If a later optimization introduces one of these, document the measured bottleneck and before/after benchmark in `docs/decisions/`.

---

# 6. Repository layout

Create approximately this structure. Minor naming changes are acceptable only when they improve cohesion without adding layers.

```text
nuclear-morphometry/
├── AGENTS.md
├── CLAUDE.md
├── README.md
├── pyproject.toml
├── uv.lock
├── .gitignore
├── .python-version
│
├── configs/
│   ├── example_2d.toml
│   ├── example_3d.toml
│   └── example_multichannel.toml
│
├── src/
│   └── nuclear_morphometry/
│       ├── __init__.py
│       ├── cli.py
│       ├── config.py
│       ├── models.py
│       ├── provenance.py
│       ├── logging_utils.py
│       │
│       ├── io/
│       │   ├── __init__.py
│       │   ├── images.py
│       │   ├── metadata.py
│       │   ├── manifest.py
│       │   └── masks.py
│       │
│       ├── segmentation/
│       │   ├── __init__.py
│       │   ├── base.py
│       │   ├── cellpose_backend.py
│       │   ├── normalize.py
│       │   └── validation.py
│       │
│       ├── measurements/
│       │   ├── __init__.py
│       │   ├── morphology_2d.py
│       │   ├── morphology_3d.py
│       │   ├── intensity.py
│       │   ├── texture.py
│       │   └── spatial.py
│       │
│       ├── qc/
│       │   ├── __init__.py
│       │   ├── flags.py
│       │   ├── annotations.py
│       │   ├── overlays.py
│       │   ├── report.py
│       │   └── viewer.py
│       │
│       ├── pipeline/
│       │   ├── __init__.py
│       │   ├── analyze.py
│       │   ├── run_state.py
│       │   └── benchmark.py
│       │
│       └── export.py
│
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── regression/
│   └── fixtures/
│
├── benchmarks/
│   └── README.md
│
├── docs/
│   ├── architecture.md
│   ├── user-guide.md
│   ├── measurement-dictionary.md
│   ├── segmentation-validation.md
│   ├── qc-protocol.md
│   └── decisions/
│
└── .github/
    └── workflows/
        └── ci.yml
```

The computational core must not import napari or Qt. GUI code belongs only under `qc/viewer.py`.

---

# 7. Engineering principles

## 7.1 Priorities, in order

1. Scientific correctness.
2. Reproducibility.
3. Segmentation validity across biological groups.
4. Performance.
5. Simplicity.
6. Convenience.
7. Generality.

Do not create abstractions for hypothetical future projects.

## 7.2 Boundary validation

Validate aggressively at system boundaries:

- Config files.
- Input paths.
- Manifest rows.
- Image dimensions.
- Channel names.
- scene indices.
- physical X/Y/Z spacing.
- device availability.
- segmentation model availability.
- output paths.

Once values are validated, internal functions should use typed structures instead of repeating defensive checks.

## 7.3 Fail loudly

Examples of conditions that must fail with actionable messages:

- 3D analysis requested but Z spacing is unavailable.
- 3D analysis requested on a maximum-intensity projection.
- CZI contains multiple scenes but no scene is specified.
- Required Hoechst channel cannot be resolved.
- Production config requests CUDA but CUDA is unavailable.
- A manifest has duplicate `(image_id, channel)` entries.
- Multiple channels assigned to an image have incompatible dimensions or physical calibration.
- A configured texture scale converts to <1 pixel.
- Existing run outputs would be overwritten unintentionally.

Never silently assume `1/1/1` voxel spacing.

---

# 8. Core domain model

Use Pydantic models (or frozen typed dataclasses where more appropriate) for clear domain invariants.

At minimum implement concepts equivalent to:

```python
class PhysicalSpacing:
    z_um: float | None
    y_um: float
    x_um: float

class ExperimentalMetadata:
    image_id: str
    cell_line: str
    sort_id: str
    condition: str
    timepoint: str
    field: str
    acquisition_batch: str | None

class ImageSource:
    path: Path
    scene: str | int | None
    channel: str
    metadata: ExperimentalMetadata

class ImageVolume:
    data: np.ndarray
    spacing: PhysicalSpacing
    axes: str
    dtype: str

class SegmentationResult:
    labels: np.ndarray
    backend: str
    model_id: str
    backend_metadata: dict[str, Any]

class RunIdentity:
    run_id: str
    config_hash: str
    git_commit: str | None
```

Exact class names may differ, but the invariants must be explicit.

---

# 9. Experimental manifest

The application must not rely on filename parsing at analysis time.

Use a **long-form manifest** with one row per image channel:

```text
image_id
cell_line
sort_id
condition
timepoint
field
channel
path
scene
acquisition_batch
```

Recommended `image_id` example:

```text
SW620_Sort01_low_48h_Field003
```

Example rows:

```text
SW620_Sort01_low_48h_Field003,SW620,Sort01,low,48h,003,Hoechst,/data/...czi,0,
SW620_Sort01_low_48h_Field003,SW620,Sort01,low,48h,003,LaminAC,/data/...czi,0,
```

The representation must support both:

1. a single multichannel CZI/TIFF containing multiple channels; and
2. separate files per channel.

If one source file contains all channels, multiple manifest rows may reference the same path with different channel selectors.

## 9.1 Manifest CLI

Implement:

```bash
nuclear-morphometry manifest build <input-root> --output manifest.csv
nuclear-morphometry manifest validate <manifest>
```

`manifest build` may parse filenames using a configurable regex and/or inspect image channel metadata, but the generated manifest must always be validated before use.

Default filename patterns should recognize the thesis naming convention such as:

```text
SW620_Sort01_low_48h_Field003_Hoechst.tif
SW620_Sort01_low_48h_Field003_LaminA-C.tif
```

Do not make filename parsing the only supported workflow.

---

# 10. Microscopy I/O

## 10.1 Input formats

Support:

- Zeiss `.czi`
- TIFF / multi-page TIFF

Use BioIO as the primary abstraction.

## 10.2 Axes

Normalize internal arrays to:

- 2D: `YX`
- 3D: `ZYX`

Channel and time dimensions must be explicitly selected before returning an analysis image.

Do not carry ambiguous axis order downstream.

## 10.3 Scenes

If a file contains more than one scene:

- `inspect` should list available scenes.
- analysis must require the manifest to specify the scene unless one unambiguous mapping can be made.

## 10.4 Physical calibration

Read:

- X pixel size in µm/pixel.
- Y pixel size in µm/pixel.
- Z step in µm/slice.

For 3D mode, all three are required.

For 2D mode, X/Y are required for calibrated physical outputs.

Store both raw pixel/voxel measurements where useful and calibrated units, but final primary morphology columns must use physical units where calibration is available.

## 10.5 CLI inspection

Implement:

```bash
nuclear-morphometry inspect <path>
```

Print a concise structured summary:

- format
- scenes
- dimensions
- axis order
- channel names
- dtype / bit depth if available
- X/Y/Z physical spacing
- number of Z planes
- number of channels
- metadata warnings

Support `--json`.

---

# 11. Configuration

Use TOML with Pydantic validation.

Provide examples in `configs/`.

A representative 3D config should look conceptually like this:

```toml
[experiment]
name = "thesis"
manifest = "manifest.csv"
output_root = "results"

[analysis]
mode = "3d"

[input]
hoechst_channel = "Hoechst"
strict_physical_spacing = true

[segmentation]
backend = "cellpose"
device = "cuda"
model = "auto"
normalize_for_segmentation = true
percentile_low = 1.0
percentile_high = 99.8
diameter_um = 0.0
tile = true
batch_size = 0

[segmentation.3d]
use_anisotropy = true
flow3d_smooth = 0.0

[measurements]
morphology = true
intensity = true
texture_2d = false
radial_distribution_2d = false

[qc]
flag_border_objects = true
exclude_border_from_default_analysis = true
save_overlays = true
overlay_samples_per_group = 5
random_seed = 20260906

[performance]
prefetch_fields = 1

[output]
save_masks = true
write_csv = true
```

The exact Cellpose parameter names must follow the installed stable Cellpose API. Keep all Cellpose-specific handling inside `segmentation/cellpose_backend.py`.

## 11.1 `model = "auto"`

Do not interpret `auto` as a permanently hard-coded model.

Implement model selection as a configuration decision:

- If `model` is explicit, load that model.
- If `model = "auto"`, use the repository's documented default model chosen after segmentation validation.
- Until a real validation dataset has selected a default, fail with a clear message instructing the user to run the segmentation benchmark, or use a safe documented temporary development default only in example/demo mode.

Do not pretend a model is scientifically validated before it is.

---

# 12. 2D versus 3D semantics

Maintain both analyses in one codebase.

## 12.1 2D mode

2D mode exists for continuity with the preliminary dataset.

If input has `Z > 1`, do **not** silently perform a maximum-intensity projection.

Config must explicitly specify one of:

```text
projection = "none"
projection = "max"
projection = "mean"
projection = "specific_plane"
```

Default is `none`.

If `projection = "none"` and the input has multiple Z planes, fail with an actionable message.

Record the projection strategy in provenance.

## 12.2 3D mode

3D mode must operate on the intact `ZYX` volume.

Required:

- valid X spacing
- valid Y spacing
- valid Z spacing
- complete stack containing the entire nucleus
- no silent projection

---

# 13. Segmentation backend

## 13.1 Interface

Define a small backend protocol, approximately:

```python
class Segmenter(Protocol):
    def segment(
        self,
        image: np.ndarray,
        spacing: PhysicalSpacing,
    ) -> SegmentationResult:
        ...
```

Downstream measurements must consume the label image, not Cellpose internals.

## 13.2 Cellpose

Implement `CellposeSegmenter` as the first production backend.

Requirements:

- CUDA support.
- Explicit device selection.
- Model loaded **once per run**, not once per image.
- `torch.inference_mode()` or the equivalent inference-only execution.
- no gradient tracking.
- 2D and true-3D support.
- physical anisotropy passed to the model when the current Cellpose API supports it.
- record model identifier/version and model weight identity/checksum when feasible.
- save segmentation backend parameters in provenance.
- no silent CPU fallback in production.

If the current Cellpose API changes from this specification, adapt only inside the backend adapter and document the decision.

## 13.3 Segmentation-only normalization

Segmentation may use a normalized copy of the Hoechst image.

**Never use the segmentation-normalized image for primary intensity measurements.**

Default segmentation normalization:

- percentile clipping/normalization with configurable low/high percentiles.
- calculated per field unless configuration later specifies an experiment-wide strategy.

Record the normalization strategy and parameters.

Primary intensity measurements must use the original image data after only unavoidable format conversion.

## 13.4 No watershed requirement

Do not implement classical watershed merely to reproduce the old CellProfiler architecture.

A custom GPU watershed should be considered only if real validation shows that Cellpose cannot produce scientifically acceptable masks and classical seeded watershed is specifically required.

---

# 14. Segmentation validation

This is a scientific gate, not optional tooling.

Implement:

```bash
nuclear-morphometry validate-segmentation \
    --prediction predicted_labels.tif \
    --reference reference_labels.tif
```

and a batch form that consumes a validation manifest.

## 14.1 Validation set design

The documentation must instruct the user to create a fixed reference set stratified across:

- SW480 / SW620.
- low / high / bulk.
- independent sorts.
- sparse fields.
- dense/touching nuclei.
- irregular nuclei.
- elongated nuclei.
- dim nuclei.
- obvious difficult cases.

Reference masks may be manually corrected in napari.

## 14.2 Metrics

Compute at least:

- object count predicted/reference
- one-to-one object matching
- mean matched IoU
- median matched IoU
- detection precision at configurable IoU threshold
- detection recall
- F1
- unmatched prediction count
- unmatched reference count

Also estimate/report potential:

- splits
- merges

Use a transparent matching algorithm, e.g. IoU matrix + Hungarian assignment.

Do not obscure the matching logic behind a black-box metric package if the same can be implemented clearly.

## 14.3 Acceptance

Do not hard-code a universal IoU threshold as a scientific truth.

Provide a documented validation report and require manual visual review of representative overlays before declaring a model accepted.

Persist the accepted model/config in a versioned config or decision record.

---

# 15. Mask persistence

Save label masks for every analyzed field.

Use a microscopy-compatible label format:

- TIFF / BigTIFF via `tifffile` is acceptable for v1.
- preserve integer labels.
- preserve `YX` or `ZYX` axes.
- preserve physical spacing in metadata where the chosen writer supports it.

Do not compress masks if compression materially slows production by default. Make compression configurable.

Mask path should be deterministic from `image_id`.

Example:

```text
results/<run-id>/masks/SW620_Sort01_low_48h_Field003_labels.tif
```

Never reuse label IDs across separate fields as though they were globally unique.

The globally unique nucleus key is:

```text
(image_id, object_number)
```

---

# 16. 2D morphology

Implement and document these primary measurements per nucleus:

- `area_px`
- `area_um2`
- `perimeter_px`
- `perimeter_um`
- `circularity`
- `form_factor` (may alias circularity if definition is identical)
- `solidity`
- `eccentricity`
- `major_axis_um`
- `minor_axis_um`
- `aspect_ratio`
- `extent`
- centroid in pixels
- centroid in physical coordinates if useful

Use clearly documented formulas.

For circularity:

\[
C = \frac{4\pi A}{P^2}
\]

Document exactly how perimeter is estimated.

## 16.1 Legacy parity

Create a measurement comparison tool:

```bash
nuclear-morphometry compare-measurements \
    --ours nuclei.parquet \
    --reference legacy_cellprofiler.csv \
    --mapping configs/cellprofiler_mapping.toml
```

The key scientific technique is:

> Compare measurements on the **same masks** before comparing end-to-end segmentation pipelines.

This isolates measurement-definition differences.

If exact CellProfiler semantics are not known for a feature, do not claim exact parity. Document the difference and provide the closest validated definition.

---

# 17. 3D morphology

All primary 3D outputs must be physically calibrated.

## 17.1 Volume

For voxel count `N`:

\[
V = N \cdot \Delta x \cdot \Delta y \cdot \Delta z
\]

Output:

```text
volume_voxels
volume_um3
```

## 17.2 Surface area

For each object:

1. crop to its bounding box with a small safe margin;
2. create a binary mask for the object;
3. use marching cubes at the object boundary with physical spacing;
4. compute triangular mesh surface area.

Output:

```text
surface_area_um2
```

Avoid marching cubes over the full field independently for every object.

## 17.3 Z-depth / nuclear height proxy

Use the number of occupied Z planes or physical Z bounding-box extent, with a clearly documented voxel convention.

Output:

```text
z_depth_slices
z_depth_um
```

The physical definition must be explicit and tested.

## 17.4 Principal axis lengths

Compute a physically calibrated equivalent-ellipsoid or equivalent principal-axis representation.

Preferred implementation:

1. obtain object voxel-center coordinates in physical units;
2. compute covariance/inertia eigenvectors/eigenvalues;
3. derive three ordered equivalent ellipsoid axes:
   - major
   - intermediate
   - minor

For a uniform solid ellipsoid, covariance along an axis is \(a^2/5\), so an equivalent full axis length can be defined from an eigenvalue \(\lambda\) as:

\[
L = 2\sqrt{5\lambda}
\]

If the implementation uses another established convention, document it rigorously and test it on synthetic ellipsoids.

Output:

```text
axis_major_um
axis_intermediate_um
axis_minor_um
```

## 17.5 Extent

Compute:

\[
\text{extent} =
\frac{\text{object volume}}
{\text{physical bounding-box volume}}
\]

## 17.6 Sphericity

When volume and surface area are valid:

\[
\Psi =
\frac{\pi^{1/3}(6V)^{2/3}}{A}
\]

Output:

```text
sphericity
```

Do not emit misleading sphericity when surface reconstruction is invalid or the object is truncated.

## 17.7 3D synthetic tests

Create synthetic:

- sphere
- ellipsoid
- cuboid
- single-voxel edge cases
- anisotropic voxel examples

Tests must check measured values against analytical or high-resolution expected values within justified tolerances.

---

# 18. Intensity measurements

For Hoechst and applicable additional channels, compute per-nucleus:

- mean intensity
- median intensity
- integrated/sum intensity
- minimum intensity
- maximum intensity
- optional standard deviation

Primary thesis outputs should emphasize mean/median/integrated values.

Always measure on the original source intensity values, not segmentation-normalized arrays.

Store image dtype/bit depth and image-level saturation metrics so intensity comparisons can be audited.

---

# 19. Texture measurements

Texture is scientifically important but must remain constrained.

## 19.1 2D only by default

Implement nucleus-level 2D texture first.

Do not automatically port all 2D texture semantics into 3D.

3D texture must remain disabled by default until its biological interpretation and scale choices are explicitly validated.

## 19.2 Required primary outputs

At minimum support:

- contrast
- entropy

Also support a **small predefined optional subset** such as:

- homogeneity
- correlation
- energy/ASM

Keep the primary/secondary distinction explicit in `measurement-dictionary.md`.

## 19.3 Gray levels

Support configurable quantization, with default compatible with the prior workflow:

```text
gray_levels = 256
```

Document the quantization strategy exactly.

Do not normalize every nucleus independently in a way that destroys between-object intensity structure without explicit justification.

## 19.4 Texture scales

Support two modes:

### Legacy pixel mode

For reproducing the old analysis:

```text
distances_px = [3, 5, 10, 20]
```

### Physical-scale mode

For calibrated final analysis:

```text
distances_um = [...]
```

Convert physical distances to valid pixel offsets using X/Y calibration.

Fail or warn when a configured physical scale cannot be represented meaningfully at the acquisition resolution.

Record the scale mode in output column names or metadata.

---

# 20. 2D radial intensity distribution

Implement optional 2D radial bins within nuclei because the old workflow used 5 radial bins.

Default:

```text
radial_bins = 5
```

Do not enable a 3D radial-distribution equivalent by default.

If later implemented, it must have its own documented definition and validation.

---

# 21. Image-level QC

Compute metrics; do not invent universal failure thresholds.

At minimum store:

- image min/max
- mean intensity
- saturation fraction
- simple focus/blur metric
- object count
- fraction of image occupied by nuclei
- segmentation runtime
- analysis runtime

A focus metric may use variance of Laplacian or another clearly documented method.

Flags should be threshold-driven from configuration and default to **measurement-only** unless a threshold has been scientifically/technically validated.

---

# 22. Object-level QC

Every object row must preserve QC columns.

At minimum:

```text
qc_border
qc_manual_debris
qc_manual_merge
qc_manual_split
qc_manual_other
qc_excluded_default
qc_exclusion_reason
```

## 22.1 Border objects

Objects touching the image/volume boundary must be flagged.

The thesis expects border-truncated nuclei to be excluded from the default biological analysis, but the object and its measurements must remain in the raw per-object dataset.

Therefore:

- keep the row
- set `qc_border = true`
- set `qc_excluded_default = true`
- preserve reason `"border"`

## 22.2 Merge/split

Do **not** create aggressive automatic merge/split exclusion from shape statistics that might erase the phenotype.

Provide manual QC annotation as the reliable path.

Optional heuristics may produce:

```text
qc_probable_merge
qc_probable_split
```

but they must:

- be disabled by default unless validated;
- never delete rows;
- be documented as heuristics.

---

# 23. Interactive napari QC

Implement:

```bash
nuclear-morphometry qc <run-dir>
```

This must launch napari and load, for a selected field:

- raw Hoechst image
- segmentation labels
- optional additional channels
- QC overlay/annotations

Requirements:

- Z scrolling for 3D stacks.
- 3D rendering of labels where napari supports it.
- show/hide channels.
- click/select a nucleus and display its object number and key measurements.
- keyboard actions or a lightweight dock widget for manual tags:
  - good
  - debris
  - merge
  - split
  - other
- save annotations to the run directory.
- annotations must be keyed by `(image_id, object_number)`.
- existing annotations must reload.
- never modify the raw image or production mask in-place.

A full distributable napari plugin is **not** required. A programmatic viewer using napari APIs is preferred for v1.

---

# 24. QC sampling and reports

Implement:

```bash
nuclear-morphometry qc-report <run-dir>
```

Generate a static report containing:

- run identity
- counts by cell line / SortID / condition
- number and fraction of border objects
- manual QC counts if available
- image saturation/focus summaries
- segmentation runtime distribution
- representative segmentation overlays

Overlay selection must be **stratified** across:

- cell line
- condition
- SortID where possible

Use a fixed configurable random seed.

Do not show only the easiest or first fields.

Save report artifacts under:

```text
results/<run-id>/qc/
```

A simple self-contained HTML report plus PNG images is sufficient. Avoid a heavy report framework.

---

# 25. Additional channels

The nuclear mask must be defined from validated Hoechst segmentation and reused for other markers unless a documented scientific reason causes revalidation.

## 25.1 H3K9Ac

Measure within Hoechst-defined nuclei:

- mean
- median
- integrated intensity
- optional 2D radial distribution

Do not interpret this as a complete chromatin accessibility assay.

## 25.2 H3K9me3

If present, same general nuclear intensity/distribution measurements.

## 25.3 Lamin A/C

Support:

- total nuclear intensity
- configurable nuclear-periphery shell intensity
- core intensity
- periphery/core ratio

Define the shell in **physical units** where calibration is available.

For example, a configurable inner shell width in µm may be created by eroding the nuclear mask by the corresponding physical distance and subtracting the core.

Handle anisotropic 3D spacing correctly if 3D Lamin analysis is used.

## 25.4 MitoTracker

Do not treat MitoTracker intensity as mtDNA copy number.

Support a perinuclear spatial measurement based on the distance from the nuclear boundary.

A clean v1 implementation may provide:

- configured outside ring(s) around each nucleus in physical units
- mean mitochondrial intensity per ring
- perinuclear/farther-region enrichment ratio

When perinuclear regions from neighboring nuclei overlap, avoid double-counting pixels by assigning each pixel to its nearest nucleus or by using another explicitly documented disambiguation strategy.

Keep this module optional and isolated under `measurements/spatial.py`.

---

# 26. Output data model

## 26.1 Canonical per-nucleus table

Canonical output:

```text
results/<run-id>/nuclei.parquet
```

Each row is one nucleus.

Required identity/metadata columns:

```text
run_id
image_id
object_number
cell_line
sort_id
condition
timepoint
field
acquisition_batch
```

Required provenance/linkage columns may include:

```text
source_path
scene
mask_path
```

Then measurement and QC columns.

CSV export:

```text
results/<run-id>/nuclei.csv
```

is for interoperability only. Parquet is canonical.

## 26.2 Field-level table

Also save:

```text
results/<run-id>/fields.parquet
```

One row per analyzed field with:

- experimental metadata
- source file(s)
- physical spacing
- dimensions
- image QC
- object count
- runtime
- errors/warnings
- segmentation model identity

## 26.3 Analysis-ready table

Provide:

```bash
nuclear-morphometry prepare-analysis <run-dir>
```

Output:

```text
analysis_ready.parquet
```

This table must preserve:

```text
SortID
Field
image_id
object_number
```

and provide a documented default inclusion column such as:

```text
include_default
```

Do not aggregate away nuclei automatically.

Do not perform final biological inferential statistics in this command.

---

# 27. Run directory and provenance

Every run creates a new immutable-ish directory:

```text
results/
└── 2026-09-06T002300Z_ab12cd3/
    ├── config.toml
    ├── provenance.json
    ├── manifest.parquet
    ├── fields.parquet
    ├── nuclei.parquet
    ├── nuclei.csv
    ├── run_state.json
    ├── masks/
    ├── qc/
    └── logs/
```

## 27.1 `provenance.json`

Record at least:

- UTC start/end time
- Git commit
- whether repository was dirty
- config SHA-256
- manifest SHA-256
- Python version
- OS/platform
- package versions
- Torch version
- CUDA availability
- CUDA runtime/version information available through Torch
- GPU name
- GPU VRAM if available
- Cellpose version
- Cellpose model identifier
- model weight identity/checksum when practical
- segmentation parameters
- random seeds
- input file size/mtime metadata

Full SHA-256 hashing of very large raw files should be optional because it can be expensive.

Implement an optional finalization command:

```bash
nuclear-morphometry finalize-run <run-dir> --hash-inputs
```

to calculate full input hashes for archival/final-thesis runs.

---

# 28. Production CLI

Implement the following coherent CLI.

Exact option spelling may be adjusted for Typer ergonomics, but functionality must exist.

```bash
nuclear-morphometry doctor

nuclear-morphometry inspect <image> [--json]

nuclear-morphometry manifest build <input-root> --output manifest.csv
nuclear-morphometry manifest validate <manifest>

nuclear-morphometry run <config.toml>

nuclear-morphometry resume <run-dir>

nuclear-morphometry benchmark <config.toml> [--limit N]

nuclear-morphometry validate-segmentation \
    --prediction <mask> \
    --reference <mask>

nuclear-morphometry compare-measurements \
    --ours <parquet> \
    --reference <csv/parquet> \
    --mapping <toml>

nuclear-morphometry qc <run-dir>

nuclear-morphometry qc-report <run-dir>

nuclear-morphometry prepare-analysis <run-dir>

nuclear-morphometry export-csv <run-dir>

nuclear-morphometry finalize-run <run-dir> --hash-inputs
```

All commands must have useful `--help`.

---

# 29. `doctor` command

`nuclear-morphometry doctor` should report:

- Python version
- package version
- Torch version
- Cellpose version
- CUDA available yes/no
- selected CUDA device
- GPU name
- GPU VRAM
- whether BioIO CZI support can import
- whether napari can import
- whether the environment appears capable of:
  - CPU development
  - CUDA production
  - GUI QC

Return nonzero status only for conditions that prevent the explicitly requested mode.

Support:

```bash
nuclear-morphometry doctor --json
```

---

# 30. Pipeline execution

## 30.1 Field processing

Conceptual execution:

```text
validated manifest
      ↓
resolve one field
      ↓
load Hoechst only
      ↓
extract/validate spacing
      ↓
create segmentation-normalized image
      ↓
Cellpose on GPU
      ↓
save mask
      ↓
measure raw Hoechst + optional channels
      ↓
compute QC
      ↓
append/write field and object results atomically
```

Avoid loading all experiment images simultaneously.

## 30.2 Model lifetime

The segmentation model must be instantiated once per run.

Wrong:

```python
for field in fields:
    model = load_model()
    ...
```

Correct:

```python
model = load_model()

for field in fields:
    ...
```

## 30.3 GPU

Use one model owner for one GPU.

Do not start many independent processes that each load the model onto the same GPU.

Use Cellpose's internal tiling/batching where appropriate.

## 30.4 I/O prefetch

A small bounded prefetch queue is allowed.

Default:

```text
prefetch_fields = 1
```

Do not build a large concurrency framework unless benchmarking shows I/O is a meaningful bottleneck.

## 30.5 CPU measurement parallelism

Start simple.

Only add process/thread parallelism for measurements after profiling shows it matters and after verifying that memory copies do not erase the benefit.

---

# 31. CUDA OOM behavior

Production CUDA OOM must be handled deliberately.

Allowed behavior:

1. clear failed tensors/cache as appropriate;
2. if configured with auto batch size, retry with a smaller inference batch size that does **not** change image resolution or segmentation semantics;
3. log the retry;
4. fail the field if the safe minimum still OOMs.

Not allowed:

- silently use CPU;
- silently resize/downsample the image;
- silently change the model;
- silently change segmentation thresholds.

---

# 32. Performance benchmarking

Implement:

```bash
nuclear-morphometry benchmark configs/example_3d.toml --limit 5
```

Report per-stage timing:

```text
I/O
segmentation normalization
GPU segmentation
mask serialization
morphology
intensity
texture
QC
Parquet/output
total
```

Also report:

- peak process RSS if practical
- peak CUDA memory if available
- image dimensions
- object count

Save benchmark JSON/Parquet to allow before/after comparison.

## 32.1 Performance rules

- Profile before optimizing.
- Avoid unnecessary large array copies.
- Use `float32` for model input unless the model/API requires otherwise.
- Do not convert large arrays NumPy → Torch → NumPy repeatedly inside one stage.
- Never reload the model per image.
- Never sacrifice scientific image resolution merely for speed without explicit user authorization.
- Performance-sensitive PRs/changes need before/after evidence.

---

# 33. Resume and failure isolation

A long experiment must survive interruption.

Implement `run_state.json` or equivalent.

Each field should have states such as:

```text
pending
running
complete
failed
```

A field is marked `complete` only after its required outputs are atomically committed.

`resume` must:

- skip validated completed fields
- retry failed/incomplete fields
- refuse to resume if the config identity is incompatible unless explicitly overridden
- preserve previous error logs

Do not append duplicate nucleus rows when resuming.

---

# 34. Logging

Use Python logging with both:

- readable console output
- per-run log file

Log at least:

- field start/end
- source image
- dimensions
- physical spacing
- segmentation model/device
- object count
- timings
- warnings
- failures
- CUDA OOM retries
- output paths

Do not flood logs with per-pixel/per-object debug data in normal mode.

---

# 35. Scientific QC protocol encoded in the software/docs

The project documentation must explicitly instruct the user to verify before accepting a final run:

- raw Z-stacks are intact for 3D;
- physical X/Y pixel size and Z-step are known;
- images are not invalidated by saturation;
- segmentation overlays were reviewed across low/high/bulk;
- border-truncated nuclei are flagged/excluded from the default analysis;
- merged/split nuclei were reviewed/quantified;
- shape-based filters did not delete plausible biological extremes;
- sufficient nuclei exist per group/sort when feasible;
- every nucleus is traceable to CellLine, SortID, Condition, Timepoint, Field;
- final biological comparisons use at least 3 independent sorts;
- terminology remains SYBR-low/high until molecular validation justifies stronger claims.

---

# 36. Testing strategy

Testing is a core deliverable.

## 36.1 Unit tests

At minimum test:

- config parsing/validation
- manifest uniqueness and grouping
- physical-spacing validation
- dimension normalization
- filename parser
- 2D geometry formulas
- 3D geometry formulas
- intensity measurements
- texture quantization
- QC border detection
- analysis inclusion logic
- provenance hashing
- resume state transitions

## 36.2 Synthetic image fixtures

Create small synthetic arrays, not large binary fixtures, for ordinary unit tests.

Examples:

- circle
- ellipse
- rectangle
- sphere
- ellipsoid
- cuboid
- anisotropic sphere-like volume
- objects touching boundary
- two labels
- saturated image
- known intensity gradients

## 36.3 Integration tests

Include CPU-only integration tests that:

- build a tiny synthetic manifest
- run the pipeline using a deterministic dummy segmentation backend
- write masks
- write Parquet
- resume successfully
- generate an analysis-ready table

The segmentation interface should make a deterministic `FixtureSegmenter` or similar test backend easy.

## 36.4 Cellpose integration tests

Mark GPU/model-download tests separately.

They should skip cleanly if:

- CUDA unavailable
- model weights unavailable
- test explicitly not enabled

Do not make ordinary CI download large model weights.

## 36.5 Regression tests

The repository should have a place for small accepted real-data/reference-mask fixtures.

Do not commit sensitive or huge microscopy data by default.

Regression validation may instead use a documented external test-data directory.

## 36.6 Scientific-output changes

Never update expected regression outputs merely because a refactor changed them.

When a scientific output changes, create a decision record explaining:

- what changed
- why
- which measurements/masks changed
- validation evidence
- whether previous results must be regenerated

---

# 37. Static analysis and style

Use:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

Configure Ruff rather than layering Flake8/Black/isort separately.

Use type hints for public/internal boundaries.

Avoid `Any` in the scientific core unless unavoidable at third-party API boundaries.

Use `numpy.typing.NDArray` or equivalent aliases for array contracts.

---

# 38. CI

Create GitHub Actions CI that runs on Linux with CPU-only dependencies where possible.

CI must run:

- Ruff check
- Ruff format check
- Pyright
- unit tests
- CPU integration tests

GPU tests must not be required for normal GitHub-hosted CI.

If Cellpose installation makes CPU CI excessively heavy, split extras cleanly while still testing the backend adapter through mocks/fixtures.

---

# 39. Dependency extras

Use sensible optional dependency groups if they reduce installation cost.

For example, conceptually:

```text
core
gpu
gui
dev
```

But do not over-fragment.

A practical install should be documented, e.g.:

```bash
uv sync --extra gpu --extra gui
```

The exact arrangement should reflect current package compatibility.

---

# 40. README

The README must be usable by a student who did not write the code.

Include:

1. What the project does.
2. Scientific scope and non-goals.
3. Installation on Linux.
4. CUDA prerequisites.
5. `nuclear-morphometry doctor`.
6. How to inspect one CZI.
7. How to create/validate a manifest.
8. How to run 2D.
9. How to run 3D.
10. How to resume.
11. How to open QC.
12. How to export tables.
13. Output directory explanation.
14. Reproducibility/provenance explanation.
15. Scientific warnings about biological replicate and phenotype-biased filtering.
16. Link to measurement dictionary.
17. Link to segmentation validation protocol.

Do not turn the README into an implementation dump.

---

# 41. Measurement dictionary

`docs/measurement-dictionary.md` is mandatory.

For every exported scientific feature include:

- exact column name
- dimensionality
- units
- mathematical definition
- source channel
- biological rationale
- primary vs secondary status
- known caveats

Example:

```text
area_um2
- dimensionality: 2D
- units: µm²
- definition: object pixel count × pixel_size_x × pixel_size_y
- source: Hoechst segmentation mask
- role: primary
- rationale: tests preliminary nuclear-size phenotype
```

This document is part of the thesis reproducibility record.

---

# 42. Decision records

Create short Markdown ADR-style files in:

```text
docs/decisions/
```

At minimum create:

```text
0001-cellpose-segmentation-backend.md
0002-no-phenotype-based-qc-filtering.md
0003-physical-units-for-3d.md
0004-parquet-as-canonical-table.md
```

Add later records for meaningful deviations discovered during implementation.

---

# 43. AGENTS.md to generate

Create `AGENTS.md` with the following intent.

It may be edited for clarity, but do not weaken these rules:

```markdown
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
- Preserve CellLine, SortID, Condition, Timepoint, Field, and ObjectNumber for every nucleus.
- SortID is the biological replicate. Nuclei are nested observations.
- Never exclude nuclei because they are biologically irregular, elongated, eccentric, low-solidity, or low-circularity.
- QC may remove/flag technical failures such as border truncation, debris, segmentation failure, saturation, or focus failure.
- Preserve rejected objects and the reason for rejection.
- Do not change validated Hoechst segmentation when adding other channels without explicit revalidation.
- Do not silently change a measurement's mathematical definition.

## Architecture

Keep separate:
- microscopy I/O and metadata
- segmentation
- measurements
- QC
- pipeline execution
- export/provenance
- GUI

The computational core must not depend on napari or Qt.

## GPU/performance

- Load the model once per run.
- Use CUDA when requested.
- Use inference mode.
- Avoid unnecessary CPU↔GPU transfers.
- Do not add CuPy, cuCIM, Dask, multiprocessing, or distributed execution without profiling evidence.
- Bound memory use.
- Do not optimize by silently downsampling or changing scientific semantics.
- Performance-sensitive changes require benchmark evidence.

## Testing

Every scientific measurement requires analytical or independently verified tests.
Maintain unit, integration, regression, and benchmark coverage.
A segmentation backend/model change requires rerunning the fixed validation set.

Never change expected scientific outputs merely to make a failing test pass.

## Outputs

Canonical tables are Parquet.
Each run preserves config, manifest, masks, measurements, QC, and provenance.

## Errors

Validate at boundaries and fail early with actionable errors.
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
```

---

# 44. CLAUDE.md to generate

Create:

```markdown
# Claude Code instructions

Read and follow `AGENTS.md` as the canonical repository instructions.

Before scientific changes, also read:
- `docs/measurement-dictionary.md`
- relevant files under `docs/decisions/`

Do not duplicate project rules here; `AGENTS.md` is the source of truth.

For substantial work:
1. inspect the existing implementation/tests;
2. state the short implementation plan;
3. implement;
4. run relevant tests, Ruff, and Pyright;
5. summarize scientific/output changes explicitly.
```

---

# 45. Default-exclusion semantics

The raw `nuclei.parquet` must contain **all successfully measured segmented objects**.

Use fields such as:

```text
include_default
qc_excluded_default
qc_exclusion_reason
```

Example logic:

- border object → keep row, `include_default=false`, reason=`border`
- manual debris → keep row, `include_default=false`, reason=`manual_debris`
- manual merge → keep row, `include_default=false`, reason=`manual_merge`
- high eccentricity only → **do not exclude**
- low solidity only → **do not exclude**
- low circularity only → **do not exclude**

This is central to the thesis.

---

# 46. Image/channel consistency

When multiple channels belong to the same field, validate:

- Y/X dimensions agree.
- Z dimensions agree when used in 3D.
- physical spacing agrees within a tiny tolerance.
- scene mapping is consistent.
- no accidental mixing of acquisition fields.

If channel alignment differs and registration is required, **do not silently resample**.

Fail and document that image registration is required.

Image registration is outside v1 unless the real dataset demonstrates a need.

---

# 47. Physical units and naming

Use SI-derived microscopy-friendly units:

- length: `um`
- area: `um2`
- volume: `um3`

Column names should carry units where appropriate.

Do not use Unicode µ in machine-facing column names; use ASCII `um`.

Human-facing docs may use `µm`.

Examples:

```text
area_um2
perimeter_um
volume_um3
surface_area_um2
z_depth_um
axis_major_um
```

---

# 48. Scientific result immutability

Once a final segmentation configuration has been accepted for a thesis dataset:

- preserve the exact config;
- preserve model identity;
- preserve generated masks;
- perform statistics from preserved masks/measurements;
- do not opportunistically rerun segmentation with a newer library/model and mix outputs.

If the model/backend changes, create a new run ID and revalidate.

---

# 49. Error output and user experience

Errors should say what the user can do next.

Bad:

```text
ValueError: spacing is None
```

Good:

```text
3D analysis requires physical Z spacing, but no Z step was found for
SW620_Sort01_low_48h_Field003.

Inspect the source with:
  nuclear-morphometry inspect <path>

Then either fix the source/manifest metadata or use an explicitly calibrated
value supported by the acquisition record. The pipeline will not assume Z=1.
```

---

# 50. Security / data safety

- Never modify raw input microscopy files.
- Never write into the source data directory unless explicitly configured.
- Refuse destructive overwrite of an existing run.
- Use atomic temp-file + rename patterns for canonical Parquet/JSON outputs where practical.
- Sanitize only application-generated filenames; preserve original source paths in metadata.
- Do not send microscopy data or metadata to remote services.
- No telemetry.

---

# 51. Out of scope for the first complete build

Do not implement these unless they become necessary from real-data validation:

- distributed cluster execution
- multi-GPU scheduling
- deep-learning model training UI
- generic workflow graph editor
- browser UI
- database-backed experiment manager
- image registration
- deconvolution
- super-resolution
- custom GPU watershed
- automatic biological statistical conclusions
- qPCR analysis
- AFM analysis

Interfaces may be clean enough to extend later, but do not add placeholder frameworks.

---

# 52. Implementation order

Implement in this order, but continue through all phases.

## Phase 1 — Bootstrap and I/O

- uv project
- package structure
- CLI
- config
- `doctor`
- `inspect`
- BioIO CZI/TIFF reading
- physical spacing
- manifest build/validation
- unit tests

## Phase 2 — Pipeline skeleton and deterministic test segmenter

- run directory
- provenance
- run state/resume
- deterministic fixture segmenter
- mask I/O
- Parquet outputs
- end-to-end synthetic integration test

This phase must prove the application architecture without requiring Cellpose.

## Phase 3 — Cellpose GPU backend

- backend adapter
- device validation
- model lifetime
- 2D
- 3D
- anisotropy support
- segmentation normalization
- timing
- CUDA OOM behavior
- model provenance

## Phase 4 — Segmentation validation tooling

- reference-mask comparison
- IoU assignment
- precision/recall/F1
- split/merge reporting
- validation report

## Phase 5 — 2D measurements

- morphology
- intensity
- texture
- radial distribution
- legacy measurement comparison
- measurement dictionary
- synthetic tests

## Phase 6 — True 3D

- volume
- surface area
- Z-depth
- three principal axes
- extent
- sphericity
- anisotropic synthetic tests

## Phase 7 — QC

- image-level QC
- object flags
- manual annotation persistence
- napari viewer
- static QC report
- stratified overlay sampling

## Phase 8 — Multi-channel

- H3K9Ac
- H3K9me3
- Lamin A/C shell/core
- MitoTracker perinuclear features
- channel consistency checks

## Phase 9 — Production hardening

- benchmark command
- resume/idempotency checks
- atomic writes
- final docs
- CI
- example configs
- comprehensive command help
- performance pass based on benchmarks only

---

# 53. Acceptance tests / Definition of Done

The project is complete only when all of the following are true.

## 53.1 Installation

From a clean clone:

```bash
uv sync --all-extras
```

or the documented equivalent succeeds on the supported Linux environment.

## 53.2 Quality gates

These pass:

```bash
uv run ruff check .
uv run ruff format --check .
uv run pyright
uv run pytest
```

## 53.3 CLI

All required commands exist and provide useful help.

## 53.4 Synthetic end-to-end

A test can:

1. generate a synthetic experiment;
2. create/validate a manifest;
3. run a deterministic segmentation backend;
4. compute 2D and/or 3D measurements;
5. save masks;
6. save field/nucleus Parquet;
7. generate QC artifacts;
8. resume without duplicate output.

## 53.5 Calibration

3D mode refuses missing Z spacing.

An anisotropic synthetic volume produces physically correct volume and axis measurements within documented tolerance.

## 53.6 QC semantics

An elongated/irregular synthetic nucleus is **not excluded** merely because of eccentricity/solidity/circularity.

A border-touching object remains in `nuclei.parquet` but is marked excluded by default for the explicit reason `border`.

## 53.7 Provenance

Every completed run has reproducible config/provenance/model/device metadata.

## 53.8 Cellpose

On a CUDA-capable machine with model weights available:

- `doctor` detects CUDA.
- the Cellpose backend runs on CUDA when requested.
- the model is not reloaded per field.
- 3D path uses physical anisotropy when supported by the current Cellpose API.
- production does not silently fall back to CPU.

## 53.9 QC viewer

`nuclear-morphometry qc <run-dir>` can open a field, show raw image + labels, and persist manual object annotations.

## 53.10 Documentation

README, architecture, measurement dictionary, segmentation validation, QC protocol, AGENTS.md, CLAUDE.md, and decision records are complete.

---

# 54. Real-data scientific validation still required after software completion

Software completion is not the same as final scientific validation.

After the application is built, the user must supply representative real microscopy images so the following can be validated:

1. Cellpose model/config selection.
2. Segmentation quality in SW480/SW620 and low/high/bulk.
3. Segmentation of irregular/elongated nuclei.
4. Split/merge rate.
5. Legacy 2D measurement parity with CellProfiler where required.
6. Exact texture scale choice.
7. Optional technical debris thresholds if any.
8. Whether channel registration is necessary.
9. Whether Lamin shell width and MitoTracker perinuclear ring widths are biologically sensible.
10. Whether final acquisitions use consistent intensity settings.

The software must make these validations easy; it must not fake them.

---

# 55. Expected final workflow

After implementation, a normal workflow should look like:

```bash
# 1. Verify environment
uv run nuclear-morphometry doctor

# 2. Inspect representative source
uv run nuclear-morphometry inspect /data/example.czi

# 3. Build and review manifest
uv run nuclear-morphometry manifest build /data/samples --output manifest.csv
uv run nuclear-morphometry manifest validate manifest.csv

# 4. Benchmark representative fields
uv run nuclear-morphometry benchmark configs/example_3d.toml --limit 5

# 5. Run full analysis
uv run nuclear-morphometry run configs/final_3d.toml

# 6. Review masks
uv run nuclear-morphometry qc results/<run-id>

# 7. Generate QC report
uv run nuclear-morphometry qc-report results/<run-id>

# 8. Prepare table for downstream statistics
uv run nuclear-morphometry prepare-analysis results/<run-id>

# 9. Optionally archive full source hashes for final thesis run
uv run nuclear-morphometry finalize-run results/<run-id> --hash-inputs
```

---

# 56. Expected performance architecture

The project should end up conceptually like this:

```text
                    RAW CZI / TIFF
                         │
                         ▼
                 BioIO metadata/I/O
                         │
            ┌────────────┴────────────┐
            │                         │
        raw Hoechst              optional channels
            │
            ▼
 segmentation-only normalization
            │
            ▼
       Cellpose / CUDA
            │
            ▼
       integer labels
            │
     ┌──────┴─────────────┐
     │                    │
     ▼                    ▼
measurements         napari / QC
(raw intensities)        │
     │                   │
     └──────────┬────────┘
                ▼
         Parquet + masks
                │
                ▼
      analysis-ready table
```

The expensive segmentation stage uses the GPU.

The application should not reproduce the old architecture in which a single CPU watershed monopolizes one thread for minutes while the GUI blocks.

---

# 57. Final implementation-agent checklist

Before declaring completion, verify explicitly:

- [ ] Full repository implemented, not scaffolding.
- [ ] `uv.lock` committed.
- [ ] `doctor` implemented.
- [ ] CZI/TIFF inspection implemented.
- [ ] Physical spacing enforced.
- [ ] Manifest workflow implemented.
- [ ] Deterministic test backend implemented.
- [ ] Cellpose backend implemented.
- [ ] CUDA production mode has no silent CPU fallback.
- [ ] 2D pipeline implemented.
- [ ] true-3D pipeline implemented.
- [ ] 2D measurement suite implemented.
- [ ] calibrated 3D suite implemented.
- [ ] texture suite constrained/documented.
- [ ] H3K9Ac/H3K9me3 optional support implemented.
- [ ] Lamin shell/core support implemented.
- [ ] MitoTracker perinuclear support implemented.
- [ ] QC flags preserve biological extremes.
- [ ] manual QC annotations implemented.
- [ ] napari viewer implemented.
- [ ] static QC report implemented.
- [ ] Parquet canonical outputs implemented.
- [ ] CSV interoperability export implemented.
- [ ] resume implemented without duplication.
- [ ] provenance implemented.
- [ ] benchmark command implemented.
- [ ] segmentation validation implemented.
- [ ] legacy measurement comparison implemented.
- [ ] README completed.
- [ ] measurement dictionary completed.
- [ ] QC protocol completed.
- [ ] architecture doc completed.
- [ ] AGENTS.md completed.
- [ ] CLAUDE.md completed.
- [ ] decision records completed.
- [ ] GitHub Actions CI implemented.
- [ ] Ruff passes.
- [ ] Ruff formatting check passes.
- [ ] Pyright passes.
- [ ] pytest passes.
- [ ] limitations/data-dependent scientific validation are documented.

---

# 58. Final instruction

Build this as **small, explicit, reproducible scientific software**, not as a framework.

The correct result is not the repository with the most abstractions or dependencies. The correct result is the repository that lets the researcher:

- load the actual microscopy data safely;
- segment nuclei quickly on the GPU;
- preserve abnormal nuclear morphology;
- measure 2D and calibrated 3D geometry correctly;
- audit every technical exclusion;
- visually verify segmentation;
- trace every nucleus to its biological replicate;
- reproduce the exact final thesis run months later.

If an implementation choice conflicts with that goal, choose the simpler scientifically defensible option and document it.
