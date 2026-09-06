# Measurement dictionary

This document is part of the thesis reproducibility record (spec section 41). For every
exported scientific feature: exact column name, dimensionality, units, mathematical
definition, source channel, biological rationale, primary/secondary status, and known
caveats.

Only measurements that are actually implemented and tested are listed here. Do not treat
this file as a preview of the full spec — see the "Not yet implemented" section at the
bottom for what `Dayana_Nuclei_Complete_Build_Spec.md` still calls for.

---

## 2D morphology (`measurements/morphology_2d.py`, spec section 16)

Source channel: Hoechst-defined nuclear segmentation mask. Role: primary unless noted.

### `area_px`
- Dimensionality: 2D. Units: pixels².
- Definition: count of pixels in the labeled object (`skimage.measure.regionprops` `area`).
- Rationale: raw measurement underlying `area_um2`; also the legacy CellProfiler unit for
  direct comparison via `compare-measurements` (not yet implemented).
- Role: secondary (audit/parity value; `area_um2` is primary).

### `area_um2`
- Dimensionality: 2D. Units: µm².
- Definition: `area_px * pixel_size_x_um * pixel_size_y_um`. Exact regardless of pixel
  anisotropy (no isotropic assumption needed for area).
- Rationale: tests the preliminary 2D nuclear-size phenotype (spec section 2: SYBR-low
  nuclei ~10-15% larger in 2D).
- Caveats: requires real X/Y physical calibration (`PhysicalSpacing`); the pipeline refuses
  to compute this with fabricated/uncalibrated spacing (see
  `docs/decisions/0006-tiff-uncalibrated-resolution-detection.md`).

### `perimeter_px` / `perimeter_um`
- Dimensionality: 2D. Units: pixels / µm.
- Definition: `skimage.measure.regionprops` `perimeter`, the Vossepoel & Smeulders
  pixel-boundary estimator (a weighted count of boundary pixel transitions), **not**
  sub-pixel contour tracing. `perimeter_um = perimeter_px * sqrt(pixel_size_x_um *
  pixel_size_y_um)` (geometric mean of X/Y pixel size — see caveat below).
- Rationale: input to `circularity`; matches the legacy CellProfiler perimeter style for
  continuity (spec section 2).
- Caveats: the geometric-mean conversion to µm is exact only for square pixels; for
  anisotropic X/Y calibration it is a documented approximation, since the underlying pixel
  estimator assumes an isotropic grid.

### `circularity` / `form_factor`
- Dimensionality: 2D (dimensionless).
- Definition: `4 * pi * area_px / perimeter_px**2`. `form_factor` is an alias of
  `circularity` (identical definition, spec section 16 permits aliasing).
- Rationale: tests the preliminary circularity/irregularity phenotype (spec section 2:
  SYBR-low nuclei less circular).
- Caveats: **never used for automatic QC exclusion** — see
  `docs/decisions/0002-no-phenotype-based-qc-filtering.md`. On a rasterized circle at
  typical nuclear radii this estimator reads ~0.85-0.95, not exactly 1.0, due to the
  pixel-perimeter estimator's known bias — this is a property of the estimator, not a bug.

### `solidity`
- Dimensionality: 2D (dimensionless, 0-1).
- Definition: `area / convex_hull_area` (`regionprops` `solidity`).
- Rationale: candidate irregularity phenotype (spec section 1.1).
- Caveats: never used for automatic QC exclusion.

### `eccentricity`
- Dimensionality: 2D (dimensionless, 0-1; 0 = circle, →1 = line segment).
- Definition: eccentricity of the ellipse with the same second central moments as the
  object (`regionprops` `eccentricity`).
- Rationale: candidate elongation phenotype (spec section 1.1).
- Caveats: never used for automatic QC exclusion — this is the primary shape statistic
  named explicitly in spec sections 3, 22, 45 as forbidden from driving exclusion.

### `major_axis_um` / `minor_axis_um`
- Dimensionality: 2D. Units: µm.
- Definition: `regionprops` `axis_major_length` / `axis_minor_length` (full length of the
  major/minor axis of the equivalent ellipse, in pixels), converted to µm via
  `sqrt(pixel_size_x_um * pixel_size_y_um)` — same isotropic-grid caveat as `perimeter_um`.
- Rationale: elongation/size phenotype support.

### `aspect_ratio`
- Dimensionality: 2D (dimensionless).
- Definition: `major_axis_length_px / minor_axis_length_px`. `inf` when
  `minor_axis_length_px == 0` (degenerate single-pixel-wide object) — not fabricated to a
  large finite number.
- Rationale: elongation phenotype support, more directly interpretable than eccentricity
  for some readers.

### `extent`
- Dimensionality: 2D (dimensionless, 0-1).
- Definition: `area / bounding_box_area` (`regionprops` `extent`).
- Rationale: secondary shape descriptor; distinguishes "fills its bounding box" (e.g. a
  square/round object) from "sparse within its bounding box" (e.g. an L-shape or an object
  with a thin protrusion).
- Role: secondary.

### `centroid_row_px` / `centroid_col_px`
- Dimensionality: 2D. Units: pixels.
- Definition: object centroid in array (row, column) = (Y, X) pixel coordinates.
- Role: secondary (linkage/QC-viewer use, e.g. locating an object for manual review).

### `centroid_y_um` / `centroid_x_um`
- Dimensionality: 2D. Units: µm.
- Definition: `centroid_row_px * pixel_size_y_um`, `centroid_col_px * pixel_size_x_um` —
  exact per-axis conversion, no isotropic assumption needed.
- Role: secondary.

### `touches_border`
- Dimensionality: 2D (boolean).
- Definition: `True` if the object's bounding box touches row 0, column 0, the last row, or
  the last column of the field.
- Rationale: feeds `qc_border`/`qc_excluded_default` (see QC section below); not itself a
  biological measurement.
- Role: secondary (QC input).

---

## Intensity (`measurements/intensity.py`, spec section 18)

Source channel: whichever channel is passed in — currently wired for Hoechst only; the
function itself is channel-agnostic (spec section 25's additional channels will reuse it
once wired). Computed **only** on the original source intensity image, never the
segmentation-normalized copy (spec 13.3) — this is enforced by the parameter name
(`intensity_image`) and stated in the function's docstring, not by a runtime check, since a
runtime check cannot distinguish "normalized" from "raw" arrays by inspection.

### `mean_intensity`, `median_intensity`, `min_intensity`, `max_intensity`, `std_intensity`
- Dimensionality: 2D or 3D (dimension-agnostic). Units: raw detector counts (dtype-dependent,
  e.g. `uint16` counts) — not further calibrated.
- Definition: `skimage.measure.regionprops` `intensity_mean` / `intensity_median` /
  `intensity_min` / `intensity_max` / `intensity_std`, restricted to the object's mask.
- Rationale: Hoechst intensity is a named biological question (spec section 1.1).
- Role: primary (mean/median/integrated per spec section 18); min/max/std secondary.

### `integrated_intensity`
- Dimensionality: 2D or 3D. Units: raw detector counts, summed.
- Definition: sum of intensity over all pixels/voxels in the object
  (`prop.image_intensity.sum()`, where `regionprops` has already zeroed non-object pixels
  within the bounding-box crop).
- Rationale: named primary intensity output (spec section 18).
- Caveats: not comparable across acquisitions with different exposure/gain settings unless
  those are held constant or explicitly normalized for — this pipeline does not currently
  perform any cross-acquisition intensity normalization (spec section 54 lists "whether
  final acquisitions use consistent intensity settings" as a real-data validation item still
  required).

---

## 2D texture (`measurements/texture.py`, spec section 19)

Source channel: Hoechst (currently); 2D only by contract (spec 19.1). Column names encode
the property and the pixel distance used: `{property}_d{distance_px}`, e.g. `contrast_d3`,
`entropy_d10`. The distance list is configuration-driven
(`measurements.texture_distances_px`, default `[3, 5, 10, 20]`, or
`measurements.texture_distances_um` converted via `um_distances_to_pixels`), so the exact
set of columns present depends on the run's config — this dictionary describes the property
definitions, not an exhaustive column list.

All properties are computed from a **masked** gray-level co-occurrence matrix (GLCM):
non-object pixels within an object's bounding box are excluded from every co-occurrence pair
(via a sentinel gray level that is discarded and the matrix renormalized before any property
is computed) — texture is never contaminated by background or neighboring objects. Each
property is computed at 4 standard angles (0°, 45°, 90°, 135°) and averaged, a standard
rotation-invariance convention.

Gray-level quantization: **per-object** min-max rescaling of the masked region's own
intensity range into `gray_levels` bins (default 256, `measurements.gray_levels`). This
means two objects with identical internal texture patterns but different absolute
brightness will produce identical texture columns — absolute brightness comparison across
objects is `measurements/intensity.py`'s job, not texture's. This is a deliberate,
documented tradeoff (spec 19.3 requires stating it explicitly), not an oversight.

### `contrast_d{n}`
- Dimensionality: 2D (dimensionless, ≥0). Role: primary.
- Definition: `skimage.feature.graycoprops(glcm, "contrast")`, i.e.
  `sum_{i,j} P(i,j) * (i-j)^2` over the masked, renormalized GLCM, averaged over angles.
- Rationale: named primary texture output (spec 19.2); candidate Hoechst
  texture/heterogeneity phenotype (spec section 1.1).

### `entropy_d{n}`
- Dimensionality: 2D (dimensionless, ≥0). Role: primary.
- Definition: `-sum(p * log2(p))` over nonzero entries of the same masked, renormalized GLCM
  used for `contrast`, averaged over angles. Not a `skimage.feature.graycoprops` property in
  the installed skimage version, so computed directly here — this is the auditable
  definition spec 19 requires.
- Rationale: named primary texture output (spec 19.2); candidate chromatin
  heterogeneity phenotype (spec section 1.1). Per spec section 1.2: Hoechst texture
  differences alone do not prove chromatin accessibility, transcriptional activity, or
  condensation state.

### `homogeneity_d{n}`
- Dimensionality: 2D (dimensionless, 0-1). Role: secondary.
- Definition: `graycoprops(glcm, "homogeneity")` — `sum_{i,j} P(i,j) / (1 + (i-j)^2)`.

### `correlation_d{n}`
- Dimensionality: 2D (dimensionless, typically -1 to 1). Role: secondary.
- Definition: `graycoprops(glcm, "correlation")`.
- Caveats: for an object with too few valid same-distance pixel pairs to form a non-trivial
  GLCM (e.g. a very small or very thin object), this and other properties are reported as
  `0.0` rather than `NaN` — there is no texture signal to measure at that scale for that
  object, and this is stated explicitly rather than silently propagating a `NaN` into
  `nuclei.parquet`.

### `energy_d{n}`
- Dimensionality: 2D (dimensionless, 0-1). Role: secondary.
- Definition: `graycoprops(glcm, "energy")` (square root of the angular second moment,
  `sqrt(sum P(i,j)^2)`). `1.0` for a perfectly uniform-intensity object (all GLCM mass in one
  cell); lower for more heterogeneous texture.

---

## 3D morphology (`measurements/morphology_3d.py`, spec section 17)

Source channel: Hoechst-defined nuclear segmentation mask, intact `ZYX` volume (never a
projection). Role: primary unless noted. Requires real X/Y/Z physical calibration —
see `docs/decisions/0003-physical-units-for-3d.md`.

### `volume_voxels` / `volume_um3`
- Dimensionality: 3D. Units: voxels / µm³.
- Definition: `volume_um3 = voxel_count * pixel_size_x_um * pixel_size_y_um * z_step_um`.
  Exact — no reconstruction or smoothing involved.
- Rationale: named primary 3D size phenotype (spec sections 1.1, 4).

### `surface_area_um2`
- Dimensionality: 3D. Units: µm². `None` when reconstruction is invalid (see caveats).
- Definition: marching cubes (`skimage.measure.marching_cubes` + `mesh_surface_area`) on
  each object's own cropped, zero-padded binary mask after Gaussian smoothing
  (`sigma=1` voxel) — never one marching-cubes pass over the full field. See
  `docs/decisions/0007-marching-cubes-surface-area-smoothing.md` for why the smoothing step
  exists: unsmoothed marching cubes on a hard voxel boundary overestimates surface area by
  ~8-9% at nuclear scale, verified against an analytical sphere.
- Caveats: `None` for border-touching objects (true surface is unknown for a truncated
  object) and for objects too small to reconstruct a meaningful isosurface (smaller than
  roughly 3×3×3 voxels at the default smoothing). The smoothing that corrects the sphere
  bias understates surface area for sharp-edged/angular objects — accepted deliberately
  since real nuclei are much closer to ellipsoids than to polyhedra.

### `z_depth_slices` / `z_depth_um`
- Dimensionality: 3D. Units: Z-planes / µm.
- Definition: count of distinct Z-indices occupied by any voxel of the object (not the
  bounding-box Z-span, so a rare non-contiguous-in-Z object is not over-counted);
  `z_depth_um = z_depth_slices * z_step_um`.
- Rationale: named nuclear-height proxy (spec sections 1.1, 17.3).

### `axis_major_um` / `axis_intermediate_um` / `axis_minor_um`
- Dimensionality: 3D. Units: µm.
- Definition: equivalent uniform-solid-ellipsoid axis lengths, derived from the eigenvalues
  of the covariance matrix of the object's voxel-center coordinates in physical units
  (population covariance, `ddof=0`, matching the uniform-solid derivation): for eigenvalue
  `λ`, axis length `L = 2 * sqrt(5 * λ)` (spec section 17.4's given convention, since for a
  uniform solid ellipsoid the covariance along an axis is exactly `a²/5`).
- Rationale: 3D elongation/size phenotype (spec section 1.1). Validated against both a
  sphere (all three axes equal) and a known-axis-length synthetic ellipsoid.

### `extent`
- Dimensionality: 3D (dimensionless, 0-1). Role: secondary.
- Definition: `volume_um3 / physical_bounding_box_volume_um3` (spec section 17.5).

### `sphericity`
- Dimensionality: 3D (dimensionless, ≤1 for a well-formed reconstruction). `None` when
  `surface_area_um2` is `None`.
- Definition: `pi^(1/3) * (6 * volume_um3)^(2/3) / surface_area_um2` (spec section 17.6).
- Rationale: named 3D shape phenotype (spec section 1.1).
- Caveats: never used for automatic QC exclusion (same rule as 2D circularity/eccentricity —
  see `docs/decisions/0002-no-phenotype-based-qc-filtering.md`); inherits every caveat of
  `surface_area_um2` above, since it's a direct function of it.

### `touches_border`
- Dimensionality: 3D (boolean). Definition: object's bounding box touches any face of the
  volume (Z, Y, or X). Role: secondary (QC input, same role as the 2D column).

---

## Object-level QC (`qc/flags.py`, spec sections 3.1, 22, 45)

### `qc_border`
- Dimensionality: n/a (boolean, per object).
- Definition: `touches_border AND qc.flag_border_objects` (config).
- Rationale: identifies technical field-of-view truncation, not a biological property.

### `qc_excluded_default`
- Dimensionality: n/a (boolean, per object).
- Definition: currently `True` only when `qc_border AND qc.exclude_border_from_default_analysis`
  (config). No shape statistic is or can be an input to this computation —
  `qc/flags.compute_object_qc`'s signature has no shape parameters at all (see
  `docs/decisions/0002-no-phenotype-based-qc-filtering.md`).
- Rationale: spec section 45's central default-exclusion semantics.
- Caveats: the row is always kept in `nuclei.parquet` regardless of this flag — nothing is
  ever deleted.

### `qc_exclusion_reason`
- Dimensionality: n/a (string or null, per object).
- Definition: `"border"` when `qc_excluded_default` is `True` for that reason; `None`
  otherwise, in the raw `nuclei.parquet` written during the pipeline run. Manual reasons
  (`manual_debris`, `manual_merge`, `manual_split`, `manual_other`, defined in `schema.py`)
  are never written into `nuclei.parquet` itself — they are stored separately (see manual
  QC annotations below) and only folded into a *derived* copy of the table by
  `qc/annotations.apply_annotations_to_nuclei`, matching spec 48's immutability
  requirement.
- Caveat: if an object is both border-excluded and manually tagged, the reason reported by
  `apply_annotations_to_nuclei` is `"border"` (the technical reason takes precedence) even
  though the corresponding `qc_manual_*` boolean is also set from the tag.

### `include_default` (analysis-ready table only, `prepare-analysis`)
- Definition: `not qc_excluded_default`, computed once by `schema.include_default_expr`. See
  `docs/decisions/0005-include-default-semantics.md` for why this is derived rather than
  stored redundantly on the raw table.

## Manual QC annotations (`qc/annotations.py`, spec sections 22.2, 23)

- Storage: `results/<run-id>/qc/annotations.json`, a list of `{image_id, object_number, tag,
  note, annotated_at}` records, keyed by `(image_id, object_number)`. One current tag per
  object (re-annotating overwrites, it is not an append-only history).
- `tag`: one of `good`, `debris`, `merge`, `split`, `other`. Maps to
  `qc_manual_debris`/`qc_manual_merge`/`qc_manual_split`/`qc_manual_other` and an
  `EXCLUSION_REASON_MANUAL_*` reason (or no exclusion, for `good`) only when folded into a
  nuclei table via `apply_annotations_to_nuclei` — never written back into
  `nuclei.parquet`.
- This module is the storage/merge layer only. The interactive `dayana-nuclei qc` viewer
  (napari, spec section 23) that produces these annotations is not yet implemented.

## Image-level QC (`qc/image_metrics.py`, spec section 21; columns live on `fields.parquet`)

Measurement-only, by design: no pass/fail threshold is invented anywhere in this module,
per spec 21 ("default to measurement-only unless a threshold has been
scientifically/technically validated"). All metrics are computed on the original
(non-segmentation-normalized) channel volume.

### `image_min_intensity` / `image_max_intensity` / `image_mean_intensity`
- Definition: min/max/mean pixel value over the whole field (all Z planes in 3D mode).

### `image_saturation_fraction`
- Definition: fraction of pixels equal to the integer dtype's maximum representable value
  (e.g. 65535 for `uint16`). `NaN` for a floating-point source image, which has no fixed
  sensor ceiling to compare against — reported as undefined rather than a fabricated 0 or 1.

### `image_focus_metric`
- Definition: variance of the discrete Laplacian of the image (`scipy.ndimage.laplace`), a
  standard focus/blur proxy (spec 21 explicitly permits this method). Higher means more
  high-frequency detail (sharper); no documented "in focus" cutoff.

### `image_occupied_fraction`
- Definition: fraction of pixels/voxels with a nonzero segmentation label.

### `segmentation_runtime_s` / `total_runtime_s`
- Definition: wall-clock seconds for the segmentation call and for the whole field
  (I/O + segmentation + measurement), respectively. Satisfies spec 21's "segmentation
  runtime" / "analysis runtime".

---

## Not yet implemented

The following spec-required measurements have no code yet and are not documented above
because there is nothing to audit: 2D radial intensity distribution (spec section 20), the
interactive napari QC viewer that produces manual annotations (spec section 23), and
additional-channel measurements (H3K9Ac, H3K9me3, Lamin A/C shell/core, MitoTracker
perinuclear — spec section 25). This section will be replaced by real entries as each is
implemented and tested, per this project's definition of done (`AGENTS.md`): documentation
must describe actual code, not planned code.
