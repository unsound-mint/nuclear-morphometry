# 0008 - Cellpose-SAM 3D mode: severe over-segmentation observed, unresolved

## Status

**Accepted as a documented open scientific risk, not a resolved issue.** This is
exactly the kind of finding spec section 14 (segmentation validation) and
section 54 (real-data validation still required after software completion)
exist to catch. Do not run a real 3D thesis analysis with this backend until
it has been re-checked against real data via `validate-segmentation`.

## Context

While verifying `segmentation/cellpose_backend.py`'s 3D path empirically on
this session's GPU (see `docs/decisions/0001` for the general Cellpose 4.x
API investigation), three synthetic 3D test volumes were segmented with
`CellposeSegmenter(model="cpsam_v2", do_3D=True)`:

1. A flat 4-plane "coin" (a disk repeated across a few Z-slices, not a
   proper sphere) — result: **0 objects detected**.
2. A hard-edged binary sphere (radius 15 px, isotropic 0.2 um spacing) —
   result: **25 objects** (a single connected sphere fragmented into 25
   pieces).
3. A Gaussian-smoothed, mildly-noised sphere (same geometry, intensity
   contrast 1000 vs. 100 background, sigma=2 blur, sigma=20 Gaussian noise —
   deliberately made more realistic/less pathological than a hard binary
   edge) — result: **434 objects** (dramatically worse fragmentation, not
   better).

The 2D path was verified working correctly on equivalent synthetic input
(clean 2-object detection on two disks, spec-conformant shape/dtype,
`tests/integration/test_cellpose_gpu.py::test_cellpose_2d_inference_returns_integer_labels`
passes). `anisotropy=1.0` (spacing was isotropic in all three 3D tests, so
this parameter was effectively a no-op) and `z_axis=0` (the fix in this same
session, see the `cellpose_backend.py` history) were both confirmed correct
— the 3D call runs without error and returns a correctly-shaped label array.
The over-segmentation is not an obvious adapter bug (wrong axis order, wrong
spacing, wrong batch handling); it looks like either a genuine limitation of
`cpsam_v2`'s 3D flow reconstruction on this class of input, or a parameter
this investigation didn't find (candidates not yet tried: an explicit
non-`None` `diameter`, `stitch_threshold` instead of `do_3D` — segmenting
plane-by-plane and stitching in Z rather than native 3D flow reconstruction
— or a different `min_size`/`cellprob_threshold`/`flow_threshold`).

Time-boxing note: this investigation stopped after the three tests above
rather than continuing to sweep parameters, because doing so without a real
reference mask set is exactly the kind of validation spec section 14 is
designed to do properly (with F1/precision/recall against ground truth, not
eyeballing an object count) — further blind parameter tweaking here would
produce a false sense of confidence without a real answer.

## Decision

1. The backend adapter is accepted as-is: it correctly drives Cellpose's 3D
   API (this is what `cellpose_backend.py` is responsible for). Segmentation
   *quality* in 3D is explicitly out of the adapter's responsibility and is
   spec section 14's job.
2. `tests/integration/test_cellpose_gpu.py`'s 3D test asserts mechanics only
   (shape, dtype, at least one object, `do_3D=True` recorded in metadata) —
   it does not assert a correct object count, matching this file's existing
   stated philosophy for the 2D tests ("does not judge segmentation
   quality").
3. `AGENTS.md`'s "current state" section flags 3D segmentation as
   unvalidated and higher-risk than 2D, so a future session does not
   mistake "the pipeline runs in 3D mode" for "3D segmentation is correct."
4. Before any real 3D thesis run: build a reference mask set (spec 14.1) and
   run `validate-segmentation` (once implemented) specifically in 3D mode.
   If real Hoechst-stained nuclei also show severe fragmentation, the
   `stitch_threshold` (plane-by-plane 2D segmentation + Z-stitching)
   alternative should be tried and compared before concluding Cellpose-SAM
   is unsuitable for this dataset's 3D analysis (spec 13.4 permits
   reconsidering the backend only after real validation shows it's needed).

## Consequences

- 2D analysis is empirically verified end-to-end on this GPU and safe to
  treat as a working default. 3D analysis is not, despite running without
  error.
- The measurement/QC/pipeline code downstream of segmentation (morphology,
  intensity, texture, QC flags) is unaffected by this finding and remains
  correct given whatever label image it receives — this is purely a
  segmentation-model-quality question, not a pipeline-architecture one.
