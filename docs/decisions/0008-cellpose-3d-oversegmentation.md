# 0008 - Cellpose-SAM 3D mode: over-segmentation traced to a real normalization bug, partially resolved

## Status

**Root cause found and fixed for the class of input that matters (smoothly-varying
fluorescence signal). A residual open question remains for pathological/synthetic
inputs and is deferred to spec section 14 real-data validation.**

## Context (original finding, since partially superseded)

While verifying `segmentation/cellpose_backend.py`'s 3D path empirically on this
session's GPU, three synthetic 3D test volumes were segmented with
`CellposeSegmenter(model="cpsam_v2", do_3D=True)`, calling `segment()` directly and
**bypassing `segmentation.normalize.normalize_percentile`** (the pipeline-level
normalization step `pipeline/analyze.py::_process_field` normally applies before
calling `segmenter.segment()`):

1. A flat 4-plane binary "coin" (a disk repeated across a few Z-slices) — **0 objects**.
2. A hard-edged binary sphere (radius 15 px, isotropic 0.2 um spacing), values `{0.0,
   1.0}` — **25 objects** (one sphere fragmented into 25 pieces).
3. A Gaussian-smoothed, mildly-noised sphere (contrast 1000 vs. 100, sigma=2 blur,
   sigma=20 noise, i.e. realistic dynamic range, *not* renormalized to `[0,1]`) —
   **434 objects**.

This was initially recorded as an unresolved, likely genuine 3D flow-reconstruction
limitation of `cpsam_v2`.

## Root cause (found via advisor review + direct experiment)

`cellpose_backend.py`'s `_eval_with_oom_retry` calls `model.eval(..., normalize=False,
...)` unconditionally, with the comment "our own segmentation/normalize.py is
authoritative." That is only true if the caller actually ran
`normalize_percentile` first. The three tests above did not — they fed
`CellposeSegmenter.segment()` directly, so case 3's `{100..1000+}`-range float32
volume reached Cellpose-SAM's 3D flow network completely unnormalized. Cellpose-SAM's
flow reconstruction assumes roughly unit-range input; case 3 (434 objects) was also
the case furthest out of `[0,1]` range, which is exactly the pattern an
un-normalized-input hypothesis predicts.

**Experiment**: re-ran all three cases with `normalize_percentile(percentile_low=1.0,
percentile_high=99.8)` applied first (i.e. the same normalization the pipeline always
performs for `normalize_for_segmentation = true`), same model, same GPU:

| case | without normalization | with normalization |
|---|---|---|
| 1. flat coin (binary) | 0 objects | 0 objects (unchanged) |
| 2. hard binary sphere | 25 objects | 25 objects (unchanged) |
| 3. smoothed+noised sphere (realistic dynamic range) | 434 objects | **1 object** |

Case 3 — the one case with a realistic, non-binary dynamic range, i.e. the one that
actually resembles real Hoechst fluorescence signal — is fully fixed by
normalization. Cases 1 and 2 are unchanged because they were already binary
`{0.0, 1.0}` images: percentile normalization is a near no-op on a two-valued
distribution (the 1st and 99.8th percentiles already sit at 0 and 1 respectively), so
it cannot have been masking a normalization bug for those two cases in the first
place.

## A real production bug, now fixed at the config layer

Independent of the synthetic test results: `config.segmentation.normalize_for_segmentation
= false` combined with `backend = "cellpose"` had no protection at all. The pipeline
would hand `CellposeSegmenter.segment()` the raw, un-normalized channel volume
(typically uint16, range 0-65535), which then reached `model.eval(normalize=False,
...)` with no normalization at any layer — reproducing exactly the case-3 failure mode
on real data. `Config._check_cellpose_requires_normalization` (`config.py`) now
rejects this combination at config-validation time (fail loud, per the project's
no-silent-fallback invariant), rather than silently producing a badly over-segmented
3D result.

## Decision

1. The backend adapter is confirmed correct for driving Cellpose's 3D API (right
   z_axis, right anisotropy, right shape/dtype output). It has never applied its own
   normalization and is not expected to — pipeline-level normalization is the
   documented contract, and it is now enforced instead of merely assumed.
2. Realistic-signal 3D segmentation (smooth intensity gradient, moderate noise,
   percentile-normalized — i.e., what real confocal Hoechst data actually looks like)
   is no longer a documented open failure. It segmented correctly (1 object for 1
   object) once fed through the same normalization path the pipeline always applies.
3. Cases 1 and 2 (flat multi-plane disk; hard 0/1 binary sphere) remain unexplained
   fragmentation/no-detection failures, but both are pathological synthetic inputs
   that do not resemble real fluorescence signal (a real nucleus is not a flat disk
   spanning 4 z-planes, nor a perfectly hard-edged binary volume) — they are not
   evidence that Cellpose-SAM mishandles real 3D nuclei. They are left unexplained
   rather than further chased with synthetic data.
4. `tests/integration/test_cellpose_gpu.py`'s 3D test now normalizes its input before
   calling `segment()`, matching real pipeline usage, and continues to assert
   mechanics only (this file's stated scope) rather than a specific object count for
   the binary-sphere case, since that specific synthetic shape is known or-effective
   for normalization and its fragmentation is not fully understood.
5. Before any real 3D thesis run: build a reference mask set (spec 14.1) and run
   `validate-segmentation` (once implemented) specifically in 3D mode against real
   Hoechst-stained nuclei. Confidence is higher than before this investigation (the
   realistic-signal-shape case now works correctly), but real-data validation is
   still required before trusting any 3D thesis result — this ADR upgrades the risk
   assessment, it does not close it.

## Consequences

- 3D segmentation is no longer flagged as "severe over-segmentation, unresolved."
  It is flagged as "verified correct on one realistic synthetic case after fixing a
  real normalization-bypass bug; two pathological synthetic edge cases remain
  unexplained; real-data validation (spec 14) still required before production use."
- `segmentation.backend = "cellpose"` with `segmentation.normalize_for_segmentation =
  false` is now a hard config error instead of a silent correctness bug.
- The measurement/QC/pipeline code downstream of segmentation is unaffected by this
  finding and remains correct given whatever label image it receives.
