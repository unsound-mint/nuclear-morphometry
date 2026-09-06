# 0002 - No phenotype-based QC filtering

## Status

Accepted. This is the central scientific commitment of the whole project
(spec sections 1.3, 3, 22, 45).

## Context

The legacy CellProfiler workflow used morphology filters (area, solidity,
eccentricity, form factor/circularity) as QC gates. For this thesis that is
scientifically dangerous: solidity, eccentricity, circularity, and
irregularity are candidate **phenotypes** SYBR-low/SYBR-high nuclei may
differ on (spec section 1.1). A QC step that removes "too eccentric" or
"too irregular" nuclei can silently delete the exact biological signal the
thesis is testing for, and would do so asymmetrically across groups if the
groups differ in prevalence of extreme morphology.

## Decision

QC may only flag/exclude **technical** failures, never shape-based
biological extremes:

- Allowed automatic exclusion criteria: border truncation (spec 22.1).
- Allowed manual exclusion criteria (via napari annotation, not yet
  implemented): debris, merge, split, other technical artifacts.
- Explicitly disallowed as automatic exclusion criteria, alone or in
  combination: area, solidity, eccentricity, circularity/form factor,
  elongation, "irregularity" in general.

This is enforced structurally, not just by convention: `qc/flags.py`'s
`compute_object_qc` takes only `touches_border`, `flag_border_objects`, and
`exclude_border_from_default` as parameters. It has no shape-statistic
inputs at all, so a future change that tried to add shape-based exclusion
would have to change the function signature -- a visible, reviewable diff --
rather than slipping in as a new branch on existing inputs.
`tests/unit/test_qc_flags.py::test_compute_object_qc_has_no_shape_parameters`
asserts this signature directly, and
`tests/integration/test_pipeline_e2e.py` includes a synthetic elongated,
low-circularity nucleus and asserts it is *not* excluded.

Every excluded row is still kept in `nuclei.parquet` with its exclusion
reason (`qc_exclusion_reason`); nothing is ever deleted (spec 45).

## Consequences

- Any exclusion heuristic based on shape statistics that a future
  contributor wants to add (e.g. `qc_probable_merge`/`qc_probable_split`
  from spec 22.2) must be implemented as a separate, explicitly-named,
  default-disabled heuristic column -- never folded into
  `qc_excluded_default`/`compute_object_qc` -- and must be validated before
  being trusted, per spec 22.2.
- Researchers filtering `analysis_ready.parquet` by `include_default` are
  guaranteed that filter never silently removed a biologically extreme
  nucleus.
