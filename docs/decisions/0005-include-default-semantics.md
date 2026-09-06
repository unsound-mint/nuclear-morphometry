# 0005 - include_default vs qc_excluded_default

## Status

Accepted.

## Context

The spec names two related but not identically-scoped concepts:

- `qc_excluded_default` / `qc_exclusion_reason`, defined in spec section 22,
  as columns on the raw per-object QC record.
- `include_default`, defined in spec sections 26.3 and 45, as a column on
  the analysis-ready table produced by `prepare-analysis`.

The spec's worked examples in section 45 use `include_default` terminology
even when describing what is clearly the same underlying decision
(`border object -> keep row, include_default=false, reason="border"`).
Nothing in the spec states explicitly whether these are the same boolean
computed twice, computed once and copied, or genuinely different values at
different pipeline stages -- and if implemented inconsistently by
independently-working modules (or agents), that ambiguity is exactly the
kind of thing that surfaces late, as a confusing mismatch between
`nuclei.parquet` and `analysis_ready.parquet`.

## Decision

They are the same underlying decision, computed in exactly one place, at
two different times:

1. `qc_excluded_default` is computed authoritatively during the main
   pipeline run, in `qc/flags.compute_object_qc`, as each object is
   measured (spec 22). This is what `nuclei.parquet` stores. The raw table
   never stores `include_default` at all.
2. `include_default` is derived from it -- `include_default = not
   qc_excluded_default` -- only when `prepare-analysis` builds
   `analysis_ready.parquet` (spec 26.3). The inversion happens in exactly
   one function, `schema.compute_include_default` (scalar form, used in
   tests) and `schema.include_default_expr` (vectorized Polars expression,
   used by `export.prepare_analysis`). No other module re-derives this
   relationship.

Rationale for not storing `include_default` on the raw table too: the raw
`nuclei.parquet` is meant to be read as "everything that was measured, plus
why anything is flagged" -- an inverted boolean alongside the flag and
reason it's inverted from is redundant and creates a second place the two
could drift out of sync if either is ever hand-edited or reprocessed
independently.

## Consequences

- Any future exclusion criterion (manual debris/merge/split annotations,
  once QC viewer support lands) only needs to update
  `qc_excluded_default`/`qc_exclusion_reason`; `include_default` picks it up
  automatically the next time `prepare-analysis` runs, with no separate
  code path to keep in sync.
- `prepare-analysis` must be re-run after any QC annotation change for
  `analysis_ready.parquet` to reflect it; it is not automatically kept live.
