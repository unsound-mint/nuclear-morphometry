# 0004 - Parquet as the canonical output format

## Status

Accepted.

## Context

Spec section 26.1 requires per-nucleus output in Parquet as canonical, with
CSV as an interoperability export only. Spec section 30.1/33 additionally
requires atomic per-field commits and resume without duplicate rows.

## Decision

- `nuclei.parquet` and `fields.parquet` are the canonical outputs.
  `nuclei.csv` is written from the same data purely for tools that can't
  read Parquet (spec 26.1) and is never read back by any pipeline code.
- Rather than one growing canonical file that a crash could corrupt
  mid-write, each field commits its own small Parquet file
  (`partial/nuclei/<image_id>.parquet`, `partial/fields/<image_id>.parquet`,
  `export.py`), written via temp-file-plus-rename (spec section 50). A
  retried field simply overwrites its own file -- there is no append path,
  so duplicate rows on resume are structurally impossible rather than
  merely tested against.
- `export.finalize_tables` rebuilds `nuclei.parquet`/`fields.parquet` by
  concatenating all committed partial files. It is idempotent and safe to
  call repeatedly (including implicitly, at the end of every `run` and
  `resume`), since it always reflects the current on-disk set of partial
  files rather than accumulating state itself.
- Every per-field nuclei frame is constructed against
  `schema.NUCLEI_TABLE_SCHEMA`, an explicit column/dtype schema, including
  when a field has zero segmented objects. This was not optional
  bookkeeping: an untyped empty `pl.DataFrame()` has zero columns, and
  concatenating that with a normal field's populated frame raises a schema
  error that would otherwise break `nuclei.parquet` for the *entire run*
  because one field found nothing (found during this build's own testing --
  see the regression test
  `test_field_with_zero_objects_does_not_break_the_run`). The same explicit
  schema also prevents a subtler version of the same failure: Polars infers
  a column's dtype from its data, so a field where every nucleus happens to
  share one `qc_exclusion_reason` value (e.g. all `None`) can infer a
  different dtype than a field with mixed values, which would also break
  the cross-field concat.

## Consequences

- Any new per-object measurement module (texture, additional channels, 3D
  morphology) must add its columns to `NUCLEI_TABLE_SCHEMA` with an
  explicit dtype, not rely on `pl.DataFrame` inference, or the same class of
  bug will resurface for whatever field happens to produce an
  all-null/all-same-value column first.
- `prepare-analysis` and any future analysis tooling should read
  `nuclei.parquet`, never the `partial/` directory directly -- the partial
  files are pipeline-internal resume state, not a supported read interface.
