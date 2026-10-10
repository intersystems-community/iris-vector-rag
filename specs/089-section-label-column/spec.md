# Feature Specification: Opt-in bitmap-indexed section label column

**Feature Branch**: `feature/ns-column`
**Created**: 2026-10-10
**Status**: Draft
**Ships in**: 0.16.0, after spec 088
**Input**: Phrase search filters sections by label with a LIKE over the JSON `metadata`.
A separate column with a bitmap index is up to 3x faster for selective labels
(`specs/088-note-search-first-slice/research-section-label-benchmark.md`). Make that
column available, opt in, without changing any result.

## What

- `ensure_section_column(vector_store) -> bool` (`iris_vector_rag.note_search.store`) adds
  `section_label VARCHAR(100)` to a chunk-shaped section table, creates the bitmap index
  `SectionLabelIdx` on it, and backfills every row from `metadata["section"]`. It is
  idempotent and returns whether it changed anything. A label over 100 characters raises
  `SectionStoreError` before the table is altered. Rows whose metadata does not parse are
  skipped and counted in the log.
- `SectionStore.store` writes the label into `section_label` in the same INSERT when the
  table has the column. Without the column it writes exactly what it wrote before.
- `IRISVectorStore.phrase_search(section=...)` filters on `section_label = '<label>'` when
  the table has the column, and on the metadata JSON otherwise. The parsed-metadata check
  that follows is unchanged.
- `section_filter_predicate(label, column=None, metadata_expr=None)`
  (`iris_vector_rag.storage.text_search`) builds either predicate. Quotes in the label are
  doubled; the column name is checked as an identifier.

## Why

The label stays in `metadata` as the source of truth. The column is a read-side index
that a table gains when a deployment chooses it, so existing tables keep working.

## Acceptance

1. On a metadata-only table, `ensure_section_column` backfills every labelled row and the
   bitmap index exists in `INFORMATION_SCHEMA.INDEXES`; a second call changes nothing.
2. A store after the column exists fills both the metadata and the column.
3. For every query in `tests/fixtures/note_search/vectors.json` that names a section,
   `phrase_search` returns the same parents, polarities and matched labels before and
   after the column exists.
4. A label containing an apostrophe is stored, filtered and found.
5. A table without the column works as before.
6. A label over 100 characters is refused before any change.

## Out of scope

Dropping the metadata label; indexing any label other than `section`; changing the
phrase or roll-up semantics.
