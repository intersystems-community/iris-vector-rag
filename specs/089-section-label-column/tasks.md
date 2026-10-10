# Tasks: Opt-in bitmap-indexed section label column

Test-first: each item's tests exist and fail before its code. Integration runs against
the shared container `iris-vector-rag-iris:51972` in schema `NSCOL`, one run at a time.

## Phase 1: pure helpers (unit tests first)

- [x] T047a `tests/unit/test_section_filter_predicate.py`, then `section_filter_predicate`
      and `json_member_predicate` in `storage/text_search.py`; the store's metadata
      predicate delegates to them.
- [x] T047b `tests/unit/test_section_store_column.py` (fake cursor and shape): the label
      is written only when the column exists, the 100-character refusal happens before
      any write or ALTER, unparsable metadata is skipped, a complete table changes
      nothing; then `SECTION_COLUMN`, `SectionStore.store` and `ensure_section_column`.

## Phase 2: phrase search

- [x] T047c `phrase_search(section=...)` uses the column predicate when the shape has
      `section_label`, and the metadata predicate otherwise.

## Phase 3: integration on real IRIS

- [x] T047d `tests/integration/test_note_search_column_iris.py`: backfill on a
      metadata-only table, bitmap index, idempotence, a long label refused before ALTER.
- [x] T047e Store after the column exists fills both metadata and column; a long label
      is refused by store with nothing written.
- [x] T047f Same phrase_search answers before and after the column for every sectioned
      query in the fixture; an unsectioned search is unchanged; a table without the
      column still filters by section.
- [x] T047g A label with an apostrophe is stored, filtered and found.

## Phase 4: documentation and gates

- [x] T047h `CHANGELOG.md`: bullet under Unreleased, note search.
- [x] T047i Run the note-search unit, contract, integration and e2e tests.
