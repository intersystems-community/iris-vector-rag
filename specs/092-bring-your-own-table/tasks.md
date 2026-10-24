# Tasks: Bring Your Own Table

**Input**: Design documents from `/specs/092-bring-your-own-table/`
**Prerequisites**: plan.md, spec.md, research.md, data-model.md, contracts/store-api.md,
quickstart.md

**Tests**: mandatory and first in every phase. Unit tests come before the code they cover. Each
user story ends with an integration test on `iris-vector-rag-iris:51972` as its phase gate:

```bash
IRIS_HOST=localhost IRIS_PORT=51972 IRIS_NAMESPACE=USER IRIS_USERNAME=_SYSTEM IRIS_PASSWORD=SYS \
IRIS_BACKEND_MODE=community /Users/tdyar/ws/iris-vector-rag-private/.venv/bin/python -m pytest <tests>
```

The commands run from `/Users/tdyar/ws/ivr-language`. Commit with `SKIP=tdd-compliance`.

## Format: `[ID] [P?] [Story] Description`

## Phase 1: Setup

- [X] T001 Port the S3 spike's team fixtures into `tests/fixtures/byot/teams.py`:
  - MedTeam `EvidenceChunk` and Štafeta `DocChunk` DDL (from `spikes/s3/setup_tables.py`), in
    schemas `BYOTMed` and `BYOTStf`;
  - a seeded generator of synthetic rows: 4 patients, dates, 2 embedding models for MedTeam,
    unit and non-unit vectors of dim 384;
  - each team's own query, as SQL strings (from `spikes/s3/team_queries.py`);
  - `create(conn)` and `drop(conn)` helpers.
- [X] T002 [P] Add a `byot_tables` fixture (in `tests/integration/test_byot_iris.py` itself, with
  the recorded catalogs in `tests/fixtures/byot/catalog/`) that creates both tables before a test and drops them after. It also
  records a catalog and row-count snapshot, for SC-004.

## Phase 2: Foundational (blocks every story)

- [X] T003 [P] (FR-001, FR-002, FR-003) Unit tests for declared roles in `tests/unit/test_byot_shape.py`:
  - a declared column outranks `ROLE_PRECEDENCE`;
  - an undeclared role still resolves by name;
  - a declared column missing from the table raises `ShapeUnresolvable` naming it;
  - a composite `id_col` gives `identifier_columns` and `identifier_sql` (`"DocRef" || '#' ||
    "Seq"`);
  - `PARENT` can be declared and is not resolved by name;
  - `attached` is true only when declared;
  - the six recorded 082 shapes are unchanged with no declaration;
  - the resolver cache keys on the mapping.
- [X] T004 [P] Unit tests for the pure module in `tests/unit/test_byot_attached.py`:
  - `ColumnMapping.from_settings` (config plus kwargs, kwargs win, string or list `id_col`);
  - `FilterSpec` validation.
- [X] T005 (FR-001, FR-002, FR-003) Add `ColumnRole.PARENT`, declared roles in `TableShape.from_snapshot(snapshot,
  declared=None)` and `_resolve_roles`, plus `identifier_columns`, `identifier_sql` and
  `attached`, in `iris_vector_rag/storage/table_shape.py`. Make `column_for` and
  `legacy_column_for` tolerate roles outside `ROLE_PRECEDENCE` and `ROLE_ALIASES`.
- [X] T006 Add the `declared` argument and the mapping in the cache key to
  `TableShapeResolver.resolve` in `iris_vector_rag/storage/shape_resolver.py`.
- [X] T007 Create `iris_vector_rag/storage/attached.py` with `ColumnMapping`, `FilterSpec`,
  `quote_identifier`, and the new errors `FilterError`, `RequiredFilterMissing` and
  `AttachedTableReadOnly` (the errors go in `iris_vector_rag/exceptions.py`, re-exported).
- [X] T008 (FR-004) Read the mapping and filter settings in `IRISVectorStore.__init__`, with kwargs
  overriding `storage:iris:` keys, and pass `declared` to the resolver in `shape()` and
  `refresh_shape()`, in `iris_vector_rag/storage/vector_store_iris.py`.
- [X] T009 (FR-004) Document every new `storage:iris:` key, unset, in
  `iris_vector_rag/config/default_config.yaml`.
- [X] T010 Gate: T003 and T004 pass, and the existing shape suites pass
  (`tests/unit/test_vector_store_shape.py`, `tests/unit/test_table_shape*.py`,
  `tests/integration/test_table_shape_six_shapes.py`).

## Phase 3: User Story 1 - Search an attached table by vector (P1) 🎯 MVP

**Goal**: The store's vector search over MedTeam's and Štafeta's tables returns the team's ids
and ranking. The table is never written. The query dimension is checked.

**Independent test**: `tests/integration/test_byot_iris.py::test_us1_*`.

- [X] T011 [P] [US1] (FR-005, FR-011, FR-012) Unit tests in `tests/unit/test_byot_store_vector.py` (fake cursor plus a
  recorded catalog of both teams' tables):
  - the vector SQL selects `identifier_sql` and the declared text column;
  - the metric is cosine on attached tables (`VECTOR_COSINE`) and dot on own tables;
  - params are `[vector]`;
  - a composite-id document id is `a#b`;
  - title and parent are in metadata;
  - the dimension mismatch is raised before execute;
  - `add_documents`, `delete_documents` and `clear_documents` raise `AttachedTableReadOnly` with
    no SQL;
  - `fetch_documents_by_ids` filters on `identifier_sql`;
  - an `embedder` callable is used by `search_by_text`, `similarity_search_with_score` and the
    LangChain `similarity_search`.
- [X] T012 [P] [US1] Unit tests for the builder in `tests/unit/test_vector_safe_helpers.py`:
  `build_bound_vector_dot_sql(metric=..., id_expression=...)`, and an invalid metric raises.
- [X] T013 [US1] Integration tests `test_us1_*` in `tests/integration/test_byot_iris.py`:
  - MedTeam top 5 equals the team SQL for both models (via `where` once US2 lands; for US1, via a
    table holding one model);
  - Štafeta composite ids are `DocumentReference/n#m`;
  - catalog and row counts are unchanged (SC-004);
  - a 384 query against a 1536 column raises.
- [X] T014 [US1] (FR-012, FR-020) Extend `build_bound_vector_dot_sql` in
  `iris_vector_rag/common/vector_sql_utils.py` with `metric` (`dot`, `cosine`) and a trusted
  `id_expression`. Allow quoted identifiers in its name checks.
- [X] T015 [US1] (FR-005, FR-011, FR-012) In `iris_vector_rag/storage/vector_store_iris.py`:
  - add the metric default (cosine when the shape is attached, else dot) and the `metric`
    setting;
  - use `identifier_sql` in the vector search, the metadata fetch and `fetch_documents_by_ids`;
  - add the `embedder` hook in `_embed_query`, routing `similarity_search_with_score` and the
    LangChain path through it;
  - add the read-only guard on the write methods;
  - put title and parent in metadata.
- [X] T016 [US1] Gate: T011 to T013 pass on IRIS, and so does the full unit suite (apart from the
  2 known ivg-credential failures from the venv's ivg 4.1.1).

## Phase 4: User Story 2 - Scope every search with real-column filters (P1)

**Goal**: `where=` with `filter_columns` and `required_filters`, bound, on every leg.

**Independent test**: `tests/integration/test_byot_iris.py::test_us2_*`.

- [X] T017 [P] [US2] (FR-006, FR-007, FR-008, FR-009, FR-020) Unit tests for `compile_where` in `tests/unit/test_byot_where.py`:
  - each op; a list gives `IN`; an empty list raises;
  - an unknown key raises naming the allowed ones;
  - a missing required filter raises `RequiredFilterMissing` naming it;
  - a filter column missing from the table raises;
  - date and timestamp columns get `CAST(? AS DATE)` / `CAST(? AS TIMESTAMP)`;
  - values are only in params; an injection string stays a param;
  - the predicate never starts with `(`; quoted names; deterministic order.
- [X] T018 [P] [US2] (FR-010) Unit tests in `tests/unit/test_byot_store_where.py`:
  - the vector search appends the predicate and passes `[vector, *params]`;
  - it combines with a metadata filter by AND;
  - a required filter is checked before any SQL;
  - `search_by_text`, `search_by_vector`, `search_by_text_with_score` and
    `similarity_search_with_score` accept and forward `where`.
- [X] T019 [US2] Integration tests `test_us2_*` in `tests/integration/test_byot_iris.py`:
  - MedTeam patient plus model `where` equals the team SQL for both models (SC-001);
  - Štafeta patient plus as-of dates (`AuthoredOn <`, `ReceivedOn <=`) equals the team SQL;
  - an `IN` list;
  - a missing required filter raises;
  - an injection value matches nothing;
  - the DB-API accepts the bound vector plus where params (the parenthesis guard).
- [X] T020 [US2] Implement `compile_where` in `iris_vector_rag/storage/attached.py`.
- [X] T021 [US2] (FR-009, FR-010) Wire `where` into `similarity_search_by_embedding` and every public search
  method in `iris_vector_rag/storage/vector_store_iris.py`, with the required-filter check first.
- [X] T022 [US2] Gate: T017 to T019 pass on IRIS.

## Phase 5: User Story 3 - Run a ready-made pipeline over the attached table (P2)

**Goal**: configuration alone runs the basic pipeline over MedTeam's table, and `where` travels
through `QueryOptions` and the engine.

**Independent test**: `tests/integration/test_byot_iris.py::test_us3_*`.

- [ ] T023 [P] [US3] (FR-010) Unit tests in `tests/unit/test_byot_engine.py`:
  - `normalize_query_params(where=...)` gives `QueryOptions.where`;
  - the engine's vector, text and fusion legs pass `where` to the store;
  - phrase, concept, global and mix raise `ValueError` when `where` is set;
  - a constructor kwarg overrides config.
- [ ] T024 [US3] Integration test `test_us3_basic_pipeline_from_config` in
  `tests/integration/test_byot_iris.py`:
  - set the mapping with `RAG_STORAGE__IRIS__*` environment variables only;
  - build the basic pipeline with a stub LLM and the test embedder;
  - query with the patient and model `where`;
  - every retrieved document is that patient's row, with its declared id and text (SC-003).
- [ ] T025 [US3] Add `where` to `QueryOptions` and `normalize_query_params` in
  `iris_vector_rag/core/query_options.py`.
- [ ] T026 [US3] (FR-010) Pass `where` in `_vector_leg`, `_text_leg` and `_retrieve_fusion`, and refuse it
  in phrase, concept, global and mix, in `iris_vector_rag/retrieval/engine.py`. Check that the
  basic pipeline's query path forwards `QueryOptions` to the engine; if it calls the store
  directly, forward `where` there (`iris_vector_rag/pipelines/basic.py`).
- [ ] T027 [US3] Gate: T023 and T024 pass on IRIS.

## Phase 6: User Story 4 - Full-text and hybrid search on the table's own index (P2)

**Goal**: `text_index`, `text_terms: prefix` and `where` on the text leg, with composite ids,
and no index creation on attached tables.

**Independent test**: `tests/integration/test_byot_iris.py::test_us4_*`.

- [ ] T028 [P] [US4] (FR-014, FR-015, FR-016) Unit tests in `tests/unit/test_byot_text.py`:
  - `search_terms(..., terms="prefix")` gives `vysaz*` from `vysazena`, `antikoa*` from
    `antikoagulace` and `ecog*` from `ECOG`; words under 4 letters stay whole; stop words follow
    `text_language`;
  - `build_text_search_sql` uses the configured index and rank procedure, selects
    `identifier_sql`, and appends the where predicate with params in order (rank, match, where…,
    order);
  - an attached table with no index raises `TextIndexMissingError` with the read-only hint and
    never calls `ensure_text_index`.
- [ ] T029 [US4] Integration tests `test_us4_*` in `tests/integration/test_byot_iris.py`:
  - Štafeta's own `cs` iFind index, `text_terms=prefix` and the as-of `where` return the same
    rows as the team's iFind SQL (SC-002);
  - rrf with `where` returns no out-of-scope row;
  - an attached table with no index raises.
- [ ] T030 [US4] (FR-015) Add the index name and prefix terms in `iris_vector_rag/storage/text_search.py`
  (`search_terms(query, language, terms)`, and `build_text_search_sql(..., where_sql)`).
- [ ] T031 [US4] (FR-014, FR-016) Wire `text_index`, `text_terms` and `where` into `text_search`,
  `_ranked_text_rows` and `_require_text_index` in
  `iris_vector_rag/storage/vector_store_iris.py`.
- [ ] T032 [US4] Gate: T028 and T029 pass on IRIS.

## Phase 7: User Story 5 - The similarity measure matches the table (P3)

**Goal**: warn when the HNSW `Distance` differs from `metric`, and measure HNSW use (SC-007).

**Independent test**: `tests/integration/test_byot_iris.py::test_us5_*`.

- [ ] T033 [P] [US5] (FR-013) Unit tests in `tests/unit/test_byot_metric.py`:
  - `catalog.hnsw_distance` parses the `Parameters` list;
  - the store logs one warning per table on a mismatch and none on a match or with no index;
  - `metric=dot` overrides the attached default.
- [ ] T034 [US5] Integration tests `test_us5_*` in `tests/integration/test_byot_iris.py`:
  - on Štafeta's table (HNSW `Distance='Cosine'`), `metric=dot` logs the warning naming the index;
  - the default does not warn;
  - non-unit vectors rank differently under cosine and dot.
- [ ] T035 [US5] (FR-013) Add `hnsw_distance(schema, table, column)` in
  `iris_vector_rag/storage/catalog.py`, and the once-per-table check in the store's vector search.
- [ ] T036 [US5] Write `scripts/perf/byot_hnsw_probe.py`: 5,000 rows, cosine and dot, `EXPLAIN`
  plus timings, with and without a patient filter. Record the results in `plan.md` under
  "Measured" (SC-007).
- [ ] T037 [US5] Gate: T033 and T034 pass on IRIS, and T036 is recorded.

## Phase 8: Polish and cross-cutting

- [ ] T038 [P] (FR-018) Write `docs/BRING_YOUR_OWN_TABLE.md` from `quickstart.md`:
  - MedTeam and Štafeta;
  - `where=` next to RAG.SDK's predicate;
  - the model-identity filter advice.

  Link it from `README.md`.
- [ ] T039 [P] (FR-019) Add "Superseded in part by 092" lines to `specs/069-attach-existing-corpus/spec.md`
  and `specs/082-table-shape/spec.md`, and a docstring pointer in
  `HybridGraphRAGPipeline.attach_existing_corpus`.
- [ ] T040 Add a CHANGELOG Unreleased entry for 092.
- [ ] T041 (FR-017) Full run:
  - unit suite;
  - integration suites `test_byot_iris.py`, `test_text_search_iris.py`,
    `test_table_shape_six_shapes.py`, `test_hybrid_rrf_modes.py` and `test_note_search_iris.py`;
  - `python3 scripts/gates/antipatterns.py` (no new findings beyond master's);
  - `bash .specify/gates/verify.sh --boundary agent`.

## Dependencies & Execution Order

- **Phase order:** Setup → Foundational (T003 to T010) → US1 → US2 → US3 and US4 (either order;
  both need US2's `where`) → US5 → Polish.
- **Story dependencies:**
  - US1 is the MVP and stands alone.
  - US2 needs US1's store path.
  - US3 and US4 need US2.
  - US5 needs US1 only, but is scheduled last as P3.
- **Within a phase:** the unit tests ([P]) are written and fail first; then the integration test;
  then the code; then the gate.

## Parallel Example

```text
Phase 2:  T003 (test_byot_shape.py) ‖ T004 (test_byot_attached.py)
US1:      T011 (test_byot_store_vector.py) ‖ T012 (test_vector_safe_helpers.py)
US2:      T017 (test_byot_where.py) ‖ T018 (test_byot_store_where.py)
Polish:   T038 (docs) ‖ T039 (spec notes)
```

## Implementation Strategy

- **MVP:** Phases 1 to 3 (US1). With them, a team can search its own table by vector with
  correct ids and ranking, and nothing is written.
- **Incremental:** US2 makes it correct for scoped clinical use (the real MVP for Rakathon-style
  users); US3 opens the pipelines; US4 adds the text leg; US5 adds the metric warning and
  measurement.
- **One commit per phase after its gate,** locally, on `092-bring-your-own-table`. Nothing is
  pushed.
