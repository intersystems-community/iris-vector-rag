# Tasks: Table Shape

**Input**: Design documents from `/specs/082-table-shape/`
**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md),
[data-model.md](./data-model.md), [contracts/table-shape.md](./contracts/table-shape.md)

**Tests**: mandatory. Unit tests are written before the code they test; each user
story phase ends with an E2E test that must pass before the next phase starts.

**Organization**: one phase per user story, in spec priority order. Phases 1 and 2
are shared prerequisites; the final phase is cross-cutting cleanup.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: can run in parallel (different files, no dependency on an unfinished task)
- **[Story]**: the user story the task serves (US1-US4)

---

## Phase 1: Setup (Shared Infrastructure)

**Purpose**: the six shapes need one definition that both the fixture capture and
the integration suite read, so neither can drift from the other.

- [X] T001 Create `tests/fixtures/shapes/ddl.py` defining `SIX_SHAPES: dict[str, str]` — one `CREATE TABLE` statement per `ShapeKind` (canonical, legacy_serial, legacy_id, fixture, mixed, double_vector) parameterised by table name, matching the shapes described in `spec.md` US1
- [X] T002 Add the `shape_fixtures` pytest fixture (returns `Path` to `tests/fixtures/shapes/`) and a `recorded_catalog` fixture in `tests/conftest.py`
- [X] T003 [P] Create `scripts/dev/capture_shape_fixtures.py`: creates each `SIX_SHAPES` table in a scratch schema against `iris-vector-rag-iris`, dumps the three catalog result sets per table to `tests/fixtures/shapes/<kind>.json`, drops the tables in a `finally`

---

## Phase 2: Foundational (Blocking Prerequisites)

**Purpose**: the three new modules and their unit tests. No call site changes, so
nothing here can break an existing caller.

**Gate**: `./scripts/ci/run-tests.sh -t unit` green with no database running.

### Tests first

- [X] T004 [P] Unit tests for `ColumnFact` derived properties in `tests/unit/test_table_shape.py`: `is_stream` true only for `%Stream.*` class types, `is_long_text` for `longvarchar`, `filterable_with_like` false for both — including the case that distinguishes `content` (longvarchar, `%Library.String`) from `chunk_text` (longvarchar, `%Stream.GlobalCharacter`)
- [X] T005 [P] Unit tests for `VectorSpec` parsing in `tests/unit/test_table_shape.py`: the real `Parameters` string `CALCSELECTIVITY,1,CAPTION,,DATATYPE,FLOAT,EXTERNALSQLNAME,,EXTERNALSQLTYPE,,INDEXNULLMARKER,,JAVATYPE,,LEN,384,STORAGEDEFAULT,GLOBALNODE,STORAGEDEFAULTTARGET,` yields `("FLOAT", 384)`; empty values are preserved; a missing `DATATYPE` or `LEN` raises `ShapeUnresolvable`
- [X] T006 [P] Unit tests for role resolution in `tests/unit/test_table_shape.py`: canonical wins for `TEXT` on the mixed shape, declared single-column key outranks name precedence for `IDENTIFIER`, name precedence applies only without a key, lookup is case-insensitive while `ColumnFact.name` keeps catalog spelling, `legacy_column_for(TEXT)` returns `content` on the mixed shape and `None` on the canonical one (FR-003)
- [X] T007 [P] Unit tests for `TableShape` errors in `tests/unit/test_table_shape.py`: `NotVectorSearchable` names the type found for a `%Library.Binary` embedding and for an absent embedding, `IdentifierNotWritable` names schema/table/column for an identity identifier, `RoleNotPresent` names the role, and every message contains schema and table (FR-011)
- [X] T008 [P] Unit tests for `from_snapshot` purity and `ShapeKind` classification in `tests/unit/test_table_shape.py`: two calls on the same snapshot return equal shapes; each of the six recorded snapshots classifies to its expected kind; an unclassifiable but resolvable shape is `UNKNOWN`, not an error
- [X] T009 [P] Unit tests for `TableShapeResolver` in `tests/unit/test_shape_resolver.py`: N resolves of one table take one snapshot (`reads == 1`), `invalidate` forces exactly one more, `invalidate_all` clears every table, two resolvers share no state, a reader returning `None` raises `TableNotFound`, and a reader raising a driver error surfaces `CatalogUnavailable` with `__cause__` set

### Implementation

- [X] T010 Implement `iris_vector_rag/storage/table_shape.py`: `ColumnRole`, `ShapeKind`, `VectorSpec`, `ColumnFact`, `TableShape` (frozen, `from_snapshot`, `column_for`, `optional_column_for`, `has`, `legacy_column_for`, `vector_spec`, `writable_identifier`, `is_canonical`), the error hierarchy, and `CANONICAL_DOCUMENT_SHAPE` / `CANONICAL_CHUNK_SHAPE`
- [X] T011 Implement `iris_vector_rag/storage/catalog.py`: `CatalogSnapshot`, the `CatalogReader` protocol, `IRISCatalogReader` (class-name lookup via `%Dictionary.CompiledClass`, columns via `INFORMATION_SCHEMA.COLUMNS` incl. `IS_IDENTITY`/`AUTO_INCREMENT`, properties via `%Dictionary.CompiledProperty` joined on `SqlFieldName`, key columns via `KEY_COLUMN_USAGE`), and `RecordedCatalog` with `from_fixture_dir` (FR-001)
- [X] T012 Implement `iris_vector_rag/storage/shape_resolver.py`: `TableShapeResolver` with per-instance cache, `resolve`, `invalidate`, `invalidate_all`, `reads`
- [X] T013 Move the canonical column-name constants into `table_shape.py` and re-export them from `iris_vector_rag/storage/constants.py` for one release
- [X] T014 Run `scripts/dev/capture_shape_fixtures.py` against `iris-vector-rag-iris` and commit the six `tests/fixtures/shapes/<kind>.json` files
- [X] T015 Integration test in `tests/integration/test_table_shape_six_shapes.py::test_recorded_fixtures_match_live_catalog`: for each shape, create the real table, take an `IRISCatalogReader` snapshot, and assert the resulting `TableShape` equals the one built from the committed fixture — the check that keeps recorded shapes honest

**Checkpoint**: the shape can be resolved and asserted on with and without a
database; no call site uses it yet.

- [X] T016 Verify the Phase 2 gate

```accept
# verifies: Phase 2 gate -- shape modules and their unit tests are green with no database
pytest tests/unit/test_table_shape.py tests/unit/test_shape_resolver.py -q
```

---

## Phase 3: User Story 1 - Search, write and delete behave the same on every observed shape (Priority: P1)

**Goal**: every column name in the vector store comes from the resolved shape, so
the reader and the writer stop disagreeing and all six shapes behave alike.

**Independent test**: create each of the six shapes on a fresh IRIS, load the same
documents through the library, run the same search, filter and delete against each,
and compare results.

### Tests first

- [X] T017 [P] [US1] Unit tests in `tests/unit/test_vector_store_shape.py` driven by `RecordedCatalog`: on the mixed shape the write path and the read path emit the same text column; on a stream metadata column the filter uses `SUBSTRING` and not `LIKE`; `TO_VECTOR` carries the shape's element type on the double-precision shape; a caller-supplied identifier against the legacy-serial shape raises `IdentifierNotWritable` before any SQL is sent (FR-008)
- [X] T018 [P] [US1] Unit test in `tests/unit/test_vector_store_shape.py`: a query embedding whose length differs from `vector_spec().dimension` is rejected naming both lengths, and no query is sent
- [X] T019 [P] [US1] Unit test in `tests/unit/test_vector_store_shape.py`: on a shape with no embedding column, a vector search raises `NotVectorSearchable` naming the table — replacing today's silent `FLOAT` fallback
- [X] T020 [US1] Integration test `tests/integration/test_table_shape_six_shapes.py::test_search_filter_write_delete_parity`: parameterised over the six shapes, load identical documents, assert identical identifiers, text, ordering and scores against the canonical shape's results, then assert delete removes exactly the named documents (FR-002)
- [X] T021 [US1] Integration test `tests/integration/test_table_shape_six_shapes.py::test_mixed_table_round_trip`: write a document to the mixed shape, read it back, assert the text read is the text written; then insert a row with only the legacy column populated and assert the per-row fallback returns its text
- [X] T022 [US1] E2E test `tests/e2e/test_table_shape_e2e.py::test_us1`: full pipeline query against a legacy-shaped table end to end, asserting answer and retrieved documents, with no column named by the caller

### Implementation

- [X] T023 [US1] Add an optional `shapes: TableShapeResolver | None` parameter to `IRISVectorStore.__init__` in `iris_vector_rag/storage/vector_store_iris.py`, defaulting to one built from its own connection; resolve the document table's shape once on first use (FR-014)
- [X] T024 [US1] Migrate the write path (`_store_documents` at `vector_store_iris.py:671`, `add_documents` at `:849`) to `shape.column_for(...)` and `shape.writable_identifier()`, deleting the inline precedence at `:701-715`
- [X] T025 [US1] Migrate the read path (`similarity_search_by_embedding` at `:1054`, `search_with_embedding` at `:308`) to the shape, deleting the contradictory precedence at `:1110-1120` and implementing the documented per-row mixed-table text fallback (FR-004)
- [X] T026 [US1] Migrate `fetch_documents_by_ids` (`:1433`), `get_document_count` (`:1517`) and `get_all_documents` (`:1540`) to the shape and delete the try-canonical-catch-legacy retries at `:1452-1500` and `:1551-1590`
- [X] T027 [US1] Migrate the delete path: replace `_get_id_column` (`:925`) with `shape.column_for(IDENTIFIER)` in `delete_documents` (`:936`) and `clear_documents` (`:1608`), and remove the method
- [X] T028 [US1] Migrate `_metadata_filter_column_expr` (`:981`), `_metadata_filter_predicate` (`:1018`) and `_apply_exact_metadata_filter` (`:1042`) to decide `SUBSTRING` versus direct comparison from `ColumnFact.filterable_with_like`, deleting their own catalog read (FR-009)
- [X] T029 [US1] Replace `_get_vector_data_type` (`:151`) and `get_embedding_column_metadata` (`:247`) with `shape.vector_spec()`, deleting the `IRIS_VECTOR_DATA_TYPE` fallback and the fixed dimension defaults (FR-001, FR-010)
- [X] T030 [US1] Verify the US1 phase gate

```accept
# verifies: US1 gate -- store behaviour is identical on every recorded shape
pytest tests/unit/test_vector_store_shape.py -q
```

**Checkpoint**: US1 is independently testable and shippable on its own.

---

## Phase 4: User Story 2 - One shape probe per table per session (Priority: P2)

**Goal**: a table's shape is resolved once per session; a search issues one ranked
query and one metadata fetch and no catalog query.

**Independent test**: count catalog queries during N searches against one table.

### Tests first

- [X] T031 [P] [US2] Unit test in `tests/unit/test_shape_resolver.py`: a store built on a counting `RecordedCatalog` performs 100 searches with `reads` unchanged after the first resolution
- [X] T032 [US2] Integration test `tests/integration/test_table_shape_six_shapes.py::test_catalog_round_trips` (SC-002): wrap the live connection in a statement-counting proxy, run 100 searches, assert exactly one shape resolution and zero catalog statements inside the loop (FR-005)
- [X] T033 [US2] Integration test in the same file: after the library itself creates or alters the table, the next operation sees the new shape with no caller action; after an out-of-band change plus an explicit refresh, the next operation uses the new shape
- [X] T034 [US2] E2E test `tests/e2e/test_table_shape_e2e.py::test_us2`: a pipeline run issuing many queries resolves each table's shape once

### Implementation

- [X] T035 [US2] Call `resolver.invalidate(...)` from every DDL path in `vector_store_iris.py` (`_ensure_table_exists` at `:366`, `_create_table_automatically` at `:1369`)
- [X] T036 [US2] Expose a `refresh_shape(table)` passthrough on `IRISVectorStore` for callers who suspect an out-of-band change, and document it in the contract's promise 14
- [X] T037 [US2] Remove the remaining per-call catalog reads in `vector_store_iris.py` (`_get_connection`-adjacent column probes and `_get_table_columns` callers) so the only catalog access is `IRISCatalogReader`
- [X] T038 [US2] Verify the US2 phase gate

```accept
# verifies: US2 gate -- one shape resolution serves many operations
pytest tests/unit/test_shape_resolver.py -q -k "cache or reads or invalidate"
```

---

## Phase 5: User Story 3 - Only one shape is ever created (Priority: P2)

**Goal**: every creation path emits the canonical shape and nothing else, and no
path alters an existing table implicitly.

**Independent test**: create the document table through each available path on a
fresh IRIS and compare the resulting shapes.

### Tests first

- [X] T039 [P] [US3] Unit test in `tests/unit/test_table_shape.py`: the DDL rendered from `CANONICAL_DOCUMENT_SHAPE` and `CANONICAL_CHUNK_SHAPE` matches the expected statement exactly, including vector element type and dimension
- [X] T040 [US3] Integration test `tests/integration/test_table_shape_six_shapes.py::test_every_creation_path_is_canonical` (SC-004): create the document table via `SchemaManager` and via the store's auto-create path on a fresh instance and assert the two resolved shapes are equal and canonical
- [X] T041 [US3] Integration test in the same file: pointing the library at each legacy shape performs no `ALTER TABLE` and no `DROP` (statement-counting proxy), and a request to create a non-canonical shape is refused with an error naming the table
- [X] T042 [US3] E2E test `tests/e2e/test_table_shape_e2e.py::test_us3`: fresh-instance bootstrap through the public entry point yields the canonical shape

### Implementation

- [X] T043 [US3] Emit the canonical DDL in `iris_vector_rag/storage/schema_manager.py` from `CANONICAL_DOCUMENT_SHAPE` (replacing the literal at `:1157-1167`) and `CANONICAL_CHUNK_SHAPE` (replacing `:1324-1335`) (FR-006)
- [X] T044 [US3] Route `schema_manager.py`'s own catalog reads (`:209`, `:1109`, `:1260-1264`, `:1302`, `:1760`, `:1887-1890`, `:2245`, `:2255-2258`) through `IRISCatalogReader` / the resolver (FR-013)
- [X] T045 [US3] Make the alter paths explicit: refuse non-canonical creation, and require an explicit destructive-migration call rather than altering on the strength of a reported type (FR-007)
- [X] T046 [US3] Verify the US3 phase gate

```accept
# verifies: US3 gate -- canonical DDL is rendered from the canonical shape
pytest tests/unit/test_table_shape.py -q -k canonical
```

---

## Phase 6: User Story 4 - Tests describe shapes, not database chatter (Priority: P3)

**Goal**: a shape-dependent test states the shape and asserts on behaviour; nothing
outside the shape owner reads the catalog.

**Independent test**: the shape-dependent storage tests pass using recorded shapes,
with no scripted cursor responses.

### Tests first

- [X] T047 [P] [US4] Port the existing shape-dependent unit tests to `RecordedCatalog`, deleting the scripted cursor/mock responses they use today (FR-015)
- [X] T048 [US4] Test for the new detector in `tests/unit/test_antipatterns_detectors.py`: a file that reads `INFORMATION_SCHEMA` or `%Dictionary` outside `iris_vector_rag/storage/catalog.py` is reported, and the same code inside it is not
- [X] T049 [US4] E2E test `tests/e2e/test_table_shape_e2e.py::test_us4`: the recorded-shape suite and the live suite reach the same verdict on the same shape

### Implementation

- [X] T050 [US4] Add the `catalog-read-outside-shape` detector to `scripts/gates/antipatterns.py` (SC-003) and record any remaining known instances in `antipatterns-baseline.txt`
- [X] T051 [US4] Verify the US4 phase gate

```accept
# verifies: US4 gate -- no catalog read outside the shape owner
python3 scripts/gates/antipatterns.py
```

---

## Phase 7: Polish & Cross-Cutting Concerns

- [ ] T052 [P] Migrate `iris_vector_rag/services/storage.py:130-146` to `shape.column_for(IDENTIFIER)`, deleting its own derivation
- [ ] T053 [P] Migrate `iris_vector_rag/validation/orchestrator.py:588-617` to the shape, deleting its `doc_id`-first precedence and its own content precedence
- [ ] T054 [P] Migrate `iris_vector_rag/pipelines/hybrid_graphrag.py:180-258` to take the embedding column and index presence from the shape, deleting its own catalog read
- [ ] T055 [P] Migrate the document-table joins in `iris_vector_rag/pipelines/graphrag.py` and `graphrag_merged.py` to `shape.column_for(IDENTIFIER)` so both agree (FR-012)
- [ ] T056 Deprecate `schema_config` column overrides: emit `DeprecationWarning` and ignore them, with a unit test asserting the warning and that the database shape wins
- [ ] T057 Update `CHANGELOG.md` for 0.15 (behaviour change on mixed tables, `IRIS_VECTOR_DATA_TYPE` no longer a fallback, `schema_config` column overrides deprecated) and document the shape in the storage docs
- [ ] T058 Run the full CI-equivalent suite and, only if `test_graphrag_handler_execute_hybrid_search` passes there, remove its entry from `tests/contract/ci_known_failures.txt`
- [ ] T059 Verify the cross-cutting gate

```accept
# verifies: Polish gate -- agent boundary clean and no scripted catalog mocks remain
bash .specify/gates/verify.sh --boundary agent
```

---

## Dependencies

```text
Phase 1 (T001-T003)
   └─> Phase 2 (T004-T016)          shape modules + fixtures; nothing else can start
          ├─> Phase 3 US1 (T017-T030)      P1, the MVP
          │      └─> Phase 4 US2 (T031-T038)   needs the store on the resolver
          ├─> Phase 5 US3 (T039-T046)      independent of US1/US2 (schema_manager only)
          └─> Phase 6 US4 (T047-T051)      needs RecordedCatalog (T011) only
                 └─> Phase 7 (T052-T059)
```

US3 and US4 can proceed in parallel with US1 once Phase 2 is done; US2 depends on
US1 because it removes reads from the paths US1 migrates. Phase 7 is last because
its detector (T050) would otherwise fire on call sites not yet migrated.

## Parallel opportunities

- Phase 2: T004-T009 are six independent test files/sections, all `[P]`.
- Phase 3: T017-T019 are `[P]` (same new file, disjoint test functions — write them
  in one pass); T024-T029 touch disjoint methods of `vector_store_iris.py` and are
  sequential only because they share a file.
- Phase 7: T052-T055 are four different files, fully parallel.

## Implementation strategy

MVP is Phase 1 + Phase 2 + Phase 3 (US1). That alone fixes the shipped defect
class: the reader and writer agree, all six shapes work, and the silent type
fallback is gone. US2 is a latency and load win, US3 prevents new legacy shapes
from being created, US4 makes the tests cheap to keep.

Test-first is not optional here: T004-T009 exist because `from_snapshot` is a pure
function of recorded catalog rows, so its tests can be written completely before
any of T010-T012 exists.
