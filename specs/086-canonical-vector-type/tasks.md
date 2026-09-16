# Tasks: Canonical Vector Type, One DDL Owner

**Input**: Design documents from `/specs/086-canonical-vector-type/`
**Prerequisites**: [plan.md](./plan.md), [spec.md](./spec.md), [research.md](./research.md)

**Tests**: mandatory. Unit tests come before the code they test; every user story phase
ends with an integration or E2E test that must pass before the next phase starts.

**Organization**: one phase per user story, in spec priority order. There is no
foundational phase — see the note under Phase 2.

## Format: `[ID] [P?] [Story] Description`

- **[P]**: can run in parallel (different files, no dependency on an unfinished task)
- **[Story]**: the user story the task serves (US1-US4)

---

## Phase 1: Setup

- [X] T001 Run `bash .specify/scripts/bash/check-container.sh` and confirm `iris-vector-rag-iris` on port `51972` is up; if it is stopped, `docker start iris-vector-rag-iris` — never substitute another container
- [X] T002 Capture the pre-change red baseline so later failures can be attributed: `./scripts/ci/run-tests.sh` on the branch point, deduped `FAILED`/`ERROR` node ids saved to `/tmp/086-baseline-before.txt`
- [X] T002a Give the worktree the project's `.env` (`IRIS_PORT=51972`). Without it `tests/conftest.py`'s `load_dotenv()` finds nothing and `auto_detect_iris_port()` falls back to `1972`, which is a **different project's container**. The first baseline attempt ran there and was discarded; the real baseline is captured in T047 from a pristine worktree at the branch point

---

## Phase 2: Foundational

Intentionally empty. Neither half of this feature needs new shared infrastructure: the
delegation target (`SchemaManager.ensure_table_schema`), the element-type resolver
(`RAG.SDK.Search.GetVectorType`) and the detector-testing split
(`tests/unit/test_antipatterns_detectors.py`) all already exist — research findings 8, 10
and 12. Writing a placeholder module here would be the only thing to undo later.

---

## Phase 3 (US1): Storing documents never destroys a populated table

**Goal**: `IRISStorage.initialize_schema` stops writing and dropping the document table
and delegates to the schema owner.

**Independent test**: populate a table, force the `store_documents` probe to fail once,
store a document, count the rows.

**Gate**: `pytest tests/test_enterprise_storage.py` plus the new integration test pass.

### Tests first

- [X] T003 [P] [US1] Unit test in `tests/test_enterprise_storage.py`: `initialize_schema` calls `SchemaManager.ensure_table_schema` with the **unqualified** `"SourceDocuments"` when `self.table_name` is `"RAG.SourceDocuments"` (FR-012, FR-015)
- [X] T004 [P] [US1] Unit test in `tests/test_enterprise_storage.py`: no statement passed to `cursor.execute` during `initialize_schema` matches `CREATE TABLE`, `DROP TABLE` or `DELETE FROM` (FR-012, FR-013)
- [X] T005 [P] [US1] Unit test in `tests/test_enterprise_storage.py`: `ensure_table_schema` returning `False` makes `initialize_schema` raise, and `self.table_name` is unchanged afterwards (FR-014)
- [X] T006 [P] [US1] Unit test in `tests/test_enterprise_storage.py`: the HNSW index step still runs with the values from `config_manager.get_vector_index_config()`, and a failure there is still logged rather than raised (FR-016)
- [X] T007 [US1] Rewrite `test_initialize_schema_fallback` and `test_initialize_schema_success` in `tests/test_enterprise_storage.py` against the delegating behaviour — the fallback assertion (`table_name == "SourceDocuments"`) asserts the mutation FR-014 removes, so it is replaced, not repaired (FR-014, FR-018)
- [X] T008 [US1] Integration test in `tests/integration/test_enterprise_storage_no_data_loss.py`: create a table with N rows against `iris-vector-rag-iris`, make the first `SELECT COUNT(*) ... WHERE 1=0` probe raise, call `store_documents` with one document, assert the row count is N+1 and the table was never dropped (FR-013)

### Implementation

- [X] T009 [US1] Rewrite `initialize_schema` in `iris_vector_rag/storage/enterprise_storage.py:99-205`: take the last dot-separated segment of `self.table_name`, call `SchemaManager(connection_manager=..., config_manager=...).ensure_table_schema(<unqualified>)`, raise a descriptive error when it returns `False`, keep the vector-index block at `:164-195` and the `commit`/`rollback`/`cursor.close` structure, and delete the two-name attempt loop, the inline `CREATE TABLE`, the unconditional `DROP TABLE`, the `DELETE FROM` fallback and the `self.table_name` assignment (FR-012, FR-013, FR-014, FR-015, FR-016, FR-017)
- [X] T010 [US1] Record in the method's docstring why it delegates rather than creating: the probe in `store_documents` (`:241-248`) cannot distinguish an absent table from an unavailable one, so anything destructive reached from there is reached by accident (FR-013)
### Discovered while running the phase gate

The index step FR-016 keeps could not survive its own catalog reads: `IRISSQLUtils`'
`check_table_exists`, `check_index_exists` and `get_table_columns` each called
`get_iris_dbapi_connection()` — which returns the process-wide **cached** connection, not a
new one — and closed it in a `finally` block. Every `initialize_schema` against a real
instance therefore ended in `<COMMUNICATION LINK ERROR> Connection closed`, and so did the
`store_documents` call that triggered it. FR-013 is untestable until this is fixed.

- [X] T011a [US1] Unit tests in `tests/test_iris_sql_utils_connection_ownership.py`: each of the three helpers leaves the connection from `get_iris_dbapi_connection` open, and still closes the cursors it opened (FR-013, FR-016)
- [X] T011b [US1] Stop closing the cached connection in those three `finally` blocks in `iris_vector_rag/common/iris_sql_utils.py`, rename `temp_connection` to `shared_connection` so the variable stops claiming ownership, and record the rule in a comment (FR-013, FR-016)

- [X] T011 [US1] **Phase gate**: `pytest tests/test_enterprise_storage.py -q` and `IRIS_BACKEND_MODE=community pytest tests/integration/test_enterprise_storage_no_data_loss.py -q` both pass

---

## Phase 4 (US2): Every creation path declares the canonical element type

**Goal**: the SDK creates `FLOAT` columns and writes into whatever element type the
column has.

**Independent test**: create through each path on a fresh IRIS, read the element type
from `%Dictionary.CompiledProperty.Parameters`, compare.

**Gate**: the two E2E tests pass, and the `.cls` files compile on the server.

### Tests first

- [ ] T012 [US2] Before touching any `.cls`: call `skill(action="describe")` for `objectscript-guardrails` and `objectscript-review` and apply both checklists to every ObjectScript edit in this phase
- [ ] T013 [P] [US2] E2E test in `tests/e2e/test_sdk_canonical_vector_type.py`: run `RAG.SDK.Schema.Initialize(384)` in a scratch schema, read `DATATYPE` and `LEN` from `%Dictionary.CompiledProperty.Parameters` for `embedding` and `chunk_embedding`, assert `FLOAT`/384, then search the table through the Python library and assert no SQLCODE -259 and no retry (FR-001, FR-002)
- [ ] T014 [P] [US2] E2E test in the same file: pre-create a `VECTOR(DOUBLE, 384)` document table, call `RAG.SDK.Pipeline.AddDocument` with an embedding, assert the write succeeds and the row is searchable — the writer named `DOUBLE` because the column is `DOUBLE` (FR-003, FR-004)
- [ ] T015 [P] [US2] E2E test in the same file: the same `AddDocument` call against a `FLOAT` table created by `Initialize`, asserting the write succeeds there too (FR-004)

### Implementation

- [ ] T016 [US2] `iris_src/src/RAG/SDK/Schema.cls:22-23`: `VECTOR(DOUBLE, dim)` → `VECTOR(FLOAT, dim)` in `GetDefaultDDL`'s `d1` and `d2`; `d3` (Entities) has no vector column and is untouched (FR-001, FR-002)
- [ ] T017 [US2] `iris_src/src/RAG/SDK/Pipeline.cls`: replace the literal `DOUBLE` in the three `TO_VECTOR` statements at `:9`, `:11` and `:25` with the type resolved by `##class(RAG.SDK.Search).GetVectorType("RAG.SourceDocuments", "embedding")`, defaulting to the canonical `FLOAT` when it cannot be resolved; also fix the two defects on the statement being rewritten — `:26` calls `st2.%Execute` on a variable never set (the prepare above assigns to `st`), and `:25` hardcodes dimension 384 instead of the caller's `dim` (FR-004)
- [ ] T018 [US2] Push both classes to IRIS and compile there — `iris_doc(mode="put", compile=true)`; a `.cls` saved to disk is not a compiled class. Record the compile output in the phase gate
- [ ] T019 [P] [US2] `sql/schema.sql:9,18`: `VECTOR(DOUBLE, 384)` → `VECTOR(FLOAT, 384)` for `embedding` and `chunk_embedding` (FR-001, FR-005)
- [ ] T020 [P] [US2] `iris_vector_rag/common/db_init_complete.sql:31`: `chunk_embedding VECTOR(DOUBLE, 384)` → `FLOAT`, making the file agree with its own `:13`, `:57`, `:75` and `:134` (FR-005)
- [ ] T021 [US2] Update the docstring at `iris_src/src/RAG/SDK/Search.cls:47-50`, which states that `RAG.SDK.Schema` creates `DOUBLE` columns: after this phase it creates `FLOAT`, and the runtime resolution stays because tables created before it are still `DOUBLE` (FR-002, FR-003)
- [ ] T022 [US2] Confirm `tests/fixtures/shapes/ddl.py:101` and `tests/unit/test_table_shape.py:537` still declare `DOUBLE` — they record 082's sixth observed shape and the renderer's parameterisation, and changing them would delete the coverage that makes FR-003 checkable (FR-003)
- [ ] T023 [US2] **Phase gate**: `IRIS_BACKEND_MODE=community pytest tests/e2e/test_sdk_canonical_vector_type.py -q` passes and both classes report a clean compile on `iris-vector-rag-iris`

---

## Phase 5 (US3): A new hardcoded element type cannot ship unnoticed

**Goal**: a `hardcoded-vector-type` detector with every existing site baselined.

**Independent test**: zero new findings on the tree as it stands; exactly one when a
literal-typed statement is added.

**Gate**: `python3 scripts/gates/antipatterns.py` exits 0.

### Tests first

- [ ] T024 [P] [US3] Unit tests in `tests/unit/test_antipatterns_detectors.py` for a pure `hardcoded_vector_types(path, source)`: reports `TO_VECTOR(?, FLOAT, 384)`, `TO_VECTOR(?,DOUBLE,"_dim_")` in ObjectScript, a `VECTOR(FLOAT, n)` column declaration, and an `ALTER TABLE ... ADD col VECTOR(FLOAT, n)` (FR-006)
- [ ] T025 [P] [US3] Unit tests in the same file for the prose cases that must **not** be reported, one per real form found in the tree: a backticked mention in a docstring (`table_shape.py:16`), bare prose in a docstring (`relation_embedding_store.py:35`) and a `#` comment (`orchestrator.py:276`) (FR-007)

### Implementation

- [ ] T026 [US3] Add `hardcoded_vector_types(path, source)` and its walker to `scripts/gates/antipatterns.py`, following the `catalog_reads`/`catalog_read_outside_shape` split so the pure half is testable; require SQL context on the line — `CREATE TABLE`, `ALTER TABLE`, `TO_VECTOR(`, or an identifier immediately before `VECTOR(` on a line ending in `,` or `NULL` (FR-006, FR-007)
- [ ] T027 [US3] Register `"hardcoded-vector-type"` in `CHECKS`; do **not** add it to `NO_BASELINE` — its purpose is to hold known sites (FR-008)
- [ ] T028 [US3] Scope the walker to shipped code: `iris_vector_rag/**/*.py`, `iris_vector_rag/**/*.sql`, `sql/**/*.sql`, `iris_src/**/*.cls`. Not `.md` — the constitution's own `TO_VECTOR(?, FLOAT, 384)` example at `.specify/memory/constitution.md:67` would otherwise become a finding against the constitution. Not `tests/` — fixtures create legacy shapes deliberately (FR-006, FR-007)
- [ ] T029 [US3] Regenerate the baseline **after** phases 3 and 4, so the lines those phases delete (`enterprise_storage.py:130`, and `:122` for the other detector) are never recorded: `python3 scripts/gates/antipatterns.py --write-baseline`, then diff the file and confirm every added line is a site listed in research finding 4, minus the deleted ones (FR-008)
- [ ] T030 [US3] Verify the detector actually fires: add one literal-typed statement to a shipped module, confirm the gate reports exactly one new finding naming that file and line, then revert the edit (FR-006)
- [ ] T031 [US3] **Phase gate**: `python3 scripts/gates/antipatterns.py` reports 0 new findings and 0 stale entries, and `pytest tests/unit/test_antipatterns_detectors.py -q` passes

---

## Phase 6 (US4): The DDL detector can see the files that hold DDL

**Goal**: `inline-standard-ddl` scans `.sql` and `.cls`, reports variable-named DDL, and
exempts only paths it walks.

**Independent test**: one unit case per blind spot, run against source handed to the
detector.

**Gate**: the detector's unit tests pass and the gate exits 0.

### Tests first

- [ ] T032 [P] [US4] Unit test in `tests/unit/test_antipatterns_detectors.py`: DDL for a standard RAG table in a `.sql` file is reported (FR-009)
- [ ] T033 [P] [US4] Unit test: the same in a `.cls` file, in the ObjectScript string form `Set d1 = "CREATE TABLE IF NOT EXISTS RAG.SourceDocuments (...)"` (FR-009)
- [ ] T034 [P] [US4] Unit test: `CREATE TABLE IF NOT EXISTS {self.entities_table} (` in a non-owner module is reported as DDL whose table the detector cannot identify (FR-010)
- [ ] T035 [P] [US4] Unit test: every path in `DDL_OWNER` is a path the detector's file set contains — the assertion that would have caught the unreachable `sql/schema.sql` exemption (FR-011)

### Implementation

- [ ] T036 [US4] Widen `inline_standard_ddl`'s file set in `scripts/gates/antipatterns.py:131-153` to package `.py` and `.sql`, `sql/**/*.sql` and `iris_src/**/*.cls`, following `legacy_fk_target:109`'s existing two-set pattern (FR-009)
- [ ] T037 [US4] Report DDL whose table name is an interpolated variable, with a message that says the detector cannot resolve it rather than claiming it is a standard table (FR-010)
- [ ] T038 [US4] Make `DDL_OWNER` honest: `sql/schema.sql` becomes a real exemption now that the file is walked; if `schema_manager.py` is the only owner the detector can reach for `.py`, say so in the comment above the set (FR-011)
- [ ] T039 [US4] Extend the baseline for the newly visible instances — the `.sql`/`.cls` DDL and the four variable-named sites (`services/storage.py:1038,1053,1069`, `common/llm_cache_iris.py:143`); `enterprise_storage.py:122` must be gone by now, not baselined (FR-008)
- [ ] T040 [US4] **Phase gate**: `pytest tests/unit/test_antipatterns_detectors.py -q` passes and `python3 scripts/gates/antipatterns.py` reports 0 new findings and 0 stale entries

---

## Phase 7: Polish and cross-cutting

- [ ] T041 [P] `docs/DEBT.md` entry 3: remove the `test_initialize_schema_fallback` row and the note pointing it at this rewrite, and correct the remaining failure count (FR-018)
- [ ] T042 [P] `docs/DEBT.md` entry 4: correct the stale line numbers for `iris_vector_rag/services/storage.py` (`352,391` → `439,478`), widen the entry from 11 `.py` `TO_VECTOR` sites to the full set the new detector baselines, and record that the detector now prevents new ones while rewriting none — deferral (c) in `spec.md` (FR-018)
- [ ] T043 [P] `docs/DEBT.md` entry 5: cross-reference it as deferral (d) so the boundary between "declares the canonical type" and "reports a non-canonical table as needing migration" is written down in both places (FR-018)
- [ ] T044 Add a `CHANGELOG.md` entry naming the behaviour change users can observe: `initialize_schema` no longer drops or empties the document table, and the SDK creates `FLOAT` columns (FR-017)
- [ ] T045 Run `markdownlint-cli2 --fix` and `prettier --write` on every `.md` touched, including this feature's four artifacts
- [ ] T046 Run both gates: `bash .specify/gates/verify.sh --boundary agent` and `python3 scripts/gates/antipatterns.py`
- [ ] T047 Run the full suite and diff deduped `FAILED`/`ERROR` node ids against `/tmp/086-baseline-before.txt` from T002; report the count honestly, separating pre-existing red from anything this feature introduced (zero expected)

---

## Dependencies

- Phase 1 before everything (T001 is a hard block: no implementation without the container).
- Phase 3 and Phase 4 are independent of each other and could run in either order; Phase 3
  is first because it is the data-loss path.
- Phase 5's T029 (baseline) depends on Phase 3 T009 and Phase 4 T016–T020 having landed —
  it must not record `enterprise_storage.py:130`.
- Phase 6's T039 (baseline) depends on Phase 3 T009 for the same reason
  (`enterprise_storage.py:122`).
- Phase 7's T047 depends on T002 having captured the baseline.
- Within each phase, every test task precedes the implementation task it covers.

## Parallel opportunities

- T003–T006 are four independent unit tests in one file; write them together, then T007.
- T013–T015 are three independent E2E cases in one new file.
- T019 and T020 are two different `.sql` files, independent of the `.cls` work.
- T024/T025 and T032–T035 are independent detector unit tests.
- T041–T043 are three independent debt entries.

## Implementation strategy

MVP is Phase 3 alone: it removes a path that can empty a customer's document table, and
it is revertable without touching anything else. Phase 4 is the second P1 and completes
the "one element type" claim. Phases 5 and 6 are the gate work that stops the class of
defect recurring, and they are deliberately last because both end in a baseline that must
reflect the code after the earlier phases, not before.

## Requirement coverage

| Requirement | Tasks                  |
| ----------- | ---------------------- |
| FR-001      | T013, T016, T019       |
| FR-002      | T013, T016, T021       |
| FR-003      | T014, T021, T022       |
| FR-004      | T014, T015, T017       |
| FR-005      | T019, T020             |
| FR-006      | T024, T026, T028, T030 |
| FR-007      | T025, T026, T028       |
| FR-008      | T027, T029, T039       |
| FR-009      | T032, T033, T036       |
| FR-010      | T034, T037             |
| FR-011      | T035, T038             |
| FR-012      | T003, T004, T009, T010 |
| FR-013      | T004, T008, T009, T010 |
| FR-014      | T005, T007, T009       |
| FR-015      | T003, T009             |
| FR-016      | T006, T009             |
| FR-017      | T009, T044             |
| FR-018      | T007, T041, T042, T043 |
