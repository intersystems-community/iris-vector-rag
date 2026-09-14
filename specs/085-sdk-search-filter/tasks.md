# Tasks: RAG.SDK search usable by an external ObjectScript consumer

**Feature**: `085-sdk-search-filter` | **Spec**: [spec.md](./spec.md) | **Plan**: [plan.md](./plan.md)

Test-first is not advisory here: the class being changed has no ObjectScript test coverage at
all, and both defects being fixed are the kind a search test would have caught. Every phase
writes its `SearchTest` methods, observes them fail, then changes `Search.cls` or `Bridge.cls`.

## Phase 1: Setup

- [X] T001 Verify the correct IRIS instance is running with `bash .specify/scripts/bash/check-container.sh`
- [X] T002 Add `scripts/dev/run-objectscript-tests.sh` that copies `iris_src/src/RAG/SDK` and `tests/objectscript/RAG/SDK/Test` into the container, compiles one class per fresh `iris session` (the whole-directory compile segfaults its workers), and runs each requested class through `%UnitTest.Manager.DebugRunTestCase` (FR-012)
- [X] T003 Confirm the harness is real by running `bash scripts/dev/run-objectscript-tests.sh RAG.SDK.Test.SchemaTest` and seeing the existing suite pass — `SchemaTest` rather than `PipelineTest` because `RAG.SDK.Pipeline` has an embedded-Python method and embedded Python does not compile on this container (research.md finding 6) (FR-012)

## Phase 2: Foundational

**Blocking**: every user story's tests need fixture tables with known contents, and no such
helper exists.

- [X] T004 Create `tests/objectscript/RAG/SDK/Test/SearchTest.cls` extending `%UnitTest.TestCase` with `OnBeforeAllTests`/`OnAfterAllTests` that create and drop the feature's fixture tables
- [X] T005 Add `MakeVec(seed, dim)` and `MakeFixture(table, vectorType, withMetadata)` helpers to `tests/objectscript/RAG/SDK/Test/SearchTest.cls`, inserting rows for two owners through `TO_VECTOR` so that scoping and ordering are both observable
- [X] T006 Add `SaveConfig()`/`RestoreConfig()` helpers to `tests/objectscript/RAG/SDK/Test/SearchTest.cls` so tests that repoint `^RAG.SDK.Config` cannot leak into other suites

## Phase 3: User Story 3 — Search works against a FLOAT embedding column (P1)

**Goal**: a search against a `VECTOR(FLOAT, N)` column returns scored rows instead of an error
object.

**Independent test**: create a `FLOAT` fixture, search it, assert scored rows come back.

**Why first**: nothing else in this feature can be tested against a `FLOAT` table while search
returns an error for every one of them.

- [X] T007 [US3] Add `TestVectorSearchAgainstFloatColumn` to `tests/objectscript/RAG/SDK/Test/SearchTest.cls` asserting a scored array, not an `error` object, from a `FLOAT` fixture — must fail before T011 (FR-009)
- [X] T008 [P] [US3] Add `TestVectorSearchAgainstDoubleColumn` to `tests/objectscript/RAG/SDK/Test/SearchTest.cls` pinning the previously working `DOUBLE` case (FR-009)
- [X] T009 [P] [US3] Add `TestGetVectorTypeResolvesElementType` to `tests/objectscript/RAG/SDK/Test/SearchTest.cls` asserting `FLOAT` for the float fixture and `DOUBLE` for the double fixture (FR-009)
- [X] T010 [P] [US3] Add `TestVectorSearchUnresolvableTypeNamesTableAndColumn` to `tests/objectscript/RAG/SDK/Test/SearchTest.cls` asserting the error text names the table and column for a table with no such column (FR-010)
- [X] T011 [US3] Add `GetVectorType(table, column)` to `iris_src/src/RAG/SDK/Search.cls`, resolving class from `INFORMATION_SCHEMA.TABLES.CLASSNAME`, property from `%Dictionary.CompiledProperty.SqlFieldName`, and element type from `Parameters.GetAt("DATATYPE")` (FR-009)
- [X] T012 [US3] Replace the literal `DOUBLE` in `VectorSearch` in `iris_src/src/RAG/SDK/Search.cls` with the resolved type, returning a named error when it cannot be resolved (FR-009, FR-010)
- [X] T043 [US3] Add `TestVectorSearchPlainVarcharTextColumn` to `tests/objectscript/RAG/SDK/Test/SearchTest.cls` — found by T008, which threw `<LIST>` rather than failing an assertion, because `VectorSearch` opened every text value as a stream OID
- [X] T044 [US3] Add `ReadText` to `iris_src/src/RAG/SDK/Search.cls` and use it in `VectorSearch` and `TextSearch`, so a plain `VARCHAR` text column no longer takes the search down
- [X] T047 [US3] Fix the schema/table split in `AttachTable` in `iris_src/src/RAG/SDK/Bridge.cls` to use the last dot rather than the first, so a schema-qualified consumer table is looked up under its own schema instead of a truncated one
- [X] T048 [US3] Use `ReadText` in `BuildBM25Index` in `iris_src/src/RAG/SDK/Bridge.cls`, which opened every text value as a stream OID for the same reason `VectorSearch` did
- [X] T013 [US3] Push `Search.cls` to IRIS and run `bash scripts/dev/run-objectscript-tests.sh RAG.SDK.Test.SearchTest`

### ⛔ PHASE GATE: all Phase 3 `SearchTest` methods pass before Phase 4 starts

## Phase 4: User Story 1 — Scoped search (P1)

**Goal**: a caller-supplied predicate scopes the search.

**Independent test**: two owners' rows in one table; search scoped to one owner returns only
that owner's rows, and returns as many as asked for when that many exist in scope.

- [X] T014 [US1] Add `TestVectorSearchWhereClauseScopesResults` to `tests/objectscript/RAG/SDK/Test/SearchTest.cls` asserting 3 requested rows all belong to the scoped owner where the unscoped top 3 would not — must fail before T019 (FR-001)
- [X] T015 [P] [US1] Add `TestVectorSearchEmptyWhereClauseUnchanged` to `tests/objectscript/RAG/SDK/Test/SearchTest.cls` asserting an omitted predicate gives the pre-085 result (FR-002)
- [X] T016 [P] [US1] Add `TestVectorSearchWhereClauseMatchingNothing` to `tests/objectscript/RAG/SDK/Test/SearchTest.cls` asserting an empty array, not an error (FR-001)
- [X] T017 [P] [US1] Add `TestVectorSearchInvalidWhereClauseNamesFailure` to `tests/objectscript/RAG/SDK/Test/SearchTest.cls` asserting a predicate over a nonexistent column returns an `error` object rather than an empty array (FR-003)
- [X] T018 [P] [US1] Add `TestVectorSearchWhereClauseRejectsStatementBreak` to `tests/objectscript/RAG/SDK/Test/SearchTest.cls` asserting predicates containing `;`, `--`, and `/*` are refused before preparation (FR-004)
- [X] T019 [US1] Add the optional `whereClause` argument to `VectorSearch` in `iris_src/src/RAG/SDK/Search.cls`, ANDed parenthesised into the existing clause, with the fragment shape check (FR-001, FR-004)
- [X] T020 [US1] Make a non-empty `whereClause` skip the IVF fast path in `VectorSearch` in `iris_src/src/RAG/SDK/Search.cls`, since that path cannot honour a predicate (FR-005)
- [X] T021 [P] [US1] Add `TestVectorSearchWhereClauseIgnoresIvfFastPath` to `tests/objectscript/RAG/SDK/Test/SearchTest.cls` asserting a scoped search stays scoped with `^RAG.SDK.Config("ivf_index_name")` set (FR-005)
- [X] T022 [US1] Add the same optional `whereClause` to `TextSearch` and `HybridSearch` in `iris_src/src/RAG/SDK/Search.cls`, including skipping the BM25 fast path when it is non-empty (FR-001, FR-005)
- [X] T023 [P] [US1] Add `TestTextSearchWhereClauseScopesResults` and `TestHybridSearchWhereClauseScopesResults` to `tests/objectscript/RAG/SDK/Test/SearchTest.cls` (FR-001)
- [X] T024 [US1] Push `Search.cls` to IRIS and run `bash scripts/dev/run-objectscript-tests.sh RAG.SDK.Test.SearchTest`

### ⛔ PHASE GATE: all Phase 4 `SearchTest` methods pass before Phase 5 starts

## Phase 5: User Story 2 — Rows carry their attributes (P1)

**Goal**: a returned row carries the row's metadata, so the caller needs no follow-up query.

**Independent test**: fixture with a known JSON metadata value; search returns it on the hit.

- [X] T025 [US2] Add `TestVectorSearchReturnsMetadata` to `tests/objectscript/RAG/SDK/Test/SearchTest.cls` asserting the hit carries the row's metadata object — must fail before T029 (FR-006)
- [X] T026 [P] [US2] Add `TestVectorSearchNoMetadataColumnOmitsKey` to `tests/objectscript/RAG/SDK/Test/SearchTest.cls` asserting the hit has no `metadata` key and the search still succeeds (FR-006)
- [X] T027 [P] [US2] Add `TestVectorSearchNonJsonMetadataCarriedThrough` to `tests/objectscript/RAG/SDK/Test/SearchTest.cls` asserting unparseable metadata comes back as text rather than failing the search (FR-007)
- [X] T028 [P] [US2] Add `TestAttachTableRecordsMetadataColumn` and `TestSetDefaultTableClearsMetadataColumn` to `tests/objectscript/RAG/SDK/Test/SearchTest.cls` (FR-008)
- [X] T029 [US2] Add the optional `metaCol` argument and the `metadata_col` key to `SetDefaultTable`, `GetDefaultTable`, and `AttachTable` in `iris_src/src/RAG/SDK/Bridge.cls`, validating the column exists when supplied (FR-008)
- [X] T030 [US2] Add `metadata_col` to `GetTableConfig` in `iris_src/src/RAG/SDK/Search.cls`, defaulting to `metadata` only for the built-in default table and to `""` for a configured one (FR-006)
- [X] T031 [US2] Select and return the metadata column in `VectorSearch` and `TextSearch` in `iris_src/src/RAG/SDK/Search.cls`, parsing JSON when possible and carrying the raw text when not (FR-006, FR-007)
- [X] T045 [US2] Add `TestGetVectorDimResolvesDeclaredWidth` to `tests/objectscript/RAG/SDK/Test/SearchTest.cls` — found by T028, whose `dimension` assertion failed because `AttachTable` counted commas in a value read back from the `VECTOR` column and got 1 for every width
- [X] T046 [US2] Add `GetVectorDim` to `iris_src/src/RAG/SDK/Search.cls` behind the same dictionary lookup as `GetVectorType`, and use it for `dimension` in `AttachTable` in `iris_src/src/RAG/SDK/Bridge.cls`
- [X] T032 [US2] Push `Search.cls` and `Bridge.cls` to IRIS and run `bash scripts/dev/run-objectscript-tests.sh RAG.SDK.Test.SearchTest`

### ⛔ PHASE GATE: all Phase 5 `SearchTest` methods pass before Phase 6 starts

## Phase 6: User Story 4 — Hybrid results keep their text (P2)

**Goal**: fusion stops discarding the text both legs already read.

**Independent test**: hybrid search over rows with known text; every fused hit carries its text.

- [X] T033 [US4] Add `TestHybridSearchRrfCarriesText` and `TestHybridSearchLinearCarriesText` to `tests/objectscript/RAG/SDK/Test/SearchTest.cls` — must fail before T035 (FR-011)
- [X] T034 [P] [US4] Add `TestHybridSearchSingleLegRowCarriesText` to `tests/objectscript/RAG/SDK/Test/SearchTest.cls` for a row matched by only one leg (FR-011)
- [X] T035 [US4] Carry text and metadata through `FuseRRF` and `FuseLinear` in `iris_src/src/RAG/SDK/Search.cls`, keeping the fused score semantics unchanged (FR-011)
- [X] T036 [US4] Replace the `^||RRFScores` / `^||LinScores` process-private globals in `iris_src/src/RAG/SDK/Search.cls` with locals, since the fusion data never outlives the method and the guardrails forbid globals where locals suffice
- [X] T037 [US4] Push `Search.cls` to IRIS and run `bash scripts/dev/run-objectscript-tests.sh RAG.SDK.Test.SearchTest`

### ⛔ PHASE GATE: all Phase 6 `SearchTest` methods pass before Phase 7 starts

## Phase 7: Polish & cross-cutting

- [X] T038 Add the four remaining feature-070 `SearchTest` methods (`TestVectorSearchReturnsTopResult`, `TestVectorSearchDimensionDetection`, `TestVectorSearchTopKRespected`, `TestDefaultTableFallback`) to `tests/objectscript/RAG/SDK/Test/SearchTest.cls` so 070's declared coverage genuinely exists (FR-012)
- [X] T039 Uncheck the false `[X]` marks in `specs/070-objectscript-sdk/tasks.md` — nine of them (T004, T014, T016, T018, T019, T020, T020a, T021, T022), with a dated correction table pointing at this feature; T011 stays checked because 085 delivered the class it claimed
- [X] T040 Run the full runnable ObjectScript suite via `bash scripts/dev/run-objectscript-tests.sh RAG.SDK.Test.SchemaTest RAG.SDK.Test.SearchTest` and confirm no regression — `PipelineTest` is excluded because `RAG.SDK.Pipeline` has an embedded-Python method and embedded Python does not compile on this container (research.md finding 6) (FR-012)
- [X] T049 Record the behaviour this feature added or changed in `specs/085-sdk-search-filter/contracts/search-api.md`: `ReadText` and `GetVectorDim` are public, `TextSearch` returns an error object for a scoped failure while still returning `[]` for an unscoped one, `AttachTable` splits schema from table on the last dot, and `dimension` comes from the dictionary
- [X] T041 Run `markdownlint-cli2 --fix` and `prettier --write` over every `.md` in `specs/085-sdk-search-filter/` — both are a no-op here: the repo-local `.markdownlint-cli2.jsonc` and `.prettierignore` exclude `specs/**`, so the gate boundary formats nothing under it
- [X] T042 Run `bash .specify/gates/verify.sh --boundary agent` and `python3 scripts/gates/antipatterns.py`

## Dependencies

```text
Phase 1 (setup) → Phase 2 (fixtures) → Phase 3 (US3, element type)
                                          → Phase 4 (US1, predicate)
                                          → Phase 5 (US2, metadata)
                                          → Phase 6 (US4, fusion)
                                          → Phase 7 (polish)
```

US3 blocks US1 and US2 because their tests need a searchable `FLOAT` fixture. US4 depends on
US2 only for carrying metadata through fusion; its text assertions stand alone.

## Parallel opportunities

Within each phase the `[P]` test tasks touch disjoint methods of the same new test class and
can be written in one pass. The implementation tasks in a phase are sequential — they edit the
same class — and each phase's push-and-run task is its gate.

## Implementation strategy

MVP is Phase 3 alone: it turns a search that fails against every canonically created table into
one that works, which is a shippable fix by itself. Phase 4 is what the consumer actually
blocked on. Phases 5 and 6 remove follow-up queries.
