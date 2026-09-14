# Implementation Plan: Canonical Vector Type, One DDL Owner

**Branch**: `086-canonical-vector-type` | **Date**: 2026-09-14 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/086-canonical-vector-type/spec.md`

## Summary

Two changes that share one cause — DDL written in more than one place.

1. Make every creation path declare the canonical element type (`FLOAT`), make the
   ObjectScript writer resolve the type it is writing into instead of naming a literal,
   and add a `hardcoded-vector-type` detector that baselines the thirty existing literal
   sites so no thirty-first appears. Widen `inline-standard-ddl` to the file types and
   the variable-named DDL it currently cannot see, and drop the exemption it cannot
   reach.
2. Delete `IRISStorage.initialize_schema`'s hand-written document-table DDL, along with
   the unconditional `DROP TABLE`, the `DELETE FROM` fallback and the `self.table_name`
   mutation, and delegate to `SchemaManager.ensure_table_schema` — which takes an
   unqualified name, so the configured `RAG.SourceDocuments` is translated on the way in.

The riskier half is (2): it is reachable from `store_documents` on any transient error
and can empty a populated table. It goes first.

## Technical Context

**Language/Version**: Python 3.11 (package), ObjectScript (`iris_src/src/RAG/SDK/`)
**Primary Dependencies**: `intersystems-irispython` DBAPI; no new dependency
**Storage**: InterSystems IRIS — container `iris-vector-rag-iris`, port `51972`, only
**Testing**: pytest (`tests/unit`, `tests/integration`, `tests/e2e`), plus the gate's own
detector unit tests in `tests/unit/test_antipatterns_detectors.py`
**Target Platform**: Linux/macOS host, IRIS in Docker
**Project Type**: single Python package with an ObjectScript SDK alongside
**Performance Goals**: none new. The delegation path adds one cached
`ensure_table_schema` call in place of a `CREATE`/`DROP` pair, so it is strictly less
work; `_ensure_standard_table` caches per table in `self._tables_validated`.
**Constraints**: no migration of existing tables (spec FR-003); declarations change only
what new tables get. `.cls` edits compile on the server, not on disk.
**Scale/Scope**: 4 declaration sites, 3 ObjectScript writer statements, 1 new detector,
3 fixes to an existing detector, 1 method rewritten, ~30 baseline lines, 2 debt entries.

## Constitution Check

_GATE: must pass before Phase 0 research. Re-checked after design._

| Principle                        | Status | Note                                                                                                                                         |
| -------------------------------- | ------ | -------------------------------------------------------------------------------------------------------------------------------------------- |
| 2 — vector SQL uses `TO_VECTOR`  | pass   | The constitution's example names `FLOAT` (`:64-68`), which is the type this feature standardises on. No statement stops using `TO_VECTOR`.    |
| Contract tests before merge      | pass   | Every phase is test-first; the destructive-path test is written before the method is touched.                                                 |
| Small, composable changes        | pass   | Two independently revertable halves; the detector work does not depend on the storage work except in baseline ordering.                       |
| Backend mode awareness           | pass   | Nothing here opens a connection that did not exist before. Integration tests run with `IRIS_BACKEND_MODE=community`.                          |
| Never log secrets                | n/a    | No new logging of caller data.                                                                                                               |

The constitution's own version header contradicts itself (`docs/DEBT.md` entry 2). Out of
scope here, as recorded there: amending it is its own explicit act.

## Project Structure

### Documentation (this feature)

```text
specs/086-canonical-vector-type/
├── plan.md       # This file
├── spec.md       # Requirements
├── research.md   # Phase 0 — twelve findings read out of the tree
└── tasks.md      # Phase 2 — task breakdown
```

No `data-model.md` or `contracts/`: this feature adds no entity and no new interface. The
one interface it leans on, `SchemaManager.ensure_table_schema`, already exists and is
described in research finding 8.

### Source code touched

```text
iris_src/src/RAG/SDK/
├── Schema.cls                      # GetDefaultDDL: DOUBLE -> FLOAT (2 statements)
└── Pipeline.cls                    # AddDocument: resolve the element type (3 statements)

sql/schema.sql                      # embedding, chunk_embedding -> FLOAT
iris_vector_rag/common/db_init_complete.sql   # chunk_embedding -> FLOAT (:31)

iris_vector_rag/storage/
└── enterprise_storage.py           # initialize_schema delegates; no DDL, no DROP

scripts/gates/
├── antipatterns.py                 # + hardcoded-vector-type; inline-standard-ddl fixes
└── antipatterns-baseline.txt       # regenerated last

tests/
├── unit/test_antipatterns_detectors.py   # both detectors, per blind spot
├── test_enterprise_storage.py            # rewrite the two initialize_schema tests
├── integration/                          # populated table survives a failed probe
└── e2e/                                  # SDK creates, writes, and searches one table
```

**Structure Decision**: existing layout; no new package or module. The new detector lives
beside the nine it joins, split into a pure function plus a walker so it can be
unit-tested against source it is handed (research finding 12).

## Phases

**Phase 1 — US1, stop the data loss.** Test first: a populated table plus a probe that
fails once, asserting the row count afterwards. Then rewrite `initialize_schema` to
translate the configured name, call `ensure_table_schema`, raise on `False`, keep the
index step, and delete the rest. Rewrite the two unit tests that drive the deleted DDL.
Gate: the enterprise storage unit suite and the new integration test pass.

**Phase 2 — US2, one element type.** ObjectScript first: read the guardrail skills, edit
`Schema.cls` and `Pipeline.cls`, push to IRIS, compile there. Then the two `.sql` files.
Gate: an E2E that runs `RAG.SDK.Schema.Initialize`, adds a document through
`RAG.SDK.Pipeline.AddDocument`, searches it through the Python library, and reads the
element type from `%Dictionary.CompiledProperty.Parameters` — plus the same against a
table pre-created as `DOUBLE`, which must still accept a write.

**Phase 3 — US3, the element-type detector.** Unit tests for the statement/prose
distinction first, then the detector, then the baseline. The baseline is written after
phases 1 and 2 so it does not record the lines they delete.

**Phase 4 — US4, the DDL detector's blind spots.** Unit tests per blind spot
(`.sql` file, `.cls` file, variable table name, unreachable exemption), then widen the
file set, add the variable-name report, and correct `DDL_OWNER`.

**Phase 5 — cross-cutting.** Update `docs/DEBT.md` entries 3 and 4, note the deferred (c)
and (d) boundaries against them, run both gates, run the full suite and report the red
baseline honestly against `master`.

## Complexity Tracking

No constitution violation to justify. One scope decision is worth recording here rather
than in a table, because it looks like scope creep and is not:

`Pipeline.cls`'s three `TO_VECTOR(?,DOUBLE,...)` statements are in scope even though the
thirty literal sites are deferred. Leaving them out would ship a `FLOAT` table with a
`DOUBLE` writer — the change would be broken on delivery, not merely incomplete
(research finding 10).
