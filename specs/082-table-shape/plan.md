# Implementation Plan: Table Shape

**Branch**: `082-table-shape` | **Date**: 2026-09-10 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/082-table-shape/spec.md`

## Summary

One module owns the physical shape of the document and chunk tables. A frozen
`TableShape` value object answers every "which column holds the identifier / the
text / the metadata / the embedding, and what type is it" question; a resolver
produces one per table per session from a single catalog read and caches it; a
catalog seam lets tests supply recorded shapes instead of scripting fake cursor
responses. Every call site that today re-derives column names — with two
contradictory precedence rules — reads the shape instead.

## Technical Context

**Language/Version**: Python 3.11 (support 3.10+; CI runs 3.11 and 3.12)
**Primary Dependencies**: `intersystems-irispython` (DBAPI), no new runtime dependency
**Storage**: InterSystems IRIS 2026.1 (`iris-vector-rag-iris`, port 51972 locally)
**Testing**: pytest; unit tests with a recorded catalog (no database), integration and
E2E against the live container
**Target Platform**: Linux/macOS server-side library
**Project Type**: single Python package (`iris_vector_rag/`)
**Performance Goals**: one catalog round trip per table per session; a search issues
one ranked query plus one metadata fetch and no catalog query
**Constraints**: additive only — 0.15 changes no public signature; Community Edition
runs with a single connection, so round trips are the scarce resource
**Scale/Scope**: two tables in scope (`RAG.SourceDocuments`, `RAG.DocumentChunks`),
six observed shapes, ~11 call sites to migrate

### Observed state (measured 2026-09-10)

| Site | What it re-derives | Precedence |
| --- | --- | --- |
| `storage/vector_store_iris.py:701-715` (write) | id, text, extra columns | `id` before `doc_id`; `text_content` before `content` |
| `storage/vector_store_iris.py:1100-1120` (read) | id, text | `id` before `doc_id`; **`content` before `text_content`** |
| `storage/vector_store_iris.py:925-934` (`_get_id_column`) | id | `id` before `doc_id` |
| `storage/vector_store_iris.py:151-179` (`_get_vector_data_type`) | vector type | `%Dictionary.CompiledProperty`, then `IRIS_VECTOR_DATA_TYPE`, then `FLOAT` |
| `storage/vector_store_iris.py:981-1017` (`_metadata_filter_column_expr`) | metadata column type | `INFORMATION_SCHEMA.COLUMNS.DATA_TYPE` |
| `storage/vector_store_iris.py:1134-1135` | id, text | `schema_config` overrides — configuration, not the database |
| `storage/vector_store_iris.py:1452-1500`, `1551-1590` | id, text | try canonical SQL, catch, retry legacy SQL |
| `services/storage.py:130-146` | id | `id` before `doc_id` |
| `validation/orchestrator.py:588-617` | id, text | `doc_id` before `id`; own content precedence |
| `pipelines/hybrid_graphrag.py:180-258` | index presence, embedding column | own catalog read |
| `storage/schema_manager.py:1887-1890`, `2255-2258` | columns for alter/verify | own catalog reads |

Two facts fall straight out of the table: the reader and the writer disagree about
the text column, which is exactly the mixed-table defect; and a search currently
pays three to five catalog round trips before its first real query.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Note |
| --- | --- | --- |
| 1. IRIS-first integration testing | Pass | Six-shape suite runs against the live container; unit tests use a recorded catalog and are additional, not a substitute |
| 2. VECTOR needs `TO_VECTOR` | Pass | The shape supplies the type and dimension that `TO_VECTOR` needs; no direct VECTOR insert is introduced |
| 3. `.DAT` fixture-first | Pass | Shape fixtures are DDL, not corpora; document loads keep their existing fixtures |
| 4. Test isolation by database state | Pass | Each shape is created in its own table name and dropped in teardown |
| 5. Embedding standards (384 default) | Pass | Dimension comes from the table, never from a default; the 384 default stays in configuration for creation |
| 6. Config & secrets hygiene | Pass | Removes `IRIS_VECTOR_DATA_TYPE` as a silent fallback (FR-001); no secret handling |
| 7. Backend mode awareness | Pass | Fewer round trips is strictly better on Community Edition; one shape cache per session, no shared global |
| Interfaces stable, breaking changes documented | Pass | Additive in 0.15; `schema_config` column overrides become deprecated no-ops with a warning |

No violations. Complexity Tracking is therefore empty and omitted.

## Project Structure

### Documentation (this feature)

```text
specs/082-table-shape/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/
│   └── table-shape.md   # Phase 1 output: the module's interface contract
├── checklists/
│   └── requirements.md  # Spec quality checklist (complete)
└── tasks.md             # Phase 2 output (/speckit.tasks)
```

### Source Code (repository root)

```text
iris_vector_rag/
├── storage/
│   ├── table_shape.py          # NEW: TableShape, roles, canonical shapes, errors
│   ├── catalog.py              # NEW: CatalogReader protocol, IRISCatalogReader,
│   │                           #      RecordedCatalog (test seam)
│   ├── shape_resolver.py       # NEW: TableShapeResolver (per-session cache)
│   ├── vector_store_iris.py    # MIGRATE: all column derivation -> shape
│   ├── schema_manager.py       # MIGRATE: canonical DDL emitted from the shape;
│   │                           #          its own catalog reads go through catalog.py
│   └── constants.py            # MIGRATE: canonical column names live with the shape
├── services/storage.py         # MIGRATE: drop its own id derivation
├── validation/orchestrator.py  # MIGRATE: drop its own id/text derivation
└── pipelines/
    ├── hybrid_graphrag.py      # MIGRATE: embedding column + index probe via shape
    ├── graphrag.py             # MIGRATE: document-table join uses shape id column
    └── graphrag_merged.py      # MIGRATE: same join

tests/
├── unit/
│   ├── test_table_shape.py            # NEW: role resolution, precedence, errors
│   ├── test_shape_resolver.py         # NEW: cache, invalidate, refresh
│   └── test_vector_store_shape.py     # NEW: store behaviour on recorded shapes
├── integration/
│   └── test_table_shape_six_shapes.py # NEW: real DDL for each shape
└── e2e/
    └── test_table_shape_e2e.py        # NEW: phase-gate E2E, one per user story

scripts/gates/antipatterns.py          # NEW detector: catalog-read-outside-shape
```

**Structure Decision**: single package, three new modules inside
`iris_vector_rag/storage/`. The shape owner sits in `storage/` because it describes
storage and has no pipeline knowledge; the catalog seam is its own module so the
recorded implementation can be imported by tests without importing a driver.

## Phasing

Phases match the spec's user stories, in priority order. Each phase is test-first
and gated by its own E2E test.

1. **Foundational** — `table_shape.py`, `catalog.py`, `shape_resolver.py` with unit
   tests. No call site changes. Gate: unit suite green, no database needed.
2. **US1 (P1)** — migrate `vector_store_iris.py` read, write and delete paths to the
   shape; six-shape integration suite. Gate: `test_table_shape_e2e.py::test_us1`.
3. **US2 (P2)** — one resolution per session: remove per-call catalog reads, add
   round-trip counting. Gate: `test_us2`.
4. **US3 (P2)** — creation always canonical: `schema_manager` emits DDL from the
   canonical shape; refuse non-canonical creation; no implicit alter. Gate: `test_us3`.
5. **US4 (P3)** — port shape-dependent unit tests to recorded shapes; delete the
   simulated catalog responses; add the antipatterns detector for SC-003.
   Gate: `test_us4` plus `python3 scripts/gates/antipatterns.py`.
6. **Polish** — migrate the remaining call sites (`services/storage.py`,
   `validation/orchestrator.py`, the three graph pipelines), deprecate
   `schema_config` column overrides, changelog.

## Risks

- **The mixed-shape read/write flip changes behaviour on a real database.** A table
  holding both `content` and `text_content` currently reads one and writes the other;
  after this change both use `text_content`. Rows written by earlier releases may hold
  their text in `content` only. Mitigation: on a mixed table, read `text_content` and
  fall back per row to the legacy column when the canonical one is null; this is a
  documented behaviour in the contract, not a precedence rule.
- **Deselected order-dependent contract test.** `test_graphrag_handler_execute_hybrid_search`
  fails in the full suite because earlier tests leave a legacy-shaped table. This
  feature should make it pass; the deselect entry is removed only after a full
  CI-equivalent run, never from an isolated run.
- **`schema_manager.py` is 2,566 lines** and carries the alter paths. Scope is limited
  to making it emit and read shapes; restructuring it belongs to feature 085.
