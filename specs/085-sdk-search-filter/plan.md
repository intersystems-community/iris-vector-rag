# Implementation Plan: RAG.SDK search usable by an external ObjectScript consumer

**Branch**: `085-sdk-search-filter` | **Date**: 2026-09-14 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/085-sdk-search-filter/spec.md`

## Summary

Make `RAG.SDK.Search` answerable by a consumer that owns its own scoping: accept an optional
SQL predicate, carry each row's metadata back with it, resolve the vector element type from
the searched table instead of hardcoding `DOUBLE`, and stop dropping text in hybrid fusion.
All four changes land under one new `%UnitTest` class, because the class being changed has no
ObjectScript test coverage today.

## Technical Context

**Language/Version**: ObjectScript (IRIS 2025.x), classes under `iris_src/src/RAG/SDK/`
**Primary Dependencies**: `%SQL.Statement`, `%DynamicObject`/`%DynamicArray`,
`%Dictionary.CompiledProperty`, `INFORMATION_SCHEMA.TABLES`; `Graph.KG.IVFIndex` /
`Graph.KG.BM25Index` on the optional fast paths
**Storage**: IRIS SQL tables, `VECTOR(FLOAT|DOUBLE, N)` embedding columns; configuration in
`^RAG.SDK.Config`
**Testing**: `%UnitTest.TestCase` classes under `tests/objectscript/RAG/SDK/Test/`, run on
`iris-vector-rag-iris` (port 51972, namespace `USER`)
**Target Platform**: IRIS server; the caller is ObjectScript in another namespace/repository
**Project Type**: single project — ObjectScript SDK inside a Python package
**Performance Goals**: one catalog read per search to resolve element type; no additional
round trip per returned row (the point of carrying metadata)
**Constraints**: no source mounted into the container, so classes are `docker cp`-ed and
`LoadDir`-compiled before tests run; predicate must not be able to introduce a second
statement
**Scale/Scope**: two production classes changed (`Search.cls`, `Bridge.cls`), one test class
added, one prior feature's task marks corrected

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-checked after Phase 1 design.*

| Principle | Status | How this feature satisfies it |
| --------- | ------ | ----------------------------- |
| 1 — IRIS-First Integration Testing | PASS | Every test is a `%UnitTest` class executing SQL against `iris-vector-rag-iris`. No mocks exist for this layer and none are introduced. Container checked before implementation, per `.specify/scripts/bash/check-container.sh`. |
| 2 — VECTOR client limitation (`TO_VECTOR` required) | PASS | Search continues to pass the query vector through `TO_VECTOR(?, <type>, dim)`; the change is that `<type>` is resolved rather than assumed. Test fixtures insert vectors through `TO_VECTOR` as well. |
| 3 — `.DAT` fixture-first | N/A | Fixtures here are three to ten rows created inline by the test class. A `.DAT` fixture would be slower to maintain than the rows it replaces and cannot express "this table has no metadata column", which is one of the cases under test. |
| 4 — Test isolation by database state | PASS | Each test creates its own fixture table with a test-specific name and drops it in `TearDown`, so no test depends on another's rows or on the shared `RAG.SourceDocuments`. |
| 5 — Embedding generation standards | PASS | 384 dimensions throughout, matching the project default. Vectors are deterministic literals, not generated — the assertions are about ordering and scoping, not about embedding quality. |
| 6 — Configuration & secrets hygiene | PASS | Configuration read from `^RAG.SDK.Config` with explicit defaults; no secrets involved. Errors name the table and column rather than echoing raw SQLCODE text alone. |
| 7 — Backend mode awareness | PASS | Single-connection safe: every test uses one `%SQL.Statement` at a time on the server side, so it runs identically under Community Edition. |

**Repository rules also treated as gates** (from `CLAUDE.md`, which the constitution does not
yet restate — see Complexity Tracking): test-first with an E2E gate per user story;
`SELECT TOP n` not `FETCH FIRST`; one container only; `objectscript-guardrails` and
`objectscript-review` checklists before any `.cls` edit.

**Post-design re-check**: no violation introduced. The design adds one catalog read per
search and no new dependency.

## Project Structure

### Documentation (this feature)

```text
specs/085-sdk-search-filter/
├── spec.md
├── plan.md              # This file
├── research.md          # Phase 0 — verified findings from the live instance
├── data-model.md        # Phase 1 — hit shape, table config, predicate
├── quickstart.md        # Phase 1 — how a consumer calls it, how to run the tests
├── contracts/
│   └── search-api.md    # Phase 1 — method signatures and return shapes
├── checklists/
│   └── requirements.md
└── tasks.md             # Phase 2 (/speckit.tasks)
```

### Source Code (repository root)

```text
iris_src/src/RAG/SDK/
├── Search.cls           # CHANGED: predicate argument, metadata in rows, resolved
│                        #          element type, fusion carries text
├── Bridge.cls           # CHANGED: metadata column in attach + default-table config
├── Pipeline.cls         # unchanged
├── Schema.cls           # unchanged (see spec Assumptions)
└── Evaluate.cls         # unchanged

tests/objectscript/RAG/SDK/Test/
├── SearchTest.cls       # NEW: the class feature 070 marked done and never wrote
├── SchemaTest.cls       # unchanged
└── PipelineTest.cls     # unchanged

specs/070-objectscript-sdk/tasks.md   # CORRECTED: false [X] marks unchecked
```

**Structure Decision**: the SDK's ObjectScript lives in `iris_src/src/RAG/SDK/` and its
`%UnitTest` classes in `tests/objectscript/RAG/SDK/Test/`, mirroring the package path. This
feature adds no directories.

## Phases

**Phase 1 — Foundational**: resolve the vector element type (US3). Nothing else can be tested
against a `FLOAT` table until search returns rows at all, so this is the blocking
prerequisite rather than a peer story.

**Phase 2 — Scoped search (US1)**: the predicate argument, the fragment shape check, and
skipping the IVF path when a predicate is present.

**Phase 3 — Rows carry attributes (US2)**: metadata column in the config surface, metadata in
the returned rows.

**Phase 4 — Hybrid keeps text (US4)**: fusion carries text under both strategies.

**Phase 5 — Polish**: correct feature 070's task marks, document the consumer-facing contract.

Each phase: its `SearchTest` methods are written and observed failing before the `Search.cls`
or `Bridge.cls` change, and the whole `SearchTest` class must pass before the next phase
starts.

## Complexity Tracking

| Violation | Why Needed | Simpler Alternative Rejected Because |
| --------- | ---------- | ------------------------------------ |
| Constitution does not cover the rules this feature is actually gated on (test-first, one container, quality gates), and its own version header (1.2.0, "added Principle XI") disagrees with its body (Principles 1–7) and footer (1.0.0) | The gates above had to be stated from `CLAUDE.md` instead of cited from the constitution | Amending the constitution mid-feature would change the authority this feature is being judged against while it is being judged. Recorded here; reconciling it is its own change. |
| Caller-supplied SQL fragment is not parameterised | The consumer composes a predicate over its own columns; there are no caller values for the SDK to bind | A metadata-key filter API would require the consumer to re-express as JSON what it already holds as SQL, and could not reach non-metadata columns such as `expires_at` |
