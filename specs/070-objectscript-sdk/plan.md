# Implementation Plan: ObjectScript SDK for IVR

**Branch**: `070-objectscript-sdk` | **Date**: 2026-04-13 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/070-objectscript-sdk/spec.md`

## Summary

Add 5 ObjectScript classes (`RAG.SDK.*`) that wrap IVR's existing SQL tables with a clean ClassMethod API. Pure ObjectScript + SQL for schema, ingest, and search. Embedded Python only for embedding generation and RAGAS evaluation. Shared `sql/schema.sql` file as single source of truth for DDL between Python IVR and ObjectScript SDK.

## Technical Context

**Language/Version**: ObjectScript (IRIS 2025.1+) + Embedded Python (for 2 methods only)
**Primary Dependencies**: IRIS SQL (VECTOR_COSINE, TO_VECTOR, %CONTAINS), IVG >= 1.27.0 (map_sql_table for Bridge)
**Storage**: Existing RAG.* SQL tables — shared with Python IVR, no new tables
**Testing**: %UnitTest against live IRIS (colbert-bench or dedicated container)
**Target Platform**: IRIS 2025.1+ (all license tiers for core; Embedded Python for AddDocumentWithEmbed/RunRAGAS)
**Project Type**: ObjectScript package (RAG.SDK.*.cls files)
**Performance Goals**: VectorSearch < 100ms for 10K docs (same SQL as Python IVR)
**Constraints**: No direct global access, no $vectorop — pure SQL. Language=python only for AddDocumentWithEmbed and RunRAGAS.
**Scale/Scope**: 5 .cls files, ~13 ClassMethods, < 500 total lines

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|---|---|---|
| P1: IRIS-First Integration Testing | PASS | %UnitTest against live IRIS |
| P2: TO_VECTOR Required | PASS | All vector inserts use TO_VECTOR() via SQL |
| P3: .DAT Fixture-First | N/A | Tests create their own data via AddDocument |
| P4: Test Isolation | PASS | Each test class initializes/drops its own schema |
| P5: Embedding 384 Default | PASS | AddDocumentWithEmbed defaults to all-MiniLM-L6-v2 (384-dim) |
| P6: Config & Secrets | PASS | No credentials in code — uses current IRIS session context |
| P7: Backend Mode Awareness | PASS | Works on any license tier (core is SQL-only) |

All gates pass.

## Project Structure

### Documentation (this feature)

```text
specs/070-objectscript-sdk/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code (repository root)

```text
sql/
└── schema.sql                    # Shared DDL — single source of truth

iris_src/
└── src/
    └── RAG/
        └── SDK/
            ├── Schema.cls        # Initialize, Drop, Status
            ├── Pipeline.cls      # AddDocument, AddDocumentBatch, AddDocumentWithEmbed
            ├── Search.cls        # VectorSearch, TextSearch, HybridSearch
            ├── Bridge.cls        # AttachTable
            └── Evaluate.cls      # RunRAGAS [Language=python]

tests/
└── objectscript/
    └── RAG/
        └── SDK/
            └── Test/
                ├── SchemaTest.cls
                ├── PipelineTest.cls
                ├── SearchTest.cls
                ├── BridgeTest.cls
                └── EvaluateTest.cls
```

**Structure Decision**: ObjectScript classes in `iris_src/src/RAG/SDK/` following IVG's pattern (`iris_src/src/Graph/KG/`). Shared SQL DDL in `sql/schema.sql`. Test classes as `%UnitTest.TestCase` subclasses.
