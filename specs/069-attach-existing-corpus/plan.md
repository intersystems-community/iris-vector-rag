# Implementation Plan: Attach Existing Corpus

**Branch**: `069-attach-existing-corpus` | **Date**: 2026-04-03 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/069-attach-existing-corpus/spec.md`

## Summary

Add `attach_existing_corpus()` method to `HybridGraphRAGPipeline` that creates a zero-copy bridge between any existing IRIS SQL table (with a VECTOR column) and IVG graph nodes + vector search. The method wraps three existing IVG engine calls (`validate_vector_table`, `map_sql_table`, dimension storage) into a single user-facing API. No data copy, no re-embedding, no new tables.

## Technical Context

**Language/Version**: Python 3.12 + ObjectScript (IRIS SQL)
**Primary Dependencies**: iris-vector-graph >= 1.27.0 (`map_sql_table`, `validate_vector_table`, `vector_search`)
**Storage**: Existing IRIS SQL tables (RAG.SourceDocuments, custom tables) — zero new tables created
**Testing**: pytest against live IRIS (colbert-bench container, port 1972)
**Target Platform**: IRIS 2025.1+ (any license tier)
**Project Type**: Single Python package (iris_vector_rag)
**Performance Goals**: < 2 seconds for attach (metadata-only, O(1) regardless of table size)
**Constraints**: Must not copy data, create tables, or re-compute embeddings
**Scale/Scope**: Acceptance testing at 10K rows; O(1) operation so scale doesn't matter

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Notes |
|---|---|---|
| P1: IRIS-First Integration Testing | PASS | Tests run against live IRIS via iris-devtester |
| P2: TO_VECTOR Required | PASS | No VECTOR inserts — we only READ existing vectors |
| P3: .DAT Fixture-First | N/A | No fixture data needed — test creates its own table |
| P4: Test Isolation | PASS | Each test creates/drops its own table |
| P5: Embedding 384 Default | PASS | Dimension auto-detected from existing data, not hardcoded |
| P6: Config & Secrets | PASS | No new credentials; uses existing connection_manager |
| P7: Backend Mode Awareness | PASS | Works on any IRIS license tier (no HNSW dependency) |

All gates pass. No violations.

## Project Structure

### Documentation (this feature)

```text
specs/069-attach-existing-corpus/
├── plan.md              # This file
├── research.md          # Phase 0 output (no unknowns — minimal)
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output
└── tasks.md             # Phase 2 output (created by /speckit.tasks)
```

### Source Code (repository root)

```text
iris_vector_rag/
└── pipelines/
    └── hybrid_graphrag.py          # Add attach_existing_corpus() method

tests/
└── test_attach_corpus.py           # Unit + integration tests
```

**Structure Decision**: Single method addition to existing `HybridGraphRAGPipeline` class. No new modules, packages, or directories needed.
