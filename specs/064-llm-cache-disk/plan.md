# Implementation Plan: LLM Cache and Connection Optimizations

**Branch**: `064-llm-cache-disk` | **Date**: 2025-12-25 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/064-llm-cache-disk/spec.md`

## Summary

Port high-performance caching, connection bypass, and evaluation logic from the `hipporag2` project into the core `iris-vector-rag` framework. The implementation includes a JSON-based `DiskCacheBackend`, proactive IRIS hardening bypass via Docker, and a standardized multi-hop evaluation package.

## Technical Context

**Language/Version**: Python 3.11+
**Primary Dependencies**: 
- `intersystems-irispython`
- `langchain-core`
- `docker` (soft dependency for connection bypass)
- `datasets` (for evaluation)
- `sentence-transformers`
**Storage**: Local disk (JSON) for cache, InterSystems IRIS for RAG state.
**Testing**: pytest + iris-devtester (NON-NEGOTIABLE)
**Target Platform**: Linux/macOS
**Project Type**: Library core enhancement
**Performance Goals**: 
- Cache hit: < 50ms
- Connection bypass: < 5s
**Constraints**: 
- MUST maintain backward compatibility with existing `memory` and `iris` cache backends.
- MUST use SHA-256 for prompt hashing.
- MUST handle missing `docker` dependency gracefully.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Status | Evidence |
|-----------|--------|----------|
| I. Library-First | PASS | All logic implemented as reusable common/evaluation components. |
| II. .DAT Fixture-First | PASS | Multi-hop dataset tests will use .DAT fixtures where applicable. |
| III. Test-First (TDD) | PASS | Contract tests for Disk Cache are prioritized. |
| IV. Backward Comp. | PASS | New backend is opt-in; connection bypass triggers only on failure. |
| V. IRIS Integration | PASS | Direct IRIS session management used for hardening bypass. |
| VI. Performance | PASS | Sub-10s connection target meets constitution standards. |

## Project Structure

### Documentation (this feature)

```text
specs/064-llm-cache-disk/
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # (Skipped: logic proven in HippoRAG2)
├── data-model.md        # Cache entry and Query models
├── quickstart.md        # Enabling disk cache
└── tasks.md             # Task breakdown
```

### Source Code (repository root)

```text
iris_vector_rag/
├── common/
│   ├── llm_cache_disk.py       # NEW: DiskCacheBackend
│   ├── llm_cache_manager.py    # UPDATED: Register disk backend
│   └── iris_connection.py      # UPDATED: Hardening bypass
└── evaluation/                 # NEW PACKAGE
    ├── __init__.py
    ├── datasets.py             # Ported: Loaders
    └── metrics.py              # Ported: Recall logic
```

**Structure Decision**: Integrated into existing core modules (`common/`) and added a new specialized package (`evaluation/`).
