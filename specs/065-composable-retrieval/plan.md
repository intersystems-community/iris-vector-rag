# Implementation Plan: Composable Query-Time Retrieval (MongoDB-Inspired DevX)

**Branch**: `claude/mongodb-vector-search-devx-ws3v6o` (speckit slot `065-composable-retrieval`) | **Date**: 2026-07-22 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/065-composable-retrieval/spec.md`

## Summary

Bring the Python developer experience to parity with MongoDB's composable vector-search model by making retrieval primitives — filtering, retrieval mode (`vector`/`text`/`hybrid`/`rrf`), fusion weights, and reranking — **query-time options on every registered pipeline**, instead of decisions baked into the pipeline *type*. Also fix the correctness and ergonomics defects that surface on first use (silently dropped `metadata_filter`/`similarity_threshold`, polymorphic `similarity_search` return type, per-query reranker rebuild, broken README import path) and add an optional native "text-in" embedding mode.

**Technical approach**: This is primarily **extraction + unification of existing code**, not greenfield. The building blocks already exist — `HybridRetrievalMethods` and `_hybrid_utils` (vector/text/rrf/fusion via `iris_graph_core`), the base-class `_retrieve_documents_by_vector` helper, `MetadataFilterManager` (parameterized filtering), and the cross-encoder reranker in `basic_rerank.py`. The plan introduces a thin **composable query layer** (a parameter normalizer, a retrieval-mode resolver, and a cached reranker resolver) that pipelines delegate to, so all new behavior is **opt-in and default-disabled** (Constitution Principle IV). When the new options are omitted, each pipeline reproduces its current behavior byte-for-byte.

## Technical Context

**Language/Version**: Python 3.10–3.12 (matches existing codebase)
**Primary Dependencies**: InterSystems IRIS (native vector search); `iris-vector-graph` ≥2.0.0 (`iris_graph_core`) for BM25 text ranking and RRF/score fusion; `sentence-transformers` `CrossEncoder` for reranking; existing `ConfigurationManager`
**Storage**: IRIS `RAG.SourceDocuments` + vector tables; `iris_graph_core` BM25/text index
**Testing**: pytest (`tests/contract/`, `tests/integration/`, `tests/unit/`, `tests/benchmarks/`); `.DAT` fixtures via iris-devtools for ≥10 entities; `pytest-benchmark` for overhead gates
**Target Platform**: Linux server (library consumed in-process)
**Project Type**: Single project (library) — `iris_vector_rag/` package
**Performance Goals**: <5ms added query overhead when composable options are disabled (Principle VI); cross-encoder loaded at most once per config per process; fusion/rerank overhead only paid when explicitly requested
**Constraints**: Zero breaking changes to existing public APIs (Principle IV); config additive-only; all metadata filtering via parameterized SQL (Principle VIII); IRIS-native retrieval only (Principle V)
**Scale/Scope**: 5 registered pipelines (`basic`, `basic_rerank`, `crag`, `graphrag`, `pylate_colbert`) plus `multi_query_rrf`; test corpora 100s–1000s docs, enterprise path to 10K

### Unknowns to resolve in Phase 0 (research.md)

- **U1**: Exact `iris_graph_core` BM25 API for the `text` mode (method name/signature; relationship to existing `kg_TXT`/`kg_RRF_FUSE`).
- **U2**: Whether `MetadataFilterManager`'s current JSON-`LIKE` filtering is parameterized (Principle VIII) and how `similarity_threshold` should be applied (pre- vs post-retrieval) without over-fetching surprises.
- **U3**: Chosen resolution for the polymorphic `IRISVectorStore.similarity_search` return type (deferred from `/speckit.clarify`): additive explicit entry points vs. versioned behavior change.
- **U4**: Reranker cache key design (what config dimensions distinguish two rerankers) and thread-safety expectations.
- **U5**: How to achieve full parity for pipelines with bespoke retrieval (`crag`, `pylate_colbert`, `graphrag`) without breaking their current behavior — delegation seam vs. mixin.
- **U6**: Native IRIS EMBEDDING availability detection for the optional "text-in" mode and its interaction with `embedding_func` precedence.

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

| Principle | Impact | Compliance approach | Gate |
|---|---|---|---|
| I. Library-First | New composable layer is a standalone, testable library component with a programmatic API; no CLI/web coupling. | Ship as `iris_vector_rag` modules with unit-testable pure functions. | ✅ PASS |
| II. .DAT Fixture-First | Integration/E2E tests exercising hybrid/rrf need ≥10 entities. | Use existing `.DAT` fixtures (e.g. `medical-graphrag-20`) via `@pytest.mark.dat_fixture`; no JSON fixtures ≥10 entities. | ✅ PASS |
| III. Test-First (TDD) | New query-time options + bug fixes need contract tests first. | `/speckit.tasks` will order contract tests (red) before implementation (green). Extend existing `tests/contract/test_hybrid_fusion_contract.py`, `test_text_search_contract.py`. | ✅ PASS (enforced in tasks) |
| IV. Backward Compatibility (NON-NEGOTIABLE) | Unifying `query()` and adding options must not break callers. | All new params optional & default to current behavior; `query_text` kept as alias for `query`; `basic_rerank` type retained; `similarity_search` fix is additive (U3). Existing test suite must pass unchanged. | ✅ PASS |
| V. IRIS Integration | Text/fusion must use IRIS-native capabilities. | `text`/`hybrid`/`rrf` use `iris_graph_core` (IRIS-native BM25/fusion); vector uses IRIS native vector search. No non-IRIS backends. | ✅ PASS |
| VI. Performance | Overhead when disabled must be <5ms. | Param normalization is O(1) dict work; retrieval/rerank cost only on explicit opt-in; reranker cached. Benchmark gate in `tests/benchmarks/`. | ✅ PASS (verified by benchmark) |
| VII. Observability | New paths need structured logs + spans. | Log chosen retrieval mode, fusion weights, rerank strategy, and degradation fallbacks; instrument retrieval + rerank steps. | ✅ PASS |
| VIII. Security-First | Metadata filtering must be injection-safe. | Route all filters through `MetadataFilterManager` parameterized path; contract test asserts SQL-injection safety (U2). | ✅ PASS (verify in U2) |
| IX. Simplicity (YAGNI) | Full parity risks over-abstraction. | Reuse existing `HybridRetrievalMethods`/`_hybrid_utils`; one thin delegation seam, not a new framework. Instruction-following rerank and API-surface changes explicitly out of scope. | ✅ PASS |
| X. PyPI Publishing | N/A this feature. | No packaging changes. | ✅ N/A |

**Result**: No violations. No entries required in Complexity Tracking. Full parity (from clarification) is achieved by *reusing* existing retrieval components behind a uniform seam, so it does not introduce new architectural complexity beyond the delegation layer.

## Project Structure

### Documentation (this feature)

```text
specs/065-composable-retrieval/
├── plan.md              # This file
├── research.md          # Phase 0 output
├── data-model.md        # Phase 1 output
├── quickstart.md        # Phase 1 output
├── contracts/           # Phase 1 output (API contracts)
│   ├── query_api.md
│   ├── retrieval_modes.md
│   └── reranker.md
├── checklists/
│   └── requirements.md  # from /speckit.specify
└── tasks.md             # /speckit.tasks output (NOT created here)
```

### Source Code (repository root)

```text
iris_vector_rag/
├── core/
│   ├── base.py                     # RAGPipeline ABC — unchanged contract; pipelines opt into the mixin
│   ├── query_options.py            # NEW: QueryOptions dataclass + normalize_query_params() (query/query_text alias, defaults)
│   ├── composable_query.py         # NEW: ComposableQueryMixin (the delegation seam) — the mixin lives here, NOT in base.py
│   └── vector_store.py             # VectorStore ABC — clarify return-type contract (U3)
├── retrieval/                      # NEW package: the composable retrieval layer
│   ├── __init__.py
│   ├── engine.py                   # RetrievalEngine: resolves mode -> strategy (vector/text/hybrid/rrf)
│   ├── modes.py                    # Mode registry + prerequisite declarations + clear-error raising (FR-012)
│   └── rerank.py                   # Reranker resolver (bool|str|callable) + process-level cache (FR-015)
├── pipelines/                      # Registered types (per __init__.py factory): basic, basic_rerank, crag,
│   │                               #   graphrag→hybrid_graphrag.py, pylate_colbert→colbert_pylate/, multi_query_rrf
│   ├── basic.py                    # Fix FR-001/002 (forward filter + threshold); delegate to composable seam
│   ├── basic_rerank.py             # Reuse cached reranker; becomes thin "basic + rerank=True" convenience
│   ├── crag.py                     # Adopt unified query() signature + composable seam
│   ├── hybrid_graphrag.py          # This IS the registered `graphrag` type (HybridGraphRAGPipeline)
│   ├── colbert_pylate/pylate_pipeline.py  # Registered `pylate_colbert` type (PyLateColBERTPipeline)
│   ├── multi_query_rrf.py
│   ├── _hybrid_utils.py            # Reused for fusion/rrf/text/vector -> Document conversion
│   └── hybrid_graphrag_retrieval.py# Reused; BM25 text path surfaced through RetrievalEngine
│   # NOTE: graphrag.py / graphrag_merged.py / iris_global_graphrag.py are NOT the registered `graphrag`
│   #       type and are OUT OF SCOPE for this feature.
├── storage/
│   ├── vector_store_iris.py        # Fix polymorphic return type (U3); ensure filter+threshold applied
│   └── metadata_filter_manager.py  # Parameterized filtering (Principle VIII)
└── config/
    └── default_config.yaml         # Additive keys: default retrieval mode, rerank defaults, text-in embedding toggle

tests/
├── contract/                       # TDD: query-signature parity, filter-applied, rerank-option, mode-selection, return-type
├── integration/                    # .DAT-fixture-backed hybrid/rrf/rerank end-to-end
├── unit/                           # QueryOptions normalization, mode resolution, reranker cache
└── benchmarks/                     # <5ms disabled-overhead gate; reranker single-load
```

**Structure Decision**: Single-project library layout (existing). One new package `iris_vector_rag/retrieval/` holds the composable layer; one new module `core/query_options.py` holds parameter normalization. Everything else is targeted edits to existing files. This keeps the new surface small (Principle IX) while giving all pipelines a single delegation seam for parity (full-parity clarification).

## Complexity Tracking

*No Constitution violations — section intentionally empty.*
