# Phase 0 Research: Composable Query-Time Retrieval

**Feature**: 065-composable-retrieval | **Date**: 2026-07-22

This document resolves the unknowns (U1–U6) enumerated in `plan.md`. Each entry follows Decision / Rationale / Alternatives.

---

## U1 — `text` mode backend: iris-vector-graph BM25

**Decision**: The `text` retrieval mode (and the text side of `hybrid`/`rrf`) is powered by **iris-vector-graph's BM25 text ranking** via `iris_graph_core`, surfaced through the new `RetrievalEngine`. The engine reuses the existing `HybridRetrievalMethods` plumbing (`iris_graph_core` engine + `_hybrid_utils` result→`Document` conversion). The exact BM25 entry point must be confirmed against the installed `iris-vector-graph` ≥2.0.0 API at implementation time; observed adjacent methods today are `iris_engine.kg_TXT(...)` (iFind text), `iris_engine.kg_RRF_FUSE(...)` (RRF), and `fusion_engine.multi_modal_search(fusion_method=..., weights=...)`.

**Rationale**: Clarification session 2026-07-22 selected BM25 explicitly over iFind (`kg_TXT`) and SQL `LIKE`. BM25 gives relevance-ranked keyword scoring comparable to MongoDB `$search`, is IRIS-native through the already-required `iris-vector-graph` package (Principle V), and reuses tested code (Principle IX).

**Alternatives considered**:
- *iFind (`kg_TXT`)* — already wired, but not BM25-ranked; rejected per clarification.
- *SQL `LIKE` substring* — zero-dependency but weakest relevance; rejected.
- *Reuse GraphRAG "enhanced text"* — couples text search to graph assumptions; rejected in favor of the BM25 primitive directly.

**Follow-up**: Implementation task must (a) confirm the BM25 method signature in the pinned `iris-vector-graph`, and (b) if BM25 requires an index that plain `basic` corpora lack, raise the clear prerequisite error mandated by FR-012 rather than silently degrading.

---

## U2 — Metadata filter parameterization + threshold application

**Decision**:
1. **Filtering** flows through `MetadataFilterManager` (key whitelist + scalar-value validation) and MUST use **bound SQL parameters** for values — no string interpolation of user values. A contract test asserts SQL-injection safety (Principle VIII).
2. **`similarity_threshold`** is applied **post-retrieval** as a score cutoff on the returned `(Document, score)` list, after over-fetching is already handled by `top_k`. Documents with score below the threshold are dropped; an all-excluded result returns an empty list with a "no matching documents" metadata flag (not an error).

**Rationale**: `MetadataFilterManager` already exists to prevent injection via a key whitelist and scalar-only values; the missing piece is that `BasicRAGPipeline.query()` never forwards the filter (the bug in FR-001/002). Post-retrieval thresholding is the least-surprising semantics and avoids coupling the threshold into every backend's SQL. Keeping the whitelist + bound params satisfies Principle VIII without new machinery.

**Alternatives considered**:
- *Pre-retrieval threshold pushed into SQL* — backend-specific, harder to guarantee across vector/BM25/fusion; rejected for simplicity.
- *Raise on empty filtered result* — violates least-surprise; rejected.

---

## U3 — Polymorphic `similarity_search` return type (deferred from clarify)

**Decision**: Resolve **additively, without changing existing behavior**. Keep `IRISVectorStore.similarity_search` behaving exactly as today (type-sniffing preserved for back-compat) but:
- Document the current polymorphism explicitly, and
- Add two clearly-named, single-return-type entry points that the new code and docs use: `search_by_text(query, k, filter) -> List[Document]` and `search_by_vector(embedding, top_k, filter) -> List[Tuple[Document, float]]` (thin wrappers over the existing `similarity_search_with_score` / `similarity_search_by_embedding`).

**Rationale**: Principle IV forbids breaking existing callers, so we cannot silently change `similarity_search`'s return type. Additive explicit methods give new code a predictable contract (FR-014) while leaving the old method untouched. Existing internal callers can migrate opportunistically.

**Alternatives considered**:
- *Versioned behavior change / deprecation of polymorphism* — cleaner long-term but breaking; deferred to a future major release with a deprecation warning (Principle IV: 2-release deprecation).

---

## U4 — Reranker cache design

**Decision**: A **process-level cache** in `retrieval/rerank.py` keyed by a tuple of `(strategy_name, model_name, extra_config_hash)`. `resolve_reranker(spec)` maps `True` → default cross-encoder (`cross-encoder/ms-marco-MiniLM-L-6-v2`), a `str` → named strategy, a `callable` → used directly (not cached, caller owns it). Cached entries store the instantiated `CrossEncoder`. Access guarded by a module lock for thread-safety of first-load.

**Rationale**: Today `basic_rerank.py` builds a fresh `CrossEncoder` per query (FR-015 target). A config-keyed cache reuses the model across queries and across pipelines, and correctly separates two different reranker configs. Callables are user-owned, so caching them is unnecessary and risks holding references.

**Alternatives considered**:
- *Instance-level cache on the pipeline* — misses cross-pipeline reuse; a developer creating pipelines per request would still reload. Rejected.
- *LRU with eviction* — YAGNI; the set of reranker configs in a process is tiny. Rejected.

---

## U5 — Full-parity delegation seam (crag / pylate_colbert / graphrag)

**Decision**: Introduce a **`ComposableQueryMixin`** (in `core/`) providing `_normalize_query(**kwargs) -> QueryOptions`, `_run_retrieval(options) -> List[(Document, score)]` (delegates to `RetrievalEngine`), and `_maybe_rerank(options, docs)`. Each pipeline's existing `query()` keeps its own generation/answer logic but routes retrieval and reranking through the mixin **only when composable options are present**; when they are absent, each pipeline takes its existing code path unchanged. Pipelines declare which retrieval modes they natively support via a `supported_retrieval_modes` attribute; requesting an unsupported/unprovisioned mode raises the FR-012 prerequisite error.

**Rationale**: Full parity (clarification) without a risky rewrite of five `query()` methods. The mixin is opt-in per call, preserving each pipeline's default behavior (Principle IV) and avoiding a monolithic base `query()` that would fight `crag`'s corrective loop and `pylate_colbert`'s late-interaction retrieval (Principle IX). Parity means "every pipeline *accepts* the options and either serves them or errors clearly," not "every pipeline reimplements every backend."

**Alternatives considered**:
- *Concrete `query()` in the base class* — would force-refactor bespoke pipelines and risk behavior drift; rejected.
- *Per-pipeline duplication of option handling* — violates DRY and drifts; rejected.

---

## U6 — Native IRIS "text-in" embedding mode

**Decision**: Add an **opt-in** config flag (e.g. `embeddings.mode: native` or `text_in: true`) that, when set and no `embedding_func` is supplied, routes embedding generation to IRIS native EMBEDDING via the store's existing `search_with_embedding` / `query_embedding_config` methods. An explicitly supplied `embedding_func` **always takes precedence** (FR-016). If native EMBEDDING is unavailable on the connected instance, raise a clear prerequisite error naming the requirement.

**Rationale**: `IRISVectorStore` already exposes `search_with_embedding()` and `query_embedding_config()`, so "text-in" is a wiring + precedence problem, not new infrastructure. Keeping it opt-in and precedence-explicit preserves the existing `embedding_func` path (Principle IV) while lowering the getting-started barrier (spec US7), analogous to MongoDB automated embedding.

**Alternatives considered**:
- *Default to native embedding* — would change behavior for existing zero-config users; rejected (must default disabled, Principle IV).

---

## Cross-cutting confirmations

- **Reuse over rebuild**: `_hybrid_utils.convert_*_to_documents`, `HybridRetrievalMethods`, `MetadataFilterManager`, and the base-class `_retrieve_documents_by_vector` helpers are reused; the new code is a thin orchestration layer.
- **Rerank ordering**: rerank runs after retrieval/fusion (spec assumption), matching `$rerank`-after-`$rankFusion`.
- **Observability**: each query logs `{retrieval_mode, weights, rerank_strategy, degraded?}` and instruments retrieval + rerank spans (Principle VII).
- **Performance gate**: a `pytest-benchmark` test asserts <5ms added overhead when no composable options are passed (Principle VI).

**All NEEDS CLARIFICATION resolved.** Ready for Phase 1 design artifacts.
