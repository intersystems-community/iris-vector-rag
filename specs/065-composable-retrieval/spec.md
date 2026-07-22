# Feature Specification: Composable Query-Time Retrieval (MongoDB-Inspired DevX)

**Feature Branch**: `claude/mongodb-vector-search-devx-ws3v6o` (speckit slot: `065-composable-retrieval`)
**Created**: 2026-07-22
**Status**: Draft
**Input**: User description: "MongoDB-inspired vector search DevX improvements for iris-vector-rag. Make retrieval primitives composable at query time (mirroring MongoDB's $rankFusion / $scoreFusion / $rerank aggregation stages) instead of split across separate pipeline types."

## Context

MongoDB's 2025–2026 vector search releases collapsed the retrieve → fuse → rerank stack into composable query-engine stages: `$rankFusion` / `$scoreFusion` (weighted hybrid fusion), `$rerank` (native cross-encoder reranking), and automated embedding (query with text, no precomputed vector). A developer expresses hybrid-plus-rerank as a single query rather than choosing infrastructure up front.

`iris-vector-rag` already has all of these primitives, but exposes them as separate *pipeline types* (`basic`, `basic_rerank`, `graphrag`) with inconsistent signatures, plus a set of correctness and ergonomics defects that surface on first use. This feature brings the developer experience to parity with MongoDB's composable model **while preserving backward compatibility** with existing pipeline usage.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Filtered search actually filters (Priority: P1)

A developer calls the primary pipeline's `query()` with a metadata filter and a similarity threshold, expecting results to be restricted accordingly. Today, on `BasicRAGPipeline`, both parameters are documented but silently discarded — the developer receives unfiltered results with no error, producing incorrect answers.

**Why this priority**: This is a correctness defect, not a missing feature. It silently returns wrong results for a documented API, which is the most damaging class of DevX failure (the developer trusts output that is wrong). It must be fixed regardless of the larger composability work.

**Independent Test**: Load a corpus with a known metadata field (e.g. `source`), issue the same query with and without `metadata_filter`, and confirm the filtered call returns only matching documents and the threshold excludes low-score results. Fully testable without any other story.

**Acceptance Scenarios**:

1. **Given** a corpus containing documents from sources A and B, **When** the developer calls `query("...", metadata_filter={"source": "A"})`, **Then** every returned document has `source == "A"` and none from B appear.
2. **Given** a `similarity_threshold` of 0.7, **When** the developer queries, **Then** no returned document has a similarity score below 0.7.
3. **Given** a filter key that is not on the allowed list, **When** the developer queries, **Then** the system raises a clear, actionable error rather than silently ignoring the filter.

---

### User Story 2 - One consistent query() across all pipelines (Priority: P1)

A developer swaps `create_pipeline("basic")` for `create_pipeline("crag")` (or any other type) expecting the "swap pipelines with one line" promise to hold. Today the first parameter name (`query` vs `query_text`), the `top_k` default (5 vs 20), and the `include_sources` default (True vs False) differ per pipeline, so identical calling code behaves differently or breaks.

**Why this priority**: It is the foundation that makes every other composable option meaningful — composable retrieval controls are only useful if the query surface is uniform. It is also a low-risk, high-trust change that directly delivers the documented promise.

**Independent Test**: Call every registered pipeline with identical keyword arguments (`query=`, `top_k=`, `generate_answer=`) and confirm all accept them, apply the same defaults, and return the same standardized response shape.

**Acceptance Scenarios**:

1. **Given** any registered pipeline type, **When** the developer calls `query("question", top_k=5)`, **Then** the call succeeds with the same parameter names and defaults across all pipelines.
2. **Given** existing code that used a pipeline-specific parameter name (e.g. `query_text=` on graphrag), **When** it runs after this change, **Then** it still works (backward-compatible alias).
3. **Given** any pipeline, **When** the developer inspects the response, **Then** it contains the same standardized keys (`query`, `answer`, `retrieved_documents`, `contexts`, `metadata`, and `sources` when requested).

---

### User Story 3 - Reranking as a query-time option on any pipeline (Priority: P2)

A developer wants reranked results without committing to a dedicated pipeline type. Instead of choosing `basic_rerank` up front (and bypassing the factory to customize the reranker), they pass a rerank option to `query()` on whatever pipeline they already have.

**Why this priority**: This is the headline MongoDB `$rerank` parity item. It is high value but depends on the unified query surface (US2) being in place first.

**Independent Test**: On a `basic` pipeline, run the same query with rerank off and rerank on, and confirm the result ordering changes and reranked scores are surfaced in metadata; confirm a custom reranker callable is honored.

**Acceptance Scenarios**:

1. **Given** a `basic` pipeline, **When** the developer calls `query("...", rerank=True)`, **Then** results are reordered by a cross-encoder reranker and reranked scores appear in document metadata.
2. **Given** `rerank="cross-encoder"` (named strategy) or `rerank=<callable>`, **When** the developer queries, **Then** the named/custom reranker is applied.
3. **Given** `rerank=False` or omitted, **When** the developer queries, **Then** behavior is identical to today's non-reranked path (backward compatible).
4. **Given** the reranker fails at runtime, **When** the developer queries, **Then** the system falls back to the unranked order and records the degradation in metadata rather than erroring out.

---

### User Story 4 - Hybrid / fusion retrieval as a query-time option (Priority: P2)

A developer selects a retrieval mode (`vector`, `text`, `hybrid`, `rrf`) and optional per-source fusion weights on `query()`, mirroring MongoDB's `$rankFusion`/`$scoreFusion`. Basic vector + text hybrid works without requiring the full knowledge-graph infrastructure.

**Why this priority**: Completes the composable-retrieval parity story alongside reranking. Slightly lower than US3 because it touches more subsystems (text search, fusion) and depends on US2.

**Independent Test**: On a pipeline backed by both vector and text search, run `retrieval="vector"`, `retrieval="text"`, and `retrieval="hybrid"` for the same query and confirm result sets differ and fusion weights shift ranking as expected.

**Acceptance Scenarios**:

1. **Given** a pipeline with vector and text search available, **When** the developer calls `query("...", retrieval="hybrid", weights={"vector": 0.7, "text": 0.3})`, **Then** results are a weighted fusion of both and metadata records the per-source scores.
2. **Given** `retrieval="rrf"`, **When** the developer queries, **Then** results are combined via reciprocal rank fusion.
3. **Given** a `retrieval` mode the current pipeline cannot satisfy (e.g. graph mode without a knowledge graph), **When** the developer queries, **Then** the system raises a clear error naming the missing prerequisite rather than silently falling back.
4. **Given** no `retrieval` argument, **When** the developer queries, **Then** behavior matches today's default for that pipeline (backward compatible).

---

### User Story 5 - Predictable search return type (Priority: P3)

A developer calling the vector store directly gets a consistent, documented return type. Today `IRISVectorStore.similarity_search` returns `List[Document]` or `List[Tuple[Document, float]]` depending on the *type* of the first argument, so downstream code must branch on runtime type.

**Why this priority**: Ergonomics/typing hazard for developers who drop below the pipeline layer. Important but lower blast radius than the pipeline-level stories.

**Independent Test**: Call the search API by text and by embedding vector and confirm each documented entry point returns a single, predictable, documented shape.

**Acceptance Scenarios**:

1. **Given** a text query, **When** the developer calls the text search entry point, **Then** it returns one documented shape regardless of arguments.
2. **Given** an embedding vector, **When** the developer calls the vector search entry point, **Then** it returns one documented shape.
3. **Given** existing code relying on the current behavior, **When** it runs, **Then** it continues to work (the change is additive/clarifying, not silently breaking).

---

### User Story 6 - Reranker is not rebuilt on every query (Priority: P3)

A developer running many reranked queries does not pay model-load cost per call. Today the cross-encoder is instantiated fresh on every query.

**Why this priority**: Pure performance/cost win with no API change; valuable but not blocking correctness or ergonomics.

**Independent Test**: Issue N reranked queries and confirm the reranker model is loaded once (measurable via load count/time), with per-query latency dropping to steady state after the first.

**Acceptance Scenarios**:

1. **Given** repeated reranked queries in one process, **When** they execute, **Then** the reranker model is initialized at most once per configuration and reused.
2. **Given** two different reranker configurations, **When** both are used, **Then** each is cached independently.

---

### User Story 7 - Zero-config "text-in" embedding mode (Priority: P3)

A developer can run basic semantic search without wiring an external `embedding_func`, using IRIS's native EMBEDDING capability — analogous to MongoDB's automated embedding where you query with text.

**Why this priority**: Meaningfully lowers the getting-started barrier, but is optional (existing `embedding_func` path remains) and depends on IRIS native embedding availability.

**Independent Test**: Configure a pipeline in text-in mode with no `embedding_func`, load documents and query with plain text, and confirm semantic results return.

**Acceptance Scenarios**:

1. **Given** native embedding is enabled and no `embedding_func` is supplied, **When** the developer loads documents and queries with text, **Then** semantic search works end to end.
2. **Given** an explicit `embedding_func` is supplied, **When** the developer queries, **Then** the explicit function takes precedence (backward compatible).

---

### User Story 8 - Documentation works on first copy-paste (Priority: P3)

A new developer copies the README quickstart and it runs. Today the README imports `from iris_rag...` but the installed package is `iris_vector_rag`, so the first example raises `ModuleNotFoundError`.

**Why this priority**: Trivial to fix but it is the literal first impression; a broken quickstart undermines trust in everything else. Low priority only because it is isolated and non-behavioral.

**Independent Test**: Execute every import statement in the README quickstart against a clean install and confirm none raise `ModuleNotFoundError`.

**Acceptance Scenarios**:

1. **Given** a clean install of the package, **When** a developer runs the README quickstart imports verbatim, **Then** all imports resolve successfully.
2. **Given** the package's own docstrings and top-level comments, **When** they reference the package name, **Then** they use the actual importable name.

### Edge Cases

- What happens when `rerank=True` is combined with `retrieval="rrf"`? (Reranking is applied *after* fusion, matching MongoDB's `$rerank`-after-`$rankFusion` ordering.)
- What happens when fusion `weights` reference a source the chosen `retrieval` mode does not produce? (System must validate and error clearly.)
- What happens when `metadata_filter` and `similarity_threshold` are both set and jointly exclude all documents? (Return an empty result set with a "no matching documents" indication, not an error.)
- What happens when a developer passes both a legacy parameter name and the canonical one (e.g. `query=` and `query_text=`) in the same call? (Define a deterministic precedence and, ideally, warn.)
- What happens when text-in embedding mode is requested but native embedding is unavailable in the connected IRIS instance? (Clear error naming the unmet prerequisite.)

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The primary pipeline's `query()` MUST apply a supplied `metadata_filter` to retrieval so that only documents matching the filter are returned.
- **FR-002**: The primary pipeline's `query()` MUST apply a supplied `similarity_threshold` so that documents below the threshold are excluded.
- **FR-003**: When a filter key is not permitted, the system MUST raise a clear, actionable error rather than ignoring the filter.
- **FR-004**: All registered pipelines MUST accept a consistent set of core query parameters with consistent names (canonical query parameter, `top_k`, `generate_answer`, `include_sources`) and consistent default values.
- **FR-005**: The system MUST preserve backward compatibility for existing pipeline-specific parameter names via aliases, without breaking existing caller code.
- **FR-006**: All pipelines MUST return the same standardized response structure (`query`, `answer`, `retrieved_documents`, `contexts`, `metadata`, and `sources` when requested).
- **FR-007**: `query()` MUST accept a rerank option expressible as a boolean, a named strategy, or a caller-supplied callable, applicable on any pipeline regardless of type.
- **FR-008**: Reranking MUST be applied after retrieval/fusion, and reranked scores MUST be surfaced in document metadata.
- **FR-009**: When reranking fails at runtime, the system MUST fall back to the pre-rerank ordering and record the degradation, rather than failing the query.
- **FR-010**: `query()` MUST accept a retrieval-mode selector supporting at least `vector`, `text`, `hybrid`, and `rrf`, applicable without requiring the developer to switch pipeline types.
- **FR-011**: `query()` MUST accept optional per-source fusion weights that influence hybrid/fusion ranking, and MUST record per-source scores in metadata.
- **FR-012**: When a requested retrieval mode's prerequisites are unmet (e.g. no knowledge graph for graph mode), the system MUST raise a clear error naming the missing prerequisite instead of silently substituting a different mode.
- **FR-013**: Omitting the new composable options (rerank, retrieval mode, weights) MUST reproduce each pipeline's current default behavior (backward compatible).
- **FR-014**: The vector store search API MUST expose a predictable, documented return type per entry point, so callers do not need to branch on runtime argument type.
- **FR-015**: The cross-encoder reranker MUST be initialized at most once per distinct reranker configuration per process and reused across queries.
- **FR-016**: The system MUST support an optional "text-in" mode where semantic search works without a caller-supplied embedding function, using IRIS native embedding; an explicitly supplied embedding function MUST take precedence.
- **FR-017**: All import examples and package self-references in the README and top-level package documentation MUST use the actual importable package name and resolve on a clean install.

### Key Entities

- **Query request**: The developer-facing inputs to `query()` — canonical query text, `top_k`, `generate_answer`, `include_sources`, `metadata_filter`, `similarity_threshold`, `rerank`, `retrieval` mode, and fusion `weights`.
- **Retrieval result**: A retrieved document plus its scores (vector score, text score, fusion score, rerank score) exposed consistently in metadata.
- **Reranker strategy**: A boolean/named/callable specification resolving to a reranking implementation, with cached instances keyed by configuration.
- **Retrieval mode**: A named strategy (`vector`, `text`, `hybrid`, `rrf`, and pipeline-specific extensions) with declared prerequisites.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: 100% of `query()` calls that supply a metadata filter return only documents matching that filter (0% leakage of non-matching documents).
- **SC-002**: A developer can switch between any two registered pipeline types by changing only the pipeline name string, with no other code changes required, for the core query parameters — verified across 100% of registered pipelines.
- **SC-003**: A developer can enable reranking on the primary pipeline by adding a single query argument, without changing pipeline type or bypassing the factory.
- **SC-004**: A developer can perform hybrid/fusion retrieval by adding a single query argument (plus optional weights), without switching pipeline type.
- **SC-005**: Repeated reranked queries in a single process load the reranker model at most once, reducing steady-state per-query reranking overhead to near zero model-load cost.
- **SC-006**: 100% of README quickstart import statements execute successfully on a clean install.
- **SC-007**: All existing pipeline usages and tests continue to pass unchanged (zero backward-compatibility regressions).
- **SC-008**: A developer can run basic semantic search with zero external embedding configuration when native embedding is available.

## Assumptions

- **Backward compatibility is additive**: Existing pipeline types (`basic`, `basic_rerank`, `graphrag`, etc.) remain available. The composable options are added to the shared query surface; `basic_rerank` continues to work as a convenience alias for "basic + rerank".
- **Canonical query parameter**: The unified first parameter is standardized to a single canonical name, with the alternate name retained as a backward-compatible alias. (Exact canonical choice deferred to `/speckit.clarify` or `/speckit.plan`.)
- **Hybrid without graph**: Query-time `hybrid`/`rrf` modes fuse vector + native IRIS text search and do NOT require the knowledge-graph / `iris_graph_core` infrastructure. Graph-based retrieval remains available only where a knowledge graph exists (e.g. graphrag).
- **Rerank ordering**: Reranking is applied after retrieval and any fusion step, matching MongoDB's `$rerank`-after-`$rankFusion` model.
- **Return-type fix is non-breaking**: Predictable return types are achieved by documenting/adding explicit entry points rather than silently changing the existing polymorphic method's behavior for current callers.
- **Native embedding availability**: "Text-in" mode depends on the connected IRIS instance supporting native EMBEDDING; where unavailable, the developer supplies an embedding function as today.

## Out of Scope

- Adopting MongoDB / Voyage AI models or services directly (this feature borrows the *ergonomics*, not the vendor stack).
- Instruction-following rerankers (natural-language rerank instructions) — a possible future enhancement, not required here.
- Automatic embedding synchronization on document mutation beyond what IRIS native embedding already provides.
- Changes to the REST API surface (this feature targets the Python developer experience; API alignment can follow).
