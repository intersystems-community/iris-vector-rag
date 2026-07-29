# Feature Specification: Dual-Level (Global/Mix) Retrieval — LightRAG-Inspired

**Feature Branch**: `claude/mongodb-vector-search-devx-ws3v6o` (speckit slot: `066-dual-level-retrieval`)
**Created**: 2026-07-22
**Status**: Draft
**Input**: "Research LightRAG and build a comparison matrix on features and opportunities for iris-vector-rag" → adopt LightRAG's dual-level retrieval as the top-ranked opportunity.

> 📊 **Full analysis**: the complete LightRAG ↔ iris-vector-rag feature comparison matrix (18 dimensions), the full ranked opportunity list, and where iris already leads live in the companion doc **[`lightrag-comparison.md`](./lightrag-comparison.md)**. This spec scopes only opportunity #1 (dual-level retrieval); other opportunities (KG lifecycle → candidate `067`, Ollama-compat API, multimodal, role-specific models, provenance) are catalogued there.

## Context

[LightRAG](https://github.com/hkuds/lightrag) (HKUDS, EMNLP 2025) is a knowledge-graph RAG framework whose signature innovation is **dual-level retrieval**: at query time it extracts *low-level* keywords (specific entities) and *high-level* keywords (broad themes/concepts) from the question, then retrieves against **entity embeddings** for the low-level signal and **relation (edge) embeddings** for the high-level signal, and fuses them. This lets it answer both precise entity questions and abstract cross-document/thematic questions, while avoiding the expensive community-report and multi-hop machinery of Microsoft GraphRAG.

A feature comparison against `iris-vector-rag` identified this as the single highest-value gap. Today the `graphrag` pipeline (HybridGraphRAGPipeline) offers `vector`, `text` (BM25), `kg` (graph traversal), `hybrid`, and `rrf` retrieval — all anchored on entity/chunk-level signals. It has **no** query-time low/high keyword split and **no** relation-embedding "theme-level" retrieval, so thematic and cross-document queries under-retrieve.

This feature adds LightRAG-style **`global`** (theme-level) and **`mix`** (comprehensive) retrieval to `iris-vector-rag`, exposed through the query-time `retrieval=` selector introduced by Feature 065 (Composable Query-Time Retrieval). It is additive and backward-compatible.

## Dependencies & Relationship to Feature 065

- **Depends on Feature 065** (composable-retrieval): reuses the `ComposableQueryMixin`, the `RetrievalEngine`/`RetrievalMode` registry, `QueryOptions`, and the `retrieval=` selector. This feature adds two new modes (`global`, `mix`) and their supporting components. If 065 has not landed, the retrieval-mode plumbing is a prerequisite.
- **Reuses** the existing knowledge-graph built by the `graphrag` entity/relation extraction, `iris_graph_core`, and IRIS native vector search.

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Theme-level answers to abstract questions (Priority: P1)

A developer asks a broad, thematic question ("What are the emerging risks discussed across these filings?") that no single entity or chunk answers well. Today `graphrag` retrieves entity/chunk matches and misses the cross-cutting themes. With `retrieval="global"`, the system extracts high-level keywords from the query and retrieves against relationship/theme embeddings, returning documents that collectively cover the theme.

**Why this priority**: This is the core LightRAG capability and the specific weakness in iris today. It delivers standalone value: better answers on thematic/cross-document queries.

**Independent Test**: On a KG-backed corpus, run a thematic query with `retrieval="global"` vs `retrieval="vector"`; the global result surfaces documents connected by relationships/themes that the vector-only result misses, and metadata records the extracted high-level keywords.

**Acceptance Scenarios**:

1. **Given** a KG-backed corpus, **When** a developer calls `query("...thematic question...", retrieval="global")`, **Then** results are retrieved via high-level keyword extraction + relation-level embeddings, and `metadata` records the extracted high-level keywords and that global mode ran.
2. **Given** a query with no extractable high-level keywords, **When** `retrieval="global"`, **Then** the system degrades gracefully (falls back to a defined behavior) and records the degradation rather than erroring.
3. **Given** the KG or relation embeddings are absent, **When** `retrieval="global"`, **Then** the system raises the clear prerequisite error (consistent with Feature 065 FR-012), naming the missing prerequisite.

---

### User Story 2 - Comprehensive "mix" retrieval (Priority: P1)

A developer wants the most complete context and does not want to choose a mode. `retrieval="mix"` combines low-level (entity), high-level (relation/theme), and naive (vector/chunk) retrieval into one fused, ranked result — LightRAG's default and most comprehensive mode.

**Why this priority**: `mix` is LightRAG's default and typically its best-performing mode; it gives iris a strong "just give me the best answer" option and completes the dual-level story.

**Independent Test**: For a query answerable partly by a specific entity and partly by a theme, `retrieval="mix"` returns a superset covering both, out-ranking any single mode on a comprehensiveness measure; metadata records per-source contributions.

**Acceptance Scenarios**:

1. **Given** a KG-backed corpus, **When** a developer calls `query("...", retrieval="mix")`, **Then** results fuse low-level, high-level, and naive retrieval with recorded per-source scores.
2. **Given** optional fusion `weights` (reusing Feature 065), **When** provided, **Then** they bias the mix fusion accordingly.
3. **Given** no `retrieval` argument, **When** a developer queries, **Then** the pipeline's existing default is used (mix is opt-in, not a silent default change) (backward compatible).

---

### User Story 3 - Query-time keyword extraction as a reusable, tunable step (Priority: P2)

The low/high keyword extraction is a distinct LLM step. A developer can inspect the extracted keywords (for debugging/provenance) and control which model performs it, so extraction can use a cheaper/faster model than generation (LightRAG's role-specific model idea for the KEYWORDS role).

**Why this priority**: Makes the new modes debuggable and cost-tunable; supports the global/mix modes but is independently valuable and testable.

**Independent Test**: Run a query in `global`/`mix` mode and confirm the response exposes the extracted low-level and high-level keywords; configure a distinct keyword-extraction model and confirm it is the one invoked for extraction.

**Acceptance Scenarios**:

1. **Given** a `global`/`mix` query, **When** it runs, **Then** `metadata` exposes the extracted `low_level_keywords` and `high_level_keywords`.
2. **Given** a configured keyword-extraction model distinct from the generation model, **When** a query runs, **Then** keyword extraction uses the configured model and generation uses the generation model.
3. **Given** no keyword-extraction model configured, **When** a query runs, **Then** it defaults to the pipeline's existing LLM (backward compatible).

---

### User Story 4 - Relation (edge) embeddings available for retrieval (Priority: P2)

For `global`/`mix` to work, relationship descriptions in the knowledge graph must be embedded and searchable. A developer indexing documents gets relation embeddings generated and stored alongside entity embeddings, and kept in sync as the KG grows.

**Why this priority**: Technical prerequisite for US1/US2, but framed as its own increment because it is independently buildable/testable (index + verify relation embeddings exist and are searchable) before the query modes consume them.

**Independent Test**: After indexing a KG-backed corpus, verify relationship embeddings exist in the store and a nearest-neighbor search over them returns relevant relationships for a theme query.

**Acceptance Scenarios**:

1. **Given** documents are indexed into the `graphrag` pipeline, **When** the KG is built, **Then** relationship descriptions are embedded and stored in an IRIS-native, searchable structure.
2. **Given** new documents are added incrementally, **When** they are indexed, **Then** new relation embeddings are added without rebuilding the entire index.

### Edge Cases

- Query yields low-level keywords but no high-level keywords (or vice versa) → the present level runs, the absent level is skipped, and metadata records which levels contributed.
- Relation-embedding index exists but is empty (KG has entities but no relationships) → `global` degrades to a defined behavior (e.g. entity-level) with a recorded note, or raises the prerequisite error — behavior to be pinned in `/speckit.clarify`.
- `mix` requested on a pipeline without a KG (e.g. `basic`) → clear prerequisite error naming the missing KG (consistent with Feature 065 FR-012).
- Keyword-extraction LLM call fails or times out → defined fallback (e.g. treat the raw query as the keyword set) with recorded degradation, not a hard failure.

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: The system MUST support a `global` retrieval mode that extracts high-level keywords from the query and retrieves using relationship/theme-level embeddings.
- **FR-002**: The system MUST support a `mix` retrieval mode that fuses low-level (entity), high-level (relation/theme), and naive (vector/chunk) retrieval into a single ranked result.
- **FR-003**: Both modes MUST be selectable via the Feature 065 query-time `retrieval=` selector without switching pipeline type, on any pipeline that has a knowledge graph and relation embeddings.
- **FR-004**: The system MUST perform query-time keyword extraction producing distinct low-level and high-level keyword sets.
- **FR-005**: The system MUST expose the extracted low-level and high-level keywords and the per-source contributions in the response `metadata` (provenance).
- **FR-006**: The system MUST allow configuring the model used for keyword extraction independently of the generation model, defaulting to the pipeline's existing LLM when unset.
- **FR-007**: The system MUST generate and store relationship (edge) embeddings in an IRIS-native searchable structure during KG construction, and MUST update them incrementally on new document ingestion without a full rebuild.
- **FR-008**: When a requested mode's prerequisites are absent (no KG, no relation embeddings), the system MUST raise a clear, named prerequisite error rather than silently falling back (consistent with Feature 065 FR-012).
- **FR-009**: When one keyword level is empty or a sub-retrieval fails, the system MUST degrade gracefully, run the available levels, and record the degradation in `metadata` (no hard failure).
- **FR-010**: Omitting `retrieval` MUST reproduce the pipeline's current default behavior; `global`/`mix` are opt-in (backward compatible; new behavior default-disabled per constitution Principle IV).
- **FR-011**: Optional fusion `weights` (Feature 065) MUST bias the `mix` fusion when supplied.
- **FR-012**: All new retrieval MUST use IRIS-native capabilities (native vector search over relation embeddings; `iris_graph_core` graph) — no non-IRIS backends (constitution Principle V).

### Key Entities

- **Query keywords**: The low-level (entity) and high-level (theme) keyword sets extracted from a query, plus the model used and any degradation flags.
- **Relationship embedding**: An embedded representation of a KG edge/relationship description, stored in an IRIS-native searchable structure, keyed to the relationship and its source documents.
- **Retrieval mode (`global`, `mix`)**: New entries in the Feature 065 `RetrievalMode` registry, with declared prerequisites (`knowledge_graph`, `relation_embeddings`) and fusion semantics.
- **Dual-level result**: Retrieved items tagged by contributing level (low/high/naive) with per-level scores, surfaced in metadata.

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: On a benchmark of thematic/cross-document questions, `retrieval="global"` and `retrieval="mix"` measurably improve answer comprehensiveness/coverage versus `retrieval="vector"` on the same corpus (define the metric during planning; target a clear, reported improvement).
- **SC-002**: A developer can select `global` or `mix` by changing a single query argument, with no pipeline-type change, on any KG-backed pipeline.
- **SC-003**: 100% of `global`/`mix` responses expose the extracted low-level and high-level keywords in `metadata`.
- **SC-004**: Keyword extraction can be pointed at a cheaper model, reducing per-query extraction cost without changing the generation model — verified by which model each step invokes.
- **SC-005**: Incremental ingestion adds relation embeddings without a full index rebuild (no full re-embed of the existing corpus).
- **SC-006**: All existing pipeline usage and tests pass unchanged; `global`/`mix` are opt-in (zero backward-compatibility regressions).
- **SC-007**: Requesting `global`/`mix` where prerequisites are absent produces a clear, named error in 100% of cases (no silent fallback).

## Assumptions

- **Builds on Feature 065**: the `retrieval=` selector, `RetrievalEngine`, `RetrievalMode` registry, and `ComposableQueryMixin` exist. If 065 is not yet merged, its retrieval-mode plumbing is a prerequisite of this feature.
- **KG availability**: `global`/`mix` apply to pipelines with a populated knowledge graph (primarily `graphrag`); other pipelines raise the prerequisite error, matching the Feature 065 parity model.
- **Relation embeddings via IRIS**: relationship descriptions are embedded with the same embedding stack as entities/chunks and stored in IRIS-native vector structures (reusing `iris_graph_core` where possible).
- **Keyword extraction is an LLM step**: low/high keyword extraction uses an LLM prompt (LightRAG-style); the exact prompt/format is an implementation detail for planning.
- **No LightRAG dependency**: this adopts LightRAG's *technique*, not its code or storage stack; iris keeps its unified IRIS backend.

## Out of Scope

- Adopting LightRAG's code, storage backends (Neo4j/Milvus/Postgres), or its Ollama-compatible API.
- Knowledge-graph editing / entity-merge / document-deletion lifecycle APIs (a separate high-value opportunity — recommend a distinct spec, e.g. `067-kg-lifecycle`).
- Multimodal ingestion (RAG-Anything-style) — separate, larger effort.
- Microsoft-GraphRAG-style community reports / multi-hop reasoning (LightRAG deliberately avoids these, and so does this feature).

## Next steps (resume locally)

Run in local dev on branch `claude/mongodb-vector-search-devx-ws3v6o`:
```bash
SPECIFY_FEATURE=066-dual-level-retrieval .specify/scripts/bash/check-prerequisites.sh --json
/speckit.clarify      # pin: empty-relation-embedding fallback, comprehensiveness metric, keyword-extraction prompt/format
/speckit.plan
/speckit.tasks
```
Note the same branch reconciliation as Feature 065: keep work on the `claude/...` branch and pass `SPECIFY_FEATURE=066-dual-level-retrieval` to the speckit scripts (they key off branch name).
