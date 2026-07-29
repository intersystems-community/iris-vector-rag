# LightRAG ↔ iris-vector-rag: Comprehensive Feature Comparison & Opportunities

**Created**: 2026-07-22
**Purpose**: Source analysis behind Feature 066 (and candidate future specs). This is the full comparison matrix and opportunity assessment that motivated the dual-level retrieval spec.
**Companion to**: [`spec.md`](./spec.md)

---

## 1. What LightRAG is

[LightRAG](https://github.com/hkuds/lightrag) (HKUDS, EMNLP 2025 — *"LightRAG: Simple and Fast Retrieval-Augmented Generation"*) is a knowledge-graph RAG framework positioned as a lighter, cheaper alternative to Microsoft GraphRAG.

**Core thesis**: build a knowledge graph via LLM entity/relation extraction, then serve **dual-level retrieval** — at query time, extract *low-level* keywords (specific entities) and *high-level* keywords (themes/concepts), retrieve **entity embeddings** for the low-level signal and **relation (edge) embeddings** for the high-level signal, and fuse. It deliberately avoids GraphRAG's expensive community reports and multi-hop reasoning to cut LLM calls at both index and query time, and emphasizes **cheap incremental updates** (set-merge new subgraphs, no global rebuild).

**Query modes**:
- `naive` — traditional vector similarity over chunks (no KG).
- `local` — low-level keywords → specific entities + their relationships.
- `global` — high-level keywords → broad themes via relationship-level retrieval.
- `hybrid` — merges `local` + `global`.
- `mix` — merges `local` + `global` + `naive` (default; most comprehensive).

**Other notable capabilities**: pluggable storage (KV / vector / graph / doc-status), optional post-retrieval reranking, role-specific model config (EXTRACT / QUERY / KEYWORDS / VLM), multimodal via RAG-Anything, REST API + WebUI, Ollama-compatible `/api/*` routes, source-attribution/citation, LLM-response caching, incremental delete + entity/relation merge + manual KG editing.

**Sources**: [GitHub](https://github.com/hkuds/lightrag) · [README](https://raw.githubusercontent.com/HKUDS/LightRAG/main/README.md) · [arXiv 2410.05779](https://arxiv.org/html/2410.05779v1) · [DeepWiki](https://deepwiki.com/HKUDS/LightRAG) · [LearnOpenCV walkthrough](https://learnopencv.com/lightrag/)

---

## 2. Comparison matrix

Legend: ✅ full · 🟡 partial/adjacent · ❌ absent

| # | Capability | LightRAG | iris-vector-rag | Gap / opportunity |
|---|---|---|---|---|
| 1 | **Center of gravity** | KG-first (graph is the product) | Vector-first, multi-paradigm; graph is one pipeline | Different bets — iris broader, LightRAG deeper on graph |
| 2 | KG construction (LLM entity/rel extraction) | ✅ core | ✅ `graphrag` + `entity_extraction`, ontology, DSPy module | Parity |
| 3 | **Dual-level retrieval** (low/high keyword extraction → entity vs relation embeddings) | ✅ signature feature | ❌ no query-time keyword split; no relation-embedding theme retrieval | **Top opportunity → Feature 066** |
| 4 | Query modes | ✅ naive/local/global/hybrid/mix | 🟡 `vector`/`text`/`kg`/`hybrid`/`rrf` (method=) — no `global`/`mix` theme mode | Add `global` + `mix` (Feature 066) |
| 5 | Hybrid fusion (RRF) | ✅ | ✅ `iris_graph_core` RRF/score fusion (Feature 065 → query-time) | Parity (improving) |
| 6 | Reranking | ✅ optional post-retrieval | ✅ cross-encoder (`basic_rerank`; 065 → composable `rerank=`) | Parity |
| 7 | Incremental insert (set-merge, no rebuild) | ✅ explicit | 🟡 `memory/` incremental indexing; graph set-merge semantics unclear | Harden graph incremental merge |
| 8 | Doc deletion + KG regeneration | ✅ | ❌ not surfaced | **Opportunity → candidate Feature 067 (KG lifecycle)** |
| 9 | Entity/relation **merge + manual KG editing API** | ✅ create/edit/merge | ❌ | **Opportunity → candidate Feature 067** |
| 10 | Storage architecture | Polyglot: Neo4j/Memgraph + Milvus/Qdrant + Postgres/Mongo (4 store types) | ✅ **single IRIS DB**: vectors+text+graph+KV in one transactional store | **iris advantage** |
| 11 | Role-specific LLM/embedding config (EXTRACT/QUERY/KEYWORDS/VLM) | ✅ | 🟡 `entity_extraction.llm` only | Opportunity (partly folded into Feature 066 KEYWORDS role) |
| 12 | Multimodal (images/tables/formulas) | ✅ RAG-Anything | ❌ | Opportunity (large lift) |
| 13 | Production REST API (auth, rate-limit, audit, streaming) | 🟡 REST + WebUI | ✅ enterprise-grade (API-key, tiered limits, WS, health) | **iris advantage** |
| 14 | Ecosystem integration API | ✅ Ollama-compatible `/api/*` | 🟡 **MCP server** (different ecosystem) | Opportunity: add Ollama-compat shim |
| 15 | Citation / provenance | ✅ source attribution/traceability | 🟡 returns `sources`/metadata | Enrich provenance |
| 16 | Conversation history / LLM cache | ✅ KV-cached | 🟡 `memory/` components | Parity-ish |
| 17 | Retrieval strategies beyond KG | ❌ | ✅ CRAG (corrective), ColBERT late-interaction, multi-query RRF | **iris advantage** |
| 18 | Enterprise (RBAC, transactions, license modes) | ❌ | ✅ RBAC, IRIS transactions, Community/Enterprise backend modes | **iris advantage** |

---

## 3. Opportunities for iris-vector-rag (ranked)

1. **Dual-level / global retrieval** (rows 3–4) — the highest-signal borrow and the subject of **Feature 066** (this spec). Add query-time low/high keyword extraction and a `global`/`mix` retrieval mode that retrieves against **relation embeddings**, not just entity/chunk embeddings. Slots into the Feature 065 `retrieval=` selector. Biggest quality gain on cross-document/thematic queries.
2. **KG lifecycle API** (rows 8–9) — doc deletion with KG regeneration + entity/relation merge + manual KG editing. Strong enterprise differentiator (curation, right-to-be-forgotten). Recommend a dedicated spec (`067-kg-lifecycle`).
3. **Role-specific model config** (row 11) — `EXTRACT`/`QUERY`/`KEYWORDS`/`VLM` roles; use a cheap model for extraction, a strong one for generation. The `KEYWORDS` role is partly pulled into Feature 066 (US3); the broader role system could be its own small config feature.
4. **Ollama-compatible API shim** (row 14) — complements the existing MCP server; makes iris usable from any Ollama client (Open WebUI, etc.). Moderate effort, big ecosystem reach.
5. **Provenance enrichment** (row 15) — richer source attribution/traceability in responses. Low effort.
6. **Multimodal ingestion** (row 12) — RAG-Anything-style parsing of images/tables/formulas. Large lift, lower priority.

---

## 4. Where iris-vector-rag already leads

- **Unified transactional backend** — one IRIS instance does vectors + BM25 text + graph + KV, vs LightRAG's Neo4j+Milvus+Postgres operational burden. Simpler ops, ACID, SQL, HNSW, RBAC/audit.
- **Enterprise API** — API-key auth, tiered rate limiting, audit logging, WebSocket streaming — beyond LightRAG's API surface. Plus an MCP server for agentic clients.
- **Multiple retrieval paradigms** — CRAG's corrective self-evaluation and ColBERT late-interaction have no LightRAG equivalent.
- **Deployment/licensing awareness** — Community/Enterprise backend modes manage the IRIS connection/license envelope.

---

## 5. Recommendation

The single most valuable adoption is **#1 (dual-level / global retrieval)** — LightRAG's core innovation, targeting iris's weakest area (thematic/cross-doc graph retrieval), composing cleanly with the in-flight Feature 065 `retrieval=` work. That is scoped in [`spec.md`](./spec.md). **#2 (KG lifecycle API)** is the strongest enterprise differentiator to spec next.
