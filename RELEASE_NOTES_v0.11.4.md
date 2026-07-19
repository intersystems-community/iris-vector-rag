# Release Notes: iris-vector-rag v0.11.4

This release represents a complete rewrite and production hardening of the
framework since the initial OEX listing (1.0.1).

## Package renamed

Previously listed on OEX as `intersystems-iris-rag` (`rag-templates` on PyPI).
Now published as **`iris-vector-rag`** on PyPI with a matching OEX module name.
Importable as `iris_vector_rag`. The `iris_rag/` shim preserves legacy imports.

## Six pipeline strategies, unified interface

All pipelines share one factory call and one response shape — `answer`,
`retrieved_documents`, `contexts`, `sources`, `metadata` — swappable with a
single `create_pipeline(type, ...)` call:

- **BasicRAG** — dense vector retrieval
- **BasicRAGReranking** — retrieval + cross-encoder reranking
- **CRAG** — Corrective RAG with relevance-gated web fallback
- **HybridGraphRAG** — GraphRAG + vector hybrid with entity extraction
- **MultiQueryRRF** — multi-query generation fused via Reciprocal Rank Fusion
- **ColBERT/PyLate (PLAID)** — late-interaction ColBERT with in-DB PLAID index

## ObjectScript SDK (`RAG.SDK.*`)

Five ObjectScript classes callable directly from IRIS without touching Python:

- `RAG.SDK.Pipeline` — run any pipeline strategy by name
- `RAG.SDK.Search` — BM25, IVFFlat, and vector search paths
- `RAG.SDK.Schema` — table initialization, schema status, pip install hook
- `RAG.SDK.Bridge` — overlay support and default table configuration
- `RAG.SDK.Evaluate` — RAGAS evaluation from ObjectScript

## Connection layer overhauled

- All IRIS connections route through `get_iris_connection()` — one path,
  no duplicates.
- Auto-detects embedded runtime via `iris.runtime.get().state` — skips TCP
  entirely when running inside IRIS or when `IRISINSTALLDIR` is set.
- Lazy `import iris.dbapi` in `connection_pool.py` — no `ImportError` at
  import time when `iris` is absent.
- Unified via `iris-embedded-python-wrapper` — handles embedded-kernel,
  embedded-local, and native-remote backends transparently.

## `attach_existing_corpus`

Zero-copy bridge: point any pipeline at tables already in IRIS without
re-ingesting data.

## Validation layer

Pre-flight checks before any pipeline runs: required tables exist, embeddings
are ≥95% non-NULL, IRIS VECTOR format valid. `auto_setup=True` creates missing
tables and embeddings on first use.

## RAGAS evaluation

Side-by-side pipeline comparison with faithfulness, context precision, and
context recall. Uses real PMC biomedical documents — no synthetic data.

## REST API and MCP server

- FastAPI REST API (`[api]` extra) with Redis-backed sessions.
- MCP server (`[mcp]` extra) — all pipelines exposed as MCP tools, usable
  from Claude and other MCP clients.

## `iris-vector-graph` integration

GraphRAG, ColBERT/PLAID, BM25, IVFFlat, and shortestPath delegate to
`iris-vector-graph` — IRIS-native graph and vector operations without leaving
the database.

## IPM/ZPM install

`zpm install iris-vector-rag` now also runs
`pip install iris-vector-rag==0.11.4` via `RAG.SDK.Schema.Install()`.

## CI/CD

- GitHub Actions CI on Python 3.11 and 3.12.
- Release workflow: tests → build → PyPI (OIDC trusted publishing) →
  GitHub Release with changelog notes attached.
