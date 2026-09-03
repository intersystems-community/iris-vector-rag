# Feature 062: IRISGraphRAGPipeline.delete_node

## Overview

Add `HybridGraphRAGPipeline.delete_node(node_id: str) -> None` as the symmetric teardown
counterpart to `index_node()`. The method removes a node from all stores managed by the
pipeline: knowledge graph tables, vector document store, and BM25 index (skipped; no
per-document delete API in iris_vector_graph 2.3.1).

## Motivation

Test teardown in opsreview (`test_021_us6b_e2e.py:123`) reaches into three private internals
to clean up what a single `index_node()` call wrote. This causes ranking pollution and flaky
cross-tenant test failures when the vector store is not cleaned between runs.

## Functional Requirements

- FR-001: `delete_node(node_id)` removes the node from `iris_engine` (Graph_KG tables).
- FR-002: `delete_node(node_id)` removes the node from `vector_store` (RAG.SourceDocuments).
- FR-003: `delete_node(node_id)` is idempotent — calling it on a non-existent node is a no-op.
- FR-004: BM25 deletion is skipped; KG deletion makes BM25 results inert (no per-doc API).
- FR-005: `delete_node(None)` raises `ValueError`.
- FR-006: `delete_node("")` raises `ValueError`.
- FR-007: Partial store failure logs `WARNING` and propagates the exception.
- FR-008: `delete_node` returns `None`.

## Non-Functional Requirements

- NF-001: No changes to `index_node()` or any other existing method.
- NF-002: Bridge adapter (`IRISGraphRAGBridge.delete_node`) deferred to opsreview repo.

## Out of Scope

- BM25 per-document delete (no API in iris_vector_graph 2.3.1).
- `IRISGraphRAGBridge.delete_node` (opsreview repo, separate feature).
- Any pipeline other than `HybridGraphRAGPipeline`.
