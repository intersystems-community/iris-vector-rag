# Research: Attach Existing Corpus

**Date**: 2026-04-03 | **Status**: Complete

## Findings

No unknowns required research — all building blocks exist in IVG >= 1.27.0.

### Decision 1: Where to add the method

- **Decision**: Add `attach_existing_corpus()` to `HybridGraphRAGPipeline` (hybrid_graphrag.py:25)
- **Rationale**: This is the class users interact with. It already holds `self.iris_engine` (an `IRISGraphEngine` instance) initialized in `_initialize_graph_core()`.
- **Alternatives considered**: Standalone function, new class `CorpusBridge` — rejected because they add unnecessary indirection over what is fundamentally 3 engine method calls.

### Decision 2: Dimension storage location

- **Decision**: Store detected dimension in `self._attached_corpora` dict keyed by `graph_label`
- **Rationale**: In-memory dict is sufficient — the dimension is re-detected on each `attach_existing_corpus` call (idempotent), and vector_search validates dimension at query time. No persistent storage needed since the `Graph_KG.table_mappings` table already persists the mapping.
- **Alternatives considered**: Add a `vector_col` column to `Graph_KG.table_mappings` — rejected because it requires an IVG schema change for a feature that only IVR uses.

### Decision 3: HNSW index detection

- **Decision**: Use `INFORMATION_SCHEMA.INDEXES` query to check for VECTOR index on the column. Warn if absent, don't fail.
- **Rationale**: HNSW index is optional — brute-force vector search still works, just slower. The user should know to run `BUILD INDEX` but shouldn't be blocked.
- **Alternatives considered**: Require HNSW index (fail if absent) — rejected because it adds an unnecessary prerequisite for a metadata-only operation.

### Decision 4: Error types

- **Decision**: `ValueError` for table/column not found. New `DimensionMismatchError(ValueError)` for query-time dimension mismatch.
- **Rationale**: `ValueError` is standard Python for invalid arguments. `DimensionMismatchError` is a subclass so it's catchable as either.
- **Alternatives considered**: Generic `RuntimeError` — rejected because dimension mismatch is a caller error (wrong query vector), not a runtime failure.
