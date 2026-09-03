# Implementation Plan: Feature 062

## Target File

`iris_vector_rag/pipelines/hybrid_graphrag.py` — add `delete_node` method after `get_hybrid_status`.

## Design

```python
def delete_node(self, node_id: str) -> None:
    # Validate
    # Call self.iris_engine.delete_node(node_id) if engine available
    # Call self.vector_store.delete_documents([node_id]) if store available
    # Log and return None
```

## BM25 Decision

`iris_vector_graph` 2.3.1 has no `bm25_delete` method. KG deletion removes the node's
text properties, making any BM25 results for that node inert. Explicit BM25 cleanup skipped.

## Testing

- Contract tests (mocked): `tests/contract/test_delete_node_contract.py`
- Integration tests (live IRIS): `tests/integration/test_delete_node_integration.py`
  — skip due to venv incompatibility (`intersystems_iris`/`sqlalchemy_iris` version mismatch)
