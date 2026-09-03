# Contract: delete_node

## Signature

```python
def delete_node(self, node_id: str) -> None
```

## Preconditions

- `node_id` must be a non-empty string.

## Postconditions

- Node removed from `iris_engine` (Graph_KG) if engine is available.
- Node removed from `vector_store` (RAG.SourceDocuments) if store is available.
- Returns `None`.
- Idempotent: no exception if node does not exist.

## Exceptions

- `ValueError` if `node_id` is `None`, empty string, or not a string.
- Any `iris_engine` or `vector_store` exception propagates after WARNING log.
