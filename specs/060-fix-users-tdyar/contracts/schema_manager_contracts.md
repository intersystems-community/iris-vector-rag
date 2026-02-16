# Contract: Graph Schema Manager (Bug 2)

This feature does not add external HTTP/GraphQL APIs. Contracts here define internal Python behaviors.

## Module: `iris_vector_rag/storage/schema_manager.py`

### Behavior Contract

- If GraphRAG pipeline requested and iris-vector-graph is missing → raise `ImportError` with install guidance.
- Ensure tables exist: `rdf_labels`, `rdf_props`, `rdf_edges`, `kg_NodeEmbeddings_optimized`.
- Table creation must be **atomic** and **idempotent**.
- Validate schema structure before PPR usage.
- Log success/failure of initialization and PPR availability.

### Example Usage

```python
from iris_vector_rag.storage.schema_manager import SchemaManager

schema = SchemaManager(connection)
schema.ensure_iris_vector_graph_tables()
schema.validate_graph_prerequisites()
```
