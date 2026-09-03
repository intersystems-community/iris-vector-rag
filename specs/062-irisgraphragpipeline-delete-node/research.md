# Research: Feature 062

## BM25 Delete API

**Decision**: Skip BM25 deletion.
**Rationale**: `iris_vector_graph` 2.3.1 has no `bm25_delete` method. KG deletion removes
the node's text properties, making any stale BM25 results inert (the node's document text
no longer exists in Graph_KG). Explicit BM25 cleanup is not needed.

## Bridge Adapter

**Decision**: Defer `IRISGraphRAGBridge.delete_node` to opsreview repo.
**Rationale**: `IRISGraphRAGBridge` lives in opsreview, a separate repository. Out of scope
for this feature. The opsreview bridge can wrap `pipeline.delete_node(node_id)` directly.

## venv Incompatibility

`intersystems_iris` package installed in `.venv` uses renamed private classes (`_IRISConnection`,
`_IRISList`) while `sqlalchemy_iris` still imports the unprefixed names. Integration tests
skip with `pytest.importorskip("iris_vector_graph")` since the import chain fails. This is a
pre-existing environment issue unrelated to Feature 062.
