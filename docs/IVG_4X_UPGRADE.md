# Upgrading to iris-vector-graph 4.x

IVR requires `iris-vector-graph>=4.2.0,<5`. ivg 4.0 changed how `Graph_KG` is
stored, so an install created under ivg 2.x has to be migrated once before
`graphrag` or `pylate_colbert` can use it.

## What changed in the layout

- `kg_NodeEmbeddings` (and `kg_NodeEmbeddings_optimized`) are keyed on
  `(graph_id, node_id)`. The old `id` column is gone; the columns are
  `emb_rowid, graph_id, node_id, emb, metadata`.
- `rdf_labels` and `rdf_props` gain a `graph_id` column.
- A named graph's vectors live in their own routed table, `kg_emb_<16 hex>`. The
  default graph still uses `kg_NodeEmbeddings`.

## How IVR reports an old install

`SchemaManager.validate_graph_prerequisites()` and
`SchemaManager.ensure_iris_vector_graph_tables()` read the `Graph_KG` columns and
list every table still in the 2.x layout in `legacy_layout_tables`. Validation
fails, and the error message names the call that fixes it. IVR never migrates on
its own: the migration moves and quarantines rows, which is the operator's call.

## Running the upgrade

Preview first, then run:

```python
from iris_vector_rag.storage.schema_manager import SchemaManager

manager = SchemaManager(connection_manager, config_manager)

preview = manager.upgrade_iris_vector_graph_storage(dry_run=True)
for step in preview.steps:
    print(step.name, step.report or step.skipped_because)

report = manager.upgrade_iris_vector_graph_storage()
print(report.names)
```

The real run calls ivg's `initialize_schema()` and then
`iris_vector_graph.migrations.upgrade_to_4_0_0(conn, dry_run=False)`. Extra
keyword arguments (`steps=`, `vector_resolver=`, ...) are passed through to
`upgrade_to_4_0_0`. Running it again on a migrated install changes nothing.

The dry run writes nothing, including DDL, so it does not run
`initialize_schema()`. On a genuine pre-4.0 install some steps need a column that
only the real run adds; ivg 4.1.1 reports those with `blocked_because` naming the
call that adds it, and `report.blocked` lists them.

## ObjectScript classes

The `kg_node_stores` step recompiles the shipped `Graph.KG` classes and rebuilds the
`^KG` globals, so the ivg ObjectScript layer has to be on the server. Since ivg
4.1.1 the wheel ships the `.cls` sources and `initialize_schema()` sends them over
the connection; nothing has to be copied into the container. A class the server
cannot compile (embedded Python missing, or `%AI.*` classes not installed) is
skipped and listed under `objectscript_skipped` in the init status. Set
`IVG_EMBEDDED_PYTHON=0` or `1` to override the embedded-Python check, and
`IVG_FORCE_CLASS_DEPLOY=1` to reinstall an unchanged set.

Some IRIS builds kill the calling process when the deploy compiles from the Native
API, and the check for embedded Python can do the same. `docs/DEBT.md` entry 10
describes what to do on such a container. IVR's own tests set
`IVG_EMBEDDED_PYTHON=0`.

## Graph-scoped search

`HybridRetrievalMethods.retrieve_via_rrf`, `retrieve_via_enhanced_text` and
`retrieve_via_hnsw_vector` accept `graph=` and pass it to ivg's `kg_RRF_FUSE`,
`kg_TXT` and `kg_KNN_VEC`. `retrieve_via_hnsw_vector` also accepts `model_key=`
to pick an embedding route. Without them, searches use the default graph, as
before.

## Deleting nodes

`engine.delete_node(node_id)` removes the node, its vectors and its edges from every
graph that holds it. That reaches the routed `kg_emb_<hash>` table of a named graph
only since ivg 4.1.1: 4.1.0 skipped it, the foreign key refused the delete, and the
call returned `False`. Pass `graph=` to delete from one graph.
`HybridGraphRAGPipeline.delete_node` calls it without `graph=`.

## PLAID

`Graph.KG.PLAIDSearch.Info` returns `nlist` and `indexed`. `nCentroids` and
`nDocs` are only the `^PLAID` subscripts those values are read from.
`PLAIDSearcher` used to check `nCentroids`, which always read 0, so every search
raised `PLAIDNotBuiltError`. It now checks `nlist`.
