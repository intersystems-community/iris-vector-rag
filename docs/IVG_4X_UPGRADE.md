# Upgrading to iris-vector-graph 4.x

IVR requires `iris-vector-graph>=4.1.0,<5`. ivg 4.0 changed how `Graph_KG` is
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

The dry run writes nothing, including DDL, so it previews one step at a time. On
a genuine pre-4.0 install ivg 4.1.0 cannot preview `docs` or `kg_node_stores`
until the earlier steps have created the columns they read. Those steps come back
with `skipped_because="cannot be previewed until the earlier steps have run: ..."`
instead of the whole preview failing.

## The ObjectScript prerequisite for `kg_node_stores`

The `kg_node_stores` step recompiles the `Graph.KG` package and rebuilds the
`^KG` globals, so the ivg ObjectScript classes have to be on the server. The
ivg 4.1.0 wheel does not ship them.

1. Take `iris_src/src` from the ivg 4.1.0 sdist and copy it into the container at
   `/tmp/src`. `initialize_schema()` deploys from there.
2. On a container that cannot compile embedded Python (see `docs/DEBT.md`,
   entry 1), leave out the `Language = python` classes. Compiling them crashes the
   callin process with signal 11.
3. After the upgrade, move `/tmp/src` out of the way. While it exists every
   `initialize_schema()` call redeploys the classes, which takes about 8 seconds.

## Graph-scoped search

`HybridRetrievalMethods.retrieve_via_rrf`, `retrieve_via_enhanced_text` and
`retrieve_via_hnsw_vector` accept `graph=` and pass it to ivg's `kg_RRF_FUSE`,
`kg_TXT` and `kg_KNN_VEC`. `retrieve_via_hnsw_vector` also accepts `model_key=`
to pick an embedding route. Without them, searches use the default graph, as
before.

## Deleting nodes in a named graph

`index_node()` writes to the default graph, and `delete_node()` removes that
node. ivg 4.1.0's `engine.delete_node()` only deletes from `kg_NodeEmbeddings`,
though, so for a node in a named graph the foreign key from its routed
`kg_emb_<hash>` table refuses the delete and the call returns `False`. Use
`engine.delete_nodes_by_prefix(node_id, graph=...)` for those. It is a prefix
match, so it also deletes any node whose ID starts with `node_id`.

## PLAID

`Graph.KG.PLAIDSearch.Info` returns `nlist` and `indexed`. `nCentroids` and
`nDocs` are only the `^PLAID` subscripts those values are read from.
`PLAIDSearcher` used to check `nCentroids`, which always read 0, so every search
raised `PLAIDNotBuiltError`. It now checks `nlist`.
