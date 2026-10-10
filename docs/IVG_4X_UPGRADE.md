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

## From 4.1.1 to 4.2.0

IVR requires `iris-vector-graph>=4.2.0,<5`: it calls `engine.bm25_upgrade` and passes
`max_df_fraction` to `bm25_search`, and both first exist in 4.2.0. There is no storage
migration between 4.1.1 and 4.2.0. `initialize_schema()` deploys the new `Graph.KG`
classes as before.

### BM25 indexes: upgrade once

ivg 4.2.0 stores BM25 indexes as schema 2 (document frequency and a per-document token
record), so `Insert` and `Delete` cost in proportion to the document, not the index. An
index built earlier is schema 1: `Search` still works, but `Insert` and `Delete` raise an
error naming `Upgrade` until `engine.bm25_upgrade(name)` has run once. Scores change
after the upgrade: IDF is computed when searching, and length normalisation now follows
the BM25 formula.

IVR owns no BM25 index. HybridGraphRAG's text leg is ivg's `kg_TXT` over the KG tables,
and the only place IVR searches BM25 is `IVGTextSearchBackend`, over an index you built
yourself. Nothing is upgraded implicitly. Preview, then run:

```python
report = manager.upgrade_bm25_indexes(dry_run=True)   # reads Info, writes nothing
report.pending        # schema-1 indexes a real run would convert
report.current        # already schema 2

report = manager.upgrade_bm25_indexes()               # or names=["logs"]
report.upgraded, report.failed, report.missing
```

To run the layout step and the BM25 step together, in ivg's order (the storage step
deploys the classes `Upgrade` lives in):

```python
result = manager.upgrade_iris_vector_graph(dry_run=True)
result = manager.upgrade_iris_vector_graph()
result.storage        # ivg's UpgradeReport
result.bm25           # IVR's Bm25UpgradeReport
```

- Only indexes that are still schema 1 are touched, so the call can be repeated.
- By default the first index that cannot be read or upgraded raises `Bm25UpgradeError`
  (`.name`, the ivg error as `__cause__`, `.report` for what was done before it).
  `continue_on_error=True` records it in the report (`report.failed`) and goes on.
- Without `names=`, the indexes are the ones ivg lists when the engine is created. An index
  another process creates afterwards is not in it; pass its name.
- `Upgrade` is one pass over the posting lists. Writers are refused while it runs and
  `Search` keeps working, so run it as a job, not on a request path.
- A missing index is reported as `missing` and not created.

`IVGTextSearchBackend` logs one warning the first time it searches a schema-1 index,
naming the call that upgrades it.

### The engine no longer logs in as `_SYSTEM`

ivg 4.1.1 opened `IRISGraphEngine._iris_obj()`, the connection for BM25, vector, PLAID and
ledger calls, as a hard-coded `_SYSTEM` / `SYS`, so those calls ran with full rights
whatever role IVR had connected as. ivg 4.2.0 uses the credentials the engine was built
with. `IRISGraphEngine.from_connect` keeps them; an engine built from a bare connection
keeps none, and uses that connection itself, shared with cursor SQL.

IVR builds its engines from the session's connection, so each of them would have taken
the shared path. IVR now records the account the session resolved on every engine it
builds (`integrations.ivg.remember_credentials`): HybridGraphRAG, the PLAID and VecIndex
engines, and the schema manager's engine. The class-method calls keep a connection of their
own, as IVR's account. For a superuser account (the `_SYSTEM` default) nothing changes. For
a restricted account the role now decides what those calls may do. Against a real IRIS, in
`tests/integration/test_ivg_least_privilege.py`:

- a role with `%DB_USER:R` and `%Native_ClassExecution:U` can search a BM25 index and
  cannot write to it (`<PROTECT>`);
- a role with `%DB_USER:RW` and `%Native_ClassExecution:U` can do both;
- a role without `%Native_ClassExecution` is refused on the first class-method call. ivg
  4.1.1 ran the same call as `_SYSTEM` (when `SYS` was still that account's password; it
  fell back to the caller's connection otherwise).

`_connection_params` is an ivg attribute, set by `from_connect`; a unit test checks
against the installed ivg that `_iris_obj` reads it. The minimum rights for each ivg
operation are in `docs/research/least-privilege-spike.md` in the iris-vector-graph
repository. That document also measures that `initialize_schema()` and the class deploy
need more than a reader or writer role has (a deployer's rights), so run
`SchemaManager.upgrade_iris_vector_graph*` and `ensure_iris_vector_graph_tables` as a
privileged account.

### The BM25 common-term cutoff

`bm25_search(..., max_df_fraction=...)` skips query terms found in more than that fraction
of the documents; if every term is over it, the rarest is kept. It is off by default.
`IVGTextSearchBackend` takes it as an argument or reads it from the configuration:

```yaml
retrieval:
  bm25:
    max_df_fraction: 0.9 # 0 = off (the default); a number from 0 to 1
```

```python
backend = IVGTextSearchBackend.from_config(engine, "isc-obs-logs", config_manager)
```

A value outside 0 to 1 raises `ValueError` naming `retrieval.bm25.max_df_fraction`. The
option is only passed to ivg when it is non-zero, and it applies to schema 2 indexes only
(ivg ignores it on schema 1).

### What did not change for IVR

- **Arno callout adoption, reader routing and the result cache** touch `kg_PAGERANK`,
  `kg_PPR`, `kg_WCC`, `kg_CDLP`, `kg_SUBGRAPH` and BFS. IVR calls none of them; it calls
  `kg_TXT`, `kg_RRF_FUSE`, `kg_KNN_VEC`, `delete_node`, the PLAID and VecIndex methods and
  the BM25 methods. ivg's own changelog says no Native Reader passes its gate on the build
  it measured, so every algorithm still answers from ObjectScript.
- **The ObjectScript fixes** in the 4.2.0 changelog (`FHIRGraph.SplitKeep`,
  `Graph.KG.Service`, `MCPTools.CypherQuery`, the embedded engine's empty-string binding,
  `IVG.CypherEngine`, `LedgerApply`) have no IVR workaround to remove: IVR does not call
  those classes or those paths.
- **The embedded-Python probe** is unchanged (`class_deploy.py` is identical in 4.1.1 and
  4.2.0), so `IVG_EMBEDDED_PYTHON=0` in IVR's tests still stands. See `docs/DEBT.md`
  entry 10 for what was and was not reproduced.

### Until 4.2.0 is on PyPI

The pin `iris-vector-graph>=4.2.0,<5` cannot be installed from PyPI until ivg 4.2.0 is
published. Install a wheel built from the `v4.2.0` tag first, then IVR:

```bash
uv build --wheel --out-dir /tmp/ivg42-dist /path/to/checkout-of-v4.2.0
uv pip install /tmp/ivg42-dist/iris_vector_graph-4.2.0-py3-none-any.whl
uv pip install --find-links /tmp/ivg42-dist -e ".[dev]"
```
