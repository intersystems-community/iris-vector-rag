# Release Notes: iris-vector-rag v0.12.0

## IRISVectorEngine — unified engine object (feature 080)

### What's new

`IRISVectorEngine` is a new top-level class that collapses the
`(connection_manager, config_manager)` pair that every pipeline constructor
previously required into a single well-typed entry point.

#### One-line construction

```python
from iris_vector_rag import IRISVectorEngine, create_pipeline

engine = IRISVectorEngine.from_config()
pipeline = create_pipeline("basic", engine=engine)
result = pipeline.query("What causes diabetes?")
```

#### All six pipelines accept `engine=`

```python
from iris_vector_rag import IRISVectorEngine, create_pipeline, create_validated_pipeline

engine = IRISVectorEngine.from_config()

# Any pipeline type
for ptype in ("basic", "basic_rerank", "crag", "graphrag", "multi_query_rrf"):
    p = create_pipeline(ptype, engine=engine)

# Production (validated) path
p = create_validated_pipeline(pipeline_type="basic", engine=engine, auto_setup=True)
```

#### Direct pipeline construction also works

```python
from iris_vector_rag import IRISVectorEngine
from iris_vector_rag.pipelines.basic import BasicRAGPipeline

engine = IRISVectorEngine.from_config()
pipeline = BasicRAGPipeline(engine)          # engine as first positional arg
```

#### Raw DBAPI connection supported

```python
from iris_vector_rag import IRISVectorEngine
from iris_vector_rag.common.iris_connection import get_iris_connection

conn = get_iris_connection()
engine = IRISVectorEngine(conn, schema_prefix="RAG")
```

### Engine properties

| Property              | Type                   | Description                              |
| --------------------- | ---------------------- | ---------------------------------------- |
| `connection`          | DBAPI connection       | Lazy-loaded; opens on first access       |
| `vector_store`        | `IRISVectorStore`      | Lazy-loaded vector store                 |
| `schema_prefix`       | `str`                  | SQL schema prefix (default `"RAG"`)      |
| `connection_manager`  | `ConnectionManager`    | Underlying manager (compat property)     |
| `config_manager`      | `ConfigurationManager` | Config manager (compat property)         |
| `embedding_dimension` | `int`                  | Embedding dimension from schema / config |

### Lazy initialization

`IRISVectorEngine.from_config()` builds `ConfigurationManager` and
`ConnectionManager` but does **not** open a DB connection until `.connection`
or `.vector_store` is first accessed (FR-008 / pure-constructors principle).

### Schema prefix flow-through

```python
engine = IRISVectorEngine.from_config(schema_prefix="MYAPP")
assert engine.schema_prefix == "MYAPP"
assert engine.vector_store.schema_manager.schema_prefix == "MYAPP"
```

### Backward compatibility

All existing call sites continue to work without modification. The
`(connection_manager, config_manager)` pair is deprecated but not removed.
`create_pipeline()` without `engine=` is unchanged.

### Also in this release

- **AGENTS.md** rewritten from stub to 370-line reference (WHERE TO LOOK
  table, CODE MAP, agent workflows, pipeline comparison, environment vars,
  ecosystem links).
- **Skill manifests** (`skills/iris-rag-pipeline`, `skills/iris-vector-search`)
  gain `source:` field for agent discoverability.
- **README** MCP server section updated to document all 8 exposed tools and
  CLI commands.
- **AGENTS.md** now cross-references `iris-agentic-dev` for ObjectScript MCP
  tool access alongside pipelines (install separately from that repo).

### Testing

- 313 unit tests (no IRIS required)
- 8 new E2E tests for `IRISVectorEngine` against real IRIS (`tests/e2e/test_engine_e2e.py`)
