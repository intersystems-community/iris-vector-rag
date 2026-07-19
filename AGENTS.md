# AGENTS.md — iris-vector-rag (IVR)

**Owner:** Thomas Dyar (Tom) — Sr. Manager, AI Platform and Ecosystems, InterSystems Corporation

> NEVER use "Tim" — that is Tim Leavitt, a colleague. Always use "Tom" in conversation.

**PyPI:** `iris-vector-rag` **Python:** 3.10+ **License:** MIT
**Repo:** `intersystems-community/iris-vector-rag`

---

## OVERVIEW

Production Python framework for RAG pipelines backed by **InterSystems IRIS Vector Search**.
Six swappable pipeline strategies share one interface and one response shape — switch via
a single factory call. IRIS replaces an external VectorDB; no Pinecone/Weaviate/pgvector needed.

**Stack:** Python 3.10+ | IRIS Vector Search (HNSW index) | iris-vector-graph (optional, GraphRAG) |
LangChain / LlamaIndex adapters | sentence-transformers | RAGAS evaluation | FastAPI MCP server

**Package name:** `iris_vector_rag` (`iris_rag/` is a thin backward-compat shim — do not add code there)

## STRUCTURE

```text
iris_vector_rag/           # Main package
  core/                    # RAGPipeline ABC, ConnectionManager, models, validators
  pipelines/               # 6 pipeline implementations
    basic.py               # Dense vector retrieval
    basic_rerank.py        # Dense + cross-encoder reranking
    crag.py                # Corrective RAG (self-critique loop)
    hybrid_graphrag.py     # GraphRAG via iris-vector-graph
    multi_query_rrf.py     # Multi-query + Reciprocal Rank Fusion
    colbert_pylate/        # ColBERT late-interaction (PyLate)
  storage/                 # IRISVectorStore, SchemaManager, CLOB handler
  embeddings/              # EmbeddingManager (sentence-transformers, configurable)
  config/                  # ConfigurationManager, default_config.yaml
  validation/              # Pre-flight validators, SetupOrchestrator
  services/                # Entity extraction, storage adapters
  mcp/                     # MCP server (8 tools — rag_basic … rag_metrics)
  api/                     # Optional REST API (FastAPI + Redis)
  common/                  # Shared utilities, LLM cache, SQL executor

tests/
  unit/                    # No IRIS required — pytest tests/unit/
  contract/                # IRIS required — schema, config, validation contracts
  e2e/                     # Full pipeline E2E against real IRIS

specs/                     # Feature specs (NNN-name/spec.md + plan.md + tasks.md)
skills/                    # Agent skill manifests (iris-rag-pipeline, iris-vector-search)
evaluation_framework/      # RAGAS side-by-side pipeline comparison
examples/                  # compare_pipelines.py and usage examples
```

## WHERE TO LOOK

| Task                  | Location                                                 | Notes                                                                               |
| --------------------- | -------------------------------------------------------- | ----------------------------------------------------------------------------------- |
| Create a pipeline     | `iris_vector_rag/__init__.py`                            | `create_pipeline(type, ...)` or `create_validated_pipeline(...)`                    |
| Pipeline base class   | `core/base.py::RAGPipeline`                              | ABC — `load_documents`, `query` interface                                           |
| Response shape        | `core/base.py`                                           | Always: `answer`, `retrieved_documents`, `contexts`, `sources`, `error`, `metadata` |
| Vector store          | `storage/vector_store_iris.py::IRISVectorStore`          | `add_documents`, `similarity_search`, `fetch_documents_by_ids`                      |
| Schema management     | `storage/schema_manager.py::SchemaManager`               | Single source of truth for dims, schema prefix, table names via `_qn()`             |
| Connection            | `core/connection.py::ConnectionManager`                  | DBAPI via `iris.dbapi.connect()`, cached + health-checked                           |
| Configuration         | `config/manager.py::ConfigurationManager`                | Precedence: `RAG_` env vars > YAML > defaults                                       |
| Embedding dims        | `storage/schema_manager.py`                              | Never hardcode; call `schema_manager.get_embedding_dimension()`                     |
| GraphRAG              | `pipelines/hybrid_graphrag.py`                           | Requires `iris-vector-graph>=2.4.6`                                                 |
| Validation pre-flight | `validation/requirements.py` + `validation/validator.py` | `BasicRAGRequirements`, etc.                                                        |
| Schema prefix         | `config/manager.py::get_schema_prefix()`                 | `IRIS_SCHEMA_PREFIX` env var; default `RAG`                                         |
| Engine entry point    | `core/engine.py::IRISVectorEngine`                       | `from_config()` for one-line construction; replaces `(cm, cfg)` pair                |
| MCP server            | `mcp/bridge.py`, `mcp/tool_schemas.py`                   | 8 tools: `rag_basic`, `rag_crag`, `rag_graphrag`, …                                 |
| MCP CLI               | `mcp/cli.py`                                             | `python -m iris_vector_rag.mcp start/stop/status/health/list-tools`                 |
| CLOB handling         | `storage/clob_handler.py::ensure_string_content()`       | Apply on every retrieved `page_content`                                             |
| Metadata filtering    | `storage/metadata_filter_manager.py`                     | 17 default keys; blocks SQL injection patterns                                      |

## CODE MAP (high-centrality symbols)

| Symbol                        | Type     | Location                       | Role                                      |
| ----------------------------- | -------- | ------------------------------ | ----------------------------------------- |
| `create_pipeline()`           | function | `__init__.py`                  | Primary factory; maps type string → class |
| `create_validated_pipeline()` | function | `__init__.py`                  | Factory with pre-flight DB checks         |
| `RAGPipeline`                 | ABC      | `core/base.py`                 | All pipelines extend this                 |
| `IRISVectorStore`             | class    | `storage/vector_store_iris.py` | Implements `VectorStore` ABC              |
| `SchemaManager`               | class    | `storage/schema_manager.py`    | Schema prefix, dims, table qualification  |
| `ConnectionManager`           | class    | `core/connection.py`           | DBAPI connection pool                     |
| `ConfigurationManager`        | class    | `config/manager.py`            | All config access goes here               |
| `EmbeddingManager`            | class    | `embeddings/manager.py`        | Embedding generation + caching            |
| `PreConditionValidator`       | class    | `validation/validator.py`      | Pre-flight table/embedding checks         |
| `BasicRAGPipeline`            | class    | `pipelines/basic.py`           | Simplest pipeline; start here             |
| `IRISVectorEngine`            | class    | `core/engine.py`               | Unified engine: wraps CM + cfg into one   |

## COMMANDS

```bash
# Install
pip install -e ".[dev]"                     # Dev; add ,evaluation,colbert,dspy,mcp,api as needed
pip install -e ".[mcp]"                     # With MCP server

# IRIS lifecycle
docker compose up -d                        # Start IRIS container (iris-vector-rag-iris, port 51972)
make docker-up / docker-down / docker-logs  # Aliases
make setup-db                               # Initialize RAG schema in IRIS

# Test
pytest tests/unit/                          # Fast, no IRIS
pytest tests/unit/ tests/contract/          # Full suite, needs IRIS
pytest tests/e2e/                           # E2E, needs IRIS + OPENAI_API_KEY
pytest tests/path/to/test.py::Class::method # Single test
IRIS_PORT=51972 pytest tests/e2e/           # Explicit port

# Lint / format
black iris_vector_rag/ tests/ && isort iris_vector_rag/ tests/
flake8 iris_vector_rag/
mypy iris_vector_rag/ --strict

# MCP server
python -m iris_vector_rag.mcp start         # Start on default port
python -m iris_vector_rag.mcp list-tools    # List 8 exposed tools
python -m iris_vector_rag.mcp status        # Server status
```

No `make test` target — invoke `pytest` directly.

## AGENT WORKFLOWS

### Index documents into IRIS

```python
from iris_vector_rag import create_pipeline

pipeline = create_pipeline("basic")
pipeline.load_documents(
    documents_path="docs/",        # or: documents=[Document(...), ...]
    chunk_size=512,
    chunk_overlap=64,
)
```

### Query a RAG pipeline

```python
from iris_vector_rag import create_pipeline

pipeline = create_pipeline("basic")          # or: crag, graphrag, basic_rerank, multi_query_rrf
result = pipeline.query(
    query_text="What causes type 2 diabetes?",
    top_k=5,
    generate_answer=True,
)
# result keys: answer, retrieved_documents, contexts, sources, error, metadata
```

### Validated pipeline (production path)

```python
from iris_vector_rag import create_validated_pipeline

pipeline = create_validated_pipeline(
    pipeline_type="basic",
    validate_requirements=True,
    auto_setup=True,               # creates tables/embeddings if missing
)
```

### Evaluate pipeline with RAGAS

```python
# Requires: pip install iris-vector-rag[evaluation]
from evaluation_framework.evaluator import PipelineEvaluator

evaluator = PipelineEvaluator(pipeline_types=["basic", "crag", "graphrag"])
results = evaluator.run(questions=test_questions, ground_truths=expected_answers)
# Metrics: faithfulness, context_precision, context_recall, answer_relevancy
```

### Extend with a custom chunker

```python
from iris_vector_rag.core.base import RAGPipeline

class MyPipeline(RAGPipeline):
    def load_documents(self, documents_path, **kwargs):
        # custom chunking logic
        chunks = my_chunker(documents_path)
        return super().load_documents(documents=chunks, **kwargs)

    def query(self, query_text: str, top_k: int = 5, **kwargs):
        # must return: answer, retrieved_documents, contexts, sources, error, metadata
        ...
```

### Switch schema prefix (multi-tenant)

```bash
IRIS_SCHEMA_PREFIX=MYAPP python -c "
from iris_vector_rag import create_pipeline
p = create_pipeline('basic')    # uses MYAPP.DocumentChunks, MYAPP.SourceDocuments, etc.
"
```

## PIPELINE COMPARISON

| Pipeline          | When to use                         | Extra deps            |
| ----------------- | ----------------------------------- | --------------------- |
| `basic`           | Baseline dense retrieval            | none                  |
| `basic_rerank`    | Better precision via cross-encoder  | sentence-transformers |
| `crag`            | Self-correcting with web fallback   | none                  |
| `graphrag`        | Entity/relationship-aware retrieval | iris-vector-graph     |
| `multi_query_rrf` | Multi-angle queries + fusion        | none                  |
| `pylate_colbert`  | ColBERT late interaction            | pylate                |

## CONVENTIONS

- **Line length:** 88 (black + isort)
- **Type hints:** strict mypy on `iris_vector_rag/`
- **`query()` signature:** always `(self, query_text: str, top_k: int = 5, **kwargs)`
- **`query()` return:** always includes `answer`, `retrieved_documents`, `contexts`, `sources`, `error`, `metadata`
- **Schema names:** route all table names through `schema_manager._qn("TableName")` — never write `"RAG.TableName"` literals
- **CLOB content:** wrap retrieved `page_content` in `ensure_string_content()` before returning
- **No DB in `__init__`:** constructors must not open connections; use lazy init via `_ensure_initialized()`
- **Config access:** always use `ConfigurationManager.get(...)` — never read env vars or YAML directly in pipelines

## ANTI-PATTERNS

- **DO NOT** add real code to `iris_rag/` — it is a shim only
- **DO NOT** hardcode `"RAG."` table prefix — use `schema_manager._qn()`
- **DO NOT** hardcode embedding dimensions — call `schema_manager.get_embedding_dimension()`
- **DO NOT** mock the database in integration/contract/e2e tests — IRIS unavailability is a SYSTEM FAILURE
- **DO NOT** add class-level mutable state to `SchemaManager` — caches must be instance-level
- **DO NOT** return a `query()` response missing `sources` or `error` at the top level
- **DO NOT** use `intersystems-iris` package — use `intersystems-irispython` (DBAPI) or `iris-embedded-python-wrapper`
- **DO NOT** write `import iris.dbapi` at module level — import lazily inside functions

## KNOWN PAIN POINTS

### Vector datatype mismatch (FLOAT vs DOUBLE)

IRIS sometimes stores embeddings as DOUBLE when FLOAT is requested. `IRISVectorStore` auto-retries
with the alternate type. If you see "FOUND DOUBLE IN SQL" in logs, it is expected fallback behavior.

### CLOB objects from IRIS

`cursor.fetchall()` may return IRIS CLOB objects instead of strings for long text. Always call
`ensure_string_content(doc.page_content)` from `storage/clob_handler.py` on retrieved content.

### `pytest.ini` is authoritative

`pyproject.toml` has a `[tool.pytest.ini_options]` block that is **stale** — `pytest.ini`
overrides it. Markers, `asyncio_mode=auto`, and the 60s timeout are set in `pytest.ini`.

### Module-scoped fixtures + repeated `load_documents`

Sharing a single `ConnectionManager` / `IRISVectorStore` across many `load_documents` calls in the
same test process causes the IRIS DBAPI C extension to segfault. Use `scope="function"` fixtures
for any test that calls `load_documents` multiple times.

### `test_basic_dimension_validation.py` requires pytest-mock

`tests/contract/test_basic_dimension_validation.py` requires `pytest-mock` (not installed by default).
Skip it or install `pytest-mock` when running the contract suite.

## ENVIRONMENT

| Variable                   | Default     | Description                                        |
| -------------------------- | ----------- | -------------------------------------------------- |
| `IRIS_HOST`                | `localhost` | IRIS superserver host                              |
| `IRIS_PORT`                | `1972`      | IRIS superserver port (project default: `51972`)   |
| `IRIS_NAMESPACE`           | `USER`      | IRIS namespace                                     |
| `IRIS_USERNAME`            | `_SYSTEM`   | IRIS username                                      |
| `IRIS_PASSWORD`            | `SYS`       | IRIS password                                      |
| `IRIS_SCHEMA_PREFIX`       | `RAG`       | SQL schema prefix (`RAG.SourceDocuments`, etc.)    |
| `OPENAI_API_KEY`           | —           | Required for LLM answer generation and E2E tests   |
| `RAG_DATABASE__IRIS__HOST` | —           | Override via env (`RAG_` prefix, `__` for nesting) |
| `RAG_DATABASE__IRIS__PORT` | —           | Port override via env                              |
| `RAG_EMBEDDINGS__MODEL`    | —           | Embedding model override                           |

## TEST FIXTURES

```python
# From tests/e2e/conftest.py
def test_foo(pipeline_dependencies):           # module-scoped shared connection + vector store
def test_foo(fresh_iris_vector_store):         # function-scoped, fresh connection per test
def test_foo(sample_biomedical_documents):     # 10 biomedical Document objects
```

Markers: `requires_database`, `requires_llm`, `requires_docker`, `contract`, `colbert`, `e2e`, `slow`

Unit tests (`tests/unit/`) run without IRIS. Contract and E2E tests require the project IRIS
container (`iris-vector-rag-iris`, port `51972`).

## AGENT SKILLS

Load these skills when working in this repo:

| Skill                | When to load                    | What it covers                                                    |
| -------------------- | ------------------------------- | ----------------------------------------------------------------- |
| `iris-rag-pipeline`  | Building or modifying pipelines | Factory patterns, pipeline interface, RAGAS evaluation            |
| `iris-vector-search` | IRIS SQL vector syntax          | `TO_VECTOR`, `VECTOR_COSINE`, HNSW index DDL, ObjectScript access |

```text
Load iris-rag-pipeline skill for pipeline patterns.
Load iris-vector-search skill for IRIS SQL vector syntax.
```

Skill manifests: `skills/iris-rag-pipeline/SKILL.md`, `skills/iris-vector-search/SKILL.md`

## ECOSYSTEM

IVR is the RAG pipeline layer; it integrates with the broader IRIS AI platform:

### iris-agentic-dev (iad)

MCP tools for ObjectScript compilation and execution. Use iad when you need to introspect
IRIS classes, run ObjectScript, or manage IRIS configuration alongside IVR pipelines.

For MCP tool orchestration across IRIS packages, use iris-agentic-dev.
Install separately — see the iris-agentic-dev repo for setup instructions.

### iris-vector-graph (ivg)

Graph database layer for entity-relationship extraction and GraphRAG. IVR depends on
`iris-vector-graph>=2.4.6` for `graphrag` and `pylate_colbert` pipeline modes.

```python
from iris_vector_rag import create_pipeline
pipeline = create_pipeline("graphrag")   # uses ivg internally
```

### iris-devtester (idt)

Container lifecycle management for tests. IVR uses idt as a dev dependency.

```bash
idt container up --name iris-vector-rag-iris --port 51972
# or:
docker compose up -d
```

### Cross-references

- **iris-agentic-dev:** compile/execute ObjectScript in the running IRIS container
- **iris-vector-graph:** GraphRAG entity extraction, ColBERT index management
- **iris-devtester:** container lifecycle, GOF fixture loading for tests
- **hipporag2-pipeline:** external RAG consumer — uses `ConnectionManager` and `IRISVectorStore`

## HUMAN APPROVAL REQUIRED

- Publishing to PyPI
- Force-pushing to `main` or `master`
- Deleting IRIS namespaces or schema tables
- Modifying security credentials in config
- Major version bumps

## LINKS

- [CLAUDE.md](CLAUDE.md) — Claude-specific context and constitutional rules
- [skills/iris-rag-pipeline/SKILL.md](skills/iris-rag-pipeline/SKILL.md) — Pipeline skill
- [skills/iris-vector-search/SKILL.md](skills/iris-vector-search/SKILL.md) — Vector search skill
- [specs/](specs/) — Feature specifications
- [iris-agentic-dev](https://github.com/intersystems-community/iris-agentic-dev) — ObjectScript MCP tools
- [iris-vector-graph](https://github.com/intersystems-community/iris-vector-graph) — GraphRAG layer
- [iris-devtester](https://github.com/intersystems-community/iris-devtester) — Container lifecycle
