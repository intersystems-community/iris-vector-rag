# Data Model: Feature 065 — iris_llm as IVR LLM Substrate

## New Entities / Protocols

### SqlExecutor (Protocol)
**File**: `iris_vector_rag/executor.py`
**Type**: `typing.Protocol` (runtime-checkable)

| Attribute | Type | Description |
|---|---|---|
| `execute(sql, params)` | `(str, Any) -> list[dict]` | Execute SQL, return rows as list of dicts |

**Implementors**:
- `MockSqlExecutor` (tests) — in-memory, configurable responses
- `IrisSyncWrapperExecutor` (ai-hub, spec 014) — wraps `iris_sync_wrapper.execute_sql_query_dict`

**Relationships**: Consumed by `GraphRAGPipeline._execute_sql()` when injected at construction.

---

### GraphRAGPipeline (modified)
**File**: `iris_vector_rag/pipelines/graphrag.py`
**Change**: Add optional `executor: SqlExecutor | None = None` constructor parameter. Add `_execute_sql(sql, params) -> list[dict]` private helper.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `executor` | `SqlExecutor \| None` | `None` | When supplied, all SQL routes through this instead of DBAPI cursor |

**State transitions**: None — purely additive. Existing `connection_manager` path unchanged when `executor=None`.

**`_execute_sql` dispatch logic**:
```
if executor → executor.execute(sql, params)
else → connection_manager.get_connection() → cursor → fetchall → dict conversion
```

---

### GraphRAGToolSet
**File**: `iris_vector_rag/tools/graphrag.py`
**Type**: `iris_llm.ToolSet` subclass
**Availability**: Only when `iris_llm` is installed

| Attribute | Type | Description |
|---|---|---|
| `_executor` | `SqlExecutor` | Injected at construction |
| `_pipeline` | `HybridGraphRAGPipeline` | Owned instance, created from executor |

**Tools exposed** (via `@tool` decorator):

| Tool name | Signature | Description |
|---|---|---|
| `search_entities` | `(query: str, limit: int = 5) -> str` | Entity search, returns JSON |
| `traverse_relationships` | `(entity_text: str, max_depth: int = 2) -> str` | Multi-hop traversal, returns JSON |
| `hybrid_search` | `(query: str, top_k: int = 5) -> str` | RRF-fused vector+graph search, returns JSON |

**Relationships**: Owns `HybridGraphRAGPipeline`; receives `SqlExecutor` from caller. Used by `FHIRGraphRAGTool` in `ai-hub`.

---

### IrisLLMDSPyAdapter
**File**: `iris_vector_rag/dspy_modules/iris_llm_lm.py`
**Type**: `dspy.BaseLM` subclass

| Attribute | Type | Description |
|---|---|---|
| `_chat` | `iris_llm.langchain.ChatIris` | Wrapped LangChain chat model |
| `provider` | `str` | `"openai"` — DSPy compatibility |
| `kwargs` | `dict` | DSPy-required metadata dict |
| `model` | `str` | Model name string |

**Methods**:
- `__call__(prompt, messages, **kwargs) -> list[str]`
- `basic_request(prompt, **kwargs) -> list[str]`

**Relationships**: Registered via `dspy.configure(lm=IrisLLMDSPyAdapter(...))` before DSPy module compilation. Wraps `ChatIris` which wraps `iris_llm.Provider`.

---

### get_llm_func modifications
**File**: `iris_vector_rag/common/utils.py`

**`get_llm_func(provider=...)` new provider branch**:

| `provider` value | Behaviour |
|---|---|
| `"openai"` | Existing — `ChatOpenAI` via `langchain_openai` |
| `"stub"` | Existing — in-memory stub |
| `"iris_llm"` | **New** — `ChatIris` via `iris_llm.langchain`; requires `OPENAI_API_KEY` or `base_url` override |

**`get_llm_func_for_embedded()` real implementation** (replaces stub):

| Step | Behaviour |
|---|---|
| Try `iris_llm` | `Provider.new_openai()` + `ChatIris` |
| Fallback | `get_llm_func(provider="stub")` with warning log |

---

### iris_globals (thin wrapper)
**File**: `iris_vector_rag/common/iris_globals.py`

| Function | Signature | Description |
|---|---|---|
| `gset(*path, value)` | `(*str, value: str) -> None` | Wraps `iris.gset`; no-op if `iris` not installed |
| `gget(*path)` | `(*str) -> str \| None` | Wraps `iris.gget`; returns `None` if not available |

**State**: Stateless — pure thin wrappers with `try/except ImportError` guards.

---

## File Structure (new and modified)

```text
iris_vector_rag/
├── executor.py                          # NEW: SqlExecutor Protocol
├── __init__.py                          # MODIFIED: export SqlExecutor
├── common/
│   ├── utils.py                         # MODIFIED: iris_llm provider branch, real get_llm_func_for_embedded
│   └── iris_globals.py                  # NEW: iris.gset/gget wrapper
├── dspy_modules/
│   └── iris_llm_lm.py                   # NEW: IrisLLMDSPyAdapter
├── pipelines/
│   └── graphrag.py                      # MODIFIED: executor param + _execute_sql helper
└── tools/
    ├── __init__.py                      # NEW: optional submodule init + import guard
    └── graphrag.py                      # NEW: GraphRAGToolSet

tests/
├── unit/
│   ├── test_sql_executor.py             # NEW: SqlExecutor protocol + MockSqlExecutor + pipeline wiring
│   ├── test_graphrag_toolset.py         # NEW: GraphRAGToolSet with MockSqlExecutor
│   └── test_iris_llm_substrate.py       # NEW: get_llm_func iris_llm branch, IrisLLMDSPyAdapter (mocked)
└── integration/
    └── test_iris_llm_external.py        # NEW: real iris_llm wheel (skipped if not installed)
```
