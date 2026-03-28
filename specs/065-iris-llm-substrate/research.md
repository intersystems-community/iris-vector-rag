# Research: Feature 065 — iris_llm as IVR LLM Substrate

## Decision 1: SqlExecutor Protocol definition and location

**Decision**: Define `SqlExecutor` as a `@runtime_checkable` `Protocol` in `iris_vector_rag/executor.py`. Export from `iris_vector_rag/__init__.py`.

**Rationale**: `executor.py` is the standard location for adapter protocols in Python libraries (see SQLAlchemy's `_typing.py`, `databases` package). A dedicated file avoids circular imports. `@runtime_checkable` is needed only if we use `isinstance()` checks — we do in tests (MockSqlExecutor verification), so include it.

**Alternatives considered**:
- `protocols.py` — valid but overengineered for a single protocol
- Inline in `__init__.py` — clutters the public API surface
- `typing_extensions` backport — not needed, Python 3.12 has full Protocol support

**Canonical pattern**:
```python
# iris_vector_rag/executor.py
from __future__ import annotations
from typing import Any, Protocol, runtime_checkable

@runtime_checkable
class SqlExecutor(Protocol):
    """Bridge: any object that can execute SQL and return list[dict] rows."""
    def execute(self, sql: str, params: Any = None) -> list[dict]:
        ...
```

**DBAPI2 adapter** (lives in `ai-hub`, not IVR):
```python
class IrisSyncWrapperExecutor:
    def __init__(self, iris_sync_wrapper):
        self._sync = iris_sync_wrapper

    def execute(self, sql: str, params=None) -> list[dict]:
        return self._sync.execute_sql_query_dict(sql, params)
```

**MockSqlExecutor** for tests (lives in `tests/`):
```python
class MockSqlExecutor:
    def __init__(self, responses: dict[str, list[dict]]):
        self._responses = responses  # sql_fragment -> rows
        self.calls: list[tuple] = []

    def execute(self, sql: str, params=None) -> list[dict]:
        self.calls.append((sql, params))
        for fragment, rows in self._responses.items():
            if fragment in sql:
                return rows
        return []
```

**Caveats**: `isinstance(obj, SqlExecutor)` with `@runtime_checkable` only checks method existence, not signatures. This is acceptable — our tests verify behavior, not static types.

---

## Decision 2: SqlExecutor injection point in GraphRAGPipeline

**Decision**: Inject `SqlExecutor` at `GraphRAGPipeline.__init__` as an optional parameter. When supplied, replace `connection_manager.get_connection() → cursor.execute()` calls in `_search_entities_in_db`, `_traverse_relationships`, and `_get_chunks_for_entities` with a private helper `_execute_sql(sql, params) -> list[dict]` that dispatches to executor or cursor.

**Rationale**: All SQL in `GraphRAGPipeline` is concentrated in 4 methods (lines 232–301, 376–397). A single `_execute_sql` helper eliminates the need to touch each call site individually and makes the executor path testable without a live IRIS instance.

**Pattern**:
```python
def _execute_sql(self, sql: str, params=None) -> list[dict]:
    if self._executor is not None:
        return self._executor.execute(sql, params)
    # existing cursor path
    connection = self.connection_manager.get_connection()
    cursor = connection.cursor()
    try:
        cursor.execute(sql, params or [])
        cols = [d[0] for d in cursor.description]
        return [dict(zip(cols, row)) for row in cursor.fetchall()]
    finally:
        cursor.close()
```

**Alternatives considered**:
- Subclass `ConnectionManager` to inject executor — more invasive, breaks existing constructor signatures
- Monkey-patch connection object — fragile, not testable
- Pass executor only to `HybridGraphRAGPipeline` — SQL is in the base class, so the base must be patched

---

## Decision 3: GraphRAGToolSet location and optional import guard

**Decision**: `iris_vector_rag/tools/graphrag.py` with guard in `iris_vector_rag/tools/__init__.py`. Top-level `iris_vector_rag/__init__.py` does NOT import from `tools/` at all.

**Rationale**: This is the established pattern in `pandas` (optional backends), `sqlalchemy` (dialects), and `matplotlib` (backends). The submodule is only loaded on explicit import — `import iris_vector_rag` never touches `iris_vector_rag.tools`.

**`tools/__init__.py`**:
```python
try:
    from iris_vector_rag.tools.graphrag import GraphRAGToolSet
    __all__ = ["GraphRAGToolSet"]
except ImportError as e:
    raise ImportError(
        "iris_vector_rag.tools requires iris_llm. "
        "Install with: pip install iris_llm-*.whl"
    ) from e
```

**`tools/graphrag.py`** top:
```python
from __future__ import annotations
from typing import TYPE_CHECKING

try:
    from iris_llm import ToolSet, tool
except ImportError as e:
    raise ImportError(
        "iris_vector_rag.tools.graphrag requires iris_llm. "
        "Install the iris_llm wheel for your platform."
    ) from e

if TYPE_CHECKING:
    from iris_vector_rag.executor import SqlExecutor
```

**pyproject.toml** for non-PyPI wheel:
```toml
[project.optional-dependencies]
iris_llm = []  # wheel installed separately; document in README

# Or if using a local path (dev only):
# iris_llm = ["iris_llm @ file:///path/to/iris_llm-0.1.0-*.whl"]
```

The empty list is correct — `iris_llm` is a compiled wheel not on PyPI. The install instructions live in README. Attempting to encode a platform-specific wheel path in pyproject.toml is fragile (arm64 vs x86_64 differ).

---

## Decision 4: IrisLLMDSPyAdapter implementation

**Decision**: Follow the existing `DirectOpenAILM(dspy.BaseLM)` pattern exactly. Override `__call__` and `basic_request`. Use `ChatIris.invoke()` internally, converting LangChain `AIMessage` to DSPy's expected `list[str]` return format.

**Rationale**: The codebase already has a working `DirectOpenAILM(dspy.BaseLM)` at line 257 of `entity_extraction_module.py`. `IrisLLMDSPyAdapter` follows the identical interface — no DSPy internals to research.

DSPy `BaseLM` interface (confirmed from codebase):
- `__init__(model: str)` — call `super().__init__(model=model)`
- `__call__(prompt=None, messages=None, **kwargs) -> list[str]`
- `basic_request(prompt: str, **kwargs) -> list[str]`
- `provider` attribute (set to `"openai"` for compatibility)
- `kwargs` dict attribute (DSPy reads this internally)

**Pattern**:
```python
class IrisLLMDSPyAdapter(dspy.BaseLM):
    def __init__(self, chat_iris, model: str = "iris_llm", **kwargs):
        super().__init__(model=model)
        self._chat = chat_iris
        self.provider = "openai"  # DSPy compatibility
        self.kwargs = {"model": model, **kwargs}

    def __call__(self, prompt=None, messages=None, **kwargs):
        from langchain_core.messages import HumanMessage, SystemMessage
        if messages:
            lc_messages = [
                SystemMessage(content=m["content"]) if m["role"] == "system"
                else HumanMessage(content=m["content"])
                for m in messages
            ]
        else:
            lc_messages = [HumanMessage(content=prompt)]
        response = self._chat.invoke(lc_messages)
        return [response.content]

    def basic_request(self, prompt: str, **kwargs):
        return self(prompt=prompt, **kwargs)
```

**Thread safety**: `iris_llm` uses tokio async runtime internally but exposes a synchronous Python API. No special handling needed for single-threaded DSPy compilation. For parallel DSPy optimization (multi-threaded), create one `IrisLLMDSPyAdapter` per thread — do not share a single instance across threads.

---

## Decision 5: get_llm_func_for_embedded implementation

**Decision**: Replace the stub with a real implementation that tries `iris_llm.Provider` first, falls back to the existing `openai` path if `OPENAI_API_KEY` is set, then falls back to `stub`.

```python
def get_llm_func_for_embedded(provider: str = "iris_llm", model_name: str = "gpt-4o-mini"):
    try:
        from iris_llm import Provider
        from iris_llm.langchain import ChatIris
        iris_provider = Provider.new_openai(api_key=os.getenv("OPENAI_API_KEY", ""))
        chat = ChatIris(provider=iris_provider, model=model_name)
        def _call(prompt: str) -> str:
            from langchain_core.messages import HumanMessage
            return chat.invoke([HumanMessage(content=prompt)]).content
        return _call
    except ImportError:
        logger.warning("iris_llm not available for embedded; falling back to stub")
        return get_llm_func(provider="stub")
```

---

## Decision 6: iris.gset globals for DSPy programs (T004)

**Decision**: Defer. The `iris_globals.py` wrapper is a prerequisite for ObjectScript bridge discovery — that bridge doesn't exist yet. Include the file as a thin no-op wrapper that logs a warning when `iris` module is absent, but do not block T001–T008 on it.

---

## Open Questions Resolved

| Question | Resolution |
|---|---|
| `HybridGraphRAGPipeline` SQL injection point | Base class `GraphRAGPipeline` via `_execute_sql()` helper |
| `iris_llm` wheel pyproject.toml encoding | Empty optional extra `[iris_llm]`; install instructions in README |
| DSPy `BaseLM` vs `dspy.LM` | Use `dspy.BaseLM` — matches existing `DirectOpenAILM` pattern |
| `@runtime_checkable` on `SqlExecutor` | Yes — needed for `isinstance()` in tests |
| `tools/` submodule guard location | `tools/__init__.py` guards the import; `graphrag.py` raises on missing `iris_llm` |
| Thread safety for `IrisLLMDSPyAdapter` | One instance per thread for DSPy parallel optimization |
