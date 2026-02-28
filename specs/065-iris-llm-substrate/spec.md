# Feature 065: iris_llm as IVR LLM Substrate

**Branch**: 065-iris-llm-substrate  
**Status**: Design / Experimental  
**Visibility**: Private only — do NOT merge to public or publish to PyPI until aicore/AI Hub architecture stabilizes

---

## Context and Motivation

This feature emerged from a broader architectural discussion about how `iris-vector-rag` (IVR), `iris-vector-graph` (IVG), `iris_llm`, and the `%AI` ObjectScript framework (from `aicore`) fit together in the same IRIS instance.

Key findings from that discussion:

1. **`iris_llm` is a Rust extension** (`_iris_llm.abi3.so`) backed by `rzf` — the Rust↔IRIS FFI bridge. It supports two modes:
   - **External**: Direct HTTP to LLM APIs (OpenAI, Anthropic, Bedrock, NIM, etc.)
   - **Embedded**: Routes through IRIS `%AI.Provider` natively via rzf when running as embedded Python inside IRIS (`%SYS.Python`)

2. **IVR currently uses `ChatOpenAI` / a plain `Callable[[str], str]`** for all LLM calls. The `get_llm_func_for_embedded()` function in `common/utils.py` is an explicit stub placeholder for this integration.

3. **The OAuth2 path**: When IRIS AI Gateway implements OAuth2 flows, `iris_llm` handles it via `Provider.new('openai', {base_url: <gateway_url>, api_key: <bearer>})`. DBAPI cannot do this. `iris_llm` is the right layer for that auth flow.

4. **IVR/IVG and `%AI.Agent`** all talk to the same IRIS tables (`RAG.SourceDocuments`, `KG.Nodes`, `KG.Edges`). IRIS is the coordination layer. No MCP server or separate process bridge is needed when running embedded.

5. **DSPy integration**: `iris_llm.ChatIris` is a LangChain `BaseChatModel`. DSPy supports LangChain models. IVR already has a `DirectOpenAILM(dspy.BaseLM)` custom adapter — the same pattern applies for `iris_llm`.

---

## Scope of This Feature (IVR-side only)

This feature covers changes **inside `iris-vector-rag-private` only**. The aicore/ObjectScript side (`Sample.AI.Tools.IVR`, `%SYS.Python` bridge) is **out of scope** — that work belongs in `aicore` and depends on AI Hub architecture decisions that are still in flux.

### In Scope

1. **`provider="iris_llm"` in `get_llm_func()`** — Add `iris_llm` as a first-class provider option alongside `openai` and `stub`. Uses `ChatIris` from `iris_llm.langchain`.

2. **`IrisLLMDSPyAdapter(dspy.BaseLM)`** — DSPy-compatible LM adapter for `iris_llm`, following the same pattern as `DirectOpenAILM` already in `dspy_modules/entity_extraction_module.py`. Lets DSPy modules (`TrakCareEntityExtractionModule`, etc.) use `iris_llm` as their LM.

3. **`get_llm_func_for_embedded()` implementation** — Replace the stub in `common/utils.py` with a real implementation using `iris_llm.Provider`. Falls back to stub if `iris_llm` is not installed.

4. **Optional dependency** — `iris_llm` is NOT added to `pyproject.toml` as a required dependency. It is an optional extra (`pip install iris-vector-rag[iris_llm]`). The wheel is not on PyPI so this references a local path or a private registry.

5. **`iris.gset` globals integration (light)** — Store DSPy compiled program metadata in `^IVR.Programs` globals when `iris` module is available (same pattern as `aicore/python/rlm/toolset.py`). Allows `%AI.Agent` ObjectScript to discover available compiled programs.

6. **`SqlExecutor` Protocol** — A `typing.Protocol` (runtime-checkable) with a single method `execute(sql: str, params=None) -> list[dict]`. Lives in `iris_vector_rag` core (no `iris_llm` dep). Allows any consumer (`ai-hub`, `iris_agent`) to supply an existing IRIS connection without IVR needing to know about DBAPI or `iris_llm` connection objects.

7. **`GraphRAGPipeline` accepts `SqlExecutor`** — Add an optional `executor: SqlExecutor | None` constructor parameter to the base class (`iris_vector_rag/pipelines/graphrag.py`). When supplied, all SQL routes through `executor.execute()` instead of the pipeline's own DBAPI connection. Existing connection path unchanged when `executor` is not supplied. `HybridGraphRAGPipeline` inherits this automatically.

8. **`GraphRAGToolSet` in `iris_vector_rag.tools`** — An `iris_llm.ToolSet` subclass in a new optional submodule `iris_vector_rag/tools/graphrag.py`. Accepts a `SqlExecutor` at construction, owns a `HybridGraphRAGPipeline` instance, exposes `search_entities`, `traverse_relationships`, and `hybrid_search` as `@tool`-decorated methods. Only importable when `iris_llm` is installed (same optional guard pattern as items 1–3 above). Consumer packages (`ai-hub`, `iris_agent`) import this — no toolset code lives in the consumer.

### Out of Scope

- Any ObjectScript classes (`Sample.AI.Tools.IVR`, etc.) — belongs in `aicore`
- MCP server changes
- `iris-vector-graph` changes
- Public release / PyPI publish
- Embedded Python `%SYS.Python` setup/docs (IRIS instance config)
- OAuth2 token acquisition flow (lives in `iris_llm` Rust core, not IVR)
- `ai-hub` wiring (`IrisSyncWrapperExecutor` adapter, `fhir_graphrag.py` changes) — separate spec in `ai-hub/specs/014-graphrag-toolset`

---

## Architecture

```
get_llm_func(provider="iris_llm", ...)
    └── ChatIris(Provider.new_openai(...))      # external mode
    └── ChatIris(Provider.new("bedrock", ...))   # external, no key needed
    └── ChatIris(Provider.new("openai", {        # OAuth2 via IRIS AI Gateway
            base_url: gateway_url,
            api_key: bearer_token
        }))
    └── ChatIris(Provider.new_embedded(...))     # embedded mode (future wheel)

IrisLLMDSPyAdapter(dspy.BaseLM)
    └── wraps ChatIris._generate()
    └── registered via dspy.configure(lm=IrisLLMDSPyAdapter(...))
    └── used by TrakCareEntityExtractionModule and any future dspy.Module

get_llm_func_for_embedded()
    └── tries iris_llm.Provider (embedded wheel)
    └── falls back to stub (no iris_llm installed)
    └── same graceful degradation pattern as aicore RLM toolset
```

---

## Validation Experiment Results (2026-02-21)

Ran against `iris_llm-0.1.0-cp39-abi3-macosx_15_0_arm64.whl` from `../ai-hub/wheels/`:

| Test | Result |
|------|--------|
| `Provider.new_openai()` + `ChatIris` + live call | ✅ "hello iris_llm" |
| `ToolSet` + `@tool` schema generation | ✅ Correct JSON Schema |
| `IrisTool.from_toolset()` + `bind_tools()` + tool call | ✅ LLM called `vector_search` correctly |
| `base_url` override (NIM/Ollama/Gateway) | ✅ Provider created OK |
| Bedrock with IAM (no explicit key) | ✅ Provider created OK |
| OAuth2 via `openai` + `base_url` + bearer as `api_key` | ✅ Schema supports it |
| Embedded provider kinds (`iris`, `embedded`, etc.) | ❌ "Unknown or disabled" — not in v0.1.0 external wheel (expected) |

**Conclusion**: External mode is production-ready now. Embedded mode requires a wheel built against the IRIS install — same wheel slot, different build target.

---

## Supported Providers (v0.1.0)

| Kind | Auth | Notes |
|------|------|-------|
| `openai` | `api_key` (env: `OPENAI_API_KEY`) | `base_url` override for Gateway/NIM/Ollama |
| `anthropic` | `api_key` (env: `ANTHROPIC_API_KEY`) | |
| `gemini` | `api_key` (env: `GOOGLE_API_KEY`) | |
| `vertex` | `project_id` + service account | IAM via `GOOGLE_APPLICATION_CREDENTIALS` |
| `bedrock` | IAM (env: `AWS_REGION`) | No explicit key needed |
| `meta` | `api_key` (env: `LLAMA_API_KEY`) | |
| `grok` | `api_key` (env: `XAI_API_KEY`) | |
| `nim` | `api_key` optional (env: `NGC_API_KEY`) | `base_url` for self-hosted |

---

## Files to Change

| File | Change |
|------|--------|
| `iris_vector_rag/common/utils.py` | Add `provider="iris_llm"` branch to `get_llm_func()`, implement `get_llm_func_for_embedded()` |
| `iris_vector_rag/dspy_modules/iris_llm_lm.py` | New file: standalone `IrisLLMDSPyAdapter(dspy.BaseLM)` |
| `iris_vector_rag/dspy_modules/iris_llm_lm.py` | New file: standalone `IrisLLMDSPyAdapter` for reuse across all dspy modules |
| `iris_vector_rag/common/iris_globals.py` | New file: thin wrapper for `iris.gset`/`iris.gget` with graceful fallback |
| `iris_vector_rag/executor.py` | New file: `SqlExecutor` Protocol definition |
| `iris_vector_rag/pipelines/graphrag.py` | Add optional `executor: SqlExecutor \| None` parameter to `GraphRAGPipeline.__init__`; add `_execute_sql` helper; route SQL through it when supplied |
| `iris_vector_rag/tools/__init__.py` | New file: optional submodule init; top-level import guard for `iris_llm` |
| `iris_vector_rag/tools/graphrag.py` | New file: `GraphRAGToolSet(ToolSet)` with `search_entities`, `traverse_relationships`, `hybrid_search` |
| `pyproject.toml` | Add `[project.optional-dependencies] iris_llm = [...]` |
| `tests/unit/test_iris_llm_substrate.py` | New: unit tests for LLM substrate paths (mocked iris_llm) |
| `tests/unit/test_sql_executor.py` | New: unit tests for `SqlExecutor` protocol + `MockSqlExecutor` + pipeline wiring |
| `tests/unit/test_graphrag_toolset.py` | New: unit tests for `GraphRAGToolSet` with `MockSqlExecutor` (no IRIS needed) |
| `tests/integration/test_iris_llm_external.py` | New: integration tests against real iris_llm (skipped if wheel not installed) |

---

## Open Questions

1. **Embedded wheel distribution**: How will the IRIS-native `iris_llm` wheel be distributed to IVR users? PyPI is not an option (IRIS-install-specific binary). Private registry? Bundled with IRIS?

2. **`dspy.LM` vs `dspy.BaseLM`**: DSPy's LM API changed between versions. `DirectOpenAILM` uses `dspy.BaseLM` — need to verify `IrisLLMDSPyAdapter` works with the DSPy version pinned in `pyproject.toml`.

3. **Thread safety**: `iris_llm` Rust core uses tokio async runtime internally. IRIS embedded Python may have threading constraints (macOS: no multithreaded call-in per rzf docs). Need to verify single-threaded usage pattern.

4. **`iris.gset` for DSPy programs**: Is this actually useful before the ObjectScript bridge exists? Could defer to when `aicore` integration is further along.

5. ~~**`HybridGraphRAGPipeline` SQL introspection**: Need to confirm the pipeline's internal SQL calls are reachable via a single executor injection point — or whether multiple call sites need patching.~~ **RESOLVED**: All cursor calls are in `GraphRAGPipeline` (base class). `HybridGraphRAGPipeline` has one additional cursor site in `_get_document_content_for_entity` which is a read helper outside the query-path scope of this feature. Injection into the base class covers all query-path methods (`_find_seed_entities`, `_expand_neighborhood`, `_get_documents_from_entities`, `_validate_knowledge_graph`).

---

## Tasks

> **See [`tasks.md`](tasks.md) for the authoritative implementation schedule (T001–T044, 8 phases).**
>
> The task list below is superseded by `tasks.md` and retained only for historical context.
> Do **not** use these T-numbers during implementation — use `tasks.md` numbering exclusively.

<details>
<summary>Original draft task list (superseded)</summary>

- [ ] T001: Add `IrisLLMDSPyAdapter` in `iris_vector_rag/dspy_modules/iris_llm_lm.py`
- [ ] T002: Add `provider="iris_llm"` to `get_llm_func()` in `common/utils.py`
- [ ] T003: Implement real `get_llm_func_for_embedded()` with `iris_llm` fallback
- [ ] T004: Add `iris_vector_rag/common/iris_globals.py` (gset/gget wrapper)
- [ ] T005: Add `[iris_llm]` optional dependency to `pyproject.toml`
- [ ] T006: Add `SqlExecutor` Protocol in `iris_vector_rag/executor.py`
- [ ] T007: Wire `SqlExecutor` into `GraphRAGPipeline`
- [ ] T008: Add `iris_vector_rag/tools/graphrag.py` (`GraphRAGToolSet`)
- [ ] T009: Unit tests — LLM substrate (mocked iris_llm)
- [ ] T010: Unit tests — `SqlExecutor` + pipeline wiring (MockSqlExecutor, no IRIS)
- [ ] T011: Unit tests — `GraphRAGToolSet` (MockSqlExecutor, no IRIS)
- [ ] T012: Integration tests (real iris_llm, skip if not installed)
- [ ] T013: Update `CHANGELOG.md` and `README.md`

</details>
