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

### Out of Scope

- Any ObjectScript classes (`Sample.AI.Tools.IVR`, etc.) — belongs in `aicore`
- MCP server changes
- `iris-vector-graph` changes
- Public release / PyPI publish
- Embedded Python `%SYS.Python` setup/docs (IRIS instance config)
- OAuth2 token acquisition flow (lives in `iris_llm` Rust core, not IVR)

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
| `iris_vector_rag/dspy_modules/entity_extraction_module.py` | Add `IrisLLMDSPyAdapter(dspy.BaseLM)` class |
| `iris_vector_rag/dspy_modules/iris_llm_lm.py` | New file: standalone `IrisLLMDSPyAdapter` for reuse across all dspy modules |
| `iris_vector_rag/common/iris_globals.py` | New file: thin wrapper for `iris.gset`/`iris.gget` with graceful fallback |
| `pyproject.toml` | Add `[project.optional-dependencies] iris_llm = [...]` |
| `tests/unit/test_iris_llm_substrate.py` | New: unit tests for all new paths (mocked iris_llm) |
| `tests/integration/test_iris_llm_external.py` | New: integration tests against real iris_llm (skipped if wheel not installed) |

---

## Relationship to Broader Architecture

```
This feature (IVR-private)
  └── enables: iris_llm as IVR LLM substrate (external + future embedded)

aicore (separate, experimental)
  └── Sample.AI.Tools.IVR — %AI.ToolSet calling IVR via %SYS.Python
  └── Sample.AI.RLMAgent enhanced with IVR vector/graph search tools
  └── depends on: this feature being stable in IVR

AI Hub (architecture TBD)
  └── may absorb or replace aicore
  └── OAuth2 flow for IRIS AI Gateway → iris_llm Rust core
  └── does NOT affect IVR's use of iris_llm as substrate

iris-vector-graph (separate)
  └── IRISGraphEngine tools as iris_llm ToolSet — future, separate feature
```

---

## Open Questions

1. **Embedded wheel distribution**: How will the IRIS-native `iris_llm` wheel be distributed to IVR users? PyPI is not an option (IRIS-install-specific binary). Private registry? Bundled with IRIS?

2. **`dspy.LM` vs `dspy.BaseLM`**: DSPy's LM API changed between versions. `DirectOpenAILM` uses `dspy.BaseLM` — need to verify `IrisLLMDSPyAdapter` works with the DSPy version pinned in `pyproject.toml`.

3. **Thread safety**: `iris_llm` Rust core uses tokio async runtime internally. IRIS embedded Python may have threading constraints (macOS: no multithreaded call-in per rzf docs). Need to verify single-threaded usage pattern.

4. **`iris.gset` for DSPy programs**: Is this actually useful before the ObjectScript bridge exists? Could defer to when `aicore` integration is further along.

---

## Tasks

- [ ] T001: Add `IrisLLMDSPyAdapter` in `iris_vector_rag/dspy_modules/iris_llm_lm.py`
- [ ] T002: Add `provider="iris_llm"` to `get_llm_func()` in `common/utils.py`
- [ ] T003: Implement real `get_llm_func_for_embedded()` with `iris_llm` fallback
- [ ] T004: Add `iris_vector_rag/common/iris_globals.py` (gset/gget wrapper)
- [ ] T005: Add `[iris_llm]` optional dependency to `pyproject.toml`
- [ ] T006: Unit tests (mocked iris_llm)
- [ ] T007: Integration tests (real iris_llm, skip if not installed)
- [ ] T008: Update `CHANGELOG.md` and `README.md`
