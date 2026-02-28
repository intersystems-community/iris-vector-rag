# Contract: iris_llm Provider Integration in get_llm_func

## Overview
`get_llm_func(provider="iris_llm", ...)` adds `iris_llm` as a first-class LLM provider alongside `openai` and `stub`. All IVR pipelines that accept an `llm_func` can use `iris_llm` transparently.

## Location
`iris_vector_rag/common/utils.py`

## Interface

```python
get_llm_func(
    provider: str = "openai",   # "openai" | "stub" | "iris_llm"  ← new
    model_name: str = "gpt-4o-mini",
    enable_cache: bool | None = None,
    **kwargs,                   # base_url, api_key override, temperature, etc.
) -> Callable[[str], str]
```

## New Branch: provider="iris_llm"

**Behaviour**:
1. Try `from iris_llm import Provider; from iris_llm.langchain import ChatIris`
2. If `ImportError` → raise `ImportError` with install instructions (do NOT silently fall back to stub in `get_llm_func` — the caller asked explicitly)
3. Create `Provider.new_openai(api_key=OPENAI_API_KEY)` — or use `base_url` kwarg for gateway/NIM
4. Return `lambda prompt: ChatIris(...).invoke([HumanMessage(prompt)]).content`

## Contract Rules

1. `provider="iris_llm"` with `iris_llm` not installed → `ImportError`, not `ValueError`
2. `provider="iris_llm"` with missing `OPENAI_API_KEY` and no `base_url` kwarg → `ValueError` (same as `openai` branch)
3. `**kwargs` forwarded: `base_url` maps to `Provider.new_openai(base_url=...)` for gateway override
4. Return type is always `Callable[[str], str]` — same as all other branches

## IrisLLMDSPyAdapter Contract

```python
class IrisLLMDSPyAdapter(dspy.BaseLM):
    """DSPy-compatible LM adapter wrapping iris_llm.ChatIris."""
```

**Required attributes** (DSPy reads these):
- `model: str` — set in `super().__init__(model=model)`
- `provider: str` — hardcoded `"openai"` for DSPy tool-call compatibility
- `kwargs: dict` — `{"model": model, ...kwargs}`

**Required methods**:
- `__call__(prompt=None, messages=None, **kwargs) -> list[str]`
- `basic_request(prompt: str, **kwargs) -> list[str]`

**Contract Rules**:
1. Returns `list[str]` — DSPy expects a list even for single responses
2. `messages` format: `[{"role": "user"|"system", "content": str}]`
3. Does NOT cache — DSPy handles caching above this layer
4. Thread safety: do not share one instance across threads in DSPy parallel optimization
