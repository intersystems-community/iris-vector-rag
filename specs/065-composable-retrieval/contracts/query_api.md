# Contract: Unified `query()` API

**Feature**: 065-composable-retrieval

## Signature (all registered pipelines)

```python
def query(
    self,
    query: str,                          # canonical; `query_text=` accepted as alias
    top_k: int = 5,
    *,
    generate_answer: bool = True,
    include_sources: bool = True,
    metadata_filter: dict | None = None,
    similarity_threshold: float = 0.0,
    retrieval: str | None = None,        # None -> pipeline default
    weights: dict[str, float] | None = None,
    rerank: bool | str | Callable | None = None,
    custom_prompt: str | None = None,
    **kwargs,
) -> dict:
```

## Behavioral contract

| ID | Given | When | Then |
|---|---|---|---|
| C-Q1 | Any registered pipeline | `query("q", top_k=5)` | Succeeds with identical param names/defaults across pipelines (FR-004). |
| C-Q2 | Caller uses legacy `query_text=` | `query(query_text="q")` | Works unchanged; `query_text` aliases `query` (FR-005). |
| C-Q3 | Both `query=` and `query_text=` passed | `query(query="a", query_text="b")` | Uses `"a"`; logs a warning. |
| C-Q4 | No composable options passed | `query("q")` | Reproduces the pipeline's pre-feature behavior exactly (FR-013). |
| C-Q5 | Any pipeline | inspect response | Contains `query, answer, retrieved_documents, contexts, metadata, execution_time` (+`sources` if requested) (FR-006). |
| C-Q6 | `top_k` out of [1,100] | `query("q", top_k=0)` | Raises the existing actionable `ValueError`. |

## Backward-compatibility assertions

- Existing test suite passes unchanged.
- `create_pipeline("basic").query("q")` returns the same documents/answer as before this feature (golden-response test).
