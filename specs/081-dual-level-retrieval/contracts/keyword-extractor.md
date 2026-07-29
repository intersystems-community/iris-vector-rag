# Contract: KeywordExtractor

**Feature**: 081-dual-level-retrieval
**Date**: 2026-07-29
**Covers**: US3 — query-time keyword extraction as a reusable, tunable step

---

## Interface

```python
from iris_vector_rag.retrieval.keyword_extractor import KeywordExtractor

extractor = KeywordExtractor(
    llm_func=pipeline.llm_func,      # default: pipeline's LLM
    language="English",               # default
)

high_kws, low_kws = extractor.extract(query="What are the systemic risks across filings?")
# high_kws: ["systemic risk", "financial stability", "cross-filing themes"]
# low_kws:  ["Basel III", "LIBOR", "Tier 1 capital"]
```

---

## Input / Output

| Parameter                     | Type      | Default  | Notes                                            |
| ----------------------------- | --------- | -------- | ------------------------------------------------ |
| `query`                       | str       | required | The user query                                   |
| Returns `high_level_keywords` | List[str] | —        | Themes/concepts; empty list on vague query       |
| Returns `low_level_keywords`  | List[str] | —        | Entities/proper nouns; empty list on vague query |

---

## Prompt Format (LightRAG-compatible)

LLM is instructed to return a single flat JSON object — no markdown fencing:

```json
{
  "high_level_keywords": ["theme1", "concept2"],
  "low_level_keywords": ["Entity A", "Proper Noun B"]
}
```

Parsing strips accidental markdown fences before `json.loads()`.

---

## Error / Degradation Behavior

| Scenario                     | Behavior                                                                       |
| ---------------------------- | ------------------------------------------------------------------------------ |
| LLM returns valid JSON       | Returns parsed lists; empty arrays allowed                                     |
| LLM returns malformed JSON   | Falls back to `([], [])`, sets `degraded=True`                                 |
| LLM call times out or raises | Falls back to `([], [])`, sets `degraded=True`                                 |
| Both arrays empty            | Caller treats as degraded; `global`/`mix` falls back to entity-level retrieval |

`degraded` and `degradation_reason` are surfaced in query response `metadata` (not raised as exceptions).

---

## Model Configuration

```python
# Use a cheaper model for extraction, a different one for generation (US3 / FR-006)
extractor = KeywordExtractor(
    llm_func=cheap_llm_func,    # e.g., a faster/smaller model
)
pipeline.keyword_extractor = extractor   # injectable on any ComposableQueryMixin pipeline
```

When `pipeline.keyword_extractor` is not set, the pipeline's default `llm_func` is used.
The model actually invoked for extraction is recorded in `metadata["extraction_model"]`.
