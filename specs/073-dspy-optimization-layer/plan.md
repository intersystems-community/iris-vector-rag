# Implementation Plan: DSPy Optimization Layer

**Spec**: 073-dspy-optimization-layer
**Created**: 2026-05-06

## Architecture Decision: Dual-Path Pattern

Every LLM call in the codebase gets a two-path pattern:

```python
def _generate_answer(self, query, documents):
    context = [d.page_content for d in documents]

    # Path A: DSPy (optimized prompts, few-shot demos)
    if self._dspy_answer_module is not None:
        return self._dspy_answer_module(context=context, question=query).answer

    # Path B: Default (existing f-string, unchanged)
    prompt = self._build_answer_prompt(query, context)
    return self.llm_func(prompt)
```

The `_dspy_answer_module` is initialized from `optimized_program` parameter at construction time. If DSPy is not installed or no program is provided, it stays `None` and the default path runs.

## Implementation Phases

### Phase 1: Foundation (P1 — must ship)

**Goal**: `_optional.py` + `signatures.py` + dual-path in `BasicRAGPipeline._generate_answer`

1. Create `iris_vector_rag/_optional.py`
   - `require_dspy(feature: str) -> module` — try/except with actionable error
   - Cache imported module in global var (avoid repeated imports)

2. Create `iris_vector_rag/dspy_modules/signatures.py`
   - Define `AnswerGeneration` signature
   - Define `RelevanceJudge` signature
   - Define `QueryExpander` signature
   - Define `EntityExtractor` signature (migrate from existing)

3. Create `iris_vector_rag/dspy_modules/modules.py`
   - `AnswerGenerationModule(dspy.Module)` wrapping `ChainOfThought(AnswerGeneration)`
   - `RelevanceJudgeModule(dspy.Module)` wrapping `ChainOfThought(RelevanceJudge)`
   - `QueryExpanderModule(dspy.Module)` wrapping `ChainOfThought(QueryExpander)`
   - Factory: `load_modules(program_path: str) -> dict[str, dspy.Module]`

4. Modify `iris_vector_rag/pipelines/basic.py`
   - Add `optimized_program: Optional[str] = None` parameter to `__init__`
   - In `__init__`: if optimized_program provided, lazy-load DSPy and create module
   - In `_generate_answer`: dual-path (DSPy module vs existing f-string)

5. Modify `iris_vector_rag/pipelines/factory.py`
   - Pass `optimized_program` through `create_pipeline()` to pipeline constructors

6. Update `pyproject.toml`
   - Add `[dspy]` extra: `dspy-ai>=2.0.0`

### Phase 2: All Pipelines (P1 — must ship)

**Goal**: Every pipeline's `_generate_answer` gets the dual-path treatment

7. Modify `basic_rerank.py` — add dual-path to `_generate_answer`
8. Modify `crag.py` — add dual-path to `_generate_answer`
9. Modify `graphrag.py` — add dual-path to `_generate_answer`
10. Modify `graphrag_merged.py` — add dual-path to `_generate_answer`
11. Modify `hybrid_graphrag.py` — add dual-path to `_generate_answer`
12. Modify `multi_query_rrf.py` — add dual-path to `_generate_answer`
13. Modify `colbert_pylate/pylate_pipeline.py` — add dual-path to `_generate_answer`

### Phase 3: Specialized Modules (P2 — high value)

**Goal**: CRAG gets RelevanceJudge, multi_query gets QueryExpander

14. Modify `crag.py` — add optional `RelevanceJudgeModule` to `RetrievalEvaluator`
    - When loaded: uses LLM to judge relevance (better than embedding sim alone)
    - When not loaded: existing embedding similarity behavior unchanged

15. Modify `multi_query_rrf.py` — add optional `QueryExpanderModule`
    - When loaded: uses compiled query expansion prompts
    - When not loaded: existing `_generate_llm_variations` f-string unchanged

16. Wire `EntityExtractor` module into `services/entity_extraction.py`
    - Replace ad-hoc dspy.Module() usage with proper EntityExtractorModule
    - Migrate logic from existing `dspy_modules/entity_extraction_module.py`

### Phase 4: Compiler CLI (P2 — enables self-serve)

**Goal**: Users can compile their own optimized programs

17. Create `iris_vector_rag/dspy_modules/compiler.py`
    - `compile_program(trainset_path, pipeline_type, output_path, optimizer, model)`
    - Supports BootstrapFewShot (default) and MIPROv2 (flag)
    - Trainset format: JSONL with `{question, answer, contexts}`

18. Create `iris_vector_rag/dspy_modules/__main__.py`
    - CLI: `python -m iris_vector_rag.dspy compile --trainset X --pipeline Y --output Z`
    - CLI: `python -m iris_vector_rag.dspy info program.json` (show metadata)

### Phase 5: Bundled Programs (P3 — polish)

**Goal**: Ship pre-compiled programs for immediate use

19. Compile bundled programs against open QA datasets (HotpotQA, NaturalQuestions)
    - `dspy_programs/bundled_basic.json`
    - `dspy_programs/bundled_crag.json`
    - `dspy_programs/bundled_graphrag.json`

20. Document "bundled" shortcut: `create_pipeline('basic', optimized_program="bundled")`

### Phase 6: Cleanup (P1 — part of bloat removal)

**Goal**: Reorganize existing dspy_modules/ into clean structure

21. Rename `entity_extraction_module.py` → `entity_extraction.py`
22. Rename `batch_entity_extraction.py` → `batch_extraction.py`
23. Rename `iris_llm_lm.py` → `iris_llm_adapter.py`
24. Rewrite `__init__.py` — clean exports only
25. Delete orphaned tests that reference old module names (update imports)

## Testing Strategy

- **Unit tests**: Each signature module works standalone (mock LLM)
- **Integration tests**: dual-path — verify DSPy path uses module, fallback uses f-string
- **Regression tests**: ALL existing tests pass without dspy installed (critical)
- **Compilation test**: BootstrapFewShot with 10 synthetic examples compiles without error

## Risk Mitigations

| Risk | Mitigation |
|---|---|
| DSPy API breaks | Pin `dspy-ai>=2.0.0,<3.0.0`; abstract behind our signatures.py |
| Import time regression | All DSPy imports are inside `_optional.py` — only triggered when `optimized_program` is set |
| Compiled programs model-specific | Document: "Bundled programs compiled for GPT-4o-mini. Recompile for your model." |
| Test flakiness with LLM calls | Mock all DSPy calls in unit tests; integration tests only in CI with API keys |
