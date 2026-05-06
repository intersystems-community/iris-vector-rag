# Tasks: DSPy Optimization Layer

**Spec**: 073-dspy-optimization-layer
**Created**: 2026-05-06

## Task Dependency Graph

```
T001 (_optional.py) ─────────────┐
T002 (signatures.py) ────────────┤
T003 (modules.py) ───────────────┼── T004 (basic.py dual-path)
T005 (pyproject.toml [dspy]) ────┘         │
                                           ├── T006-T012 (all other pipelines)
                                           │
T013 (RelevanceJudge in crag.py) ──────────┤── requires T004 pattern
T014 (QueryExpander in multi_query_rrf) ───┘
T015 (EntityExtractor wiring) ─────────────── requires T002
                                           
T016 (compiler.py) ────── T017 (__main__.py CLI)
                                           
T018 (bundled programs) ─── requires T016 + API key + dataset
                                           
T019-T023 (dspy_modules/ cleanup) ─── can run in parallel with T006-T012
                                           
T024 (factory.py wiring) ─── requires T004
T025 (tests) ─── requires all above
```

## Tasks

### T001: Create `_optional.py` helper [P1]
- **File**: `iris_vector_rag/_optional.py`
- **Action**: New file
- **Deps**: None
- **Acceptance**: `require_dspy("answer generation")` raises helpful ImportError when dspy not installed; returns dspy module when installed; caches result

### T002: Create `signatures.py` [P1]
- **File**: `iris_vector_rag/dspy_modules/signatures.py`
- **Action**: New file
- **Deps**: T001
- **Acceptance**: 4 signatures defined (AnswerGeneration, RelevanceJudge, QueryExpander, EntityExtractor); each has typed InputField/OutputField; importable without dspy via lazy gate

### T003: Create `modules.py` [P1]
- **File**: `iris_vector_rag/dspy_modules/modules.py`
- **Action**: New file
- **Deps**: T002
- **Acceptance**: `AnswerGenerationModule`, `RelevanceJudgeModule`, `QueryExpanderModule` classes; `load_program(path) -> dict` factory; all use ChainOfThought wrapper

### T004: Dual-path in `basic.py` [P1]
- **File**: `iris_vector_rag/pipelines/basic.py`
- **Action**: Modify
- **Deps**: T001, T003
- **Acceptance**: `__init__` accepts `optimized_program`; `_generate_answer` has two paths; without program: exact same behavior as today

### T005: Add `[dspy]` extra to pyproject.toml [P1]
- **File**: `pyproject.toml`
- **Action**: Modify
- **Deps**: None
- **Acceptance**: `pip install iris-vector-rag[dspy]` installs `dspy-ai>=2.0.0`

### T006: Dual-path in `basic_rerank.py` [P1] [P]
- **File**: `iris_vector_rag/pipelines/basic_rerank.py`
- **Deps**: T004 (pattern established)

### T007: Dual-path in `crag.py` [P1] [P]
- **File**: `iris_vector_rag/pipelines/crag.py`
- **Deps**: T004

### T008: Dual-path in `graphrag.py` [P1] [P]
- **File**: `iris_vector_rag/pipelines/graphrag.py`
- **Deps**: T004

### T009: Dual-path in `graphrag_merged.py` [P1] [P]
- **File**: `iris_vector_rag/pipelines/graphrag_merged.py`
- **Deps**: T004

### T010: Dual-path in `hybrid_graphrag.py` [P1] [P]
- **File**: `iris_vector_rag/pipelines/hybrid_graphrag.py`
- **Deps**: T004

### T011: Dual-path in `multi_query_rrf.py` [P1] [P]
- **File**: `iris_vector_rag/pipelines/multi_query_rrf.py`
- **Deps**: T004

### T012: Dual-path in `colbert_pylate/pylate_pipeline.py` [P1] [P]
- **File**: `iris_vector_rag/pipelines/colbert_pylate/pylate_pipeline.py`
- **Deps**: T004

### T013: RelevanceJudge in CRAG [P2]
- **File**: `iris_vector_rag/pipelines/crag.py`
- **Action**: Modify `RetrievalEvaluator` class
- **Deps**: T003, T007
- **Acceptance**: When DSPy module loaded, evaluator uses LLM-based relevance judgment; when not loaded, embedding similarity path unchanged

### T014: QueryExpander in multi_query_rrf [P2]
- **File**: `iris_vector_rag/pipelines/multi_query_rrf.py`
- **Action**: Modify `_generate_llm_variations`
- **Deps**: T003, T011
- **Acceptance**: When DSPy module loaded, query expansion uses optimized prompts; when not loaded, existing f-string unchanged

### T015: EntityExtractor wiring in services [P2]
- **File**: `iris_vector_rag/services/entity_extraction.py`
- **Action**: Modify to use `EntityExtractorModule` from modules.py
- **Deps**: T003
- **Acceptance**: Replace ad-hoc `dspy.Module()` with proper EntityExtractorModule; existing try/except pattern preserved

### T016: Create `compiler.py` [P2]
- **File**: `iris_vector_rag/dspy_modules/compiler.py`
- **Action**: New file
- **Deps**: T002, T003
- **Acceptance**: `compile_program(trainset_path, pipeline_type, output_path)` works with BootstrapFewShot; handles errors gracefully

### T017: Create CLI `__main__.py` [P2]
- **File**: `iris_vector_rag/dspy_modules/__main__.py`
- **Action**: New file
- **Deps**: T016
- **Acceptance**: `python -m iris_vector_rag.dspy compile --help` shows usage; `compile` subcommand works end-to-end

### T018: Compile bundled programs [P3]
- **Action**: Run compiler against HotpotQA/NaturalQuestions
- **Deps**: T016, API key, dataset
- **Acceptance**: JSON files in `dspy_programs/` directory; each <100KB; load successfully into pipelines

### T019: Rename entity_extraction_module.py → entity_extraction.py [P1] [P]
- **File**: `iris_vector_rag/dspy_modules/`
- **Deps**: None (can run in parallel with T006-T012)

### T020: Rename batch_entity_extraction.py → batch_extraction.py [P1] [P]
- **File**: `iris_vector_rag/dspy_modules/`
- **Deps**: None

### T021: Rename iris_llm_lm.py → iris_llm_adapter.py [P1] [P]
- **File**: `iris_vector_rag/dspy_modules/`
- **Deps**: None

### T022: Rewrite dspy_modules/__init__.py [P1]
- **File**: `iris_vector_rag/dspy_modules/__init__.py`
- **Deps**: T019-T021
- **Acceptance**: Clean exports; lazy imports via _optional.py; no crash without dspy

### T023: Update test imports [P1]
- **Files**: `tests/unit/test_graphrag_bug_fixes.py`, `tests/contract/test_entity_types_batch_extraction.py`, `tests/integration/test_iris_llm_external.py`, `tests/unit/test_iris_llm_substrate.py`
- **Deps**: T019-T021
- **Acceptance**: All existing tests pass with new module paths

### T024: Wire `optimized_program` through factory.py [P1]
- **File**: `iris_vector_rag/pipelines/factory.py`
- **Deps**: T004
- **Acceptance**: `create_pipeline('basic', optimized_program='path/to/program.json')` works

### T025: Test suite [P1]
- **Files**: New test files
- **Deps**: All above
- **Acceptance**:
  - Unit: signatures importable, modules instantiate, compile mock works
  - Integration: dual-path selects correctly based on `optimized_program`
  - Regression: `pytest tests/` passes without dspy installed (critical gate)

---

## Execution Order

**Week 1 — Foundation (T001-T005, T019-T023)**:
- Create _optional.py, signatures.py, modules.py
- Add [dspy] extra to pyproject.toml
- Rename/reorganize existing dspy_modules/ files
- Update test imports

**Week 2 — All Pipelines (T004, T006-T012, T024)**:
- Implement dual-path in basic.py (reference implementation)
- Apply same pattern to all other pipelines (parallelizable)
- Wire through factory.py

**Week 3 — Specialized + CLI (T013-T017)**:
- RelevanceJudge in CRAG
- QueryExpander in multi_query_rrf
- EntityExtractor wiring
- Compiler + CLI

**Week 4 — Bundled Programs + Polish (T018, T025)**:
- Compile bundled programs
- Full test suite
- Documentation updates
