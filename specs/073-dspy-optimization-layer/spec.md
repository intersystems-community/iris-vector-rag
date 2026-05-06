# Feature Specification: DSPy Optimization Layer

**Feature Branch**: `073-dspy-optimization-layer`
**Created**: 2026-05-06
**Status**: Draft
**Priority**: High
**Input**: OpenExchange reviewer feedback (bloat/clarity), existing spec-063 DSPy entity extraction work, codebase audit showing disconnected dspy_modules/

## Executive Summary

Add a horizontal DSPy optimization layer that improves every pipeline's LLM calls without changing the user-facing API. Users who install `pip install iris-vector-rag[dspy]` get access to pre-compiled optimized prompts and the ability to compile custom ones against their own data. Users without DSPy installed see zero difference — the existing code paths remain unchanged.

## Problem Statement

Today:
- Every pipeline has a `_generate_answer()` method with a hand-rolled f-string prompt
- Prompt quality is fixed at author time — no data-driven improvement possible
- `dspy_modules/` exists but is disconnected from all pipelines (orphaned from spec-063)
- `optimization/` exists but is never imported by any pipeline
- DSPy isn't even listed in pyproject.toml (optional or required)
- The reviewer correctly identified this as "code written for AI agents, not for humans"

After this spec:
- Every pipeline LLM call point is declared as a DSPy Signature
- Without DSPy: exact same f-string path as today (zero regression)
- With DSPy: loads compiled programs with better instructions + few-shot demos
- One command to compile custom prompts against user's own Q&A data
- `dspy_modules/` is cleaned up, reorganized, and properly wired

## User Scenarios & Testing

### User Story 1 — Zero-Config Improvement (Priority: P1)

A developer installs `pip install iris-vector-rag[dspy]` and points their pipeline at the bundled pre-compiled program. Answer quality improves without any training data or configuration changes.

**Acceptance Scenarios**:

1. **Given** a user has `iris-vector-rag[dspy]` installed, **When** they create any pipeline with `optimized_program="bundled"`, **Then** the pipeline uses pre-compiled DSPy prompts for answer generation
2. **Given** a user does NOT have dspy installed, **When** they create any pipeline normally, **Then** behavior is identical to today — no import errors, no warnings, no changes
3. **Given** a pre-compiled program is loaded, **When** the pipeline generates an answer, **Then** the prompt includes optimized instructions and few-shot examples from the compiled JSON

---

### User Story 2 — Custom Compilation (Priority: P2)

A data scientist with domain-specific Q&A pairs wants to compile optimized prompts for their use case (e.g., medical, legal, financial). They run a single CLI command and get a JSON file they can load into any pipeline.

**Acceptance Scenarios**:

1. **Given** a JSONL file with `{question, answer, contexts}` entries, **When** the user runs `python -m iris_vector_rag.dspy compile --trainset data.jsonl --pipeline basic --output my_rag.json`, **Then** DSPy compiles an optimized program and saves it
2. **Given** a compiled program file, **When** the user passes `optimized_program="my_rag.json"` to `create_pipeline()`, **Then** all LLM calls in that pipeline use the optimized prompts
3. **Given** compilation fails (API error, bad data), **Then** clear error messages explain what went wrong and how to fix it

---

### User Story 3 — Per-Pipeline Specialized Modules (Priority: P2)

An advanced user wants to optimize not just answer generation but also CRAG's relevance evaluation and multi_query_rrf's query expansion.

**Acceptance Scenarios**:

1. **Given** CRAG pipeline with DSPy enabled, **When** a relevance judge module is compiled, **Then** CRAG uses the LLM-based relevance scorer instead of embedding similarity alone
2. **Given** multi_query_rrf with DSPy enabled, **When** a query expander module is compiled, **Then** query generation uses optimized prompts instead of the hard-coded f-string
3. **Given** graphrag with DSPy enabled, **When** an entity extractor module is compiled, **Then** entity extraction uses the optimized DSPy program (same as existing spec-063 work)

---

### User Story 4 — Graceful Degradation (Priority: P1)

A production deployment configured with DSPy encounters issues (missing API key, corrupted program file, dspy not installed). The system must continue working with zero downtime.

**Acceptance Scenarios**:

1. **Given** `optimized_program` points to a non-existent file, **When** pipeline initializes, **Then** logs a warning and falls back to default f-string prompts
2. **Given** DSPy is configured but the import fails (not installed), **When** pipeline initializes, **Then** raises a clear `ImportError` with message: `Install with: pip install "iris-vector-rag[dspy]"`
3. **Given** compilation is interrupted, **When** user re-runs, **Then** previous partial state is recoverable (checkpointing)

---

## Requirements

### Functional Requirements

- **FR-001**: All 6 pipelines MUST support loading pre-compiled DSPy programs via `optimized_program` parameter in `create_pipeline()`
- **FR-002**: A `_optional.py` helper MUST gate DSPy imports with actionable error messages
- **FR-003**: Four DSPy Signatures MUST be defined: `AnswerGeneration`, `RelevanceJudge`, `QueryExpander`, `EntityExtractor`
- **FR-004**: Each pipeline's `_generate_answer` MUST have a two-path implementation (DSPy path + fallback path)
- **FR-005**: A CLI command MUST exist for compiling optimized programs: `python -m iris_vector_rag.dspy compile`
- **FR-006**: Pre-compiled "bundled" programs MUST ship with the package for immediate use
- **FR-007**: The `[dspy]` extra in pyproject.toml MUST include `dspy-ai>=2.0.0`
- **FR-008**: Existing `dspy_modules/entity_extraction_module.py` and `batch_entity_extraction.py` MUST be reorganized under the new structure
- **FR-009**: `iris_llm_lm.py` (IrisLLM DSPy adapter) MUST be preserved and documented

### Non-Functional Requirements

- **NFR-001**: Zero performance regression for users without DSPy installed
- **NFR-002**: Import time for `iris_vector_rag` MUST NOT increase (lazy imports only)
- **NFR-003**: Compiled program files MUST be JSON (human-readable, version-controllable)
- **NFR-004**: Compilation using BootstrapFewShot MUST work with as few as 10 examples

### Constraints

- **C-001**: DSPy is OPTIONAL — base package MUST work without it
- **C-002**: No changes to the public API of `create_pipeline()` or `pipeline.query()` beyond the new `optimized_program` parameter
- **C-003**: The existing f-string prompts remain the default — DSPy is an opt-in upgrade

## Data Model

### DSPy Signatures (4 canonical)

```
AnswerGeneration
  Input:  context: list[str], question: str
  Output: answer: str

RelevanceJudge
  Input:  question: str, passages: list[str]
  Output: status: Literal["confident", "ambiguous", "disoriented"], reasoning: str

QueryExpander
  Input:  question: str
  Output: queries: list[str]

EntityExtractor
  Input:  text: str, entity_types: list[str]
  Output: entities: list[str], relationships: list[str]
```

### Compiled Program File Format (JSON)

```json
{
  "version": "1.0",
  "compiled_with": "dspy.MIPROv2",
  "pipeline": "basic",
  "modules": {
    "answer_generation": {
      "signature": "AnswerGeneration",
      "demos": [...],
      "instructions": "..."
    }
  },
  "metadata": {
    "trainset_size": 50,
    "metric": "SemanticF1",
    "score": 0.61,
    "compiled_at": "2026-05-06T12:00:00Z"
  }
}
```

### File Structure (target state)

```
iris_vector_rag/
  _optional.py                    # NEW: require_dspy(), require_torch() helpers
  dspy_modules/
    __init__.py                   # REWRITE: clean exports
    signatures.py                 # NEW: 4 canonical signatures
    modules.py                    # NEW: ChainOfThought wrappers for each signature
    compiler.py                   # NEW: CLI compiler logic
    __main__.py                   # NEW: `python -m iris_vector_rag.dspy compile`
    entity_extraction.py          # MOVED from entity_extraction_module.py (cleaned up)
    batch_extraction.py           # MOVED from batch_entity_extraction.py (cleaned up)
    iris_llm_adapter.py           # MOVED from iris_llm_lm.py (cleaned up)
  dspy_programs/
    bundled_basic.json            # Pre-compiled: basic pipeline
    bundled_crag.json             # Pre-compiled: CRAG pipeline
    bundled_graphrag.json         # Pre-compiled: GraphRAG pipeline
```

## Pipeline Integration Points

| Pipeline | Module used | Trigger |
|---|---|---|
| basic | `AnswerGeneration` | Always (if DSPy loaded) |
| basic_rerank | `AnswerGeneration` | Always (if DSPy loaded) |
| crag | `AnswerGeneration` + `RelevanceJudge` | Answer: always; Relevance: opt-in |
| graphrag | `AnswerGeneration` + `EntityExtractor` | Answer: always; Entities: opt-in |
| graphrag_merged | `AnswerGeneration` + `EntityExtractor` | Same as graphrag |
| hybrid_graphrag | `AnswerGeneration` + `EntityExtractor` | Same as graphrag |
| multi_query_rrf | `AnswerGeneration` + `QueryExpander` | Answer: always; Expander: opt-in |
| colbert_pylate | `AnswerGeneration` | Always (if DSPy loaded) |

## Success Criteria

- **SC-001**: All 6 pipelines accept `optimized_program` parameter without error
- **SC-002**: With bundled program loaded, answer generation uses optimized prompts (verifiable by inspecting prompt sent to LLM)
- **SC-003**: Without dspy installed, all existing tests pass unchanged (zero regression)
- **SC-004**: `python -m iris_vector_rag.dspy compile --help` shows usage instructions
- **SC-005**: A 50-example trainset compiles in <5 minutes on GPT-4o-mini
- **SC-006**: The `dspy_modules/` subpackage has <10 files (consolidated from current state)

## Dependencies

- `dspy-ai>=2.0.0` (optional, in `[dspy]` extra)
- Existing: `openai>=1.0.0` (for compilation against OpenAI models)
- Existing: `services/entity_extraction.py` (already has DSPy fallback pattern)

## Risks

- **R-001**: DSPy API stability — DSPy is pre-1.0 and changes frequently. Mitigation: pin version, abstract behind our own signatures.py
- **R-002**: Pre-compiled programs may degrade on different LLM providers. Mitigation: ship programs compiled for GPT-4o-mini (cheapest/most common), document how to recompile for other models.
- **R-003**: Users may expect DSPy to magically improve retrieval quality — it only optimizes LLM prompts, not vector search. Mitigation: clear docs about what DSPy does and doesn't do.
