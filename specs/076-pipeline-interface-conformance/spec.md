# Feature Specification: Pipeline Interface Conformance (AUD-006)

**Feature Branch**: `076-pipeline-interface-conformance`
**Created**: 2026-07-19
**Status**: Draft

## Context

The README and docs claim "all six pipelines share the same interface." Research
found four concrete divergences:

1. `load_documents()` ABC declares `-> None` but `basic.py` and `crag.py` return a
   stats `Dict`. `graphrag.py` and `multi_query_rrf.py` return `None` (implicit).
2. `MultiQueryRRFPipeline.__init__` accepts `connection_manager`/`config_manager`
   but not `llm_func` or `embedding_func`; `create_pipeline()` passes none of the
   standard dependencies to it.
3. `sources` appears at the top level of `query()` results only when
   `include_sources=True` in basic/rerank; in `graphrag.py` it's always present;
   in `crag.py` it's inside `metadata` only.
4. `query()` ABC declares `query_text` as the param name; `basic.py` and
   `multi_query_rrf.py` use `query`, breaking callers that pass it as a keyword arg.

## User Scenarios & Testing

### User Story 1 — load_documents returns a consistent IngestionResult (Priority: P1)

A developer calls `load_documents()` on any pipeline and processes the result the
same way — checking `result.documents_loaded`, `result.documents_failed`. Today
some pipelines return a dict, some return None.

**Why this priority**: Data integrity — callers need to know if ingestion succeeded.
The P0 fix (AUD-001) already returns a dict from basic; the ABC must match reality,
and all pipelines must converge.

**Independent Test**: Call `load_documents()` on each of the 6 pipelines with a
mock vector store; assert all return an `IngestionResult`-shaped dict with
`documents_loaded`, `documents_failed` keys.

**Acceptance Scenarios**:

1. **Given** any of the 6 pipelines, **When** `load_documents()` succeeds, **Then** result is a dict with `documents_loaded >= 0` and `documents_failed == 0`.
2. **Given** any of the 6 pipelines, **When** `load_documents()` fails, **Then** result raises `IngestionError` (not returns None or silently succeeds).
3. **Given** `graphrag.py` or `multi_query_rrf.py`, **When** `load_documents()` completes, **Then** result matches the same shape as `basic.py`.

---

### User Story 2 — query() returns consistent keys on all pipelines (Priority: P1)

A developer calls `pipeline.query("What is RAG?")` on any pipeline and reliably
accesses `result["answer"]`, `result["contexts"]`, `result["sources"]`,
`result["metadata"]`, and `result["error"]`.

**Why this priority**: RAGAS evaluation and downstream tooling depend on consistent
response shape. Missing keys cause `KeyError` at integration time.

**Independent Test**: Call `query()` on each pipeline with a mock vector store and
mock LLM; assert all six return dicts containing all required keys.

**Acceptance Scenarios**:

1. **Given** any of the 6 pipelines, **When** `query()` returns, **Then** result contains `answer`, `retrieved_documents`, `contexts`, `sources`, `metadata`, `error` keys.
2. **Given** `crag.py`, **When** `query()` returns, **Then** `sources` is a top-level key (not buried inside `metadata`).
3. **Given** any pipeline called as `pipeline.query(query_text="What is RAG?")`, **Then** it does not raise `TypeError` from unknown keyword arg.

---

### User Story 3 — create_pipeline passes standard deps to all pipelines (Priority: P2)

A developer calls `create_pipeline('multi_query_rrf', llm_func=my_llm)` and
expects `my_llm` to be used. Today it's silently dropped.

**Why this priority**: Kwargs silently dropped is a trust failure; affects
multi_query_rrf and potentially others.

**Independent Test**: Call `create_pipeline('multi_query_rrf', llm_func=mock_fn)`,
assert the pipeline's internal llm_func is `mock_fn`.

**Acceptance Scenarios**:

1. **Given** `create_pipeline('multi_query_rrf', llm_func=my_func)`, **Then** `pipeline.llm_func is my_func`.
2. **Given** `create_pipeline('multi_query_rrf', connection_manager=cm, config_manager=cfg)`, **Then** pipeline uses those objects.
3. **Given** any pipeline type, **When** `create_pipeline(type, llm_func=x)` is called, **Then** `x` is used (not silently dropped).

---

### Edge Cases

- `load_documents()` with `documents=[]` — should return `documents_loaded=0, documents_failed=0` consistently.
- `query()` with no documents loaded — all pipelines should return `answer` key (possibly None), not raise `KeyError`.
- `query_text` vs `query` positional arg — must work as both positional and keyword.

## Requirements

### Functional Requirements

- **FR-001**: ABC `RAGPipeline.load_documents()` return type MUST be updated to `Dict[str, Any]` with required keys `documents_loaded: int`, `documents_failed: int`.
- **FR-002**: `graphrag.py` and `multi_query_rrf.py` `load_documents()` MUST return the same stats dict shape as `basic.py`.
- **FR-003**: All six `query()` methods MUST return a dict containing `answer`, `retrieved_documents`, `contexts`, `sources`, `metadata`, `error` keys.
- **FR-004**: `crag.py query()` MUST promote `sources` to a top-level key.
- **FR-005**: ABC `query()` param MUST be named `query_text`; all pipelines MUST match that name (or accept both via compat alias).
- **FR-006**: `MultiQueryRRFPipeline.__init__` MUST accept `llm_func` and `embedding_func` params.
- **FR-007**: `create_pipeline()` MUST pass `connection_manager`, `config_manager`, `llm_func`, and `embedding_func` to all pipeline constructors that accept them.
- **FR-008**: A parameterized conformance test suite MUST run all 6 pipelines through construction, ingestion (mocked store), and query (mocked store + LLM) asserting response shape.

### Key Entities

- **IngestionResult**: `{documents_loaded: int, documents_failed: int, embeddings_generated: int}` — uniform return from all `load_documents()`.
- **QueryResult**: `{answer, retrieved_documents, contexts, sources, metadata, error}` — uniform return from all `query()`.

## Success Criteria

### Measurable Outcomes

- **SC-001**: Parameterized conformance test passes for all 6 pipelines — 0 failures.
- **SC-002**: `mypy --strict` on `core/base.py` reports no errors related to return types.
- **SC-003**: `create_pipeline('multi_query_rrf', llm_func=x).llm_func is x` passes as an assertion.
- **SC-004**: All 219+ existing unit tests continue to pass.
