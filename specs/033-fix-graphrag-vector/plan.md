# Implementation Plan: Fix GraphRAG Vector Retrieval Logic

**Branch**: `033-fix-graphrag-vector` | **Date**: 2025-10-06 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/033-fix-graphrag-vector/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path → ✅ COMPLETE
2. Fill Technical Context → ✅ COMPLETE
3. Fill Constitution Check section → ✅ COMPLETE
4. Evaluate Constitution Check → ✅ PASS
5. Execute Phase 0 → research.md → ✅ COMPLETE
6. Execute Phase 1 → contracts, data-model.md, quickstart.md → ✅ COMPLETE
7. Re-evaluate Constitution Check → ✅ PASS
8. Plan Phase 2 → Describe task generation approach → ✅ COMPLETE
9. STOP - Ready for /tasks command → ✅ READY
```

## Summary

**Primary Requirement**: Fix GraphRAG vector search retrieval which currently returns 0 documents despite having 2,376 valid documents with embeddings in RAG.SourceDocuments.

**Current State**: Schema is healthy (all tables exist), knowledge graph populated (22,305 entities, 90,298 relationships), but vector search returns empty results for all queries. RAGAS shows 0% context precision/recall.

**Technical Approach**: Investigate and fix the vector search retrieval logic in GraphRAG pipeline, focusing on:
1. Embedding dimension validation (384D all-MiniLM-L6-v2)
2. Vector search SQL query correctness
3. IRIS VECTOR_DOT_PRODUCT function usage
4. Top-K parameter configuration (default K=10)
5. Query embedding generation workflow
6. Result ranking and filtering logic

**Success Criteria**: Achieve context precision >30%, context recall >20%, retrieval returns documents for all relevant queries.

## Technical Context

**Language/Version**: Python 3.12
**Primary Dependencies**:
- iris_rag framework (RAGPipeline base class)
- sentence-transformers (all-MiniLM-L6-v2 embedding model)
- intersystems-iris-dbapi (IRIS database connectivity)
- ragas (evaluation framework)
- langchain (LLM integration)

**Storage**: InterSystems IRIS 2025.3.0 (vector database with VECTOR_DOT_PRODUCT)
**Testing**: pytest with contract/integration/e2e markers, RAGAS evaluation
**Target Platform**: Linux/macOS server (Docker IRIS on port 11972)
**Project Type**: Single project (RAG framework extension)
**Performance Goals**:
- Vector search <500ms p95 latency
- Support 2,376+ documents (current corpus size)
- Retrieval quality: >30% context precision, >20% context recall

**Constraints**:
- Must work with existing 384D embeddings (no reindexing)
- Cannot modify schema (Feature 032 already fixed)
- Must maintain backward compatibility with other pipelines
- IRIS VECTOR_DOT_PRODUCT requires specific SQL syntax

**Scale/Scope**:
- 2,376 existing documents with embeddings
- 22,305 entities, 90,298 relationships (knowledge graph)
- Fix applies to GraphRAG pipeline only (not basic/crag/rerank)

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**I. Framework-First Architecture**:
- ✓ Component extends RAGPipeline base class (GraphRAGPipeline already exists)
- ✓ No application-specific logic (pure framework fix)
- ✓ CLI interface exposed (via existing RAGAS evaluation scripts)

**II. Pipeline Validation & Requirements**:
- ✓ Automated requirement validation included (iris_rag.validation.requirements.GraphRAGRequirements)
- ✓ Setup procedures idempotent (schema already initialized)

**III. Test-Driven Development**:
- ✓ Contract tests written before implementation (will create vector_search_contract.py)
- ✓ Performance tests for 10K+ scenarios (using existing RAGAS framework)
- ✓ Must execute against live IRIS database (port 11972 Docker instance)

**IV. Performance & Enterprise Scale**:
- ✓ Incremental indexing supported (existing documents not modified)
- ✓ IRIS vector operations optimized (will use proven VECTOR_DOT_PRODUCT patterns)

**V. Production Readiness**:
- ✓ Structured logging included (FR-004: diagnostic logging when 0 results)
- ✓ Health checks implemented (existing IRIS connectivity validation)
- ✓ Docker deployment ready (existing docker-compose.yml)

**VI. Explicit Error Handling**:
- ✓ No silent failures (will raise clear exceptions on dimension mismatch, connection failures)
- ✓ Clear exception messages (FR-004 requirement)
- ✓ Actionable error context (log query embedding dims, document count, SQL query)

**VII. Standardized Database Interfaces**:
- ✓ Uses proven SQL/vector utilities (common/vector_sql_utils.py, common/iris_sql_utils.py)
- ✓ No ad-hoc IRIS queries (will follow established patterns)
- ✓ New patterns contributed back (if new vector search pattern discovered)

**Constitution Check Result**: ✅ PASS (all requirements satisfied)

## Project Structure

### Documentation (this feature)
```
specs/033-fix-graphrag-vector/
├── spec.md              # Feature specification (complete)
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (investigation findings)
├── data-model.md        # Phase 1 output (vector search data flow)
├── quickstart.md        # Phase 1 output (how to validate fix)
├── contracts/           # Phase 1 output (contract tests)
│   ├── vector_search_contract.md
│   └── ragas_validation_contract.md
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
iris_rag/
├── pipelines/
│   ├── graphrag.py             # GraphRAG pipeline (MODIFY: fix vector search)
│   └── graphrag_merged.py      # HybridGraphRAG (MODIFY: same fix)
├── storage/
│   └── vector_store_iris.py    # IRIS vector operations (VERIFY: correct usage)
├── core/
│   └── vector_store.py         # Abstract base (VERIFY: interface compliance)
└── validation/
    └── requirements.py         # GraphRAGRequirements (NO CHANGE)

common/
├── vector_sql_utils.py         # VERIFY: VECTOR_DOT_PRODUCT helpers
├── iris_sql_utils.py           # VERIFY: parameterized query helpers
└── database.py                 # VERIFY: connection management

tests/
├── contract/
│   ├── test_vector_search_contract.py      # NEW: vector search contract tests
│   └── test_ragas_validation_contract.py   # NEW: RAGAS acceptance tests
├── integration/
│   └── test_graphrag_vector_search.py      # NEW: integration tests
└── e2e/
    └── test_graphrag_pipeline_e2e.py       # MODIFY: add vector search validation

scripts/
├── simple_working_ragas.py     # EXISTING: RAGAS evaluation (use for validation)
└── test_graphrag_validation.py # EXISTING: GraphRAG smoke tests (use for validation)
```

**Structure Decision**: Single project structure - this is a bug fix within the existing iris_rag framework, modifying GraphRAG pipeline's vector search logic. No new major components, just fixing existing retrieval code.

## Phase 0: Outline & Research

**Status**: ✅ COMPLETE (leveraging Feature 032 investigation findings)

### Research Questions Addressed

1. **Why does vector search return 0 results?**
   - **Findings from Feature 032 investigation**:
     - Schema is healthy: RAG.SourceDocuments exists with 2,376 documents
     - All documents have non-null embeddings (384D)
     - Logs show "Vector search returned 0 results" for all queries
   - **Hypothesis**: Vector search SQL query or embedding comparison logic is broken

2. **What IRIS VECTOR_DOT_PRODUCT syntax is correct?**
   - **Research**: Review existing working pipelines (basic_rag, crag) for proven patterns
   - **Reference**: common/vector_sql_utils.py contains working examples
   - **Key**: IRIS requires specific SQL syntax for vector similarity search

3. **How to validate embedding dimensions match?**
   - **Requirement**: FR-005 - validate dimensions before querying
   - **Approach**: Compare query embedding dimension to stored embedding dimension
   - **Expected**: Both should be 384D for all-MiniLM-L6-v2 model

4. **What diagnostic logging is needed?**
   - **Requirement**: FR-004 - log detailed info when 0 results
   - **Log items**:
     - Query embedding dimensions
     - Total documents in RAG.SourceDocuments
     - SQL query executed
     - VECTOR_DOT_PRODUCT scores (if any returned)
     - Top-K parameter value

### Consolidated Findings → research.md

**Decision**: Fix vector search in GraphRAG pipeline by:
1. Reviewing working vector search in BasicRAG pipeline
2. Identifying differences in GraphRAG vector search implementation
3. Applying proven VECTOR_DOT_PRODUCT SQL patterns
4. Adding dimension validation
5. Adding diagnostic logging

**Rationale**:
- BasicRAG works with same database and embeddings
- GraphRAG must have implementation-specific bug
- Reuse proven patterns rather than inventing new approaches

**Alternatives Considered**:
- ❌ Rebuild embeddings (unnecessary - embeddings are valid)
- ❌ Change schema (unnecessary - schema is correct)
- ✅ Fix retrieval logic (minimal, targeted fix)

## Phase 1: Design & Contracts

**Status**: ✅ COMPLETE

### Data Model (data-model.md)

**Vector Search Data Flow**:
```
1. User Query (str) → Embedding Model → Query Embedding (384D vector)
2. Query Embedding → IRIS VECTOR_DOT_PRODUCT → Similarity Scores
3. RAG.SourceDocuments → Filter by score → Top-K Documents
4. Top-K Documents → Rank by relevance → Retrieved Contexts
```

**Key Entities**:
- **QueryEmbedding**: 384D float vector from all-MiniLM-L6-v2
- **DocumentEmbedding**: 384D float vector stored in RAG.SourceDocuments.embedding
- **SimilarityScore**: Float from VECTOR_DOT_PRODUCT (higher = more similar)
- **RetrievedDocument**: doc_id, text_content, metadata, similarity_score

**Validation Rules**:
- Query embedding MUST be 384D
- Document embeddings MUST be 384D
- K parameter MUST be configurable (default K=10)
- VECTOR_DOT_PRODUCT MUST return non-null scores

**State Transitions**:
1. Initial: Query received
2. Embedded: Query → QueryEmbedding (384D)
3. Searched: QueryEmbedding → SimilarityScores
4. Ranked: SimilarityScores → Top-K Documents
5. Retrieved: Documents returned to LLM

### Contracts (contracts/)

**Contract 1: Vector Search Correctness** (vector_search_contract.md)
```python
# Given: 2,376 documents with 384D embeddings in RAG.SourceDocuments
# When: Query "What are the symptoms of diabetes?" is embedded and searched
# Then: Vector search MUST return K=10 documents
# And: All returned documents MUST have similarity scores
# And: Scores MUST be sorted descending (most similar first)
# And: At least 1 document MUST be relevant to query
```

**Contract 2: Dimension Validation** (dimension_validation_contract.md)
```python
# Given: Query embedding generator (all-MiniLM-L6-v2)
# When: Query is embedded
# Then: Resulting vector MUST be exactly 384 dimensions
# And: If document embedding dimension != 384, MUST raise DimensionMismatchError
# And: Error message MUST include both dimensions and suggestion
```

**Contract 3: RAGAS Acceptance** (ragas_validation_contract.md)
```python
# Given: GraphRAG pipeline with fixed vector search
# When: RAGAS evaluation runs on 5 test queries
# Then: Context precision MUST be >30%
# And: Context recall MUST be >20%
# And: Overall performance MUST improve from 14.4% baseline
# And: All queries MUST retrieve at least 1 document
```

**Contract 4: Diagnostic Logging** (diagnostic_logging_contract.md)
```python
# Given: Vector search executes and returns 0 results
# When: Logging level is INFO or DEBUG
# Then: Log MUST contain:
#   - "Vector search returned 0 results"
#   - "Query embedding dimensions: {dims}"
#   - "Total documents in RAG.SourceDocuments: {count}"
#   - "SQL query: {sql}"
#   - "Top-K parameter: {k}"
```

### Quickstart (quickstart.md)

**How to Validate the Fix**:
```bash
# 1. Ensure IRIS running on port 11972 with test data
docker ps | grep iris  # Should show container on port 11972

# 2. Verify test data exists
.venv/bin/python -c "
from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager
config = ConfigurationManager()
conn = ConnectionManager(config).get_connection()
cursor = conn.cursor()
cursor.execute('SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL')
print(f'Documents with embeddings: {cursor.fetchone()[0]}')
"
# Expected output: Documents with embeddings: 2376

# 3. Run contract tests (should FAIL before fix, PASS after)
.venv/bin/pytest tests/contract/test_vector_search_contract.py -v

# 4. Run RAGAS evaluation (should show >30% precision, >20% recall after fix)
IRIS_HOST=localhost IRIS_PORT=11972 RAGAS_PIPELINES="graphrag" \\
  .venv/bin/python scripts/simple_working_ragas.py

# 5. Verify results in outputs/reports/ragas_evaluations/
cat outputs/reports/ragas_evaluations/simple_ragas_report_*.json | \\
  python -m json.tool | grep -A 5 '"graphrag"'
# Expected: context_precision >0.3, context_recall >0.2
```

**Quick Smoke Test**:
```python
# Test vector search directly
from iris_rag import create_pipeline

pipeline = create_pipeline("graphrag", validate_requirements=True)
result = pipeline.query("What are the symptoms of diabetes?")

print(f"Contexts retrieved: {len(result.contexts)}")  # Should be >0
print(f"Answer: {result.answer[:100]}...")  # Should not be "No relevant documents"
```

## Phase 2: Task Generation Plan

**Approach**: Generate tasks.md with TDD workflow:

1. **Contract Test Tasks** (write tests first):
   - T001: Create test_vector_search_contract.py (expect FAIL)
   - T002: Create test_dimension_validation_contract.py (expect FAIL)
   - T003: Create test_ragas_validation_contract.py (expect FAIL)
   - T004: Create test_diagnostic_logging_contract.py (expect FAIL)

2. **Investigation Tasks** (understand the bug):
   - T005: Compare GraphRAG vector search with BasicRAG vector search
   - T006: Identify differences in VECTOR_DOT_PRODUCT SQL usage
   - T007: Review common/vector_sql_utils.py for proven patterns
   - T008: Document root cause in investigation notes

3. **Implementation Tasks** (fix the bug):
   - T009: Fix vector search SQL query in GraphRAGPipeline.query()
   - T010: Add embedding dimension validation (FR-005)
   - T011: Add diagnostic logging when 0 results (FR-004)
   - T012: Make top-K configurable (FR-006, default K=10)
   - T013: Apply same fixes to HybridGraphRAGPipeline

4. **Validation Tasks** (verify the fix):
   - T014: Run contract tests → all PASS
   - T015: Run RAGAS evaluation → context precision >30%, recall >20%
   - T016: Run smoke tests on 5 sample queries
   - T017: Update investigation findings with resolution

5. **Documentation Tasks**:
   - T018: Update FINDINGS.md with fix details
   - T019: Add vector search debugging guide to docs/
   - T020: Document top-K configuration in README

**Task Dependencies**:
- T001-T004 (contract tests) → T005-T008 (investigation) → T009-T013 (implementation) → T014-T017 (validation) → T018-T020 (docs)

**Estimation**: ~8-12 hours total (TDD cycle: 2h contracts, 3h investigation, 4h implementation, 2h validation, 1h docs)

## Progress Tracking

- [x] Load feature spec
- [x] Fill Technical Context
- [x] Initial Constitution Check (PASS)
- [x] Phase 0: Research complete (leveraging Feature 032 findings)
- [x] Phase 0: research.md created with investigation findings
- [x] Phase 1: Contracts written (4 contract specifications)
- [x] Phase 1: data-model.md created with vector search flow
- [x] Phase 1: quickstart.md created with validation procedures
- [x] Phase 1: contracts/vector_search_contract.md created (VSC-001)
- [x] Phase 1: contracts/dimension_validation_contract.md created (DVC-002)
- [x] Phase 1: contracts/ragas_validation_contract.md created (RAG-003)
- [x] Phase 1: contracts/diagnostic_logging_contract.md created (LOG-004)
- [x] Post-Design Constitution Check (PASS)
- [x] Phase 2: Task generation approach documented
- [ ] Phase 2: tasks.md creation (blocked - requires /tasks command)
- [ ] Phase 3-4: Implementation & validation (blocked - requires tasks.md)

## Complexity Tracking

**Potential Violations**: None identified

**Justifications**: N/A

**Mitigations**:
- Reusing proven patterns from BasicRAG avoids complexity
- Contract-first TDD prevents scope creep
- Diagnostic logging aids future debugging

## Next Steps

✅ **Plan Phase Complete** - Ready for `/tasks` command

**To proceed**:
1. Run `/tasks` to generate tasks.md with full TDD breakdown
2. Begin implementation starting with T001 (contract tests)
3. Follow TDD cycle: red (failing tests) → green (passing implementation) → refactor (cleanup)
4. Validate with RAGAS after each major task completion
5. Document resolution in investigation findings

**Success Criteria Reminder**:
- All contract tests pass (T014)
- RAGAS shows context precision >30%, recall >20% (T015)
- GraphRAG retrieves documents for all relevant queries (T016)
- No regression in other pipelines (basic, crag, rerank)
