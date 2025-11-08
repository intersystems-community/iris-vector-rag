
# Implementation Plan: HippoRAG2 Pipeline Implementation

**Branch**: `049-implement-a-hipporag2` | **Date**: 2025-11-04 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/Users/intersystems-community/ws/rag-templates/specs/049-implement-a-hipporag2/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → If not found: ERROR "No feature spec at {path}"
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Project Type from file system structure or context (web=frontend+backend, mobile=app+api)
   → Set Structure Decision based on project type
3. Fill the Constitution Check section based on the content of the constitution document.
4. Evaluate Constitution Check section below
   → If violations exist: Document in Complexity Tracking
   → If no justification possible: ERROR "Simplify approach first"
   → Update Progress Tracking: Initial Constitution Check
5. Execute Phase 0 → research.md
   → If NEEDS CLARIFICATION remain: ERROR "Resolve unknowns"
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, agent-specific template file (e.g., `CLAUDE.md` for Claude Code, `.github/copilot-instructions.md` for GitHub Copilot, `GEMINI.md` for Gemini CLI, `QWEN.md` for Qwen Code or `AGENTS.md` for opencode).
7. Re-evaluate Constitution Check section
   → If new violations: Refactor design, return to Phase 1
   → Update Progress Tracking: Post-Design Constitution Check
8. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
9. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary

Implement a HippoRAG2 pipeline for the rag-templates framework that provides neurobiologically-inspired retrieval-augmented generation with improved multi-hop reasoning capabilities. The pipeline integrates deeply with InterSystems IRIS database and iris-vector-graph for optimized knowledge graph storage and traversal. Core capabilities include: (1) LLM-based entity extraction and relationship identification, (2) knowledge graph construction with vector embeddings for passages and entities, (3) multi-stage retrieval combining vector similarity and graph-based associative connections, and (4) question answering with support for both OpenAI API and local LLM deployments. The implementation must support enterprise scale (100K-1M documents, 1M-10M entities) with transaction-based checkpointing for reliability.

## Technical Context
**Language/Version**: Python 3.11+ (existing rag-templates framework requirement)
**Primary Dependencies**: InterSystems IRIS 2025.3+, iris-vector-graph 2.0+, LangChain, Sentence Transformers, OpenAI API, LiteLLM (for local LLM support)
**Storage**: InterSystems IRIS (vector tables for embeddings, iris-vector-graph tables for knowledge graph, metadata tables for checkpointing)
**Testing**: pytest with contract/integration/e2e markers, RAGAS evaluation framework, HotpotQA benchmark dataset
**Target Platform**: Linux/macOS server environments (Docker deployment supported)
**Project Type**: Single framework component (RAG pipeline implementation extending RAGPipeline base class)
**Performance Goals**: Index 10K documents in <1 hour (167 docs/min), retrieval <2s at 1M document scale, support concurrent queries
**Constraints**: Development-grade performance targets, transaction-based checkpointing with <5% overhead, retry logic with exponential backoff (max 3 attempts)
**Scale/Scope**: Enterprise scale (100K-1M documents, 1M-10M entities, 50M relationships), HotpotQA benchmark evaluation, multi-hop reasoning capability

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**I. Framework-First Architecture**:
- ✓ Component extends RAGPipeline base class (spec FR-034: integrate with pipeline factory pattern)
- ✓ No application-specific logic (framework component, reusable pipeline)
- ✓ CLI interface exposed (via create_pipeline() factory, Make targets)

**II. Pipeline Validation & Requirements**:
- ✓ Automated requirement validation included (spec FR-033: validate configuration at initialization)
- ✓ Setup procedures idempotent (spec FR-008a/b/c: checkpointing with resume capability)

**III. Test-Driven Development**:
- ✓ Contract tests written before implementation (Phase 1 will generate failing contract tests)
- ✓ Performance tests for 10K+ scenarios (spec NFR-004: 10K documents in <1 hour)

**IV. Performance & Enterprise Scale**:
- ✓ Incremental indexing supported (spec FR-009: incremental indexing without full reindex)
- ✓ IRIS vector operations optimized (spec FR-023/024/025: IRIS native vector search, iris-vector-graph integration)

**V. Production Readiness**:
- ✓ Structured logging included (spec FR-041: log warnings for failed documents, FR-039/040: progress visibility)
- ✓ Health checks implemented (spec FR-041a/b: operational counters with status endpoint)
- ✓ Docker deployment ready (existing rag-templates Docker infrastructure)

**VI. Explicit Error Handling**:
- ✓ No silent failures (spec FR-002a: retry logic with warnings, skip failed documents explicitly)
- ✓ Clear exception messages (design requirement from constitution)
- ✓ Actionable error context (spec FR-041: log warnings with document context)

**VII. Standardized Database Interfaces**:
- ✓ Uses proven SQL/vector utilities (will use existing IRISVectorStore, iris-vector-graph APIs)
- ✓ No ad-hoc IRIS queries (leverage framework abstractions)
- ✓ New patterns contributed back (checkpointing patterns will extend framework utilities)

**Initial Constitution Check: PASS** ✅ (All requirements satisfied by spec)

---

## Post-Design Constitution Check
*Re-evaluation after Phase 1 design artifacts*

**I. Framework-First Architecture**:
- ✅ Data model designed for reusable framework component (8 entities, clear abstractions)
- ✅ API contracts enforce standard interfaces (query(), load_documents() match RAGPipeline base class)
- ✅ No application-specific logic in design (all components generic, configurable)

**II. Pipeline Validation & Requirements**:
- ✅ Configuration validation specified in contracts (config.yaml with validation rules)
- ✅ Checkpoint schema ensures idempotent operations (indexing_progress table with session tracking)

**III. Test-Driven Development**:
- ✅ Contract tests defined in YAML (16 test cases across 3 contracts, all must fail initially)
- ✅ Performance test scenarios documented (NFR-004: 10K docs, NFR-005: 1M docs retrieval)

**IV. Performance & Enterprise Scale**:
- ✅ Incremental indexing via checkpointing (data model includes IndexingProgress entity)
- ✅ IRIS optimization strategies documented (batch inserts, vector indexes, iris-vector-graph for graph)

**V. Production Readiness**:
- ✅ Logging specified (tqdm progress, warnings for failures, operational counters)
- ✅ Health check capability (metrics API in contracts: queries_processed, documents_indexed)
- ✅ Docker deployment ready (quickstart includes IRIS Docker setup)

**VI. Explicit Error Handling**:
- ✅ No silent failures (contracts specify exception types for all error conditions)
- ✅ Clear exception messages (ValueError, RuntimeError, LLMAPIException with context)
- ✅ Actionable error context (retry logic, checkpoint recovery, troubleshooting guide in quickstart)

**VII. Standardized Database Interfaces**:
- ✅ Uses iris-vector-graph for KG (data model specifies iris-vector-graph tables)
- ✅ No ad-hoc queries (all IRIS interactions via defined schemas and IRISVectorStore)
- ✅ New checkpoint pattern documented (IndexingProgress schema, batch commit strategy)

**Post-Design Constitution Check: PASS** ✅ (No design violations, ready for implementation)

**Complexity Tracking**: No violations, no deviations to document

---

## Project Structure

### Documentation (this feature)
```
specs/[###-feature]/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
iris_rag/
├── pipelines/
│   └── hipporag2_pipeline.py           # HippoRAG2Pipeline class extending RAGPipeline
├── core/
│   ├── rag_pipeline.py                 # Base class (existing)
│   └── models.py                       # Entity, Relationship, KnowledgeGraph models
├── services/
│   ├── entity_extraction.py           # LLM-based entity extraction with retry logic
│   ├── graph_builder.py               # Knowledge graph construction
│   ├── checkpoint_manager.py          # Transaction-based checkpointing
│   └── hipporag2_retrieval.py         # Multi-stage retrieval (vector + graph)
├── storage/
│   └── iris_kg_store.py               # IRIS knowledge graph storage via iris-vector-graph
├── config/
│   └── hipporag2_config.yaml          # Pipeline-specific configuration
└── evaluation/
    └── hotpotqa_evaluator.py          # HotpotQA benchmark evaluation

tests/
├── contract/
│   ├── test_hipporag2_pipeline_contract.py    # Pipeline interface contract
│   ├── test_entity_extraction_contract.py     # Entity extraction API contract
│   └── test_retrieval_contract.py             # Retrieval API contract
├── integration/
│   ├── test_hipporag2_indexing.py             # Full indexing workflow with IRIS
│   ├── test_hipporag2_retrieval.py            # Multi-hop retrieval with real KG
│   ├── test_checkpoint_resume.py              # Interruption and resume behavior
│   └── test_hotpotqa_evaluation.py            # HotpotQA benchmark integration
├── unit/
│   ├── test_entity_extraction.py              # Entity extraction logic (mocked LLM)
│   ├── test_graph_builder.py                  # Graph construction logic
│   └── test_checkpoint_manager.py             # Checkpointing logic (mocked IRIS)
└── e2e/
    └── test_hipporag2_e2e.py                  # End-to-end multi-hop QA workflow

config/
└── hipporag2_pipeline_config.yaml             # Pipeline registration config

docs/
└── hipporag2/
    ├── architecture.md                        # Architecture overview
    └── quickstart.md                          # Getting started guide
```

**Structure Decision**: Single project structure (Option 1). This is a framework component that extends the existing rag-templates architecture. The HippoRAG2 pipeline integrates into iris_rag/pipelines/ alongside existing pipelines (BasicRAG, CRAG, GraphRAG), reuses framework services and storage abstractions, and follows the established testing hierarchy (contract/integration/unit/e2e).

## Phase 0: Outline & Research ✅ COMPLETE

**Research Questions Addressed**:
1. HippoRAG2 architecture & three-store design (chunks, entities, facts)
2. Entity extraction with OpenIE (online/offline modes, retry logic)
3. Knowledge graph storage via iris-vector-graph
4. Multi-stage retrieval algorithm (entity linking → graph expansion → passage ranking)
5. Transaction-based checkpointing for resume capability
6. Embedding model flexibility (Sentence Transformers, OpenAI, custom)
7. Progress visibility with tqdm + basic operational counters
8. HotpotQA evaluation integration
9. Configuration management with YAML validation

**Key Architectural Decision**: Separate repository strategy - HippoRAG2 pipeline as standalone package consuming rag-templates as dependency (Framework-First Architecture principle)

**Output**: ✅ `research.md` (9 research questions, all decisions documented with rationale)

## Phase 1: Design & Contracts ✅ COMPLETE

**Data Model** (`data-model.md`):
- 8 core entities: Document, Passage, Entity, Relationship, KnowledgeGraph, Query, RetrievalResult, Answer
- 1 checkpoint entity: IndexingProgress
- 7 IRIS tables with schemas (passages, passage_embeddings, entities, entity_embeddings, relationships, indexing_progress, knowledge_graphs)
- Validation rules for cross-entity integrity and indexing invariants

**API Contracts** (`contracts/`):
- `hipporag2_pipeline_contract.yaml`: query() and load_documents() interface (RAGAS-compatible response format)
- `entity_extraction_contract.yaml`: extract_entities() with retry logic (spec FR-002a)
- `retrieval_contract.yaml`: retrieve() multi-stage retrieval (spec FR-012, FR-016, FR-017)

**Quickstart Guide** (`quickstart.md`):
- 10-section guide: Installation → IRIS setup → 9-document example → Multi-hop query → Checkpoint resume → HotpotQA evaluation → Metrics → Troubleshooting
- Validates spec acceptance scenario 2: "What county is Erik Hort's birthplace a part of?" → "Rockland County"

**Agent Context Update**:
- ✅ Executed `.specify/scripts/bash/update-agent-context.sh claude`
- Updated CLAUDE.md with Python 3.11+, IRIS 2025.3+, iris-vector-graph dependencies

**Contract Tests**: Ready for generation in Phase 2 (tests/contract/ directory)

**Output**: ✅ data-model.md, ✅ contracts/ (3 YAML files), ✅ quickstart.md, ✅ CLAUDE.md updated

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
1. Load `.specify/templates/tasks-template.md` as base
2. Generate tasks from Phase 1 design artifacts:
   - **From contracts/** (3 YAML files):
     * Contract test tasks for each interface (query, load_documents, extract_entities, retrieve)
     * Test implementation tasks (must fail initially per TDD requirement)
   - **From data-model.md** (9 entities):
     * IRIS table creation tasks (7 tables)
     * Entity model class tasks (8 core entities + 1 checkpoint)
     * Migration script task for schema versioning
   - **From quickstart.md** (10 sections):
     * Integration test for 9-document indexing scenario
     * Multi-hop query test (Erik Hort → Montebello → Rockland County)
     * Checkpoint resume test
     * HotpotQA evaluation test
   - **From research.md** (9 decisions):
     * Service implementation tasks (EntityExtraction, GraphBuilder, CheckpointManager, HippoRAG2Retriever)
     * Configuration validation task
     * Retry logic implementation task

**Task Categories**:
- **[P] Parallel Tasks**: Independent file creation (models, utils, configs)
- **[S] Sequential Tasks**: Dependent implementations (tests → services → pipeline)
- **[T] Test Tasks**: All contract/integration/e2e tests (TDD-first)
- **[I] Implementation Tasks**: Make failing tests pass

**Ordering Strategy** (TDD + Dependency Order):
1. **Phase 2.1 - Schema & Models** [P]:
   - Task 1-7: Create IRIS tables (passages, entities, relationships, checkpointing, etc.)
   - Task 8-16: Implement entity models (Document, Passage, Entity, Relationship, etc.)
2. **Phase 2.2 - Contract Tests** [T]:
   - Task 17-24: Write failing contract tests (pipeline, entity extraction, retrieval)
3. **Phase 2.3 - Core Services** [I] [S]:
   - Task 25-28: Implement EntityExtractionService (with retry logic FR-002a)
   - Task 29-31: Implement KnowledgeGraphBuilder
   - Task 32-34: Implement CheckpointManager (transaction-based FR-008a/b/c)
   - Task 35-38: Implement HippoRAG2Retriever (3-stage algorithm)
4. **Phase 2.4 - Pipeline Integration** [I] [S]:
   - Task 39-42: Implement HippoRAG2Pipeline class
   - Task 43-45: Implement configuration validation
   - Task 46-48: Integrate with rag-templates factory pattern
5. **Phase 2.5 - Integration Tests** [T]:
   - Task 49-52: Write integration tests (indexing, retrieval, checkpoint resume)
6. **Phase 2.6 - Evaluation** [I] [T]:
   - Task 53-55: Implement HotpotQAEvaluator
   - Task 56-58: Write HotpotQA evaluation tests
7. **Phase 2.7 - Documentation & Deployment**:
   - Task 59-60: Finalize quickstart examples
   - Task 61-62: Create Docker deployment configs

**Estimated Output**: 60-65 numbered, ordered tasks in tasks.md

**Dependency Graph** (Critical Path):
```
IRIS Schema → Entity Models → Contract Tests (fail) → Services → Pipeline → Integration Tests (pass) → E2E Tests (pass)
```

**Parallelization Opportunities**:
- Schema creation tasks (1-7) can run in parallel
- Model creation tasks (8-16) can run in parallel after schema
- Service implementations (25-38) can run in parallel after models
- Test writing can happen concurrently with implementation (TDD workflow)

**Performance Validation Tasks** (from NFR-004, NFR-005):
- Task 63: 10K document indexing performance test (<1 hour target)
- Task 64: 1M document retrieval latency test (<2s target)
- Task 65: Checkpoint overhead measurement test (<5% target)

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)  
**Phase 4**: Implementation (execute tasks.md following constitutional principles)  
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking
*Fill ONLY if Constitution Check has violations that must be justified*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |


## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command) ✅
- [x] Phase 1: Design complete (/plan command) ✅
- [x] Phase 2: Task planning complete (/plan command - describe approach only) ✅
- [x] Phase 3: Tasks generated (/tasks command) ✅ - **65 tasks across 12 phases**
- [ ] Phase 4: Implementation complete - **NEXT STEP**
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS ✅
- [x] Post-Design Constitution Check: PASS ✅
- [x] All NEEDS CLARIFICATION resolved (none in Technical Context) ✅
- [x] Complexity deviations documented (none) ✅

**Artifacts Generated**:
- [x] research.md (9 research questions, architectural decision)
- [x] data-model.md (9 entities, 7 IRIS tables, validation rules)
- [x] contracts/ (3 YAML files: pipeline, entity_extraction, retrieval)
- [x] quickstart.md (10-section getting started guide)
- [x] CLAUDE.md updated (agent context with new dependencies)
- [x] tasks.md (65 numbered tasks across 12 phases, TDD-ordered)

**Ready for Phase 4 (Implementation)**: ✅ All design and planning artifacts complete

---
*Based on Constitution v1.2.0 - See `/.specify/memory/constitution.md`*
