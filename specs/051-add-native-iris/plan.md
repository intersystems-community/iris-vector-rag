
# Implementation Plan: IRIS EMBEDDING Support with Optimized Model Caching

**Branch**: `051-add-native-iris` | **Date**: 2025-01-06 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/Users/tdyar/ws/rag-templates/specs/051-add-native-iris/spec.md`

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

Add native IRIS EMBEDDING data type support to rag-templates to solve critical performance issue (DP-442038) where embedding model reloads for each row cause 720x slowdown. Implement optimized model caching using double-checked locking pattern (from Feature 050) to reduce 1,746 row vectorization from 20 minutes to <30 seconds. Extend with entity extraction capabilities for GraphRAG pipelines to automatically build knowledge graphs during vectorization.

**Core Requirements**: (1) Cache SentenceTransformer models in memory with >95% hit rate, (2) Integrate with IRIS %Embedding.Config for configuration management, (3) Extract entities during vectorization for GraphRAG, (4) Support all RAG pipelines with EMBEDDING-based auto-vectorization option.

## Technical Context
**Language/Version**: Python 3.11+ (existing framework)
**Primary Dependencies**: sentence-transformers, torch (GPU support), InterSystems IRIS 2025.3+, iris-vector-graph (for GraphRAG integration)
**Storage**: InterSystems IRIS database with %Embedding.Config table and EMBEDDING column support
**Testing**: pytest with live IRIS database (constitutional requirement), contract tests (TDD), integration tests with actual models
**Target Platform**: Linux/macOS server with CUDA/MPS GPU support (optional, CPU fallback)
**Project Type**: single (RAG framework extension)
**Performance Goals**: 50x improvement over baseline (20min → 30sec for 1,746 rows), >95% cache hit rate, <5s model load time, >80% GPU utilization when available
**Constraints**: Thread-safe caching for concurrent requests, graceful degradation when GPU memory exhausted, zero embedding failures for valid input
**Scale/Scope**: Support 10K+ document bulk loads, concurrent vectorization across multiple IRIS processes, medical domain entity extraction (Disease, Symptom, Medication types)

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**I. Framework-First Architecture**:
- ✓ Component designed as reusable EMBEDDING integration layer (not pipeline-specific)
- ✓ No application-specific logic (generic entity extraction, model caching)
- ✓ CLI interface exposed via Make targets and direct Python module execution

**II. Pipeline Validation & Requirements**:
- ✓ Automated %Embedding.Config validation before table creation (FR-010)
- ✓ Setup procedures idempotent (model cache can be rebuilt, configs re-validated)

**III. Test-Driven Development**:
- ✓ Contract tests written before implementation (model cache, entity extraction)
- ✓ Performance tests for 10K+ scenarios with actual IRIS EMBEDDING columns
- ✓ Live IRIS database required for integration tests (constitutional requirement)

**IV. Performance & Enterprise Scale**:
- ✓ Incremental indexing supported (EMBEDDING columns auto-vectorize on INSERT/UPDATE)
- ✓ IRIS vector operations optimized (leverages native EMBEDDING type, no manual vector storage)
- ✓ Concurrent operations supported (thread-safe model cache, FR-004)

**V. Production Readiness**:
- ✓ Structured logging included (model loading events, cache stats, FR-022)
- ✓ Health checks implemented (config validation, model availability checks)
- ✓ Docker deployment ready (extends existing IRIS containers)

**VI. Explicit Error Handling**:
- ✓ No silent failures (all embedding errors logged with context, FR-026)
- ✓ Clear exception messages (invalid config errors, FR-023)
- ✓ Actionable error context (row ID, text hash, model name in errors)

**VII. Standardized Database Interfaces**:
- ✓ Uses proven SQL utilities for %Embedding.Config queries
- ✓ No ad-hoc IRIS queries (standardized config table access pattern)
- ✓ New EMBEDDING integration patterns will be contributed back to framework

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
├── embeddings/
│   ├── manager.py           # Existing: Model cache (Feature 050)
│   ├── iris_embedding.py    # NEW: IRIS EMBEDDING integration layer
│   └── entity_extractor.py  # NEW: Entity extraction for GraphRAG
├── config/
│   ├── embedding_config.py  # NEW: EMBEDDING configuration models
│   └── default_config.yaml  # UPDATED: Add EMBEDDING settings
├── pipelines/
│   ├── basic_rag.py         # UPDATED: Add EMBEDDING support
│   ├── crag.py              # UPDATED: Add EMBEDDING support
│   ├── graphrag.py          # UPDATED: Add EMBEDDING + entity extraction
│   └── ...                  # All pipelines updated
└── storage/
    ├── iris_vector_store.py # UPDATED: Query %Embedding.Config table
    └── iris_embedding_ops.py # NEW: EMBEDDING column operations

tests/
├── contract/
│   ├── iris_embedding_contract.yaml  # NEW: EMBEDDING integration contract
│   └── entity_extraction_contract.yaml  # NEW: Entity extraction contract
├── integration/
│   ├── test_iris_embedding_e2e.py   # NEW: End-to-end EMBEDDING tests
│   └── test_embedding_cache_perf.py # EXTENDED: Performance tests
└── unit/
    ├── test_iris_embedding_config.py  # NEW: Config validation tests
    └── test_entity_extraction.py      # NEW: Entity extraction logic tests
```

**Structure Decision**: Single project structure (Option 1). This feature extends the existing iris_rag framework with new EMBEDDING integration modules. Main additions are in `iris_rag/embeddings/` for IRIS EMBEDDING integration, `iris_rag/config/` for configuration models, and updates across all pipelines to support EMBEDDING-based vectorization.

## Phase 0: Outline & Research
1. **Extract unknowns from Technical Context** above:
   - For each NEEDS CLARIFICATION → research task
   - For each dependency → best practices task
   - For each integration → patterns task

2. **Generate and dispatch research agents**:
   ```
   For each unknown in Technical Context:
     Task: "Research {unknown} for {feature context}"
   For each technology choice:
     Task: "Find best practices for {tech} in {domain}"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all NEEDS CLARIFICATION resolved

## Phase 1: Design & Contracts
*Prerequisites: research.md complete*

1. **Extract entities from feature spec** → `data-model.md`:
   - Entity name, fields, relationships
   - Validation rules from requirements
   - State transitions if applicable

2. **Generate API contracts** from functional requirements:
   - For each user action → endpoint
   - Use standard REST/GraphQL patterns
   - Output OpenAPI/GraphQL schema to `/contracts/`

3. **Generate contract tests** from contracts:
   - One test file per endpoint
   - Assert request/response schemas
   - Tests must fail (no implementation yet)

4. **Extract test scenarios** from user stories:
   - Each story → integration test scenario
   - Quickstart test = story validation steps

5. **Update agent file incrementally** (O(1) operation):
   - Run `.specify/scripts/bash/update-agent-context.sh claude`
     **IMPORTANT**: Execute it exactly as specified above. Do not add or remove any arguments.
   - If exists: Add only NEW tech from current plan
   - Preserve manual additions between markers
   - Update recent changes (keep last 3)
   - Keep under 150 lines for token efficiency
   - Output to repository root

**Output**: data-model.md, /contracts/*, failing tests, quickstart.md, agent-specific file

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
The /tasks command will load `.specify/templates/tasks-template.md` as the base and generate tasks from Phase 1 design documents:

1. **Contract Test Tasks** (from contracts/*.yaml):
   - iris_embedding_contract.yaml → `test_iris_embedding_contract.py` [P]
   - entity_extraction_contract.yaml → `test_entity_extraction_contract.py` [P]
   - Each contract has 8-10 test scenarios → individual test functions

2. **Model/Entity Creation Tasks** (from data-model.md):
   - EmbeddingConfig model → `iris_rag/config/embedding_config.py` [P]
   - CachedModelInstance dataclass → `iris_rag/embeddings/manager.py` (extend existing)
   - EntityExtractionResult model → `iris_rag/embeddings/entity_extractor.py` [P]
   - ValidationResult model → `iris_rag/config/embedding_config.py` [P]

3. **Integration Module Tasks** (from data-model.md relationships):
   - IRIS EMBEDDING integration → `iris_rag/embeddings/iris_embedding.py`
   - Entity extractor → `iris_rag/embeddings/entity_extractor.py`
   - Config validation → `iris_rag/config/embedding_config.py`
   - EMBEDDING operations → `iris_rag/storage/iris_embedding_ops.py`

4. **Pipeline Integration Tasks** (from spec.md FR-012):
   - Update BasicRAGPipeline for EMBEDDING support
   - Update CRAGPipeline for EMBEDDING support
   - Update HybridGraphRAGPipeline for EMBEDDING + entity extraction
   - Update PyLateColBERTPipeline for EMBEDDING support
   - Update BasicRAGRerankingPipeline for EMBEDDING support

5. **Integration Test Tasks** (from quickstart.md workflows):
   - test_iris_embedding_e2e.py: End-to-end EMBEDDING workflow
   - test_embedding_cache_perf.py: Verify 95% cache hit rate (extend existing)
   - test_entity_extraction_graphrag.py: Entity extraction → GraphRAG integration
   - test_bulk_vectorization.py: 1,746 rows in <30 seconds benchmark

6. **Performance Benchmark Tasks** (from contracts):
   - benchmark_cache_hit_performance: 50x improvement target
   - benchmark_10k_documents: Enterprise scale test
   - benchmark_batch_extraction_efficiency: Batch vs single extraction
   - benchmark_entity_extraction_accuracy: 85% accuracy target

**Ordering Strategy**:
1. **TDD Order**: Contract tests before implementation
   - Task 1-2: Write contract tests (failing tests)
   - Task 3-10: Implement models/services to pass contract tests
   - Task 11-15: Integration tests
   - Task 16-20: Pipeline updates
   - Task 21-25: Performance benchmarks

2. **Dependency Order**:
   - Models first (EmbeddingConfig, EntityExtractionResult)
   - Core services next (iris_embedding.py, entity_extractor.py)
   - Storage layer (iris_embedding_ops.py)
   - Pipeline integration (update all 5 pipelines)
   - End-to-end validation (quickstart workflow tests)

3. **Parallelization** ([P] marker):
   - Contract tests can run in parallel (independent files)
   - Model creation can happen in parallel (different modules)
   - Pipeline updates can happen in parallel (independent files)
   - Performance benchmarks can run in parallel (separate test functions)

**Estimated Task Breakdown**:
- Contract tests: 2 tasks ([P] parallelizable)
- Model/entity creation: 4 tasks ([P] parallelizable)
- Core implementation: 6 tasks (sequential dependencies)
- Pipeline updates: 5 tasks ([P] parallelizable)
- Integration tests: 4 tasks (sequential after implementation)
- Performance benchmarks: 4 tasks ([P] parallelizable)
- Documentation: 2 tasks (update READMEs, migration guide)

**Total Estimated Tasks**: 27 numbered, dependency-ordered tasks

**Key Constraints**:
- Contract tests MUST be written before any implementation (TDD)
- Performance benchmarks MUST verify targets (95% cache hit, 50x speedup, 85% entity accuracy)
- All pipelines MUST support EMBEDDING-based vectorization as optional mode
- Zero breaking changes to existing manual vectorization workflows

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
- [x] Phase 0: Research complete (/plan command) - See research.md
- [x] Phase 1: Design complete (/plan command) - data-model.md, contracts/*, quickstart.md, CLAUDE.md updated
- [x] Phase 2: Task planning complete (/plan command) - Described task generation approach (27 tasks estimated)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS (all 7 principles satisfied)
- [x] Post-Design Constitution Check: PASS (Phase 1 complete, no new violations)
- [x] All NEEDS CLARIFICATION resolved (none in Technical Context)
- [x] Complexity deviations documented (none - no constitutional violations)

---
*Based on Constitution v1.2.0 - See `/.specify/memory/constitution.md`*
