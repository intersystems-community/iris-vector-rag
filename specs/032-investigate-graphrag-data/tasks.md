# Tasks: GraphRAG Data/Setup Investigation

**Feature**: 032-investigate-graphrag-data | **Branch**: `032-investigate-graphrag-data`
**Input**: Design documents from `/Users/intersystems-community/ws/rag-templates/specs/032-investigate-graphrag-data/`
**Prerequisites**: plan.md ✅, research.md ✅, data-model.md ✅, contracts/ ✅, quickstart.md ✅

## Execution Flow (main)
```
1. Load plan.md from feature directory
   → ✅ Tech stack: Python 3.12, pytest, IRIS database
   → ✅ Structure: Diagnostic scripts in /scripts, tests in /tests/contract
2. Load design documents:
   → ✅ data-model.md: 4 entities (KnowledgeGraphState, EntityExtractionStatus, etc.)
   → ✅ contracts/: 2 contract files (graph_inspector, entity_extraction_verification)
   → ✅ research.md: GraphRAG architecture, entity extraction, schema investigation
   → ✅ quickstart.md: 3-step investigation workflow
3. Generate tasks by category:
   → Setup: Environment verification (1 task)
   → Tests: 2 contract test files [P] (2 tasks)
   → Core: 3 diagnostic scripts (3 tasks - sequential for shared imports)
   → Investigation: Execute diagnostics and analyze (3 tasks)
   → Validation: Verify contracts pass, document findings (2 tasks)
4. Task rules applied:
   → Contract tests in different files = [P]
   → Diagnostic scripts share imports = sequential
   → Investigation tasks depend on scripts = sequential
5. Total tasks: 11 (T001-T011)
6. Parallel opportunities: T002-T003 (contract tests)
7. TDD enforced: Tests (T002-T003) before implementation (T004-T006)
8. ✅ Validation: All contracts have tests, investigation workflow complete
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in task descriptions

## Path Conventions
Single Python framework project:
- Scripts: `/Users/intersystems-community/ws/rag-templates/scripts/`
- Tests: `/Users/intersystems-community/ws/rag-templates/tests/contract/`
- Specs: `/Users/intersystems-community/ws/rag-templates/specs/032-investigate-graphrag-data/`

---

## Phase 3.1: Setup

- [ ] **T001** Verify development environment for diagnostic investigation
  - **File**: N/A (environment check)
  - **Action**: Verify IRIS database running, Python .venv active, iris_rag installed
  - **Tests**: Run `docker ps | grep iris`, `python -c "from iris_rag.services.entity_extraction import OntologyAwareEntityExtractor; print('OK')"`
  - **Success**: IRIS running, iris_rag importable, entity extraction service available

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3

**CRITICAL: These tests MUST be written and MUST FAIL before ANY diagnostic script implementation**

- [ ] **T002** [P] Write contract test for graph inspector in `/Users/intersystems-community/ws/rag-templates/tests/contract/test_graph_inspector_contract.py`
  - **Contract**: `/Users/intersystems-community/ws/rag-templates/specs/032-investigate-graphrag-data/contracts/graph_inspector_contract.md`
  - **Tests to implement**:
    - `test_graph_inspector_exists()` - Verify script file exists
    - `test_graph_inspector_executable()` - Verify script can be executed
    - `test_graph_inspector_output_format()` - Verify valid JSON output with required fields
    - `test_graph_inspector_empty_graph_detection()` - Verify exit code 1 when graph empty
    - `test_graph_inspector_tables_missing_detection()` - Verify exit code 2 when tables missing
    - `test_graph_inspector_sample_limit()` - Verify sample_entities ≤ 5
    - `test_graph_inspector_document_link_consistency()` - Verify linked + orphaned == total
    - `test_graph_inspector_completeness_score_bounds()` - Verify score 0.0-1.0
    - `test_graph_inspector_suggestions_on_error()` - Verify suggestions when exit != 0
    - `test_graph_inspector_connection_error_handling()` - Verify exit code 3 on connection error
  - **Success**: All 10 tests FAIL with "FileNotFoundError: scripts/inspect_knowledge_graph.py"
  - **Dependencies**: None
  - **Parallel**: Can run with T003

- [ ] **T003** [P] Write contract test for entity extraction verifier in `/Users/intersystems-community/ws/rag-templates/tests/contract/test_entity_extraction_contract.py`
  - **Contract**: `/Users/intersystems-community/ws/rag-templates/specs/032-investigate-graphrag-data/contracts/entity_extraction_verification_contract.md`
  - **Tests to implement**:
    - `test_entity_extraction_verifier_exists()` - Verify script file exists
    - `test_entity_extraction_verifier_executable()` - Verify script can be executed
    - `test_entity_extraction_verifier_output_format()` - Verify valid JSON output
    - `test_service_availability_detection()` - Verify service import status checked
    - `test_llm_configuration_check()` - Verify LLM config validation
    - `test_ontology_status_check()` - Verify ontology plugin status
    - `test_extraction_config_validity()` - Verify extraction method is valid
    - `test_confidence_threshold_bounds()` - Verify threshold 0.0-1.0
    - `test_test_extraction_execution()` - Verify test extraction runs
    - `test_ingestion_hook_detection()` - Verify hook invocation tracking
    - `test_not_invoked_diagnosis()` - Verify exit code 1 when not invoked
    - `test_import_error_handling()` - Verify exit code 2 on import failure
  - **Success**: All 12 tests FAIL with "FileNotFoundError: scripts/verify_entity_extraction.py"
  - **Dependencies**: None
  - **Parallel**: Can run with T002

## Phase 3.3: Core Implementation (ONLY after tests are failing)

**Sequential execution required - scripts share common imports and patterns**

- [ ] **T004** Implement graph inspector diagnostic script in `/Users/intersystems-community/ws/rag-templates/scripts/inspect_knowledge_graph.py`
  - **Contract**: `/Users/intersystems-community/ws/rag-templates/specs/032-investigate-graphrag-data/contracts/graph_inspector_contract.md`
  - **Implementation requirements**:
    - Import `iris_rag.core.connection.ConnectionManager`, `iris_rag.config.manager.ConfigurationManager`
    - Check table existence: `RAG.Entities`, `RAG.Relationships`, `RAG.Communities`
    - Query counts from each table
    - Sample up to 5 entities with document_id
    - Calculate document link statistics (linked vs orphaned)
    - Calculate data quality score (entities with embeddings / total entities)
    - Generate diagnosis with severity, message, suggestions, next_steps
    - Output JSON to stdout
    - Exit with appropriate code: 0 (success), 1 (empty), 2 (missing tables), 3 (connection error)
  - **Success**: Contract tests T002 pass (all 10 assertions)
  - **Dependencies**: T002 (tests must fail first)
  - **No [P]**: Shares imports with other diagnostic scripts

- [ ] **T005** Implement entity extraction verifier script in `/Users/intersystems-community/ws/rag-templates/scripts/verify_entity_extraction.py`
  - **Contract**: `/Users/intersystems-community/ws/rag-templates/specs/032-investigate-graphrag-data/contracts/entity_extraction_verification_contract.md`
  - **Implementation requirements**:
    - Try importing `iris_rag.services.entity_extraction.OntologyAwareEntityExtractor`
    - Check LLM configuration (provider, model, API key set)
    - Check ontology status (enabled, domain, concept count)
    - Load extraction config (method, threshold, enabled types, max entities)
    - Detect if extraction called during ingestion (check hook invocation count)
    - Run test extraction on sample text: "COVID-19 is caused by SARS-CoV-2 virus."
    - Generate diagnosis with root cause identification
    - Output JSON to stdout
    - Exit with code: 0 (functional), 1 (not invoked), 2 (service error), 3 (config error)
  - **Success**: Contract tests T003 pass (all 12 assertions)
  - **Dependencies**: T003 (tests must fail first), T004 (shared patterns)
  - **No [P]**: Sequential with T004 to establish diagnostic patterns

- [ ] **T006** Implement pipeline data comparison script in `/Users/intersystems-community/ws/rag-templates/scripts/compare_pipeline_data.py`
  - **Reference**: `/Users/intersystems-community/ws/rag-templates/specs/032-investigate-graphrag-data/data-model.md` (PipelineDataComparison entity)
  - **Implementation requirements**:
    - Query vector table counts for each pipeline type (basic, basic_rerank, crag, graphrag, pylate_colbert)
    - Query metadata table counts
    - Query knowledge graph counts (GraphRAG only)
    - Calculate data completeness score per pipeline
    - Compare retrieval success rates (from RAGAS reports if available)
    - Generate diagnosis comparing GraphRAG vs working pipelines
    - Output JSON to stdout with `pipelines` dict
    - Exit with code: 0 (all OK), 1 (some missing data), 2 (connection error)
  - **Success**: Script executes, outputs valid JSON, identifies GraphRAG data gap
  - **Dependencies**: T004, T005 (shared diagnostic patterns)
  - **No [P]**: Sequential to maintain consistency with diagnostic format

## Phase 3.4: Investigation Execution

**Sequential execution required - each step depends on previous findings**

- [ ] **T007** Run graph inspector on current system and capture results
  - **Command**: `python scripts/inspect_knowledge_graph.py > investigation/graph_state.json`
  - **Expected result**: Exit code 1 (empty graph) or 2 (tables missing)
  - **Analysis**: Document table existence and entity counts
  - **Output**: `/Users/intersystems-community/ws/rag-templates/specs/032-investigate-graphrag-data/investigation/graph_state.json`
  - **Success**: Graph state documented, empty graph or missing tables confirmed
  - **Dependencies**: T004 (script must exist)

- [ ] **T008** Run entity extraction verifier and capture results
  - **Command**: `python scripts/verify_entity_extraction.py > investigation/extraction_status.json`
  - **Expected result**: Exit code 1 (service exists but not invoked)
  - **Analysis**: Document service availability, LLM config, ontology status, invocation status
  - **Output**: `/Users/intersystems-community/ws/rag-templates/specs/032-investigate-graphrad-data/investigation/extraction_status.json`
  - **Success**: Extraction service status documented, not-invoked status confirmed
  - **Dependencies**: T005 (script must exist), T007 (graph state provides context)

- [ ] **T009** Run pipeline comparison and analyze root cause
  - **Command**: `python scripts/compare_pipeline_data.py > investigation/pipeline_comparison.json`
  - **Expected result**: GraphRAG shows incomplete data vs basic/crag
  - **Analysis**: Confirm other pipelines have data, GraphRAG missing knowledge graph data
  - **Output**: `/Users/intersystems-community/ws/rag-templates/specs/032-investigate-graphrag-data/investigation/pipeline_comparison.json`
  - **Success**: Root cause identified: Entity extraction not invoked during load_data
  - **Dependencies**: T006 (script must exist), T007-T008 (provide context)

## Phase 3.5: Validation & Documentation

**Parallel opportunities for final validation**

- [ ] **T010** [P] Verify all contract tests pass with implemented scripts
  - **Command**: `pytest tests/contract/test_graph_inspector_contract.py tests/contract/test_entity_extraction_contract.py -v`
  - **Expected**: All 22 contract tests pass (10 for graph inspector, 12 for extraction verifier)
  - **Validation**:
    - Graph inspector returns valid JSON with all required fields
    - Entity extraction verifier detects service status correctly
    - Exit codes match contract specifications
    - Suggestions provided for error states
  - **Success**: All contract assertions pass, diagnostic scripts meet specifications
  - **Dependencies**: T004, T005 (scripts implemented)
  - **Parallel**: Can run with T011

- [ ] **T011** [P] Document investigation findings and fix recommendations
  - **File**: `/Users/intersystems-community/ws/rag-templates/specs/032-investigate-graphrag-data/investigation/FINDINGS.md`
  - **Content**:
    - Summary of diagnostic results (graph state, extraction status, pipeline comparison)
    - Root cause: Entity extraction not invoked during `GraphRAGPipeline.load_documents()`
    - Evidence: Service functional (test extraction works), but invocation_count == 0
    - Fix recommendations: 3 options from quickstart.md (modify load_documents, separate target, unified loader)
    - Next steps: Implement fix, validate with RAGAS, update documentation
  - **Success**: Investigation complete, root cause documented, fix path specified
  - **Dependencies**: T007-T009 (investigation results)
  - **Parallel**: Can run with T010

---

## Dependencies Graph

```
T001 (Setup)
  ↓
T002 [P] ←──┐
T003 [P] ←──┘ (Contract tests - can run in parallel)
  ↓
T004 (Graph inspector) ← Must fail tests first
  ↓
T005 (Extraction verifier) ← Sequential for pattern consistency
  ↓
T006 (Pipeline comparison) ← Sequential for pattern consistency
  ↓
T007 (Run graph inspector) ← Investigation step 1
  ↓
T008 (Run extraction verifier) ← Investigation step 2
  ↓
T009 (Run pipeline comparison) ← Investigation step 3, identify root cause
  ↓
T010 [P] ←──┐
T011 [P] ←──┘ (Validation & docs - can run in parallel)
```

## Parallel Execution Examples

### Example 1: Contract Tests (Phase 3.2)
```bash
# Launch T002 and T003 together:
# Terminal 1:
pytest tests/contract/test_graph_inspector_contract.py -v

# Terminal 2:
pytest tests/contract/test_entity_extraction_contract.py -v

# Expected: Both fail with FileNotFoundError (TDD - scripts don't exist yet)
```

### Example 2: Final Validation (Phase 3.5)
```bash
# Launch T010 and T011 together:
# Terminal 1:
pytest tests/contract/test_graph_inspector_contract.py tests/contract/test_entity_extraction_contract.py -v

# Terminal 2:
# Document findings in investigation/FINDINGS.md based on investigation/*.json results

# Expected: T010 all tests pass, T011 root cause documented
```

## Task Execution Notes

### TDD Workflow
1. **MUST** run T002-T003 first (contract tests)
2. **VERIFY** tests fail (FileNotFoundError expected)
3. **THEN** implement T004-T006 (diagnostic scripts)
4. **VERIFY** tests pass after implementation

### Investigation Workflow
1. Run diagnostic scripts (T007-T009) sequentially
2. Capture JSON output to investigation/ directory
3. Analyze results to confirm hypothesis
4. Document root cause in FINDINGS.md

### Success Criteria
- ✅ All 22 contract tests pass
- ✅ Graph inspector detects empty knowledge graph (exit code 1)
- ✅ Extraction verifier detects service not invoked (exit code 1)
- ✅ Pipeline comparison shows GraphRAG data gap
- ✅ Root cause documented with fix recommendations

## Validation Checklist

*GATE: All items must be checked before investigation complete*

- [ ] All contracts have corresponding tests (T002-T003 ✅)
- [ ] All diagnostic scripts implemented (T004-T006 ✅)
- [ ] All tests come before implementation (T002-T003 before T004-T006 ✅)
- [ ] Parallel tasks truly independent (T002-T003, T010-T011 ✅)
- [ ] Each task specifies exact file path (✅)
- [ ] Investigation results documented (T011 ✅)
- [ ] Root cause identified and fix path specified (T011 ✅)

---

**Tasks Ready for Execution** - 11 tasks, 4 parallel opportunities, TDD enforced
