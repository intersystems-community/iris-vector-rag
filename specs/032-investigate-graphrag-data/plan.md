# Implementation Plan: GraphRAG Data/Setup Investigation

**Branch**: `032-investigate-graphrag-data` | **Date**: 2025-10-06 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/Users/intersystems-community/ws/rag-templates/specs/032-investigate-graphrag-data/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → ✓ Spec loaded successfully
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Project Type: Single Python framework project (RAG system)
   → Structure: Framework diagnostic scripts + investigation
3. Fill the Constitution Check section
   → Investigation feature - diagnostic focus, not new pipeline
4. Evaluate Constitution Check section
   → Investigation features exempt from some requirements (no new pipeline)
   → Update Progress Tracking: Initial Constitution Check
5. Execute Phase 0 → research.md
   → Research GraphRAG architecture, entity extraction, knowledge graph schema
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, CLAUDE.md
   → Diagnostic contracts for graph inspection
7. Re-evaluate Constitution Check section
   → Diagnostic scripts follow framework patterns
   → Update Progress Tracking: Post-Design Constitution Check
8. Plan Phase 2 → Describe task generation approach
9. STOP - Ready for /tasks command
```

## Summary

**Primary Requirement**: Investigate why GraphRAG pipeline returns no documents (0% performance) despite successful document loading in RAGAS evaluation and elsewhere.

**Observable Problem**:
- GraphRAG returns "No relevant documents found to answer the query" for all queries
- Other pipelines (basic, basic_rerank, crag, pylate_colbert) retrieve documents successfully from same data
- GraphRAG shows empty contexts `[]`, 0% context precision/recall, 0.14 overall score
- No crashes, but zero useful results

**Investigation Scope**:
1. Verify knowledge graph tables exist and are accessible
2. Check if entity extraction runs during `make load-data`
3. Inspect knowledge graph state (entity/relationship/community counts)
4. Validate GraphRAG query logic accesses correct tables
5. Document root cause and required fixes

## Technical Context

**Language/Version**: Python 3.12 (current framework version)
**Primary Dependencies**:
- iris_rag framework (GraphRAG, HybridGraphRAGPipeline)
- InterSystems IRIS database (knowledge graph storage)
- Entity extraction service (iris_rag.services.entity_extraction)
- Knowledge graph schema manager
**Storage**: IRIS database with knowledge graph tables (RAG.Entities, RAG.Relationships, RAG.Communities)
**Testing**: pytest with `@pytest.mark.requires_database` for IRIS connectivity
**Target Platform**: macOS/Linux development environment with Docker IRIS
**Project Type**: Single framework project (diagnostic investigation)
**Performance Goals**: Identify why GraphRAG has 0% retrieval vs other pipelines with >60% retrieval
**Constraints**: Investigation only - no changes to GraphRAG logic unless absolutely necessary
**Scale/Scope**: 10-document RAGAS test set, but issue affects all GraphRAG usage

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**I. Framework-First Architecture**:
- N/A No new pipeline component (investigation/diagnostic)
- ✓ Diagnostic scripts will use CLI interface
- ✓ Results format suitable for framework consumers

**II. Pipeline Validation & Requirements**:
- N/A Investigation feature, not pipeline creation
- ✓ Diagnostic checks will be idempotent

**III. Test-Driven Development**:
- ✓ Contract tests for graph inspection utilities
- N/A Performance tests not applicable (diagnostic feature)
- ✓ IRIS connectivity required for all diagnostic tests

**IV. Performance & Enterprise Scale**:
- N/A Investigation feature
- ✓ Diagnostic queries will use proven IRIS patterns

**V. Production Readiness**:
- ✓ Structured logging for diagnostic output
- ✓ Clear error messages for missing data
- N/A Docker deployment not required (diagnostic script)

**VI. Explicit Error Handling**:
- ✓ All diagnostic checks report explicit results
- ✓ Clear messages for empty tables, missing entities
- ✓ Actionable next steps in findings

**VII. Standardized Database Interfaces**:
- ✓ Uses framework's database utilities
- ✓ No ad-hoc queries - reuses existing patterns
- ✓ Any new inspection utilities contributed back

**Summary**: Investigation feature is constitutionally compliant. Focuses on diagnostics, not new implementation. Uses established framework patterns for database inspection.

## Project Structure

### Documentation (this feature)
```
specs/032-investigate-graphrag-data/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0: GraphRAG architecture analysis
├── data-model.md        # Phase 1: Knowledge graph schema expectations
├── quickstart.md        # Phase 1: Investigation execution guide
├── contracts/           # Phase 1: Diagnostic tool contracts
│   ├── graph_inspector_contract.md
│   └── entity_extraction_verification_contract.md
└── tasks.md             # Phase 2: (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
# Investigation diagnostic scripts location
scripts/
├── inspect_knowledge_graph.py      # NEW: Inspect graph state
├── verify_entity_extraction.py     # NEW: Check extraction pipeline
└── compare_pipeline_data.py        # NEW: Compare data availability

# Existing framework components under investigation
iris_rag/
├── pipelines/
│   ├── graphrag.py                 # Original GraphRAG (NOT used)
│   └── hybrid_graphrag.py          # Current GraphRAG implementation
├── services/
│   └── entity_extraction.py        # Entity extraction service
└── storage/
    └── schema_manager.py           # Knowledge graph schema

# Test structure for diagnostic validation
tests/
├── contract/
│   ├── test_graph_inspector_contract.py
│   └── test_entity_extraction_contract.py
└── integration/
    └── test_graphrag_data_investigation.py
```

**Structure Decision**: Single framework project structure with diagnostic scripts in `/scripts` directory following existing patterns. Tests validate diagnostic accuracy before investigation. No new pipeline components - only inspection and verification utilities.

## Phase 0: Outline & Research

**Research Goals**:
1. Understand GraphRAG/HybridGraphRAGPipeline architecture
2. Identify entity extraction trigger points in document loading
3. Map knowledge graph schema (tables, relationships, expectations)
4. Analyze retrieval flow from query to knowledge graph access
5. Compare with working pipelines (basic, crag) data requirements

**Research Tasks**:

### R1: GraphRAG Architecture Analysis
**Question**: How does HybridGraphRAGPipeline retrieve documents? What data sources does it query?

**Investigation areas**:
- Source file: `iris_rag/pipelines/hybrid_graphrag.py`
- Retrieval methods and their data dependencies
- Knowledge graph vs vector store usage
- Query execution flow

### R2: Entity Extraction Pipeline
**Question**: When and how does entity extraction occur during document ingestion?

**Investigation areas**:
- Source file: `iris_rag/services/entity_extraction.py`
- Trigger points in load_data workflow
- Configuration flags that enable/disable extraction
- Dependencies on LLM/NLP services

### R3: Knowledge Graph Schema
**Question**: What tables/schema are required for GraphRAG? What's the expected state after loading 10 documents?

**Investigation areas**:
- Source file: `iris_rag/storage/schema_manager.py`
- Knowledge graph table definitions (Entities, Relationships, Communities)
- Minimum viable data for retrieval
- Schema initialization during setup

### R4: Comparison with Working Pipelines
**Question**: What data do basic/crag pipelines access vs GraphRAG?

**Investigation areas**:
- Basic pipeline: vector store only
- CRAG pipeline: vector store + metadata
- GraphRAG: vector store + knowledge graph
- Identify GraphRAG's unique data requirements

**Output**: `research.md` with architectural findings, entity extraction workflow, schema expectations, and comparison matrix.

## Phase 1: Design & Contracts

*Prerequisites: research.md complete*

### 1. Data Model (`data-model.md`)

**Entities** (from spec + research):

**KnowledgeGraphState**:
- entity_count: int (number of entities in RAG.Entities)
- relationship_count: int (number of relationships in RAG.Relationships)
- community_count: int (number of communities in RAG.Communities)
- document_entity_links: int (entities linked to source documents)
- orphaned_entities: int (entities without document links)
- extraction_enabled: bool (whether extraction ran during load)

**EntityExtractionStatus**:
- service_available: bool (entity_extraction module accessible)
- llm_configured: bool (LLM available for extraction)
- extraction_triggered: bool (extraction called during ingestion)
- documents_processed: int (documents that went through extraction)
- entities_created: int (entities successfully created)
- errors: List[str] (extraction errors if any)

**PipelineDataComparison**:
- pipeline_name: str
- vector_table_rows: int
- metadata_table_rows: int
- knowledge_graph_rows: int (GraphRAG only)
- retrieval_success_rate: float

### 2. API Contracts (`/contracts/`)

**Contract 1: Graph Inspector** (`contracts/graph_inspector_contract.md`)
```python
# COMMAND: python scripts/inspect_knowledge_graph.py
# EXIT CODE: 0 (success), 1 (empty graph), 2 (tables missing)
#
# OUTPUT FORMAT (JSON):
# {
#   "tables_exist": {"entities": bool, "relationships": bool, "communities": bool},
#   "counts": {"entities": int, "relationships": int, "communities": int},
#   "sample_entities": [{"id": str, "name": str, "type": str}, ...],
#   "document_links": {"total_entities": int, "linked": int, "orphaned": int}
# }
#
# ASSERTIONS:
# - If tables_exist all true: count values must be >= 0
# - If any table missing: exit code 2
# - If all counts == 0: exit code 1 (empty graph)
# - sample_entities limited to 5 for brevity
```

**Contract 2: Entity Extraction Verification** (`contracts/entity_extraction_verification_contract.md`)
```python
# COMMAND: python scripts/verify_entity_extraction.py
# EXIT CODE: 0 (enabled), 1 (disabled), 2 (service error)
#
# OUTPUT FORMAT (JSON):
# {
#   "service_status": {"available": bool, "import_error": str | null},
#   "llm_status": {"configured": bool, "provider": str, "model": str},
#   "ingestion_hooks": {"extraction_called": bool, "hook_location": str},
#   "test_extraction": {"success": bool, "entities_found": int, "error": str | null}
# }
#
# ASSERTIONS:
# - If service_status.available false: exit code 2
# - If llm_status.configured false: exit code 1
# - If ingestion_hooks.extraction_called false: exit code 1
# - test_extraction runs mini extraction test if possible
```

### 3. Contract Tests

**`tests/contract/test_graph_inspector_contract.py`**:
```python
def test_graph_inspector_exists():
    """Verify inspector script exists."""
    assert Path("scripts/inspect_knowledge_graph.py").exists()

def test_graph_inspector_output_format():
    """Verify inspector outputs valid JSON with required fields."""
    result = subprocess.run(["python", "scripts/inspect_knowledge_graph.py"],
                          capture_output=True, text=True)
    output = json.loads(result.stdout)
    assert "tables_exist" in output
    assert "counts" in output
    assert "sample_entities" in output
    assert "document_links" in output

def test_graph_inspector_empty_graph_detection():
    """Verify inspector detects empty knowledge graph."""
    # This test expects to FAIL initially (graph is empty)
    result = subprocess.run(["python", "scripts/inspect_knowledge_graph.py"],
                          capture_output=True)
    assert result.returncode == 1  # Empty graph
```

**`tests/contract/test_entity_extraction_contract.py`**:
```python
def test_entity_extraction_verifier_exists():
    """Verify extraction verifier script exists."""
    assert Path("scripts/verify_entity_extraction.py").exists()

def test_entity_extraction_verifier_output_format():
    """Verify verifier outputs valid JSON."""
    result = subprocess.run(["python", "scripts/verify_entity_extraction.py"],
                          capture_output=True, text=True)
    output = json.loads(result.stdout)
    assert "service_status" in output
    assert "llm_status" in output
    assert "ingestion_hooks" in output
```

### 4. Quickstart Guide (`quickstart.md`)

Investigation execution steps for developers to run diagnostic scripts, interpret results, and identify root cause.

### 5. Update CLAUDE.md

Run the agent context update script to add this investigation context to CLAUDE.md.

**Output**: data-model.md, /contracts/*, failing contract tests, quickstart.md, updated CLAUDE.md

## Phase 2: Task Planning Approach

*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:

1. **Contract Implementation Tasks** (TDD order):
   - T001: Write contract test for graph inspector (FAIL expected)
   - T002: Implement graph inspector script to pass contract
   - T003: Write contract test for extraction verifier (FAIL expected)
   - T004: Implement extraction verifier script to pass contract

2. **Investigation Execution Tasks**:
   - T005: Run graph inspector on current system
   - T006: Run extraction verifier on current system
   - T007: Analyze results and identify root cause
   - T008: Document findings in investigation report

3. **Validation Tasks**:
   - T009: Verify diagnostic scripts against contract tests
   - T010: Execute quickstart guide end-to-end
   - T011: Document fix recommendations

**Ordering Strategy**:
- TDD: Tests before implementation (T001→T002, T003→T004)
- Diagnostic before analysis (T005-T006 before T007)
- Documentation last (T008, T011)
- T001, T003 can run in parallel [P]
- T002, T004 can run in parallel [P]
- T005, T006 can run in parallel [P]

**Estimated Output**: 11 tasks in tasks.md with [P] markers for parallel execution

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation

*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)
**Phase 4**: Implementation (diagnostic scripts + investigation execution)
**Phase 5**: Validation (contract tests pass, root cause documented, fix path identified)

## Complexity Tracking

*No constitutional violations - investigation feature follows framework patterns*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| N/A       | N/A        | N/A                                 |

## Progress Tracking

**Phase Status**:
- [x] Phase 0: Research complete (/plan command) ✅
- [x] Phase 1: Design complete (/plan command) ✅
- [x] Phase 2: Task planning complete (/plan command - describe approach only) ✅
- [x] Phase 3: Tasks generated (/tasks command) ✅
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Deliverables Created**:
- [x] research.md (R1-R4 complete with hypothesis)
- [x] data-model.md (4 entities, 3 contracts defined)
- [x] contracts/graph_inspector_contract.md (10 assertions, 3 scenarios)
- [x] contracts/entity_extraction_verification_contract.md (12 assertions, 4 scenarios)
- [x] quickstart.md (investigation workflow guide)
- [x] CLAUDE.md updated (agent context)
- [x] tasks.md (11 tasks, TDD workflow, 4 parallel opportunities)

**Gate Status**:
- [x] Initial Constitution Check: PASS (investigation feature compliant)
- [x] Post-Design Constitution Check: PASS (diagnostic scripts follow patterns)
- [x] All NEEDS CLARIFICATION resolved (hypothesis formed from research)
- [x] Complexity deviations documented (none - no violations)

**Research Findings**:
- ✅ GraphRAG architecture analyzed (HybridGraphRAGPipeline extends GraphRAGPipeline)
- ✅ Entity extraction service located and analyzed (OntologyAwareEntityExtractor)
- ✅ Knowledge graph schema documented (RAG.Entities, RAG.Relationships, RAG.Communities)
- ✅ Pipeline comparison complete (Basic/CRAG work, GraphRAG needs KG data)
- ✅ **Hypothesis formed**: Entity extraction not invoked during load_data workflow

---
*Based on Constitution v1.6.0 - See `/Users/intersystems-community/ws/rag-templates/.specify/memory/constitution.md`*
