# Tasks: Fix RAGAS GraphRAG Evaluation Workflow

**Input**: Design documents from `/Users/intersystems-community/ws/rag-templates/specs/040-fix-ragas-evaluation/`
**Prerequisites**: plan.md (complete), data-model.md (complete), quickstart.md (complete)

## Execution Flow (main)
```
1. Load plan.md from feature directory ✓
   → Extracted: Python 3.12, single script modification (scripts/simple_working_ragas.py)
   → Tech stack: iris_rag framework, RAGAS, sentence-transformers
2. Load optional design documents ✓
   → data-model.md: Evaluation workflow states, entity check results
   → quickstart.md: Validation procedure (baseline, fix, verification steps)
   → No contracts/ (bug fix, not new API)
3. Generate tasks by category:
   → Baseline: Capture current failure state
   → Core: Entity check function, auto-load function, skip logic
   → Integration: Modify evaluation loop to call entity check
   → Validation: Run quickstart procedures, verify fix
4. Apply task rules ✓
   → Helper functions [P] (different logical blocks in same file)
   → Integration depends on helpers (sequential)
   → Validation last
5. Number tasks sequentially (T001-T008)
6. Generate dependency graph ✓
7. Validate task completeness ✓
   → All workflow states covered
   → Baseline→Implementation→Validation flow maintained
8. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] Description`
- **NO [P] markers**: All tasks modify the same file (scripts/simple_working_ragas.py)
- Include exact line numbers or function names for modifications
- Each task is self-contained and can be validated independently

## Path Conventions
- **Target file**: `/Users/intersystems-community/ws/rag-templates/scripts/simple_working_ragas.py` (single file to modify)
- **Reference files** (read-only):
  - `/Users/intersystems-community/ws/rag-templates/iris_rag/pipelines/graphrag.py` (GraphRAG.load_documents method)
  - `/Users/intersystems-community/ws/rag-templates/data/sample_10_docs/` (test documents)
- **Validation**: `/Users/intersystems-community/ws/rag-templates/specs/040-fix-ragas-evaluation/quickstart.md`

---

## Phase 3.1: Baseline Capture

- [ ] **T001** Capture current GraphRAG failure state (baseline verification)
  - **File**: No file modifications - diagnostic only
  - **Action**: Run `make test-ragas-sample` and verify GraphRAG fails with "Knowledge graph is empty" error
  - **Expected result**: Error message "Knowledge graph is empty. No entities found in RAG.Entities table."
  - **Verification**:
    ```bash
    # Check entity tables are empty
    python -c "import iris; conn = iris.connect('localhost', 1972, 'USER', '_SYSTEM', 'SYS'); cursor = conn.cursor(); cursor.execute('SELECT COUNT(*) FROM RAG.Entities'); print(f'Entities: {cursor.fetchone()[0]}'); cursor.close(); conn.close()"
    ```
  - **Output**: Save output to `/tmp/baseline_graphrag_failure.log` for comparison

---

## Phase 3.2: Core Implementation (Entity Check & Auto-Load)

- [ ] **T002** Add entity data check function to scripts/simple_working_ragas.py
  - **File**: `/Users/intersystems-community/ws/rag-templates/scripts/simple_working_ragas.py`
  - **Action**: Add new function `check_graphrag_prerequisites()` after imports, before main()
  - **Function signature**:
    ```python
    def check_graphrag_prerequisites() -> Dict[str, Any]:
        """Check if GraphRAG prerequisites (entity data) are met.

        Returns:
            {
                "has_entities": bool,
                "entities_count": int,
                "relationships_count": int,
                "sufficient_data": bool
            }
        """
    ```
  - **Implementation requirements**:
    - Connect to IRIS database using existing connection pattern from script
    - Query `SELECT COUNT(*) FROM RAG.Entities`
    - Query `SELECT COUNT(*) FROM RAG.EntityRelationships`
    - Return dictionary with counts and `sufficient_data = (entities_count > 0)`
  - **Verification**: Function returns correct entity counts from database

- [ ] **T003** Add auto-load function to scripts/simple_working_ragas.py
  - **File**: `/Users/intersystems-community/ws/rag-templates/scripts/simple_working_ragas.py`
  - **Action**: Add new function `load_documents_with_entities()` after check_graphrag_prerequisites()
  - **Function signature**:
    ```python
    def load_documents_with_entities(
        pipeline: Any,  # GraphRAGPipeline instance
        documents_path: str,
        logger: logging.Logger
    ) -> Dict[str, Any]:
        """Load documents using GraphRAG pipeline to extract entities.

        Args:
            pipeline: GraphRAG pipeline instance
            documents_path: Path to documents directory
            logger: Logger instance for output

        Returns:
            {
                "documents_loaded": int,
                "entities_extracted": int,
                "relationships_extracted": int,
                "success": bool,
                "error": Optional[str]
            }
        """
    ```
  - **Implementation requirements**:
    - Call `pipeline.load_documents(documents_path)` (GraphRAG's method that extracts entities)
    - Re-run entity check after loading to get final counts
    - Log progress: "Loading documents with entity extraction..."
    - Handle exceptions and return error message if load fails
    - Return success=True if no exceptions, even if entities_extracted=0
  - **Verification**: Function successfully calls GraphRAG.load_documents() and returns entity counts

- [ ] **T004** Add skip logic logging to scripts/simple_working_ragas.py
  - **File**: `/Users/intersystems-community/ws/rag-templates/scripts/simple_working_ragas.py`
  - **Action**: Add helper function `log_graphrag_skip()` for consistent skip messages
  - **Function signature**:
    ```python
    def log_graphrag_skip(
        logger: logging.Logger,
        reason: str,
        entities_count: int
    ) -> None:
        """Log informational message when GraphRAG evaluation is skipped.

        Args:
            logger: Logger instance
            reason: Human-readable skip reason
            entities_count: Current entity count from database
        """
    ```
  - **Implementation requirements**:
    - Log at INFO level (not WARNING or ERROR - this is expected behavior)
    - Message format: "⏭️  Skipping GraphRAG evaluation: {reason}"
    - Include entity count in message: "Entity count: {entities_count}"
    - Include actionable guidance: "To enable GraphRAG: load documents with entity extraction"
  - **Verification**: Function logs clear, informational skip message

---

## Phase 3.3: Integration (Modify Evaluation Loop)

- [ ] **T005** Integrate entity check into main evaluation loop in scripts/simple_working_ragas.py
  - **File**: `/Users/intersystems-community/ws/rag-templates/scripts/simple_working_ragas.py`
  - **Action**: Modify the pipeline evaluation loop (find where pipelines are tested sequentially)
  - **Integration point**: Before creating/evaluating GraphRAG pipeline, add conditional check
  - **Logic to implement**:
    ```python
    # Before GraphRAG pipeline evaluation:
    if pipeline_name == "graphrag":
        # Check entity data
        entity_check = check_graphrag_prerequisites()
        logger.info(f"GraphRAG entity check: {entity_check['entities_count']} entities, {entity_check['relationships_count']} relationships")

        if not entity_check["sufficient_data"]:
            # Mode 1: Auto-load (default for now)
            logger.info("No entity data found. Auto-loading documents with entity extraction...")

            # Create GraphRAG pipeline first
            pipeline = create_pipeline("graphrag", validate_requirements=True, auto_setup=True)

            # Load documents with entity extraction
            load_result = load_documents_with_entities(
                pipeline=pipeline,
                documents_path=os.getenv("EVAL_PMC_DIR", "data/sample_10_docs"),
                logger=logger
            )

            if load_result["success"]:
                logger.info(f"Entity extraction complete: {load_result['entities_extracted']} entities, {load_result['relationships_extracted']} relationships")
                # Continue with evaluation
            else:
                logger.error(f"Entity extraction failed: {load_result.get('error', 'Unknown error')}")
                # Skip GraphRAG evaluation
                log_graphrag_skip(logger, "Entity extraction failed", 0)
                continue  # Skip to next pipeline
    ```
  - **Constraints**:
    - Must not affect basic, basic_rerank, crag, or pylate_colbert pipelines
    - Must preserve existing error handling patterns in the script
    - Must use existing environment variables (EVAL_PMC_DIR)
  - **Verification**: GraphRAG evaluation now checks entities and auto-loads if needed

---

## Phase 3.4: Validation & Testing

- [ ] **T006** Run quickstart baseline verification (Step 1)
  - **File**: No file modifications - validation only
  - **Action**: Execute quickstart.md Step 1 baseline tests
  - **Commands**:
    ```bash
    # Clean entity tables
    python -c "import iris; conn = iris.connect('localhost', 1972, 'USER', '_SYSTEM', 'SYS'); cursor = conn.cursor(); cursor.execute('TRUNCATE TABLE RAG.Entities'); cursor.execute('TRUNCATE TABLE RAG.EntityRelationships'); conn.commit(); cursor.close(); conn.close()"

    # Run evaluation
    make test-ragas-sample

    # Verify GraphRAG NO LONGER fails with "knowledge graph is empty"
    # Instead, should show entity extraction happened
    ```
  - **Success criteria**:
    - GraphRAG evaluation completes (may have low scores, but shouldn't crash)
    - Entity count > 0 after evaluation
    - No "knowledge graph is empty" error

- [ ] **T007** Verify entity extraction occurred (quickstart Step 3)
  - **File**: No file modifications - validation only
  - **Action**: Check that entity extraction populated knowledge graph tables
  - **Verification**:
    ```bash
    python -c "
    import iris
    conn = iris.connect('localhost', 1972, 'USER', '_SYSTEM', 'SYS')
    cursor = conn.cursor()
    cursor.execute('SELECT COUNT(*) FROM RAG.Entities')
    entity_count = cursor.fetchone()[0]
    cursor.execute('SELECT COUNT(*) FROM RAG.EntityRelationships')
    relationship_count = cursor.fetchone()[0]
    cursor.close()
    conn.close()
    print(f'✓ Entities: {entity_count}' if entity_count > 0 else '❌ No entities')
    print(f'✓ Relationships: {relationship_count}')
    "
    ```
  - **Success criteria**:
    - Entities count > 0
    - Relationships count >= 0 (can be zero if no relationships found)
    - GraphRAG evaluation results in latest JSON report

- [ ] **T008** Run regression tests (quickstart Step 4)
  - **File**: No file modifications - validation only
  - **Action**: Verify other pipelines still work correctly
  - **Verification**:
    ```bash
    # Run full evaluation
    make test-ragas-sample

    # Check all pipeline results
    python -c "
    import json
    import glob
    report_files = sorted(glob.glob('outputs/reports/ragas_evaluations/simple_ragas_report_*.json'), reverse=True)
    if report_files:
        with open(report_files[0]) as f:
            data = json.load(f)
        for pipeline in ['basic', 'basic_rerank', 'crag', 'pylate_colbert', 'graphrag']:
            if pipeline in data:
                success_rate = data[pipeline].get('success_rate', 0)
                print(f'{pipeline}: success_rate={success_rate}')
            else:
                print(f'{pipeline}: ❌ Missing from results')
    "
    ```
  - **Success criteria**:
    - All 5 pipelines present in results (basic, basic_rerank, crag, graphrag, pylate_colbert)
    - No pipeline crashes or fails to run
    - GraphRAG no longer shows "knowledge graph is empty" error

---

## Dependencies

**Sequential Execution Required** (all tasks modify/test the same file):
```
T001 (baseline)
  ↓
T002 (entity check function)
  ↓
T003 (auto-load function)
  ↓
T004 (skip logging function)
  ↓
T005 (integrate into evaluation loop) - DEPENDS ON: T002, T003, T004
  ↓
T006 (validation: baseline check)
  ↓
T007 (validation: entity extraction)
  ↓
T008 (validation: regression tests)
```

**Logical Flow**:
- **T001**: Establish baseline (problem exists)
- **T002-T004**: Build helper functions (can be done sequentially, one after another)
- **T005**: Integration (uses all helpers)
- **T006-T008**: Validation (verify fix works)

**No Parallel Execution**: All tasks modify or test the same script file (`scripts/simple_working_ragas.py`), so they must run sequentially.

---

## Success Criteria

✅ **Baseline (T001)**:
- GraphRAG fails with "Knowledge graph is empty" error (confirms problem exists)

✅ **Implementation (T002-T005)**:
- Entity check function returns correct counts from database
- Auto-load function calls GraphRAG.load_documents() successfully
- Skip logging provides clear, actionable messages
- Integration point correctly detects GraphRAG pipeline and runs entity check

✅ **Validation (T006-T008)**:
- GraphRAG evaluation completes without "knowledge graph is empty" error
- Entity tables populated (entities_count > 0)
- All 5 pipelines produce results in evaluation report
- No regression (basic/crag/pylate_colbert still work)

---

## Notes

- **Single file modification**: All code changes in `scripts/simple_working_ragas.py`
- **No TDD for this fix**: Modifying existing script, validation via make target (not formal unit tests)
- **Backward compatible**: Fix only activates for GraphRAG pipeline, others unaffected
- **Auto-load mode**: Current implementation uses auto-load by default (skip mode can be added later)
- **Environment variables**: Uses existing EVAL_PMC_DIR for document path
- **Entity extraction performance**: Expect 10-30 seconds for 71 sample documents

---

## Task Generation Metadata

**Source documents analyzed**:
- ✓ plan.md (technical context, fix approach)
- ✓ data-model.md (evaluation workflow states, entity check results)
- ✓ quickstart.md (validation procedure, before/after tests)
- ✗ contracts/ (not applicable - bug fix, no new API)

**Validation checklist**:
- ✓ Baseline capture before implementation
- ✓ Helper functions before integration
- ✓ Integration depends on helpers
- ✓ Validation verifies fix works
- ✓ Regression tests ensure no breakage
- ✓ Exact file paths specified
- ✓ Success criteria measurable

**Estimated completion time**: 2-3 hours
- T001: 10 minutes (baseline capture)
- T002-T004: 60 minutes (helper functions)
- T005: 45 minutes (integration)
- T006-T008: 45 minutes (validation)

---

*Generated by /tasks command for Feature 040-fix-ragas-evaluation*
*Based on plan.md, data-model.md, and quickstart.md*
