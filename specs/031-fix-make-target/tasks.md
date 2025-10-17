# Tasks: Fix RAGAS Make Target Pipeline List

**Input**: Design documents from `/specs/031-fix-make-target/`
**Prerequisites**: plan.md ✓, research.md ✓, data-model.md ✓, contracts/ ✓, quickstart.md ✓

## Execution Flow (main)
```
1. Load plan.md from feature directory ✓
   → Tech stack: Python 3.11+, Make, pytest
   → Structure: Single project (scripts/utils/, tests/infrastructure/)
2. Load optional design documents ✓
   → contracts/: helper_script_contract.md → 5 contract tests
   → data-model.md: Minimal (PipelineType read-only metadata)
   → research.md: Factory introspection pattern, Makefile integration
3. Generate tasks by category ✓
   → Tests: 5 contract tests for helper script + Makefile
   → Core: Helper script implementation + Makefile modification
   → Integration: End-to-end RAGAS verification
   → Polish: Documentation updates
4. Apply task rules ✓
   → Test files are different → [P] for parallel
   → Helper script is single file → sequential
   → Tests before implementation (TDD)
5. Number tasks sequentially ✓
6. Generate dependency graph ✓
7. Create parallel execution examples ✓
8. Validate task completeness ✓
   → All 5 contracts have tests
   → All implementation tasks depend on tests
9. Return: SUCCESS (tasks ready for execution)
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- All file paths are absolute from repository root

---

## Phase 3.1: Setup
No setup tasks needed - existing infrastructure (scripts/, tests/, Makefile already present)

---

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3
**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

- [ ] **T001** [P] Contract test: Helper script exists
  - **File**: `tests/infrastructure/test_makefile_targets.py`
  - **Task**: Add `test_get_pipeline_types_script_exists()` function
  - **Contract**: Verify `scripts/utils/get_pipeline_types.py` file exists
  - **Expected**: FAIL (script doesn't exist yet)
  - **Test Code**:
    ```python
    def test_get_pipeline_types_script_exists():
        """Verify helper script file exists at expected location."""
        script_path = Path("scripts/utils/get_pipeline_types.py")
        assert script_path.exists(), "Helper script not found"
        assert script_path.is_file(), "Helper script path is not a file"
    ```

- [ ] **T002** [P] Contract test: Output format validation
  - **File**: `tests/infrastructure/test_makefile_targets.py`
  - **Task**: Add `test_get_pipeline_types_output_format()` function
  - **Contract**: Verify comma-separated output with no spaces
  - **Expected**: FAIL (script doesn't exist yet)
  - **Test Code**:
    ```python
    def test_get_pipeline_types_output_format():
        """Verify helper script outputs comma-separated list."""
        result = subprocess.run(
            ["python", "scripts/utils/get_pipeline_types.py"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0, f"Script failed: {result.stderr}"

        output = result.stdout.strip()
        assert "," in output, "Output must contain commas"
        assert " " not in output, "Output must not contain spaces"

        pipelines = output.split(",")
        assert len(pipelines) > 0, "Must have at least one pipeline"
        for name in pipelines:
            assert name.isidentifier() or "_" in name, f"Invalid pipeline name: {name}"
    ```

- [ ] **T003** [P] Contract test: Factory matching
  - **File**: `tests/infrastructure/test_makefile_targets.py`
  - **Task**: Add `test_get_pipeline_types_matches_factory()` function
  - **Contract**: Verify helper output matches factory available_types
  - **Expected**: FAIL (script doesn't exist yet)
  - **Test Code**:
    ```python
    def test_get_pipeline_types_matches_factory():
        """Verify helper script output matches iris_rag factory."""
        # Get helper script output
        result = subprocess.run(
            ["python", "scripts/utils/get_pipeline_types.py"],
            capture_output=True,
            text=True,
        )
        assert result.returncode == 0
        script_output = set(result.stdout.strip().split(","))

        # Expected factory types (as of 2025-10-06)
        expected_types = {"basic", "basic_rerank", "crag", "graphrag", "pylate_colbert"}

        assert script_output == expected_types, \
            f"Helper output {script_output} doesn't match factory {expected_types}"
    ```

- [ ] **T004** [P] Contract test: Makefile integration
  - **File**: `tests/infrastructure/test_makefile_targets.py`
  - **Task**: Add `test_ragas_target_uses_dynamic_pipelines()` function
  - **Contract**: Verify Makefile test-ragas-sample target calls helper
  - **Expected**: FAIL (Makefile not modified yet)
  - **Test Code**:
    ```python
    def test_ragas_target_uses_dynamic_pipelines():
        """Verify Makefile test-ragas-sample target calls helper script."""
        makefile_path = Path("Makefile")
        makefile_content = makefile_path.read_text()

        # Find test-ragas-sample target
        lines = makefile_content.split('\n')
        found_target = False
        found_helper_call = False

        for line in lines:
            if 'test-ragas-sample:' in line:
                found_target = True
            if found_target and 'get_pipeline_types.py' in line:
                found_helper_call = True
                break

        assert found_target, "test-ragas-sample target not found in Makefile"
        assert found_helper_call, "Makefile doesn't call get_pipeline_types.py helper"
    ```

- [ ] **T005** [P] Contract test: Environment variable override
  - **File**: `tests/infrastructure/test_makefile_targets.py`
  - **Task**: Add `test_ragas_target_respects_env_override()` function
  - **Contract**: Verify RAGAS_PIPELINES env var preserves override capability
  - **Expected**: FAIL (Makefile not modified yet)
  - **Test Code**:
    ```python
    def test_ragas_target_respects_env_override():
        """Verify RAGAS_PIPELINES env var overrides helper script."""
        makefile_path = Path("Makefile")
        makefile_content = makefile_path.read_text()

        # Check for pattern: ${RAGAS_PIPELINES:-$(shell ...)}
        # This ensures env var takes precedence
        assert "RAGAS_PIPELINES:-" in makefile_content or \
               "RAGAS_PIPELINES:=" in makefile_content, \
               "Makefile doesn't preserve env var override capability"
    ```

**Verification After Test Phase**:
```bash
# Run all new tests - they MUST FAIL
pytest tests/infrastructure/test_makefile_targets.py::test_get_pipeline_types_script_exists -v
pytest tests/infrastructure/test_makefile_targets.py::test_get_pipeline_types_output_format -v
pytest tests/infrastructure/test_makefile_targets.py::test_get_pipeline_types_matches_factory -v
pytest tests/infrastructure/test_makefile_targets.py::test_ragas_target_uses_dynamic_pipelines -v
pytest tests/infrastructure/test_makefile_targets.py::test_ragas_target_respects_env_override -v

# Expected result: 5 FAILED (this confirms tests are working)
```

---

## Phase 3.3: Core Implementation (ONLY after tests are failing)

- [ ] **T006** Implement helper script: scripts/utils/get_pipeline_types.py
  - **File**: `scripts/utils/get_pipeline_types.py` (NEW)
  - **Task**: Create helper script that extracts pipeline types from iris_rag factory
  - **Requirements**:
    1. Import iris_rag factory module
    2. Extract available_types list from `_create_pipeline_legacy` function
    3. Output comma-separated string to stdout
    4. Exit with code 0 on success, 1 on error
    5. Write errors to stderr
  - **Implementation Guide**:
    ```python
    #!/usr/bin/env python
    """Extract available pipeline types from iris_rag factory."""
    import sys
    import inspect
    import re

    try:
        from iris_rag import _create_pipeline_legacy
    except ImportError:
        print("ERROR: Cannot import iris_rag. Is the package installed?", file=sys.stderr)
        print("       Ensure you have run: uv sync or pip install -e .", file=sys.stderr)
        sys.exit(1)

    try:
        # Get source code of factory function
        source = inspect.getsource(_create_pipeline_legacy)

        # Extract available_types list
        # Pattern: available_types = ["type1", "type2", ...]
        match = re.search(r'available_types\s*=\s*\[([\s\S]*?)\]', source)
        if not match:
            raise ValueError("Cannot find available_types list in factory source")

        # Parse list items
        list_content = match.group(1)
        types = re.findall(r'"([^"]+)"', list_content)

        if not types:
            print("ERROR: No pipeline types available from factory", file=sys.stderr)
            print("       This indicates a bug in iris_rag factory - please report", file=sys.stderr)
            sys.exit(1)

        # Output comma-separated list
        print(','.join(types))

    except Exception as e:
        print(f"ERROR: Cannot extract pipeline types: {e}", file=sys.stderr)
        print("       Factory source may have changed - please update helper script", file=sys.stderr)
        sys.exit(1)
    ```
  - **Dependencies**: Tests T001-T003 must exist and fail

- [ ] **T007** Modify Makefile: Dynamic pipeline discovery in test-ragas-sample target
  - **File**: `Makefile` (MODIFY)
  - **Task**: Update test-ragas-sample target to use helper script
  - **Line**: ~684 (approximate - search for "test-ragas-sample:")
  - **Change**:
    ```makefile
    # OLD (hardcoded):
    export RAGAS_PIPELINES=${RAGAS_PIPELINES:-"basic,basic_rerank,crag,graphrag,pylate_colbert"};

    # NEW (dynamic):
    export RAGAS_PIPELINES=$${RAGAS_PIPELINES:-$$(python scripts/utils/get_pipeline_types.py)};
    ```
  - **Verification**:
    - If RAGAS_PIPELINES env var is set → use it
    - If not set → run helper script and use output
    - If helper fails → make fails with clear error
  - **Dependencies**: T006 must be complete

- [ ] **T008** Ensure scripts/utils/ directory exists
  - **File**: `scripts/utils/` (DIRECTORY)
  - **Task**: Create directory if it doesn't exist
  - **Command**: `mkdir -p scripts/utils/`
  - **Dependencies**: None (can run before T006)

---

## Phase 3.4: Integration & Verification

- [ ] **T009** Run contract tests - verify all pass
  - **Command**: `pytest tests/infrastructure/test_makefile_targets.py -v -k "pipeline_types"`
  - **Expected**: 5/5 tests PASS
  - **If failing**: Debug and fix implementation (T006, T007)
  - **Dependencies**: T006, T007 must be complete

- [ ] **T010** Integration test: Default behavior (all pipelines)
  - **Command**: `make test-ragas-sample`
  - **Expected**:
    - Helper script is called
    - All 5 factory pipelines are tested
    - RAGAS report contains results for: basic, basic_rerank, crag, graphrag, pylate_colbert
  - **Verification**:
    ```bash
    # Check latest RAGAS report
    cat outputs/reports/ragas_evaluations/simple_ragas_report_*.json | jq 'keys'
    # Should show: ["basic", "basic_rerank", "crag", "graphrag", "pylate_colbert"]
    ```
  - **Dependencies**: T009 must pass

- [ ] **T011** Integration test: Environment variable override
  - **Command**: `RAGAS_PIPELINES=basic,crag make test-ragas-sample`
  - **Expected**:
    - Helper script is NOT called
    - Only 2 pipelines tested (basic, crag)
    - RAGAS report contains only those 2
  - **Verification**:
    ```bash
    cat outputs/reports/ragas_evaluations/simple_ragas_report_*.json | jq 'keys'
    # Should show: ["basic", "crag"]
    ```
  - **Dependencies**: T010 must pass

- [ ] **T012** Error handling test: Verify helper fails gracefully
  - **Test**: Temporarily break PYTHONPATH and verify make fails with clear error
  - **Command**: `PYTHONPATH=/invalid make test-ragas-sample`
  - **Expected**:
    - Make fails immediately
    - Error message: "ERROR: Cannot import iris_rag..."
    - No partial RAGAS evaluation
  - **Dependencies**: T009 must pass

---

## Phase 3.5: Polish

- [ ] **T013** [P] Update documentation: Add feature to README.md
  - **File**: `README.md` (MODIFY)
  - **Task**: Document dynamic pipeline discovery feature
  - **Section**: Find "RAGAS Evaluation" or "Testing" section
  - **Add**:
    ```markdown
    ### Dynamic Pipeline Discovery

    RAGAS evaluation targets automatically test all available pipelines:

    ```bash
    # Test all factory pipelines (auto-discovered)
    make test-ragas-sample

    # Override to test specific pipelines only
    RAGAS_PIPELINES=basic,crag make test-ragas-sample
    ```

    The pipeline list is dynamically queried from the iris_rag factory, so new
    pipelines are automatically included without Makefile updates.
    ```
  - **Dependencies**: T010, T011 must pass

- [ ] **T014** [P] Update CLAUDE.md: Document helper script
  - **File**: `CLAUDE.md` (VERIFY/UPDATE)
  - **Task**: Verify agent context was updated correctly by update-agent-context.sh
  - **Check**: Should already contain Python 3.11+ and build system context
  - **Add if missing**: Reference to `scripts/utils/get_pipeline_types.py` as a build utility
  - **Dependencies**: None (can run anytime)

- [ ] **T015** Clean up: Remove any debug code or temporary files
  - **Task**: Review all modified files for debug prints, commented code
  - **Files to check**:
    - `scripts/utils/get_pipeline_types.py`
    - `Makefile`
    - `tests/infrastructure/test_makefile_targets.py`
  - **Dependencies**: All previous tasks complete

- [ ] **T016** Final verification: Run quickstart guide
  - **File**: Follow steps in `specs/031-fix-make-target/quickstart.md`
  - **Task**: Execute all 7 quickstart steps and verify success criteria
  - **Expected**: All steps pass, all success criteria met
  - **Dependencies**: All implementation and integration tasks complete

---

## Dependencies

**Critical Path**:
1. **Tests first** (T001-T005) → Must fail before implementation
2. **Implementation** (T006-T008) → Make tests pass
3. **Verification** (T009-T012) → Confirm everything works
4. **Polish** (T013-T016) → Documentation and cleanup

**Parallel Opportunities**:
- T001-T005 can all be written in parallel (different test functions)
- T008 can run before T006 (just create directory)
- T013-T014 can run in parallel (different files)

**Blocking Relationships**:
```
T001-T005 (tests) ──→ T006 (helper script)
                  ──→ T007 (Makefile)
T008 (mkdir) ─────────→ T006 (helper script)

T006, T007 ──→ T009 (verify tests pass)
T009 ──→ T010 (default behavior)
T010 ──→ T011 (env override)
T010 ──→ T012 (error handling)

T010, T011 ──→ T013 (README)
All ──→ T015 (cleanup)
T015 ──→ T016 (final verification)
```

---

## Parallel Execution Examples

### Phase 1: Write All Tests in Parallel
```bash
# Launch T001-T005 together (5 test functions, same file but different tests):
# Use Task agent with multiple prompts:

Task 1: "Add test_get_pipeline_types_script_exists() to tests/infrastructure/test_makefile_targets.py
  - Verify scripts/utils/get_pipeline_types.py file exists
  - Assert is_file() and exists()
  - Test must FAIL (script doesn't exist yet)"

Task 2: "Add test_get_pipeline_types_output_format() to tests/infrastructure/test_makefile_targets.py
  - Run helper script with subprocess
  - Verify comma-separated output, no spaces
  - Test must FAIL (script doesn't exist yet)"

Task 3: "Add test_get_pipeline_types_matches_factory() to tests/infrastructure/test_makefile_targets.py
  - Compare helper output to expected factory types
  - Expected: {basic, basic_rerank, crag, graphrag, pylate_colbert}
  - Test must FAIL (script doesn't exist yet)"

Task 4: "Add test_ragas_target_uses_dynamic_pipelines() to tests/infrastructure/test_makefile_targets.py
  - Parse Makefile content
  - Verify get_pipeline_types.py is called
  - Test must FAIL (Makefile not modified yet)"

Task 5: "Add test_ragas_target_respects_env_override() to tests/infrastructure/test_makefile_targets.py
  - Check Makefile has RAGAS_PIPELINES:- pattern
  - Verify env var override capability
  - Test must FAIL (Makefile not modified yet)"
```

### Phase 2: Documentation in Parallel
```bash
# Launch T013-T014 together (different files):

Task 1: "Update README.md with Dynamic Pipeline Discovery section
  - Add to RAGAS Evaluation section
  - Document make test-ragas-sample behavior
  - Document RAGAS_PIPELINES env var override"

Task 2: "Verify CLAUDE.md was updated by update-agent-context.sh
  - Check for Python 3.11+ context
  - Add reference to scripts/utils/get_pipeline_types.py if missing"
```

---

## Validation Checklist

*GATE: Verify before marking tasks complete*

- [x] All 5 contracts have corresponding tests (T001-T005)
- [x] Helper script implementation task exists (T006)
- [x] Makefile modification task exists (T007)
- [x] All tests come before implementation (T001-T005 before T006-T007)
- [x] Parallel tasks are truly independent (test functions are independent)
- [x] Each task specifies exact file path
- [x] No task modifies same file as another [P] task (tests are functions, not file rewrites)
- [x] Integration verification tasks exist (T009-T012)
- [x] Documentation tasks exist (T013-T014)
- [x] Final verification via quickstart (T016)

---

## Notes

### Important Reminders
- **TDD**: Tests T001-T005 MUST fail before implementing T006-T007
- **Verification**: Run `pytest -v` after T006 to confirm tests now pass
- **Commit**: Commit after each task (especially after T005, T007, T009, T016)
- **Quickstart**: T016 is your final verification - follow every step

### Common Pitfalls to Avoid
- ❌ Implementing helper script before writing tests
- ❌ Modifying Makefile without testing helper script first
- ❌ Not verifying tests actually fail in T001-T005
- ❌ Skipping integration tests (T010-T012)
- ❌ Not following quickstart guide for final validation

### Success Criteria
After completing all tasks:
- ✅ 5/5 contract tests pass
- ✅ `make test-ragas-sample` tests all 5 factory pipelines
- ✅ `RAGAS_PIPELINES=basic,crag make test-ragas-sample` tests only 2
- ✅ Helper script fails clearly on import errors
- ✅ Quickstart guide completes successfully
- ✅ Documentation is updated

---

**Task Generation Status**: ✓ Complete
**Total Tasks**: 16 (5 test, 3 implementation, 4 integration, 4 polish)
**Estimated Time**: 2-3 hours (tests: 1h, implementation: 30m, verification: 1h, docs: 30m)
**Ready for Execution**: Yes - start with T001
