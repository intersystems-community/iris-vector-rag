# Tasks: Repository Cleanup and Organization

**Input**: Design documents from `/specs/037-clean-up-this/`
**Prerequisites**: plan.md ✓, research.md ✓, data-model.md ✓, contracts/ ✓, quickstart.md ✓

## Execution Flow (main)

```
1. Load plan.md from feature directory ✅
   → Tech stack: Python 3.11, git, pytest
   → Structure: Single project (maintenance task)
2. Load optional design documents ✅
   → data-model.md: 13 entities (FileCategory, RepositoryFile, etc.)
   → contracts/cleanup-operations.md: 10 operations
   → research.md: 4 decision areas
   → quickstart.md: 12 execution steps
3. Generate tasks by category ✅
   → Setup: Cleanup script structure
   → Tests: 10 contract tests (one per operation)
   → Core: 10 operation implementations
   → Integration: Full cleanup workflow
   → Polish: Documentation, final validation
4. Apply task rules ✅
   → Contract tests = [P] (different files)
   → Operations = sequential (depend on tests passing)
5. Number tasks sequentially (T001-T024) ✅
6. Generate dependency graph ✅
7. Create parallel execution examples ✅
8. Validate task completeness ✅
   → All 10 contracts have tests ✓
   → All 13 entities in data model ✓
   → All 10 operations implemented ✓
9. Return: SUCCESS (tasks ready for execution) ✅
```

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

## Path Conventions

Repository cleanup uses script-based approach rather than traditional application structure:
- Scripts: `scripts/cleanup/` (new directory for cleanup operations)
- Tests: `tests/contract/` for contract tests
- Documentation: Updates to existing docs

## Phase 3.1: Setup

- [x] **T001** Create cleanup script structure
  - Create `scripts/cleanup/` directory
  - Create `scripts/cleanup/__init__.py`
  - Create `scripts/cleanup/models.py` for data classes
  - Create `scripts/cleanup/operations.py` for cleanup functions
  - Create `scripts/cleanup/main.py` for CLI entry point

- [x] **T002** [P] Install and configure cleanup dependencies
  - Verify pytest installed (8.4.1 ✓)
  - Add gitpython to dependencies if not present (3.1.44 ✓)
  - Configure linting for new scripts directory

- [x] **T003** Create baseline test snapshot
  - Run `pytest --tb=short -v > baseline_tests.txt`
  - Record baseline test count (manual validation approach)
  - Create `cleanup_log.txt` for operation tracking

## Phase 3.2: Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3

**CRITICAL: These tests MUST be written and MUST FAIL before ANY implementation**

- [ ] **T004** [P] Contract test scan_repository() in `tests/contract/test_scan_repository_contract.py`
  - Test valid repository scan
  - Test exclude_dirs parameter
  - Test invalid repo_root raises ValueError
  - Test permission errors
  - Verify FileInventory postconditions

- [ ] **T005** [P] Contract test classify_files() in `tests/contract/test_classify_files_contract.py`
  - Test file classification into categories
  - Test essential files detection
  - Test relocatable files detection
  - Test removable patterns (temp, old outputs, historical)
  - Verify ClassificationReport postconditions

- [ ] **T006** [P] Contract test remove_files() in `tests/contract/test_remove_files_contract.py`
  - Test file removal
  - Test dry_run mode
  - Test removal failure handling
  - Test bytes_freed calculation
  - Verify RemovalReport postconditions

- [ ] **T007** [P] Contract test move_files() in `tests/contract/test_move_files_contract.py`
  - Test file relocation
  - Test directory creation
  - Test dry_run mode
  - Test git mv integration
  - Verify MoveReport postconditions

- [ ] **T008** [P] Contract test consolidate_duplicates() in `tests/contract/test_consolidate_duplicates_contract.py`
  - Test duplicate detection by basename
  - Test keeping newest version
  - Test removal of older duplicates
  - Verify duplicate removal results

- [ ] **T009** [P] Contract test validate_tests() in `tests/contract/test_validate_tests_contract.py`
  - Test test suite execution
  - Test baseline comparison
  - Test pass rate validation (100%)
  - Test test count matching
  - Verify TestReport postconditions

- [ ] **T010** [P] Contract test rollback_changes() in `tests/contract/test_rollback_changes_contract.py`
  - Test git restore --staged
  - Test git restore (working directory)
  - Test rollback verification
  - Test rollback for different operation types

- [ ] **T011** [P] Contract test check_broken_links() in `tests/contract/test_check_broken_links_contract.py`
  - Test Markdown link parsing
  - Test relative path validation
  - Test broken link detection
  - Verify DocumentationUpdateReport structure

- [ ] **T012** [P] Contract test update_documentation() in `tests/contract/test_update_documentation_contract.py`
  - Test link updates based on path_mapping
  - Test relative path recalculation
  - Test dry_run mode
  - Verify documentation update results

- [ ] **T013** [P] Contract test update_documentation_index() in `tests/contract/test_update_documentation_index_contract.py`
  - Test DOCUMENTATION_INDEX.md parsing
  - Test link updates
  - Test table structure preservation
  - Verify index update success

## Phase 3.3: Core Implementation (ONLY after tests are failing)

- [ ] **T014** [P] Implement data models in `scripts/cleanup/models.py`
  - FileCategory enum
  - RepositoryFile dataclass
  - TopLevelFile, DocumentationFile, GeneratedOutput, StatusFile, TemporaryFile subclasses
  - FileInventory, ClassificationReport, RemovalReport, MoveReport, TestReport, DocumentationUpdateReport
  - BrokenLink dataclass
  - Validation functions

- [ ] **T015** Implement scan_repository() in `scripts/cleanup/operations.py`
  - Walk repository directory tree
  - Exclude .git, .venv, node_modules
  - Create RepositoryFile instances with metadata
  - Return FileInventory
  - Must pass T004 contract tests

- [ ] **T016** Implement classify_files() in `scripts/cleanup/operations.py`
  - Define ESSENTIAL_FILES list
  - Implement classification logic (Essential/Relocatable/Removable/Review)
  - Pattern matching for temporary files
  - Pattern matching for old outputs
  - Duplicate detection logic
  - Return ClassificationReport
  - Must pass T005 contract tests

- [ ] **T017** Implement remove_files() in `scripts/cleanup/operations.py`
  - Implement dry_run mode
  - File deletion with error handling
  - Track bytes freed
  - Record failures
  - Return RemovalReport
  - Must pass T006 contract tests

- [ ] **T018** Implement move_files() in `scripts/cleanup/operations.py`
  - Implement dry_run mode
  - Create target directories
  - Use git mv for tracked files
  - Use shutil.move for untracked files
  - Track moved files
  - Return MoveReport
  - Must pass T007 contract tests

- [ ] **T019** Implement consolidate_duplicates() in `scripts/cleanup/operations.py`
  - Group files by basename
  - Sort by modification time (newest first)
  - Mark oldest as duplicates
  - Use remove_files() for deletion
  - Return RemovalReport
  - Must pass T008 contract tests

- [ ] **T020** Implement validate_tests() in `scripts/cleanup/operations.py`
  - Execute pytest command
  - Parse test output for counts
  - Compare to baseline if provided
  - Return TestReport with comparison
  - Must pass T009 contract tests

- [ ] **T021** Implement rollback_changes() in `scripts/cleanup/operations.py`
  - Execute git restore --staged
  - Execute git restore (working directory)
  - Verify rollback success
  - Return RollbackReport
  - Must pass T010 contract tests

- [ ] **T022** Implement check_broken_links() in `scripts/cleanup/operations.py`
  - Parse Markdown files for links `[text](path)`
  - Validate relative path targets
  - Track broken links with context
  - Return DocumentationUpdateReport
  - Must pass T011 contract tests

- [ ] **T023** Implement update_documentation() in `scripts/cleanup/operations.py`
  - Scan .md files for links
  - Replace old paths with new paths from mapping
  - Recalculate relative paths
  - Update file contents
  - Return DocumentationUpdateReport
  - Must pass T012 contract tests

- [ ] **T024** Implement update_documentation_index() in `scripts/cleanup/operations.py`
  - Parse DOCUMENTATION_INDEX.md
  - Update links based on path_mapping
  - Preserve table formatting
  - Write updated content
  - Return success boolean
  - Must pass T013 contract tests

## Phase 3.4: Integration

- [ ] **T025** Create cleanup CLI in `scripts/cleanup/main.py`
  - Argument parsing (--dry-run, --batch, --skip-tests)
  - Orchestrate all cleanup operations
  - Follow batch sequence from quickstart.md
  - Test validation after each batch
  - Rollback on failure
  - Logging to cleanup_log.txt

- [ ] **T026** Integration test: Full cleanup workflow in `tests/integration/test_cleanup_workflow.py`
  - Set up test repository with sample files
  - Run full cleanup sequence
  - Validate file removals
  - Validate file moves
  - Validate test pass rate maintained
  - Validate documentation updated

- [ ] **T027** Integration test: Rollback on test failure in `tests/integration/test_cleanup_rollback.py`
  - Simulate test failure after cleanup batch
  - Verify rollback_changes() is triggered
  - Verify files restored to original state
  - Verify git status clean

## Phase 3.5: Polish

- [ ] **T028** [P] Unit tests for file classification logic in `tests/unit/test_classification.py`
  - Test ESSENTIAL_FILES detection
  - Test temporary file patterns
  - Test old output patterns
  - Test duplicate detection
  - Test status file identification

- [ ] **T029** [P] Unit tests for link parsing in `tests/unit/test_link_parsing.py`
  - Test Markdown link extraction
  - Test relative path resolution
  - Test broken link detection
  - Test path mapping updates

- [ ] **T030** [P] Update CLAUDE.md with cleanup script usage
  - Document cleanup script location
  - Document CLI arguments
  - Document batch execution approach
  - Document rollback procedure

- [ ] **T031** Create cleanup execution guide in `docs/CLEANUP_GUIDE.md`
  - Document when to run cleanup
  - Document safety checks
  - Document validation steps
  - Reference quickstart.md

- [ ] **T032** Manual validation using `quickstart.md`
  - Execute all 12 steps from quickstart
  - Verify each batch completes successfully
  - Verify all 136 tests pass
  - Verify documentation links valid
  - Verify top-level directory cleaner

## Dependencies

```
Setup (T001-T003)
  ↓
Tests (T004-T013) [ALL PARALLEL - different files]
  ↓
Models (T014) [blocks all implementations]
  ↓
Operations (T015-T024) [sequential - must pass corresponding tests]
  ↓
Integration (T025-T027)
  ↓
Polish (T028-T032) [T028-T031 parallel, T032 sequential]
```

### Detailed Dependencies

- **T001-T003**: No dependencies (setup)
- **T004-T013**: Depend on T001 (test structure exists)
- **T014**: Depends on T004-T013 (tests written, must fail)
- **T015**: Depends on T014 (models), must pass T004
- **T016**: Depends on T014 (models), must pass T005
- **T017**: Depends on T014 (models), must pass T006
- **T018**: Depends on T014 (models), must pass T007
- **T019**: Depends on T017 (uses remove_files), must pass T008
- **T020**: Depends on T014 (models), must pass T009
- **T021**: Depends on T014 (models), must pass T010
- **T022**: Depends on T014 (models), must pass T011
- **T023**: Depends on T014 (models), T022 (link parsing), must pass T012
- **T024**: Depends on T014 (models), must pass T013
- **T025**: Depends on T015-T024 (all operations)
- **T026**: Depends on T025 (CLI implementation)
- **T027**: Depends on T025, T021 (rollback implementation)
- **T028-T031**: Depend on T015-T024 (operations implemented)
- **T032**: Depends on T025-T031 (everything complete)

## Parallel Execution Examples

### Phase 3.2: All Contract Tests in Parallel

These tests are completely independent (different files):

```bash
# Launch T004-T013 together (10 contract tests):
pytest tests/contract/test_scan_repository_contract.py &
pytest tests/contract/test_classify_files_contract.py &
pytest tests/contract/test_remove_files_contract.py &
pytest tests/contract/test_move_files_contract.py &
pytest tests/contract/test_consolidate_duplicates_contract.py &
pytest tests/contract/test_validate_tests_contract.py &
pytest tests/contract/test_rollback_changes_contract.py &
pytest tests/contract/test_check_broken_links_contract.py &
pytest tests/contract/test_update_documentation_contract.py &
pytest tests/contract/test_update_documentation_index_contract.py &
wait
```

### Phase 3.5: Unit Tests and Documentation in Parallel

```bash
# Launch T028-T031 together:
pytest tests/unit/test_classification.py &  # T028
pytest tests/unit/test_link_parsing.py &    # T029
# Edit CLAUDE.md (T030) in parallel
# Edit docs/CLEANUP_GUIDE.md (T031) in parallel
```

## Batch Execution Sequence

The cleanup operations must be executed in specific batches (from quickstart.md):

1. **BATCH 1**: Remove temporary/cache files (T017)
2. **BATCH 2**: Remove old evaluation reports (T017)
3. **BATCH 3**: Remove historical tracking files (T017)
4. **BATCH 4**: Consolidate duplicate documentation (T019)
5. **BATCH 5**: Move status files to docs/ (T018)
6. **BATCH 6**: Update documentation links (T023)
7. **BATCH 7**: Update DOCUMENTATION_INDEX.md (T024)

After each batch: Run T020 (validate_tests) and T021 (rollback_changes if tests fail)

## Notes

- **[P] tasks** = different files, no dependencies, can execute in parallel
- **TDD critical**: All contract tests (T004-T013) MUST fail before implementation starts
- **Test validation**: Run full pytest suite after each implementation task to verify contract tests pass
- **Batch approach**: Cleanup operations executed in batches with test validation between each
- **Rollback safety**: All changes reversible via git history (NFR-001)
- **No database operations**: This is file system only, no IRIS database involved
- **Constitutional compliance**: Maintenance task exempt from RAG pipeline requirements

## Task Generation Rules Applied

1. **From Contracts** (cleanup-operations.md):
   - 10 contract operations → 10 contract test tasks [P] (T004-T013)
   - 10 operations → 10 implementation tasks (T015-T024)

2. **From Data Model** (data-model.md):
   - 13 entities → 1 consolidated model task (T014) [P]
   - Relationships → integrated in operations (T015-T024)

3. **From User Stories** (spec.md acceptance scenarios):
   - 5 acceptance scenarios → covered by integration tests (T026-T027)
   - Quickstart scenarios → manual validation task (T032)

4. **Ordering**:
   - Setup (T001-T003) → Tests (T004-T013) → Models (T014) → Operations (T015-T024) → Integration (T025-T027) → Polish (T028-T032)

## Validation Checklist

- [x] All contracts have corresponding tests (10 operations = 10 contract tests)
- [x] All entities have model tasks (13 entities in T014)
- [x] All tests come before implementation (T004-T013 before T015-T024)
- [x] Parallel tasks truly independent (different files confirmed)
- [x] Each task specifies exact file path (all tasks have paths)
- [x] No task modifies same file as another [P] task (verified)

## Success Criteria

After completing all tasks:
- ✓ All 136 tests still pass
- ✓ Status files moved to docs/
- ✓ Old evaluation reports removed
- ✓ Historical tracking files removed
- ✓ Duplicate documentation consolidated
- ✓ All documentation links valid
- ✓ DOCUMENTATION_INDEX.md updated
- ✓ Top-level directory cleaner (~15-20 files vs ~20-25)
- ✓ Git history shows all cleanup commits
- ✓ Cleanup operations fully tested and validated

---

**Tasks Generated**: 2025-10-08
**Ready for Implementation**: Phase 4
**Total Tasks**: 32 (10 contract tests [P], 1 model task [P], 10 operations, 3 integration, 8 polish)
