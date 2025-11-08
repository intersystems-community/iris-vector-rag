# Tasks: Root Directory Cleanup

**Feature**: 052-i-thought-we
**Input**: Design documents from `/Users/tdyar/ws/rag-templates/specs/052-i-thought-we/`
**Prerequisites**: plan.md, research.md, data-model.md, contracts/, quickstart.md

## Overview

This task list reduces the repository root directory from 991 files to <20 essential files by organizing configuration files, scripts, documentation, and artifacts into appropriate subdirectories while maintaining all CI/CD and development workflow functionality.

**Tech Stack**: Python 3.11+, Bash, Git 2.0+, docker-compose, make
**Scope**: File reorganization (no code changes to iris_rag/ framework)
**Validation**: Contract tests in `specs/052-i-thought-we/contracts/test_root_directory_contract.py`

## Format: `[ID] [P?] Description`
- **[P]**: Can run in parallel (different files, no dependencies)
- Include exact file paths in descriptions

---

## Phase 3.1: Setup & Preparation (5 minutes)

- [ ] **T001** [P] Create target directory structure
  - Create `config/`, `config/docker/`, `docs/`, `scripts/setup/`, `scripts/upload/`, `scripts/docker/`, `tests/artifacts/`, `archive/eval_results/`
  - Verify directories created with proper permissions
  - File: Repository structure (multiple directories)

- [ ] **T002** [P] Create git recovery tag
  - Run: `git tag before-cleanup-052`
  - Provides rollback point if needed
  - File: Git repository metadata

- [ ] **T003** [P] Verify current file count baseline
  - Count files in root: `ls -1 | wc -l` (expected: ~991)
  - Document baseline in commit message
  - File: Root directory analysis

---

## Phase 3.2: Contract Tests First (TDD) ⚠️ MUST COMPLETE BEFORE 3.3

**CRITICAL: These tests MUST be written and MUST FAIL before ANY cleanup**

- [ ] **T004** Run contract tests to verify they fail initially
  - Run: `pytest specs/052-i-thought-we/contracts/test_root_directory_contract.py -v`
  - Expected: Most tests FAIL (files still in root)
  - Validates test coverage before cleanup
  - File: `specs/052-i-thought-we/contracts/test_root_directory_contract.py` (already exists)

---

## Phase 3.3: Low-Risk Cleanup (10 minutes)

**ONLY proceed after T004 shows failing tests**

- [ ] **T005** [P] Delete obsolete log files (942 indexing logs)
  - Delete: `indexing_CONTINUOUS_RUN_*.log` (942 files)
  - Delete: `evaluation.log`, `cleanup_log.txt`
  - Verify: `ls *.log 2>/dev/null` returns empty
  - Files: Root directory `.log` files

- [ ] **T006** [P] Delete temporary and generated files
  - Delete: `.DS_Store` (find and remove all)
  - Delete: `.coverage`, `coverage.json`
  - Delete: `htmlcov/`, `coverage_html/` directories
  - Delete: `comprehensive_ragas_results_*` directories in root
  - Files: Temporary artifacts

- [ ] **T007** [P] Move documentation files to docs/
  - Move: `CONTRIBUTING.md` → `docs/CONTRIBUTING.md`
  - Move: `USER_GUIDE.md` → `docs/USER_GUIDE.md` (if exists)
  - Move: `CLAUDE.md` → `docs/CLAUDE.md` (if exists)
  - Verify: `README.md` remains in root
  - Files: `CONTRIBUTING.md`, `USER_GUIDE.md`, `CLAUDE.md`

- [ ] **T008** [P] Archive old evaluation results
  - Move: Old `eval_results/comprehensive_ragas_results_*` → `archive/eval_results/`
  - Keep: 5 most recent evaluation results (optional)
  - Or: Move all to archive
  - Files: `eval_results/` directory contents

- [ ] **T009** Update .gitignore for logs and artifacts
  - Add: `logs/`, `*.log`, `tests/artifacts/`, `archive/`
  - Add: `htmlcov/`, `coverage_html/`, `.coverage`, `coverage.json`
  - Ensure: `.DS_Store`, `Thumbs.db` entries exist
  - File: `.gitignore`

- [ ] **T010** Commit Phase 1 changes
  - Run: `git add -A`
  - Commit: "chore: cleanup root directory - remove logs, temp files, move docs (Phase 1)"
  - File: Git repository

- [ ] **T011** Validate Phase 1 cleanup
  - Run contract tests: `pytest specs/052-i-thought-we/contracts/test_root_directory_contract.py::TestRootDirectoryContract::test_no_log_files_in_root -v`
  - Run contract tests: `pytest specs/052-i-thought-we/contracts/test_root_directory_contract.py::TestRootDirectoryContract::test_no_temporary_files_in_root -v`
  - Run contract tests: `pytest specs/052-i-thought-we/contracts/test_root_directory_contract.py::TestRootDirectoryContract::test_contributing_in_docs -v`
  - Expected: These 3 tests now PASS
  - File: Contract test validation

---

## Phase 3.4: Medium-Risk Cleanup (15 minutes)

- [ ] **T012** [P] Move shell scripts to scripts/ subdirectories
  - Move: `activate_env.sh` → `scripts/activate_env.sh`
  - Move: `docker-entrypoint-mcp.sh` → `scripts/docker/docker-entrypoint-mcp.sh`
  - Move: `upload_to_pypi.sh` → `scripts/upload/upload_to_pypi.sh`
  - Move: `setup_iris_env.sh` → `scripts/setup/setup_iris_env.sh` (if exists)
  - Maintain: Executable permissions (chmod +x)
  - Files: Shell script files in root

- [ ] **T013** Update Makefile script references
  - Update: `activate_env.sh` → `scripts/activate_env.sh`
  - Update: `setup_iris_env.sh` → `scripts/setup/setup_iris_env.sh`
  - Update: Any other script paths in make targets
  - Test: `make -n setup-env` (dry run)
  - File: `Makefile`

- [ ] **T014** Commit Phase 2 changes
  - Run: `git add -A`
  - Commit: "chore: move scripts to scripts/ directory and update Makefile (Phase 2)"
  - File: Git repository

- [ ] **T015** Validate Phase 2 cleanup
  - Run contract tests: `pytest specs/052-i-thought-we/contracts/test_root_directory_contract.py::TestRootDirectoryContract::test_no_shell_scripts_in_root -v`
  - Run contract tests: `pytest specs/052-i-thought-we/contracts/test_root_directory_contract.py::TestRootDirectoryContract::test_scripts_directory_exists -v`
  - Test: `make setup-env` (if safe to run)
  - Expected: Script-related tests now PASS
  - File: Contract test validation

---

## Phase 3.5: High-Risk Cleanup (20 minutes)

- [ ] **T016** [P] Move configuration files to config/
  - Move: `.flake8` → `config/.flake8`
  - Move: `.coveragerc` → `config/.coveragerc`
  - Move: `.coveragerc.ci` → `config/.coveragerc.ci` (if exists)
  - Verify: Tools will find via parent directory search
  - Files: Configuration files in root

- [ ] **T017** [P] Move docker-compose variants to config/docker/
  - Move: `docker-compose.api.yml` → `config/docker/docker-compose.api.yml`
  - Move: `docker-compose.full.yml` → `config/docker/docker-compose.full.yml`
  - Move: `docker-compose.licensed.yml` → `config/docker/docker-compose.licensed.yml`
  - Move: `docker-compose.mcp.yml` → `config/docker/docker-compose.mcp.yml`
  - Move: `docker-compose.test.yml` → `config/docker/docker-compose.test.yml`
  - Move: `docker-compose.iris-only.yml` → `config/docker/docker-compose.iris-only.yml`
  - Verify: `docker-compose.yml` remains in root
  - Files: Docker Compose variant files

- [ ] **T018** Update Makefile docker-compose paths
  - Update all docker-compose variant references to `config/docker/`
  - Example: `docker-compose -f docker-compose.licensed.yml` → `docker-compose -f config/docker/docker-compose.licensed.yml`
  - Update targets: docker-up-licensed, docker-up-mcp, docker-up-api, etc.
  - File: `Makefile`

- [ ] **T019** Update docker-compose files with new script paths
  - In `config/docker/docker-compose.mcp.yml`: Update `docker-entrypoint-mcp.sh` path
  - Change: `./docker-entrypoint-mcp.sh` → `./scripts/docker/docker-entrypoint-mcp.sh`
  - Verify: All entrypoint and volume paths updated
  - Files: `config/docker/docker-compose.*.yml`

- [ ] **T020** [P] Move test artifacts to tests/artifacts/
  - Move: `htmlcov/` → `tests/artifacts/htmlcov/` (if exists)
  - Move: `coverage_html/` → `tests/artifacts/coverage_html/` (if exists)
  - Move: Remaining `eval_results/` → `tests/artifacts/ragas/` (if not archived)
  - Files: Test artifact directories

- [ ] **T021** Commit Phase 3 changes
  - Run: `git add -A`
  - Commit: "chore: move config files to config/, update tool paths (Phase 3)"
  - File: Git repository

- [ ] **T022** Validate Phase 3 cleanup - Tool configurations
  - Test flake8: `flake8 . --count` (should find config/.flake8)
  - Test coverage: `pytest --cov=iris_rag --cov-report=html` (should find config/.coveragerc)
  - Test pre-commit: `pre-commit run --all-files` (should find .pre-commit-config.yaml in root)
  - Expected: All tools work correctly
  - File: Tool validation

- [ ] **T023** Validate Phase 3 cleanup - Docker Compose
  - Test primary: `docker-compose config` (verify no errors)
  - Test variant: `docker-compose -f config/docker/docker-compose.licensed.yml config`
  - Test variant: `docker-compose -f config/docker/docker-compose.mcp.yml config`
  - Expected: All compose files valid
  - File: Docker Compose validation

---

## Phase 3.6: Validation & Documentation (5 minutes)

- [ ] **T024** Run full contract test suite
  - Run: `pytest specs/052-i-thought-we/contracts/test_root_directory_contract.py -v`
  - Expected: All 19+ contract tests PASS
  - Address: Any failures before proceeding
  - File: `specs/052-i-thought-we/contracts/test_root_directory_contract.py`

- [ ] **T025** Verify root file count target met
  - Count: `ls -1 | wc -l` (should be ≤15 visible files)
  - Count: `ls -1a | grep -v '^\\.$' | grep -v '^\\.\\.$' | wc -l` (all files including hidden)
  - Expected: ≤20 total files in root
  - File: Root directory validation

- [ ] **T026** Update README.md documentation
  - Update: Docker Compose examples to reference new paths
  - Example: `docker-compose -f config/docker/docker-compose.licensed.yml up -d`
  - Update: Any script references
  - File: `README.md`

- [ ] **T027** Update docs/CLAUDE.md documentation
  - Update: Script paths in development commands
  - Update: Docker Compose variant paths
  - Update: Any affected Makefile target examples
  - File: `docs/CLAUDE.md` (moved from root)

- [ ] **T028** Update .claude configuration (if exists)
  - Update: Path to CLAUDE.md if .claude config references it
  - Check: `~/.claude/` or project-specific config
  - File: `.claude/` configuration

- [ ] **T029** Final validation - Run full test suite
  - Run: `make test` (all pytest tests)
  - Expected: All tests pass (no regressions from file moves)
  - File: Full test suite

- [ ] **T030** Final validation - Verify Makefile targets
  - Test: `make setup-env` (if safe)
  - Test: `make install`
  - Test: `make docker-up && make docker-down`
  - Expected: All critical make targets work
  - File: Makefile validation

- [ ] **T031** Commit Phase 4 documentation updates
  - Run: `git add -A`
  - Commit: "docs: update documentation for new directory structure (Phase 4)"
  - File: Git repository

---

## Dependencies

**Sequential Dependencies**:
- T001-T003 (Setup) must complete before T004 (Contract tests)
- T004 (Contract tests) must FAIL before proceeding to T005-T031
- T005-T008 (Phase 1 cleanup) → T009 (.gitignore) → T010 (Commit) → T011 (Validate)
- T011 validation PASS required before T012-T015 (Phase 2)
- T012 (Move scripts) → T013 (Update Makefile) → T014 (Commit) → T015 (Validate)
- T015 validation PASS required before T016-T023 (Phase 3)
- T016-T017 (Move configs) → T018-T019 (Update paths) → T020 (Move artifacts) → T021 (Commit) → T022-T023 (Validate)
- T022-T023 validation PASS required before T024 (Full contract tests)
- T024 (All tests PASS) required before T026-T028 (Documentation)
- T026-T028 (Docs) → T029-T030 (Final validation) → T031 (Final commit)

**Parallel Opportunities**:
- T001, T002, T003 can run in parallel (setup tasks)
- T005, T006, T007, T008 can run in parallel (Phase 1 cleanup - different files)
- T016, T017 can run in parallel (move configs - different files)
- Cannot parallelize: T013 depends on T012, T018-T019 depend on T017

---

## Parallel Execution Examples

### Setup Phase (T001-T003)
```bash
# Run in parallel
Task: "Create target directory structure"
Task: "Create git recovery tag"
Task: "Verify current file count baseline"
```

### Phase 1 Cleanup (T005-T008)
```bash
# Run in parallel - different file operations
Task: "Delete obsolete log files (942 indexing logs)"
Task: "Delete temporary and generated files"
Task: "Move documentation files to docs/"
Task: "Archive old evaluation results"
```

### Phase 3 Config Moves (T016-T017)
```bash
# Run in parallel - different files
Task: "Move configuration files to config/"
Task: "Move docker-compose variants to config/docker/"
```

---

## Rollback Procedure

If any task fails or validation shows issues:

```bash
# Option 1: Rollback to recovery tag
git reset --hard before-cleanup-052

# Option 2: Revert last commit
git revert HEAD

# Option 3: Restore specific files
git checkout HEAD~1 -- Makefile config/ docs/
```

---

## Success Criteria Checklist

After completing all tasks:

- ✅ Root directory has ≤15 visible files: `ls -1 | wc -l`
- ✅ All 19+ contract tests pass: `pytest specs/052-i-thought-we/contracts/ -v`
- ✅ Linting works: `flake8 .`
- ✅ Coverage works: `pytest --cov=iris_rag --cov-report=html`
- ✅ Pre-commit works: `pre-commit run --all-files`
- ✅ Docker compose works: `docker-compose up -d && docker-compose down`
- ✅ Docker variants work: `docker-compose -f config/docker/docker-compose.licensed.yml config`
- ✅ Makefile targets work: `make test`
- ✅ Documentation updated: README.md, docs/CLAUDE.md reference new paths
- ✅ Full test suite passes: `make test`
- ✅ Git status clean: `git status`

---

## Task Summary

**Total Tasks**: 31
**Estimated Time**: 50-60 minutes
**Phases**: 4 (Setup, Low-Risk, Medium-Risk, High-Risk, Validation)
**Parallel Tasks**: 8 tasks can run in parallel (marked [P])
**Sequential Tasks**: 23 tasks (dependencies require order)

**Breakdown by Phase**:
- Phase 3.1 (Setup): 3 tasks, 5 minutes
- Phase 3.2 (Contract Tests): 1 task, 2 minutes
- Phase 3.3 (Low-Risk): 7 tasks, 10 minutes
- Phase 3.4 (Medium-Risk): 4 tasks, 15 minutes
- Phase 3.5 (High-Risk): 8 tasks, 20 minutes
- Phase 3.6 (Validation): 8 tasks, 10 minutes

---

## Notes

- **[P] tasks**: Different files, no dependencies, can run in parallel
- **Contract tests**: Must FAIL initially (T004), then PASS incrementally (T011, T015, T024)
- **Incremental validation**: Validate after each phase before proceeding
- **Git commits**: Commit after each major phase for rollback capability
- **No code changes**: This is file reorganization only, iris_rag/ framework untouched
- **Backward compatibility**: All CI/CD and development workflows must continue working

---

## Related Documents

- [spec.md](./spec.md) - Feature specification with 16 functional requirements
- [research.md](./research.md) - Research findings on best practices
- [data-model.md](./data-model.md) - 8 file categories and organization rules
- [quickstart.md](./quickstart.md) - Detailed execution guide with bash commands
- [contracts/test_root_directory_contract.py](./contracts/test_root_directory_contract.py) - 19 validation tests
- [plan.md](./plan.md) - Implementation plan with constitution check
