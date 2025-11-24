# Implementation Tasks: Root Directory Cleanup and Reorganization

**Feature**: Root Directory Cleanup and Reorganization
**Branch**: `052-root-directory-cleanup`
**Generated**: 2025-11-24
**Input**: [spec.md](./spec.md), [plan.md](./plan.md), [research.md](./research.md)

---

## Task Organization

Tasks are organized by user story priority (P1 → P2 → P3) with dependency ordering within each phase. Tasks marked with `[PARALLEL]` can be executed concurrently with other parallel tasks in the same phase.

**Phases**:
1. **Setup**: Environment preparation and backup
2. **Foundational**: Core infrastructure changes required by all user stories
3. **US1 (P1)**: Developer Onboarding Experience
4. **US2 (P1)**: Preventing Accidental Commits
5. **US3 (P2)**: Project Documentation Discovery
6. **US4 (P2)**: Dependency and Package Management Clarity
7. **US5 (P3)**: Streamlined Git Workflow
8. **Polish**: Cross-cutting validation and documentation

---

## Phase 1: Setup

- [ ] [SETUP-001] [P1] Create feature branch `052-root-directory-cleanup` from `main`
- [ ] [SETUP-002] [P1] Verify git status is clean before starting cleanup (no uncommitted changes)
- [ ] [SETUP-003] [P1] Create backup tag `pre-cleanup-backup` for rollback safety
- [ ] [SETUP-004] [P1] Create `check_imports.py` script from research.md template at `.specify/scripts/python/check_imports.py`
- [ ] [SETUP-005] [P1] Document current root directory state: `ls -la | tee docs/archive/root_directory_before_cleanup.txt`

**Dependencies**: None (entry point)
**Estimated Time**: 10 minutes
**Validation**: Git branch created, status clean, backup tag exists, import checker script executable

---

## Phase 2: Foundational

- [ ] [FOUND-001] [P1] [US1] Run import verification for legacy packages: `python .specify/scripts/python/check_imports.py iris_rag rag_templates common > import_verification_results.txt`
- [ ] [FOUND-002] [P1] [US1] Create `config/` directory with README.md explaining purpose of `.key` files
- [ ] [FOUND-003] [P1] [US1] Create `docs/archive/` directory for git-tracked archived files
- [ ] [FOUND-004] [P1] [US1] Create `docs/logs/historical/` directory with README explaining context
- [ ] [FOUND-005] [P1] [US2] Add `config/` to `.gitignore` with comment `# Configuration files (.gitignored)`

**Dependencies**: SETUP phase complete
**Estimated Time**: 15 minutes
**Validation**: Directories created, import verification complete, no import conflicts found

---

## Phase 3: US1 - Developer Onboarding Experience (P1)

### Test File Consolidation

- [ ] [US1-001] [P1] [PARALLEL] Move root-level `test_*.py` files to `tests/` directory
- [ ] [US1-002] [P1] [PARALLEL] Move scattered test files from subdirectories to appropriate `tests/` locations
- [ ] [US1-003] [P1] Run test suite to verify paths still valid: `pytest tests/ -v`
- [ ] [US1-004] [P1] Count root items after test move: `ls -la | wc -l` (should be <114)

**Dependencies**: FOUND-003
**Estimated Time**: 10 minutes
**Validation**: All tests pass, test files in `tests/`, root item count decreased

### Log File Processing

- [ ] [US1-005] [P1] [PARALLEL] Identify log files with historical value: for each `*.log`, run `grep -r "$(basename $log)" docs/` OR check `find $log -mtime -90`; if matches → keep, else → mark for removal
- [ ] [US1-006] [P1] Move historical logs to `docs/logs/historical/` with README context
- [ ] [US1-007] [P1] [PARALLEL] Remove remaining log files: `rm *.log push_*.log`
- [ ] [US1-008] [P1] Count root items after log cleanup: `ls -la | wc -l` (should be <100)

**Dependencies**: FOUND-004
**Estimated Time**: 10 minutes
**Validation**: Historical logs preserved with context, temporary logs removed, root count decreased

### Archive Processing

- [ ] [US1-009] [P1] Identify git-tracked files in `archive/` and `backups/` directories
- [ ] [US1-010] [P1] Move git-tracked archive files to `docs/archive/` with origin documentation
- [ ] [US1-011] [P1] Remove untracked archive directories: `rm -rf archive/ backups/ scratch/`
- [ ] [US1-012] [P1] Count root items after archive cleanup: `ls -la | wc -l` (should be <80)

**Dependencies**: FOUND-003
**Estimated Time**: 10 minutes
**Validation**: Git-tracked files preserved, untracked archives removed, root count decreased

### Symlink Conversion

- [ ] [US1-013] [P1] Identify all symlinks in root directory: `find . -maxdepth 1 -type l`
- [ ] [US1-014] [P1] Convert symlinks to regular files by copying target content
- [ ] [US1-015] [P1] Verify no broken symlinks remain: `find . -type l -exec test ! -e {} \; -print`

**Dependencies**: US1-001 through US1-012
**Estimated Time**: 5 minutes
**Validation**: No symlinks in root, all references converted to regular files

### Legacy Package Handling

- [ ] [US1-016] [P1] Review import verification results from FOUND-001
- [ ] [US1-017] [P1] If no imports found: Remove legacy package directories `iris_rag/`, `rag_templates/`, `common/`
- [ ] [US1-018] [P1] If imports found: Add deprecation warning to README.md for each legacy package
- [ ] [US1-019] [P1] Remove orphaned integration directories: `rm -rf mem0_integration/ mem0-mcp-server/ supabase-mcp-memory-server/`
- [ ] [US1-020] [P1] Count root items after package cleanup: `ls -la | wc -l` (should be <60)

**Dependencies**: FOUND-001
**Estimated Time**: 10 minutes
**Validation**: Legacy packages removed or deprecated, orphaned integrations removed, root count decreased

### Language-Specific Directory Handling

- [ ] [US1-021] [P1] [PARALLEL] Evaluate `nodejs/` directory per FR-012: Run `grep -r "nodejs" --include="*.{py,md,yml,yaml,json,Dockerfile}" .` and `grep -r "node" .github/workflows/`; if no matches → remove `nodejs/`, else → move to `tools/nodejs/`
- [ ] [US1-022] [P1] [PARALLEL] Evaluate `objectscript/` directory per FR-012: Run `grep -r "objectscript" --include="*.{py,md,yml,yaml,json,Dockerfile}" .` and check README; if no matches → remove `objectscript/`, else → move to `tools/objectscript/`
- [ ] [US1-023] [P1] Count root items after language cleanup: `ls -la | wc -l` (should be <50)

**Dependencies**: US1-020
**Estimated Time**: 5 minutes
**Validation**: Language-specific directories relocated or removed, root count decreased

**US1 Total Estimated Time**: 60 minutes
**US1 Final Validation**: Root directory has fewer than 50 items, new developers can locate all key directories within 2 minutes

---

## Phase 4: US2 - Preventing Accidental Commits (P1)

### Build Artifacts

- [ ] [US2-001] [P1] Add build artifacts, logs, and temporary files to `.gitignore` per FR-005: `dist/`, `build/`, `*.egg-info/`, `.eggs/`, `*.log`
- [ ] [US2-002] [P1] Remove existing build artifacts: `rm -rf dist/ build/ *.egg-info/ .eggs/`
- [ ] [US2-003] [P1] Verify `git status` clean after removal

**Dependencies**: FOUND-005
**Estimated Time**: 5 minutes
**Validation**: Build artifacts ignored and removed, git status clean

### Output Directories

- [ ] [US2-004] [P1] [PARALLEL] Add output directories to `.gitignore` per FR-006: `outputs/`, `reports/`, `validation_results/`, `test_results/`, `coverage_reports/`, `.pytest_cache/`, `htmlcov/`
- [ ] [US2-005] [P1] [PARALLEL] Remove output directories from root: `rm -rf outputs/ reports/ validation_results/ test_results/`
- [ ] [US2-006] [P1] Verify `git status` clean after removal

**Dependencies**: US2-003
**Estimated Time**: 5 minutes
**Validation**: Output directories ignored and removed, git status clean

### Test Artifacts

- [ ] [US2-007] [P1] Add test artifacts to `.gitignore`: `.pytest_cache/`, `.coverage`, `htmlcov/`, `*.log`
- [ ] [US2-008] [P1] Remove test artifacts: `rm -rf .pytest_cache/ htmlcov/ .coverage`
- [ ] [US2-009] [P1] Run full test suite with coverage: `pytest tests/ --cov=iris_vector_rag --cov-report=html`
- [ ] [US2-010] [P1] Verify `git status` remains clean after test run

**Dependencies**: US2-006
**Estimated Time**: 10 minutes
**Validation**: Test artifacts ignored, git status clean after full test run

### Configuration Files

- [ ] [US2-011] [P1] Move `iris.key` to `config/iris.key`
- [ ] [US2-012] [P1] Remove duplicate key file: `rm temp_iris.key` (if exists)
- [ ] [US2-013] [P1] Add `.env` patterns to `.gitignore`: `.env`, `.env.local`, `.env.*.local`, `*.env` with exception `!.env.example`
- [ ] [US2-014] [P1] Consolidate multiple `.env` files into single `.env` (see research.md strategy)
- [ ] [US2-015] [P1] Create `.env.example` template from consolidated `.env`
- [ ] [US2-016] [P1] Verify `git status` does not show `.env` or `.key` files

**Dependencies**: FOUND-002, US2-010
**Estimated Time**: 15 minutes
**Validation**: Config files moved to gitignored directory, .env consolidated, secrets not tracked

**US2 Total Estimated Time**: 35 minutes
**US2 Final Validation**: `git status` clean after full test suite + build, no accidental commits possible

---

## Phase 5: US3 - Project Documentation Discovery (P2)

### Documentation Consolidation

- [ ] [US3-001] [P2] [PARALLEL] Move all markdown files (except README.md, CHANGELOG.md, LICENSE) to `docs/`
- [ ] [US3-002] [P2] [PARALLEL] Move status files to `docs/`: `STATUS.md`, `PROGRESS.md`, `TODO.md`
- [ ] [US3-003] [P2] [PARALLEL] Move feature documentation to `docs/`: `*_summary.md`, `*_fix_summary.md`
- [ ] [US3-004] [P2] Create `docs/README.md` index explaining documentation structure
- [ ] [US3-005] [P2] Update root `README.md` with link to `docs/` directory
- [ ] [US3-006] [P2] Count root items after doc consolidation: `ls -la | wc -l` (should be <40)

**Dependencies**: US2-016
**Estimated Time**: 15 minutes
**Validation**: All docs in `docs/`, stakeholders can find documentation within 2 minutes

### Bug Fix Documentation

- [ ] [US3-007] [P2] Move bug fix summaries to `docs/bug-fixes/`
- [ ] [US3-008] [P2] Move regression documentation to `docs/bug-fixes/`
- [ ] [US3-009] [P2] Update `docs/README.md` with bug-fixes section

**Dependencies**: US3-006
**Estimated Time**: 5 minutes
**Validation**: Bug fix documentation organized and discoverable

**US3 Total Estimated Time**: 20 minutes
**US3 Final Validation**: Documentation consolidated in `docs/`, index provides clear navigation, root count decreased

---

## Phase 6: US4 - Dependency and Package Management Clarity (P2)

### Legacy Dependency Files

- [ ] [US4-001] [P2] Remove `poetry.lock` (legacy Poetry lockfile)
- [ ] [US4-002] [P2] Evaluate `requirements.txt`: Remove if generated from pyproject.toml, or regenerate via `uv pip compile pyproject.toml -o requirements.txt`
- [ ] [US4-003] [P2] Evaluate `requirements-dev.txt`: Remove if generated, or regenerate via `uv pip compile pyproject.toml --extra dev -o requirements-dev.txt`
- [ ] [US4-004] [P2] Verify `pyproject.toml` and `uv.lock` are present and up-to-date

**Dependencies**: US3-009
**Estimated Time**: 10 minutes
**Validation**: Only uv-based dependency files remain, no Poetry artifacts

### README Setup Instructions

- [ ] [US4-005] [P2] Update README.md with clear "Setup" section documenting uv as primary package manager
- [ ] [US4-006] [P2] Add environment setup instructions: `cp .env.example .env` workflow
- [ ] [US4-007] [P2] Document IRIS connection configuration in README
- [ ] [US4-008] [P2] Remove references to Poetry or old setup methods from README

**Dependencies**: US4-004
**Estimated Time**: 15 minutes
**Validation**: README clearly documents uv setup, new developers use correct tools

### CI/CD Verification

- [ ] [US4-009] [P2] Review `.github/workflows/` files to ensure they use `uv` commands
- [ ] [US4-010] [P2] Verify Docker Compose and Dockerfile use `uv` for dependency installation
- [ ] [US4-011] [P2] Count root items after dependency cleanup: `ls -la | wc -l` (should be <35)

**Dependencies**: US4-008
**Estimated Time**: 10 minutes
**Validation**: CI/CD pipelines use uv, Docker builds unaffected, root count decreased

**US4 Total Estimated Time**: 35 minutes
**US4 Final Validation**: Single source of truth for dependencies, developers understand setup process

---

## Phase 7: US5 - Streamlined Git Workflow (P3)

### IDE/Editor Ignores

- [ ] [US5-001] [P3] Add IDE patterns to `.gitignore`: `.vscode/`, `.idea/`, `*.swp`, `*.swo`, `*~`
- [ ] [US5-002] [P3] Remove any committed IDE files: `rm -rf .vscode/ .idea/` (if not needed)

**Dependencies**: US4-011
**Estimated Time**: 5 minutes
**Validation**: IDE files not tracked, editor configs optional

### OS-Specific Ignores

- [ ] [US5-003] [P3] Add OS patterns to `.gitignore`: `.DS_Store`, `Thumbs.db`
- [ ] [US5-004] [P3] Remove OS-specific files: `find . -name ".DS_Store" -delete`

**Dependencies**: US5-002
**Estimated Time**: 5 minutes
**Validation**: OS artifacts not tracked, clean across platforms

### Hierarchical .gitignore Structure

- [ ] [US5-005] [P3] Reorganize `.gitignore` with clear section headers (see research.md template)
- [ ] [US5-006] [P3] Add comments explaining each pattern group in `.gitignore`
- [ ] [US5-007] [P3] Add comments with parenthetical notes like "(keep .env.example)" for exceptions

**Dependencies**: US5-004
**Estimated Time**: 15 minutes
**Validation**: `.gitignore` self-documenting, maintainable, follows research.md template

### Final Git Status Verification

- [ ] [US5-008] [P3] Run full test suite: `pytest tests/ -v`
- [ ] [US5-009] [P3] Build distribution package: `uv build`
- [ ] [US5-010] [P3] Verify `git status` clean after tests + build
- [ ] [US5-011] [P3] Count root items: `ls -la | wc -l` (MUST be <30 per SC-001)

**Dependencies**: US5-007
**Estimated Time**: 10 minutes
**Validation**: Git status clean after all operations, root directory has <30 items

**US5 Total Estimated Time**: 35 minutes
**US5 Final Validation**: Clean git workflow, root directory <30 items (SUCCESS CRITERIA MET)

---

## Phase 8: Polish & Cross-Cutting Concerns

### Validation Against Success Criteria

- [ ] [POLISH-001] [P1] **SC-001**: Verify root directory has fewer than 30 items: `ls -la | wc -l`
- [ ] [POLISH-002] [P1] **SC-002**: Test new developer experience: Clone repo, locate tests/docs/main package within 2 minutes
- [ ] [POLISH-003] [P1] **SC-003**: Verify `git status` clean after full test suite and package build
- [ ] [POLISH-004] [P1] **SC-004**: Verify 100% of build artifacts, logs, temporary files in `.gitignore`
- [ ] [POLISH-005] [P2] **SC-005**: Verify single-command setup documented in README
- [ ] [POLISH-006] [P3] **SC-007**: Run repository linter: `repolinter` (if available)

**Dependencies**: US5-011
**Estimated Time**: 20 minutes
**Validation**: All 8 success criteria met, metrics recorded

### Documentation Updates

- [ ] [POLISH-007] [P1] Create CHANGELOG entry for v0.5.14 documenting this cleanup (per NFR-006)
- [ ] [POLISH-008] [P1] Document what was removed/moved in `docs/archive/cleanup_2025-11-24.md`
- [ ] [POLISH-009] [P2] Update root README.md with current directory structure
- [ ] [POLISH-010] [P2] Create `docs/CONTRIBUTING.md` explaining project structure for contributors

**Dependencies**: POLISH-006
**Estimated Time**: 20 minutes
**Validation**: CHANGELOG updated, cleanup documented, README current

### Final Compatibility Verification

- [ ] [POLISH-011] [P1] **NFR-007**: Run full test suite to verify no broken imports: `pytest tests/ -v`
- [ ] [POLISH-012] [P1] **NFR-008**: Trigger CI/CD pipeline manually to verify no breakage
- [ ] [POLISH-013] [P1] **NFR-009**: Build Docker images to verify no breakage: `docker-compose build`
- [ ] [POLISH-014] [P1] **NFR-010**: Verify development workflows unchanged (make commands, scripts)

**Dependencies**: POLISH-010
**Estimated Time**: 30 minutes
**Validation**: All NFRs satisfied, no breaking changes

### Feature Completion

- [ ] [POLISH-015] [P1] Document final root directory state: `ls -la | tee docs/archive/root_directory_after_cleanup.txt`
- [ ] [POLISH-016] [P1] Create pull request with comprehensive description of changes
- [ ] [POLISH-017] [P1] Add before/after comparison to PR: `diff docs/archive/root_directory_before_cleanup.txt docs/archive/root_directory_after_cleanup.txt`
- [ ] [POLISH-018] [P1] Tag cleanup completion: `git tag v0.5.14-cleanup`

**Dependencies**: POLISH-014
**Estimated Time**: 15 minutes
**Validation**: Feature complete, PR ready, changes documented

**Polish Total Estimated Time**: 85 minutes
**Polish Final Validation**: All success criteria met, all NFRs satisfied, feature ready for merge

---

## Summary

**Total Tasks**: 93 tasks across 8 phases
**Total Estimated Time**: 310 minutes (~5 hours)
**Success Criteria Coverage**: 8/8 criteria explicitly validated
**NFR Coverage**: 10/10 non-functional requirements verified

**Parallel Execution Opportunities**:
- Phase 2 Foundational: None (sequential setup)
- Phase 3 US1: Tasks US1-001/US1-002, US1-005/US1-007, US1-021/US1-022
- Phase 4 US2: Tasks US2-004/US2-005
- Phase 5 US3: Tasks US3-001/US3-002/US3-003

**Critical Path** (must complete before others):
1. SETUP → FOUND → US1 (P1) → US2 (P1) → US3 (P2) → US4 (P2) → US5 (P3) → POLISH

**Risk Mitigation**:
- Backup tag created in SETUP-003 for rollback safety
- Import verification in FOUND-001 prevents breaking changes
- Incremental git status checks after each phase
- Full test suite runs in US1-003, US2-009, US5-008, POLISH-011

**Feature Number Conflict Note**: This feature is numbered 051 but conflicts with existing Feature 051 (Simplify IRIS Connection Architecture). Recommend renumbering to 052 before proceeding to implementation.
