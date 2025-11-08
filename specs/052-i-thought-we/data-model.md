# Data Model: Root Directory Cleanup

**Feature**: 052-i-thought-we
**Date**: 2025-11-07

## Overview

This document defines the file categories, organization rules, and directory structure for the root directory cleanup. It serves as the canonical reference for where files should be located and why.

---

## File Categories

### Category 1: Essential Root Files (KEEP IN ROOT)

**Definition**: Files that MUST remain in the repository root directory for proper tool operation or GitHub visibility.

**Rules**:
- Maximum 15 files allowed in this category
- Each file must have a clear justification
- No subdirectory alternative exists

**Files**:

| File | Purpose | Justification | Cannot Move Because |
|------|---------|---------------|---------------------|
| `README.md` | Project overview | GitHub displays this on landing page | GitHub convention |
| `LICENSE` | Legal license | GitHub displays in sidebar | GitHub convention |
| `pyproject.toml` | Python package config | pip/setuptools searches root | Package manager convention |
| `Makefile` | Build automation | make searches root directory | Make convention |
| `docker-compose.yml` | Default Docker setup | docker-compose searches root | Docker Compose convention |
| `.gitignore` | Version control ignores | Git searches root | Git convention |
| `.env.example` | Environment template | Developer convenience | Convention |
| `.pre-commit-config.yaml` | Pre-commit hooks config | pre-commit requires root location | Tool requirement |

**Total**: 8 files

---

### Category 2: Configuration Files (MOVE TO config/)

**Definition**: Files that configure development tools, linters, test frameworks, and Docker variants.

**Rules**:
- All files go to `config/` root or appropriate subdirectory
- Tools that support parent directory search can be moved
- Update CI/CD paths if necessary

**Destination Directory Structure**:
```
config/
├── .flake8                          # Linting configuration
├── .coveragerc                      # Coverage configuration
├── .coveragerc.ci                   # CI-specific coverage config
└── docker/                          # Docker Compose variants
    ├── docker-compose.api.yml
    ├── docker-compose.full.yml
    ├── docker-compose.licensed.yml
    ├── docker-compose.mcp.yml
    ├── docker-compose.test.yml
    └── docker-compose.iris-only.yml
```

**Files to Move**:

| Current Path | New Path | Tool Impact | Mitigation |
|--------------|----------|-------------|------------|
| `.flake8` | `config/.flake8` | flake8 searches parent dirs | None needed (automatic) |
| `.coveragerc` | `config/.coveragerc` | coverage searches parent dirs | None needed (automatic) |
| `.coveragerc.ci` | `config/.coveragerc.ci` | coverage searches parent dirs | None needed (automatic) |
| `docker-compose.api.yml` | `config/docker/docker-compose.api.yml` | Requires -f flag | Update Makefile targets |
| `docker-compose.full.yml` | `config/docker/docker-compose.full.yml` | Requires -f flag | Update Makefile targets |
| `docker-compose.licensed.yml` | `config/docker/docker-compose.licensed.yml` | Requires -f flag | Update Makefile targets |
| `docker-compose.mcp.yml` | `config/docker/docker-compose.mcp.yml` | Requires -f flag | Update Makefile targets |
| `docker-compose.test.yml` | `config/docker/docker-compose.test.yml` | Requires -f flag | Update Makefile targets |
| `docker-compose.iris-only.yml` | `config/docker/docker-compose.iris-only.yml` | Requires -f flag | Update Makefile targets |

**Total**: 9 files

---

### Category 3: Shell Scripts (MOVE TO scripts/)

**Definition**: Executable shell scripts (.sh files) used for setup, deployment, utilities, and automation.

**Rules**:
- All .sh files go to `scripts/` with logical subdirectories
- Maintain executable permissions (chmod +x)
- Update Makefile and documentation references

**Destination Directory Structure**:
```
scripts/
├── activate_env.sh                  # Environment activation
├── setup/
│   ├── setup_iris_env.sh           # IRIS setup scripts
│   └── ... (other setup scripts)
├── upload/
│   └── upload_to_pypi.sh           # PyPI upload script
├── docker/
│   └── docker-entrypoint-mcp.sh    # Docker entrypoint for MCP
└── ci/
    └── ... (CI-specific scripts)
```

**Files to Move**:

| Current Path | New Path | Referenced By | Update Required |
|--------------|----------|---------------|-----------------|
| `activate_env.sh` | `scripts/activate_env.sh` | Makefile, docs | Yes - Makefile |
| `docker-entrypoint-mcp.sh` | `scripts/docker/docker-entrypoint-mcp.sh` | docker-compose.mcp.yml | Yes - Docker compose |
| `upload_to_pypi.sh` | `scripts/upload/upload_to_pypi.sh` | Manual usage, docs | Yes - Documentation |
| `setup_iris_env.sh` | `scripts/setup/setup_iris_env.sh` | Makefile, docs | Yes - Makefile |

**Total**: 4+ files (exact count determined during implementation)

---

### Category 4: Documentation Files (MOVE TO docs/)

**Definition**: Markdown files providing user guides, contributor documentation, and technical references.

**Rules**:
- All .md files EXCEPT README.md go to `docs/`
- Maintain relative links (update if necessary)
- Update GitHub Pages config if applicable

**Destination Directory Structure**:
```
docs/
├── CONTRIBUTING.md                  # Contribution guidelines
├── USER_GUIDE.md                   # User documentation
├── CLAUDE.md                       # Claude Code instructions
├── API_REFERENCE.md                # API documentation (if exists)
├── DEVELOPMENT.md                  # Development guide
└── ... (other .md files)
```

**Files to Move**:

| Current Path | New Path | GitHub Recognizes | Update Required |
|--------------|----------|-------------------|-----------------|
| `CONTRIBUTING.md` | `docs/CONTRIBUTING.md` | Yes (GitHub finds in docs/) | No |
| `USER_GUIDE.md` | `docs/USER_GUIDE.md` | No | Update references |
| `CLAUDE.md` | `docs/CLAUDE.md` | No | Update .claude config |

**Total**: 3+ files

---

### Category 5: Log Files (MOVE TO logs/ OR DELETE)

**Definition**: Log files generated by application execution, test runs, and continuous indexing operations.

**Rules**:
- 942 indexing logs go to `logs/indexing/` (if needed) or DELETE
- All logs/ directory added to .gitignore
- Individual log files (.log) deleted if obsolete

**Destination Directory Structure**:
```
logs/                               # .gitignored directory
└── indexing/                       # Historical indexing logs (if kept)
    └── indexing_CONTINUOUS_RUN_*.log
```

**Files to Move or Delete**:

| Current Path | Decision | Rationale |
|--------------|----------|-----------|
| `indexing_CONTINUOUS_RUN_*.log` (942 files) | DELETE | Obsolete, regenerable, bloat repository |
| `evaluation.log` | MOVE to `logs/` or DELETE | If obsolete: delete; if needed: move |
| `cleanup_log.txt` | DELETE | One-time cleanup, not needed |

**Total**: 942+ files (DELETE recommended)

---

### Category 6: Test Artifacts (MOVE TO tests/artifacts/)

**Definition**: Generated files from test execution, coverage reports, and evaluation results.

**Rules**:
- All test-generated content goes to `tests/artifacts/`
- Add tests/artifacts/ to .gitignore
- Keep directory structure for organization

**Destination Directory Structure**:
```
tests/
├── artifacts/                      # .gitignored directory
│   ├── coverage_html/             # HTML coverage reports
│   ├── htmlcov/                   # Alternative coverage HTML
│   ├── results/                   # Test result files
│   └── ragas/                     # RAGAS evaluation results
├── contract/                       # Contract tests (existing)
├── integration/                    # Integration tests (existing)
└── unit/                           # Unit tests (existing)
```

**Files to Move**:

| Current Path | New Path | .gitignore Entry |
|--------------|----------|------------------|
| `htmlcov/` | `tests/artifacts/htmlcov/` | `tests/artifacts/` |
| `coverage_html/` | `tests/artifacts/coverage_html/` | `tests/artifacts/` |
| `.coverage` | DELETE (regenerated) | `.coverage` (already ignored) |
| `coverage.json` | DELETE (regenerated) | `coverage.json` |
| `comprehensive_ragas_results_*/` | `tests/artifacts/ragas/` or archive/ | `tests/artifacts/` |

**Total**: 5+ directories, multiple generated files

---

### Category 7: Archived Content (MOVE TO archive/)

**Definition**: Old evaluation results, deprecated scripts, and historical artifacts that may be useful for reference but are no longer actively used.

**Rules**:
- Keep most recent 5 evaluation runs accessible
- Move older content to archive/
- Add archive/ to .gitignore (or keep in git for history)

**Destination Directory Structure**:
```
archive/
├── eval_results/                   # Old evaluation results
│   ├── comprehensive_ragas_results_20250619_*/
│   ├── comprehensive_ragas_results_20250620_*/
│   └── ... (60+ older result directories)
└── deprecated_scripts/             # Old scripts no longer used
    └── ... (if any exist)
```

**Files to Archive**:

| Current Path | New Path | Keep in Git? |
|--------------|----------|--------------|
| `eval_results/comprehensive_ragas_results_*` (old) | `archive/eval_results/` | No - add to .gitignore |
| `future_tests_not_ready/` | `archive/future_tests_not_ready/` | Decision during implementation |

**Total**: 60+ directories

---

### Category 8: Temporary/Generated Files (DELETE)

**Definition**: Files that should never be in version control - generated artifacts, OS-specific files, and IDE configurations.

**Rules**:
- DELETE immediately
- Ensure .gitignore prevents future commits
- No archive needed (regenerable)

**Files to Delete**:

| File | Type | Regenerable? | .gitignore Entry |
|------|------|--------------|------------------|
| `.DS_Store` | macOS metadata | Yes | `.DS_Store` |
| `.coverage` | Coverage data | Yes | `.coverage` |
| `coverage.json` | Coverage JSON | Yes | `coverage.json` |
| `*.pyc` | Python bytecode | Yes | `*.pyc` |
| `__pycache__/` | Python cache | Yes | `__pycache__/` |

**Total**: Varies (found during implementation)

---

## Validation Rules

### Rule 1: Essential Root Files Limit
- **Constraint**: Maximum 15 files in repository root (excluding directories)
- **Validation**: `ls -1 | wc -l` ≤ 15
- **Current State**: 991 files
- **Target State**: ≤15 files

### Rule 2: Configuration Tool Compatibility
- **Constraint**: All tools (flake8, pytest, coverage, pre-commit) must find their configs
- **Validation**: Run full test suite and linting
- **Test Commands**:
  ```bash
  flake8 .
  pytest --cov=iris_rag --cov-report=html
  pre-commit run --all-files
  ```

### Rule 3: Docker Compose Functionality
- **Constraint**: All docker-compose commands must work
- **Validation**: Test all 7 compose file variants
- **Test Commands**:
  ```bash
  docker-compose up -d && docker-compose down
  docker-compose -f config/docker/docker-compose.licensed.yml up -d && docker-compose down
  # ... (test all 6 variants)
  ```

### Rule 4: Makefile Target Preservation
- **Constraint**: All Makefile targets must execute successfully
- **Validation**: Test critical make targets
- **Test Commands**:
  ```bash
  make setup-env
  make install
  make test
  make docker-up && make docker-down
  ```

### Rule 5: CI/CD Pipeline Success
- **Constraint**: All CI/CD workflows must pass
- **Validation**: Run .gitlab-ci.yml and .github/workflows/*
- **Test**: Commit to branch and verify CI passes

---

## Migration Sequence

### Phase 1: Low-Risk Moves (No Tool Impact)
1. Move log files (942 indexing logs) to `logs/indexing/` or DELETE
2. Move documentation files to `docs/`
3. Move old evaluation results to `archive/eval_results/`
4. Delete temporary files (.DS_Store, .coverage, coverage.json)

**Validation**: `git status` shows expected moves, no tool errors

---

### Phase 2: Medium-Risk Moves (Makefile Updates)
1. Move shell scripts to `scripts/` with subdirectories
2. Update Makefile targets to reference new paths
3. Test all Makefile targets

**Validation**: `make test && make docker-up && make docker-down`

---

### Phase 3: High-Risk Moves (Tool Configuration)
1. Move `.flake8` and `.coveragerc` to `config/`
2. Move docker-compose variants to `config/docker/`
3. Update Makefile docker targets
4. Test all tools and CI/CD

**Validation**: Full test suite + linting + docker-compose + CI

---

## Rollback Plan

If any validation fails:

```bash
# Phase 1 rollback (low risk - simple git reset)
git reset --hard HEAD

# Phase 2 rollback (medium risk - revert specific commits)
git revert <commit-hash>

# Phase 3 rollback (high risk - manual restoration)
git checkout HEAD -- config/ Makefile
mv config/.flake8 .
mv config/.coveragerc .
# ... (restore other critical files)
```

---

## Success Criteria

✅ Repository root contains ≤15 files (excluding directories)
✅ All CI/CD workflows pass
✅ All Makefile targets execute successfully
✅ All docker-compose variants work
✅ All development tools find their configurations
✅ GitHub repository landing page shows clean structure
✅ Documentation updated with new paths

---

## Related Documents

- [spec.md](./spec.md) - Feature specification with functional requirements
- [research.md](./research.md) - Research findings on best practices
- [plan.md](./plan.md) - Implementation plan
- [quickstart.md](./quickstart.md) - Step-by-step execution guide
