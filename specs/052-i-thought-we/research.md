# Research: Root Directory Cleanup

**Feature**: 052-i-thought-we
**Date**: 2025-11-07
**Status**: Complete

## Overview

This document consolidates research on best practices for organizing repository root directories, handling configuration file relocation, and maintaining backward compatibility during file reorganization.

---

## Research Task 1: Repository Organization Best Practices

**Question**: What are industry best practices for organizing repository root directories?

**Findings**:

### Essential Root Files (Keep in Root)
- **README.md** - Project overview and quick start guide
- **LICENSE** - Legal license information
- **pyproject.toml / setup.py / package.json** - Package configuration
- **Makefile** - Common build commands (for projects using make)
- **docker-compose.yml** - Primary Docker orchestration (if project uses Docker)
- **.gitignore** - Version control ignore rules
- **.env.example** - Environment variable template (never .env with secrets)

### Common Subdirectory Patterns
- **docs/** - Documentation files (guides, API references, architecture)
- **scripts/** - Shell scripts, utilities, automation tools
- **config/** - Configuration files (.flake8, .coveragerc, docker variants)
- **tests/** - Test suites with subdirectories (unit/, integration/, contract/)
- **logs/** - Log files (should be .gitignored, not committed)
- **archive/** - Deprecated or old content

### Files to Remove Entirely
- **Generated artifacts** - .coverage, coverage.json, htmlcov/, *.pyc
- **Temporary files** - .DS_Store, Thumbs.db, *.swp, *.tmp
- **Build artifacts** - dist/, build/, *.egg-info
- **Editor configs** - .idea/, .vscode/ (use .gitignore)
- **Old log files** - Unless needed for historical analysis

**Decision**: Follow the essential + subdirectory pattern, keeping only ~10-15 files in root.

**Rationale**:
- Improves repository browsing experience on GitHub
- Makes project structure clear to new contributors
- Reduces cognitive load when finding files
- Aligns with open-source best practices (React, Django, FastAPI projects)

**Alternatives Considered**:
- Flat structure (keep all 991 files) - Rejected: unmaintainable clutter
- Aggressive nesting (move everything to subdirs) - Rejected: breaks tool expectations

---

## Research Task 2: Configuration File Relocation

**Question**: How do development tools (flake8, pytest, coverage) find configuration files when they're not in root?

**Findings**:

### Tool Search Behavior
- **flake8**: Searches for `.flake8`, `setup.cfg`, `tox.ini` in current directory → parent directories
- **pytest**: Searches for `pytest.ini`, `pyproject.toml`, `tox.ini` in current directory → parent directories
- **coverage**: Searches for `.coveragerc`, `setup.cfg`, `pyproject.toml` in current directory → parent directories
- **pre-commit**: Requires `.pre-commit-config.yaml` in repository root

### Recommended Locations

| Tool | Current Location | Recommended Location | Rationale |
|------|-----------------|---------------------|-----------|
| flake8 | `.flake8` (root) | `config/.flake8` | Supports subdirectory search |
| coverage | `.coveragerc` (root) | `config/.coveragerc` | Supports subdirectory search |
| pytest | `pyproject.toml` (root) | Keep in root | Tool configuration in package file |
| pre-commit | `.pre-commit-config.yaml` (root) | Keep in root | **MUST be in root** |
| docker-compose | `docker-compose.*.yml` (root) | `config/docker/` | Requires `-f` flag update |

### Integration Approach
1. **Option 1: Keep in Root** - Least disruptive, tool finds configs automatically
2. **Option 2: Move to config/ + Update Paths** - Cleaner root, requires CI/CD updates
3. **Option 3: Consolidate into pyproject.toml** - Modern approach, one config file

**Decision**:
- Move `.flake8` and `.coveragerc` to `config/`
- Keep `.pre-commit-config.yaml` in root (required by pre-commit)
- Keep tool configs in `pyproject.toml` where possible
- Move docker-compose variants to `config/docker/`

**Rationale**:
- Tools will find configs via parent directory search
- Reduces root clutter without breaking functionality
- pyproject.toml is already configured for pytest, black, isort
- Docker compose variants are rarely used individually

**Alternatives Considered**:
- Keep all configs in root - Rejected: defeats cleanup purpose
- Move all to pyproject.toml - Rejected: some tools don't support it (.flake8)

---

## Research Task 3: Docker Compose File Consolidation

**Question**: How to consolidate 7 docker-compose variants without breaking existing workflows?

**Findings**:

### Current Docker Compose Files (Root Directory)
1. `docker-compose.yml` - Primary setup (IRIS + Redis + API + Streamlit)
2. `docker-compose.api.yml` - API service only
3. `docker-compose.full.yml` - Full development stack
4. `docker-compose.licensed.yml` - Licensed IRIS version (port 21972)
5. `docker-compose.mcp.yml` - Model Context Protocol server
6. `docker-compose.test.yml` - Testing environment
7. `docker-compose.iris-only.yml` - IRIS database only

### Usage Patterns
```bash
# Current (files in root):
docker-compose up -d                               # Uses docker-compose.yml
docker-compose -f docker-compose.licensed.yml up -d

# After moving to config/docker/:
docker-compose up -d                               # Still uses docker-compose.yml (root)
docker-compose -f config/docker/docker-compose.licensed.yml up -d
```

### Consolidation Strategy

**Option 1: Move All to config/docker/** (Rejected)
- Breaks default `docker-compose up -d` command
- Requires `-f` flag for every invocation
- High disruption to existing workflows

**Option 2: Keep Primary, Move Variants** (Selected)
- Keep `docker-compose.yml` in root (default file)
- Move 6 variants to `config/docker/`
- Update Makefile targets to use `-f config/docker/...`

**Option 3: Merge All into One File with Profiles** (Considered)
- Use Docker Compose profiles (--profile flag)
- Single file reduces clutter
- Rejected: Complex, high migration effort

**Decision**: Keep `docker-compose.yml` in root, move 6 variants to `config/docker/`

**Rationale**:
- Preserves default docker-compose behavior
- Reduces root clutter from 7 files to 1
- Minimal disruption (only variant usage requires update)
- Makefile updates centralize path changes

**Implementation**:
```bash
# Update Makefile
docker-up-licensed:
	docker-compose -f config/docker/docker-compose.licensed.yml up -d

docker-up-mcp:
	docker-compose -f config/docker/docker-compose.mcp.yml up -d

# Update documentation
# CLAUDE.md, README.md, docs/DEVELOPMENT.md
```

---

## Research Task 4: Backward Compatibility Validation

**Question**: How to ensure CI/CD and development workflows continue working after reorganization?

**Findings**:

### Critical Paths to Test

#### CI/CD Workflows (.github/workflows/, .gitlab-ci.yml)
- Docker compose file references
- Configuration file paths (.flake8, .coveragerc)
- Script invocations (scripts/*.sh)
- Test artifact paths (coverage reports, htmlcov/)

#### Makefile Targets
- `make setup-env` - Environment activation
- `make install` - Dependency installation
- `make test` - Test suite execution
- `make docker-up` - Docker compose up
- `make load-data` - Data loading scripts

#### Development Commands
- `pytest` - Must find pytest.ini or pyproject.toml
- `flake8 .` - Must find .flake8 config
- `coverage run` - Must find .coveragerc
- `pre-commit run` - Must find .pre-commit-config.yaml
- `docker-compose up -d` - Must find docker-compose.yml

### Validation Strategy

**Phase 1: Pre-Move Validation**
```bash
# Capture current state
make test > before_cleanup.log
git status > before_git_status.txt
docker-compose config > before_docker_config.yml
```

**Phase 2: Incremental Migration**
1. Move log files first (low risk)
2. Move documentation files (low risk)
3. Move scripts with Makefile updates (medium risk)
4. Move configuration files (high risk - test thoroughly)
5. Move docker-compose variants last (high risk - test all make targets)

**Phase 3: Post-Move Validation**
```bash
# Verify all workflows pass
make test
make docker-up && make docker-down
pytest --cov=iris_rag --cov-report=html
flake8 .
pre-commit run --all-files

# Compare outputs
diff before_cleanup.log after_cleanup.log
diff before_docker_config.yml after_docker_config.yml
```

**Decision**: Incremental migration with validation at each step

**Rationale**:
- Reduces blast radius if something breaks
- Allows rollback at any point
- Validates assumptions before full migration
- Git commits provide checkpoint recovery

**Alternatives Considered**:
- Big bang migration (move everything at once) - Rejected: high risk
- Gradual migration over multiple PRs - Rejected: too slow for simple task

---

## Research Task 5: Handling Old Evaluation Results

**Question**: What to do with eval_results/ directory containing 65+ subdirectories?

**Findings**:

### Current State
```bash
eval_results/
├── comprehensive_ragas_results_20250619_165432/
├── comprehensive_ragas_results_20250620_091234/
├── comprehensive_ragas_results_20250621_143056/
├── ... (62+ more timestamped directories)
├── evaluation.log
└── cleanup_log.txt
```

### Storage Analysis
- **Size**: Unknown (need to check disk usage)
- **Age**: Timestamped directories suggest historical data
- **Usage**: Likely used for performance tracking and regression testing
- **Reference**: May be referenced in documentation or scripts

### Options

**Option 1: Delete Entirely**
- Pros: Maximum cleanup, removes 65+ directories
- Cons: Loses historical data, may break documentation references
- Risk: High

**Option 2: Archive to archive/eval_results/**
- Pros: Preserves history, reduces root clutter
- Cons: Still takes disk space
- Risk: Low

**Option 3: Keep Most Recent, Delete Old**
- Pros: Retains recent data for reference
- Cons: Requires determining what "recent" means
- Risk: Medium

**Option 4: Move to docs/benchmarks/**
- Pros: Keeps results accessible, proper categorization
- Cons: Clutters docs/ directory
- Risk: Medium

**Decision**: Archive old results to `archive/eval_results/`, keep most recent (last 5 runs) in root or docs/

**Rationale**:
- Preserves historical data for regression analysis
- Removes bulk of clutter from root
- Keeps recent results accessible for current development
- Can be fully deleted later if disk space is concern

**Implementation**:
```bash
# Keep 5 most recent results
mkdir -p archive/eval_results
ls -t eval_results/comprehensive_ragas_results_* | tail -n +6 | xargs -I {} mv eval_results/{} archive/eval_results/

# Or move all to archive and keep structure
mv eval_results archive/eval_results
mkdir eval_results  # Fresh directory for new results
```

---

## Summary of Decisions

| Area | Decision | Impact |
|------|----------|--------|
| **Root Files** | Keep only 10-15 essential files | 991 → <20 files |
| **Configuration** | Move `.flake8`, `.coveragerc` to `config/` | Tools will find via parent search |
| **Docker Compose** | Keep primary in root, move 6 variants to `config/docker/` | Preserves default behavior |
| **Scripts** | Move all to `scripts/` with subdirs | Centralized location |
| **Logs** | Move 942 logs to `logs/indexing/` | .gitignore ensures future logs go here |
| **Documentation** | Move to `docs/`, keep README.md in root | Clear documentation structure |
| **Test Artifacts** | Move to `tests/artifacts/` | Proper test organization |
| **Eval Results** | Archive old to `archive/`, keep recent 5 | Preserves history, reduces clutter |
| **Validation** | Incremental migration with validation | Low-risk deployment |

---

## Next Steps (Phase 1)

1. Create `data-model.md` - Define file categories and organization rules
2. Create `contracts/` - Validation tests for CI/CD workflows
3. Create `quickstart.md` - Guide for executing the reorganization
4. Re-evaluate Constitution Check after design complete
