# Quickstart: Root Directory Cleanup

**Feature**: 052-i-thought-we
**Branch**: `052-i-thought-we`
**Est. Time**: 30-45 minutes

## Overview

This guide provides step-by-step instructions for executing the root directory cleanup, reducing 991 files to <20 while maintaining all CI/CD and development workflow functionality.

---

## Prerequisites

✅ Working directory: `/Users/tdyar/ws/rag-templates`
✅ Branch: `052-i-thought-we` (already created)
✅ Git status clean: All Feature 051 changes committed
✅ Backup: Current state backed up (optional but recommended)

**Optional Backup**:
```bash
cd /Users/tdyar/ws/rag-templates
git stash  # If any uncommitted changes
git tag before-cleanup-052  # Create recovery point
```

---

## Execution Steps

### Phase 1: Low-Risk Moves (10 minutes)

**Step 1.1: Delete obsolete log files**

```bash
cd /Users/tdyar/ws/rag-templates

# Count log files before deletion
ls indexing_CONTINUOUS_RUN_*.log 2>/dev/null | wc -l  # Should show ~942

# Delete all indexing logs (obsolete, regenerable)
rm -f indexing_CONTINUOUS_RUN_*.log

# Delete other obsolete logs
rm -f evaluation.log cleanup_log.txt

# Verify deletion
ls *.log 2>/dev/null  # Should show no .log files
```

**Step 1.2: Move documentation files to docs/**

```bash
# Create docs/ directory if it doesn't exist
mkdir -p docs

# Move documentation files
mv CONTRIBUTING.md docs/
mv USER_GUIDE.md docs/ 2>/dev/null || true  # If exists
mv CLAUDE.md docs/ 2>/dev/null || true  # If exists

# Verify README.md remains in root
ls -la README.md
```

**Step 1.3: Archive old evaluation results**

```bash
# Create archive directory
mkdir -p archive/eval_results

# Move old evaluation result directories
# Keep 5 most recent, archive the rest
if [ -d eval_results ]; then
    ls -t eval_results/comprehensive_ragas_results_* 2>/dev/null | tail -n +6 | while read dir; do
        mv "$dir" archive/eval_results/
    done
fi

# Or move all to archive if preferred:
# mv eval_results archive/eval_results 2>/dev/null || true
# mkdir eval_results  # Fresh directory for new results
```

**Step 1.4: Delete temporary files**

```bash
# Delete macOS metadata
find . -name ".DS_Store" -delete

# Delete coverage artifacts (regenerable)
rm -f .coverage coverage.json
rm -rf htmlcov coverage_html

# Delete comprehensive RAGAS results in root (if not already moved)
rm -rf comprehensive_ragas_results_* 2>/dev/null || true
```

**Validation 1.1: Verify low-risk changes**

```bash
# Check git status
git status

# Should show:
# - deleted: indexing_CONTINUOUS_RUN_*.log (942 files)
# - deleted: evaluation.log, cleanup_log.txt
# - renamed: CONTRIBUTING.md -> docs/CONTRIBUTING.md
# - deleted: .DS_Store, .coverage, coverage.json, htmlcov/, coverage_html/

# Stage changes
git add -A
git commit -m "chore: cleanup root directory - remove logs, temp files, move docs (Phase 1)"
```

---

### Phase 2: Medium-Risk Moves (15 minutes)

**Step 2.1: Move shell scripts to scripts/**

```bash
# Create scripts subdirectories
mkdir -p scripts/setup
mkdir -p scripts/upload
mkdir -p scripts/docker
mkdir -p scripts/ci

# Move scripts to appropriate locations
mv activate_env.sh scripts/ 2>/dev/null || true
mv docker-entrypoint-mcp.sh scripts/docker/ 2>/dev/null || true
mv upload_to_pypi.sh scripts/upload/ 2>/dev/null || true
mv setup_iris_env.sh scripts/setup/ 2>/dev/null || true

# Find any remaining .sh files in root
ls *.sh 2>/dev/null
# If any found, move to scripts/ or appropriate subdirectory
```

**Step 2.2: Update Makefile references**

Open `Makefile` and update script paths:

```makefile
# Before:
activate-env:
	./activate_env.sh

# After:
activate-env:
	./scripts/activate_env.sh
```

Update all script references in Makefile:
- `activate_env.sh` → `scripts/activate_env.sh`
- `setup_iris_env.sh` → `scripts/setup/setup_iris_env.sh`
- Any other script references

**Validation 2.1: Test Makefile targets**

```bash
# Test critical make targets
make setup-env || echo "OK if fails (may need adjustment)"
make install || echo "OK"

# If any failures, review Makefile updates
```

**Step 2.3: Commit Phase 2 changes**

```bash
git add -A
git commit -m "chore: move scripts to scripts/ directory and update Makefile (Phase 2)"
```

---

### Phase 3: High-Risk Moves (20 minutes)

**Step 3.1: Move configuration files**

```bash
# Create config directory structure
mkdir -p config/docker

# Move configuration files
mv .flake8 config/ 2>/dev/null || true
mv .coveragerc config/ 2>/dev/null || true
mv .coveragerc.ci config/ 2>/dev/null || true

# Move docker-compose variants (keep primary in root)
mv docker-compose.api.yml config/docker/ 2>/dev/null || true
mv docker-compose.full.yml config/docker/ 2>/dev/null || true
mv docker-compose.licensed.yml config/docker/ 2>/dev/null || true
mv docker-compose.mcp.yml config/docker/ 2>/dev/null || true
mv docker-compose.test.yml config/docker/ 2>/dev/null || true
mv docker-compose.iris-only.yml config/docker/ 2>/dev/null || true

# Verify primary docker-compose.yml remains in root
ls -la docker-compose.yml
```

**Step 3.2: Update Makefile docker targets**

Open `Makefile` and update docker-compose variant paths:

```makefile
# Before:
docker-up-licensed:
	docker-compose -f docker-compose.licensed.yml up -d

# After:
docker-up-licensed:
	docker-compose -f config/docker/docker-compose.licensed.yml up -d
```

Update all docker-compose variant references:
- `docker-compose.api.yml` → `config/docker/docker-compose.api.yml`
- `docker-compose.full.yml` → `config/docker/docker-compose.full.yml`
- `docker-compose.licensed.yml` → `config/docker/docker-compose.licensed.yml`
- `docker-compose.mcp.yml` → `config/docker/docker-compose.mcp.yml`
- `docker-compose.test.yml` → `config/docker/docker-compose.test.yml`
- `docker-compose.iris-only.yml` → `config/docker/docker-compose.iris-only.yml`

**Step 3.3: Update docker-entrypoint references**

If `docker-entrypoint-mcp.sh` was referenced in docker-compose files:

```bash
# Find docker-compose files referencing entrypoint
grep -r "docker-entrypoint-mcp.sh" config/docker/

# Update paths:
# entrypoint: ./docker-entrypoint-mcp.sh
# → entrypoint: ./scripts/docker/docker-entrypoint-mcp.sh
```

**Step 3.4: Move test artifacts**

```bash
# Create test artifacts directory
mkdir -p tests/artifacts

# If any artifacts directories remain in root, move them
mv htmlcov tests/artifacts/ 2>/dev/null || true
mv coverage_html tests/artifacts/ 2>/dev/null || true

# Move evaluation results if they're in root
if [ -d eval_results ]; then
    mv eval_results tests/artifacts/ragas 2>/dev/null || true
fi
```

**Step 3.5: Update .gitignore**

Add/verify entries in `.gitignore`:

```bash
# Add if not already present
cat >> .gitignore << 'EOF'

# Logs
logs/
*.log

# Test artifacts
tests/artifacts/
htmlcov/
coverage_html/
.coverage
coverage.json

# Archive
archive/

# Temporary files
.DS_Store
Thumbs.db
EOF
```

**Validation 3.1: Test all tools and workflows**

```bash
# Test linting (flake8 should find config/.flake8 via parent search)
flake8 .

# Test coverage (coverage should find config/.coveragerc via parent search)
pytest --cov=iris_rag --cov-report=html

# Test pre-commit (must find .pre-commit-config.yaml in root)
pre-commit run --all-files

# Test primary docker-compose
docker-compose up -d
docker-compose down

# Test docker-compose variant (licensed IRIS)
docker-compose -f config/docker/docker-compose.licensed.yml up -d
docker-compose -f config/docker/docker-compose.licensed.yml down

# Test Makefile targets
make test
```

**Step 3.6: Commit Phase 3 changes**

```bash
git add -A
git commit -m "chore: move config files to config/, update tool paths (Phase 3)"
```

---

### Phase 4: Validation & Documentation (5 minutes)

**Step 4.1: Run contract tests**

```bash
# Run contract tests to verify cleanup success
pytest specs/052-i-thought-we/contracts/test_root_directory_contract.py -v

# All tests should pass
# If any fail, review and fix before proceeding
```

**Step 4.2: Update documentation**

Update `README.md` to reference new locations:

```markdown
## Quick Start

...

### 2. Start IRIS Database

```bash
# Start IRIS with Docker Compose
docker-compose up -d

# Or use licensed IRIS variant
docker-compose -f config/docker/docker-compose.licensed.yml up -d
```
```

Update `docs/CLAUDE.md` (formerly `CLAUDE.md`):

```markdown
## Development Commands

### Environment Setup
```bash
# Initial setup
make setup-env    # Now references scripts/activate_env.sh
...
```

**Step 4.3: Final validation**

```bash
# Count files in root
ls -1 | wc -l  # Should be ≤15

# Verify structure
tree -L 2 -d
# Should show:
# .
# ├── archive
# │   └── eval_results
# ├── config
# │   └── docker
# ├── docs
# ├── logs (if kept)
# ├── scripts
# │   ├── ci
# │   ├── docker
# │   ├── setup
# │   └── upload
# ├── tests
# │   └── artifacts
# └── iris_rag (unchanged)

# Final commit
git add -A
git commit -m "docs: update documentation for new directory structure (Phase 4)"
```

---

## Rollback Procedure

If anything breaks during cleanup:

```bash
# Option 1: Rollback to before-cleanup tag
git reset --hard before-cleanup-052

# Option 2: Revert specific commits
git log --oneline  # Find commit hash
git revert <commit-hash>

# Option 3: Restore specific files
git checkout HEAD~1 -- Makefile config/ docs/
```

---

## Success Criteria Checklist

After completing all phases, verify:

- ✅ Root directory has ≤15 files: `ls -1 | wc -l`
- ✅ Contract tests pass: `pytest specs/052-i-thought-we/contracts/ -v`
- ✅ Linting works: `flake8 .`
- ✅ Coverage works: `pytest --cov=iris_rag --cov-report=html`
- ✅ Pre-commit works: `pre-commit run --all-files`
- ✅ Docker compose works: `docker-compose up -d && docker-compose down`
- ✅ Makefile targets work: `make test && make docker-up`
- ✅ Documentation updated: README.md, docs/CLAUDE.md reference new paths
- ✅ Git status clean: `git status`

---

## Next Steps

After successful cleanup on `052-i-thought-we` branch:

1. **Push to internal repository**:
   ```bash
   git push origin 052-i-thought-we
   ```

2. **Create merge request** to `main` branch

3. **Sync to sanitized repository** (after merge):
   ```bash
   cd /Users/tdyar/ws/rag-templates
   git checkout main
   git pull origin main
   ./scripts/sync_to_sanitized.sh

   cd /Users/tdyar/ws/rag-templates-sanitized
   git pull github main --rebase
   git add -A
   git commit -m "chore: sync root directory cleanup from internal repository"
   git push github main
   ```

4. **No PyPI release needed** (non-code changes)

---

## Troubleshooting

### Issue: flake8 not finding config

```bash
# Verify config exists
ls -la config/.flake8

# Test in subdirectory (should find via parent search)
cd iris_rag
flake8 .
cd ..

# If still fails, check flake8 version supports parent directory search
flake8 --version
```

### Issue: Docker compose variant not found

```bash
# Verify path
ls -la config/docker/docker-compose.licensed.yml

# Use absolute path if needed
docker-compose -f $(pwd)/config/docker/docker-compose.licensed.yml up -d
```

### Issue: Makefile target fails

```bash
# Check Makefile syntax
make -n <target>  # Dry run

# Verify script paths
ls -la scripts/activate_env.sh
ls -la scripts/setup/setup_iris_env.sh
```

### Issue: Contract tests fail

```bash
# Run specific failing test
pytest specs/052-i-thought-we/contracts/test_root_directory_contract.py::TestRootDirectoryContract::test_root_file_count_within_limit -v

# Review error message for specific violation
# Fix and re-run
```

---

## Estimated Timeline

| Phase | Tasks | Time | Risk |
|-------|-------|------|------|
| **Phase 1** | Delete logs, move docs, delete temp files | 10 min | Low |
| **Phase 2** | Move scripts, update Makefile | 15 min | Medium |
| **Phase 3** | Move configs, update docker paths | 20 min | High |
| **Phase 4** | Validation, documentation | 5 min | Low |
| **Total** | | **50 min** | |

**Recommendation**: Allocate 1 hour for execution + validation.

---

## Related Documents

- [spec.md](./spec.md) - Feature specification
- [research.md](./research.md) - Research findings
- [data-model.md](./data-model.md) - File categories and organization
- [plan.md](./plan.md) - Implementation plan
- [contracts/test_root_directory_contract.py](./contracts/test_root_directory_contract.py) - Validation tests
