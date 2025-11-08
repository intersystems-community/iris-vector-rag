<!--
Sync Impact Report:
- Version change: 1.7.0 → 1.7.1 (PATCH)
- Version bump rationale: Enhanced Step 4 (PyPI Publishing) with detailed procedures
- List of modified principles: Principle VIII (Git & Release Workflow)
- Modified sections:
  * Step 4: PyPI Publishing - Added comprehensive publishing procedures
  * When to publish (semantic versioning guidance)
  * Version bumping procedure (files to update)
  * Build and validation steps (clean, build, test locally)
  * Upload procedure (secure token, verification)
  * Post-upload verification (install from PyPI, test imports)
- Removed sections: N/A
- Templates requiring updates:
  ✅ No template updates needed (workflow-specific, not template-affecting)
- Follow-up TODOs: None
-->

# RAG-Templates Constitution

## Core Principles

### I. Framework-First Architecture

Every component MUST be designed as a reusable framework element. Pipeline implementations MUST extend the abstract RAGPipeline base class. Components MUST be independently testable and documented with clear interfaces. No application-specific logic in framework components. All components MUST expose functionality via CLI interface (Make targets or direct CLI commands).

**Rationale**: Ensures the framework can be consumed by multiple applications while maintaining clean separation of concerns, reusability, and operational accessibility.

### II. Pipeline Validation & Requirements

All pipelines MUST implement automated requirement validation and setup orchestration. Pipeline creation MUST fail fast with clear error messages when prerequisites are missing. Setup procedures MUST be idempotent and recoverable.

**Rationale**: Enterprise environments require reliable, self-validating components that can detect and resolve configuration issues automatically.

### III. Test-Driven Development with Live Database Validation (NON-NEGOTIABLE)

Tests MUST be written before implementation. All contract tests MUST fail initially, then pass after implementation. Unit, integration, and end-to-end tests are mandatory for all pipeline implementations.

**Integration Testing Absolute Requirement (CRITICAL)**:

When developing ANY feature that integrates with an external system (IRIS database, APIs, vector stores, etc.), integration tests with the ACTUAL system are MANDATORY and CANNOT be skipped. This principle is non-negotiable:

1. **NO MOCKING OF INTEGRATION POINTS FOR INTEGRATION TESTS**: Integration tests MUST use the real system
2. **IRIS-Specific Rule**: For ANY feature involving IRIS (embeddings, vector operations, storage, schema), you MUST create and run integration tests against a real IRIS instance BEFORE claiming completion
3. **Test Container Requirement**: Use `iris-devtester` package with `IRISContainer` for repeatable, isolated IRIS testing
4. **Test Coverage**: Integration tests MUST verify the complete integration path, not just Python code in isolation

**Example Violation**: Creating "Feature 051: IRIS EMBEDDING Support" without actual IRIS integration tests is UNACCEPTABLE. Contract tests alone DO NOT prove the feature works.

**Correct Approach**:
- Use `from iris_devtester import IRISContainer`
- Create fixtures: `@pytest.fixture def iris_container(): with IRISContainer.community() as container: yield container`
- Test actual IRIS operations: SQL DDL, data insertion, vector queries, Python callbacks from IRIS
- Verify performance claims with real workloads

**IRIS Database Requirement**: All RAG framework tests that involve data storage, vector operations, schema management, or pipeline operations MUST execute against a running IRIS database instance. Tests MUST use either:
1. **Framework-managed Docker IRIS** (preferred): Use `iris-devtester` with `IRISContainer.community()` for automatic container management
2. **External IRIS instance**: When configured via environment variables

**IRIS Docker Management Procedures**:
- **Required IRIS Image**: ALL Docker Compose files MUST use `intersystemsdc/iris-community:2025.3.0EHAT.127.0-linux-arm64v8` (note: NOT iris-lockeddown)
- **Licensed IRIS Priority**: Use `intersystemsdc/iris-community:2025.3.0EHAT.127.0-linux-arm64v8` or similar licensed images when available
- **Standardized Port Mapping**: ALL Docker Compose files MUST follow the framework's port mapping strategy:
  * **Container ports**: Always use IRIS standard ports (1972 SuperServer, 52773 Management Portal)
  * **Host port ranges**:
    - Default IRIS: `11972:1972` and `152773:52773` (docker-compose.yml)
    - Licensed IRIS: `21972:1972` and `252773:52773` (docker-compose.licensed.yml)
    - Reserved: `1972:1972` and `52773:52773` (for existing IRIS installations)
  * **Rationale**: Predictable ports avoid conflicts, enable easy configuration, support multiple IRIS instances
- **Port Discovery**: Use the framework's `common/iris_port_discovery.py` utility to locate running IRIS instances across mapped ports
- **Connection Management**: Use `common/iris_connection_manager.py` for standardized IRIS connections with automatic fallback patterns
- **Health Validation**: Always run `python evaluation_framework/test_iris_connectivity.py` before database-dependent testing

**Test Categories with IRIS Requirements**:
- `@pytest.mark.requires_database`: MUST connect to live IRIS
- `@pytest.mark.integration`: MUST use IRIS for data operations
- `@pytest.mark.e2e`: MUST use complete IRIS + vector workflow
- `@pytest.mark.clean_iris`: MUST start from fresh/empty IRIS instance to validate complete setup workflow
- Unit tests MAY mock IRIS for isolated component testing

**Clean IRIS Testing Requirement (NON-NEGOTIABLE)**: All RAG pipeline implementations MUST provide test variants that start from a completely clean/fresh IRIS database instance. These tests validate the complete setup orchestration including schema creation, data ingestion, and pipeline initialization. Clean IRIS tests are essential for:
- Validating auto-setup and orchestration capabilities
- Testing schema migration and upgrade procedures
- Ensuring reproducible deployment scenarios
- Verifying framework self-sufficiency in new environments

**Database Health Validation**: All test suites MUST verify IRIS health before execution using established connectivity tests from `evaluation_framework/test_iris_connectivity.py`.

**Container Management Best Practices**:
- **Avoid Port Conflicts**: Never force-stop existing IRIS containers; use dynamic port allocation instead
- **Image Selection**: Check `docker images | grep iris` to identify available licensed IRIS images before attempting pulls
- **Configuration Updates**: Modify docker-compose files to use available images and dynamic ports rather than creating new containers
- **Environment Variables**: Set `IRIS_HOST=localhost`, `IRIS_PORT=<discovered_port>`, `IRIS_USERNAME=_SYSTEM`, `IRIS_PASSWORD=SYS` for testing

Performance tests required for enterprise scale scenarios (10K+ documents) MUST execute against IRIS with actual vector operations and RAGAS evaluation.

**Rationale**: RAG systems are fundamentally dependent on IRIS database for document storage, vector embeddings, and search operations. Testing without live database connections provides false validation and cannot detect real-world integration failures, performance issues, or data consistency problems.

### IV. Performance & Enterprise Scale

All pipelines MUST support incremental indexing and concurrent operations. Vector operations MUST be optimized for IRIS database capabilities. Memory usage MUST be monitored and bounded. Performance benchmarks required for 1K, 10K document scenarios with RAGAS evaluation.

**Rationale**: Enterprise RAG applications must handle large document corpora efficiently without degrading user experience.

### V. Production Readiness Standards

All components MUST include structured logging, error handling, and health checks. Configuration MUST be externalized via environment variables and YAML. Docker deployment MUST be supported with comprehensive monitoring capabilities.

**Rationale**: Production RAG systems require operational excellence for reliability, debugging, and maintenance.

### VI. Explicit Error Handling (NON-NEGOTIABLE)

Domain errors only: no silent failures, every bug MUST be explicit. All error conditions MUST surface as clear exceptions with actionable messages. No swallowed exceptions or undefined behavior. Failed operations MUST provide specific context about what failed and why.

**Rationale**: RAG systems process critical knowledge; silent failures can lead to incorrect or missing information being returned to users, which is unacceptable in enterprise environments.

### VII. Standardized Database Interfaces

All database interactions MUST use proven, standardized utilities from the framework's SQL and vector helper modules. No ad-hoc database queries or direct IRIS API calls outside established patterns. New database patterns MUST be contributed back to shared utilities after validation.

**Rationale**: IRIS database interactions have complex edge cases and performance considerations. Hard-won fixes and optimizations must be systematized to prevent teams from rediscovering the same issues.

### VIII. Git & Release Workflow (NON-NEGOTIABLE)

All code changes MUST follow the standard git workflow for the public iris-vector-rag repository:

**Step 1: Standard Git Workflow**
```bash
# Stage changes
git add <changed-files>

# Commit with conventional commit message
git commit -m "<type>: <description>"

# Push to GitHub
git push github main
```

**Step 2: PyPI Publishing (when applicable)**

**When to Publish:**
- New features added (MINOR version bump: 0.2.3 → 0.3.0)
- Bug fixes or patches (PATCH version bump: 0.2.3 → 0.2.4)
- Breaking changes (MAJOR version bump: 0.2.3 → 1.0.0)
- Skip if only documentation, tests, or internal changes

**Complete Version Bump Workflow:**

```bash
# 1. Update version in BOTH files (Example: 0.2.3 → 0.2.4)
# Edit these files:
# - pyproject.toml (line 7): version = "0.2.4"
# - iris_rag/__init__.py (line 21): __version__ = "0.2.4"

# 2. Clean previous builds
rm -rf dist/ build/ *.egg-info

# 3. Build source distribution and wheel
python -m build

# 4. Validate distributions
python -m twine check dist/*

# 5. Test local installation
pip install --force-reinstall --no-deps dist/iris_vector_rag-*.whl
python -c "import iris_rag; print(f'iris_rag version: {iris_rag.__version__}'); from iris_rag import create_pipeline; print('✅ Package imports successfully')"

# 6. Commit version bump
git add pyproject.toml iris_rag/__init__.py
git commit -m "chore: bump version to 0.2.4 for [brief description]

[Detailed changelog entry describing what changed in this version]"

# 7. Upload to PyPI (requires ~/.pypirc with token)
python -m twine upload dist/*

# 8. Verify PyPI upload
curl -s https://pypi.org/pypi/iris-vector-rag/json | python3 -c "import sys, json; data=json.load(sys.stdin); print(f\"✅ Latest PyPI version: {data['info']['version']}\"); print(f\"📅 Upload date: {list(data['releases'][data['info']['version']])[0]['upload_time']}\"); print(f\"🔗 URL: https://pypi.org/project/iris-vector-rag/{data['info']['version']}/\")"

# 9. Push version bump commit
git push github main
```

**PyPI Authentication Setup (one-time):**

Create `~/.pypirc` with your PyPI API token:
```ini
[pypi]
username = __token__
password = pypi-AgEIcHlwaS5vcmc...  # Your PyPI API token
```

**Important Notes:**
- Always test package installation before uploading to PyPI
- Version bumps should include comprehensive changelog in commit message
- Verify PyPI upload succeeded before announcing new version

**Repository Configuration:**
- Primary remote: `github` → `https://github.com/intersystems-community/iris-vector-rag.git`
- **Default branch**: `main`

**One-Time GitHub Setup (if not already configured):**

If the GitHub repository default branch is `master` instead of `main`:
1. Go to: https://github.com/intersystems-community/iris-vector-rag/settings/branches
2. Under "Default branch", click the switch icon next to `master`
3. Select `main` from the dropdown
4. Click "Update" and confirm the change
5. (Optional) Delete the old `master` branch protection rules if they exist

This ensures pushes to `main` are immediately visible on GitHub.

**Critical Rules**:
- ALWAYS pull and rebase before pushing to avoid diverged history issues
- ALWAYS test package installation before PyPI upload (`pip install dist/*.whl`)
- PyPI package name is `iris-vector-rag`, Python module name remains `iris_rag`
- Follow conventional commit message format (e.g., `feat:`, `fix:`, `chore:`)

**Rationale**: Proper git workflow and thorough testing before PyPI publication ensures stable releases and prevents breaking changes for users.

## Enterprise Requirements

Production deployments MUST include:

- Vector store persistence and backup procedures
- API rate limiting and authentication
- Observability stack (logging, metrics, tracing)
- Security scanning and vulnerability management
- Multi-environment configuration management (dev, staging, prod)

## Development Standards

**Package Management**: All Python projects MUST use uv for dependency management, virtual environment creation, and package installation. Traditional pip/virtualenv workflows are deprecated in favor of uv's superior performance and reliability.

Code MUST pass linting (black, isort, flake8, mypy) before commits. All public APIs MUST include comprehensive docstrings. Breaking changes MUST follow semantic versioning. Dependencies MUST be pinned and regularly updated for security.

Documentation MUST include quickstart guides, API references, and integration examples. Agent-specific guidance files (CLAUDE.md) MUST be maintained for AI development assistance.

## VIII. Git & Release Workflow (NON-NEGOTIABLE)

All feature development MUST follow standardized git workflows to prevent diverged branches, rejected pushes, and deployment conflicts. The fork-based strategy (private development repo + public fork) requires discipline and clear procedures.

### Repository Structure (MANDATORY)

**Three-Remote Configuration**:
```bash
origin   → https://github.com/isc-tdyar/iris-vector-rag-private.git  (PRIVATE - main development)
fork     → https://github.com/isc-tdyar/iris-vector-rag.git          (PUBLIC - for PRs only)
upstream → https://github.com/intersystems-community/iris-vector-rag.git  (PUBLIC - community repo)
```

**Directory Structure**:
- Working directory: `/Users/tdyar/ws/iris-vector-rag-private/`
- Private repo contains ALL files: code + `.claude/` + `.specify/` + `specs/` + tracking files
- Public fork NEVER receives private files (removed before push)

**Rationale**: GitHub does not allow private forks of public repositories. This three-remote strategy enables private development with full version control while maintaining clean public contributions.

### Private Files (NEVER Push to Public Fork)

The following files/directories are private development artifacts and MUST NEVER appear in PRs to public repo:
- `.claude/` - Claude Code commands and personal AI assistant setup
- `.specify/` - Feature specification system, constitution, scripts, templates
- `specs/` - Feature planning documents and specifications
- `STATUS.md`, `PROGRESS.md`, `TODO.md` - Development tracking files
- `FORK_WORKFLOW.md` - Workflow documentation

**These files live ONLY in**:
- ✅ Private repo (`origin`) - tracked with full version control
- ✅ Local development machine
- ❌ NEVER in public fork (`fork`)
- ❌ NEVER in community repo (`upstream`)

### Environment Management

**Local Virtual Environment REQUIRED**: ALL development and testing MUST use the local `.venv` environment created by `uv`. Global Python environments (system Python, miniconda, pyenv) MUST NOT be used for package installation or testing.

**Enforcement**:
```bash
# ALWAYS activate local venv before any development work
source .venv/bin/activate

# Verify you're using local environment
which python  # MUST show /path/to/project/.venv/bin/python
```

**Rationale**: Global environments cause version conflicts, test failures with stale packages, and unpredictable behavior. The local `.venv` ensures reproducible builds and consistent test results.

### Pre-Development Checklist

Before starting ANY feature work, MUST execute:
```bash
# 1. Verify you're in private repo directory
pwd  # MUST show /Users/tdyar/ws/iris-vector-rag-private

# 2. Sync with all remotes
git fetch origin    # Private development repo
git fetch fork      # Public fork (for PRs)
git fetch upstream  # Community repo

# 3. Verify branch status
git status
git branch -vv  # Check tracking relationships

# 4. Ensure working on latest master
git checkout master
git pull origin master --ff-only  # Fast-forward only from private repo

# 5. Activate local environment
source .venv/bin/activate
which python  # Verify local venv path
```

**Rationale**: Starting with stale branches or wrong environments leads to diverged histories, rejected pushes, and wasted time resolving conflicts later.

### Feature Development Workflow

**Branch Creation via /specify**:

The `/specify` command creates feature branches using `.specify/scripts/bash/create-new-feature.sh`. This script:
1. Determines next feature number (e.g., 053)
2. Creates branch name from description (e.g., `053-update-to-iris`)
3. Executes `git checkout -b <branch-name>`
4. Creates `specs/<branch-name>/` directory with templates

**CRITICAL**: The `/specify` script creates the branch but does NOT sync with remotes. You MUST verify branch creation succeeded:

```bash
# After /specify completes, verify branch
git branch  # Should show new feature branch with *
git status  # Should show "On branch 053-feature-name"

# If branch creation failed (no-git mode):
git checkout -b 053-feature-name  # Create manually
```

**Common /specify Issues**:
- Branch exists but spec files incomplete → Re-run `/specify` or manually create files
- Working directory dirty → Commit/stash changes before `/specify`
- Git not detected → Verify `.git/` directory exists

**Standard Feature Development**:
```bash
# 1. Start with /specify command (creates branch + spec directory)
# 2. Verify branch creation
git status

# 3. Complete planning phase
# - /plan generates plan.md, research.md, data-model.md
# - /tasks generates tasks.md
# - Review and refine planning documents

# 4. Implementation phase
# - /implement executes tasks (or manual development)
# - Write contract tests FIRST (TDD principle)
# - Implement features to pass tests
# - Run tests frequently: pytest specs/<feature>/contracts/

# 5. Commit early and often
git add <modified-files>
git commit -m "feat: <description>"  # Conventional commits

# 6. Never commit to wrong branch
git branch  # Verify you're on feature branch, NOT main
```

### Daily Development Workflow

**Phase 1: Daily Commits to Private Repo**
```bash
# 1. Verify feature complete
pytest specs/<feature>/contracts/ -v  # All tests pass
git status  # Working directory clean

# 2. Switch to master and merge feature branch
git checkout master
git merge <feature-branch> --no-edit  # Fast-forward preferred

# 3. Push to private repo (includes ALL files)
git push origin master  # Goes to private repo automatically

# 4. Verify push
git log --oneline -5  # Check commit history
```

### Creating Pull Requests to Public Repo

**Phase 1: Create Clean PR Branch (Remove Private Files)**
```bash
# 1. Create PR branch from your completed feature
git checkout -b pr/<feature-name> master

# 2. Remove private files from THIS branch only
git rm -r --cached .claude/
git rm -r --cached .specify/
git rm -r --cached specs/
git rm --cached STATUS.md PROGRESS.md TODO.md FORK_WORKFLOW.md

# 3. Commit the removal
git commit -m "chore: remove private development files for public PR"

# 4. Verify private files removed
git ls-files | grep -E "(\.claude|\.specify|specs/|STATUS\.md)" || echo "Clean ✅"
```

**Phase 2: Push to Public Fork**
```bash
# 1. Push PR branch to public fork
git push fork pr/<feature-name>

# 2. Verify on GitHub
# Visit: https://github.com/isc-tdyar/iris-vector-rag
# Should see new branch without private files
```

**Phase 3: Create Pull Request**
```bash
# On GitHub:
# 1. Go to https://github.com/intersystems-community/iris-vector-rag
# 2. Click "New Pull Request"
# 3. Click "compare across forks"
# 4. Set:
#    Base repository: intersystems-community/iris-vector-rag (base: main)
#    Head repository: isc-tdyar/iris-vector-rag (compare: pr/<feature-name>)
# 5. Create PR with descriptive title and summary
# 6. Verify PR does NOT contain .claude/, .specify/, specs/, or tracking files
```

**Phase 4: After PR Merged (Sync Back)**
```bash
# 1. Fetch updates from community repo
git fetch upstream

# 2. Merge into your private master
git checkout master
git merge upstream/main

# 3. Push updates to private repo
git push origin master

# 4. Delete PR branch (local and remote)
git branch -d pr/<feature-name>
git push fork --delete pr/<feature-name>
```

**Rationale**: Three-remote fork workflow keeps private development artifacts safe while enabling clean public contributions. Private repo is source of truth, public fork is only for PRs.

### Version Bump & PyPI Publishing

**When to Bump Version**: After feature complete and merged to main, before PyPI publish.

**Procedure**:
```bash
# 1. Determine version increment (semantic versioning)
# MAJOR.MINOR.PATCH (e.g., 0.2.5)
# - MAJOR: Breaking changes
# - MINOR: New features (backward compatible)
# - PATCH: Bug fixes

# 2. Update version in TWO files (BOTH required!)
# File 1: pyproject.toml
vim pyproject.toml  # Update version = "X.Y.Z"

# File 2: iris_rag/__init__.py
vim iris_rag/__init__.py  # Update __version__ = "X.Y.Z"

# 3. Commit version bump
git add pyproject.toml iris_rag/__init__.py
git commit -m "chore: bump version to X.Y.Z for <feature-name>

<One-line summary of what changed>

<Optional: detailed changelog if multiple features>"

# 4. Build and test package
python -m build  # Creates dist/ with .whl and .tar.gz
python -m twine check dist/*  # Validate package

# 5. Test local install
pip install --force-reinstall --no-deps dist/iris_vector_rag-X.Y.Z-*.whl
python -c "import iris_rag; print(iris_rag.__version__)"  # Verify version

# 6. Upload to PyPI (requires ~/.pypirc with credentials)
./scripts/upload/upload_to_pypi.sh
# OR manually:
python -m twine upload dist/*

# 7. Verify PyPI publication
curl -s https://pypi.org/pypi/iris-vector-rag/json | \
  python -c "import sys, json; data=json.load(sys.stdin); print(f\"Version: {data['info']['version']}\")"
```

**Complete PyPI Publishing Workflow Documentation**: See `.specify/memory/constitution.md` sections on version management and release procedures.

### Troubleshooting Common Git Issues

**Issue 1: Rejected Push (non-fast-forward)**
```
! [rejected]        main -> main (non-fast-forward)
```
**Cause**: Remote has commits you don't have locally.
**Fix**:
```bash
git fetch origin  # Get remote state
git log origin/main..main  # Your commits
git log main..origin/main  # Their commits
git pull origin main --rebase  # Rebase your commits on top
git push origin main  # Now should succeed
```

**Issue 2: Diverged Branches (Different Commit Hashes)**
```
Your branch and 'origin/main' have diverged
```
**Cause**: Same changes with different commit hashes (common with cherry-picks).
**Fix**:
```bash
# Verify you have the correct version locally
git log --oneline -10
# If local is correct source of truth:
git push origin main --force-with-lease
```

**Issue 3: Working on Wrong Branch**
```bash
# Accidentally committed to main instead of feature branch
git status  # Shows "On branch main"
```
**Fix**:
```bash
# Move commits to correct branch
git checkout -b <feature-branch>  # Creates branch with current commits
git checkout main
git reset --hard origin/main  # Reset main to match remote
git checkout <feature-branch>  # Continue work on feature
```

**Issue 4: Merge Conflicts During Cherry-pick**
```
CONFLICT (content): Merge conflict in pyproject.toml
```
**Fix**:
```bash
git status  # Review conflicted files
vim <conflicted-file>  # Resolve conflicts manually
git add <resolved-file>
git cherry-pick --continue  # Or --abort if unresolvable
```

**Issue 5: Stale Remote Tracking Information**
```bash
# Local thinks remote is at old commit
```
**Fix**:
```bash
git remote update --prune  # Refresh all remotes
git fetch --all --prune  # Alternative command
```

### Constitutional Compliance

**Pre-commit Checklist**:
- [ ] Using local `.venv` environment (verify with `which python`)
- [ ] On correct branch (verify with `git branch`)
- [ ] All tests passing (run `pytest`)
- [ ] Working directory clean or changes staged (verify with `git status`)
- [ ] Conventional commit message (e.g., `feat:`, `fix:`, `chore:`)

**Pre-push Checklist**:
- [ ] Synced with remote (`git fetch origin`)
- [ ] Fast-forward merge preferred (avoid `--no-ff`)
- [ ] Use `--force-with-lease` instead of `--force` if required
- [ ] Pushed to internal GitLab BEFORE public GitHub

**Pre-release Checklist**:
- [ ] Version bumped in BOTH `pyproject.toml` AND `iris_rag/__init__.py`
- [ ] Package builds successfully (`python -m build`)
- [ ] Local install test passes
- [ ] All contract tests passing (`pytest specs/*/contracts/`)
- [ ] Sanitized branch updated and pushed to GitHub

## Governance

This constitution supersedes all other development practices. All pull requests MUST verify constitutional compliance. Complexity that violates simplicity principles MUST be justified in writing with alternatives considered.

### AI Architecting Principles

Development with AI tools MUST follow constraint-based architecture, not "vibecoding". Constitutional validation gates serve as constraint checklists that prevent repeating known bugs and design mistakes. Every bug fix MUST be captured as a new validation rule or enhanced guideline. AI development MUST work within established frameworks, patterns, and validation loops.

**Constraint Philosophy**: Less freedom = less chaos. Constraints are superpowers that prevent regression and ensure consistency.

Amendment procedure: Proposed changes require documentation of impact, team approval, and migration plan for affected components. Version increments follow semantic versioning based on change scope.

**Version**: 1.8.0 | **Ratified**: 2025-01-27 | **Last Amended**: 2025-11-08
