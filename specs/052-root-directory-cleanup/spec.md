# Feature Specification: Root Directory Cleanup and Reorganization

**Feature Branch**: `052-root-directory-cleanup`
**Created**: 2025-11-24
**Status**: Draft
**Input**: User description: "Major cleanup and reorganization of root directory structure to improve project maintainability and developer experience"

## Clarifications

### Session 2025-11-24

- Q: What happens when archived directories (`archive/`, `backups/`) contain git-tracked files that need to be preserved? → A: Move git-tracked files from archives to `docs/archive/` or `docs/backups/` with documentation of origin; remove untracked archive files
- Q: How does system handle symlinks or hardlinks pointing to files in consolidated directories? → A: Convert all symlinks to regular files (copy target content) during consolidation to eliminate link complexity
- Q: What happens if legacy package directories (iris_rag/, rag_templates/) are still referenced by import statements? → A: Verify no active imports exist via codebase grep before removal; if imports found, keep directory with clear deprecation warning in README
- Q: How are orphaned configuration files (iris.key, temp_iris.key, multiple .env files) handled safely? → A: Consolidate `.env` files per FR-021; move `.key` files to `.gitignore`d `config/` directory with README explaining purpose
- Q: What happens to log files with historical value (e.g., indexing_WITH_DSPY_FULL.log) that might be referenced in documentation? → A: Move historically valuable logs to `docs/logs/historical/` with README explaining context; remove logs older than 3 months

## User Scenarios & Testing *(mandatory)*

### User Story 1 - Developer Onboarding Experience (Priority: P1)

New developers joining the project need to quickly understand the project structure and locate relevant files without being overwhelmed by clutter or confused by inconsistent organization.

**Why this priority**: First impressions matter critically for developer productivity and project adoption. A clean root directory is the first thing developers see and directly impacts their confidence in the project's quality and maintainability.

**Independent Test**: Can be fully tested by having a new developer clone the repository and complete a simple task (e.g., "find the test suite", "locate documentation", "run the application") within 5 minutes without asking for help. Delivers immediate value by reducing onboarding friction.

**Acceptance Scenarios**:

1. **Given** a new developer clones the repository, **When** they view the root directory, **Then** they see only essential project files and clearly organized directories
2. **Given** a developer needs to find test files, **When** they look in the root directory, **Then** they see a clear `tests/` directory with no test files scattered in the root
3. **Given** a developer needs documentation, **When** they check the root, **Then** they find a `docs/` directory with all documentation consolidated
4. **Given** a developer runs `ls -la` in root, **When** reviewing the output, **Then** they see fewer than 30 items (down from 114+)

---

### User Story 2 - Preventing Accidental Commits (Priority: P1)

Developers need the build system to automatically ignore temporary files, logs, and build artifacts to prevent accidentally committing noise to version control.

**Why this priority**: Accidental commits of logs, build artifacts, and temporary files pollute git history, increase repository size, and cause merge conflicts. This is a constant pain point that should be eliminated at the infrastructure level.

**Independent Test**: Can be tested by creating various temporary files (logs, test outputs, build artifacts) and verifying `git status` shows clean working directory. Delivers immediate value by preventing common mistakes.

**Acceptance Scenarios**:

1. **Given** a developer runs tests that generate log files, **When** they check `git status`, **Then** log files are automatically ignored
2. **Given** build artifacts are generated in `dist/` or `build/`, **When** developer checks `git status`, **Then** these directories are ignored
3. **Given** test output files are created (`.txt`, `.log`), **When** developer commits changes, **Then** only source code changes are staged
4. **Given** temporary directories like `outputs/`, `reports/`, `coverage_reports/` exist, **When** checking version control, **Then** these are not tracked

---

### User Story 3 - Project Documentation Discovery (Priority: P2)

Project stakeholders (developers, managers, contributors) need to find project documentation, status reports, and changelogs in a predictable location.

**Why this priority**: Scattered documentation creates information silos and makes it hard for stakeholders to understand project status, progress, and decisions. Consolidation improves transparency.

**Independent Test**: Can be tested by asking a non-developer (e.g., project manager) to find specific information (current status, changelog, progress) within 2 minutes. Delivers value by improving project transparency.

**Acceptance Scenarios**:

1. **Given** stakeholder needs project status, **When** they check `docs/` directory, **Then** they find `STATUS.md`, `PROGRESS.md`, and `TODO.md` in one place
2. **Given** developer needs feature documentation, **When** they check `docs/`, **Then** they find all markdown documentation files consolidated
3. **Given** user wants to see changelogs, **When** they check root directory, **Then** they find `CHANGELOG.md` easily accessible at root level (standard practice)
4. **Given** contributor needs to understand recent bug fixes, **When** they check `docs/`, **Then** they find bug fix summaries and regression documentation

---

### User Story 4 - Dependency and Package Management Clarity (Priority: P2)

Developers need to understand which package manager and dependency files are current and which are legacy/deprecated.

**Why this priority**: Multiple dependency files (poetry.lock, requirements.txt, requirements-dev.txt, uv.lock) create confusion about which is the source of truth. This leads to dependency conflicts and inconsistent environments.

**Independent Test**: Can be tested by asking a developer to set up the development environment and verifying they use the correct tool (uv) without consulting anyone. Delivers value by standardizing setup process.

**Acceptance Scenarios**:

1. **Given** project uses `uv` for dependency management, **When** developer checks root, **Then** they see `pyproject.toml` and `uv.lock` as primary files
2. **Given** legacy files exist (poetry.lock, requirements.txt), **When** reviewing root directory, **Then** deprecated files are clearly marked or moved to archive
3. **Given** developer follows README instructions, **When** setting up environment, **Then** instructions reference only current package manager (uv)
4. **Given** CI/CD pipeline runs, **When** installing dependencies, **Then** only uv.lock and pyproject.toml are used

---

### User Story 5 - Streamlined Git Workflow (Priority: P3)

Developers need a clean `git status` output that shows only meaningful changes without noise from build artifacts, logs, or temporary files.

**Why this priority**: Developers check `git status` dozens of times per day. A noisy output slows down workflow and increases the chance of committing wrong files. Clean output improves developer velocity.

**Independent Test**: Can be tested by running a full test suite with coverage, building the package, and verifying `git status` remains clean. Delivers value through improved daily workflow efficiency.

**Acceptance Scenarios**:

1. **Given** developer runs tests with coverage, **When** they check `git status`, **Then** coverage files and reports are not shown
2. **Given** developer builds distribution packages, **When** they check `git status`, **Then** `dist/` and `build/` artifacts are ignored
3. **Given** developer runs examples or scripts, **When** output files are generated, **Then** these temporary outputs don't appear in `git status`
4. **Given** developer works on feature, **When** they commit, **Then** only relevant source files appear in staged changes

---

### Edge Cases

*(All edge cases resolved through clarifications - see Clarifications section above)*

## Requirements *(mandatory)*

### Functional Requirements

#### Directory Structure

- **FR-001**: Root directory MUST contain fewer than 30 items (down from 114+)
- **FR-002**: All test files MUST be moved from root to `tests/` directory
- **FR-003**: Log files MUST be processed as follows: historically valuable logs (referenced in documentation via `grep -r "logfilename.log" docs/` OR file mtime <3 months) moved to `docs/logs/historical/` with README explaining context; all other logs removed
- **FR-004**: All temporary output files MUST be removed from root directory
- **FR-005**: Build artifacts (`dist/`, `build/`, `*.egg-info/`, `.eggs/`), logs (`*.log`), and temporary files MUST be in .gitignore
- **FR-006**: Output directories that regenerate on test/build runs (`outputs/`, `reports/`, `validation_results/`, `test_results/`, `coverage_reports/`, `.pytest_cache/`, `htmlcov/`) MUST be in .gitignore
- **FR-007**: Documentation files MUST be consolidated in `docs/` directory, except `README.md`, `CHANGELOG.md`, and `LICENSE` which remain at root
- **FR-008**: Archive and backup directories MUST be processed as follows: git-tracked files moved to `docs/archive/` or `docs/backups/` with origin documentation; untracked archive files removed
- **FR-008a**: Symlinks and hardlinks MUST be converted to regular files (copying target content) during consolidation to eliminate link complexity

#### Package Structure

- **FR-009**: System MUST have ONE primary package directory (`iris_vector_rag/`)
- **FR-010**: Legacy package directories (`iris_rag/`, `rag_templates/`, `common/`) MUST be verified via codebase grep for active imports; if no imports found, remove; if imports exist, keep with deprecation warning in README
- **FR-011**: Orphaned integration directories (`mem0_integration/`, `mem0-mcp-server/`, `supabase-mcp-memory-server/`) MUST be removed or moved to appropriate location
- **FR-012**: Language-specific directories (`nodejs/`, `objectscript/`) MUST be evaluated using these criteria: (1) Check for import references in codebase via grep, (2) Check for references in Docker files/workflows, (3) Check for documentation in README. If no references found → remove; if referenced → move to `tools/<language>/`

#### Dependency Management

- **FR-013**: Project MUST use one primary package manager (uv) with `pyproject.toml` and `uv.lock` as source of truth
- **FR-014**: Legacy dependency files (`poetry.lock`) MUST be removed
- **FR-015**: Generated dependency files (`requirements.txt`, `requirements-dev.txt`) MUST be either removed or generated via uv export
- **FR-016**: README MUST document the primary package manager and setup process

#### Git Hygiene

- **FR-017**: `.gitignore` MUST use clear section headers and comments for maintainability (see research.md for hierarchical structure template)
- **FR-018**: `.gitignore` MUST exclude environment-specific files (.env, *.key) while keeping examples (.env.example)
- **FR-019**: `.gitignore` MUST exclude IDE files (`.vscode/`, `.idea/`, `*.swp`, `*.swo`, `*~`) and OS artifacts (`.DS_Store`, `Thumbs.db`)
- **FR-020**: Git history MUST NOT be rewritten (no force pushes) - cleanup affects only working directory

#### Configuration Files

- **FR-021**: Multiple `.env` files MUST be consolidated to single `.env` (gitignored) with `.env.example` for reference
- **FR-022**: Orphaned `.key` files (iris.key, temp_iris.key) MUST be moved to `.gitignore`d `config/` directory with README explaining their purpose; redundant duplicates removed
- **FR-023**: Push log files (`push_*.log`) MUST be removed as they provide no ongoing value

### Key Entities

- **Root Directory Structure**: The organization of files and directories at repository root - represents project's first impression and navigability
- **Ignored Files Patterns**: Collections of file patterns in `.gitignore` - determines what gets tracked in version control
- **Legacy Artifacts**: Outdated files, logs, and directories that no longer serve a purpose - represent technical debt
- **Package Directory**: The primary Python package (`iris_vector_rag/`) - distinguishes from legacy or duplicate package dirs
- **Documentation Files**: Project documentation and status reports - need clear single location for discoverability

## Success Criteria *(mandatory)*

### Measurable Outcomes

- **SC-001**: Root directory contains fewer than 30 items (currently 114), representing 75% reduction in visual clutter
- **SC-002**: New developers can locate test files, documentation, and main package directory within 2 minutes of cloning
- **SC-003**: `git status` output remains clean (no untracked files) after running full test suite and building package
- **SC-004**: 100% of build artifacts, logs, and temporary files are automatically ignored by version control
- **SC-005**: All team members can set up development environment using single command documented in README
- **SC-006**: Zero accidental commits of log files, build artifacts, or temporary outputs in next 30 days
- **SC-007**: Project passes standard repository linter (e.g., repolinter) checks for directory organization
- **SC-008**: 90% of developers report improved ability to navigate project structure in post-cleanup survey

## Scope *(mandatory)*

### In Scope

- Reorganization of root directory file structure
- Consolidation of documentation files into `docs/`
- Cleanup of log files and temporary outputs
- Update of `.gitignore` for comprehensive coverage
- Removal or documentation of legacy package directories
- Consolidation of dependency management files
- Update of README with current directory structure
- Documentation of what was removed/moved and why

### Out of Scope

- Refactoring of code within `iris_vector_rag/` package (internal structure unchanged)
- Migration of build system or package manager (uv is already in use)
- Changes to CI/CD pipeline configuration (GitHub workflows remain unchanged)
- Rewriting git history or removing files from git history (affects only working directory)
- Changes to test infrastructure or test file organization within `tests/` directory
- Updates to Docker configuration or Kubernetes manifests

### Dependencies

- No external dependencies - this is pure organizational work
- Requires careful review of which files are safe to delete vs. archive
- Should be coordinated with any active feature branches to minimize merge conflicts

### Assumptions

- Project is transitioning to or has transitioned to `uv` for dependency management (based on presence of uv.lock)
- Log files in root directory are from historical debugging sessions and not actively monitored
- Archive and backup directories are developer-local artifacts not intended for version control
- Output directories (outputs/, reports/, validation_results/) regenerate automatically and don't need preservation
- Multiple .env files represent different configuration attempts, with .env.example being the canonical template
- Legacy package directories (iris_rag/, rag_templates/, common/) are either unused or duplicates of iris_vector_rag/

## Non-Functional Requirements

### Performance

- **NFR-001**: Cleanup operations MUST complete within 15 minutes of developer time
- **NFR-002**: Git operations (status, commit, push) MUST not slow down after cleanup (same or faster)
- **NFR-003**: Repository size MUST not increase (may decrease if large logs removed)

### Maintainability

- **NFR-004**: Directory structure MUST follow Python project best practices (PEP standards)
- **NFR-005**: `.gitignore` MUST use clear sections and comments for future maintainability
- **NFR-006**: CHANGELOG MUST document this reorganization for transparency

### Compatibility

- **NFR-007**: Cleanup MUST NOT break existing imports or module references
- **NFR-008**: Cleanup MUST NOT affect CI/CD pipeline execution
- **NFR-009**: Cleanup MUST NOT break Docker builds or deployments
- **NFR-010**: Changes MUST be backwards compatible with existing development workflows
