# Feature Specification: Root Directory Cleanup

**Feature Branch**: `052-i-thought-we`
**Created**: 2025-11-07
**Status**: Draft
**Input**: User description: "I thought we cleaned up the root directory, but on main branch in https://github.com/intersystems-community/iris-vector-rag -- I see way too many files in top level dir! there is .flake8 file, many scripts, many .md files, etc - let's do a top-to-bottom cleanup"

## Execution Flow (main)
```
1. Parse user description from Input
   → Repository root directory has excessive clutter (991 files)
2. Extract key concepts from description
   → Actors: Repository maintainers, contributors, users browsing GitHub
   → Actions: Organize files, move to appropriate subdirectories, remove obsolete files
   → Data: Configuration files, scripts, logs, documentation, test artifacts
   → Constraints: Must maintain functionality, follow best practices
3. For each unclear aspect:
   → [RESOLVED] File retention policy determined from analysis
4. Fill User Scenarios & Testing section
   → Repository browsing experience improved
5. Generate Functional Requirements
   → Each requirement specifies file organization rules
6. Identify Key Entities (if data involved)
   → File categories identified
7. Run Review Checklist
   → No implementation details included
8. Return: SUCCESS (spec ready for planning)
```

---

## ⚡ Quick Guidelines
- ✅ Focus on WHAT users need and WHY
- ❌ Avoid HOW to implement (no tech stack, APIs, code structure)
- 👥 Written for business stakeholders, not developers

---

## User Scenarios & Testing

### Primary User Story
As a repository contributor or user browsing the GitHub repository, I want a clean and organized root directory so that I can quickly understand the project structure, find relevant files, and navigate the codebase without being overwhelmed by clutter.

### Acceptance Scenarios
1. **Given** a user visits the GitHub repository homepage, **When** they view the root directory listing, **Then** they should see only essential project files (README, LICENSE, configuration files) and well-organized subdirectories
2. **Given** a developer wants to run the project, **When** they look at the root directory, **Then** they should find clear entry points (docker-compose.yml, setup scripts) without searching through hundreds of files
3. **Given** a contributor wants to understand the codebase, **When** they browse the repository, **Then** they should see a logical organization with documentation in docs/, scripts in scripts/, tests in tests/, etc.
4. **Given** the cleanup is complete, **When** running existing workflows (CI/CD, local development, Docker), **Then** all functionality should continue working without breaking changes

### Edge Cases
- What happens when obsolete log files (942 indexing logs) are removed?
  - System should continue logging to appropriate locations (logs/ directory)
- How does system handle moved configuration files (.flake8, .coveragerc)?
  - Tools should still find configuration through standard search paths or updated references
- What about deprecated scripts that might still be referenced?
  - Must verify no active references before removal or update references

---

## Requirements

### Functional Requirements

**File Organization:**
- **FR-001**: System MUST organize root directory to contain only essential project files (README.md, LICENSE, pyproject.toml, docker-compose files, Makefile)
- **FR-002**: System MUST move all log files (942+ indexing logs, evaluation.log, cleanup_log.txt) to a logs/ directory or remove if obsolete
- **FR-003**: System MUST consolidate all shell scripts (activate_env.sh, setup scripts, upload scripts) into scripts/ directory
- **FR-004**: System MUST move all test results and coverage artifacts (htmlcov/, coverage_html/, comprehensive_ragas_results*/) to appropriate test artifact directories
- **FR-005**: System MUST organize documentation files (CONTRIBUTING.md, USER_GUIDE.md, CLAUDE.md) in docs/ directory while keeping README.md in root

**Configuration Management:**
- **FR-006**: System MUST relocate development configuration files (.flake8, .coveragerc, .coveragerc.ci, .pre-commit-config.yaml) to appropriate locations where tools can find them
- **FR-007**: System MUST consolidate multiple docker-compose files (docker-compose.yml, docker-compose.api.yml, docker-compose.full.yml, docker-compose.licensed.yml, docker-compose.mcp.yml, docker-compose.test.yml, docker-compose.iris-only.yml) with clear naming
- **FR-008**: System MUST move .gitlab-ci.yml, baseline_tests.txt, and other CI/test artifacts to appropriate directories

**Data Cleanup:**
- **FR-009**: System MUST remove obsolete directories (comprehensive_ragas_results_20250619_*, future_tests_not_ready if truly not ready)
- **FR-010**: System MUST remove temporary/generated files that should not be in version control (.DS_Store, .coverage, coverage.json)
- **FR-011**: System MUST evaluate and remove or archive old evaluation results (eval_results/ directory with 65+ subdirectories)

**Backward Compatibility:**
- **FR-012**: System MUST ensure all CI/CD pipelines continue functioning after reorganization
- **FR-013**: System MUST ensure local development workflows (make commands, docker commands) continue working
- **FR-014**: System MUST update any hardcoded paths in scripts, configuration, or documentation

**Documentation:**
- **FR-015**: System MUST update README.md to reflect new directory structure
- **FR-016**: System MUST add/update CONTRIBUTING.md with guidance on where to place new files

### Key Entities

- **Root Directory Files**: Essential project files that must remain in root
  - README.md (project overview)
  - LICENSE (legal)
  - pyproject.toml (Python package configuration)
  - docker-compose.yml (primary Docker setup)
  - Makefile (common commands)
  - .gitignore (version control configuration)

- **Configuration Files**: Development and tool configurations
  - Linting configs (.flake8, .pre-commit-config.yaml)
  - Coverage configs (.coveragerc, .coveragerc.ci)
  - Docker configs (multiple docker-compose variants)
  - CI/CD configs (.gitlab-ci.yml, .github/)

- **Scripts**: Executable files for setup, deployment, utilities
  - Environment setup (activate_env.sh, setup scripts)
  - Deployment (upload_to_pypi.sh)
  - Docker utilities (docker-entrypoint-mcp.sh)

- **Artifacts**: Generated files from tests, builds, and operations
  - Coverage reports (htmlcov/, coverage_html/, .coverage, coverage.json)
  - Test results (comprehensive_ragas_results*)
  - Evaluation results (eval_results/, evaluation_framework/)
  - Log files (942+ indexing_CONTINUOUS_RUN_*.log files)

- **Documentation**: User and developer guides
  - User-facing (README.md, USER_GUIDE.md)
  - Contributor-facing (CONTRIBUTING.md, CLAUDE.md)
  - API/technical docs (docs/ directory)

---

## Review & Acceptance Checklist

### Content Quality
- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

### Requirement Completeness
- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable (e.g., "root directory contains only X essential files")
- [x] Scope is clearly bounded (reorganization only, no feature changes)
- [x] Dependencies and assumptions identified (CI/CD must continue working)

---

## Execution Status

- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked (none remaining)
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [x] Review checklist passed

---

## Additional Context

### Current State Analysis
- **Total files in root**: 991 files
- **Log files**: 942 indexing logs (indexing_CONTINUOUS_RUN_*.log)
- **Markdown files**: 4 (README.md, CONTRIBUTING.md, CLAUDE.md, USER_GUIDE.md)
- **Docker compose variants**: 7 different configurations
- **Old result directories**: Multiple timestamped comprehensive_ragas_results directories
- **Coverage artifacts**: Multiple locations (htmlcov/, coverage_html/, .coverage, coverage.json)

### Expected Outcome
A clean root directory with:
- ~10-15 essential files visible in GitHub root view
- Organized subdirectories (scripts/, docs/, logs/, config/)
- No generated/temporary files in version control
- Clear project structure for new contributors
- Maintained functionality for all workflows

### Success Metrics
- Root directory file count reduced from 991 to <20 files
- All CI/CD tests pass after reorganization
- Docker compose and development workflows function unchanged
- GitHub repository landing page shows clean, professional structure
