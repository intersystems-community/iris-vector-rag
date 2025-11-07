
# Implementation Plan: Root Directory Cleanup

**Branch**: `052-i-thought-we` | **Date**: 2025-11-07 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/Users/tdyar/ws/rag-templates/specs/052-i-thought-we/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path
   → If not found: ERROR "No feature spec at {path}"
2. Fill Technical Context (scan for NEEDS CLARIFICATION)
   → Detect Project Type from file system structure or context (web=frontend+backend, mobile=app+api)
   → Set Structure Decision based on project type
3. Fill the Constitution Check section based on the content of the constitution document.
4. Evaluate Constitution Check section below
   → If violations exist: Document in Complexity Tracking
   → If no justification possible: ERROR "Simplify approach first"
   → Update Progress Tracking: Initial Constitution Check
5. Execute Phase 0 → research.md
   → If NEEDS CLARIFICATION remain: ERROR "Resolve unknowns"
6. Execute Phase 1 → contracts, data-model.md, quickstart.md, agent-specific template file (e.g., `CLAUDE.md` for Claude Code, `.github/copilot-instructions.md` for GitHub Copilot, `GEMINI.md` for Gemini CLI, `QWEN.md` for Qwen Code or `AGENTS.md` for opencode).
7. Re-evaluate Constitution Check section
   → If new violations: Refactor design, return to Phase 1
   → Update Progress Tracking: Post-Design Constitution Check
8. Plan Phase 2 → Describe task generation approach (DO NOT create tasks.md)
9. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary

Reduce repository root directory clutter from 991 files to <20 essential files by organizing configuration files, scripts, documentation, and artifacts into appropriate subdirectories while maintaining all CI/CD and development workflow functionality.

## Technical Context
**Language/Version**: Python 3.11+, Bash (shell scripts), Git 2.0+
**Primary Dependencies**: git, bash, make, docker-compose
**Storage**: File system reorganization (moving files to subdirectories)
**Testing**: pytest for CI/CD validation after reorganization
**Target Platform**: GitHub repository structure (intersystems-community/iris-vector-rag)
**Project Type**: Repository maintenance (not application code)
**Performance Goals**: N/A (file organization task)
**Constraints**: Must maintain all CI/CD workflows, docker-compose functionality, development commands (make targets), and tool configurations
**Scale/Scope**: 991 files → <20 files in root directory, organize 942 log files, consolidate 7 docker-compose variants, relocate 4+ markdown files

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**I. Framework-First Architecture**: N/A (repository organization, not framework code)

**II. Pipeline Validation & Requirements**: N/A (no pipeline implementation)

**III. Test-Driven Development**: ✓ Validation tests for CI/CD workflows | N/A (no 10K+ scenarios - file organization task)

**IV. Performance & Enterprise Scale**: N/A (file organization, not data operations)

**V. Production Readiness**: ✓ Maintains existing Docker deployment | ✓ Preserves CI/CD workflows | N/A (no new logging/health checks)

**VI. Explicit Error Handling**: ✓ CI/CD failures will surface immediately if reorganization breaks workflows | ✓ Git operations provide clear error messages

**VII. Standardized Database Interfaces**: N/A (no database operations)

**VIII. Git & Release Workflow**: ✓ Changes committed to internal repository (rag-templates) | ✓ Sync to sanitized repository (iris-vector-rag) | ✓ No PyPI release needed (non-code changes)

## Project Structure

### Documentation (this feature)
```
specs/[###-feature]/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)

**Target Structure** (after cleanup):
```
/Users/tdyar/ws/rag-templates/  (root - max 20 files)
├── README.md                    # Project overview (KEEP)
├── LICENSE                      # License file (KEEP)
├── pyproject.toml              # Python package config (KEEP)
├── Makefile                     # Build commands (KEEP)
├── docker-compose.yml          # Primary Docker setup (KEEP)
├── .gitignore                  # Version control config (KEEP)
├── .env.example                # Environment template (KEEP)
│
├── archive/                    # NEW: Archived/deprecated content
│   ├── old_results/           # Old evaluation results (FR-011)
│   └── deprecated_scripts/    # Old scripts no longer used
│
├── config/                     # Configuration files
│   ├── .flake8               # Linting config (FR-006)
│   ├── .coveragerc           # Coverage config (FR-006)
│   ├── .pre-commit-config.yaml
│   └── docker/               # Docker compose variants (FR-007)
│       ├── docker-compose.api.yml
│       ├── docker-compose.full.yml
│       ├── docker-compose.licensed.yml
│       ├── docker-compose.mcp.yml
│       ├── docker-compose.test.yml
│       └── docker-compose.iris-only.yml
│
├── docs/                       # Documentation (FR-005)
│   ├── CONTRIBUTING.md
│   ├── USER_GUIDE.md
│   ├── CLAUDE.md
│   └── ... (other .md files)
│
├── logs/                       # NEW: Log files (FR-002)
│   └── indexing/             # 942 indexing logs
│
├── scripts/                    # Shell scripts (FR-003)
│   ├── activate_env.sh
│   ├── setup/
│   └── upload/
│
├── tests/                      # Test suite (existing)
│   ├── artifacts/            # NEW: Test artifacts (FR-004)
│   │   ├── coverage_html/
│   │   ├── htmlcov/
│   │   └── results/
│   ├── contract/
│   ├── integration/
│   └── unit/
│
└── iris_rag/                   # Source code (existing, unchanged)
    └── ... (framework code)
```

**Current State** (before cleanup):
- 991 files in root directory
- 942 log files (indexing_CONTINUOUS_RUN_*.log)
- 7 docker-compose variants in root
- 4+ markdown files scattered
- Multiple coverage artifacts (.coverage, coverage.json, htmlcov/, coverage_html/)
- Temporary files (.DS_Store)

**Structure Decision**: This is a single Python project (Option 1) with repository maintenance scope. The cleanup reorganizes existing files without changing the iris_rag/ framework structure. Focus is on root directory organization per FR-001 through FR-011.

## Phase 0: Outline & Research
1. **Extract unknowns from Technical Context** above:
   - For each NEEDS CLARIFICATION → research task
   - For each dependency → best practices task
   - For each integration → patterns task

2. **Generate and dispatch research agents**:
   ```
   For each unknown in Technical Context:
     Task: "Research {unknown} for {feature context}"
   For each technology choice:
     Task: "Find best practices for {tech} in {domain}"
   ```

3. **Consolidate findings** in `research.md` using format:
   - Decision: [what was chosen]
   - Rationale: [why chosen]
   - Alternatives considered: [what else evaluated]

**Output**: research.md with all NEEDS CLARIFICATION resolved

## Phase 1: Design & Contracts
*Prerequisites: research.md complete*

1. **Extract entities from feature spec** → `data-model.md`:
   - Entity name, fields, relationships
   - Validation rules from requirements
   - State transitions if applicable

2. **Generate API contracts** from functional requirements:
   - For each user action → endpoint
   - Use standard REST/GraphQL patterns
   - Output OpenAPI/GraphQL schema to `/contracts/`

3. **Generate contract tests** from contracts:
   - One test file per endpoint
   - Assert request/response schemas
   - Tests must fail (no implementation yet)

4. **Extract test scenarios** from user stories:
   - Each story → integration test scenario
   - Quickstart test = story validation steps

5. **Update agent file incrementally** (O(1) operation):
   - Run `.specify/scripts/bash/update-agent-context.sh claude`
     **IMPORTANT**: Execute it exactly as specified above. Do not add or remove any arguments.
   - If exists: Add only NEW tech from current plan
   - Preserve manual additions between markers
   - Update recent changes (keep last 3)
   - Keep under 150 lines for token efficiency
   - Output to repository root

**Output**: data-model.md, /contracts/*, failing tests, quickstart.md, agent-specific file

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
- Load `.specify/templates/tasks-template.md` as base
- Generate tasks from Phase 1 design docs (contracts, data model, quickstart)
- Each contract → contract test task [P]
- Each entity → model creation task [P] 
- Each user story → integration test task
- Implementation tasks to make tests pass

**Ordering Strategy**:
- TDD order: Tests before implementation 
- Dependency order: Models before services before UI
- Mark [P] for parallel execution (independent files)

**Estimated Output**: 25-30 numbered, ordered tasks in tasks.md

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)  
**Phase 4**: Implementation (execute tasks.md following constitutional principles)  
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking
*Fill ONLY if Constitution Check has violations that must be justified*

No constitution violations - this is a repository maintenance task, not a framework implementation. Most constitution principles are N/A for file organization.

## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [x] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [x] Complexity deviations documented (none)

---
*Based on Constitution v1.2.0 - See `/.specify/memory/constitution.md`*
