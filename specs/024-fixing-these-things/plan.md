
# Implementation Plan: RAG-Templates Quality Improvement Initiative

**Branch**: `024-fixing-these-things` | **Date**: 2025-10-02 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/024-fixing-these-things/spec.md`

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
Fix 105 failing tests, increase code coverage from 9% to 60% overall and 80% for critical modules, establish CI/CD quality gates, and align test expectations with actual implementations. The approach focuses on systematic test repair, coverage improvement through TDD practices, and automated quality enforcement.

## Technical Context
**Language/Version**: Python 3.11+ (based on existing codebase)
**Primary Dependencies**: pytest, coverage.py, pytest-cov, Docker (for IRIS database)
**Storage**: IRIS database in Docker containers for test data
**Testing**: pytest with coverage plugins, pytest-randomly disabled due to conflicts
**Target Platform**: Linux/macOS development environments, CI/CD pipelines
**Project Type**: single - RAG framework library
**Performance Goals**: Test execution time max 2x current baseline (per clarifications)
**Constraints**: Must maintain backward compatibility for pipeline factory APIs
**Scale/Scope**: 349 tests, 12,775 lines of code, 5 critical modules

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**I. Framework-First Architecture**: N/A - This is a quality improvement initiative, not a new component

**II. Pipeline Validation & Requirements**: ✓ Will enhance existing validation | ✓ Will ensure setup procedures are idempotent

**III. Test-Driven Development**: ✓ Fixing tests to follow TDD principles | ✓ Will add performance tests for enterprise scenarios

**IV. Performance & Enterprise Scale**: ✓ Performance benchmarks will be established | ✓ Test execution time constraints defined

**V. Production Readiness**: ✓ CI/CD pipeline will be established | ✓ Docker test environment will be standardized

**VI. Explicit Error Handling**: ✓ Fixing AttributeError and mock issues | ✓ Tests will have clear failure messages

**VII. Standardized Database Interfaces**: ✓ Tests will use IRIS Docker instances | ✓ No ad-hoc database queries in tests

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
```
iris_rag/
├── config/          # Critical module - needs 80% coverage
├── validation/      # Critical module - needs 80% coverage
├── pipelines/       # Critical module - needs 80% coverage
├── services/        # Critical module - needs 80% coverage
├── storage/         # Critical module - needs 80% coverage
├── core/
├── embeddings/
├── memory/
├── ontology/
├── optimization/
├── testing/         # Coverage framework we built
├── utils/
└── visualization/

common/
├── *.py            # Shared utilities

tests/
├── conftest.py     # Test configuration
├── unit/           # Unit tests (many failing)
├── integration/    # Integration tests
├── e2e/           # End-to-end tests
└── contract/      # Contract tests (to be added)

.github/workflows/  # CI/CD configuration (to be added)
├── ci.yml
└── coverage.yml
```

**Structure Decision**: Single project structure - this is a Python framework library with existing test structure. Focus will be on fixing tests in the existing directories and adding contract tests for API stability.

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
- Categorize failing tests by failure type (AttributeError, Mock, Config, Type)
- Create repair tasks for each category [P]
- Generate coverage tasks for each critical module
- Create CI/CD setup tasks
- Add documentation tasks

**Ordering Strategy**:
1. **Environment Setup**: Docker test environment, dependencies
2. **Test Infrastructure**: Fix conftest.py, shared fixtures
3. **Test Repair by Category** [P]:
   - AttributeError fixes (40% of failures)
   - Mock configuration fixes (30% of failures)
   - Config/setup fixes (20% of failures)
   - Type mismatch fixes (10% of failures)
4. **Coverage Enhancement** [P]:
   - Config module tests
   - Validation module tests
   - Pipelines module tests
   - Services module tests
   - Storage module tests
5. **Contract Tests**: API stability tests for public interfaces
6. **CI/CD Integration**: GitHub Actions workflows, quality gates
7. **Documentation**: Test setup guide, contribution guidelines

**Estimated Output**: 35-40 numbered, ordered tasks in tasks.md
- Setup tasks: 5
- Test repair tasks: 15-20 (grouped by failure type)
- Coverage tasks: 10-15 (one per critical module area)
- Contract tests: 3-5
- CI/CD: 3-5
- Documentation: 2-3

**Parallelization**: Most test repair and coverage tasks can run in parallel (marked [P])

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)  
**Phase 4**: Implementation (execute tasks.md following constitutional principles)  
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking
*Fill ONLY if Constitution Check has violations that must be justified*

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |


## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [x] Complexity deviations documented (none required)

---
*Based on Constitution v1.2.0 - See `/.specify/memory/constitution.md`*
