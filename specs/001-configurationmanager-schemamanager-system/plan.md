
# Implementation Plan: ConfigurationManager → SchemaManager System

**Branch**: `001-configurationmanager-schemamanager-system` | **Date**: 2025-01-30 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/001-configurationmanager-schemamanager-system/spec.md`

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
RAG framework developers need a reliable configuration management system that automatically handles database schema migrations and vector dimension consistency across all pipeline components. The system must prevent configuration drift, dimension mismatches, and schema inconsistencies that could break production deployments.

## Technical Context
**Language/Version**: Python 3.12+
**Primary Dependencies**: InterSystems IRIS, YAML configuration files, environment variable management
**Storage**: InterSystems IRIS database with vector search capabilities, HNSW indexes with ACORN=1 optimization
**Testing**: pytest with live IRIS database validation, @pytest.mark.requires_database markers
**Target Platform**: Docker containerized environment supporting licensed IRIS instances
**Project Type**: single - Python framework component
**Performance Goals**: <50ms configuration access, <5s schema migrations, enterprise scale 10K+ documents
**Constraints**: IRIS database dependency, transaction-safe migrations with automatic rollback, constitutional compliance required
**Scale/Scope**: Framework component serving all RAG pipeline implementations, vector dimension authority for entire system

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**I. Framework-First Architecture**: ✓ Component provides reusable framework utilities | ✓ No application-specific logic | ✓ CLI interface via Make targets exposed

**II. Pipeline Validation & Requirements**: ✓ Automated requirement validation included | ✓ Setup procedures idempotent with transaction safety

**III. Test-Driven Development**: ✓ Contract tests for all public interfaces | ✓ Performance tests against live IRIS with 10K+ scenarios

**IV. Performance & Enterprise Scale**: ✓ Incremental schema updates supported | ✓ IRIS vector operations with ACORN=1 optimization

**V. Production Readiness**: ✓ Structured logging with audit trails | ✓ Health checks via database connectivity validation | ✓ Docker deployment with licensed IRIS

**VI. Explicit Error Handling**: ✓ No silent failures - all errors surface as clear exceptions | ✓ Clear exception messages with actionable context | ✓ Transaction rollback on failure with detailed logging

**VII. Standardized Database Interfaces**: ✓ Uses proven common/iris_sql_utils patterns | ✓ No ad-hoc IRIS queries - standardized connection management | ✓ Audit methods replace direct SQL access

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
├── config/
│   ├── __init__.py
│   ├── manager.py              # ConfigurationManager - central config authority
│   └── default_config.yaml     # Default configuration template
├── storage/
│   ├── __init__.py
│   ├── schema_manager.py       # SchemaManager - vector dimension authority
│   └── vector_store_iris.py    # IRIS vector store integration
└── core/
    ├── __init__.py
    ├── connection.py           # Standardized IRIS connection management
    └── models.py               # Core data models

common/
├── iris_sql_utils.py          # Proven SQL utilities for IRIS operations
└── iris_dbapi_connector.py    # Standardized IRIS database connector

tests/
├── unit/
│   ├── config/
│   │   └── test_manager.py
│   └── storage/
│       └── test_schema_manager.py
├── integration/
│   └── storage/
│       └── test_schema_manager.py
└── contract/
    ├── test_configuration_manager_contract.py
    └── test_schema_manager_contract.py
```

**Structure Decision**: Single project structure selected as this is a framework component providing reusable configuration and schema management utilities. The implementation leverages existing iris_rag structure with new config/ module for ConfigurationManager and enhanced storage/schema_manager.py for SchemaManager functionality.

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

| Violation | Why Needed | Simpler Alternative Rejected Because |
|-----------|------------|-------------------------------------|
| [e.g., 4th project] | [current need] | [why 3 projects insufficient] |
| [e.g., Repository pattern] | [specific problem] | [why direct DB access insufficient] |


## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [ ] Phase 0: Research complete (/plan command)
- [ ] Phase 1: Design complete (/plan command)
- [ ] Phase 2: Task planning complete (/plan command - describe approach only)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [ ] Initial Constitution Check: PASS
- [ ] Post-Design Constitution Check: PASS
- [ ] All NEEDS CLARIFICATION resolved
- [ ] Complexity deviations documented

---
*Based on Constitution v1.2.0 - See `/.specify/memory/constitution.md`*
