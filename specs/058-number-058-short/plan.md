
# Implementation Plan: Cloud Configuration Flexibility

**Branch**: `058-number-058-short` | **Date**: 2025-01-12 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/058-number-058-short/spec.md`

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

Enable flexible cloud deployment configuration for iris-vector-rag by implementing environment variable support, respecting config file specifications, and supporting variable vector dimensions (128-8192). This addresses 9 documented pain points from FHIR-AI-Hackathon-Kit AWS migration that blocked cloud deployments. Technical approach: Extend existing ConfigurationManager with 12-factor app configuration priority (env vars > config > defaults), add preflight validation for vector dimensions and namespaces, and ensure 100% backward compatibility through optional configuration with sensible defaults.

**Impact**: Reduce cloud deployment time from 65 minutes to under 25 minutes (60% reduction) and unblock NVIDIA NIM, OpenAI embedding models.

## Technical Context
**Language/Version**: Python 3.11+
**Primary Dependencies**: PyYAML (existing), python-dotenv (existing), iris-dbapi (existing), pydantic (for validation)
**Storage**: InterSystems IRIS database with vector tables (RAG.Entities, RAG.EntityRelationships)
**Testing**: pytest with contract tests (TDD approach), integration tests against live IRIS
**Target Platform**: Linux/macOS/Windows (Python cross-platform), AWS IRIS, Azure IRIS, GCP IRIS, on-premises IRIS
**Project Type**: Single project (iris-vector-rag framework package)
**Performance Goals**: Configuration validation overhead < 100ms at startup, zero impact on query performance
**Constraints**: 100% backward compatibility (existing local deployments must work unchanged), no breaking changes to public APIs
**Scale/Scope**: Extends existing ConfigurationManager (~15 files affected), adds 5-8 new configuration validation utilities, 15-20 contract tests

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**I. Framework-First Architecture**:
- ✓ Extends existing ConfigurationManager (framework component, not RAGPipeline)
- ✓ No application-specific logic (pure configuration management)
- ✓ CLI interface exposed via init_tables --config flag

**II. Pipeline Validation & Requirements**:
- ✓ Automated validation for vector dimensions, namespace permissions, config file existence
- ✓ Setup procedures idempotent (can run init_tables multiple times safely)

**III. Test-Driven Development**:
- ✓ Contract tests written before implementation (TDD principle)
- N/A Performance tests for 10K+ scenarios (configuration is startup-time, not query-time)

**IV. Performance & Enterprise Scale**:
- N/A Incremental indexing (this is configuration feature)
- N/A IRIS vector operations (no changes to vector operations)

**V. Production Readiness**:
- ✓ Structured logging for configuration sources (env var, file, defaults)
- ✓ Health checks for config validation at startup
- ✓ Docker deployment ready (environment variable support for containers)

**VI. Explicit Error Handling**:
- ✓ No silent failures (fail fast on invalid config)
- ✓ Clear exception messages (ConfigValidationError with actionable details)
- ✓ Actionable error context (e.g., "Vector dimension mismatch: configured 1024, existing table has 384. Run migration guide at...")

**VII. Standardized Database Interfaces**:
- ✓ Uses existing ConnectionManager and ConfigurationManager patterns
- ✓ No ad-hoc IRIS queries (extends existing utilities)
- ✓ New validation patterns will be contributed to config/ module

**Initial Constitution Check**: ✅ PASS - All applicable principles satisfied

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
iris_vector_rag/
├── config/
│   ├── manager.py                    # EXTEND: Add env var priority, validation
│   ├── backend_modes.py              # EXISTING: No changes needed
│   ├── embedding_config.py           # EXTEND: Add vector dimension validation
│   ├── pipeline_config_service.py    # EXTEND: Use enhanced ConfigurationManager
│   ├── default_config.yaml           # EXTEND: Add new configuration schema
│   └── validators.py                 # NEW: Configuration validation utilities
├── core/
│   ├── connection.py                 # EXTEND: Use env vars for connection params
│   └── models.py                     # REVIEW: Ensure vector dimension flexibility
├── cli/
│   └── init_tables.py                # EXTEND: Respect --config flag
└── services/
    └── storage.py                    # EXTEND: Support schema-prefixed table names

config/
├── default.yaml                      # EXTEND: Add cloud deployment examples
├── examples/
│   ├── aws.yaml                      # NEW: AWS IRIS configuration template
│   ├── azure.yaml                    # NEW: Azure IRIS configuration template
│   └── local.yaml                    # NEW: Local development configuration template

tests/
├── contract/
│   └── test_cloud_config_contract.py # NEW: 15-20 contract tests
├── integration/
│   └── test_config_integration.py    # NEW: Integration tests with live IRIS
└── unit/
    └── test_config_validators.py     # NEW: Unit tests for validation logic
```

**Structure Decision**: Single project structure (iris-vector-rag framework package). This feature extends the existing configuration management system without adding new top-level components. Primary changes are in iris_vector_rag/config/ module with supporting changes in core/connection.py and cli/init_tables.py.

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
- [x] Phase 0: Research complete (/plan command) - research.md created
- [x] Phase 1: Design complete (/plan command) - data-model.md, contracts/, quickstart.md created
- [x] Phase 2: Task planning complete (/plan command - approach described below)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS (re-evaluated, no new violations)
- [x] All NEEDS CLARIFICATION resolved (zero markers in Technical Context)
- [x] Complexity deviations documented (none - feature complies with all principles)

**Artifacts Generated**:
- ✅ `/specs/058-number-058-short/plan.md` (this file)
- ✅ `/specs/058-number-058-short/research.md` (9 research areas, 420 lines)
- ✅ `/specs/058-number-058-short/data-model.md` (4 core entities, 2 validators, 360 lines)
- ✅ `/specs/058-number-058-short/contracts/test_cloud_config_contract.py` (22 contract tests, 460 lines)
- ✅ `/specs/058-number-058-short/quickstart.md` (Cloud deployment guide, 380 lines)
- ✅ `CLAUDE.md` updated with configuration patterns

**Next Command**: `/speckit.tasks` to generate tasks.md from design artifacts

---
*Based on Constitution v1.8.0 - See `/.specify/memory/constitution.md`*
