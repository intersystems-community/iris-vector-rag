
# Implementation Plan: Fix iris-vector-rag Entity Types Configuration Bug

**Branch**: `062-fix-iris-vector` | **Date**: 2025-01-16 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/062-fix-iris-vector/spec.md`

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

**Primary Requirement**: Fix entity extraction service to honor configured entity types from YAML configuration instead of defaulting to healthcare-specific types (USER, MODULE, VERSION). Currently, `EntityExtractionService.extract_batch_with_dspy()` does not accept or pass the `entity_types` parameter to `TrakCareEntityExtractionModule`, causing configured types (PERSON, TITLE, LOCATION) to be ignored.

**Technical Approach**:
1. Add `entity_types` parameter to `EntityExtractionService.extract_batch_with_dspy()` method
2. Read entity types from configuration when parameter is None
3. Pass entity types to `TrakCareEntityExtractionModule.forward()` method
4. Maintain backward compatibility with existing code that doesn't specify entity types
5. Use domain-neutral defaults (PERSON, ORGANIZATION, LOCATION) instead of healthcare-specific defaults

## Technical Context
**Language/Version**: Python 3.12 (iris-vector-rag 0.5.4)
**Primary Dependencies**: DSPy, iris-vector-rag, YAML configuration
**Storage**: InterSystems IRIS Database (RAG.Entities table)
**Testing**: pytest with contract tests and integration tests
**Target Platform**: iris-vector-rag framework package (multi-platform: Linux, macOS, Windows)
**Project Type**: Single project (Python package framework)
**Performance Goals**: No performance degradation from configuration changes (NFR-002)
**Constraints**:
- Must maintain backward compatibility (FR-005)
- Configuration changes must not require code modifications (NFR-001)
- Must work with existing HippoRAG pipeline without changes
**Scale/Scope**:
- Fix applies to all entity extraction operations
- Affects HotpotQA question answering (currently F1=0.000 for bridge questions)
- Impacts all users of iris-vector-rag entity extraction

**Bug Context** (from original report):
- **Current Behavior**: `TrakCareEntityExtractionModule` defaults to healthcare types [USER, ORGANIZATION, PRODUCT, MODULE, VERSION]
- **Root Cause**: `extract_batch_with_dspy()` lacks `entity_types` parameter, cannot pass config to module
- **Files Affected**:
  - `iris_vector_rag/services/entity_extraction.py:880` - missing parameter
  - `iris_vector_rag/dspy_modules/entity_extraction_module.py:93` - accepts entity_types but never receives them
- **Test Evidence**: HotpotQA Q2 fails because "Chief of Protocol" (TITLE type) not extracted

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**I. Framework-First Architecture**:
- ✓ Component is framework element (EntityExtractionService)
- ✓ No application-specific logic (pure bug fix in framework)
- ✓ Existing CLI interface preserved

**II. Pipeline Validation & Requirements**:
- ✓ Fix maintains existing validation patterns
- ✓ Configuration reading idempotent
- ✓ No setup procedure changes

**III. Test-Driven Development**:
- ✓ Contract tests required before implementation
- N/A Performance tests (bug fix, not new feature)
- ✓ Integration tests against real IRIS database required

**IV. Performance & Enterprise Scale**:
- N/A Incremental indexing (not affected by this fix)
- N/A IRIS vector operations (not affected by this fix)
- ✓ No performance degradation (NFR-002)

**V. Production Readiness**:
- ✓ Logging preserved (no changes to logging)
- N/A Health checks (not affected by configuration bug fix)
- ✓ Docker deployment unchanged

**VI. Explicit Error Handling**:
- ✓ Must add validation for invalid entity types (FR-006)
- ✓ Clear error messages required (NFR-003)
- ✓ No silent failures when config invalid

**VII. Standardized Database Interfaces**:
- N/A No database interface changes
- ✓ Configuration reading uses existing patterns
- ✓ No new IRIS queries required

**Initial Assessment**: ✅ PASS - Bug fix follows constitutional principles

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
├── services/
│   └── entity_extraction.py          # FIX: Add entity_types parameter to extract_batch_with_dspy()
├── dspy_modules/
│   └── entity_extraction_module.py   # VERIFY: Already accepts entity_types parameter
└── config/
    └── manager.py                     # READ: Configuration reading (no changes needed)

tests/
├── contract/
│   └── test_entity_types_config.py   # NEW: Contract tests for entity_types configuration
├── integration/
│   └── test_entity_types_integration.py  # NEW: Integration tests against IRIS
└── unit/
    └── test_entity_extraction_service.py  # UPDATE: Add unit tests for new parameter

config/
└── hipporag2.yaml                     # REFERENCE: Example config with entity_types
```

**Structure Decision**: Single project (iris-vector-rag Python package). This is a bug fix to the existing EntityExtractionService, not a new component. Changes are minimal and localized to entity extraction service layer.

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
- Each contract rule → contract test task (8 tests from CR-001 to CR-008)
- Implementation tasks to satisfy contracts
- Integration test tasks for IRIS database validation
- Validation tasks for HotpotQA Question 2 fix

**Ordering Strategy** (TDD - Tests First):
1. Contract tests (MUST fail initially)
   - CR-001: Parameter acceptance test
   - CR-002: Config fallback test
   - CR-003: Default fallback test
   - CR-004: Empty list validation test
   - CR-005: Type filtering test
   - CR-006: Unknown type warning test
   - CR-007: Module integration test
   - CR-008: Backward compatibility test

2. Implementation tasks (make tests pass)
   - Add entity_types parameter to extract_batch_with_dspy()
   - Implement config reading logic
   - Implement validation logic
   - Implement entity_types threading to module

3. Integration tests (IRIS database)
   - Test with real IRIS database
   - Verify only configured types stored
   - Test HotpotQA Question 2 scenario

4. Validation tasks
   - Run quickstart guide
   - Verify F1 score improvement
   - Verify no regressions

**Estimated Output**: ~15-20 numbered, ordered tasks in tasks.md

**Key Tasks** (preview):
1. [P] Create contract test file: test_entity_types_config.py
2. [P] Write CR-001: Parameter acceptance test (MUST FAIL)
3. [P] Write CR-002: Config fallback test (MUST FAIL)
4. [P] Write CR-003: Default fallback test (MUST FAIL)
5. [P] Write CR-004: Empty list validation test (MUST FAIL)
6. [P] Write CR-005: Type filtering test (MUST FAIL)
7. [P] Write CR-006: Unknown type warning test (MUST FAIL)
8. [P] Write CR-007: Module integration test (MUST FAIL)
9. [P] Write CR-008: Backward compatibility test (MUST FAIL)
10. Update EntityExtractionService.extract_batch_with_dspy() signature
11. Implement entity_types parameter reading from config
12. Implement validation (empty list, unknown types)
13. Thread entity_types to TrakCareEntityExtractionModule.forward()
14. Run contract tests (all should PASS now)
15. [P] Create integration test: test_entity_types_integration.py
16. Test with IRIS database (configured types only)
17. Test HotpotQA Question 2 (verify Chief of Protocol extracted)
18. Run quickstart validation script
19. Verify F1 score > 0.0 for Question 2
20. Final regression testing

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
  - ✅ research.md created with all decisions documented
  - ✅ No NEEDS CLARIFICATION remaining
  - ✅ Technical approach validated
- [x] Phase 1: Design complete (/plan command)
  - ✅ data-model.md created
  - ✅ contracts/entity_types_api_contract.md created (8 contract rules)
  - ✅ quickstart.md created with validation script
  - ✅ CLAUDE.md updated with feature context
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
  - ✅ Task generation strategy documented
  - ✅ TDD ordering strategy defined (tests first)
  - ✅ 20 tasks previewed
- [ ] Phase 3: Tasks generated (/tasks command) - **NEXT STEP**
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS (no violations)
- [x] Post-Design Constitution Check: PASS (no new violations)
- [x] All NEEDS CLARIFICATION resolved
- [x] Complexity deviations documented (none - simple bug fix)

**Artifacts Generated**:
- ✅ /specs/062-fix-iris-vector/plan.md (this file)
- ✅ /specs/062-fix-iris-vector/research.md
- ✅ /specs/062-fix-iris-vector/data-model.md
- ✅ /specs/062-fix-iris-vector/contracts/entity_types_api_contract.md
- ✅ /specs/062-fix-iris-vector/quickstart.md
- ✅ CLAUDE.md (updated)

---
*Based on Constitution v1.2.0 - See `/.specify/memory/constitution.md`*
