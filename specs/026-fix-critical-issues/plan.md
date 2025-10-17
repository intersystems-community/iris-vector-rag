# Implementation Plan: Fix Critical Issues from Analyze Command

**Branch**: `026-fix-critical-issues` | **Date**: 2025-10-04 | **Spec**: [spec.md](./spec.md)
**Input**: Feature specification from `/specs/026-fix-critical-issues/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path ✅
   → Spec found at /Users/intersystems-community/ws/rag-templates/specs/026-fix-critical-issues/spec.md
2. Fill Technical Context (scan for NEEDS CLARIFICATION) ✅
   → Detect Project Type: Python testing framework (pytest-based)
   → Structure Decision: Single project (tests/ directory structure)
3. Fill Constitution Check section ✅
   → Based on constitution v1.6.0
4. Evaluate Constitution Check section ✅
   → Violations: None (testing framework improvements align with all principles)
   → Update Progress Tracking: Initial Constitution Check PASS
5. Execute Phase 0 → research.md ✅
   → Coverage warning approaches, error message standards, TDD validation
6. Execute Phase 1 → contracts, data-model.md, quickstart.md ✅
   → Coverage contracts, error message contracts, TDD validation contracts
7. Re-evaluate Constitution Check section ✅
   → No new violations
   → Update Progress Tracking: Post-Design Constitution Check PASS
8. Plan Phase 2 → Describe task generation approach ✅
9. STOP - Ready for /tasks command ✅
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary

**Primary Requirement**: Add missing compliance checks identified by the /analyze command: coverage threshold warnings, error message validation, and TDD contract test verification to ensure the testing framework follows constitutional principles.

**Technical Approach**:
1. Implement pytest hooks for coverage threshold warnings (60% overall, 80% critical)
2. Create error message validation framework with three-part structure
3. Build TDD compliance validator using git history analysis
4. Add requirement-to-task mapping validation
5. Ensure all edge cases have corresponding tests

## Technical Context

**Language/Version**: Python 3.12 (same as existing framework)
**Primary Dependencies**: pytest 8.4.1, pytest-cov 6.1.1, coverage 7.6.1, GitPython 3.1.43
**Storage**: Coverage data (.coverage files), Git repository (history analysis)
**Testing**: pytest with custom hooks and plugins
**Target Platform**: macOS Darwin (development), Linux (CI/CD)
**Project Type**: Single project (testing framework enhancements)
**Performance Goals**:
  - Coverage calculation < 5 seconds overhead
  - TDD validation < 10 seconds for full history scan
  - Error message validation near-zero overhead
**Constraints**:
  - Must not break existing 206 passing tests
  - Must integrate with existing pytest workflow
  - Must follow pytest plugin architecture
**Scale/Scope**:
  - 21 implementation tasks in tasks.md (originally estimated ~20-25)
  - 17 functional requirements + 3 non-functional requirements to track
  - 3 compliance areas (coverage, errors, TDD)
  - 4 contract specifications

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**I. Framework-First Architecture**: ✓ Components are pytest plugins/hooks | ✓ No application-specific logic | ✓ CLI interface via pytest commands

**II. Pipeline Validation & Requirements**: ✓ Automated validation for all compliance checks | ✓ Validation procedures idempotent

**III. Test-Driven Development**: ✓ Contract tests for validators written first | ✓ Validators themselves follow TDD

**IV. Performance & Enterprise Scale**: ✓ Minimal overhead on test execution | ✓ Scales with codebase size

**V. Production Readiness**: ✓ Structured logging for violations | ✓ Health checks for validator status | ✓ Works in Docker environments

**VI. Explicit Error Handling**: ✓ No silent validation failures | ✓ Clear violation messages | ✓ Actionable remediation steps

**VII. Standardized Database Interfaces**: N/A - No database interactions in this feature

**GATE RESULT**: ✅ PASS - All constitutional principles satisfied

## Project Structure

### Documentation (this feature)
```
specs/026-fix-critical-issues/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── data-model.md        # Phase 1 output (/plan command)
├── quickstart.md        # Phase 1 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
│   ├── coverage_warning_contract.md
│   ├── error_message_contract.md
│   ├── tdd_validation_contract.md
│   └── task_mapping_contract.md
└── tasks.md             # Phase 2 output (/tasks command - NOT created by /plan)
```

### Source Code (repository root)
```
tests/
├── plugins/             # NEW: Pytest plugins
│   ├── __init__.py
│   ├── coverage_warnings.py
│   ├── error_message_validator.py
│   └── tdd_compliance.py
├── contract/            # Contract tests for validators
│   ├── test_coverage_warning_contract.py
│   ├── test_error_message_contract.py
│   ├── test_tdd_validation_contract.py
│   └── test_task_mapping_contract.py
├── unit/
└── e2e/

scripts/
├── validate_tdd_compliance.py  # Standalone TDD validator
└── validate_task_mapping.py    # Requirement-to-task mapper

.coveragerc                     # Updated with critical module patterns
pytest.ini                      # Updated with new plugin registrations
```

**Structure Decision**: Single project structure with new `tests/plugins/` directory for pytest plugins. Validators implemented as pytest hooks and standalone scripts for flexibility. Contract tests validate the validators themselves follow TDD.

## Phase 0: Outline & Research

### Research Tasks

1. **Pytest Hook Architecture for Coverage Warnings**
   - Research: How to implement pytest hooks that run after coverage collection
   - Research: Best practices for non-failing warnings in pytest output
   - Research: Integration with pytest-cov plugin architecture

2. **Error Message Validation Patterns**
   - Research: Automated validation of error message quality
   - Research: Three-part error message structure standards
   - Research: pytest assertion rewriting for better messages

3. **TDD Compliance Validation Approaches**
   - Research: Git history analysis for test state detection
   - Research: Identifying contract test patterns in codebase
   - Research: CI/CD integration strategies for TDD validation

4. **Critical Module Detection Strategies**
   - Research: Configuration vs convention for critical modules
   - Research: pytest marker strategies for critical code paths
   - Research: Coverage threshold customization patterns

5. **Task Mapping Validation**
   - Research: Requirement ID extraction from markdown
   - Research: Task-to-requirement linking strategies
   - Research: Gap analysis reporting formats

### Research Output

See [research.md](./research.md) for detailed findings, decisions, rationales, and alternatives considered for each research task.

**Key Decisions**:
- Use pytest_terminal_summary hook for coverage warnings
- Implement error validators as pytest plugin with configurable rules
- Use GitPython for TDD compliance checking with commit analysis
- Critical modules defined by directory patterns in .coveragerc
- Task mapping uses regex-based requirement extraction

## Phase 1: Design & Contracts

### Data Model (compliance entities)

See [data-model.md](./data-model.md) for complete entity definitions.

**Key Entities**:
1. **CoverageWarning**: module_path, current_coverage, threshold, severity, timestamp
2. **ErrorMessageValidation**: test_name, error_message, validation_components, status
3. **ContractTestState**: test_file, initial_state, implementation_commit, compliance
4. **RequirementTaskMapping**: requirement_id, task_ids, coverage_status
5. **CriticalModuleConfig**: module_pattern, threshold, rationale

### API Contracts

See [contracts/](./contracts/) directory for detailed contract specifications.

**Contract Files**:
1. `coverage_warning_contract.md`: Pytest hook behavior for coverage warnings
2. `error_message_contract.md`: Error message validation rules and format
3. `tdd_validation_contract.md`: Git-based TDD compliance checking
4. `task_mapping_contract.md`: Requirement-to-task validation logic

### Contract Tests

Contract tests validate that our validators work correctly:
1. Test coverage warnings trigger at correct thresholds
2. Test error messages are properly validated
3. Test TDD compliance detection works with git history
4. Test requirement mapping finds all gaps

### Quickstart

See [quickstart.md](./quickstart.md) for developer guide.

**Quickstart validates**:
1. Developer can see coverage warnings without test failures
2. Developer gets feedback on error message quality
3. Developer can verify TDD compliance before PR
4. Developer can validate all requirements have tasks

### Agent Context Update

Execute: `.specify/scripts/bash/update-agent-context.sh claude`

Updates CLAUDE.md with:
- New pytest plugin architecture
- Coverage warning thresholds
- Error message validation rules
- TDD compliance checking procedures
- Task mapping validation

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

### Task Generation Strategy

**Generate tasks from Phase 1 design docs**:

1. **From coverage_warning_contract.md**:
   - Task: Create pytest plugin for coverage warnings
   - Task: Implement pytest_terminal_summary hook
   - Task: Add .coveragerc critical module patterns
   - Task: Create contract test for coverage warnings

2. **From error_message_contract.md**:
   - Task: Create error message validator plugin
   - Task: Implement three-part validation rules
   - Task: Add pytest hook for assertion messages
   - Task: Create contract test for error validation

3. **From tdd_validation_contract.md**:
   - Task: Create TDD compliance validator script
   - Task: Implement git history analysis
   - Task: Add pre-commit hook option
   - Task: Create contract test for TDD validation

4. **From task_mapping_contract.md**:
   - Task: Create requirement extraction script
   - Task: Implement task mapping validation
   - Task: Add gap reporting
   - Task: Create contract test for mapping

5. **From quickstart.md**:
   - Task: Create example test with violations
   - Task: Document plugin usage
   - Task: Add CI/CD integration examples

### Ordering Strategy

**TDD order** (tests before implementation):
1. Contract tests first (all fail initially)
2. Plugin infrastructure setup
3. Core validation logic
4. Integration with pytest
5. Documentation and examples

**Dependency order**:
1. Base plugin architecture
2. Individual validators (can be parallel [P])
3. Integration hooks
4. Standalone scripts
5. CI/CD integration

**Parallel execution** [P]:
- Coverage, error, and TDD validators are independent
- Contract tests can run in parallel
- Documentation can be written alongside code

### Estimated Output

**20-25 numbered, ordered tasks** in tasks.md:
- 4 contract test tasks (one per validator)
- 4 plugin implementation tasks
- 4 validation logic tasks
- 2 standalone script tasks
- 3 configuration tasks
- 2 documentation tasks
- 1 CI/CD integration task

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)
**Phase 4**: Implementation (execute tasks.md following constitutional principles)
**Phase 5**: Validation (run tests, execute quickstart.md, performance validation)

## Complexity Tracking
*Fill ONLY if Constitution Check has violations that must be justified*

No violations - all changes align with constitutional principles.

## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning complete (/plan command - describe approach only)
- [x] Phase 3: Tasks generated (/tasks command) - 21 tasks in tasks.md
- [x] Phase 4: Implementation complete - All 21 tasks executed
- [x] Phase 5: Validation passed - Contract tests passing, plugins functional

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved (used reasonable defaults)
- [x] Complexity deviations documented (none)

---
*Based on Constitution v1.6.0 - See `.specify/memory/constitution.md`*