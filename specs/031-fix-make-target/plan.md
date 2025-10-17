# Implementation Plan: Fix RAGAS Make Target Pipeline List

**Branch**: `031-fix-make-target` | **Date**: 2025-10-06 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/031-fix-make-target/spec.md`

## Execution Flow (/plan command scope)
```
1. Load feature spec from Input path ✓
   → Spec loaded, no errors
2. Fill Technical Context ✓
   → Project Type: Single Python package (iris_rag framework)
   → Structure Decision: Framework-based, extends existing Makefile infrastructure
3. Fill the Constitution Check section ✓
4. Evaluate Constitution Check section
   → STATUS: PASS with notes (N/A for most - this is Makefile/tooling, not pipeline)
   → Update Progress Tracking: Initial Constitution Check
5. Execute Phase 0 → research.md
   → Research factory introspection patterns
6. Execute Phase 1 → contracts, data-model.md, quickstart.md
   → Generate helper script contract
   → Update Makefile to call helper
7. Re-evaluate Constitution Check section
   → STATUS: PASS (no new violations)
   → Update Progress Tracking: Post-Design Constitution Check
8. Plan Phase 2 → Describe task generation approach
9. STOP - Ready for /tasks command
```

**IMPORTANT**: The /plan command STOPS at step 7. Phases 2-4 are executed by other commands:
- Phase 2: /tasks command creates tasks.md
- Phase 3-4: Implementation execution (manual or via tools)

## Summary
Fix the `make test-ragas-sample` target to dynamically query the iris_rag factory for available pipeline types instead of hardcoding them. This eliminates the need for manual Makefile updates when pipelines are added, removed, or renamed. The solution will create a helper script that queries the factory and outputs a comma-separated pipeline list, which the Makefile will invoke before running RAGAS evaluations.

**Technical Approach**: Create a lightweight Python helper script (`scripts/utils/get_pipeline_types.py`) that imports the iris_rag factory and outputs available pipeline types. The Makefile will execute this script and use the output to set `RAGAS_PIPELINES` dynamically, while still respecting user overrides via environment variables.

## Technical Context
**Language/Version**: Python 3.11+ (matches existing pyproject.toml)
**Primary Dependencies**:
- iris_rag package (internal - the factory we're querying)
- Make (build orchestration)
- No external dependencies needed

**Storage**: N/A (no data persistence)
**Testing**: pytest (existing framework infrastructure tests in tests/infrastructure/)
**Target Platform**: Developer workstations (macOS, Linux) running make commands
**Project Type**: Single Python package with Makefile build system
**Performance Goals**: Helper script execution <100ms (startup + import + output)
**Constraints**:
- Must preserve backward compatibility with RAGAS_PIPELINES env var override
- Must not break existing make targets
- Must fail clearly if no pipelines available
**Scale/Scope**:
- Single helper script (~50 lines)
- Single Makefile modification (2-3 lines in test-ragas-sample target)
- 1 infrastructure test added

## Constitution Check
*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**I. Framework-First Architecture**:
- ✓ Component is a utility script (not a pipeline, no RAGPipeline extension needed)
- ✓ No application-specific logic (pure factory introspection)
- ✓ CLI interface exposed (script outputs to stdout for make consumption)

**II. Pipeline Validation & Requirements**:
- N/A - This is tooling for the build system, not a pipeline

**III. Test-Driven Development**:
- ✓ Infrastructure test will be written in tests/infrastructure/test_makefile_targets.py
- N/A - No database operations (pure Python introspection)

**IV. Performance & Enterprise Scale**:
- N/A - This is a development tool, not a runtime component

**V. Production Readiness**:
- ✓ Clear error messages (script will exit with status code and message on failure)
- N/A - Not deployed to production (development tooling only)

**VI. Explicit Error Handling**:
- ✓ No silent failures (script will exit 1 with clear error if factory unavailable)
- ✓ Clear exception messages (import failures, no pipelines found)
- ✓ Actionable error context (tells user what's wrong and how to fix)

**VII. Standardized Database Interfaces**:
- N/A - No database operations

**Constitutional Compliance**: PASS (8 checks applicable, 0 violations)

## Project Structure

### Documentation (this feature)
```
specs/031-fix-make-target/
├── plan.md              # This file (/plan command output)
├── research.md          # Phase 0 output (/plan command)
├── contracts/           # Phase 1 output (/plan command)
│   └── helper_script_contract.md
└── quickstart.md        # Phase 1 output (/plan command)
```

### Source Code (repository root)
```
# Single project structure (existing framework)
scripts/
├── utils/
│   └── get_pipeline_types.py  # NEW: Factory introspection helper

Makefile  # MODIFIED: test-ragas-sample target

tests/
└── infrastructure/
    └── test_makefile_targets.py  # MODIFIED: Add dynamic pipeline test
```

**Structure Decision**: Single project structure. This is a utility enhancement to the existing iris_rag framework's build system. The helper script goes in `scripts/utils/` alongside other build utilities, and the Makefile modification is in-place.

## Phase 0: Outline & Research
**Status**: ✓ Complete

### Research Questions
1. **How to safely import iris_rag factory without full initialization?**
   - Decision: Import only the factory module, not create pipelines
   - Rationale: Avoid database connections, just need the available_types list
   - Implementation: `from iris_rag import _create_pipeline_legacy` and extract available_types

2. **What's the current factory interface for listing pipelines?**
   - Finding: Available types are hardcoded in `_create_pipeline_legacy` function (lines 152-158 of iris_rag/__init__.py)
   - Current list: `["basic", "basic_rerank", "crag", "graphrag", "pylate_colbert"]`
   - Extraction strategy: Parse the function source or hardcode list (discuss trade-offs)

3. **How should the helper script output results?**
   - Decision: Simple stdout with comma-separated values
   - Rationale: Easy for Make to capture with `$(shell ...)`
   - Format: `basic,basic_rerank,crag,graphrag,pylate_colbert`

4. **What error conditions must be handled?**
   - iris_rag package not importable → exit 1 with clear message
   - No pipelines available → exit 1 (something is very wrong)
   - Any other exception → exit 1 with traceback for debugging

5. **How to preserve env var override behavior?**
   - Makefile pattern: `${RAGAS_PIPELINES:-$(shell python scripts/utils/get_pipeline_types.py)}`
   - If RAGAS_PIPELINES set → use it
   - If not set → run helper script and use output

### Research Output
See [research.md](research.md) for detailed findings and alternatives considered.

**Key Decisions**:
- **Factory Introspection**: Parse available_types from factory function source
- **Script Location**: `scripts/utils/get_pipeline_types.py` (consistent with existing utilities)
- **Output Format**: Comma-separated list to stdout
- **Error Handling**: Exit codes + stderr messages
- **Makefile Integration**: Shell command substitution with fallback pattern

## Phase 1: Design & Contracts
**Status**: ✓ Complete

### Data Model
**Entity**: PipelineType (read-only, factory-defined)
- **Attributes**:
  - name: string (e.g., "basic", "graphrag")
  - defined: boolean (always true if in list)
- **Source**: iris_rag.__init__._create_pipeline_legacy function
- **Access Pattern**: Import module → extract available_types list
- **No persistence** (runtime introspection only)

See [data-model.md](data-model.md) for complete entity specifications.

### API Contracts

**Contract 1: Helper Script CLI Interface**
```
COMMAND: python scripts/utils/get_pipeline_types.py
STDOUT: "basic,basic_rerank,crag,graphrag,pylate_colbert"
STDERR: (empty on success)
EXIT CODE: 0
```

**Contract 2: Helper Script Error Cases**
```
CASE: iris_rag not importable
STDOUT: (empty)
STDERR: "ERROR: Cannot import iris_rag. Is the package installed?"
EXIT CODE: 1

CASE: No pipelines available
STDOUT: (empty)
STDERR: "ERROR: No pipeline types available from factory"
EXIT CODE: 1
```

**Contract 3: Makefile Integration**
```makefile
# Before (hardcoded)
export RAGAS_PIPELINES=${RAGAS_PIPELINES:-"basic,basic_rerank,crag,graphrag,pylate_colbert"};

# After (dynamic with fallback)
export RAGAS_PIPELINES=$${RAGAS_PIPELINES:-$$(python scripts/utils/get_pipeline_types.py)};
```

See [contracts/](contracts/) for complete specifications and test scenarios.

### Testing Strategy

**Contract Tests** (tests/infrastructure/test_makefile_targets.py):
1. `test_get_pipeline_types_script_exists()` - Verify script file exists
2. `test_get_pipeline_types_output_format()` - Verify comma-separated output
3. `test_get_pipeline_types_matches_factory()` - Verify output matches factory
4. `test_ragas_target_uses_dynamic_pipelines()` - Verify Makefile calls helper
5. `test_ragas_target_respects_env_override()` - Verify env var precedence

**Integration Tests**:
1. Run `make test-ragas-sample` and verify all factory pipelines tested
2. Set `RAGAS_PIPELINES=basic,crag` and verify only those 2 tested
3. Rename a pipeline in factory and verify make target uses new name

See [quickstart.md](quickstart.md) for test execution guide.

### Agent Context Update
Running update script to add this feature to CLAUDE.md...

## Phase 2: Task Planning Approach
*This section describes what the /tasks command will do - DO NOT execute during /plan*

**Task Generation Strategy**:
- Load `.specify/templates/tasks-template.md` as base
- Generate tasks in TDD order (tests first, then implementation)
- Each contract → contract test task [P]
- Helper script implementation → single task
- Makefile modification → single task
- Integration verification → single task

**Ordering Strategy**:
1. **Contract Tests** [P] - Can be written in parallel:
   - Task 1: test_get_pipeline_types_script_exists
   - Task 2: test_get_pipeline_types_output_format
   - Task 3: test_get_pipeline_types_matches_factory
   - Task 4: test_ragas_target_uses_dynamic_pipelines
   - Task 5: test_ragas_target_respects_env_override

2. **Implementation** (sequential, depends on tests):
   - Task 6: Implement scripts/utils/get_pipeline_types.py
   - Task 7: Modify Makefile test-ragas-sample target

3. **Verification** (depends on implementation):
   - Task 8: Run all contract tests (should pass)
   - Task 9: Run make test-ragas-sample (integration test)
   - Task 10: Documentation update (README.md if needed)

**Estimated Output**: 10 numbered tasks in dependency order

**IMPORTANT**: This phase is executed by the /tasks command, NOT by /plan

## Phase 3+: Future Implementation
*These phases are beyond the scope of the /plan command*

**Phase 3**: Task execution (/tasks command creates tasks.md)
**Phase 4**: Implementation (execute tasks.md following constitutional principles)
**Phase 5**: Validation (run tests, execute quickstart.md, verify all 5 pipelines tested)

## Complexity Tracking
*No constitutional violations requiring justification*

This feature is purely additive tooling that enhances the build system. It introduces no architectural complexity, no new dependencies, and follows all constitutional principles.

## Progress Tracking
*This checklist is updated during execution flow*

**Phase Status**:
- [x] Phase 0: Research complete (/plan command)
- [x] Phase 1: Design complete (/plan command)
- [x] Phase 2: Task planning approach described (/plan command)
- [ ] Phase 3: Tasks generated (/tasks command)
- [ ] Phase 4: Implementation complete
- [ ] Phase 5: Validation passed

**Gate Status**:
- [x] Initial Constitution Check: PASS
- [x] Post-Design Constitution Check: PASS
- [x] All NEEDS CLARIFICATION resolved
- [x] Complexity deviations documented (N/A - none)

---
*Based on Constitution v1.6.0 - See `/.specify/memory/constitution.md`*
