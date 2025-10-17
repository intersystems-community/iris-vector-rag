# Feature Specification: Fix RAGAS Make Target Pipeline List

**Feature Branch**: `031-fix-make-target`
**Created**: 2025-10-06
**Status**: Draft
**Input**: User description: "fix make target test-ragas-sample to pick up the latest list of pipelines"

## Execution Flow (main)
```
1. Parse user description from Input
   → Feature is about make target automatically using current available pipelines
2. Extract key concepts from description
   → Actors: Developers running RAGAS evaluations
   → Actions: Execute make test-ragas-sample, dynamically detect available pipelines
   → Data: Pipeline list from iris_rag factory
   → Constraints: Must work without manual Makefile updates
3. For each unclear aspect:
   → [RESOLVED] All aspects clear from context
4. Fill User Scenarios & Testing section
   → User runs make test-ragas-sample and all available pipelines are tested
5. Generate Functional Requirements
   → All requirements testable via make execution and RAGAS output
6. Identify Key Entities
   → Pipeline types (data entities from factory)
7. Run Review Checklist
   → No [NEEDS CLARIFICATION] markers
   → No implementation details in requirements
8. Return: SUCCESS (spec ready for planning)
```

---

## ⚡ Quick Guidelines
- ✅ Focus on WHAT users need and WHY
- ❌ Avoid HOW to implement (no tech stack, APIs, code structure)
- 👥 Written for business stakeholders, not developers

---

## User Scenarios & Testing *(mandatory)*

### Primary User Story
As a developer running RAGAS evaluations, when I execute `make test-ragas-sample`, the system should automatically test all currently available RAG pipelines without requiring me to manually update the Makefile whenever pipelines are added or removed.

**Current Problem**: The Makefile hardcodes a pipeline list that becomes outdated when:
- New pipelines are added to the factory
- Pipeline names change
- Pipelines are temporarily disabled
- This leads to failed evaluations or missing coverage

**Desired Outcome**: The make target queries the iris_rag factory for the current list of available pipelines and uses that list dynamically.

### Acceptance Scenarios

1. **Given** a developer has just added a new pipeline to iris_rag factory, **When** they run `make test-ragas-sample`, **Then** the new pipeline is automatically included in the RAGAS evaluation without modifying the Makefile

2. **Given** the iris_rag factory has 5 available pipelines, **When** a developer runs `make test-ragas-sample`, **Then** all 5 pipelines are tested and results appear in the RAGAS report

3. **Given** a pipeline name has been changed in the factory, **When** a developer runs `make test-ragas-sample`, **Then** the evaluation uses the new name automatically without Makefile changes

4. **Given** a pipeline has been temporarily removed from the factory, **When** a developer runs `make test-ragas-sample`, **Then** the evaluation skips that pipeline without errors

### Edge Cases
- What happens when the factory has zero available pipelines? (Should fail with clear error message)
- What happens when a pipeline is listed but fails to initialize? (Should report failure for that specific pipeline, continue with others)
- What happens if a user manually overrides with `RAGAS_PIPELINES` env var? (Should respect user override)

## Requirements *(mandatory)*

### Functional Requirements

- **FR-001**: System MUST dynamically query iris_rag factory for the current list of available pipeline types at evaluation time

- **FR-002**: System MUST use the factory's pipeline list as the default when `RAGAS_PIPELINES` environment variable is not set

- **FR-003**: System MUST preserve user's ability to override the pipeline list via `RAGAS_PIPELINES` environment variable

- **FR-004**: System MUST fail with a clear error message if no pipelines are available from the factory

- **FR-005**: System MUST test all available pipelines when developer runs `make test-ragas-sample` without environment variable overrides

- **FR-006**: System MUST NOT require Makefile modifications when new pipelines are added to the iris_rag factory

- **FR-007**: System MUST report which pipelines were tested in the evaluation output

- **FR-008**: System MUST handle pipeline initialization failures gracefully, continuing to test remaining pipelines

### Key Entities *(include if feature involves data)*

- **Pipeline Type**: A string identifier representing an available RAG pipeline (e.g., "basic", "crag", "graphrag")
  - Attributes: name, availability status
  - Source: iris_rag factory's available_types list
  - Lifecycle: Defined at factory level, queried at runtime

- **RAGAS Evaluation**: A test execution that evaluates multiple pipelines
  - Attributes: pipelines tested, results per pipeline, timestamp
  - Relationships: Tests multiple Pipeline Types

---

## Review & Acceptance Checklist
*GATE: Automated checks run during main() execution*

### Content Quality
- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

### Requirement Completeness
- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable (pipeline count in RAGAS output)
- [x] Scope is clearly bounded (only test-ragas-sample target)
- [x] Dependencies and assumptions identified (iris_rag factory must expose available types)

---

## Execution Status
*Updated by main() during processing*

- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked (none found)
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [x] Review checklist passed

---

## Additional Context

### Current State
The Makefile currently hardcodes:
```
export RAGAS_PIPELINES=${RAGAS_PIPELINES:-"basic,basic_rerank,crag,graphrag,pylate_colbert"};
```

This list becomes stale when:
- The factory adds/removes pipelines
- Pipeline names change
- Pipelines are refactored

### Business Impact
**Problem**: Developers waste time manually syncing Makefile with factory changes, leading to:
- Missed test coverage (new pipelines not evaluated)
- Broken CI/CD (references to removed pipelines)
- Developer friction (extra manual step for every pipeline change)

**Value**: Automatic pipeline discovery reduces maintenance overhead and ensures complete test coverage without manual coordination.

### Success Metrics
- Zero Makefile updates required when adding new pipelines
- RAGAS evaluation coverage matches factory availability (100%)
- Clear error messages when pipeline count is unexpected
