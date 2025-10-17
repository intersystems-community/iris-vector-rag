# Feature Specification: Configurable Test Backend Modes (Enterprise & Community)

**Feature Branch**: `035-make-2-modes`
**Created**: 2025-10-08
**Status**: Draft
**Input**: User description: "make 2 modes of back ends configurabble test options - enterprise and community - and use ../iris-devtools as appropriate to help manage the IRIS database state"

## Execution Flow (main)
```
1. Parse user description from Input
   → Feature requires configurable test backend modes
2. Extract key concepts from description
   → Actors: Test runners, developers
   → Actions: Configure backend mode, manage database state
   → Data: IRIS database configuration, connection limits
   → Constraints: Community edition license limits vs enterprise
3. For each unclear aspect:
   → [NEEDS CLARIFICATION: What specific database state management operations?]
   → [NEEDS CLARIFICATION: How should mode selection be configured (env var, config file, pytest marker)?]
   → [NEEDS CLARIFICATION: Should iris-devtools be optional or required dependency?]
4. Fill User Scenarios & Testing section
   → User flow: Developer selects backend mode → Tests run with appropriate configuration
5. Generate Functional Requirements
   → Each requirement must be testable
6. Identify Key Entities
   → Backend configuration, database state manager
7. Run Review Checklist
   → Spec has clarification needs marked
8. Return: SUCCESS (spec ready for planning after clarification)
```

---

## ⚡ Quick Guidelines
- ✅ Focus on WHAT users need and WHY
- ❌ Avoid HOW to implement (no tech stack, APIs, code structure)
- 👥 Written for business stakeholders, not developers

---

## Clarifications

### Session 2025-10-08

- Q: How should users configure the backend mode? → A: Combination - Config file default + env var override
- Q: What should happen when iris-devtools is not available at ../iris-devtools? → A: Tests fail immediately with clear error message (iris-devtools is a required dev dependency)
- Q: What specific database state management operations should iris-devtools provide? → A: Container lifecycle management, schema reset, connection validation, and health checks
- Q: What should happen when IRIS edition doesn't match the configured backend mode? → A: Fail immediately with error requiring manual correction
- Q: What is the maximum concurrent connection limit for community mode? → A: 1 connection

---

## User Scenarios & Testing

### Primary User Story
As a developer running tests, I need to select between enterprise and community backend modes so that tests can execute with appropriate database connection settings and avoid license exhaustion issues specific to community edition.

### Acceptance Scenarios

1. **Given** a test suite configured for community mode, **When** tests execute, **Then** they should use connection pooling and sequential execution strategies appropriate for community edition license limits

2. **Given** a test suite configured for enterprise mode, **When** tests execute, **Then** they should use parallel execution and unlimited connections without artificial throttling

3. **Given** iris-devtools is available in ../iris-devtools, **When** tests need to manage database state (container lifecycle, schema reset, connection validation, health checks), **Then** the system should use iris-devtools utilities to perform these operations

4. **Given** a developer wants to switch backend modes, **When** they change the configuration file setting or set an environment variable override, **Then** subsequent test runs should use the new backend mode without code changes

5. **Given** tests running in community mode, **When** connection limits are reached, **Then** the system should gracefully queue or retry connections instead of failing with license errors

### Edge Cases

- What happens when iris-devtools is not available at ../iris-devtools? Tests fail immediately with a clear error message indicating the required development dependency is missing.
- How does the system handle switching modes mid-test-session? Mode is validated once at test session start and cannot be changed mid-session (requires new test run).
- What happens when enterprise mode is selected but only community edition IRIS is running? System fails immediately with a clear error message indicating the mismatch and requiring manual correction of the backend mode configuration.

## Requirements

### Functional Requirements

- **FR-001**: System MUST support two distinct backend configuration modes: "enterprise" and "community"

- **FR-002**: System MUST allow users to select backend mode via configuration file (default) with environment variable override capability

- **FR-003**: Community mode MUST limit concurrent database connections to prevent license exhaustion

- **FR-004**: Community mode MUST use sequential test execution strategies to avoid connection pooling issues

- **FR-005**: Enterprise mode MUST allow parallel test execution without connection limits

- **FR-006**: System MUST integrate with iris-devtools from ../iris-devtools for container lifecycle management (start/stop/restart), schema reset between tests, connection validation, and health/readiness checks

- **FR-007**: System MUST fail test execution with a clear error message when iris-devtools is not available at ../iris-devtools (iris-devtools is a required development dependency)

- **FR-008**: System MUST detect IRIS edition (community vs enterprise) and fail immediately with a clear error message when the detected edition does not match the configured backend mode, requiring manual correction

- **FR-009**: Backend configuration MUST be validated before test execution begins

- **FR-010**: System MUST provide clear documentation of differences between enterprise and community modes

- **FR-011**: Community mode MUST limit connection pool to maximum 1 concurrent connection; enterprise mode MUST allow unlimited concurrent connections

- **FR-012**: System MUST log which backend mode is active at test session start

- **FR-013**: Database state management operations (container lifecycle, schema reset, connection validation, health checks) MUST be available identically in both enterprise and community modes

### Non-Functional Requirements

- **NFR-001**: Mode switching MUST NOT require code changes to test files

- **NFR-002**: Community mode MUST prevent >95% of license exhaustion errors during test execution

- **NFR-003**: Enterprise mode performance MUST NOT be degraded by community mode safeguards

- **NFR-004**: Configuration validation errors MUST provide actionable error messages

### Key Entities

- **Backend Configuration**: Represents the selected mode (enterprise or community) with associated connection limits, execution strategies, and database management capabilities

- **Database State Manager**: Coordinates IRIS database operations using iris-devtools utilities, with behavior varying by backend mode

- **Connection Pool**: Manages database connections with size and retry strategies determined by backend mode

---

## Review & Acceptance Checklist

### Content Quality
- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

### Requirement Completeness
- [x] No [NEEDS CLARIFICATION] markers remain (5 clarifications resolved)
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable (>95% license error prevention, 1 connection limit for community)
- [x] Scope is clearly bounded (iris-devtools required dev dependency)
- [x] Dependencies and assumptions identified (iris-devtools from ../iris-devtools)

---

## Execution Status

- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked and resolved (5 questions)
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [x] Review checklist passed
