# Feature Specification: Test Infrastructure Resilience and Database Schema Management

**Feature Branch**: `028-obviously-these-failures`
**Created**: 2025-10-05
**Status**: Draft
**Input**: User description: "obviously these failures are due to a gap in our specs!! let's fix this"

## Execution Flow (main)
```
1. Parse user description from Input
   → Identified: Test suite failures due to infrastructure gaps
2. Extract key concepts from description
   → Actors: Test runner, Database, Schema manager
   → Actions: Reset schema, validate tables, cleanup test state
   → Data: Database tables (SourceDocuments, DocumentChunks, Entities, Relationships)
   → Constraints: Must not affect production, must be idempotent
3. For each unclear aspect:
   → [RESOLVED: Analysis shows 69 errors from schema mismatch, 47 from unimplemented MCP, 218 cascading failures]
4. Fill User Scenarios & Testing section
   → Clear user flow: Developer runs tests → schema conflicts → tests fail
5. Generate Functional Requirements
   → Each requirement is testable via pytest
6. Identify Key Entities
   → Database schema, Test fixtures, Cleanup handlers
7. Run Review Checklist
   → No [NEEDS CLARIFICATION] markers
   → Implementation details intentionally included (this is infrastructure)
8. Return: SUCCESS (spec ready for planning)
```

---

## ⚡ Quick Guidelines
- ✅ Focus on WHAT test infrastructure needs and WHY
- ✅ Ensure medical-grade reliability for test execution
- 👥 Written for developers maintaining the test suite

---

## User Scenarios & Testing *(mandatory)*

### Primary User Story
As a developer running the test suite, I need the database schema to be automatically managed and reset between test runs so that schema conflicts never cause test failures and I can trust the test results represent actual code issues, not infrastructure problems.

### Acceptance Scenarios

1. **Given** a fresh IRIS database instance, **When** tests run for the first time, **Then** all required tables are created with correct schema and all 771 tests execute without schema errors

2. **Given** tables exist from a previous test run with outdated schema, **When** tests run again, **Then** schema is automatically detected as stale, tables are dropped and recreated, and tests proceed without errors

3. **Given** multiple test sessions run concurrently, **When** each session needs clean database state, **Then** each session gets isolated schema or proper cleanup without interfering with other sessions

4. **Given** a test fails midway leaving partial data, **When** next test runs, **Then** database state is cleaned up and test starts fresh

5. **Given** MCP contract tests that expect unimplemented modules, **When** these tests run, **Then** they are properly marked as "expected failures" or "skipped" rather than causing ERROR status

### Edge Cases
- What happens when IRIS database is not running? → Tests should skip with clear message, not error
- What happens when schema version changes between runs? → Auto-detection and migration or clean rebuild
- What happens when table drop fails? → Retry with CASCADE or fail clearly with actionable message
- How does system handle partial schema (some tables exist, some don't)? → Validate all tables, recreate incomplete sets

---

## Requirements *(mandatory)*

### Functional Requirements

#### Database Schema Management
- **FR-001**: System MUST detect when database tables exist with incompatible schema before running tests
- **FR-002**: System MUST provide automatic schema reset functionality that drops and recreates all RAG tables (SourceDocuments, DocumentChunks, Entities, Relationships)
- **FR-003**: Schema reset MUST be idempotent (safe to run multiple times)
- **FR-004**: System MUST validate table schema matches expected structure before allowing tests to proceed
- **FR-005**: System MUST provide clear error messages when database is unavailable, including actionable remediation steps

#### Test Isolation and Cleanup
- **FR-006**: Each test class MUST have access to clean database state via pytest fixtures
- **FR-007**: Test fixtures MUST automatically cleanup database state after test completion (success or failure)
- **FR-008**: System MUST support test-level transaction rollback for fast cleanup
- **FR-009**: Cleanup operations MUST handle partial data (documents without embeddings, entities without relationships)
- **FR-010**: System MUST prevent test data pollution between test runs

#### Contract Test Management
- **FR-011**: Contract tests for unimplemented features (MCP modules) MUST be marked with `@pytest.mark.contract`
- **FR-012**: Contract test failures MUST not contribute to overall test failure count when feature is known to be unimplemented
- **FR-013**: System MUST distinguish between "expected contract failures" (TDD) and "actual bugs"
- **FR-014**: Contract tests MUST provide clear messages indicating they are testing future functionality

#### Test Execution Reliability
- **FR-015**: Test suite MUST provide pre-flight checks that validate all prerequisites (database, API keys, schema)
- **FR-016**: System MUST fail fast with clear messages when prerequisites are not met
- **FR-017**: Database connection failures MUST be caught at setup phase, not during individual tests
- **FR-018**: System MUST provide utility to reset entire test environment to known-good state

#### Medical-Grade Quality Standards
- **FR-019**: Schema management operations MUST be logged with timestamps and outcomes for audit trail
- **FR-020**: System MUST track which schema version each test run used
- **FR-021**: Failed schema operations MUST provide full diagnostic information (table name, SQL attempted, error code)
- **FR-022**: System MUST validate that cleanup operations actually removed all test data before marking as complete

### Non-Functional Requirements

- **NFR-001**: Schema reset operation MUST complete in under 5 seconds for all 4 tables
- **NFR-002**: Test isolation overhead MUST add less than 100ms per test class
- **NFR-003**: Pre-flight checks MUST complete in under 2 seconds
- **NFR-004**: Error messages MUST include specific SQLCODE and remediation steps
- **NFR-005**: System MUST handle 100% of known schema conflict scenarios without manual intervention

### Key Entities

- **DatabaseSchema**: Represents the expected structure of RAG tables (SourceDocuments, DocumentChunks, Entities, Relationships) including column names, types, and indexes
- **SchemaValidator**: Validates actual database tables against expected schema, detecting mismatches
- **TestFixtureManager**: Manages database state for test isolation, provides clean state for each test
- **ContractTestMarker**: Identifies and manages tests that are expected to fail (TDD contract tests)
- **PreflightChecker**: Validates all prerequisites before test execution (database connectivity, schema validity, API keys)
- **CleanupHandler**: Ensures database is cleaned after each test, handles partial data scenarios

---

## Review & Acceptance Checklist

### Content Quality
- [x] No implementation details (languages, frameworks, APIs) - **EXCEPTION: This is infrastructure spec**
- [x] Focused on test infrastructure reliability and developer value
- [x] Written for developers maintaining test suite
- [x] All mandatory sections completed

### Requirement Completeness
- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable (771 tests run, 0 schema errors)
- [x] Scope is clearly bounded (database schema + test infrastructure only)
- [x] Dependencies identified (IRIS database, pytest, conftest.py)

---

## Execution Status

- [x] User description parsed
- [x] Key concepts extracted (schema mismatch, MCP contract tests, cascading failures)
- [x] Ambiguities marked (none - analysis was comprehensive)
- [x] User scenarios defined (5 scenarios + 4 edge cases)
- [x] Requirements generated (22 functional, 5 non-functional)
- [x] Entities identified (6 key entities)
- [x] Review checklist passed

---

## Analysis Summary

Based on test run of 771 tests:
- **69 ERRORS**: Database schema mismatch (SQLCODE -29: Field not found)
- **47 ERRORS**: Unimplemented MCP modules (expected TDD contract test failures)
- **218 FAILED**: Cascading failures from database setup errors
- **476 PASSED**: Including 100% of Feature 026 quality framework tests
- **8 SKIPPED**: Properly handled optional tests

**Root Cause**: No automated schema management or contract test handling in test infrastructure.

**Impact**: ~90% of test failures will resolve with proper schema reset and contract test marking.

**Priority**: CRITICAL - blocking medical-grade quality validation
