# Feature Specification: Fix Critical Bugs in v0.5.3 (Connection API + Schema Initialization)

**Feature Branch**: `060-fix-users-tdyar`
**Created**: 2025-01-13
**Updated**: 2025-01-13 (Added iris.connect() bug fix from FHIR-AI feedback)
**Status**: Draft
**Input**:
1. Bug Report: "/Users/tdyar/ws/hipporag2-pipeline/BUG_REPORT_IRIS_VECTOR_RAG_PPR_SCHEMA.md"
2. FHIR-AI v0.5.3 Testing Feedback: "iris.connect() AttributeError in iris_dbapi_connector.py line 210"

## Overview

This feature addresses two critical bugs identified in v0.5.3:

**Bug 1: iris.connect() AttributeError (CRITICAL - Breaks All Connections)**
The connection code at `iris_dbapi_connector.py:210` attempts to call `iris.connect()`, which does not exist in the intersystems-irispython package. This breaks all database connections in v0.5.3, making the framework unusable. The correct API is `iris.createConnection()` (native IRIS), `iris.dbapi.connect()` (DBAPI), or `iris_devtester.IRISContainer().dbapi_connection()` (recommended).

**Bug 2: Silent iris-vector-graph Schema Failures (HIGH - Degrades Retrieval Quality)**
When the system has the iris-vector-graph package installed, graph-based retrieval features (particularly Personalized PageRank for multi-hop question answering) fail silently because required database tables are not automatically created during pipeline initialization. This causes the system to fall back to basic uniform scoring without any indication to users that advanced retrieval capabilities are degraded.

## User Scenarios & Testing *(mandatory)*

### Primary User Stories

**Story 1: Database Connection Functionality**
As a framework user, when I create a database connection using the iris-vector-rag framework, I need the connection to work correctly so that I can store and retrieve vectors without AttributeError crashes.

**Story 2: Automatic Graph Schema Initialization**
As a data processing pipeline operator, when I install iris-vector-graph to enable advanced graph-based retrieval features, I need the system to automatically initialize all required database tables so that Personalized PageRank and graph-based re-ranking work correctly without manual database setup steps.

### Acceptance Scenarios

**Connection Bug Scenarios:**
1. **Given** ConnectionManager is initialized with valid IRIS credentials, **When** a connection is created, **Then** the connection uses the correct `iris.createConnection()` API (not the non-existent `iris.connect()`)

2. **Given** ConnectionManager attempts to connect to IRIS, **When** the connection is established, **Then** no AttributeError is raised about missing iris.connect() method

3. **Given** a valid IRIS connection configuration, **When** IRISVectorStore or SchemaManager creates a connection, **Then** the connection succeeds and database operations work correctly

**Graph Schema Scenarios:**
4. **Given** iris-vector-graph is installed and a pipeline is initialized with a clean database, **When** the pipeline setup runs, **Then** all required graph tables (rdf_labels, rdf_props, rdf_edges, kg_NodeEmbeddings_optimized) are automatically created

5. **Given** iris-vector-graph is installed and PPR retrieval is requested, **When** the system attempts to compute personalized page rank scores, **Then** the operation succeeds without "Table not found" errors

6. **Given** iris-vector-graph is NOT installed and a pipeline is initialized, **When** the pipeline setup runs, **Then** the system proceeds normally without attempting to create graph tables

7. **Given** iris-vector-graph is installed but table creation fails, **When** the initialization error occurs, **Then** the system provides a clear error message indicating which tables failed to initialize and why

8. **Given** iris-vector-graph tables exist and PPR is requested, **When** multi-hop questions are processed, **Then** the system uses graph-based re-ranking instead of falling back to uniform scoring

### Edge Cases

**Connection Edge Cases:**
- What happens when iris.createConnection() fails (invalid credentials)?
- How does the system handle SSL connection requirements?
- What happens when DBAPI connection pool is exhausted?

**Graph Schema Edge Cases:**
- What happens when iris-vector-graph is installed after initial pipeline setup?
- How does the system handle partial table creation (some tables exist, others don't)?
- What happens if iris-vector-graph tables exist but have incorrect schema versions?
- How does the system behave when database permissions prevent table creation?
- What happens when multiple pipelines try to initialize tables concurrently?

## Requirements *(mandatory)*

### Functional Requirements

**Connection API Fix (Bug 1 - CRITICAL):**
- **FR-001**: System MUST use correct IRIS connection API (`iris.createConnection()`, `iris.dbapi.connect()`, or `iris_devtester.IRISContainer().dbapi_connection()`) instead of non-existent `iris.connect()`
- **FR-002**: System MUST successfully establish database connections without AttributeError crashes
- **FR-003**: System MUST replace all instances of `iris.connect()` calls in `iris_dbapi_connector.py` with correct API
- **FR-004**: System MUST pass all connection-related tests (ConnectionManager, IRISVectorStore, SchemaManager)

**Automatic Detection and Initialization (Bug 2):**
- **FR-005**: System MUST automatically detect when iris-vector-graph package is installed in the Python environment
- **FR-006**: System MUST automatically create all required graph tables (rdf_labels, rdf_props, rdf_edges, kg_NodeEmbeddings_optimized) during pipeline initialization when iris-vector-graph is detected
- **FR-007**: System MUST skip graph table creation when iris-vector-graph is not installed without raising errors
- **FR-008**: System MUST verify that all required tables exist before attempting Personalized PageRank operations

**Error Handling and Feedback:**
- **FR-009**: System MUST provide clear error messages when table creation fails, indicating which specific table(s) failed and the underlying database error
- **FR-010**: System MUST log successful graph table initialization with confirmation of which tables were created
- **FR-011**: System MUST fail fast with a descriptive error (not silent fallback) when PPR is requested but required tables are missing
- **FR-012**: System MUST distinguish between "iris-vector-graph not installed" (expected) and "tables missing but should exist" (error condition)

**Schema Management:**
- **FR-013**: System MUST integrate graph table initialization into the existing SchemaManager initialization flow
- **FR-014**: System MUST use the same table creation patterns that already exist in SchemaManager for graph tables
- **FR-015**: System MUST handle idempotent table creation (safe to run multiple times, no errors if tables already exist)

**Validation and Testing:**
- **FR-016**: System MUST validate that created tables have the correct schema structure
- **FR-017**: System MUST provide a validation method to check if all graph prerequisites are met before PPR operations
- **FR-018**: System MUST log whether PPR functionality is available or degraded based on table initialization status

### Success Criteria

**Measurable Outcomes:**
1. **Connection Success**: 100% of connection attempts succeed without AttributeError (currently 0% in v0.5.3)
2. **Test Pass Rate**: 6/6 FHIR-AI tests pass (currently 3/6 in v0.5.3)
3. **Table Creation Success**: 100% of required graph tables are created when iris-vector-graph is installed
4. **Error Reduction**: Zero "Table not found" errors during PPR operations when iris-vector-graph is available
5. **Silent Failure Elimination**: PPR failures are explicitly reported (not silent fallback to uniform scoring)
6. **Initialization Time**: Graph table creation adds less than 5 seconds to pipeline initialization
7. **Backward Compatibility**: Pipelines without iris-vector-graph continue to work exactly as before

**Qualitative Measures:**
1. **Framework Usability**: All database operations work correctly (ConnectionManager, IRISVectorStore, SchemaManager)
2. **Developer Experience**: No manual database setup steps required when iris-vector-graph is installed
3. **Error Clarity**: Error messages clearly indicate missing prerequisites and remediation steps
4. **Observability**: Logs clearly show whether PPR functionality is enabled or disabled

### Key Entities

- **IRIS Connection**: Database connection established via intersystems-irispython package
  - Attributes: host, port, namespace, user, password, SSL settings
  - Correct APIs: `iris.createConnection()` (native), `iris.dbapi.connect()` (DBAPI), `iris_devtester.IRISContainer().dbapi_connection()` (recommended)
  - Incorrect API: `iris.connect()` (does not exist - causes AttributeError)
  - Used by: ConnectionManager, IRISVectorStore, SchemaManager

- **Graph Table Schemas**: Database table definitions required by iris-vector-graph for graph-based operations
  - Attributes: table name, column definitions, indexes, foreign key relationships
  - Tables include: rdf_labels (node labels), rdf_props (node properties), rdf_edges (graph edges), kg_NodeEmbeddings_optimized (optimized embeddings for graph nodes)
  - Relationships: Tables must be created in dependency order (nodes before edges)

- **Pipeline Initialization Context**: The setup phase when database schemas are validated and created
  - Attributes: connection information, available packages, schema state
  - Determines whether graph features should be initialized
  - Responsible for coordinating SchemaManager operations

- **PPR Operation**: Personalized PageRank computation for graph-based retrieval re-ranking
  - Prerequisites: All graph tables must exist and contain data
  - Behavior: Should fail explicitly when prerequisites are not met (not silent fallback)

### Non-Functional Requirements

- **Stability**: Database connections must work 100% of the time without AttributeErrors (Bug 1 fix)
- **Reliability**: Table creation must be atomic (all tables succeed or all fail)
- **Idempotency**: Running initialization multiple times must be safe and produce the same result
- **Performance**: Schema validation checks must complete in under 1 second
- **Observability**: All initialization steps must generate appropriate log messages (INFO for success, ERROR for failures)
- **Backward Compatibility**: v0.5.4 must pass all tests that passed in v0.5.2 (ConfigurationManager, Environment variables, Document model) PLUS the 3 tests that fail in v0.5.3 (ConnectionManager, IRISVectorStore, SchemaManager)

## Assumptions & Constraints

### Assumptions

1. The `iris.connect()` method call at `iris_dbapi_connector.py:210` is the primary cause of connection failures in v0.5.3
2. The SchemaManager already knows about graph table definitions (confirmed in schema_manager.py:1864)
3. The iris-vector-graph package is an optional dependency (not required for core functionality)
4. Users who install iris-vector-graph expect graph features to work automatically
5. Database connection has sufficient permissions to create tables
6. The CloudConfiguration API implemented in v0.5.3 for dimension reading should be preserved (it fixed the dimension bug)

### Constraints

1. Must maintain backward compatibility with pipelines that don't use iris-vector-graph
2. Cannot require manual database setup steps (must be automatic)
3. Must work with existing SchemaManager table creation patterns
4. Cannot introduce breaking changes to pipeline initialization API
5. Must use only documented intersystems-irispython APIs (not non-existent methods like `iris.connect()`)

## Dependencies

### Internal Dependencies
- **iris_dbapi_connector.py**: Connection establishment code that needs iris.connect() fix (line 210)
- SchemaManager with graph table definitions (iris_vector_rag/storage/schema_manager.py)
- Pipeline initialization logic that calls SchemaManager
- Logging system for initialization feedback
- ConnectionManager that uses iris_dbapi_connector

### External Dependencies
- **intersystems-irispython>=5.1.2**: IRIS database driver with correct connection APIs
- iris-vector-graph package (optional, detected at runtime for Bug 2 fix)
- IRIS database with table creation permissions

## Success Metrics

### Primary Metrics
- **Connection Success Rate**: Percentage of connection attempts that succeed (target: 100%, currently 0% in v0.5.3)
- **Test Pass Rate**: Ratio of passing FHIR-AI tests (target: 6/6, currently 3/6 in v0.5.3)
- **PPR Success Rate**: Percentage of PPR operations that succeed without table errors (target: 100% when iris-vector-graph is installed)
- **Silent Failure Rate**: Number of silent fallbacks to uniform scoring (target: 0)
- **Initialization Failures**: Number of table creation errors per 1000 pipeline setups (target: < 1)

### Secondary Metrics
- **Setup Time Impact**: Additional seconds added to pipeline initialization (target: < 5s)
- **Error Message Clarity**: Percentage of users who can diagnose issues from error messages alone (target: > 90%)
- **Documentation Reduction**: Reduction in manual setup documentation needed (target: eliminate all database setup steps)
- **Regression Prevention**: All v0.5.2 passing tests continue to pass in v0.5.4 (CloudConfiguration, environment variables, document model)

## Out of Scope

The following are explicitly excluded from this feature:

- SSL connection configuration changes (Bug 1 fix only changes connection API, not SSL settings)
- Connection pooling improvements (only fixes the API call)
- Automatic installation of iris-vector-graph package (user must install manually)
- Schema migration or version upgrade logic (only handles initial creation)
- Performance optimization of graph operations (only focuses on initialization)
- Data population or seeding of graph tables (only creates empty tables)
- Automatic detection of schema version mismatches between iris-vector-rag and iris-vector-graph
- Custom table naming or schema customization beyond defaults
- Retry logic for transient database connection failures during initialization
- Reverting CloudConfiguration API changes (v0.5.3 dimension fix should be preserved)

---

## Review & Acceptance Checklist

### Content Quality
- [x] No implementation details (languages, frameworks, APIs)
- [x] Focused on user value and business needs
- [x] Written for non-technical stakeholders
- [x] All mandatory sections completed

### Requirement Completeness
- [x] No [NEEDS CLARIFICATION] markers remain
- [x] Requirements are testable and unambiguous
- [x] Success criteria are measurable
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

---

## Execution Status

- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked (none - bug report provided complete context)
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [x] Review checklist passed
