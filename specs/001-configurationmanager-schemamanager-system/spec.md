# Feature Specification: ConfigurationManager → SchemaManager System

**Feature Branch**: `001-configurationmanager-schemamanager-system`
**Created**: 2025-01-27
**Status**: Draft
**Input**: User description: "configurationmanager -> schemamanager system design - might all be specified and perfectly implemented, but I want you to check"

---

## ⚡ Quick Guidelines
- ✅ Focus on WHAT users need and WHY
- ❌ Avoid HOW to implement (no tech stack, APIs, code structure)
- 👥 Written for business stakeholders, not developers

---

## Clarifications

### Session 2025-01-27
- Q: When multiple configuration sources conflict (environment variables, YAML files, defaults), what should be the exact precedence order for resolving conflicts? → A: Environment variables override YAML files override defaults
- Q: If a schema migration fails partway through execution, should the system automatically rollback all changes, require manual intervention, or provide both options? → A: Automatic rollback with detailed error logging
- Q: Should vector dimension consistency validation occur during system startup, before each pipeline operation, or only when explicitly requested? → A: During system startup only
- Q: Should the system support dynamic configuration reloading during runtime without service restart, or is restart-based configuration change acceptable? → B: Support hot reload for non-critical settings only
- Q: When schema operations exceed performance targets (>5s for migrations, >50ms for config access), should the system log warnings, fail operations, or continue with degraded performance? → A: Log warnings and continue with degraded performance

## User Scenarios & Testing *(mandatory)*

### Primary User Story
RAG framework developers need a reliable configuration management system that automatically handles database schema migrations and vector dimension consistency across all pipeline components. The system must prevent configuration drift, dimension mismatches, and schema inconsistencies that could break production deployments.

### Acceptance Scenarios
1. **Given** a new RAG pipeline deployment, **When** the system starts up, **Then** configuration is automatically loaded from YAML files with environment variable overrides applied correctly
2. **Given** a configuration change to vector dimensions, **When** the system detects the change, **Then** automatic schema migration is triggered to update all affected tables
3. **Given** multiple embedding models in use, **When** any component requests vector dimensions, **Then** the correct dimension is returned based on the table and model context
4. **Given** schema metadata exists in the database, **When** checking schema status, **Then** current vs expected configuration differences are accurately identified

### Edge Cases
- What happens when schema migration fails halfway through (rollback behavior)?
- How does system handle missing or corrupted schema metadata tables?
- What occurs when environment variables contain invalid type casting values?
- How does the system respond to unknown embedding models or table names?

## Requirements *(mandatory)*

### Functional Requirements
- **FR-001**: System MUST load configuration from YAML files with environment variable overrides using RAG_ prefix and __ delimiters, with clear precedence order where environment variables override YAML files which override defaults
- **FR-002**: System MUST validate all required configuration keys and fail fast with clear error messages if missing, with support for hot reload of non-critical settings only while requiring restart for critical configuration changes
- **FR-003**: System MUST provide centralized vector dimension authority for all tables based on embedding models and configuration with validation during system startup only
- **FR-004**: System MUST automatically detect schema migration needs by comparing current vs expected table configurations with comprehensive status reporting
- **FR-005**: System MUST perform safe schema migrations with automatic transaction rollback on failure and detailed error logging for troubleshooting
- **FR-006**: System MUST maintain schema version metadata for all managed tables with audit trail capabilities for compliance and debugging
- **FR-007**: System MUST support table-specific configurations including embedding columns, foreign keys, and indexes with validation and consistency checking
- **FR-008**: System MUST validate vector dimension consistency across all pipeline components during startup only to minimize runtime performance impact
- **FR-009**: System MUST create and manage HNSW vector indexes with ACORN=1 optimization when available, with performance monitoring and degradation handling
- **FR-010**: System MUST provide audit methods for integration testing that replace direct SQL access patterns while maintaining performance targets of sub-50ms configuration access and sub-5s migration operations, logging warnings when targets are exceeded but continuing with degraded performance

### Key Entities *(include if feature involves data)*
- **ConfigurationManager**: Central authority for loading and accessing all system configuration from YAML files and environment variables
- **SchemaManager**: Vector dimension authority and database schema migration manager that ensures table consistency
- **SchemaMetadata**: Database table tracking schema versions, vector dimensions, and configuration for each managed table
- **TableConfiguration**: Specifications for each table including embedding columns, dimensions, foreign keys, and index requirements
- **MigrationPlan**: Transaction-safe procedures for updating table schemas while preserving data integrity

## Database Requirements *(mandatory for data-dependent features)*

### IRIS Database Dependency
- **Database Type**: InterSystems IRIS with vector search capabilities and ACORN=1 optimization
- **Test Environment**: Framework-managed Docker container (`docker-compose -f docker-compose.iris-only.yml up -d`)
- **Health Check**: IRIS connectivity validation required before test execution using `evaluation_framework/test_iris_connectivity.py`
- **Data Operations**: All configuration validation, schema operations, and vector index management use live database
- **Performance Targets**: <50ms configuration access, <5s schema migrations measured against actual IRIS instance

### Constitutional Compliance Requirements
- **Test Categories**: All validation tasks MUST use `@pytest.mark.requires_database` marker
- **Database Health**: IRIS container health verification required before any data-dependent testing
- **Schema Operations**: Real schema creation, migration, and rollback testing on live database
- **Vector Operations**: Actual HNSW index creation with ACORN=1 optimization verification
- **Error Handling**: Database connection failures, transaction rollbacks, and schema corruption testing

### Database Validation Requirements

**CRITICAL**: All ConfigurationManager and SchemaManager validation MUST execute against live IRIS database:

#### Required Pre-Implementation Steps
1. **Database Health Check**: `python evaluation_framework/test_iris_connectivity.py`
2. **Container Management**: `docker-compose -f docker-compose.iris-only.yml up -d`
3. **Schema Validation**: Actual schema creation and migration testing with real DDL execution
4. **Vector Operations**: Real HNSW index creation with ACORN=1 optimization and dimension validation

#### Invalid Test Patterns (Constitutional Violations)
❌ Import-only validation without database connectivity
❌ Mock objects for ConfigurationManager/SchemaManager database operations
❌ Configuration loading validation without schema verification
❌ "Integration" testing that doesn't integrate with IRIS database
❌ Performance claims without actual database performance measurement

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
- [x] Success criteria are measurable
- [x] Scope is clearly bounded
- [x] Dependencies and assumptions identified

---

## Execution Status
*Updated by main() during processing*

- [x] User description parsed
- [x] Key concepts extracted
- [x] Ambiguities marked
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [x] Review checklist passed

---