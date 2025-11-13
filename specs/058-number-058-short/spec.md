# Feature Specification: Cloud Configuration Flexibility

**Feature Branch**: `058-cloud-config-flexibility`
**Created**: 2025-01-12
**Status**: Draft
**Input**: User description: "Address FHIR-AI-Hackathon-Kit feedback: Enable flexible configuration for cloud deployments by supporting environment variables, respecting config files, configurable vector dimensions, and documented namespace requirements"

## Executive Summary

The iris-vector-rag library currently has hardcoded configuration values that block cloud deployments (AWS IRIS, Azure, GCP). Users report 65-minute migration times with workarounds, versus an expected 25 minutes with proper configuration support. This feature addresses 9 pain points documented in FHIR-AI-Hackathon-Kit project feedback to enable seamless cloud deployments.

**Impact**: Reduce cloud migration time by 60% (40 minutes saved per deployment) and unblock NVIDIA NIM, OpenAI, and other variable-dimension embedding models.

---

## User Scenarios & Testing

### Primary User Story
As a developer deploying GraphRAG to AWS IRIS (or other cloud environments), I need to configure connection settings, vector dimensions, and table schemas through configuration files and environment variables without modifying source code, so I can deploy to cloud infrastructure with different namespace restrictions and embedding model requirements.

### Acceptance Scenarios

1. **Environment Variable Configuration**
   - **Given** I have AWS IRIS credentials in environment variables (IRIS_HOST, IRIS_PORT, IRIS_USERNAME, IRIS_PASSWORD, IRIS_NAMESPACE)
   - **When** I initialize the connection manager
   - **Then** It uses my environment variables instead of hardcoded localhost values
   - **And** I can connect to AWS IRIS without code modifications

2. **Config File Respected by init_tables()**
   - **Given** I have a config file specifying vector_dimension=1024 and table_schema="SQLUser"
   - **When** I run the init_tables() command with --config flag
   - **Then** Tables are created with 1024-dimensional vectors in SQLUser schema
   - **And** I don't need to write workaround scripts

3. **Configurable Vector Dimensions**
   - **Given** I'm using NVIDIA NIM embeddings (1024 dimensions) instead of default SentenceTransformers (384 dimensions)
   - **When** I configure vector_dimension=1024 in my config file
   - **Then** All vector storage operations use 1024-dimensional vectors
   - **And** My embeddings aren't truncated or rejected

4. **Namespace Configuration**
   - **Given** AWS IRIS requires %SYS namespace (DEMO namespace has restricted access)
   - **When** I specify namespace="%SYS" in my configuration
   - **Then** All database operations use the %SYS namespace
   - **And** I don't encounter permission denied errors

5. **Schema-Prefixed Table Names**
   - **Given** I need tables in SQLUser schema for cloud deployment
   - **When** I configure table_schema="SQLUser" in my config
   - **Then** Tables are created as SQLUser.Entities and SQLUser.EntityRelationships
   - **And** I can run incremental syncs without schema conflicts

### Edge Cases
- **What happens when environment variables conflict with config file values?** Environment variables take precedence (12-factor app pattern), documented priority order
- **What happens when vector dimensions don't match existing tables?** System detects mismatch and provides clear error message with migration guidance
- **What happens when namespace doesn't exist or user lacks permissions?** System performs preflight check and fails fast with actionable error message
- **What happens when config file is missing or malformed?** System falls back to documented defaults and logs warnings about missing configuration

---

## Requirements

### Functional Requirements
- **FR-001**: System MUST read connection parameters (host, port, username, password, namespace) from environment variables when present
- **FR-002**: System MUST respect config file specifications for all init_tables() operations, including vector dimensions and schema names
- **FR-003**: System MUST support configurable vector dimensions from 128 to 8192 to accommodate different embedding models
- **FR-004**: System MUST allow table schema prefix configuration (e.g., "SQLUser", "DEMO", "%SYS") for cloud namespace requirements
- **FR-005**: System MUST provide configuration priority order: environment variables > config file > defaults
- **FR-006**: System MUST validate configuration at startup and fail fast with clear error messages for invalid settings
- **FR-007**: System MUST document all required namespace permissions and access requirements for cloud deployments
- **FR-008**: System MUST preserve backward compatibility with existing local deployments using default values
- **FR-009**: System MUST detect vector dimension mismatches between config and existing tables, providing migration guidance
- **FR-010**: Configuration documentation MUST include examples for AWS IRIS, Azure, and on-premises deployments
- **FR-011**: System MUST log all configuration sources used (environment variable, config file path, defaults) for troubleshooting

### Success Criteria
- **SC-001**: Cloud deployment time reduces from 65 minutes to under 25 minutes (60% reduction)
- **SC-002**: Zero code modifications required to deploy to AWS IRIS, Azure, or GCP
- **SC-003**: Users can switch embedding models (384-dim to 1024-dim) through configuration only
- **SC-004**: All 9 documented pain points from FHIR-AI-Hackathon-Kit feedback are resolved
- **SC-005**: Configuration documentation includes copy-paste examples for all major cloud providers
- **SC-006**: Existing local deployments continue working without any changes (100% backward compatible)
- **SC-007**: init_tables() respects --config flag 100% of the time (currently 0%)
- **SC-008**: Namespace permission errors decrease to zero through preflight validation

### Key Entities
- **ConnectionConfiguration**: Represents connection parameters (host, port, credentials, namespace) with priority-based resolution from environment variables, config files, and defaults
- **VectorConfiguration**: Encapsulates vector storage settings (dimension, distance metric, index type) with validation against supported ranges
- **TableConfiguration**: Defines table schema, names, and namespace requirements with cloud-specific overrides
- **ConfigurationSource**: Tracks where each configuration value originated (env var, file, default) for debugging and audit trails

---

## Assumptions & Dependencies

### Assumptions
1. Users have valid IRIS credentials with appropriate namespace permissions
2. Configuration files follow YAML format (current project standard)
3. Vector dimension changes require table recreation or migration (destructive operation requiring user acknowledgment)
4. Cloud providers (AWS, Azure, GCP) support standard IRIS connection protocols
5. Configuration validation overhead (< 100ms) is acceptable at startup
6. Default values work for 80% of local development scenarios

### Dependencies
- Existing ConfigurationManager system must be extended (not replaced)
- IRIS database version supports configurable namespaces and schemas
- No breaking changes to existing pipeline APIs
- Documentation system supports code examples and deployment guides

### Out of Scope
- Automatic table migration between vector dimensions (requires separate migration tool)
- Configuration GUI or web interface (CLI/file-based only)
- Multi-region failover configuration
- Encryption of configuration files (handled by infrastructure/secrets management)
- Configuration versioning and rollback (handled by git/infrastructure)

---

## Risks & Mitigations

### Configuration Complexity (Medium Risk)
- **Risk**: Too many configuration options confuse users
- **Mitigation**: Provide environment-specific templates (aws.yaml, azure.yaml, local.yaml) with sensible defaults
- **Mitigation**: Use configuration validation to catch 90% of errors before deployment

### Backward Compatibility (Low Risk)
- **Risk**: Changes break existing deployments
- **Mitigation**: Make all configuration optional with backward-compatible defaults
- **Mitigation**: Comprehensive testing with existing test suite to detect regressions

### Vector Dimension Mismatch (Medium Risk)
- **Risk**: Users change vector dimensions without recreating tables, causing data corruption
- **Mitigation**: Preflight validation detects mismatches and blocks operations
- **Mitigation**: Clear error messages with migration instructions

### Documentation Lag (Low Risk)
- **Risk**: Configuration options added faster than documentation updates
- **Mitigation**: Auto-generate config schema documentation from validation code
- **Mitigation**: Include configuration examples in integration tests (docs as code)

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
- [x] Ambiguities marked (0 clarifications needed - made informed assumptions)
- [x] User scenarios defined
- [x] Requirements generated
- [x] Entities identified
- [x] Review checklist passed

---
