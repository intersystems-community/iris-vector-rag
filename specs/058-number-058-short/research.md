# Research: Cloud Configuration Flexibility

**Feature**: 058-cloud-config-flexibility
**Date**: 2025-01-12
**Status**: Complete

## Overview

This research document captures technical decisions for enabling flexible cloud deployment configuration in iris-vector-rag. The feature addresses 9 documented pain points from FHIR-AI-Hackathon-Kit AWS migration that blocked cloud deployments.

## Key Research Areas

### 1. Configuration Priority Pattern

**Decision**: Use 12-factor app configuration pattern (environment variables > config file > defaults)

**Rationale**:
- Industry-standard pattern for cloud-native applications
- Enables containerized deployments (Docker, Kubernetes)
- Allows runtime configuration without code changes
- Backward compatible (defaults work for existing local deployments)

**Alternatives Considered**:
- CLI flags only: Rejected - too verbose, not suitable for containers
- Config file only: Rejected - blocks containerized deployments
- Environment variables only: Rejected - no sensible defaults for local development

**Implementation Approach**:
- Extend existing ConfigurationManager in iris_vector_rag/config/manager.py
- Add `_load_env_variables()` method already exists - enhance with IRIS-specific vars
- Environment variable naming: `IRIS_HOST`, `IRIS_PORT`, `IRIS_USERNAME`, `IRIS_PASSWORD`, `IRIS_NAMESPACE`
- No RAG_ prefix for IRIS connection vars (simpler for users)

**Reference**: Feature 035 (backend modes) demonstrates successful env var pattern in this codebase

### 2. Vector Dimension Validation

**Decision**: Preflight validation with clear error messages and migration guidance

**Rationale**:
- Vector dimension mismatches cause silent data corruption
- Users changing from 384 (SentenceTransformers) to 1024 (NVIDIA NIM) need explicit validation
- Enterprise requirement: Fail fast with actionable errors

**Alternatives Considered**:
- Auto-migration: Rejected - requires separate migration tool (out of scope)
- Silent truncation: Rejected - violates Constitution Principle VI (explicit error handling)
- Schema flexibility: Rejected - IRIS vector tables have fixed dimensions

**Implementation Approach**:
- Create `VectorDimensionValidator` in iris_vector_rag/config/validators.py
- Validation steps:
  1. Read configured dimension from config
  2. Query existing table schema: `SELECT VECTOR_DIM FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_NAME='Entities'`
  3. Compare configured vs existing
  4. On mismatch: Raise ConfigValidationError with migration guide URL

**Validation Range**: 128-8192 dimensions (supports all major embedding models)

### 3. Namespace Configuration & Permissions

**Decision**: Configurable namespace with documented permission requirements for each cloud provider

**Rationale**:
- AWS IRIS requires %SYS namespace (DEMO has restricted access)
- Azure/GCP may have different namespace restrictions
- Enterprise requirement: Clear documentation prevents 20-minute debugging sessions

**Alternatives Considered**:
- Auto-detect namespace: Rejected - no reliable way to determine best namespace
- Hardcode per cloud provider: Rejected - not flexible enough
- No validation: Rejected - silent permission failures violate Principle VI

**Implementation Approach**:
- Add `IRIS_NAMESPACE` environment variable (default: "USER")
- Create `NamespaceValidator` in iris_vector_rag/config/validators.py
- Validation steps:
  1. Test namespace access: `SELECT 1 FROM %SYS.ALL`
  2. Test write permissions: `CREATE TABLE test_permissions (id INT)`
  3. Cleanup test table
  4. On failure: Raise ConfigValidationError with required permissions list

**Documentation Additions**:
- config/examples/aws.yaml - AWS-specific namespace guidance
- config/examples/azure.yaml - Azure-specific namespace guidance
- docs/cloud-deployment.md - Complete cloud deployment guide

### 4. Schema-Prefixed Table Names

**Decision**: Configurable table schema prefix (e.g., "SQLUser", "DEMO", "%SYS")

**Rationale**:
- Cloud providers enforce schema isolation
- Users need `SQLUser.Entities` instead of `RAG.Entities` for AWS
- Enables multiple iris-vector-rag deployments in same IRIS instance

**Alternatives Considered**:
- Fully qualified table names: Rejected - too verbose, error-prone
- Schema per pipeline: Rejected - unnecessary complexity
- No schema support: Rejected - blocks cloud deployments

**Implementation Approach**:
- Add `table_schema` configuration option (default: "RAG")
- Modify iris_vector_rag/services/storage.py table references:
  ```python
  # OLD: f"INSERT INTO RAG.Entities ..."
  # NEW: f"INSERT INTO {self.config.table_schema}.Entities ..."
  ```
- Update cli/init_tables.py to use configured schema
- Ensure all SQL queries use schema prefix

**Backward Compatibility**: Default "RAG" schema preserves existing behavior

### 5. Configuration File Format

**Decision**: Continue using YAML format with enhanced schema

**Rationale**:
- Existing codebase uses YAML extensively (config/*.yaml)
- YAML is human-readable, supports comments, widely adopted
- No need to introduce new format

**Alternatives Considered**:
- TOML: Rejected - requires new dependency, no clear advantage
- JSON: Rejected - no comments, less human-friendly
- Python files: Rejected - security risk, not suitable for user-provided config

**Schema Enhancements**:
```yaml
database:
  iris:
    host: ${IRIS_HOST:localhost}         # Env var with default
    port: ${IRIS_PORT:1972}
    username: ${IRIS_USERNAME:_SYSTEM}
    password: ${IRIS_PASSWORD:SYS}
    namespace: ${IRIS_NAMESPACE:USER}

storage:
  vector_dimension: ${VECTOR_DIMENSION:384}  # Configurable (128-8192)
  table_schema: ${TABLE_SCHEMA:RAG}          # Schema prefix

tables:
  entities: "Entities"              # Table name (schema prefix added automatically)
  relationships: "EntityRelationships"
```

**Validation Library**: Use pydantic for schema validation (already in dependencies via FastAPI)

### 6. init_tables CLI Enhancement

**Decision**: Make init_tables respect --config flag (currently ignores it)

**Rationale**:
- Pain Point #2: init_tables ignores --config flag
- Users forced to write workaround scripts
- Breaks expected CLI behavior

**Current Behavior** (bug):
```bash
# User runs:
python -m iris_rag.cli.init_tables --config aws.yaml

# But init_tables.py loads default config instead:
config = ConfigurationManager()  # No config_path passed!
```

**Fixed Behavior**:
```python
# cli/init_tables.py
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', help='Path to configuration file')
    args = parser.parse_args()

    # FIX: Pass config_path to ConfigurationManager
    config = ConfigurationManager(config_path=args.config)

    # Use config for table creation
    vector_dim = config.get('storage.vector_dimension', 384)
    schema = config.get('storage.table_schema', 'RAG')
    ...
```

**Testing**: Contract test will verify --config flag is respected

### 7. Backward Compatibility Strategy

**Decision**: All configuration is optional with sensible defaults

**Rationale**:
- Constitution Principle: 100% backward compatibility required
- Existing local deployments must work without changes
- Success Criterion SC-006: Existing deployments continue working

**Compatibility Matrix**:
| Configuration | Default Value | Existing Behavior | New Behavior |
|---------------|--------------|-------------------|--------------|
| IRIS_HOST | localhost | ✓ Works | ✓ Works |
| IRIS_PORT | 1972 | ✓ Works | ✓ Works |
| IRIS_NAMESPACE | USER | ✓ Works | ✓ Works |
| vector_dimension | 384 | ✓ Works | ✓ Works |
| table_schema | RAG | ✓ Works | ✓ Works |

**Testing Strategy**:
- Contract test: `test_backward_compatibility` verifies existing code works
- Integration test: Run full pipeline with no env vars or config file
- Ensure default behavior identical to v0.4.x

### 8. Configuration Validation Performance

**Decision**: < 100ms validation overhead at startup

**Rationale**:
- Startup-time validation acceptable (not query-time)
- Enterprise requirement: Fast deployment for CI/CD
- Fail fast principle: Better to fail at startup than during operation

**Validation Steps** (estimated timing):
1. Load config file (5-10ms)
2. Parse environment variables (1-2ms)
3. Validate vector dimension (10-20ms - single DB query)
4. Validate namespace permissions (20-30ms - two test queries)
5. Validate table schema (10-15ms - metadata query)

**Total**: 46-77ms (well within 100ms budget)

**Optimization**: Cache validation results for pipeline lifetime

### 9. Error Messages & Troubleshooting

**Decision**: Actionable error messages with migration guidance

**Rationale**:
- Constitution Principle VI: Explicit error handling
- Enterprise requirement: Self-service troubleshooting
- Reduces support burden

**Error Message Template**:
```
ConfigValidationError: Vector dimension mismatch detected

Configured dimension: 1024 (from VECTOR_DIMENSION env var)
Existing table dimension: 384 (RAG.Entities)

This mismatch will cause data corruption. To fix:

Option 1: Recreate tables with new dimension
  $ python -m iris_rag.cli.init_tables --drop --config your-config.yaml
  WARNING: This will delete all existing data!

Option 2: Change configuration to match existing tables
  Set VECTOR_DIMENSION=384 (or remove from config)

Option 3: Migrate data to new dimension (requires external tool)
  See: https://docs.iris-vector-rag.com/migration/vector-dimensions

Current table schema query:
  SELECT VECTOR_DIM FROM INFORMATION_SCHEMA.COLUMNS
  WHERE TABLE_NAME='Entities' AND COLUMN_NAME='embedding'
```

**Documentation Pages Required**:
- docs/troubleshooting/vector-dimensions.md
- docs/troubleshooting/namespace-permissions.md
- docs/troubleshooting/cloud-deployment.md

## Technology Stack Decisions

### Configuration Management
- **Library**: Extend existing ConfigurationManager (no new dependencies)
- **Validation**: pydantic for schema validation (already in dependencies)
- **Environment Variables**: python-dotenv (already in dependencies)
- **Rationale**: Reuse existing infrastructure, zero new dependencies

### Testing Framework
- **Contract Tests**: pytest with contract fixtures (TDD approach)
- **Integration Tests**: pytest with live IRIS database (Constitution Principle III)
- **Mocking**: unittest.mock for unit tests only (not integration tests)
- **Rationale**: Proven testing infrastructure in iris-vector-rag

## Integration Points

### 1. ConnectionManager (iris_vector_rag/core/connection.py)
**Changes Required**:
- Read IRIS_HOST, IRIS_PORT, IRIS_USERNAME, IRIS_PASSWORD from env vars
- Priority: env vars > config file > defaults
- No breaking changes to public API

### 2. ConfigurationManager (iris_vector_rag/config/manager.py)
**Changes Required**:
- Enhance _load_env_variables() to support IRIS connection vars
- Add validation hooks for vector dimensions, namespaces
- Maintain backward compatibility with existing env var pattern (RAG_)

### 3. Storage Layer (iris_vector_rag/services/storage.py)
**Changes Required**:
- Use configurable table schema prefix
- All SQL queries must use f"{schema}.TableName" pattern
- Ensure schema prefix applied consistently

### 4. CLI Tools (iris_vector_rag/cli/init_tables.py)
**Changes Required**:
- Fix --config flag to actually use provided config file
- Add validation before table creation
- Show configuration summary before proceeding

## Risks & Mitigations

### Risk 1: Configuration Complexity
**Mitigation**: Provide cloud-specific templates (aws.yaml, azure.yaml, local.yaml) with inline comments

### Risk 2: Breaking Changes
**Mitigation**: Comprehensive backward compatibility tests, all config optional with defaults

### Risk 3: Validation Overhead
**Mitigation**: Cache validation results, optimize DB queries, target < 100ms

### Risk 4: Documentation Lag
**Mitigation**: Auto-generate config schema docs from pydantic models, examples in integration tests

## Success Metrics (from Specification)

1. **SC-001**: Cloud deployment time reduces from 65 minutes to under 25 minutes (60% reduction)
2. **SC-002**: Zero code modifications required to deploy to AWS IRIS, Azure, or GCP
3. **SC-003**: Users can switch embedding models (384-dim to 1024-dim) through configuration only
4. **SC-006**: Existing local deployments continue working without any changes (100% backward compatible)
5. **SC-007**: init_tables() respects --config flag 100% of the time (currently 0%)

## Next Steps

With research complete, proceed to Phase 1 (Design & Contracts):
1. Create data-model.md (configuration entities)
2. Generate contract tests (15-20 tests for all requirements)
3. Create quickstart.md (cloud deployment guide)
4. Update CLAUDE.md with configuration patterns

---
**Research Status**: ✅ COMPLETE - All technical decisions documented, zero NEEDS CLARIFICATION remaining
