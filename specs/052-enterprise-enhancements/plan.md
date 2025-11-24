# Implementation Plan: Enterprise Enhancements for RAG System

**Branch**: `051-enterprise-enhancements` | **Date**: 2025-11-22 | **Spec**: [spec.md](spec.md)
**Input**: Feature specification from `/specs/051-enterprise-enhancements/spec.md`

## Summary

Add six enterprise-grade enhancements to iris-vector-rag to enable production deployments in multi-tenant, security-sensitive environments. Enhancements include configurable metadata filtering for tenant isolation, collection management APIs for operational visibility, RBAC integration for access control, OpenTelemetry instrumentation for observability, flexible connection management, and batch operations for performance. All enhancements maintain 100% backward compatibility.

**Technical Approach**: Extend existing `iris_vector_rag` modules (`storage`, `core`, `config`) with optional features that default to disabled. Use interface-based design (RBAC policy, telemetry providers) for extensibility without coupling to specific implementations.

## Technical Context

**Language/Version**: Python 3.10+ (existing codebase uses 3.10-3.12)
**Primary Dependencies**:
- Existing: InterSystems IRIS DB driver, sentence-transformers, openai, anthropic
- New: opentelemetry-api, opentelemetry-sdk (for monitoring), PyYAML (for config extension)

**Storage**: InterSystems IRIS database (RAG.SourceDocuments table - existing)
**Testing**: pytest (existing test infrastructure)
**Target Platform**: Linux/macOS/Windows servers (existing platform support)
**Project Type**: Single Python package with library + CLI (existing structure)
**Performance Goals**:
- Metadata filtering: <5ms overhead vs. unfiltered queries
- Collection list API: <2 seconds for 1000 collections
- Bulk loading: 10x+ speedup vs. one-by-one (target: 10K docs in <10s)
- Monitoring: <5% overhead when enabled, 0% when disabled

**Constraints**:
- **Backward Compatibility**: Zero breaking changes (all features opt-in)
- **Security**: Custom metadata fields must prevent SQL injection
- **Performance**: No degradation when features disabled
- **Data Isolation**: Multi-tenant filtering must guarantee complete isolation

**Scale/Scope**:
- Collections: 1-1,000 per deployment
- Documents per collection: 1,000-1,000,000
- Concurrent users: 1-100
- Metadata fields: Default 17 + up to 50 custom fields

## Constitution Check

*GATE: Must pass before Phase 0 research. Re-check after Phase 1 design.*

**Constitution**: `.specify/memory/constitution.md` (Version 1.0.0)

### Core Principles Check

✅ **I. Library-First Design**: All enhancements extend existing library modules (storage/, security/, monitoring/) without requiring CLI/API usage
✅ **II. .DAT Fixture-First Testing**: Integration tests for User Story 3 (RBAC with 100+ scenarios) will use .DAT fixtures
✅ **III. Test-First (TDD)**: Contract tests written before implementation for all 6 user stories (83 contract tests total)
✅ **IV. Backward Compatibility**: All features default to disabled, existing code continues working unchanged
✅ **V. InterSystems IRIS Integration**: All vector operations use IRIS native capabilities, batch operations use IRIS transactions
✅ **VI. Performance Standards**: Requirements defined (metadata filtering <5ms, bulk loading 10K docs <10s, monitoring <5% overhead)
✅ **VII. Observability**: OpenTelemetry integration planned (User Story 4) with structured logging
✅ **VIII. Security-First**: SQL injection prevention (User Story 1), RBAC policy interface (User Story 3), audit logging
✅ **IX. Simplicity**: No premature abstractions, interface-based design for extensibility only where needed

### Violations: **NONE**

All enhancements follow iris-vector-rag constitution principles. Library-first design, TDD approach, backward compatibility, and security requirements fully satisfied.

## Project Structure

### Documentation (this feature)

```text
specs/051-enterprise-enhancements/
├── spec.md              # Feature specification (completed)
├── plan.md              # This file (/speckit.plan command output)
├── research.md          # Phase 0 output (to be generated)
├── data-model.md        # Phase 1 output (to be generated)
├── quickstart.md        # Phase 1 output (to be generated)
├── contracts/           # Phase 1 output (to be generated)
│   ├── metadata_filtering.yaml    # Custom metadata filter API contracts
│   ├── collection_management.yaml # Collection CRUD API contracts
│   ├── rbac_integration.yaml      # RBAC policy interface contracts
│   ├── monitoring.yaml            # Telemetry API contracts
│   └── batch_operations.yaml      # Bulk loading API contracts
├── checklists/
│   └── requirements.md  # Spec quality checklist (completed)
└── tasks.md             # Phase 2 output (/speckit.tasks command - NOT created by /speckit.plan)
```

### Source Code (repository root)

```text
iris_vector_rag/
├── storage/
│   ├── vector_store_iris.py         # EXTEND: Add collection management methods
│   ├── metadata_filter_manager.py   # NEW: Custom filter validation
│   └── batch_operations.py          # NEW: Bulk loading logic
│
├── security/
│   ├── rbac.py                      # NEW: RBAC policy interface
│   └── __init__.py                  # NEW: Security module init
│
├── monitoring/
│   ├── telemetry.py                 # NEW: OpenTelemetry integration
│   ├── cost_tracking.py             # NEW: LLM cost calculation
│   └── __init__.py                  # NEW: Monitoring module init
│
├── core/
│   └── models.py                    # EXTEND: Add monitoring/collection entities
│
├── config/
│   ├── default_config.yaml          # EXTEND: Add new configuration sections
│   └── manager.py                   # EXTEND: Handle new config keys
│
└── cli/
    └── collection.py                # NEW: Collection management CLI commands

tests/
├── contract/
│   ├── test_metadata_filtering_contract.py    # NEW: Metadata filter contracts
│   ├── test_collection_management_contract.py # NEW: Collection API contracts
│   ├── test_rbac_integration_contract.py      # NEW: RBAC policy contracts
│   ├── test_monitoring_contract.py            # NEW: Telemetry contracts
│   └── test_batch_operations_contract.py      # NEW: Bulk loading contracts
│
├── integration/
│   ├── test_custom_metadata_filters.py        # NEW: E2E metadata filtering
│   ├── test_collection_lifecycle.py           # NEW: E2E collection mgmt
│   ├── test_rbac_enforcement.py               # NEW: E2E RBAC with mock policy
│   ├── test_telemetry_spans.py                # NEW: E2E monitoring
│   └── test_bulk_loading.py                   # NEW: E2E batch operations
│
└── unit/
    ├── storage/
    │   ├── test_metadata_filter_manager.py    # NEW: Unit tests for filter validation
    │   └── test_batch_operations.py           # NEW: Unit tests for bulk loading
    ├── security/
    │   └── test_rbac_policy.py                # NEW: Unit tests for RBAC interface
    └── monitoring/
        └── test_telemetry.py                  # NEW: Unit tests for monitoring
```

**Structure Decision**: Single Python package with library-first design (Option 1). Enhancements extend existing modules (`storage/`, `core/`, `config/`) and add new modules (`security/`, `monitoring/`) following iris-vector-rag's existing architecture. No web/mobile components needed.

## Complexity Tracking

**No violations requiring justification** - All enhancements follow existing patterns:
- New modules (`security/`, `monitoring/`) align with existing modular structure
- Interface-based RBAC design matches iris-vector-rag's extensibility patterns
- Configuration-driven approach consistent with current `config/` module

---

## Phase 0: Research & Technical Decisions

**Status**: Not started (to be completed by generating `research.md`)

### Research Questions

1. **Configuration Schema Design**
   - **Question**: How to extend YAML configuration for custom metadata fields without breaking existing configs?
   - **Research**: YAML merge strategies, backward-compatible schema evolution
   - **Output**: Configuration migration strategy

2. **RBAC Policy Interface**
   - **Question**: What interface design allows integration with diverse authorization systems (LDAP, OAuth, IRIS Security)?
   - **Research**: Policy-based access control patterns, Python ABC patterns for extensibility
   - **Output**: RBACPolicy abstract base class design

3. **OpenTelemetry Integration**
   - **Question**: How to instrument existing code with minimal intrusion? How to achieve 0% overhead when disabled?
   - **Research**: OpenTelemetry Python SDK, context propagation, conditional instrumentation
   - **Output**: Telemetry decorator/context manager patterns

4. **Bulk Loading Performance**
   - **Question**: What batch size and transaction strategy optimizes throughput without memory issues?
   - **Research**: IRIS batch INSERT performance, transaction isolation levels, connection pooling
   - **Output**: Batch operation algorithm with configurable strategies

5. **Metadata Schema Discovery**
   - **Question**: What sampling size provides accurate schema without performance impact?
   - **Research**: Statistical sampling for schema inference, JSON type inference algorithms
   - **Output**: Schema discovery sampling strategy

**Next Step**: Generate `research.md` with findings for each question.

---

## Phase 1: Design & Contracts

**Status**: Not started (dependencies: Phase 0 research complete)

### Deliverables

1. **data-model.md** - Entity models for:
   - `CustomMetadataField` (name, validation rules, merged with defaults)
   - `Collection` (id, document_count, size, created_at, last_updated)
   - `PermissionPolicy` (user, resource, operation, decision)
   - `MonitoringMetric` (operation_type, timestamp, duration, tokens, cost)
   - `BulkOperation` (id, status, progress, success_count, error_count)
   - `MetadataSchema` (field_name, type, frequency, examples, stats)

2. **contracts/** - OpenAPI/YAML specifications for:
   - `metadata_filtering.yaml` - Configuration schema + filter validation API
   - `collection_management.yaml` - CRUD operations for collections
   - `rbac_integration.yaml` - RBACPolicy interface methods
   - `monitoring.yaml` - Telemetry span creation + export APIs
   - `batch_operations.yaml` - Bulk loading API with progress tracking

3. **quickstart.md** - Step-by-step guide for:
   - Configuring custom metadata fields
   - Using collection management APIs
   - Implementing custom RBAC policies
   - Enabling monitoring
   - Running bulk document loads

**Next Step**: Generate these artifacts after research is complete.

---

## Phase 2: Task Breakdown (NOT part of this command)

**Status**: Deferred to `/speckit.tasks` command

Will break down implementation into atomic tasks with:
- Test file creation (TDD approach)
- Implementation file changes
- Configuration updates
- Documentation updates
- Integration testing

---

## Implementation Phases (High-Level)

### Phase 1: P1 Features (Week 1) - Foundation

**Features**:
1. Custom Metadata Filtering
2. Collection Management API
3. RBAC Integration Hooks

**Why First**: These are P1 (critical) and foundational - multi-tenancy and security unblock enterprise deployment.

**Deliverables**:
- `iris_vector_rag/storage/metadata_filter_manager.py` (new)
- `iris_vector_rag/storage/vector_store_iris.py` (extend with collection methods)
- `iris_vector_rag/security/rbac.py` (new)
- Contract tests for all three features
- Integration tests with real IRIS database

### Phase 2: P2 Features (Week 2) - Operations

**Features**:
4. OpenTelemetry Monitoring
5. Bulk Document Loading

**Why Second**: These are P2 (high value) but not blocking for initial deployment. Build on Phase 1 foundation.

**Deliverables**:
- `iris_vector_rag/monitoring/telemetry.py` (new)
- `iris_vector_rag/storage/batch_operations.py` (new)
- Contract tests for monitoring and bulk ops
- Performance benchmarks for bulk loading

### Phase 3: P3 Features (Week 3) - Polish

**Features**:
6. Metadata Schema Discovery

**Why Third**: This is P3 (nice-to-have) - improves developer experience but not required for production.

**Deliverables**:
- Extend `iris_vector_rag/storage/vector_store_iris.py` with sample_metadata_schema() method
- Contract tests for schema discovery
- CLI command for schema inspection

---

## Success Criteria Mapping

Each functional requirement maps to a success criterion:

| Feature | Functional Requirements | Success Criteria | Validation Method |
|---------|------------------------|------------------|-------------------|
| Custom Metadata Filtering | FR-001 to FR-005 | SC-001 | Configuration test + filter query test |
| Collection Management | FR-006 to FR-011 | SC-002 | API response time < 2s for 1000 collections |
| RBAC Integration | FR-012 to FR-017 | SC-003, SC-009 | Permission denial tests + error message clarity |
| Monitoring | FR-018 to FR-023 | SC-004, SC-010 | Telemetry data collection + overhead measurement |
| Bulk Loading | FR-024 to FR-028 | SC-005 | 10K document benchmark |
| Schema Discovery | FR-029 to FR-033 | SC-006 | Schema inference accuracy + response time |

**Overall Success**: SC-007 (95% adoption), SC-008 (zero breaking changes)

---

## Risk Mitigation

### Risk 1: Backward Compatibility Breakage
**Mitigation**: All features default to disabled. Run full existing test suite before/after each phase.
**Validation**: `pytest tests/` passes with zero changes to non-enhancement tests.

### Risk 2: Performance Regression
**Mitigation**: Benchmark core operations (query, index) before/after with features disabled.
**Validation**: <5% performance difference when features disabled (measured via pytest-benchmark).

### Risk 3: RBAC Policy Misuse
**Mitigation**: Provide secure-by-default example policies. Document security review checklist.
**Validation**: Penetration testing with common misconfigurations.

### Risk 4: Monitoring Overhead
**Mitigation**: Use conditional compilation patterns (e.g., `if self.telemetry_enabled:`) to avoid overhead.
**Validation**: Profile with and without monitoring enabled.

---

## Testing Strategy

### Contract Tests (TDD Approach)
**Purpose**: Define expected behavior before implementation

**Coverage**:
- 5 contract test files (one per enhancement)
- ~8-12 tests per file
- Total: ~40-60 contract tests

**Example** (`test_metadata_filtering_contract.py`):
```python
def test_custom_field_configuration():
    """Contract: Custom metadata fields merge with defaults"""
    # Test that adding custom field doesn't remove defaults
    # Test that duplicate field names are handled
    # Test that invalid field names are rejected
    pass  # Fails initially (TDD)

def test_metadata_filter_validation():
    """Contract: Only configured fields allowed in filters"""
    # Test that unconfigured field raises clear error
    # Test that error message lists allowed fields
    pass  # Fails initially (TDD)
```

### Integration Tests
**Purpose**: Validate end-to-end workflows with real IRIS database

**Coverage**:
- 5 integration test files (one per enhancement)
- ~10-15 tests per file
- Total: ~50-75 integration tests

**Infrastructure**: Use existing IRIS test database (docker-compose)

### Unit Tests
**Purpose**: Test individual components in isolation

**Coverage**:
- 4 unit test files for new modules
- ~15-20 tests per file
- Total: ~60-80 unit tests

---

## Configuration Changes

### New Configuration Sections

**File**: `iris_vector_rag/config/default_config.yaml`

```yaml
# NEW: Custom Metadata Filtering
storage:
  iris:
    custom_filter_keys:
      - tenant_id
      - collection_id
      - security_level
      # Users add their fields here

# NEW: Collection Management (no config needed - uses existing storage config)

# NEW: RBAC Integration (optional)
security:
  rbac:
    enabled: false  # Disabled by default
    policy_class: null  # User provides: "myapp.security.MyRBACPolicy"

# NEW: Monitoring
telemetry:
  enabled: false  # Disabled by default
  service_name: "iris-rag-api"
  otlp:
    endpoint: "http://localhost:4318"
  sampling:
    ratio: 0.1  # 10% sampling

# NEW: Bulk Operations
batch_operations:
  default_batch_size: 1000
  show_progress: false  # Enable for CLI
  error_handling: "continue"  # continue|stop|rollback
```

---

## API Surface Changes

### New Public APIs

**iris_vector_rag.storage.IRISVectorStore** (extended):
```python
# Collection Management
def create_collection(collection_id: str, metadata: Optional[Dict] = None) -> bool
def list_collections() -> List[Dict[str, Any]]
def get_collection_info(collection_id: str) -> Dict[str, Any]
def delete_collection(collection_id: str) -> int
def collection_exists(collection_id: str) -> bool

# Bulk Operations
def add_documents_batch(
    documents: List[Document],
    embeddings: Optional[List[List[float]]] = None,
    batch_size: int = 1000,
    show_progress: bool = False,
    error_handling: str = "continue"
) -> Dict[str, Any]

# Schema Discovery
def sample_metadata_schema(
    collection_id: Optional[str] = None,
    sample_size: int = 100
) -> Dict[str, Dict[str, Any]]
```

**iris_vector_rag.security.RBACPolicy** (new interface):
```python
class RBACPolicy(ABC):
    @abstractmethod
    def check_collection_access(user: str, collection_id: Optional[str], operation: str) -> bool

    @abstractmethod
    def filter_documents(user: str, documents: List[Document]) -> List[Document]
```

**iris_vector_rag.monitoring.Telemetry** (new):
```python
def configure_telemetry(
    enabled: bool = True,
    service_name: str = "iris-rag",
    endpoint: str = "http://localhost:4318"
)

@contextmanager
def trace_operation(operation_name: str, **attributes) -> Span
```

---

## Dependencies

### New Python Packages

**File**: `pyproject.toml` (to be updated)

```toml
[project.dependencies]
# Existing dependencies (unchanged)
# ...

# NEW: Monitoring (optional extra)
opentelemetry-api = {version = ">=1.20.0", optional = true}
opentelemetry-sdk = {version = ">=1.20.0", optional = true}
opentelemetry-exporter-otlp = {version = ">=1.20.0", optional = true}

# NEW: Batch operations (built-in tqdm)
tqdm = ">=4.66.0"

[project.optional-dependencies]
monitoring = [
    "opentelemetry-api>=1.20.0",
    "opentelemetry-sdk>=1.20.0",
    "opentelemetry-exporter-otlp>=1.20.0"
]

enterprise = [
    "opentelemetry-api>=1.20.0",
    "opentelemetry-sdk>=1.20.0",
    "opentelemetry-exporter-otlp>=1.20.0",
    "tqdm>=4.66.0"
]
```

**Installation**:
```bash
# Core features (no new dependencies)
pip install iris-vector-rag

# With monitoring
pip install iris-vector-rag[monitoring]

# All enterprise features
pip install iris-vector-rag[enterprise]
```

---

## Migration Path for Existing Users

### Zero Changes Required

**Scenario**: Existing iris-vector-rag users upgrade to version with enhancements

**Impact**: **NONE** - All features default to disabled

**Example**:
```python
# Existing code continues working unchanged
from iris_vector_rag import create_pipeline

pipeline = create_pipeline('basic')
results = pipeline.query("What is diabetes?", top_k=5)
# ✅ No changes needed
```

### Opt-In Adoption

**Scenario**: User wants to enable custom metadata filtering

**Migration**:
1. Add config file or update existing:
```yaml
storage:
  iris:
    custom_filter_keys:
      - tenant_id
```

2. Use in queries:
```python
results = pipeline.query(
    "What is diabetes?",
    top_k=5,
    metadata_filter={"tenant_id": "tenant_123"}  # ✅ Now allowed
)
```

**Validation**: If unconfigured field used, clear error message:
```
VectorStoreConfigurationError: Filter key 'tenant_id' not in allowed list.
Add to storage:iris:custom_filter_keys in config.
Allowed keys: [category, year, source_type, ...]
```

---

## Development Workflow

### Phase 0: Research (Current)
1. Generate `research.md` with technical decisions
2. Review research findings
3. Approve technical approach before Phase 1

### Phase 1: Design
1. Generate `data-model.md` with entity models
2. Generate contract YAML files in `contracts/`
3. Generate `quickstart.md` with usage examples
4. Review design artifacts
5. Run `.specify/scripts/bash/update-agent-context.sh claude` to update agent context

### Phase 2: Implementation (via `/speckit.tasks`)
1. Generate `tasks.md` with atomic implementation tasks
2. For each task:
   - Write contract test (TDD)
   - Run test (should fail)
   - Implement feature
   - Run test (should pass)
   - Write integration test
   - Run integration test
3. Run full test suite after each phase
4. Benchmark performance after each phase

---

## Appendix: Technical Debt & Future Work

### Known Limitations

1. **Metadata Consistency**: Assumes metadata fields have consistent types within a collection. Schema evolution not handled.
   - **Mitigation**: Document assumption in API docs
   - **Future**: Add schema migration utilities

2. **RBAC Policy Caching**: No caching of permission decisions - may impact performance at scale.
   - **Mitigation**: Document that users should implement caching in their policy
   - **Future**: Add optional built-in policy cache

3. **Monitoring Data Retention**: No automatic cleanup of telemetry data.
   - **Mitigation**: Document manual cleanup procedures
   - **Future**: Add automatic retention policies

### Follow-Up Enhancements (Not in Scope)

- **Dynamic RBAC Policies**: Runtime policy updates without restart
- **Advanced Schema Migration**: Automatic metadata type conversions
- **Distributed Tracing**: Cross-service trace propagation
- **Automated Performance Tuning**: ML-based batch size optimization

---

## Conclusion

This implementation plan provides a phased approach to adding six enterprise enhancements while maintaining 100% backward compatibility. All enhancements follow iris-vector-rag's existing architecture patterns and use interface-based design for extensibility.

**Next Steps**:
1. **Execute Phase 0**: Generate `research.md` to resolve technical questions
2. **Execute Phase 1**: Generate `data-model.md`, `contracts/`, and `quickstart.md`
3. **Run `/speckit.tasks`**: Generate atomic implementation tasks

**Ready for**: Research phase (Phase 0)
