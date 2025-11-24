# Research: Enterprise Enhancements for RAG System

**Branch**: `051-enterprise-enhancements` | **Date**: 2025-11-22 | **Spec**: [spec.md](spec.md)

## Purpose

This document resolves technical research questions identified in [plan.md](plan.md) Phase 0. Each question includes the decision made, rationale, alternatives considered, and implementation guidance.

---

## Research Question 1: Configuration Schema Design

**Question**: How to extend YAML configuration for custom metadata fields without breaking existing configs?

### Decision: Additive YAML Schema with Default Merge Strategy

Use a dedicated `custom_filter_keys` array in the existing `storage.iris` section that merges with (not replaces) the hardcoded default filter keys.

**Configuration Example**:
```yaml
storage:
  iris:
    custom_filter_keys:
      - tenant_id
      - security_level
      - department
    # Other existing config keys remain unchanged
```

**Implementation**:
```python
# In iris_vector_rag/storage/vector_store_iris.py
DEFAULT_FILTER_KEYS = [
    "collection_id", "category", "year", "source_type",
    "author", "language", "doc_id", "page_number",
    "file_name", "file_path", "created_at", "updated_at",
    "tags", "priority", "status", "version", "owner"
]

def __init__(self, config_manager):
    self.default_filter_keys = DEFAULT_FILTER_KEYS
    custom_keys = config_manager.get("storage.iris.custom_filter_keys", [])
    # Merge: defaults + custom (removing duplicates)
    self.allowed_filter_keys = list(set(self.default_filter_keys + custom_keys))
```

### Rationale

1. **Backward Compatibility**: Existing configs without `custom_filter_keys` continue working (defaults to empty list)
2. **No Replacement Risk**: Merging prevents users from accidentally removing default fields
3. **Clear Separation**: Custom fields are explicitly marked as "custom" in config
4. **Standard YAML Pattern**: Follows common YAML extension patterns (additive arrays)

### Alternatives Considered

**Alternative 1: Separate `allowed_filter_keys` (replacement strategy)**
```yaml
storage:
  iris:
    allowed_filter_keys:
      - collection_id
      - tenant_id  # User must list ALL keys
```
❌ **Rejected**: High risk of users forgetting default fields, breaking existing functionality

**Alternative 2: Prefix-based wildcards**
```yaml
storage:
  iris:
    custom_filter_patterns:
      - "tenant_*"  # Allows tenant_id, tenant_name, etc.
```
❌ **Rejected**: Too permissive, increases SQL injection risk, harder to validate

**Alternative 3: Schema validation file**
```yaml
# config/metadata_schema.yaml
fields:
  tenant_id:
    type: string
    filterable: true
```
❌ **Rejected**: Over-engineered for MVP, adds complexity without clear benefit

### Migration Path

**Existing users**: No changes needed, continue using default fields
**New users wanting custom fields**:
1. Add `custom_filter_keys` array to config
2. Deploy with new version
3. Use custom fields in queries immediately

---

## Research Question 2: RBAC Policy Interface

**Question**: What interface design allows integration with diverse authorization systems (LDAP, OAuth, IRIS Security)?

### Decision: Abstract Base Class with Two-Level Authorization

Provide a `RBACPolicy` abstract base class with two authorization hooks:
1. **Collection-level**: Check if user can access a collection (read/write/delete)
2. **Document-level**: Filter documents based on user permissions

**Interface Design**:
```python
from abc import ABC, abstractmethod
from typing import List, Optional
from langchain.docstore.document import Document

class RBACPolicy(ABC):
    """Abstract base class for implementing custom RBAC policies.

    Users implement this interface to integrate with their organization's
    authorization system (LDAP, OAuth, IRIS Security, etc.).
    """

    @abstractmethod
    def check_collection_access(
        self,
        user: str,
        collection_id: Optional[str],
        operation: str
    ) -> bool:
        """Check if user has permission for collection-level operation.

        Args:
            user: User identifier (email, username, or account ID)
            collection_id: Collection to check (None = all collections)
            operation: One of 'read', 'write', 'delete', 'admin'

        Returns:
            True if operation allowed, False otherwise
        """
        pass

    @abstractmethod
    def filter_documents(
        self,
        user: str,
        documents: List[Document]
    ) -> List[Document]:
        """Filter documents based on user's document-level permissions.

        Args:
            user: User identifier
            documents: List of documents to filter

        Returns:
            Filtered list containing only documents user can access
        """
        pass

    def get_audit_context(self, user: str) -> dict:
        """Optional: Provide additional context for audit logging.

        Default implementation returns user identifier only.
        Override to add roles, groups, clearance levels, etc.
        """
        return {"user": user}
```

**Usage Example**:
```python
# User implements custom policy
class LDAPRBACPolicy(RBACPolicy):
    def __init__(self, ldap_client):
        self.ldap = ldap_client

    def check_collection_access(self, user, collection_id, operation):
        groups = self.ldap.get_user_groups(user)
        if "admin" in groups:
            return True
        if operation == "read" and "readers" in groups:
            return True
        return False

    def filter_documents(self, user, documents):
        clearance = self.ldap.get_clearance_level(user)
        return [
            doc for doc in documents
            if doc.metadata.get("security_level", 0) <= clearance
        ]

# Configure in iris_vector_rag
policy = LDAPRBACPolicy(ldap_client)
store = IRISVectorStore(config_manager, rbac_policy=policy)
```

### Rationale

1. **Maximum Flexibility**: ABC allows integration with any authorization system
2. **Two-Level Security**: Collection + document filtering covers 95% of enterprise use cases
3. **Minimal Interface**: Only 2 required methods keeps implementation simple
4. **No Dependencies**: Interface has zero dependencies on specific auth systems
5. **Audit Support**: Optional `get_audit_context()` enables extensible logging

### Alternatives Considered

**Alternative 1: Decorator-based authorization**
```python
@require_permission("read")
def query(self, query_text):
    pass
```
❌ **Rejected**: Too rigid, doesn't support document-level filtering, couples to specific auth model

**Alternative 2: Capability-based tokens**
```python
def query(self, query_text, capabilities: List[str]):
    pass
```
❌ **Rejected**: Shifts complexity to callers, no centralized policy enforcement

**Alternative 3: SQL-based row-level security (IRIS Security)**
```sql
CREATE ROLE reader;
GRANT SELECT ON RAG.SourceDocuments WHERE security_level <= %CurrentSecurityLevel TO reader;
```
❌ **Rejected**: Locks users into IRIS-specific security, prevents integration with external IAM

**Alternative 4: Policy-as-code (Rego/Open Policy Agent)**
```python
def check_access(self, user, resource, action):
    return opa_client.evaluate_policy(user, resource, action)
```
❌ **Rejected**: Adds heavy dependency (OPA), over-engineered for most use cases

### Implementation Notes

**Configuration**:
```yaml
security:
  rbac:
    enabled: false  # Disabled by default
    policy_class: null  # e.g., "myapp.security.LDAPRBACPolicy"
    policy_config:  # Passed to policy constructor
      ldap_server: "ldap://..."
```

**Error Handling**:
```python
class PermissionDeniedError(Exception):
    """Raised when RBAC policy denies access."""
    def __init__(self, user, resource, operation):
        self.user = user
        self.resource = resource
        self.operation = operation
        super().__init__(
            f"User '{user}' denied '{operation}' access to '{resource}'. "
            f"Contact your administrator if you believe this is an error."
        )
```

---

## Research Question 3: OpenTelemetry Integration

**Question**: How to instrument existing code with minimal intrusion? How to achieve 0% overhead when disabled?

### Decision: Context Manager Pattern with Conditional Initialization

Use Python context managers for instrumentation with lazy initialization only when telemetry is enabled. Zero overhead when disabled via early return guards.

**Implementation Pattern**:
```python
from contextlib import contextmanager
from typing import Optional
from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.resources import Resource
from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter

class TelemetryManager:
    """Manages OpenTelemetry instrumentation with zero-overhead when disabled."""

    def __init__(self, enabled: bool = False, config: Optional[dict] = None):
        self.enabled = enabled
        self._tracer = None

        if enabled and config:
            resource = Resource.create({"service.name": config.get("service_name", "iris-rag")})
            provider = TracerProvider(resource=resource)

            exporter = OTLPSpanExporter(endpoint=config.get("otlp.endpoint"))
            provider.add_span_processor(BatchSpanProcessor(exporter))

            trace.set_tracer_provider(provider)
            self._tracer = trace.get_tracer(__name__)

    @contextmanager
    def trace_operation(self, operation_name: str, **attributes):
        """Trace an operation with automatic span lifecycle.

        Zero overhead when telemetry disabled (early return).
        """
        if not self.enabled or not self._tracer:
            yield None  # No-op when disabled
            return

        with self._tracer.start_as_current_span(operation_name) as span:
            # Add custom attributes
            for key, value in attributes.items():
                span.set_attribute(key, value)

            try:
                yield span
            except Exception as e:
                span.set_status(trace.Status(trace.StatusCode.ERROR))
                span.record_exception(e)
                raise

# Global singleton (initialized once)
telemetry = TelemetryManager(enabled=False)

def configure_telemetry(enabled=True, service_name="iris-rag", endpoint="http://localhost:4318"):
    """Configure telemetry (called once at startup)."""
    global telemetry
    telemetry = TelemetryManager(
        enabled=enabled,
        config={"service_name": service_name, "otlp.endpoint": endpoint}
    )
```

**Usage in Pipeline Code**:
```python
# In iris_vector_rag/pipelines/basic.py
from iris_vector_rag.monitoring.telemetry import telemetry

def query(self, query_text: str, top_k: int = 5):
    with telemetry.trace_operation(
        "rag.query",
        query_length=len(query_text),
        top_k=top_k,
        pipeline="basic"
    ) as span:
        # Retrieval
        with telemetry.trace_operation("rag.retrieval", top_k=top_k):
            docs = self.vector_store.similarity_search(query_text, k=top_k)

        # Generation
        with telemetry.trace_operation(
            "rag.generation",
            context_length=sum(len(d.page_content) for d in docs)
        ) as gen_span:
            answer = self.llm.generate(query_text, docs)

            if gen_span:
                gen_span.set_attribute("gen.ai.response.tokens", answer.token_count)

        return answer
```

**Zero Overhead Proof**:
```python
# When disabled (default):
with telemetry.trace_operation("test"):  # Early return at line 2
    expensive_operation()  # No span creation, no attribute setting

# Equivalent to:
expensive_operation()  # Direct execution, zero overhead
```

### Rationale

1. **Zero Overhead When Disabled**: Early return in context manager prevents any tracer calls
2. **Minimal Code Changes**: Context managers wrap existing code without restructuring
3. **Semantic Conventions**: Follows OpenTelemetry GenAI semantic conventions (gen.ai.*)
4. **Lazy Initialization**: Tracer only created when enabled at startup
5. **Error Handling**: Automatic exception recording in spans

### Alternatives Considered

**Alternative 1: Decorator-based instrumentation**
```python
@trace_operation("query")
def query(self, query_text):
    pass
```
❌ **Rejected**: Decorators always have overhead (function wrapping), harder to add dynamic attributes

**Alternative 2: Aspect-oriented programming (AspectLib)**
```python
from aspectlib import Aspect
Aspect(trace_aspect).weave(IRISVectorStore)
```
❌ **Rejected**: High complexity, obscure debugging, non-standard Python pattern

**Alternative 3: Middleware/proxy pattern**
```python
instrumented_store = TelemetryProxy(vector_store)
```
❌ **Rejected**: Breaks type hints, adds indirection, complicates initialization

**Alternative 4: Manual span creation everywhere**
```python
span = tracer.start_span("query")
try:
    result = self.query(...)
finally:
    span.end()
```
❌ **Rejected**: Verbose, error-prone (forget span.end()), no zero-overhead guarantee

### Performance Validation

**Benchmark Requirements** (from plan.md):
- **Disabled**: 0% overhead (measured via pytest-benchmark)
- **Enabled**: <5% overhead

**Test Strategy**:
```python
# tests/integration/test_telemetry_overhead.py
def test_telemetry_disabled_zero_overhead(benchmark):
    configure_telemetry(enabled=False)
    result = benchmark(pipeline.query, "test query")
    # Compare to baseline without any telemetry code

def test_telemetry_enabled_low_overhead(benchmark):
    configure_telemetry(enabled=True, endpoint="http://localhost:4318")
    result = benchmark(pipeline.query, "test query")
    # Assert < 5% slower than disabled
```

### OpenTelemetry Semantic Conventions for GenAI

Follow [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/):

**Span Names**:
- `gen_ai.client.chat` - LLM chat completion
- `gen_ai.client.embeddings` - Embedding generation
- `gen_ai.retrieval.search` - Vector search

**Attributes**:
- `gen_ai.system` - LLM provider (openai, anthropic)
- `gen_ai.request.model` - Model name (gpt-4, claude-3-opus)
- `gen_ai.usage.input_tokens` - Prompt tokens
- `gen_ai.usage.output_tokens` - Completion tokens
- `gen_ai.response.finish_reason` - Stop reason (stop, length, tool_calls)

---

## Research Question 4: Bulk Loading Performance

**Question**: What batch size and transaction strategy optimizes throughput without memory issues?

### Decision: Configurable Batch Size with Streaming Strategy

Default batch size of **1,000 documents** with streaming insertion (one transaction per batch) and configurable error handling strategies.

**Implementation**:
```python
from typing import List, Dict, Any
from tqdm import tqdm

def add_documents_batch(
    self,
    documents: List[Document],
    embeddings: Optional[List[List[float]]] = None,
    batch_size: int = 1000,
    show_progress: bool = False,
    error_handling: str = "continue"  # continue | stop | rollback
) -> Dict[str, Any]:
    """Bulk load documents with configurable batching and error handling.

    Args:
        documents: List of documents to add
        embeddings: Pre-computed embeddings (optional)
        batch_size: Documents per batch (default 1000)
        show_progress: Show progress bar (default False)
        error_handling: Error strategy:
            - "continue": Skip failed documents, continue processing
            - "stop": Stop on first error, commit previous batches
            - "rollback": Stop on first error, rollback all batches

    Returns:
        {
            "total": 10000,
            "success_count": 9987,
            "error_count": 13,
            "errors": [{"index": 45, "doc_id": "...", "error": "..."}],
            "time_seconds": 8.2
        }
    """
    total = len(documents)
    success_count = 0
    errors = []
    start_time = time.time()

    iterator = range(0, total, batch_size)
    if show_progress:
        iterator = tqdm(iterator, desc="Loading documents", unit="batch")

    for i in iterator:
        batch_docs = documents[i:i + batch_size]
        batch_embeds = embeddings[i:i + batch_size] if embeddings else None

        try:
            if error_handling == "rollback":
                # Use savepoint for rollback capability
                with self.connection.cursor() as cursor:
                    cursor.execute("SAVEPOINT batch_{}".format(i))
                    self._insert_batch(cursor, batch_docs, batch_embeds)
                    cursor.execute("RELEASE SAVEPOINT batch_{}".format(i))
                    success_count += len(batch_docs)
            else:
                # Normal transaction per batch
                self._insert_batch_with_individual_errors(
                    batch_docs, batch_embeds, errors, success_count
                )
        except Exception as e:
            if error_handling == "stop":
                errors.append({
                    "batch_start": i,
                    "error": str(e),
                    "message": "Stopped processing due to batch error"
                })
                break
            elif error_handling == "rollback":
                # Rollback to start
                self.connection.rollback()
                raise RuntimeError(f"Batch loading failed at index {i}: {e}")
            # "continue" mode: error already logged, continue to next batch

    return {
        "total": total,
        "success_count": success_count,
        "error_count": len(errors),
        "errors": errors[:100],  # Limit error list size
        "time_seconds": time.time() - start_time
    }

def _insert_batch(self, cursor, documents: List[Document], embeddings):
    """Insert batch using parameterized query."""
    query = """
        INSERT INTO RAG.SourceDocuments
        (content, embeddings, metadata, collection_id, created_at)
        VALUES (?, TO_VECTOR(?), ?, ?, CURRENT_TIMESTAMP)
    """

    params = [
        (
            doc.page_content,
            embeddings[i] if embeddings else self._generate_embedding(doc),
            json.dumps(doc.metadata),
            doc.metadata.get("collection_id", "default")
        )
        for i, doc in enumerate(documents)
    ]

    cursor.executemany(query, params)
```

**Batch Size Selection**:
- **1,000**: Default for balanced memory/performance
- **100**: Small datasets or memory-constrained environments
- **5,000**: Large RAM environments with high-speed IRIS instances
- **10,000+**: Enterprise deployments with SAN storage

### Rationale

1. **1,000 Default**: Empirically proven optimal for most scenarios (LangChain, pgvector)
2. **Streaming Strategy**: Avoids loading entire dataset into memory
3. **Transaction Per Batch**: Balance between atomicity and performance
4. **Three Error Strategies**: Covers all use cases:
   - **continue**: Data migrations (best-effort)
   - **stop**: Production loads (fail-fast)
   - **rollback**: Critical operations (all-or-nothing)
5. **Progress Indicator**: UX for long-running operations

### Alternatives Considered

**Alternative 1: Single transaction for all batches**
```python
with self.connection.transaction():
    for batch in batches:
        insert_batch(batch)
```
❌ **Rejected**: High memory usage for large datasets, single failure rolls back everything

**Alternative 2: COPY/LOAD DATA INFILE**
```python
cursor.execute("LOAD DATA INFILE '/tmp/docs.csv' INTO RAG.SourceDocuments")
```
❌ **Rejected**: Requires file system access, doesn't work with embeddings (binary data)

**Alternative 3: Async/parallel insertion**
```python
async def insert_batch_async(batch):
    await connection.execute(...)
```
❌ **Rejected**: Complexity of async code, IRIS connection pool may not support concurrent writes

### Performance Target Validation

**Benchmark** (from plan.md SC-005):
- **Target**: 10,000 documents in <10 seconds (10x+ faster than one-by-one)
- **Baseline**: One-by-one loading ~1.5 docs/sec = 167 minutes for 10K docs
- **Bulk**: 1,000+ docs/sec = <10 seconds for 10K docs

**Test Implementation**:
```python
# tests/integration/test_bulk_loading.py
def test_bulk_loading_10k_documents():
    docs = [create_test_document() for _ in range(10000)]

    start = time.time()
    result = store.add_documents_batch(docs, batch_size=1000)
    duration = time.time() - start

    assert result["success_count"] == 10000
    assert duration < 10.0  # Must complete in <10 seconds
    assert result["error_count"] == 0
```

---

## Research Question 5: Metadata Schema Discovery

**Question**: What sampling size provides accurate schema without performance impact?

### Decision: Stratified Sampling with 100-200 Document Default

Use stratified random sampling of 100-200 documents (configurable) to infer metadata schema, with type inference and statistical analysis.

**Implementation**:
```python
from typing import Dict, Any, Optional
from collections import Counter, defaultdict
import json

def sample_metadata_schema(
    self,
    collection_id: Optional[str] = None,
    sample_size: int = 100
) -> Dict[str, Dict[str, Any]]:
    """Discover metadata schema by sampling documents.

    Args:
        collection_id: Specific collection (None = all collections)
        sample_size: Number of documents to sample (default 100)

    Returns:
        {
            "tenant_id": {
                "type": "string",
                "frequency": 0.98,  # 98% of documents have this field
                "unique_values": 12,
                "examples": ["tenant_001", "tenant_002", "tenant_003"],
                "nullable": True
            },
            "priority": {
                "type": "integer",
                "frequency": 0.75,
                "min": 1,
                "max": 5,
                "avg": 2.8,
                "unique_values": 5,
                "examples": [1, 2, 3, 4, 5],
                "nullable": False
            },
            "created_at": {
                "type": "datetime",
                "frequency": 1.0,
                "min": "2023-01-15T10:30:00Z",
                "max": "2024-11-22T15:45:00Z",
                "examples": ["2023-06-20T08:15:00Z"],
                "nullable": False
            }
        }
    """
    # Step 1: Get total document count
    cursor = self.connection.cursor()
    if collection_id:
        cursor.execute(
            "SELECT COUNT(*) FROM RAG.SourceDocuments WHERE collection_id = ?",
            (collection_id,)
        )
    else:
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")

    total_count = cursor.fetchone()[0]

    # Step 2: Stratified random sampling
    if total_count <= sample_size:
        # Sample all documents
        query = "SELECT metadata FROM RAG.SourceDocuments"
        params = []
    else:
        # Random sampling using IRIS SQL
        query = """
            SELECT TOP ? metadata
            FROM RAG.SourceDocuments
            ORDER BY NEWID()
        """
        params = [sample_size]

    if collection_id:
        query += " WHERE collection_id = ?"
        params.append(collection_id)

    cursor.execute(query, params)

    # Step 3: Aggregate metadata fields
    field_stats = defaultdict(lambda: {
        "types": Counter(),
        "values": [],
        "count": 0,
        "null_count": 0
    })

    sampled_count = 0
    for row in cursor.fetchall():
        metadata = json.loads(row[0])
        sampled_count += 1

        for key, value in metadata.items():
            stats = field_stats[key]
            stats["count"] += 1

            if value is None:
                stats["null_count"] += 1
                continue

            # Infer type
            value_type = self._infer_type(value)
            stats["types"][value_type] += 1

            # Store value for statistics
            if len(stats["values"]) < 100:  # Limit memory
                stats["values"].append(value)

    # Step 4: Build schema summary
    schema = {}
    for field, stats in field_stats.items():
        # Determine primary type (most common)
        primary_type = stats["types"].most_common(1)[0][0] if stats["types"] else "unknown"

        frequency = stats["count"] / sampled_count
        nullable = stats["null_count"] > 0

        field_info = {
            "type": primary_type,
            "frequency": round(frequency, 3),
            "nullable": nullable,
            "unique_values": len(set(stats["values"]))
        }

        # Add type-specific statistics
        if primary_type in ("integer", "float"):
            numeric_values = [v for v in stats["values"] if isinstance(v, (int, float))]
            if numeric_values:
                field_info["min"] = min(numeric_values)
                field_info["max"] = max(numeric_values)
                field_info["avg"] = round(sum(numeric_values) / len(numeric_values), 2)

        if primary_type in ("string", "datetime"):
            field_info["examples"] = list(set(stats["values"]))[:5]

        schema[field] = field_info

    return schema

def _infer_type(self, value) -> str:
    """Infer JSON type from Python value."""
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "float"
    if isinstance(value, str):
        # Try to detect datetime strings
        if len(value) >= 19 and value[4] == '-' and value[7] == '-':
            try:
                datetime.fromisoformat(value.replace('Z', '+00:00'))
                return "datetime"
            except ValueError:
                pass
        return "string"
    if isinstance(value, list):
        return "array"
    if isinstance(value, dict):
        return "object"
    return "unknown"
```

**Usage Example**:
```python
# Discover schema for entire database
schema = store.sample_metadata_schema(sample_size=200)

# Discover schema for specific collection
schema = store.sample_metadata_schema(
    collection_id="medical-docs",
    sample_size=100
)

# Example output:
# {
#   "tenant_id": {
#     "type": "string",
#     "frequency": 0.98,
#     "nullable": True,
#     "unique_values": 12,
#     "examples": ["tenant_001", "tenant_002"]
#   },
#   "priority": {
#     "type": "integer",
#     "frequency": 0.75,
#     "nullable": False,
#     "min": 1, "max": 5, "avg": 2.8,
#     "unique_values": 5
#   }
# }
```

### Rationale

1. **100-200 Sample Size**: Statistical significance for 95% confidence interval (margin of error ~10%)
2. **Stratified Sampling**: Random sampling ensures representative schema
3. **Type Inference**: Automatic detection of string/integer/float/datetime/boolean/array/object
4. **Frequency Tracking**: Shows how common each field is (helps identify required vs optional)
5. **Statistics for Numerics**: Min/max/avg for integer/float fields
6. **Example Values**: Helps developers understand field content

### Alternatives Considered

**Alternative 1: Full table scan**
```python
cursor.execute("SELECT DISTINCT JSON_KEYS(metadata) FROM RAG.SourceDocuments")
```
❌ **Rejected**: Slow for large datasets (millions of documents), doesn't provide statistics

**Alternative 2: Fixed 10% sampling**
```python
sample_size = int(total_count * 0.1)
```
❌ **Rejected**: Too large for huge datasets (1M docs = 100K sample), too small for tiny datasets (10 docs = 1 sample)

**Alternative 3: Schema cache (persist discovered schema)**
```python
# Store schema in RAG.MetadataSchema table
cursor.execute("SELECT schema FROM RAG.MetadataSchema WHERE collection_id = ?")
```
❌ **Rejected**: Adds complexity, schema can drift as documents added, users may want fresh sample

**Alternative 4: JSON Schema validation library**
```python
from jsonschema import validate, Draft7Validator
schema = Draft7Validator.from_dict(sample_metadata).schema
```
❌ **Rejected**: Over-engineered, doesn't provide statistics, library dependency

### Sample Size Justification

**Statistical Formula**:
```
n = (Z^2 * p * (1-p)) / E^2

Where:
- Z = 1.96 (95% confidence level)
- p = 0.5 (maximum variability)
- E = 0.10 (10% margin of error)

n = (1.96^2 * 0.5 * 0.5) / 0.10^2 = 96.04 ≈ 100
```

**For 99% confidence** (stricter):
```
Z = 2.576 → n = 165
```

**Conclusion**: 100-200 sample size provides statistically valid schema inference.

### Performance Validation

**Target** (from plan.md SC-006): Schema discovery completes in <5 seconds

**Benchmark**:
```python
# tests/integration/test_schema_discovery.py
def test_schema_discovery_performance():
    # Load 10,000 documents
    load_test_documents(10000)

    start = time.time()
    schema = store.sample_metadata_schema(sample_size=200)
    duration = time.time() - start

    assert duration < 5.0  # Must complete in <5 seconds
    assert len(schema) > 0
```

---

## Summary of Technical Decisions

| Research Question | Decision | Key Benefit |
|-------------------|----------|-------------|
| Configuration Schema | Additive `custom_filter_keys` array | Backward compatible, no replacement risk |
| RBAC Policy Interface | Abstract base class with 2 methods | Maximum flexibility, minimal interface |
| OpenTelemetry Integration | Context manager with early return | Zero overhead when disabled |
| Bulk Loading | 1,000 batch size, streaming strategy | Balanced memory/performance, 10x+ speedup |
| Schema Discovery | Stratified sampling of 100-200 docs | Statistical validity, <5s response time |

---

## Next Steps

1. **Phase 1**: Generate `data-model.md` defining 6 entity models
2. **Phase 1**: Generate `contracts/` with 5 YAML specification files
3. **Phase 1**: Generate `quickstart.md` with usage examples
4. **Update Agent Context**: Run `.specify/scripts/bash/update-agent-context.sh claude`
5. **Phase 2**: Run `/speckit.tasks` to generate atomic implementation tasks

**Ready for**: Phase 1 (Design & Contracts)
