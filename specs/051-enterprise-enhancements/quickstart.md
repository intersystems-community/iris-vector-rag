# Quickstart Guide: Enterprise Enhancements for RAG System

**Branch**: `051-enterprise-enhancements` | **Date**: 2025-11-22 | **Spec**: [spec.md](spec.md)

## Introduction

This guide provides step-by-step instructions for using the six enterprise enhancements added to iris-vector-rag. Each section includes:
- **Purpose**: What problem it solves
- **Configuration**: How to enable the feature
- **Usage Examples**: Python code and CLI commands
- **Common Patterns**: Best practices and tips

All enhancements are **opt-in** (disabled by default) for 100% backward compatibility.

---

## Table of Contents

1. [Custom Metadata Filtering](#1-custom-metadata-filtering) (P1)
2. [Collection Management](#2-collection-management) (P1)
3. [RBAC Integration](#3-rbac-integration) (P1)
4. [OpenTelemetry Monitoring](#4-opentelemetry-monitoring) (P2)
5. [Bulk Document Loading](#5-bulk-document-loading) (P2)
6. [Metadata Schema Discovery](#6-metadata-schema-discovery) (P3)

---

## 1. Custom Metadata Filtering

### Purpose
Enable multi-tenant deployments by filtering documents by custom business attributes like `tenant_id`, `security_level`, or `department`.

### Configuration

Add custom filter fields to your YAML configuration:

```yaml
# config/custom_config.yaml
storage:
  iris:
    custom_filter_keys:
      - tenant_id
      - security_level
      - department
```

### Usage Examples

**Python Library**:
```python
from iris_rag import create_pipeline

# Create pipeline with custom config
pipeline = create_pipeline("basic", config_path="config/custom_config.yaml")

# Query with custom metadata filters
result = pipeline.query(
    query="What are the symptoms of diabetes?",
    top_k=5,
    metadata_filter={
        "tenant_id": "tenant_001",
        "security_level": 3,
        "category": "research-paper"
    }
)

print(f"Answer: {result['answer']}")
print(f"Documents: {len(result['retrieved_documents'])}")
```

**REST API**:
```bash
curl -X POST http://localhost:8000/api/v1/basic/_search \
  -H "Authorization: ApiKey <your-key>" \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What are the symptoms of diabetes?",
    "top_k": 5,
    "metadata_filter": {
      "tenant_id": "tenant_001",
      "security_level": 3
    }
  }'
```

### List Allowed Filter Keys

```python
from iris_rag.storage import IRISVectorStore

store = IRISVectorStore(config_manager)

# Get all allowed keys
allowed = store.get_allowed_filter_keys()

print("Default keys:", allowed["default_keys"])
print("Custom keys:", allowed["custom_keys"])
print("All keys:", allowed["all_keys"])
```

### Error Handling

```python
from iris_rag.exceptions import VectorStoreConfigurationError

try:
    result = pipeline.query(
        query="test",
        metadata_filter={"invalid_field": "value"}
    )
except VectorStoreConfigurationError as e:
    print(f"Error: {e.message}")
    print(f"Rejected keys: {e.rejected_keys}")
    print(f"Allowed keys: {e.allowed_keys}")
```

### Common Patterns

**Multi-Tenant Isolation**:
```python
# Each tenant sees only their documents
def query_for_tenant(tenant_id: str, query: str):
    return pipeline.query(
        query=query,
        metadata_filter={"tenant_id": tenant_id}
    )
```

**Security-Level Filtering**:
```python
# Users see documents at or below their clearance
def query_with_clearance(user_clearance: int, query: str):
    # Note: Requires documents tagged with security_level metadata
    return pipeline.query(
        query=query,
        metadata_filter={"security_level": user_clearance}
    )
```

---

## 2. Collection Management

### Purpose
View, create, and delete document collections for operational management and data organization.

### No Configuration Required
Collection management APIs work out-of-the-box with existing iris-vector-rag setup.

### Usage Examples

**List All Collections**:
```python
from iris_rag.storage import IRISVectorStore

store = IRISVectorStore(config_manager)

# List all collections with statistics
collections = store.list_collections()

for collection in collections:
    print(f"Collection: {collection['collection_id']}")
    print(f"  Documents: {collection['document_count']}")
    print(f"  Size: {collection['total_size_bytes'] / 1024 / 1024:.2f} MB")
    print(f"  Last Updated: {collection['last_updated']}")
```

**Get Collection Details**:
```python
# Get detailed info for specific collection
info = store.get_collection_info("medical-docs-2024")

print(f"Collection: {info['collection_id']}")
print(f"Documents: {info['document_count']}")
print(f"Created: {info['created_at']}")
print(f"Metadata: {info['metadata']}")
```

**Create Collection**:
```python
# Explicit collection creation (optional, auto-created on first document)
success = store.create_collection(
    collection_id="new-collection",
    metadata={
        "department": "Research",
        "owner": "research@company.com",
        "retention_years": 5
    }
)
```

**Delete Collection**:
```python
# Delete collection and all documents
deleted_count = store.delete_collection("old-test-collection")
print(f"Deleted {deleted_count} documents")
```

**Check Existence**:
```python
# Check before operations
if store.collection_exists("medical-docs"):
    info = store.get_collection_info("medical-docs")
else:
    print("Collection not found")
```

### REST API Examples

**List Collections**:
```bash
curl -X GET http://localhost:8000/api/v1/collections \
  -H "Authorization: ApiKey <your-key>"
```

**Get Collection Details**:
```bash
curl -X GET http://localhost:8000/api/v1/collections/medical-docs-2024 \
  -H "Authorization: ApiKey <your-key>"
```

**Delete Collection**:
```bash
curl -X DELETE http://localhost:8000/api/v1/collections/test-collection \
  -H "Authorization: ApiKey <your-key>"
```

### Common Patterns

**Cleanup Test Data**:
```python
# Delete all test collections
all_collections = store.list_collections()
for collection in all_collections:
    if collection['collection_id'].startswith("test-"):
        deleted = store.delete_collection(collection['collection_id'])
        print(f"Deleted test collection: {collection['collection_id']} ({deleted} docs)")
```

**Monitor Collection Growth**:
```python
# Track collection size over time
import time

while True:
    info = store.get_collection_info("production-docs")
    print(f"Documents: {info['document_count']}, Size: {info['total_size_bytes'] / 1024 / 1024:.2f} MB")
    time.sleep(3600)  # Check every hour
```

---

## 3. RBAC Integration

### Purpose
Restrict document access based on user permissions using your organization's authorization system (LDAP, OAuth, IRIS Security).

### Implementation Steps

**Step 1: Implement RBACPolicy Interface**

```python
# myapp/security.py
from iris_rag.security import RBACPolicy
from typing import List, Optional
from langchain.docstore.document import Document

class LDAPRBACPolicy(RBACPolicy):
    def __init__(self, ldap_server: str, bind_dn: str):
        self.ldap = LDAPClient(ldap_server, bind_dn)

    def check_collection_access(
        self,
        user: str,
        collection_id: Optional[str],
        operation: str
    ) -> bool:
        """Check if user has permission for operation."""
        groups = self.ldap.get_user_groups(user)

        # Admin can do anything
        if "admin" in groups:
            return True

        # Read permission
        if operation == "read" and "readers" in groups:
            return True

        # Write permission
        if operation == "write" and "writers" in groups:
            return True

        return False

    def filter_documents(
        self,
        user: str,
        documents: List[Document]
    ) -> List[Document]:
        """Filter documents by security clearance."""
        clearance = self.ldap.get_clearance_level(user)

        return [
            doc for doc in documents
            if doc.metadata.get("security_level", 0) <= clearance
        ]

    def get_audit_context(self, user: str) -> dict:
        """Provide audit context."""
        return {
            "user": user,
            "roles": self.ldap.get_user_groups(user),
            "clearance": self.ldap.get_clearance_level(user)
        }
```

**Step 2: Configure RBAC in YAML**

```yaml
# config/rbac_config.yaml
security:
  rbac:
    enabled: true
    policy_class: "myapp.security.LDAPRBACPolicy"
    policy_config:
      ldap_server: "ldap://ldap.company.com"
      bind_dn: "cn=admin,dc=company,dc=com"
```

**Step 3: Use RBAC-Enabled Vector Store**

```python
from iris_rag.storage import IRISVectorStore
from iris_rag.config import ConfigurationManager
from myapp.security import LDAPRBACPolicy

# Initialize RBAC policy
policy = LDAPRBACPolicy(
    ldap_server="ldap://ldap.company.com",
    bind_dn="cn=admin,dc=company,dc=com"
)

# Create vector store with RBAC
config_manager = ConfigurationManager(config_path="config/rbac_config.yaml")
store = IRISVectorStore(
    config_manager=config_manager,
    rbac_policy=policy
)

# Query with user context
results = store.similarity_search(
    query="diabetes symptoms",
    k=5,
    user="john.doe@company.com"  # User identifier
)
# Documents automatically filtered by RBAC policy
```

### Usage Examples

**Query with Permission Check**:
```python
from iris_rag.security import PermissionDeniedError

try:
    result = pipeline.query(
        query="confidential data",
        user="john.doe@company.com"
    )
except PermissionDeniedError as e:
    print(f"Access denied: {e.message}")
    print(f"Contact your administrator")
```

**Upload with Write Permission**:
```python
try:
    store.add_documents(
        documents=[...],
        user="admin@company.com"
    )
except PermissionDeniedError as e:
    print(f"Upload denied: {e.message}")
```

### Common Patterns

**Role-Based Access**:
```python
class RoleBasedRBACPolicy(RBACPolicy):
    def check_collection_access(self, user, collection_id, operation):
        user_roles = self.get_user_roles(user)

        # Admins have full access
        if "admin" in user_roles:
            return True

        # Collection owners have full access to their collections
        owner = self.get_collection_owner(collection_id)
        if owner == user:
            return True

        # Other users: read-only for public collections
        if operation == "read" and self.is_public_collection(collection_id):
            return True

        return False
```

**Clearance-Level Filtering**:
```python
class ClearanceLevelRBACPolicy(RBACPolicy):
    def filter_documents(self, user, documents):
        user_clearance = self.get_clearance_level(user)

        return [
            doc for doc in documents
            if doc.metadata.get("security_level", 0) <= user_clearance
        ]
```

---

## 4. OpenTelemetry Monitoring

### Purpose
Monitor RAG system performance in production with query latency, token usage, and cost tracking.

### Configuration

```yaml
# config/telemetry_config.yaml
telemetry:
  enabled: true
  service_name: "iris-rag-production"
  otlp:
    endpoint: "http://otel-collector:4318"
    protocol: "http"
  sampling:
    ratio: 0.1  # 10% sampling
```

### Usage Examples

**Enable Telemetry at Runtime**:
```python
from iris_rag.monitoring.telemetry import configure_telemetry

# Enable telemetry
configure_telemetry(
    enabled=True,
    service_name="iris-rag-api",
    endpoint="http://localhost:4318"
)
```

**Instrument Custom Code**:
```python
from iris_rag.monitoring.telemetry import telemetry

def custom_operation():
    with telemetry.trace_operation(
        "custom.operation",
        operation_type="data_processing",
        record_count=1000
    ) as span:
        # Your code here
        process_data()

        # Add custom attributes
        if span:
            span.set_attribute("processing.success", True)
```

**Query Telemetry Status**:
```python
from iris_rag.monitoring.telemetry import telemetry

status = telemetry.get_status()
print(f"Enabled: {status['enabled']}")
print(f"Service: {status['service_name']}")
print(f"Spans exported: {status['total_spans_exported']}")
print(f"Overhead: {status['overhead_percentage']:.2f}%")
```

**Calculate LLM Cost**:
```python
from iris_rag.monitoring.cost_tracking import calculate_llm_cost

cost = calculate_llm_cost(
    llm_model="gpt-4",
    prompt_tokens=1234,
    completion_tokens=567
)
print(f"Estimated cost: ${cost:.4f} USD")
# Returns: $0.0804
```

### Setup OpenTelemetry Collector

**Docker Compose**:
```yaml
# docker-compose.telemetry.yml
version: '3.8'
services:
  jaeger:
    image: jaegertracing/all-in-one:latest
    ports:
      - "16686:16686"  # Jaeger UI
      - "4318:4318"    # OTLP HTTP receiver
    environment:
      - COLLECTOR_OTLP_ENABLED=true
```

**Start Services**:
```bash
# Start Jaeger all-in-one
docker-compose -f docker-compose.telemetry.yml up -d

# Open Jaeger UI
open http://localhost:16686
```

### Common Patterns

**Monitor Query Performance**:
```python
# Queries automatically instrumented when telemetry enabled
result = pipeline.query(query="diabetes symptoms")

# View traces in Jaeger UI:
# - Service: "iris-rag-production"
# - Operation: "rag.query"
# - Spans: query → retrieval → generation
# - Attributes: query_length, top_k, tokens, cost
```

**Track LLM Usage**:
```python
# Token counts and costs automatically tracked
# View in Jaeger UI under span attributes:
# - gen.ai.usage.input_tokens
# - gen.ai.usage.output_tokens
# - estimated_cost_usd
```

---

## 5. Bulk Document Loading

### Purpose
Load large volumes of documents (10,000+) efficiently for initial setup or data migrations.

### No Configuration Required
Bulk operations work out-of-the-box. Optionally configure defaults:

```yaml
# config/batch_config.yaml
batch_operations:
  default_batch_size: 1000
  show_progress: true  # Enable for CLI
  error_handling: "continue"
```

### Usage Examples

**Basic Bulk Loading**:
```python
from iris_rag.storage import IRISVectorStore
from langchain.docstore.document import Document

store = IRISVectorStore(config_manager)

# Prepare documents
documents = [
    Document(
        page_content="Document text...",
        metadata={"collection_id": "medical-docs", "category": "research"}
    )
    for _ in range(10000)
]

# Bulk load
result = store.add_documents_batch(
    documents=documents,
    batch_size=1000,
    error_handling="continue"
)

print(f"Total: {result['total']}")
print(f"Success: {result['success_count']}")
print(f"Errors: {result['error_count']}")
print(f"Time: {result['time_seconds']:.2f} seconds")
print(f"Throughput: {result['throughput_docs_per_sec']:.0f} docs/sec")
```

**With Progress Bar** (CLI):
```python
result = store.add_documents_batch(
    documents=documents,
    batch_size=1000,
    show_progress=True
)
# Output:
# Loading documents: 100%|██████████| 10/10 [00:08<00:00,  1.22batch/s]
```

**Stop on Error**:
```python
# Production loading: fail fast
try:
    result = store.add_documents_batch(
        documents=documents,
        batch_size=1000,
        error_handling="stop"
    )
except Exception as e:
    print(f"Bulk loading failed: {e}")
    # Check result['errors'] for details
```

**Rollback on Error**:
```python
# Critical operation: all-or-nothing
try:
    result = store.add_documents_batch(
        documents=critical_documents,
        batch_size=500,
        error_handling="rollback"
    )
except RuntimeError as e:
    print(f"Rollback triggered: {e}")
    # No partial data committed
```

**Pre-Computed Embeddings**:
```python
from sentence_transformers import SentenceTransformer

# Pre-compute embeddings for faster loading
model = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode([doc.page_content for doc in documents])

# Bulk load with pre-computed embeddings
result = store.add_documents_batch(
    documents=documents,
    embeddings=embeddings.tolist(),
    batch_size=1000
)
# Significantly faster (no embedding generation)
```

### Common Patterns

**Data Migration**:
```python
# Migrate from old system
def migrate_documents(old_collection_id: str, new_collection_id: str):
    # Load from old system
    old_docs = load_from_old_system(old_collection_id)

    # Transform documents
    new_docs = [
        Document(
            page_content=doc['text'],
            metadata={
                "collection_id": new_collection_id,
                "legacy_id": doc['id'],
                "migrated_at": datetime.now().isoformat()
            }
        )
        for doc in old_docs
    ]

    # Bulk load
    result = store.add_documents_batch(
        documents=new_docs,
        batch_size=1000,
        show_progress=True,
        error_handling="stop"
    )

    return result
```

**Parallel Batch Loading**:
```python
from concurrent.futures import ThreadPoolExecutor

def load_collection_batch(collection_id: str, documents: List[Document]):
    return store.add_documents_batch(documents, batch_size=1000)

# Load multiple collections in parallel
collections = {
    "medical-docs": medical_documents,
    "legal-docs": legal_documents,
    "marketing-docs": marketing_documents
}

with ThreadPoolExecutor(max_workers=3) as executor:
    futures = [
        executor.submit(load_collection_batch, coll_id, docs)
        for coll_id, docs in collections.items()
    ]

    for future in futures:
        result = future.result()
        print(f"Loaded {result['success_count']} documents")
```

---

## 6. Metadata Schema Discovery

### Purpose
Discover what metadata fields exist in document collections without manual documentation.

### No Configuration Required
Schema discovery works out-of-the-box.

### Usage Examples

**Discover Schema for Collection**:
```python
from iris_rag.storage import IRISVectorStore

store = IRISVectorStore(config_manager)

# Discover schema for specific collection
schema = store.sample_metadata_schema(
    collection_id="medical-docs-2024",
    sample_size=200
)

# Print schema
for field_name, field_info in schema.items():
    print(f"\nField: {field_name}")
    print(f"  Type: {field_info['type']}")
    print(f"  Frequency: {field_info['frequency'] * 100:.1f}%")
    print(f"  Nullable: {field_info['nullable']}")
    print(f"  Unique Values: {field_info['unique_value_count']}")

    if field_info['type'] in ('integer', 'float'):
        print(f"  Min: {field_info['min_value']}")
        print(f"  Max: {field_info['max_value']}")
        print(f"  Avg: {field_info['avg_value']}")

    if 'examples' in field_info:
        print(f"  Examples: {field_info['examples']}")
```

**Discover Schema for All Collections**:
```python
# Sample across all collections
schema = store.sample_metadata_schema(
    collection_id=None,  # All collections
    sample_size=100
)
```

**CLI Usage**:
```bash
# Discover schema via CLI (future enhancement)
iris-rag schema discover --collection medical-docs-2024 --sample-size 200
```

### Example Output

```
Field: tenant_id
  Type: string
  Frequency: 98.0%
  Nullable: True
  Unique Values: 12
  Examples: ['tenant_001', 'tenant_002', 'tenant_003']

Field: priority
  Type: integer
  Frequency: 75.0%
  Nullable: False
  Unique Values: 5
  Min: 1
  Max: 5
  Avg: 2.8
  Examples: [1, 2, 3, 4, 5]

Field: created_at
  Type: datetime
  Frequency: 100.0%
  Nullable: False
  Unique Values: 87
  Examples: ['2023-06-20T08:15:00Z', '2024-03-15T14:30:00Z']
  Min: 2023-01-15T10:30:00Z
  Max: 2024-11-22T15:45:00Z
```

### Common Patterns

**Build Dynamic Query UI**:
```python
# Discover schema to build filter dropdowns
schema = store.sample_metadata_schema(collection_id="production-docs")

# Build UI dropdowns from schema
for field_name, field_info in schema.items():
    if field_info['type'] == 'string' and field_info['unique_value_count'] < 20:
        # Create dropdown with examples
        print(f"Filter: {field_name}")
        print(f"  Options: {field_info['examples']}")
```

**Validate Metadata Before Loading**:
```python
# Discover expected schema
expected_schema = store.sample_metadata_schema(collection_id="medical-docs")

# Validate new documents match expected schema
def validate_document_metadata(doc: Document) -> bool:
    for field, value in doc.metadata.items():
        if field not in expected_schema:
            print(f"Warning: Unexpected field '{field}'")
            return False

        expected_type = expected_schema[field]['type']
        actual_type = type(value).__name__

        if expected_type != actual_type:
            print(f"Warning: Field '{field}' type mismatch: expected {expected_type}, got {actual_type}")
            return False

    return True
```

---

## Troubleshooting

### Custom Metadata Filtering

**Error: "Filter key 'X' not in allowed list"**
- **Cause**: Field not configured in `storage.iris.custom_filter_keys`
- **Solution**: Add field to config and restart

**Error: "Invalid field name 'X'"**
- **Cause**: Field name contains special characters or starts with digit
- **Solution**: Use alphanumeric + underscores only, start with letter

### Collection Management

**Error: "Collection not found"**
- **Cause**: Collection doesn't exist or was deleted
- **Solution**: Check `list_collections()` or create collection explicitly

**Performance: List collections slow**
- **Cause**: Many collections (>1000)
- **Solution**: Expected behavior (should still complete in <2 seconds)

### RBAC Integration

**Error: "Permission check failed"**
- **Cause**: RBAC policy raised exception
- **Solution**: Check policy implementation and logs

**Error: "RBAC policy not configured"**
- **Cause**: `security.rbac.policy_class` not set
- **Solution**: Implement policy class and add to config

### OpenTelemetry Monitoring

**No spans exported**
- **Cause**: Telemetry disabled or OTLP collector not running
- **Solution**: Check `telemetry.enabled=true` and collector endpoint

**High overhead**
- **Cause**: Sampling ratio too high (e.g., 1.0 = 100%)
- **Solution**: Reduce `sampling.ratio` to 0.1 (10%)

### Bulk Operations

**Error: "Batch size must be at least 1"**
- **Cause**: Invalid `batch_size` parameter
- **Solution**: Use positive integer (recommended: 1000)

**Slow performance**
- **Cause**: Small batch size (e.g., 10)
- **Solution**: Increase to 1000 or use pre-computed embeddings

---

## Next Steps

1. **Read Feature Spec**: See [spec.md](spec.md) for detailed requirements
2. **Review Data Model**: See [data-model.md](data-model.md) for entity definitions
3. **Check API Contracts**: See [contracts/](contracts/) for API specifications
4. **Run Tests**: See implementation guide for TDD workflow

For questions or issues, consult the project documentation or file an issue on GitHub.
