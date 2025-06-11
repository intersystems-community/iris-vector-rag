# Migration Guide: Personal Assistant to RAG Templates

## Overview

This guide provides step-by-step instructions for migrating existing Personal Assistant RAG implementations to the new [`iris_rag`](../iris_rag/__init__.py:1) architecture. The migration preserves functionality while providing improved modularity, testability, and maintainability.

## Pre-Migration Assessment

### 1. Current Implementation Analysis

**Inventory Checklist:**
- [ ] Identify existing RAG pipeline implementations
- [ ] Document current configuration format
- [ ] List custom embedding functions
- [ ] Catalog existing document storage
- [ ] Note performance requirements
- [ ] Identify integration points

**Assessment Commands:**
```bash
# Analyze current codebase structure
find . -name "*.py" -path "*/rag/*" | head -20
grep -r "initialize_iris_rag_pipeline" . --include="*.py"
grep -r "embedding_func\|llm_func" . --include="*.py"
```

### 2. Compatibility Check

**Required Dependencies:**
```bash
# Check current IRIS connectivity
python -c "import iris; print('IRIS available')"
# Check embedding libraries
python -c "import sentence_transformers; print('Embeddings available')"
```

**Version Requirements:**
- Python >= 3.8
- InterSystems IRIS >= 2023.1
- Compatible embedding libraries

## Configuration Migration

### 3. Configuration Format Transformation

#### Before: Legacy PA Configuration
```python
# Legacy Personal Assistant config
pa_config = {
    "iris_host": "localhost",
    "iris_port": 1972,
    "iris_namespace": "USER", 
    "iris_user": "testuser",
    "iris_password": "testpassword",
    "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
    "llm_model_name": "gpt-3.5-turbo",
    "schema": "RAG",
    "chunk_size": 1000,
    "top_k": 5
}
```

#### After: RAG Templates Configuration
```yaml
# config/rag_config.yaml
database:
  iris:
    host: localhost
    port: 1972
    namespace: USER
    username: testuser
    password: testpassword

embeddings:
  model_name: sentence-transformers/all-MiniLM-L6-v2
  batch_size: 32

llm:
  model_name: gpt-3.5-turbo
  api_key: ${LLM_API_KEY}

pipelines:
  basic:
    chunk_size: 1000
    chunk_overlap: 200
    default_top_k: 5
    embedding_batch_size: 32

storage:
  schema: RAG
  table_prefix: rag_
```

### 4. Environment Variable Migration

#### Before: Direct Configuration
```python
config = {
    "iris_password": "hardcoded_password",
    "api_key": "hardcoded_key"
}
```

#### After: Environment-Based Configuration
```bash
# .env file
RAG_DATABASE__IRIS__PASSWORD=secure_password
RAG_LLM__API_KEY=secure_api_key
RAG_EMBEDDINGS__MODEL_NAME=sentence-transformers/all-MiniLM-L6-v2
```

## Code Migration Patterns

### 5. Pipeline Initialization Migration

#### Before: Legacy Initialization
```python
from basic_rag.pipeline import BasicRAGPipeline
from common.iris_connector_jdbc import get_iris_connection
from common.utils import get_embedding_func, get_llm_func

# Legacy approach
iris_connector = get_iris_connection(config)
embedding_func = get_embedding_func(config["embedding_model_name"])
llm_func = get_llm_func(config["llm_model_name"])

pipeline = BasicRAGPipeline(
    iris_connector=iris_connector,
    embedding_func=embedding_func,
    llm_func=llm_func,
    schema=config["schema"]
)
```

#### After: RAG Templates Approach
```python
from iris_rag import create_pipeline
from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager

# New modular approach
config_manager = ConfigurationManager("config/rag_config.yaml")
connection_manager = ConnectionManager(config_manager)

pipeline = create_pipeline(
    pipeline_type="basic",
    connection_manager=connection_manager,
    config_manager=config_manager
)
```

### 6. Personal Assistant Adapter Pattern

#### Migration Using Adapter
```python
from iris_rag.adapters.personal_assistant import PersonalAssistantAdapter

# Seamless migration for PA
adapter = PersonalAssistantAdapter()

# Use existing PA config format
pipeline = adapter.initialize_iris_rag_pipeline(
    pa_specific_config=legacy_pa_config
)

# Existing PA code continues to work
result = adapter.query("What is machine learning?")
```

### 7. Document Loading Migration

#### Before: Manual Document Processing
```python
# Legacy document loading
documents = []
for file_path in document_paths:
    with open(file_path, 'r') as f:
        content = f.read()
    
    chunks = split_text(content, chunk_size=1000)
    embeddings = embedding_func(chunks)
    
    for chunk, embedding in zip(chunks, embeddings):
        store_document(chunk, embedding, file_path)
```

#### After: Automated Pipeline Processing
```python
# New streamlined approach
pipeline.load_documents(
    documents_path="data/documents/",
    chunk_documents=True,
    generate_embeddings=True
)

# Or with custom documents
from iris_rag.core.models import Document

documents = [
    Document(page_content=content, metadata={"source": path})
    for path, content in document_data.items()
]

pipeline.load_documents(
    documents_path="",
    documents=documents
)
```

### 8. Query Execution Migration

#### Before: Manual Query Processing
```python
# Legacy query execution
query_embedding = embedding_func([query_text])[0]
retrieved_docs = vector_search(query_embedding, top_k=5)
context = format_context(retrieved_docs)
prompt = create_prompt(query_text, context)
answer = llm_func(prompt)

result = {
    "query": query_text,
    "answer": answer,
    "documents": retrieved_docs
}
```

#### After: Unified Pipeline Execution
```python
# New unified approach
result = pipeline.execute(
    query_text="What is machine learning?",
    top_k=5,
    include_sources=True
)

# Standard return format guaranteed
print(result["query"])           # Original query
print(result["answer"])          # Generated answer
print(result["retrieved_documents"])  # Source documents
print(result["sources"])         # Source metadata
```

## Data Migration Procedures

### 9. Document Storage Migration

#### Step 1: Export Existing Documents
```python
# Export from legacy storage
def export_legacy_documents(legacy_connection):
    cursor = legacy_connection.cursor()
    cursor.execute("SELECT id, content, metadata, embedding FROM RAG.Documents")
    
    documents = []
    for row in cursor.fetchall():
        doc_id, content, metadata_json, embedding_str = row
        documents.append({
            "id": doc_id,
            "content": content,
            "metadata": json.loads(metadata_json),
            "embedding": json.loads(embedding_str)
        })
    
    return documents
```

#### Step 2: Import to New Storage
```python
# Import to new iris_rag storage
from iris_rag.core.models import Document

def migrate_documents(exported_docs, new_pipeline):
    migrated_documents = []
    
    for doc_data in exported_docs:
        document = Document(
            page_content=doc_data["content"],
            metadata=doc_data["metadata"]
        )
        # Preserve original ID if needed
        document.id = doc_data["id"]
        migrated_documents.append(document)
    
    # Store with existing embeddings
    embeddings = [doc_data["embedding"] for doc_data in exported_docs]
    new_pipeline.storage.store_documents(migrated_documents, embeddings)
```

### 10. Embedding Migration

#### Preserve Existing Embeddings
```python
# Migration script preserving embeddings
def migrate_with_embeddings(source_config, target_config):
    # Initialize both systems
    legacy_pipeline = initialize_legacy_pipeline(source_config)
    new_pipeline = create_pipeline("basic", target_config)
    
    # Export documents with embeddings
    documents = export_legacy_documents(legacy_pipeline.iris_connector)
    
    # Import preserving embeddings
    migrate_documents(documents, new_pipeline)
    
    print(f"Migrated {len(documents)} documents with embeddings preserved")
```

#### Re-generate Embeddings (if model changed)
```python
# Migration with embedding regeneration
def migrate_with_new_embeddings(source_config, target_config):
    # Export content only
    documents = export_document_content(source_config)
    
    # Load into new pipeline (will generate new embeddings)
    new_pipeline = create_pipeline("basic", target_config)
    new_pipeline.load_documents(
        documents_path="",
        documents=documents,
        generate_embeddings=True
    )
```

## Testing Migrated Systems

### 11. Validation Test Suite

#### Functional Validation
```python
def test_migration_functionality(original_pipeline, migrated_pipeline):
    """Test that migrated pipeline produces equivalent results"""
    
    test_queries = [
        "What is machine learning?",
        "Explain neural networks",
        "How does RAG work?"
    ]
    
    for query in test_queries:
        # Test retrieval consistency
        original_docs = original_pipeline.retrieve_documents(query, top_k=5)
        migrated_docs = migrated_pipeline.query(query, top_k=5)
        
        assert len(original_docs) == len(migrated_docs)
        
        # Test answer generation
        original_result = original_pipeline.execute(query)
        migrated_result = migrated_pipeline.execute(query)
        
        assert migrated_result["query"] == query
        assert "answer" in migrated_result
        assert "retrieved_documents" in migrated_result
```

#### Performance Validation
```python
def test_migration_performance(pipeline, test_queries):
    """Validate performance meets requirements"""
    
    import time
    
    for query in test_queries:
        start_time = time.time()
        result = pipeline.execute(query)
        execution_time = time.time() - start_time
        
        # Performance requirements
        assert execution_time < 5.0  # Max 5 seconds
        assert len(result["retrieved_documents"]) > 0
        assert result["metadata"]["processing_time"] < 5.0
```

### 12. Integration Testing

#### End-to-End Test
```python
def test_end_to_end_migration():
    """Complete migration test"""
    
    # 1. Setup test environment
    test_config = load_test_config()
    
    # 2. Initialize migrated pipeline
    pipeline = create_pipeline("basic", test_config)
    
    # 3. Load test documents
    pipeline.load_documents("test_data/sample_docs/")
    
    # 4. Test query execution
    result = pipeline.execute("Test query")
    
    # 5. Validate standard format
    assert "query" in result
    assert "answer" in result
    assert "retrieved_documents" in result
    
    # 6. Test document count
    doc_count = pipeline.get_document_count()
    assert doc_count > 0
```

## Production Deployment Strategy

### 13. Phased Deployment

#### Phase 1: Parallel Deployment
```python
# Run both systems in parallel
class DualPipelineService:
    def __init__(self, legacy_config, new_config):
        self.legacy_pipeline = initialize_legacy_pipeline(legacy_config)
        self.new_pipeline = create_pipeline("basic", new_config)
        self.comparison_mode = True
    
    def query(self, query_text):
        if self.comparison_mode:
            # Compare results
            legacy_result = self.legacy_pipeline.execute(query_text)
            new_result = self.new_pipeline.execute(query_text)
            
            # Log comparison metrics
            self.log_comparison(legacy_result, new_result)
            
            # Return new result but log differences
            return new_result
        else:
            return self.new_pipeline.execute(query_text)
```

#### Phase 2: Gradual Cutover
```python
# Gradual traffic migration
class GradualMigrationService:
    def __init__(self, legacy_pipeline, new_pipeline):
        self.legacy_pipeline = legacy_pipeline
        self.new_pipeline = new_pipeline
        self.migration_percentage = 0  # Start with 0% on new system
    
    def query(self, query_text):
        import random
        
        if random.randint(1, 100) <= self.migration_percentage:
            return self.new_pipeline.execute(query_text)
        else:
            return self.legacy_pipeline.execute(query_text)
    
    def increase_migration_percentage(self, increment=10):
        self.migration_percentage = min(100, self.migration_percentage + increment)
```

### 14. Monitoring and Validation

#### Production Monitoring
```python
def setup_migration_monitoring(pipeline):
    """Setup monitoring for migrated pipeline"""
    
    import logging
    
    # Performance monitoring
    def log_performance_metrics(result):
        processing_time = result["metadata"]["processing_time"]
        num_retrieved = result["metadata"]["num_retrieved"]
        
        logging.info(f"Query processed in {processing_time:.2f}s, retrieved {num_retrieved} docs")
        
        # Alert if performance degrades
        if processing_time > 10.0:
            logging.warning(f"Slow query detected: {processing_time:.2f}s")
    
    # Wrap pipeline execution
    original_execute = pipeline.execute
    
    def monitored_execute(query_text, **kwargs):
        result = original_execute(query_text, **kwargs)
        log_performance_metrics(result)
        return result
    
    pipeline.execute = monitored_execute
```

## Rollback Procedures

### 15. Emergency Rollback

#### Immediate Rollback Script
```bash
#!/bin/bash
# emergency_rollback.sh

echo "Initiating emergency rollback..."

# 1. Stop new service
systemctl stop iris-rag-service

# 2. Restore legacy service
systemctl start legacy-rag-service

# 3. Update load balancer
curl -X POST "http://loadbalancer/config" -d '{"route": "legacy"}'

# 4. Verify legacy service
curl -X POST "http://legacy-rag/health" || exit 1

echo "Rollback completed successfully"
```

#### Data Rollback
```python
def rollback_data_migration(backup_timestamp):
    """Rollback data to pre-migration state"""
    
    # 1. Stop current pipeline
    pipeline.clear_knowledge_base()
    
    # 2. Restore from backup
    backup_file = f"migration_backup_{backup_timestamp}.json"
    with open(backup_file, 'r') as f:
        backup_data = json.load(f)
    
    # 3. Restore documents
    restore_documents(backup_data["documents"])
    
    # 4. Verify restoration
    doc_count = pipeline.get_document_count()
    assert doc_count == backup_data["document_count"]
    
    print(f"Data rollback completed: {doc_count} documents restored")
```

### 16. Rollback Validation

#### Post-Rollback Testing
```python
def validate_rollback(legacy_pipeline):
    """Validate system after rollback"""
    
    test_queries = ["test query 1", "test query 2"]
    
    for query in test_queries:
        try:
            result = legacy_pipeline.execute(query)
            assert "answer" in result
            print(f"✓ Query '{query}' successful")
        except Exception as e:
            print(f"✗ Query '{query}' failed: {e}")
            return False
    
    return True
```

## Common Migration Issues

### 17. Troubleshooting Guide

#### Issue: Configuration Loading Errors
```python
# Problem: Environment variables not loading
# Solution: Check environment variable format
from iris_rag.config.manager import ConfigurationManager

try:
    config = ConfigurationManager("config/rag_config.yaml")
    iris_host = config.get("database:iris:host")
    print(f"IRIS host: {iris_host}")
except Exception as e:
    print(f"Config error: {e}")
    # Check environment variables
    import os
    print("RAG_ environment variables:")
    for key, value in os.environ.items():
        if key.startswith("RAG_"):
            print(f"  {key}={value}")
```

#### Issue: Connection Failures
```python
# Problem: Cannot connect to IRIS
# Solution: Validate connection parameters
from iris_rag.core.connection import ConnectionManager

def diagnose_connection_issue(config_manager):
    try:
        conn_manager = ConnectionManager(config_manager)
        result = conn_manager.execute("SELECT 1")
        print("✓ Connection successful")
    except Exception as e:
        print(f"✗ Connection failed: {e}")
        
        # Check configuration
        host = config_manager.get("database:iris:host")
        port = config_manager.get("database:iris:port")
        print(f"Attempting connection to {host}:{port}")
```

#### Issue: Embedding Dimension Mismatch
```python
# Problem: Existing embeddings incompatible with new model
# Solution: Re-generate embeddings or use compatible model
def fix_embedding_mismatch(pipeline, documents):
    try:
        # Try loading with existing embeddings
        pipeline.load_documents(documents_path="", documents=documents)
    except ValueError as e:
        if "dimension mismatch" in str(e):
            print("Embedding dimension mismatch detected")
            print("Re-generating embeddings with current model...")
            
            # Re-generate embeddings
            pipeline.load_documents(
                documents_path="",
                documents=documents,
                generate_embeddings=True  # Force regeneration
            )
```

#### Issue: Performance Degradation
```python
# Problem: Slower performance after migration
# Solution: Optimize configuration and indexing
def optimize_performance(pipeline):
    # 1. Check index status
    doc_count = pipeline.get_document_count()
    print(f"Document count: {doc_count}")
    
    # 2. Optimize batch sizes
    config = pipeline.config_manager
    batch_size = config.get("pipelines:basic:embedding_batch_size", 32)
    print(f"Current batch size: {batch_size}")
    
    # 3. Suggest optimizations
    if doc_count > 10000 and batch_size < 64:
        print("Suggestion: Increase embedding_batch_size to 64 for large datasets")
```

## Migration Checklist

### 18. Pre-Migration Checklist
- [ ] Backup existing data and configuration
- [ ] Test new system in development environment
- [ ] Validate configuration transformation
- [ ] Prepare rollback procedures
- [ ] Set up monitoring and alerting

### 19. Migration Execution Checklist
- [ ] Deploy new system in parallel
- [ ] Migrate configuration files
- [ ] Transfer document data
- [ ] Validate functionality
- [ ] Monitor performance metrics
- [ ] Gradually increase traffic to new system

### 20. Post-Migration Checklist
- [ ] Verify all functionality working
- [ ] Monitor performance for 24-48 hours
- [ ] Remove legacy system (after validation period)
- [ ] Update documentation
- [ ] Train team on new system

---

**Migration Support**: For additional assistance, refer to the [API Reference](API_REFERENCE.md) and [Implementation Guide](IMPLEMENTATION_GUIDE.md).

*Last Updated: 2025-06-07*  
*Version: 1.0*