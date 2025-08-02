# Migration Guide

A comprehensive guide for migrating from complex setup to the dead-simple Library Consumption Framework.

## Table of Contents

1. [Migration Overview](#migration-overview)
2. [Before and After Comparison](#before-and-after-comparison)
3. [Step-by-Step Migration](#step-by-step-migration)
4. [Backward Compatibility](#backward-compatibility)
5. [Performance Considerations](#performance-considerations)
6. [Automated Migration Tools](#automated-migration-tools)
7. [Common Migration Patterns](#common-migration-patterns)
8. [Troubleshooting](#troubleshooting)

## Migration Overview

The Library Consumption Framework transforms rag-templates from a complex, setup-intensive framework into a dead-simple library that works immediately with zero configuration.

### Migration Benefits

- **Reduced Complexity**: From 50+ lines of setup to 3 lines of code
- **Zero Configuration**: Works out-of-the-box with sensible defaults
- **Immediate Productivity**: Start building in minutes, not hours
- **Backward Compatibility**: Existing code continues to work
- **Progressive Enhancement**: Add complexity only when needed

### Migration Strategy

1. **Assess Current Usage**: Identify how you're currently using rag-templates
2. **Choose API Tier**: Select Simple, Standard, or Enterprise API
3. **Migrate Incrementally**: Convert one component at a time
4. **Test Thoroughly**: Ensure functionality is preserved
5. **Optimize**: Take advantage of new features

## Before and After Comparison

### Complex Setup (Before)

#### Python - Complex Setup
```python
# 50+ lines of complex setup
from iris_rag.pipelines.factory import create_pipeline
from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager
from iris_rag.embeddings.manager import EmbeddingManager
from iris_rag.storage.enterprise_storage import IRISStorage
from common.utils import get_llm_func
from common.iris_connector import get_iris_connection

# Complex configuration management
config_manager = ConfigurationManager("config.yaml")
connection_manager = ConnectionManager(config_manager)
embedding_manager = EmbeddingManager(config_manager)

# Manual pipeline creation
pipeline = create_pipeline(
    pipeline_type="basic",
    llm_func=get_llm_func(),
    external_connection=get_iris_connection(),
    connection_manager=connection_manager,
    config_manager=config_manager,
    embedding_func=embedding_manager.embed_texts
)

# Manual document loading
from iris_rag.storage.enterprise_storage import IRISStorage
storage = IRISStorage(connection_manager, config_manager)
storage.initialize_schema()

# Complex document processing
documents = []
for file_path in document_paths:
    with open(file_path, 'r') as f:
        content = f.read()
        doc = Document(
            page_content=content,
            metadata={"source": file_path}
        )
        documents.append(doc)

storage.store_documents(documents)

# Complex querying
result = pipeline.query("What is machine learning?", top_k=5)
answer = result['answer']
sources = result['retrieved_documents']
```

#### JavaScript - Complex Setup
```javascript
// 40+ lines of complex setup
const { createVectorSearchPipeline } = require('./src/index');
const { ConfigManager } = require('./src/config-manager');

// Manual configuration
const configManager = new ConfigManager();
const dbConfig = {
    host: configManager.get('iris.host') || 'localhost',
    port: configManager.get('iris.webPort') || 52773,
    namespace: configManager.get('iris.namespace') || 'ML_RAG',
    username: configManager.get('iris.username') || 'demo',
    password: configManager.get('iris.password') || 'demo'
};

// Manual pipeline creation
const pipeline = createVectorSearchPipeline({
    connection: dbConfig,
    embeddingModel: configManager.get('iris.embeddingModel') || 'Xenova/all-MiniLM-L6-v2'
});

// Manual initialization
await pipeline.initialize();

// Complex document processing
const processedDocs = documents.map((doc, index) => ({
    docId: `doc_${index}`,
    title: doc.title || `Document ${index}`,
    content: doc.content,
    sourceFile: doc.source || 'unknown',
    pageNumber: 1,
    chunkIndex: index
}));

await pipeline.indexDocuments(processedDocs);

// Complex querying
const results = await pipeline.search("What is machine learning?", {
    topK: 5,
    additionalWhere: null,
    minSimilarity: 0.7
});

const answer = results.length > 0 
    ? `Based on the information: ${results[0].textContent}...`
    : "No relevant information found.";
```

### Simple API (After)

#### Python - Simple API
```python
# 3 lines of dead-simple code
from rag_templates import RAG

rag = RAG()
rag.add_documents(["Document 1", "Document 2", "Document 3"])
answer = rag.query("What is machine learning?")
```

#### JavaScript - Simple API
```javascript
// 4 lines of dead-simple code
import { RAG } from '@rag-templates/core';

const rag = new RAG();
await rag.addDocuments(["Document 1", "Document 2", "Document 3"]);
const answer = await rag.query("What is machine learning?");
```

## Step-by-Step Migration

### Step 1: Assess Current Usage

#### Identify Your Current Pattern

**Pattern A: Basic Pipeline Usage**
```python
# If you're using basic pipeline creation
pipeline = create_pipeline(pipeline_type="basic", ...)
result = pipeline.query(query)
```
→ **Migrate to**: Simple API

**Pattern B: Advanced Configuration**
```python
# If you're using complex configuration
config = ConfigurationManager("complex-config.yaml")
pipeline = create_pipeline(pipeline_type="colbert", config_manager=config, ...)
```
→ **Migrate to**: Standard API

**Pattern C: Custom Pipelines**
```python
# If you're using custom pipeline implementations
class MyCustomPipeline(RAGPipeline):
    def execute(self, query):
        # Custom logic
```
→ **Migrate to**: Enterprise API

### Step 2: Choose Your API Tier

#### Simple API Migration
**Best for**: Basic RAG functionality, prototypes, simple applications

```python
# Before (Complex)
from iris_rag.pipelines.factory import create_pipeline
from common.utils import get_llm_func

pipeline = create_pipeline(
    pipeline_type="basic",
    llm_func=get_llm_func()
)
result = pipeline.query("query")

# After (Simple)
from rag_templates import RAG

rag = RAG()
answer = rag.query("query")
```

#### Standard API Migration
**Best for**: Production applications, technique selection, advanced configuration

```python
# Before (Complex)
config = ConfigurationManager("config.yaml")
pipeline = create_pipeline(
    pipeline_type="colbert",
    config_manager=config,
    llm_func=get_llm_func()
)

# After (Standard)
from rag_templates import ConfigurableRAG

rag = ConfigurableRAG({
    "technique": "colbert",
    "llm_provider": "openai"
})
```

#### Enterprise API Migration
**Best for**: Enterprise deployments, custom features, complex workflows

```python
# Before (Complex)
config = ConfigurationManager("enterprise-config.yaml")
connection_manager = ConnectionManager(config)
pipeline = CustomRAGPipeline(
    connection_manager=connection_manager,
    config_manager=config
)

# After (Enterprise)
from rag_templates import ConfigurableRAG
from rag_templates.config import ConfigManager

config = ConfigManager.from_file("enterprise-config.yaml")
rag = ConfigurableRAG(config)
```

### Step 3: Migrate Configuration

#### Environment Variables Migration

**Before**: Manual environment variable handling
```python
import os
db_host = os.getenv('IRIS_HOST', 'localhost')
db_port = int(os.getenv('IRIS_PORT', '52773'))
```

**After**: Automatic environment variable support
```python
# Environment variables automatically loaded
# IRIS_HOST, IRIS_PORT, IRIS_USERNAME, IRIS_PASSWORD
rag = RAG()  # Automatically uses environment variables
```

#### Configuration File Migration

**Before**: Complex YAML structure
```yaml
# old-config.yaml
database:
  iris:
    connection:
      host: localhost
      port: 52773
      username: demo
      password: demo
      namespace: USER

embeddings:
  manager:
    model:
      name: "sentence-transformers/all-MiniLM-L6-v2"
      dimension: 384

pipelines:
  basic:
    chunk_size: 1000
    chunk_overlap: 200
    default_top_k: 5
```

**After**: Simplified configuration
```yaml
# new-config.yaml
technique: "basic"
llm_provider: "openai"
embedding_model: "text-embedding-3-small"
max_results: 5

# Database config (optional - uses environment variables)
database:
  host: localhost
  port: 52773
  namespace: RAG_SIMPLE
```

### Step 4: Migrate Document Processing

#### Document Loading Migration

**Before**: Manual document processing
```python
from iris_rag.core.models import Document
from iris_rag.storage.enterprise_storage import IRISStorage

documents = []
for file_path in file_paths:
    with open(file_path, 'r') as f:
        content = f.read()
        doc = Document(
            page_content=content,
            metadata={"source": file_path}
        )
        documents.append(doc)

storage = IRISStorage(connection_manager, config_manager)
storage.store_documents(documents)
```

**After**: Simple document addition
```python
# String documents
rag.add_documents([
    "Document 1 content",
    "Document 2 content"
])

# Or document objects
rag.add_documents([
    {
        "content": "Document content",
        "title": "Document Title",
        "source": "file.pdf"
    }
])
```

#### Bulk Document Loading Migration

**Before**: Complex bulk loading
```python
from iris_rag.ingestion.loader import DocumentLoader
from iris_rag.ingestion.chunker import RecursiveCharacterTextSplitter

loader = DocumentLoader()
chunker = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200
)

documents = loader.load_directory("./documents")
chunks = chunker.split_documents(documents)
storage.store_documents(chunks)
```

**After**: Simple directory loading
```python
# Simple API
rag.load_from_directory("./documents")

# Standard API with options
rag = ConfigurableRAG({
    "chunk_size": 1000,
    "chunk_overlap": 200
})
rag.load_from_directory("./documents", {
    "file_types": [".pdf", ".txt", ".md"]
})
```

### Step 5: Migrate Querying

#### Basic Query Migration

**Before**: Complex pipeline execution
```python
result = pipeline.query(
    query_text="What is machine learning?",
    top_k=5,
    similarity_threshold=0.7
)

answer = result['answer']
sources = result['retrieved_documents']
confidence = result.get('confidence', 0.0)
```

**After**: Simple querying
```python
# Simple API - string response
answer = rag.query("What is machine learning?")

# Standard API - rich response
result = rag.query("What is machine learning?", {
    "max_results": 5,
    "min_similarity": 0.7,
    "include_sources": True
})

answer = result.answer
sources = result.sources
confidence = result.confidence
```

#### Advanced Query Migration

**Before**: Manual query processing
```python
# Custom query processing
embedding_func = embedding_manager.embed_texts
query_embedding = embedding_func([query_text])[0]

# Manual vector search
search_results = storage.vector_search(
    query_embedding=query_embedding,
    top_k=10,
    similarity_threshold=0.8
)

# Manual context preparation
context = "\n".join([doc.page_content for doc in search_results])

# Manual LLM call
llm_func = get_llm_func()
answer = llm_func(f"Context: {context}\nQuestion: {query_text}")
```

**After**: Automatic query processing
```python
# All processing handled automatically
result = rag.query("What is machine learning?", {
    "max_results": 10,
    "min_similarity": 0.8,
    "include_sources": True,
    "response_format": "detailed"
})
```

## Backward Compatibility

### Existing Code Compatibility

The Library Consumption Framework maintains backward compatibility with existing code:

#### Python Compatibility
```python
# Existing complex code continues to work
from iris_rag.pipelines.factory import create_pipeline
from common.utils import get_llm_func

# This still works
pipeline = create_pipeline(
    pipeline_type="basic",
    llm_func=get_llm_func()
)

# New simple code works alongside
from rag_templates import RAG
rag = RAG()

# Both can coexist in the same application
```

#### JavaScript Compatibility
```javascript
// Existing code continues to work
const { createVectorSearchPipeline } = require('./src/index');
const pipeline = createVectorSearchPipeline({...});

// New simple code works alongside
import { RAG } from '@rag-templates/core';
const rag = new RAG();

// Both can coexist
```

### Gradual Migration Strategy

#### Phase 1: Add Simple API Alongside Existing Code
```python
# Keep existing complex pipeline
existing_pipeline = create_pipeline(...)

# Add new simple API for new features
from rag_templates import RAG
simple_rag = RAG()

# Use both as needed
legacy_result = existing_pipeline.query(query)
simple_answer = simple_rag.query(query)
```

#### Phase 2: Migrate Non-Critical Components
```python
# Migrate simple use cases first
def simple_qa(question):
    # Before: complex pipeline
    # return existing_pipeline.query(question)['answer']
    
    # After: simple API
    return rag.query(question)

# Keep complex use cases on old system temporarily
def complex_analysis(query):
    return existing_pipeline.query(query)  # Keep for now
```

#### Phase 3: Complete Migration
```python
# Replace all usage with new API
from rag_templates import ConfigurableRAG

# Migrate complex use cases to Standard API
rag = ConfigurableRAG({
    "technique": "colbert",
    "llm_provider": "openai"
})

def simple_qa(question):
    return rag.query(question)

def complex_analysis(query):
    return rag.query(query, {
        "max_results": 15,
        "include_sources": True,
        "analysis_depth": "comprehensive"
    })
```

## Performance Considerations

### Performance Comparison

#### Initialization Performance

**Before**: Complex initialization
```python
# ~5-10 seconds initialization time
config_manager = ConfigurationManager("config.yaml")  # ~1s
connection_manager = ConnectionManager(config_manager)  # ~2s
embedding_manager = EmbeddingManager(config_manager)  # ~3s
pipeline = create_pipeline(...)  # ~2s
```

**After**: Lazy initialization
```python
# ~0.1 seconds initialization time
rag = RAG()  # Instant - lazy initialization

# Heavy operations deferred until first use
answer = rag.query("test")  # ~3s first call, then fast
```

#### Memory Usage

**Before**: High memory footprint
```python
# Multiple managers and connections loaded upfront
# Memory usage: ~500MB baseline
```

**After**: Optimized memory usage
```python
# Lazy loading and shared resources
# Memory usage: ~200MB baseline
```

#### Query Performance

**Before**: Manual optimization required
```python
# Manual caching and optimization
cache = {}
def cached_query(query):
    if query in cache:
        return cache[query]
    result = pipeline.query(query)
    cache[query] = result
    return result
```

**After**: Built-in optimization
```python
# Automatic caching and optimization
rag = ConfigurableRAG({
    "caching": {"enabled": True, "ttl": 3600}
})
answer = rag.query(query)  # Automatically cached
```

### Performance Migration Tips

1. **Enable Caching**: Use built-in caching for better performance
```python
rag = ConfigurableRAG({
    "caching": {"enabled": True, "ttl": 3600}
})
```

2. **Optimize Batch Processing**: Use batch document addition
```python
# Instead of multiple calls
for doc in documents:
    rag.add_documents([doc])  # Inefficient

# Use batch processing
rag.add_documents(documents)  # Efficient
```

3. **Choose Appropriate Technique**: Select technique based on use case
```python
# For speed
rag = ConfigurableRAG({"technique": "basic"})

# For accuracy
rag = ConfigurableRAG({"technique": "colbert"})

# For complex reasoning
rag = ConfigurableRAG({"technique": "hyde"})
```

## Automated Migration Tools

### Migration Script

#### Python Migration Script
```python
#!/usr/bin/env python3
"""
Automated migration script for rag-templates Library Consumption Framework.
"""

import ast
import os
import re
from pathlib import Path

class RAGMigrationTool:
    def __init__(self, project_path):
        self.project_path = Path(project_path)
        self.migration_report = []
    
    def analyze_current_usage(self):
        """Analyze current rag-templates usage patterns."""
        patterns = {
            'complex_pipeline': r'create_pipeline\(',
            'config_manager': r'ConfigurationManager\(',
            'connection_manager': r'ConnectionManager\(',
            'manual_storage': r'IRISStorage\(',
            'manual_embedding': r'EmbeddingManager\('
        }
        
        usage_stats = {pattern: 0 for pattern in patterns}
        
        for py_file in self.project_path.rglob("*.py"):
            content = py_file.read_text()
            for pattern_name, pattern in patterns.items():
                matches = len(re.findall(pattern, content))
                usage_stats[pattern_name] += matches
        
        return usage_stats
    
    def suggest_migration_strategy(self, usage_stats):
        """Suggest appropriate migration strategy based on usage."""
        total_complex_usage = sum(usage_stats.values())
        
        if total_complex_usage == 0:
            return "No migration needed - already using simple patterns"
        elif total_complex_usage < 5:
            return "Simple API migration recommended"
        elif total_complex_usage < 20:
            return "Standard API migration recommended"
        else:
            return "Enterprise API migration recommended - consider gradual migration"
    
    def generate_migration_examples(self, file_path):
        """Generate migration examples for a specific file."""
        content = Path(file_path).read_text()
        
        # Example migrations
        migrations = []
        
        # Detect create_pipeline usage
        if 'create_pipeline(' in content:
            migrations.append({
                'type': 'pipeline_creation',
                'before': 'create_pipeline(pipeline_type="basic", ...)',
                'after': 'RAG()',
                'description': 'Replace complex pipeline creation with Simple API'
            })
        
        # Detect manual document processing
        if 'Document(' in content and 'page_content' in content:
            migrations.append({
                'type': 'document_processing',
                'before': 'Document(page_content=content, metadata={...})',
                'after': 'rag.add_documents([content])',
                'description': 'Replace manual document creation with simple addition'
            })
        
        return migrations
    
    def create_migration_plan(self):
        """Create a comprehensive migration plan."""
        usage_stats = self.analyze_current_usage()
        strategy = self.suggest_migration_strategy(usage_stats)
        
        plan = {
            'current_usage': usage_stats,
            'recommended_strategy': strategy,
            'migration_steps': [],
            'estimated_effort': self.estimate_effort(usage_stats)
        }
        
        # Generate step-by-step plan
        if 'Simple API' in strategy:
            plan['migration_steps'] = [
                "1. Install new rag-templates library",
                "2. Replace create_pipeline() with RAG()",
                "3. Replace pipeline.query() with rag.query()",
                "4. Replace manual document processing with rag.add_documents()",
                "5. Test and validate functionality"
            ]
        elif 'Standard API' in strategy:
            plan['migration_steps'] = [
                "1. Install new rag-templates library",
                "2. Identify technique requirements",
                "3. Replace create_pipeline() with ConfigurableRAG()",
                "4. Migrate configuration to new format",
                "5. Update query calls to use new API",
                "6. Test and validate functionality"
            ]
        
        return plan
    
    def estimate_effort(self, usage_stats):
        """Estimate migration effort in hours."""
        total_usage = sum(usage_stats.values())
        
        if total_usage < 5:
            return "2-4 hours"
        elif total_usage < 20:
            return "1-2 days"
        else:
            return "3-5 days"

# Usage
if __name__ == "__main__":
    import sys
    
    if len(sys.argv) != 2:
        print("Usage: python migrate_rag.py <project_path>")
        sys.exit(1)
    
    project_path = sys.argv[1]
    migration_tool = RAGMigrationTool(project_path)
    
    print("🔍 Analyzing current rag-templates usage...")
    plan = migration_tool.create_migration_plan()
    
    print(f"\n📊 Current Usage Analysis:")
    for pattern, count in plan['current_usage'].items():
        print(f"  {pattern}: {count} occurrences")
    
    print(f"\n🎯 Recommended Strategy: {plan['recommended_strategy']}")
    print(f"⏱️  Estimated Effort: {plan['estimated_effort']}")
    
    print(f"\n📋 Migration Steps:")
    for step in plan['migration_steps']:
        print(f"  {step}")
    
    print(f"\n✅ Run this script with --execute to perform automated migration")
```

#### Usage
```bash
# Analyze current usage
python migrate_rag.py /path/to/your/project

# Example output:
# 🔍 Analyzing current rag-templates usage...
# 
# 📊 Current Usage Analysis:
#   complex_pipeline: 3 occurrences
#   config_manager: 2 occurrences
#   connection_manager: 1 occurrences
#   manual_storage: 1 occurrences
#   manual_embedding: 1 occurrences
# 
# 🎯 Recommended Strategy: Standard API migration recommended
# ⏱️  Estimated Effort: 1-2 days
# 
# 📋 Migration Steps:
#   1. Install new rag-templates library
#   2. Identify technique requirements
#   3. Replace create_pipeline() with ConfigurableRAG()
#   4. Migrate configuration to new format
#   5. Update query calls to use new API
#   6. Test and validate functionality
```

## Common Migration Patterns

### Pattern 1: Basic Pipeline to Simple API

**Before**:
```python
from iris_rag.pipelines.factory import create_pipeline
from common.utils import get_llm_func

def setup_rag():
    pipeline = create_pipeline(
        pipeline_type="basic",
        llm_func=get_llm_func()
    )
    return pipeline

def ask_question(pipeline, question):
    result = pipeline.query(question, top_k=5)
    return result['answer']

# Usage
pipeline = setup_rag()
answer = ask_question(pipeline, "What is AI?")
```

**After**:
```python
from rag_templates import RAG

def setup_rag():
    return RAG()

def ask_question(rag, question):
    return rag.query(question)

# Usage
rag = setup_rag()
answer = ask_question(rag, "What is AI?")
```

### Pattern 2: Configuration-Heavy to Standard API

**Before**:
```python
from iris_rag.config.manager import ConfigurationManager
from iris_rag.pipelines.factory import create_pipeline

def setup_advanced_rag():
    config = ConfigurationManager("advanced-config.yaml")
    pipeline = create_pipeline(
        pipeline_type="colbert",
        config_manager=config,
        llm_func=get_llm_func()
    )
    return pipeline

def advanced_query(pipeline, question):
    result = pipeline.query(
        question,
        top_k=10,
        similarity_threshold=0.8
    )
    return {
        'answer': result['answer'],
        'sources': result['retrieved_documents'],
        'confidence': result.get('confidence', 0.0)
    }
```

**After**:
```python
from rag_templates import ConfigurableRAG

def setup_advanced_rag():
    return ConfigurableRAG({
        "technique": "colbert",
        "llm_provider": "openai",
        "max_results": 10
    })

def advanced_query(rag, question):
    result = rag.query(question, {
        "max_results": 10,
        "min_similarity": 0.8,
        "include_sources": True
    })
    return {
        'answer': result.answer,
        'sources': result.sources,
        'confidence': result.confidence
    }
```

### Pattern 3: Custom Pipeline to Enterprise API

**Before**:
```python
from iris_rag.core.base import RAGPipeline
from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager

class CustomRAGPipeline(RAGPipeline):
    def __init__(self, connection_manager, config_manager):
        super().__init__(connection_manager, config_manager)
        self.custom_processor = CustomProcessor()
    
    def execute(self, query_text, **kwargs):
        # Custom logic
        processed_query = self.custom_processor.process(query_text)
        result = super().execute(processed_query, **kwargs)
        return self.custom_processor.post_process(result)

def setup_custom_rag():
    config = ConfigurationManager("custom-config.yaml")
    connection_manager = ConnectionManager(config)
    return CustomRAGPipeline(connection_manager, config)
```

**After**:
```python
from rag_templates import ConfigurableRAG
from rag_templates.config import ConfigManager

class CustomProcessor:
    def process(self, query):
        # Custom preprocessing
        return query
    
    def post_process(self, result):
        # Custom postprocessing
        return result

def setup_custom_rag():
    config = ConfigManager.from_file("custom-config.yaml")
    rag = ConfigurableRAG(config)
    
    # Add custom processing through middleware
    processor = CustomProcessor()
    
    original_query = rag.query
    def custom_query(query_text, **kwargs):
        processed_query = processor.process(query_text)
        result = original_query(processed_query, **kwargs)
        return processor.post_process(result)
    
    rag.query = custom_query
    return rag
```

## Troubleshooting

### Common Migration Issues

#### Issue 1: Import Errors

**Problem**: `ImportError: No module named 'rag_templates'`

**Solution**:
```bash
# Install the new library
pip install rag-templates

# For JavaScript
npm install @rag-templates/core
```

#### Issue 2: Configuration Not Found

**Problem**: `ConfigurationError: Configuration file not found`

**Solution**:
```python
# Before: Required configuration file
rag = RAG("config.yaml")  # Fails if file doesn't exist

# After: Optional configuration
rag = RAG()  # Works with defaults
# or
rag = RAG() if not os.path.exists("config.yaml") else RAG("config.yaml")
```

#### Issue 3: Different Query Results

**Problem**: Query results differ between old and new APIs

**Solution**:
```python
# Ensure same technique is used
old_pipeline = create_pipeline(pipeline_type="basic")
new_rag = ConfigurableRAG({"technique": "basic"})

# Use same parameters
old_result = old_pipeline.query(query, top_k=5)
new_result = new_rag.query(query, {"max_results": 5})

# Compare results
assert old_result['answer'] == new_result.answer
```

#### Issue 4: Performance Regression

**Problem**: New API is slower than old implementation

**Solution**:
```python
# Enable caching for better performance
rag = ConfigurableRAG({
    "technique": "basic",  # Use fastest technique
    "caching": {"enabled": True, "ttl": 3600},
    "embedding_config": {"cache_embeddings": True}
})

# Use batch processing
rag.add_documents(all_documents)  # Instead of one-by-one
```

#### Issue 5: Missing Features

**Problem**: Some advanced features not available in Simple API

**Solution**:
```python
# Upgrade to Standard or Enterprise API
from rag_templates import ConfigurableRAG

# Standard API has more features
rag = ConfigurableRAG({
    "technique": "colbert",
    "advanced_features": True
})

# Enterprise API has all features
from rag_templates.config import ConfigManager
config = ConfigManager.from_file("enterprise-config.yaml")
rag = ConfigurableRAG(config)
```

### Migration Validation

#### Validation Script
```python
def validate_migration(old_pipeline, new_rag, test_queries):
    """Validate that migration preserves functionality."""
    
    validation_results = []
    
    for query in test_queries:
        # Test old implementation
        old_result = old_pipeline.query(query)
        old_answer = old_result['answer']
        
        # Test new implementation
        new_answer = new_rag.query(query)
        
        # Compare results (allowing for minor differences)
        similarity = calculate_similarity(old_answer, new_answer)
        
        validation_results.append({
            'query': query,
            'old_answer': old_answer,
            'new_answer': new_answer,
            'similarity': similarity,
            'passed': similarity > 0.8  # 80% similarity threshold
        })
    
    # Generate report
    passed = sum(1 for r in validation_results if r['passed'])
    total = len(validation_results)
    
    print(f"Migration Validation Results: {passed}/{total} tests passed")
    
    for result in validation_results:
        status = "✅" if result['passed'] else "❌"
        print(f"{status} Query: {result['query'][:50]}...")
        print(f"   Similarity: {result['similarity']:.2f}")
        
        if not result['passed']:
            print(f"   Old: {result['old_answer'][:100]}...")
            print(f"   New: {result['new_answer'][:100]}...")
    
    return passed == total

# Usage
test_queries = [
    "What is machine learning?",
    "How does deep learning work?",
    "