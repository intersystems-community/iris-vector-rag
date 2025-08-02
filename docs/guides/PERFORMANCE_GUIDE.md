# RAG Templates Performance Guide

## Overview

This guide provides comprehensive performance optimization strategies for the RAG templates system in production environments. It covers pipeline optimization, IRIS database tuning, vector search performance, memory management, scaling strategies, and monitoring best practices using the actual [`iris_rag`](../../iris_rag/) architecture.

## Table of Contents

1. [Pipeline Performance Optimization](#pipeline-performance-optimization)
2. [IRIS Database Tuning](#iris-database-tuning)
3. [Vector Search Performance](#vector-search-performance)
4. [Memory Management](#memory-management)
5. [Scaling Strategies](#scaling-strategies)
6. [Performance Monitoring](#performance-monitoring)
7. [Benchmarking & Testing](#benchmarking--testing)
8. [Troubleshooting Performance Issues](#troubleshooting-performance-issues)

## Pipeline Performance Optimization

### RAG Pipeline Architecture

The RAG templates use a modular architecture with clear separation between retrieval, augmentation, and generation phases. Each pipeline inherits from [`RAGPipeline`](../../iris_rag/core/base.py) base class and uses the [`ConnectionManager`](../../iris_rag/core/connection.py) for database operations.

#### Performance Characteristics by Technique

Based on recent benchmark results from [`outputs/reports/benchmarks/`](../../outputs/reports/benchmarks/):

| Technique | Throughput (QPS) | Scalability | Best Use Case |
|-----------|------------------|-------------|---------------|
| BasicRAG | 73.30 q/s | Linear | Simple queries, fast responses |
| HyDE | 122.37 q/s | Good | Hypothetical document expansion |
| ColBERT | 4.23 q/s | Excellent | Token-level matching, high accuracy |
| CRAG | Variable | Good | Complex reasoning, accuracy critical |
| NodeRAG | Variable | Good | SQL-based reasoning |
| GraphRAG | Variable | Excellent | Knowledge graph queries |

**🚀 ColBERT Performance Notes**: While ColBERT shows lower throughput due to its sophisticated token-level matching, it provides superior accuracy for complex queries. The [`ColBERTRAGPipeline`](../../iris_rag/pipelines/colbert.py) implementation uses optimized batch processing for token embeddings.

### Pipeline Optimization Strategies

#### 1. Use iris_rag Architecture

The current system uses the [`iris_rag`](../../iris_rag/) package architecture with optimized implementations:

```python
from iris_rag.pipelines.basic import BasicRAGPipeline
from iris_rag.pipelines.colbert import ColBERTRAGPipeline
from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager

# Initialize with proper configuration
config_manager = ConfigurationManager()
connection_manager = ConnectionManager(config_manager)

# Create optimized pipeline
pipeline = BasicRAGPipeline(
    connection_manager=connection_manager,
    config_manager=config_manager
)
```

#### 2. Leverage Vector Database Optimizations

The system uses native IRIS VECTOR columns with proper indexing:

```python
# Vector operations use the insert_vector utility for consistency
from common.db_vector_utils import insert_vector

# All vector insertions use standardized format
success = insert_vector(
    cursor=cursor,
    table_name="RAG.SourceDocuments",
    vector_column_name="document_embedding_vector",
    vector_data=embedding,
    target_dimension=384,
    key_columns={"doc_id": doc_id}
)
```

#### 3. Optimize Configuration Parameters

Key performance parameters in [`config/config.yaml`](../../config/config.yaml):

```yaml
# Pipeline Configuration
pipelines:
  basic:
    chunk_size: 1000              # Optimize for your document size
    chunk_overlap: 200            # Balance context vs performance
    default_top_k: 5              # Limit retrieved documents
    embedding_batch_size: 32      # Batch embeddings for efficiency
  colbert:
    candidate_pool_size: 100      # Stage 1 retrieval size
    
# Storage Backend Configuration
storage:
  backends:
    iris:
      vector_dimension: 384       # Match your embedding model
      
# Testing Configuration
testing:
  min_docs_e2e: 1000             # Minimum for meaningful tests
```

#### 4. Implement LLM Caching

The system includes built-in LLM caching for performance:

```python
from common.llm_cache_manager import get_global_cache_manager

# LLM caching is automatically enabled
cache_manager = get_global_cache_manager()

# Monitor cache performance
cache_stats = cache_manager.get_cache_stats()
print(f"Cache hit rate: {cache_stats['metrics']['hit_rate']:.2%}")
print(f"Average cached response time: {cache_stats['metrics']['avg_response_time_cached']:.2f}ms")
```

#### 5. Batch Processing Optimization

Optimize batch sizes based on available memory and document characteristics:

```python
def optimize_batch_size(document_count, available_memory_gb):
    """Calculate optimal batch size based on system resources"""
    base_batch_size = 32  # From config.yaml embedding_batch_size
    
    if available_memory_gb >= 32:
        return min(128, document_count // 10)
    elif available_memory_gb >= 16:
        return min(64, document_count // 20)
    else:
        return base_batch_size
```

## IRIS Database Tuning

### Essential Performance Indexes

Create these indexes for optimal performance with the current schema:

```sql
-- Critical performance indexes for token operations (ColBERT)
CREATE INDEX idx_token_embeddings_doc_sequence 
ON RAG.DocumentTokenEmbeddings (doc_id, token_sequence_index);

CREATE INDEX idx_token_embeddings_sequence_only 
ON RAG.DocumentTokenEmbeddings (token_sequence_index);

-- Composite index for document identification
CREATE INDEX idx_source_docs_doc_id_title 
ON RAG.SourceDocuments (doc_id, title);

-- Vector search optimization for current tables
CREATE INDEX idx_document_vector_embedding 
ON RAG.SourceDocuments (document_embedding_vector) USING HNSW;

-- Additional performance indexes
CREATE INDEX idx_source_docs_embedding_not_null
ON RAG.SourceDocuments (doc_id) WHERE document_embedding_vector IS NOT NULL;
```

### HNSW Index Configuration

For production deployments with IRIS Enterprise Edition:

```sql
-- HNSW index with optimized parameters for current schema
CREATE INDEX idx_vector_hnsw ON RAG.SourceDocuments (document_embedding_vector) 
USING HNSW WITH (
    M = 16,           -- Number of connections (higher = better recall, more memory)
    EF_CONSTRUCTION = 200,  -- Construction parameter (higher = better quality)
    EF_SEARCH = 100   -- Search parameter (higher = better recall, slower search)
);

-- For ColBERT token embeddings (if using HNSW)
CREATE INDEX idx_token_vector_hnsw ON RAG.DocumentTokenEmbeddings (token_embedding_vector)
USING HNSW WITH (
    M = 8,            -- Lower M for token embeddings (more numerous)
    EF_CONSTRUCTION = 100,
    EF_SEARCH = 50
);
```

### Query Optimization

#### Use Proper Vector Search Syntax

Always use the [`common.db_vector_utils.insert_vector()`](../../common/db_vector_utils.py) utility for vector operations:

```sql
-- Optimized vector search with current schema
SELECT TOP 10 doc_id, title, text_content,
       VECTOR_COSINE(document_embedding_vector, 
                     TO_VECTOR(?, DOUBLE, 384)) AS similarity
FROM RAG.SourceDocuments
WHERE document_embedding_vector IS NOT NULL
ORDER BY similarity DESC;
```

**Important**: Always use `TOP` instead of `LIMIT` for IRIS SQL compatibility.

#### Connection Pool Configuration

Use the [`ConnectionManager`](../../iris_rag/core/connection.py) with proper configuration:

```python
from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager

# Configuration from config.yaml
config_manager = ConfigurationManager()
connection_manager = ConnectionManager(config_manager)

# Database configuration in config/config.yaml:
# database:
#   db_host: "localhost"
#   db_port: 1972
#   db_user: "SuperUser"
#   db_password: "SYS"
#   db_namespace: "USER"
```

### Database Maintenance

Regular maintenance tasks for optimal performance:

```sql
-- Update table statistics for current schema
UPDATE STATISTICS FOR TABLE RAG.SourceDocuments;
UPDATE STATISTICS FOR TABLE RAG.DocumentTokenEmbeddings;

-- Rebuild indexes periodically
REBUILD INDEX idx_vector_hnsw ON RAG.SourceDocuments;
REBUILD INDEX idx_token_embeddings_doc_sequence ON RAG.DocumentTokenEmbeddings;

-- Monitor index usage
SELECT * FROM INFORMATION_SCHEMA.INDEX_USAGE 
WHERE TABLE_NAME IN ('SourceDocuments', 'DocumentTokenEmbeddings');
```

## Vector Search Performance

### Embedding Generation Optimization

#### Batch Embedding Generation

```python
def optimized_batch_embeddings(texts, batch_size=32):
    """Generate embeddings in optimized batches"""
    embeddings = []
    
    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        batch_embeddings = embedding_model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
            normalize_embeddings=True  # Normalize for cosine similarity
        )
        embeddings.extend(batch_embeddings)
    
    return embeddings
```

#### Embedding Caching Strategy

```python
import pickle
import os
from pathlib import Path

class EmbeddingCache:
    def __init__(self, cache_dir="./embedding_cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_key(self, text):
        """Generate cache key from text hash"""
        return hashlib.md5(text.encode()).hexdigest()
    
    def get_embedding(self, text):
        """Get cached embedding or compute new one"""
        cache_key = self.get_cache_key(text)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        # Compute and cache new embedding
        embedding = self.embedding_func([text])[0]
        with open(cache_file, 'wb') as f:
            pickle.dump(embedding, f)
        
        return embedding
```

### Vector Search Optimization

#### Approximate Nearest Neighbor (ANN) Configuration

```python
# HNSW parameters for different use cases
HNSW_CONFIGS = {
    "speed_optimized": {
        "M": 8,
        "EF_CONSTRUCTION": 100,
        "EF_SEARCH": 50
    },
    "balanced": {
        "M": 16,
        "EF_CONSTRUCTION": 200,
        "EF_SEARCH": 100
    },
    "accuracy_optimized": {
        "M": 32,
        "EF_CONSTRUCTION": 400,
        "EF_SEARCH": 200
    }
}
```

#### Query Result Filtering

```python
def optimized_vector_search(query_embedding, top_k=10, similarity_threshold=0.7):
    """Optimized vector search with filtering"""
    sql = """
        SELECT doc_id, title, text_content, similarity
        FROM (
            SELECT doc_id, title, text_content,
                   VECTOR_COSINE(document_embedding_vector, 
                                TO_VECTOR(?, DOUBLE, 384)) AS similarity
            FROM RAG.SourceDocuments
            WHERE document_embedding_vector IS NOT NULL
        ) ranked
        WHERE similarity >= ?
        ORDER BY similarity DESC
        LIMIT ?
    """
    
    return cursor.execute(sql, [query_embedding, similarity_threshold, top_k])
```

## Memory Management

### Chunking Strategies

#### Adaptive Chunking

```python
def adaptive_chunk_size(document_length, target_chunks=10):
    """Calculate optimal chunk size based on document length"""
    base_chunk_size = 512
    max_chunk_size = 2048
    min_chunk_size = 256
    
    calculated_size = document_length // target_chunks
    return max(min_chunk_size, min(max_chunk_size, calculated_size))

def smart_chunking(text, chunk_size=None, overlap=0.1):
    """Intelligent text chunking with sentence boundary preservation"""
    if chunk_size is None:
        chunk_size = adaptive_chunk_size(len(text))
    
    sentences = text.split('. ')
    chunks = []
    current_chunk = ""
    
    for sentence in sentences:
        if len(current_chunk) + len(sentence) <= chunk_size:
            current_chunk += sentence + ". "
        else:
            if current_chunk:
                chunks.append(current_chunk.strip())
            current_chunk = sentence + ". "
    
    if current_chunk:
        chunks.append(current_chunk.strip())
    
    return chunks
```

### Memory Pool Management

```python
import gc
from typing import Optional

class MemoryManager:
    def __init__(self, max_memory_gb: float = 8.0):
        self.max_memory_bytes = max_memory_gb * 1024 * 1024 * 1024
        self.embedding_cache = {}
        
    def check_memory_usage(self):
        """Monitor current memory usage"""
        import psutil
        process = psutil.Process()
        return process.memory_info().rss
    
    def cleanup_if_needed(self):
        """Cleanup memory if usage exceeds threshold"""
        current_memory = self.check_memory_usage()
        
        if current_memory > self.max_memory_bytes * 0.8:  # 80% threshold
            # Clear embedding cache
            self.embedding_cache.clear()
            
            # Force garbage collection
            gc.collect()
            
            print(f"Memory cleanup performed. Usage: {current_memory / 1024**3:.2f}GB")
```

### Garbage Collection Optimization

```python
def optimize_gc_for_rag():
    """Configure garbage collection for RAG workloads"""
    import gc
    
    # Increase GC thresholds for better performance
    gc.set_threshold(1000, 15, 15)  # Increased from defaults
    
    # Disable automatic GC during critical operations
    gc.disable()
    
    # Manual GC after batch operations
    def cleanup_after_batch():
        gc.collect()
        gc.enable()
```

## Scaling Strategies

### Horizontal Scaling

#### Load Balancing Configuration

```python
class LoadBalancedRAG:
    def __init__(self, iris_connections):
        self.connections = iris_connections
        self.current_connection = 0
    
    def get_connection(self):
        """Round-robin connection selection"""
        conn = self.connections[self.current_connection]
        self.current_connection = (self.current_connection + 1) % len(self.connections)
        return conn
    
    def parallel_search(self, query, num_workers=4):
        """Parallel search across multiple connections"""
        from concurrent.futures import ThreadPoolExecutor
        
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            futures = []
            for i in range(num_workers):
                conn = self.get_connection()
                future = executor.submit(self._search_worker, conn, query)
                futures.append(future)
            
            results = []
            for future in futures:
                results.extend(future.result())
            
            return self._merge_results(results)
```

### Vertical Scaling

#### Resource Allocation Guidelines

| Document Count | RAM | CPU Cores | Storage | IRIS Config |
|----------------|-----|-----------|---------|-------------|
| 1K-5K | 8GB | 4 cores | 50GB SSD | Default |
| 5K-25K | 16GB | 8 cores | 100GB SSD | Increased buffers |
| 25K-100K | 32GB | 16 cores | 500GB SSD | Memory-optimized |
| 100K+ | 64GB+ | 24+ cores | 1TB+ SSD | Enterprise config |

#### IRIS Memory Configuration

```objectscript
// Optimize IRIS memory settings for large datasets
Set ^%SYS("BUFFERS") = 50000        // Increase buffer pool
Set ^%SYS("LOCKSIZ") = 16777216     // Increase lock table
Set ^%SYS("ROUTINES") = 512         // Routine buffer size
Set ^%SYS("GMHEAP") = 268435456     // Global memory heap
```

### Auto-Scaling Implementation

```python
class AutoScalingRAG:
    def __init__(self, base_config):
        self.base_config = base_config
        self.performance_metrics = []
    
    def monitor_performance(self, response_time, memory_usage):
        """Monitor performance metrics for scaling decisions"""
        self.performance_metrics.append({
            'timestamp': time.time(),
            'response_time': response_time,
            'memory_usage': memory_usage
        })
        
        # Keep only recent metrics
        cutoff = time.time() - 300  # 5 minutes
        self.performance_metrics = [
            m for m in self.performance_metrics 
            if m['timestamp'] > cutoff
        ]
    
    def should_scale_up(self):
        """Determine if scaling up is needed"""
        if len(self.performance_metrics) < 10:
            return False
        
        recent_response_times = [m['response_time'] for m in self.performance_metrics[-10:]]
        avg_response_time = sum(recent_response_times) / len(recent_response_times)
        
        return avg_response_time > 5.0  # Scale up if avg > 5 seconds
```

## Performance Monitoring

### Built-in Monitoring System

The system includes comprehensive monitoring via [`iris_rag.monitoring`](../../iris_rag/monitoring/):

#### Performance Monitor Usage

```python
from iris_rag.monitoring.performance_monitor import PerformanceMonitor, QueryPerformanceData
from iris_rag.monitoring.metrics_collector import MetricsCollector
from iris_rag.config.manager import ConfigurationManager
from datetime import datetime

# Initialize monitoring
config_manager = ConfigurationManager()
perf_monitor = PerformanceMonitor(config_manager)
metrics_collector = MetricsCollector()

# Start real-time monitoring
perf_monitor.start_monitoring()
metrics_collector.start_collection()

# Record query performance
query_data = QueryPerformanceData(
    query_text="What is machine learning?",
    pipeline_type="basic_rag",
    execution_time_ms=150.5,
    retrieval_time_ms=45.2,
    generation_time_ms=105.3,
    documents_retrieved=5,
    tokens_generated=150,
    timestamp=datetime.now(),
    success=True
)

perf_monitor.record_query_performance(query_data)

# Get performance summary
summary = perf_monitor.get_performance_summary(time_window_minutes=60)
print(f"Average response time: {summary['execution_time_stats']['avg_ms']:.2f}ms")
print(f"Success rate: {summary['success_rate']:.1f}%")
```

#### Key Performance Indicators (KPIs)

The monitoring system tracks:

- **Query Performance**: Execution time, retrieval time, generation time
- **System Metrics**: CPU usage, memory usage, disk usage
- **Database Metrics**: Document counts, vector query performance
- **Cache Performance**: LLM cache hit rates and speedup ratios

### Real-time Monitoring Dashboard

```python
# Get real-time status
status = perf_monitor.get_real_time_status()
print(f"Monitoring active: {status['monitoring_active']}")
print(f"Recent queries (5min): {status['recent_performance']['total_queries']}")

# Export metrics for analysis
perf_monitor.export_metrics(
    filepath="outputs/performance_metrics.json",
    time_window_minutes=60
)

# Collect cache metrics
cache_metrics = metrics_collector.collect_cache_metrics()
print(f"LLM Cache hit rate: {cache_metrics['llm_cache_hit_rate']:.2%}")
print(f"Cache speedup: {cache_metrics['llm_cache_speedup_ratio']:.1f}x")
```

### Alerting System

The [`PerformanceMonitor`](../../iris_rag/monitoring/performance_monitor.py) includes built-in threshold checking:

```python
# Configure performance thresholds
perf_monitor.thresholds = {
    'query_time_warning_ms': 1000,
    'query_time_critical_ms': 5000,
    'retrieval_time_warning_ms': 500,
    'retrieval_time_critical_ms': 2000,
    'generation_time_warning_ms': 3000,
    'generation_time_critical_ms': 10000
}

# Alerts are automatically logged when thresholds are exceeded
# Check logs for performance warnings and critical alerts
```

## Benchmarking & Testing

### Available Benchmarking Tools

The system includes comprehensive benchmarking capabilities:

#### Make Commands for Testing

```bash
# Run comprehensive tests with 1000 documents
make test-1000

# Run RAGAS evaluation on all pipelines
make eval-all-ragas-1000

# Quick performance debugging
make ragas-debug

# Full benchmark suite
make ragas-full

# Individual pipeline testing
make debug-ragas-basic
make debug-ragas-colbert
make debug-ragas-hyde
```

#### Benchmark Scripts

Key benchmarking scripts in [`scripts/utilities/evaluation/`](../../scripts/utilities/evaluation/):

- [`comprehensive_rag_benchmark_with_ragas.py`](../../scripts/utilities/evaluation/comprehensive_rag_benchmark_with_ragas.py) - Full RAGAS evaluation
- [`enterprise_rag_benchmark_final.py`](../../scripts/utilities/evaluation/enterprise_rag_benchmark_final.py) - Enterprise-scale benchmarks

#### Benchmark Results

Results are stored in [`outputs/reports/benchmarks/`](../../outputs/reports/benchmarks/) with:
- JSON results files
- Markdown reports
- Performance visualizations (radar charts, bar charts)

### Performance Regression Testing

Use the built-in monitoring system for regression testing:

```python
from iris_rag.monitoring.performance_monitor import PerformanceMonitor

# Establish baseline
baseline_summary = perf_monitor.get_performance_summary(time_window_minutes=60)
baseline_avg = baseline_summary['execution_time_stats']['avg_ms']

# After changes, compare performance
current_summary = perf_monitor.get_performance_summary(time_window_minutes=60)
current_avg = current_summary['execution_time_stats']['avg_ms']

regression_threshold = 1.2  # 20% slower is regression
if current_avg > baseline_avg * regression_threshold:
    print(f"REGRESSION DETECTED: {current_avg:.2f}ms vs {baseline_avg:.2f}ms baseline")
else:
    print(f"Performance OK: {current_avg:.2f}ms vs {baseline_avg:.2f}ms baseline")
```

## Troubleshooting Performance Issues

### Common Performance Problems

#### 1. Slow Vector Search

**Symptoms**: High query latency, timeouts
**Diagnosis**:
```sql
-- Check if HNSW indexes exist
SELECT * FROM INFORMATION_SCHEMA.INDEXES 
WHERE TABLE_NAME = 'SourceDocuments' 
AND INDEX_TYPE = 'HNSW';

-- Check vector search query plans
EXPLAIN SELECT * FROM RAG.SourceDocuments 
WHERE VECTOR_COSINE(document_embedding_vector, TO_VECTOR(?, DOUBLE, 384)) > 0.7;

-- Check for NULL embeddings
SELECT COUNT(*) as total_docs,
       COUNT(document_embedding_vector) as embedded_docs
FROM RAG.SourceDocuments;
```

**Solutions**:
- Create HNSW indexes on vector columns
- Optimize HNSW parameters (M, EF_CONSTRUCTION, EF_SEARCH)
- Ensure all documents have embeddings
- Use proper vector search syntax with `TO_VECTOR()`

#### 2. Memory Leaks

**Symptoms**: Increasing memory usage, OOM errors
**Diagnosis**:
```python
import tracemalloc

tracemalloc.start()

# Run your RAG pipeline
result = rag_pipeline.query(query)

# Check memory usage
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
```

**Solutions**:
- Implement proper garbage collection
- Clear embedding caches periodically
- Use memory pools for large operations
- Monitor with built-in [`MetricsCollector`](../../iris_rag/monitoring/metrics_collector.py)

#### 3. Database Connection Issues

**Symptoms**: Connection timeouts, pool exhaustion
**Diagnosis**:
```python
# Monitor connection pool status using ConnectionManager
from iris_rag.core.connection import ConnectionManager

def check_connection_health(connection_manager):
    try:
        connection = connection_manager.get_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT 1")
        print("Connection: Healthy")
        return True
    except Exception as e:
        print(f"Connection: Unhealthy - {e}")
        return False
```

**Solutions**:
- Use proper [`ConnectionManager`](../../iris_rag/core/connection.py) configuration
- Implement connection health checks
- Add connection retry logic
- Monitor database metrics with built-in monitoring

#### 4. ColBERT Token Embedding Performance

**Symptoms**: Slow ColBERT queries, high memory usage
**Diagnosis**:
```sql
-- Check token embedding count
SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings;

-- Check for missing token embeddings
SELECT d.doc_id, d.title
FROM RAG.SourceDocuments d
LEFT JOIN RAG.DocumentTokenEmbeddings t ON d.doc_id = t.doc_id
WHERE t.doc_id IS NULL;

-- Check token embedding distribution
SELECT doc_id, COUNT(*) as token_count
FROM RAG.DocumentTokenEmbeddings
GROUP BY doc_id
ORDER BY token_count DESC
LIMIT 10;
```

**Solutions**:
- Ensure all documents have token embeddings
- Use batch processing for token embedding generation
- Implement proper indexing on token tables
- Consider token embedding caching strategies

### Performance Profiling

```python
import cProfile
import pstats

def profile_rag_pipeline(rag_pipeline, query):
    """Profile RAG pipeline performance"""
    profiler = cProfile.Profile()
    
    profiler.enable()
    result = rag_pipeline.query(query)
    profiler.disable()
    
    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
    
    return result
```

### Optimization Checklist

- [ ] Use [`iris_rag`](../../iris_rag/) architecture for optimized implementations
- [ ] Create appropriate database indexes (HNSW for vectors)
- [ ] Configure HNSW parameters for your use case
- [ ] Implement LLM caching with [`llm_cache_manager`](../../common/llm_cache_manager.py)
- [ ] Optimize batch sizes in [`config.yaml`](../../config/config.yaml)
- [ ] Monitor memory usage and implement cleanup
- [ ] Set up performance monitoring with [`iris_rag.monitoring`](../../iris_rag/monitoring/)
- [ ] Run regular performance regression tests
- [ ] Profile slow operations to identify bottlenecks
- [ ] Use [`ConnectionManager`](../../iris_rag/core/connection.py) for database connections
- [ ] Always use [`insert_vector`](../../common/db_vector_utils.py) utility for vector operations
- [ ] Follow IRIS SQL rules (use `TOP` instead of `LIMIT`)

## Conclusion

This performance guide provides a comprehensive framework for optimizing the RAG templates system in production environments. The key to success is:

1. **Use the iris_rag architecture** for optimized, production-ready implementations
2. **Monitor continuously** with built-in monitoring tools
3. **Scale incrementally** based on actual usage patterns
4. **Test regularly** with comprehensive benchmarking tools
5. **Follow best practices** for IRIS database optimization

For specific implementation details, refer to the actual code in [`iris_rag/`](../../iris_rag/) and benchmark results in [`outputs/reports/benchmarks/`](../../outputs/reports/benchmarks/).

### Quick Start Commands

```bash
# Set up environment
make setup-env
make install
make setup-db

# Run performance tests
make test-1000
make ragas-full

# Monitor performance
python -c "
from iris_rag.monitoring.performance_monitor import PerformanceMonitor
monitor = PerformanceMonitor()
monitor.start_monitoring()
print('Performance monitoring started')
"
```

### Additional Resources

- [Configuration Guide](../../config/config.yaml) - System configuration options
- [Monitoring Documentation](../../iris_rag/monitoring/) - Built-in monitoring capabilities
- [Benchmark Results](../../outputs/reports/benchmarks/) - Historical performance data
- [Testing Guide](../../Makefile) - Available testing commands
- [Database Utilities](../../common/db_vector_utils.py) - Vector operation utilities