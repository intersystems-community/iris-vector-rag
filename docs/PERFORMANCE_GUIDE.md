# RAG Templates Performance Guide

## Overview

This guide provides comprehensive performance optimization strategies for RAG templates in production environments. It covers pipeline optimization, IRIS database tuning, vector search performance, memory management, scaling strategies, and monitoring best practices.

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

The RAG templates use a modular architecture with clear separation between retrieval, augmentation, and generation phases. Each pipeline inherits from [`RAGPipeline`](../rag_templates/core/base.py) base class.

#### Performance Characteristics by Technique

| Technique | Avg Response Time | Scalability | Best Use Case |
|-----------|------------------|-------------|---------------|
| BasicRAG | 20-30ms (100 docs) | Linear | Simple queries, fast responses |
| CRAG | 2.33s (V2) | Good | Complex reasoning, accuracy critical |
| HyDE | 5.56s (V2) | Good | Hypothetical document expansion |
| GraphRAG | 1.63s (V2) | Excellent | Knowledge graph queries |
| **ColBERT** | **~0.039s per doc (Optimized)** | **Excellent** | **Token-level matching, enterprise-ready** |
| NodeRAG | Variable | Good | SQL-based reasoning |

**🚀 ColBERT Performance Breakthrough (June 2025)**: The [`_retrieve_documents_with_colbert`](../iris_rag/pipelines/colbert.py) method has been optimized with **~99.4% performance improvement**, transforming from I/O-bound to compute-bound behavior through batch loading and single-pass parsing optimizations.

### Pipeline Optimization Strategies

#### 0. ColBERT Performance Breakthrough (June 2025)

**Major Achievement**: The [`_retrieve_documents_with_colbert`](../iris_rag/pipelines/colbert.py) method has been dramatically optimized, achieving a **~99.4% performance improvement**.

**Key Optimizations Implemented:**

1. **Batch Loading of Token Embeddings**: All 206,306+ token embeddings are now loaded from [`RAG.DocumentTokenEmbeddings`](../iris_rag/storage/iris.py) in a **single SQL query**, eliminating the N+1 query problem.

2. **Efficient Single-Pass Parsing**: Each embedding string is parsed into a numerical array **only once** during the batch load, dramatically reducing computational overhead.

3. **In-Memory Processing**: MaxSim calculations are performed on pre-loaded and pre-parsed in-memory data.

**Performance Impact:**
- **Before**: ~6-9 seconds per document (I/O-bound behavior)
- **After**: ~0.039 seconds per document (compute-bound behavior)
- **Improvement**: ~99.4% reduction in retrieval processing time
- **Database Queries**: Reduced from O(n) to O(1) for token embeddings

**Behavioral Shift:**
- **Previous**: I/O-bound with inefficient database access patterns
- **Current**: Compute-bound focused on MaxSim calculations
- **Result**: Enterprise-ready performance for large document collections

This optimization demonstrates that with proper engineering, even the most complex RAG techniques can achieve production-ready performance while maintaining their advanced semantic capabilities.

#### 1. Use V2 Implementations

V2 pipelines provide significant performance improvements by using native VECTOR columns:

```python
# V2 pipelines avoid IRIS SQL parser bugs and provide:
# - 2-6x faster performance for most techniques
# - True vector search across entire corpus
# - Better scalability for large document collections

# Example: CRAG V2 vs Original
# Original: 13.51s → V2: 2.33s (5.80x faster)
```

#### 2. Optimize Document Sampling

For techniques that sample documents, optimize batch sizes:

```python
# BasicRAG optimization
class OptimizedBasicRAG(BasicRAG):
    def __init__(self, sample_size=500):  # Increased from 100
        self.sample_size = sample_size
    
    def retrieve_documents(self, query_text, top_k=5):
        # Use larger sample for better coverage
        sql = f"""
            SELECT TOP {self.sample_size} doc_id, title, text_content, embedding
            FROM {self.schema}.SourceDocuments_V2
            WHERE document_embedding_vector IS NOT NULL
            ORDER BY NEWID()  -- Random sampling
        """
```

#### 3. Implement Result Caching

Cache frequently accessed results to reduce computation:

```python
from functools import lru_cache
import hashlib

class CachedRAGPipeline(RAGPipeline):
    @lru_cache(maxsize=1000)
    def _cached_embedding(self, text):
        """Cache embeddings for frequently used queries"""
        return self.embedding_func([text])[0]
    
    @lru_cache(maxsize=500)
    def _cached_similarity_search(self, query_hash, top_k):
        """Cache similarity search results"""
        # Implementation depends on specific technique
        pass
```

#### 4. Batch Processing Optimization

Optimize batch sizes based on available memory and document characteristics:

```python
def optimize_batch_size(document_count, available_memory_gb):
    """Calculate optimal batch size based on system resources"""
    base_batch_size = 50
    
    if available_memory_gb >= 32:
        return min(200, document_count // 10)
    elif available_memory_gb >= 16:
        return min(100, document_count // 20)
    else:
        return base_batch_size
```

## IRIS Database Tuning

### Essential Performance Indexes

Create these indexes for optimal performance:

```sql
-- Critical performance indexes for token operations
CREATE INDEX idx_token_embeddings_doc_sequence 
ON RAG.DocumentTokenEmbeddings (doc_id, token_sequence_index);

CREATE INDEX idx_token_embeddings_sequence_only 
ON RAG.DocumentTokenEmbeddings (token_sequence_index);

-- Composite index for document identification
CREATE INDEX idx_source_docs_doc_id_title 
ON RAG.SourceDocuments (doc_id, title);

-- Vector search optimization (V2 tables)
CREATE INDEX idx_document_vector_embedding 
ON RAG.SourceDocuments_V2 (document_embedding_vector) USING HNSW;
```

### HNSW Index Configuration

For production deployments with IRIS Enterprise Edition:

```sql
-- HNSW index with optimized parameters
CREATE INDEX idx_vector_hnsw ON RAG.SourceDocuments_V2 (document_embedding_vector) 
USING HNSW WITH (
    M = 16,           -- Number of connections (higher = better recall, more memory)
    EF_CONSTRUCTION = 200,  -- Construction parameter (higher = better quality)
    EF_SEARCH = 100   -- Search parameter (higher = better recall, slower search)
);
```

### Query Optimization

#### Use Proper Vector Search Syntax

```sql
-- Optimized V2 vector search (avoids parser bugs)
SELECT TOP 10 doc_id, title, text_content,
       VECTOR_COSINE(document_embedding_vector, 
                     TO_VECTOR(:query_embedding, DOUBLE, 384)) AS similarity
FROM RAG.SourceDocuments_V2
WHERE document_embedding_vector IS NOT NULL
ORDER BY similarity DESC;
```

#### Connection Pool Configuration

```python
from rag_templates.core.connection import ConnectionManager

# Optimize connection pooling
config = {
    "database:iris": {
        "driver": "intersystems_iris.dbapi._DBAPI",
        "host": "localhost",
        "port": 1972,
        "namespace": "USER",
        "username": "demo",
        "password": "demo",
        "pool_size": 10,        # Connection pool size
        "max_overflow": 20,     # Additional connections
        "pool_timeout": 30,     # Connection timeout
        "pool_recycle": 3600    # Recycle connections hourly
    }
}
```

### Database Maintenance

Regular maintenance tasks for optimal performance:

```sql
-- Update table statistics
UPDATE STATISTICS FOR TABLE RAG.SourceDocuments_V2;

-- Rebuild indexes periodically
REBUILD INDEX idx_vector_hnsw ON RAG.SourceDocuments_V2;

-- Monitor index usage
SELECT * FROM INFORMATION_SCHEMA.INDEX_USAGE 
WHERE TABLE_NAME LIKE 'SourceDocuments%';
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
            FROM RAG.SourceDocuments_V2
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

### Key Performance Indicators (KPIs)

#### Response Time Metrics

```python
import time
from dataclasses import dataclass
from typing import List

@dataclass
class PerformanceMetrics:
    query_time: float
    retrieval_time: float
    generation_time: float
    total_time: float
    documents_retrieved: int
    memory_usage: float

class PerformanceMonitor:
    def __init__(self):
        self.metrics: List[PerformanceMetrics] = []
    
    def measure_query_performance(self, rag_pipeline, query):
        """Comprehensive performance measurement"""
        start_time = time.time()
        
        # Measure retrieval phase
        retrieval_start = time.time()
        documents = rag_pipeline.query(query)
        retrieval_time = time.time() - retrieval_start
        
        # Measure generation phase
        generation_start = time.time()
        result = rag_pipeline.execute(query)
        generation_time = time.time() - generation_start
        
        total_time = time.time() - start_time
        
        metrics = PerformanceMetrics(
            query_time=total_time,
            retrieval_time=retrieval_time,
            generation_time=generation_time,
            total_time=total_time,
            documents_retrieved=len(documents),
            memory_usage=self._get_memory_usage()
        )
        
        self.metrics.append(metrics)
        return result, metrics
```

### Real-time Monitoring Dashboard

```python
def generate_performance_report(metrics: List[PerformanceMetrics]):
    """Generate comprehensive performance report"""
    if not metrics:
        return "No metrics available"
    
    total_queries = len(metrics)
    avg_response_time = sum(m.total_time for m in metrics) / total_queries
    p95_response_time = sorted([m.total_time for m in metrics])[int(0.95 * total_queries)]
    
    report = f"""
    Performance Report
    ==================
    Total Queries: {total_queries}
    Average Response Time: {avg_response_time:.3f}s
    P95 Response Time: {p95_response_time:.3f}s
    Average Documents Retrieved: {sum(m.documents_retrieved for m in metrics) / total_queries:.1f}
    Average Memory Usage: {sum(m.memory_usage for m in metrics) / total_queries:.2f}GB
    """
    
    return report
```

### Alerting System

```python
class PerformanceAlerting:
    def __init__(self, thresholds):
        self.thresholds = thresholds
        self.alert_history = []
    
    def check_alerts(self, metrics: PerformanceMetrics):
        """Check for performance threshold violations"""
        alerts = []
        
        if metrics.total_time > self.thresholds.get('max_response_time', 10.0):
            alerts.append(f"High response time: {metrics.total_time:.2f}s")
        
        if metrics.memory_usage > self.thresholds.get('max_memory_gb', 16.0):
            alerts.append(f"High memory usage: {metrics.memory_usage:.2f}GB")
        
        for alert in alerts:
            self._send_alert(alert)
    
    def _send_alert(self, message):
        """Send alert notification"""
        print(f"ALERT: {message}")
        # Implement actual alerting (email, Slack, etc.)
```

## Benchmarking & Testing

### Performance Regression Testing

```python
class PerformanceRegressionTest:
    def __init__(self, baseline_metrics):
        self.baseline = baseline_metrics
    
    def run_regression_test(self, rag_pipeline, test_queries):
        """Run performance regression test"""
        current_metrics = []
        
        for query in test_queries:
            start_time = time.time()
            result = rag_pipeline.execute(query)
            end_time = time.time()
            
            current_metrics.append(end_time - start_time)
        
        return self._compare_with_baseline(current_metrics)
    
    def _compare_with_baseline(self, current_metrics):
        """Compare current performance with baseline"""
        current_avg = sum(current_metrics) / len(current_metrics)
        baseline_avg = sum(self.baseline) / len(self.baseline)
        
        regression_threshold = 1.2  # 20% slower is regression
        
        if current_avg > baseline_avg * regression_threshold:
            return {
                'status': 'REGRESSION',
                'current_avg': current_avg,
                'baseline_avg': baseline_avg,
                'degradation': (current_avg / baseline_avg - 1) * 100
            }
        
        return {'status': 'PASS', 'current_avg': current_avg, 'baseline_avg': baseline_avg}
```

### Load Testing Framework

```python
import concurrent.futures
import random

class LoadTester:
    def __init__(self, rag_pipeline, test_queries):
        self.rag_pipeline = rag_pipeline
        self.test_queries = test_queries
    
    def run_load_test(self, concurrent_users=10, duration_seconds=60):
        """Run load test with concurrent users"""
        results = []
        start_time = time.time()
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrent_users) as executor:
            futures = []
            
            while time.time() - start_time < duration_seconds:
                query = random.choice(self.test_queries)
                future = executor.submit(self._execute_query, query)
                futures.append(future)
                
                time.sleep(0.1)  # 10 QPS per user
            
            for future in concurrent.futures.as_completed(futures):
                results.append(future.result())
        
        return self._analyze_load_test_results(results)
```

## Troubleshooting Performance Issues

### Common Performance Problems

#### 1. Slow Vector Search

**Symptoms**: High query latency, timeouts
**Diagnosis**:
```sql
-- Check if HNSW indexes exist
SELECT * FROM INFORMATION_SCHEMA.INDEXES 
WHERE TABLE_NAME = 'SourceDocuments_V2' 
AND INDEX_TYPE = 'HNSW';

-- Check vector search query plans
EXPLAIN SELECT * FROM RAG.SourceDocuments_V2 
WHERE VECTOR_COSINE(document_embedding_vector, TO_VECTOR(?, DOUBLE, 384)) > 0.7;
```

**Solutions**:
- Create HNSW indexes on vector columns
- Optimize HNSW parameters (M, EF_CONSTRUCTION, EF_SEARCH)
- Use V2 pipelines to avoid SQL parser bugs

#### 2. Memory Leaks

**Symptoms**: Increasing memory usage, OOM errors
**Diagnosis**:
```python
import tracemalloc

tracemalloc.start()

# Run your RAG pipeline
result = rag_pipeline.execute(query)

# Check memory usage
current, peak = tracemalloc.get_traced_memory()
print(f"Current memory usage: {current / 1024 / 1024:.1f} MB")
print(f"Peak memory usage: {peak / 1024 / 1024:.1f} MB")
```

**Solutions**:
- Implement proper garbage collection
- Clear embedding caches periodically
- Use memory pools for large operations

#### 3. Database Connection Issues

**Symptoms**: Connection timeouts, pool exhaustion
**Diagnosis**:
```python
# Monitor connection pool status
def check_connection_pool_health(connection_manager):
    active_connections = len(connection_manager._connections)
    print(f"Active connections: {active_connections}")
    
    # Test connection health
    for name, conn in connection_manager._connections.items():
        try:
            cursor = conn.cursor()
            cursor.execute("SELECT 1")
            print(f"Connection {name}: Healthy")
        except Exception as e:
            print(f"Connection {name}: Unhealthy - {e}")
```

**Solutions**:
- Increase connection pool size
- Implement connection health checks
- Add connection retry logic

### Performance Profiling

```python
import cProfile
import pstats

def profile_rag_pipeline(rag_pipeline, query):
    """Profile RAG pipeline performance"""
    profiler = cProfile.Profile()
    
    profiler.enable()
    result = rag_pipeline.execute(query)
    profiler.disable()
    
    # Analyze results
    stats = pstats.Stats(profiler)
    stats.sort_stats('cumulative')
    stats.print_stats(20)  # Top 20 functions
    
    return result
```

### Optimization Checklist

- [ ] Use V2 pipelines for better performance
- [ ] Create appropriate database indexes
- [ ] Configure HNSW parameters for your use case
- [ ] Implement embedding caching
- [ ] Optimize batch sizes for your hardware
- [ ] Monitor memory usage and implement cleanup
- [ ] Set up performance monitoring and alerting
- [ ] Run regular performance regression tests
- [ ] Profile slow operations to identify bottlenecks
- [ ] Configure connection pooling appropriately

## Conclusion

This performance guide provides a comprehensive framework for optimizing RAG templates in production environments. The key to success is:

1. **Start with V2 pipelines** for immediate performance gains
2. **Monitor continuously** to identify bottlenecks early
3. **Scale incrementally** based on actual usage patterns
4. **Test regularly** to prevent performance regressions

For specific implementation details, refer to the existing performance documentation and benchmark results in the project.