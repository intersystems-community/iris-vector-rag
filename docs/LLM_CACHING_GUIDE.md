# LLM Caching Guide

This guide provides comprehensive documentation for the intelligent LLM response caching layer in the RAG Templates framework. The caching system significantly reduces API costs and improves response times by storing LLM responses in the IRIS database.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [API Reference](#api-reference)
- [Performance Optimization](#performance-optimization)
- [Monitoring and Analytics](#monitoring-and-analytics)
- [Troubleshooting](#troubleshooting)
- [Best Practices](#best-practices)

## Overview

The LLM caching layer provides intelligent caching of Large Language Model responses to:

- **Reduce API Costs**: Cache frequently asked questions to avoid repeated API calls
- **Improve Performance**: Return cached responses instantly (10-100x faster)
- **Enhance Reliability**: Continue operation even when LLM APIs are unavailable
- **Enable Analytics**: Track usage patterns and optimize cache configuration

### Key Features

- **IRIS Backend**: Leverages existing IRIS database infrastructure
- **Langchain Integration**: Seamless integration with Langchain's caching system
- **Intelligent Key Generation**: SHA256-based cache keys with configurable parameters
- **TTL Support**: Automatic expiration of cached responses
- **Performance Metrics**: Built-in hit/miss tracking and performance monitoring
- **Graceful Fallback**: Continues operation even if cache is unavailable
- **Auto-cleanup**: Automatic removal of expired cache entries

## Architecture

```
┌─────────────────┐    ┌─────────────────┐    ┌─────────────────┐
│   Application   │    │ Cache Manager   │    │ IRIS Database   │
│                 │────│                 │────│                 │
│ • LLM Calls     │    │ • Key Gen       │    │ • llm_cache     │
│ • RAG Pipeline  │    │ • TTL Mgmt      │    │ • Persistence   │
│ • Langchain     │    │ • Metrics       │    │ • Indexing      │
└─────────────────┘    └─────────────────┘    └─────────────────┘
         │                       │                       │
         └───────────────────────┼───────────────────────┘
                                 │
                    ┌─────────────────┐
                    │ Configuration   │
                    │                 │
                    │ • YAML Config   │
                    │ • Env Variables │
                    │ • Validation    │
                    └─────────────────┘
```

### Components

1. **Cache Manager** ([`common/llm_cache_manager.py`](../common/llm_cache_manager.py)): Main orchestration layer
2. **IRIS Backend** ([`common/llm_cache_iris.py`](../common/llm_cache_iris.py)): IRIS database integration
3. **Configuration** ([`common/llm_cache_config.py`](../common/llm_cache_config.py)): Configuration management
4. **Langchain Wrapper**: Integration with Langchain's caching system

## Quick Start

### Basic Setup

```python
from common.llm_cache_manager import setup_langchain_cache
from common.llm_cache_config import load_cache_config

# Setup caching with default configuration
config = load_cache_config()
cache = setup_langchain_cache(config)

# Caching is now automatically applied to all LLM calls
```

### Using with RAG Pipelines

```python
from iris_rag import create_pipeline
from common.llm_cache_manager import setup_langchain_cache

# Enable caching
setup_langchain_cache()

# Create pipeline - caching is automatically enabled
pipeline = create_pipeline("basic", config_path="config.yaml")
result = pipeline.execute("What is machine learning?")  # Cache miss
result = pipeline.execute("What is machine learning?")  # Cache hit!
```

### Manual LLM Function Usage

```python
from common.utils import get_llm_func
from common.llm_cache_manager import get_cache_stats

# Create LLM function with caching enabled
llm_func = get_llm_func(
    provider="openai",
    model_name="gpt-3.5-turbo",
    enable_cache=True
)

# First call - cache miss
response1 = llm_func("Explain quantum computing")

# Second call - cache hit
response2 = llm_func("Explain quantum computing")

# Check performance
stats = get_cache_stats()
print(f"Hit rate: {stats['metrics']['hit_rate']:.2%}")
```

## Configuration

### Configuration File

The caching layer is configured via [`config/cache_config.yaml`](../config/cache_config.yaml):

```yaml
llm_cache:
  # Core settings
  enabled: true
  backend: "iris"
  ttl_seconds: 3600
  normalize_prompts: false
  max_cache_size: 1000
  
  # IRIS-specific configuration
  iris:
    table_name: "llm_cache"
    schema: "USER"
    connection_timeout: 30
    cleanup_batch_size: 1000
    auto_cleanup: true
    cleanup_interval: 86400
  
  # Cache key generation
  key_generation:
    include_temperature: true
    include_max_tokens: true
    include_model_name: true
    hash_algorithm: "sha256"
    normalize_whitespace: true
    normalize_case: false
  
  # Monitoring
  monitoring:
    enabled: true
    log_operations: false
    track_stats: true
    metrics_interval: 300
  
  # Error handling
  error_handling:
    graceful_fallback: true
    max_retries: 3
    retry_delay: 1
    operation_timeout: 10
```

### Environment Variables

Override configuration with environment variables:

```bash
# Core settings
export LLM_CACHE_ENABLED=true
export LLM_CACHE_BACKEND=iris
export LLM_CACHE_TTL=3600
export LLM_CACHE_NORMALIZE_PROMPTS=false
export LLM_CACHE_MAX_SIZE=1000

# IRIS settings
export LLM_CACHE_TABLE=llm_cache
export LLM_CACHE_IRIS_SCHEMA=USER
```

### Configuration Options

| Setting | Description | Default | Options |
|---------|-------------|---------|---------|
| `enabled` | Enable/disable caching | `true` | `true`, `false` |
| `backend` | Cache backend type | `"iris"` | `"memory"`, `"iris"` |
| `ttl_seconds` | Time-to-live for cache entries | `3600` | Any positive integer |
| `table_name` | IRIS table name | `"llm_cache"` | Any valid table name |
| `hash_algorithm` | Hash algorithm for keys | `"sha256"` | `"sha256"`, `"md5"`, `"sha1"` |

## API Reference

### Core Functions

#### `setup_langchain_cache(config=None)`

Setup Langchain cache based on configuration.

```python
from common.llm_cache_manager import setup_langchain_cache
from common.llm_cache_config import CacheConfig

# Use default configuration
cache = setup_langchain_cache()

# Use custom configuration
config = CacheConfig(enabled=True, backend="iris", ttl_seconds=7200)
cache = setup_langchain_cache(config)
```

#### `get_cache_stats()`

Get cache performance statistics.

```python
from common.llm_cache_manager import get_cache_stats

stats = get_cache_stats()
print(f"Hit rate: {stats['metrics']['hit_rate']:.2%}")
print(f"Total requests: {stats['metrics']['total_requests']}")
```

#### `clear_global_cache()`

Clear all cached entries.

```python
from common.llm_cache_manager import clear_global_cache

clear_global_cache()
```

### Cache Manager Class

#### `LangchainCacheManager`

Main cache management class.

```python
from common.llm_cache_manager import LangchainCacheManager
from common.llm_cache_config import load_cache_config

config = load_cache_config()
manager = LangchainCacheManager(config)

# Setup cache
cache = manager.setup_cache()

# Get statistics
stats = manager.get_cache_stats()

# Clear cache
manager.clear_cache()
```

### IRIS Backend

#### `IRISCacheBackend`

IRIS database backend for cache storage.

```python
from common.llm_cache_iris import IRISCacheBackend
from common.utils import get_iris_connector

iris_connector = get_iris_connector()
backend = IRISCacheBackend(
    iris_connector=iris_connector,
    table_name="llm_cache",
    ttl_seconds=3600
)

# Store value
backend.set("key", "value", ttl=7200)

# Retrieve value
value = backend.get("key")

# Get cache info
info = backend.get_cache_info()
```

## Performance Optimization

### Cache Key Optimization

The cache key generation affects cache hit rates. Configure key generation parameters:

```yaml
key_generation:
  include_temperature: true    # Include temperature in cache key
  include_max_tokens: true     # Include max_tokens in cache key
  include_model_name: true     # Include model name in cache key
  normalize_whitespace: true   # Normalize whitespace in prompts
  normalize_case: false        # Normalize case in prompts
```

### TTL Configuration

Configure appropriate TTL values based on your use case:

- **Short TTL (1 hour)**: For rapidly changing content
- **Medium TTL (24 hours)**: For general Q&A
- **Long TTL (7 days)**: For stable reference content

```python
from common.llm_cache_config import CacheConfig

# Short TTL for dynamic content
config = CacheConfig(ttl_seconds=3600)

# Long TTL for stable content
config = CacheConfig(ttl_seconds=604800)
```

### Cleanup Configuration

Configure automatic cleanup to maintain performance:

```yaml
iris:
  auto_cleanup: true
  cleanup_interval: 86400      # 24 hours
  cleanup_batch_size: 1000     # Process 1000 entries at a time
```

## Monitoring and Analytics

### Cache Statistics

Monitor cache performance with built-in metrics:

```python
from common.llm_cache_manager import get_cache_stats

stats = get_cache_stats()

print(f"Cache enabled: {stats['enabled']}")
print(f"Backend: {stats['backend']}")
print(f"Hit rate: {stats['metrics']['hit_rate']:.2%}")
print(f"Total requests: {stats['metrics']['total_requests']}")
print(f"Cache hits: {stats['metrics']['hits']}")
print(f"Cache misses: {stats['metrics']['misses']}")
```

### Database Analytics

Get detailed cache information from the database:

```python
from common.llm_cache_iris import IRISCacheBackend
from common.utils import get_iris_connector

backend = IRISCacheBackend(get_iris_connector(), "llm_cache")
info = backend.get_cache_info()

print(f"Total entries: {info['total_entries']}")
print(f"Active entries: {info['active_entries']}")
print(f"Expired entries: {info['expired_entries']}")
```

### Performance Metrics

Track response time improvements:

```python
import time
from common.llm_cache_manager import get_global_cache_manager

manager = get_global_cache_manager()

# Measure cache hit performance
start_time = time.time()
response = llm_func("What is AI?")  # Cache hit
hit_time = time.time() - start_time

print(f"Cache hit response time: {hit_time:.3f}s")
```

## Troubleshooting

### Common Issues

#### Cache Not Working

1. **Check if caching is enabled**:
   ```python
   from common.llm_cache_manager import is_langchain_cache_configured
   print(f"Cache configured: {is_langchain_cache_configured()}")
   ```

2. **Verify configuration**:
   ```python
   from common.llm_cache_config import load_cache_config
   config = load_cache_config()
   print(f"Cache enabled: {config.enabled}")
   print(f"Backend: {config.backend}")
   ```

3. **Check IRIS connection**:
   ```python
   from common.utils import get_iris_connector
   try:
       connector = get_iris_connector()
       print("IRIS connection successful")
   except Exception as e:
       print(f"IRIS connection failed: {e}")
   ```

#### Low Cache Hit Rate

1. **Check key generation settings**: Ensure consistent parameters
2. **Review TTL settings**: May be too short for your use case
3. **Analyze prompt variations**: Use normalization options

#### Performance Issues

1. **Monitor cache size**: Large caches may impact performance
2. **Enable auto-cleanup**: Remove expired entries regularly
3. **Optimize TTL**: Balance between hit rate and freshness

### Debug Mode

Enable debug logging for troubleshooting:

```python
import logging

# Enable debug logging
logging.getLogger('common.llm_cache_manager').setLevel(logging.DEBUG)
logging.getLogger('common.llm_cache_iris').setLevel(logging.DEBUG)

# Enable operation logging
config = load_cache_config()
config.log_operations = True
```

### Error Handling

The cache system includes graceful error handling:

```yaml
error_handling:
  graceful_fallback: true      # Continue without cache on errors
  max_retries: 3               # Retry failed operations
  retry_delay: 1               # Delay between retries
  operation_timeout: 10        # Timeout for cache operations
```

## Best Practices

### Configuration

1. **Use IRIS backend** for production deployments
2. **Configure appropriate TTL** based on content freshness requirements
3. **Enable auto-cleanup** to maintain performance
4. **Use environment variables** for deployment-specific settings

### Key Generation

1. **Include model parameters** in cache keys for accuracy
2. **Normalize whitespace** to improve hit rates
3. **Consider case sensitivity** based on your use case
4. **Use SHA256 hashing** for security and collision resistance

### Monitoring

1. **Track hit rates** to optimize configuration
2. **Monitor cache size** to prevent performance degradation
3. **Set up alerts** for cache failures
4. **Regular cleanup** of expired entries

### Security

1. **Secure IRIS connection** with proper authentication
2. **Use appropriate schema** for cache table isolation
3. **Regular backups** of cache data if needed
4. **Monitor access patterns** for unusual activity

### Performance

1. **Batch cleanup operations** to minimize impact
2. **Use connection pooling** for IRIS connections
3. **Monitor response times** for both hits and misses
4. **Optimize cache key generation** for your use case

## Examples

### Basic Caching Example

```python
from common.llm_cache_manager import setup_langchain_cache
from common.utils import get_llm_func

# Setup caching
setup_langchain_cache()

# Create LLM function
llm_func = get_llm_func("openai", "gpt-3.5-turbo", enable_cache=True)

# First call - cache miss
response1 = llm_func("What is machine learning?")

# Second call - cache hit
response2 = llm_func("What is machine learning?")

assert response1 == response2  # Same response from cache
```

### Custom Configuration Example

```python
from common.llm_cache_config import CacheConfig
from common.llm_cache_manager import setup_langchain_cache

# Custom configuration
config = CacheConfig(
    enabled=True,
    backend="iris",
    ttl_seconds=7200,  # 2 hours
    table_name="custom_llm_cache",
    include_temperature=False,  # Don't include temperature in key
    normalize_whitespace=True
)

# Setup with custom config
cache = setup_langchain_cache(config)
```

### Monitoring Example

```python
import time
from common.llm_cache_manager import get_cache_stats, get_global_cache_manager

# Setup monitoring
manager = get_global_cache_manager()

# Perform some operations
llm_func("Question 1")
llm_func("Question 2")
llm_func("Question 1")  # Cache hit

# Get statistics
stats = get_cache_stats()
print(f"Hit rate: {stats['metrics']['hit_rate']:.2%}")
print(f"Average cached response time: {stats['metrics']['avg_response_time_cached']:.3f}s")
print(f"Average uncached response time: {stats['metrics']['avg_response_time_uncached']:.3f}s")
```

### Cleanup Example

```python
from common.llm_cache_iris import IRISCacheBackend
from common.utils import get_iris_connector

# Manual cleanup
backend = IRISCacheBackend(get_iris_connector(), "llm_cache")
deleted_count = backend.cleanup_expired()
print(f"Cleaned up {deleted_count} expired entries")

# Get cache info
info = backend.get_cache_info()
print(f"Active entries: {info['active_entries']}")
print(f"Expired entries: {info['expired_entries']}")
```

---

For more information, see the [API Reference](API_REFERENCE.md) and [Developer Guide](DEVELOPER_GUIDE.md).