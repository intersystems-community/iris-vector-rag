# LLM Caching Layer Specification

## 1. Overview

This specification defines a lightweight LLM caching layer that leverages existing Langchain capabilities to reduce OpenAI API calls while maintaining simplicity and minimal dependencies. The implementation builds upon the existing [`get_llm_func`](common/utils.py:212) pattern which already uses Langchain's `ChatOpenAI`, enabling seamless integration with Langchain's built-in caching mechanisms.

## 2. Objectives

- **Primary Goal**: Reduce OpenAI API costs by caching LLM responses using Langchain's native caching
- **Secondary Goals**: 
  - Improve response times for repeated queries
  - Maintain existing [`get_llm_func`](common/utils.py:212) patterns and interfaces
  - Ensure minimal code changes to existing pipelines
  - Provide configurable caching strategies
  - Replace custom cache implementation with Langchain standard

## 3. Scope

### 3.1 In Scope
- Integration with Langchain's built-in caching mechanisms (`langchain.llm_cache`)
- Configuration-driven cache backend selection
- TTL-based cache expiration via Langchain
- Cache hit/miss metrics for monitoring
- Integration with existing RAG pipeline patterns
- Test coverage for caching functionality
- Migration from existing [`iris_rag/llm/cache.py`](iris_rag/llm/cache.py:1) to Langchain caching
- Enhancement of existing [`get_llm_func`](common/utils.py:212) to support caching

### 3.2 Out of Scope
- Custom cache implementations beyond Langchain capabilities
- Complex cache invalidation strategies beyond TTL
- Distributed caching across multiple instances
- Cache warming strategies
- Advanced cache analytics beyond basic hit/miss metrics
- Maintaining the existing custom cache implementation

## 4. Technical Requirements

### 4.1 Langchain Cache Backend Options

The implementation shall support the following Langchain cache backends in order of preference:

1. **langchain.cache.InMemoryCache** (Default)
   - Zero external dependencies
   - Suitable for development and single-instance deployments
   - Automatic cleanup on process restart

2. **langchain.cache.SQLiteCache** (Recommended for production)
   - Persistent caching across restarts
   - No additional service dependencies
   - File-based storage

3. **langchain.cache.RedisCache** (Optional)
   - Only if Redis is already available in the environment
   - Shared cache across multiple instances
   - Advanced TTL capabilities

### 4.2 Cache Key Generation

- **Default Strategy**: Use Langchain's default cache key generation based on:
  - LLM model name and parameters
  - Input prompt/messages
  - Temperature and other generation parameters
- **Custom Strategy**: Optional prompt normalization for better cache hits:
  - Whitespace normalization
  - Case normalization (configurable)

### 4.3 Cache Configuration

Configuration shall be managed through environment variables and config files:

```yaml
llm_cache:
  enabled: true
  backend: "sqlite"  # "memory", "sqlite", "redis"
  ttl_seconds: 3600  # 1 hour default
  sqlite_path: "./cache/llm_cache.db"
  redis_url: "redis://localhost:6379/0"  # if redis backend
  normalize_prompts: false
  max_cache_size: 1000  # for memory backend
```

Environment Variables:
```bash
LLM_CACHE_ENABLED=true
LLM_CACHE_BACKEND=sqlite
LLM_CACHE_TTL=3600
LLM_CACHE_SQLITE_PATH=./cache/llm_cache.db
LLM_CACHE_REDIS_URL=redis://localhost:6379/0
LLM_CACHE_NORMALIZE_PROMPTS=false
LLM_CACHE_MAX_SIZE=1000
```

### 4.4 Integration Points

#### 4.4.1 Current Architecture Analysis
The existing [`get_llm_func`](common/utils.py:212) already creates Langchain `ChatOpenAI` instances:
```python
# Current implementation in common/utils.py:255
_llm_instance = ChatOpenAI(model_name=model_name, openai_api_key=api_key, **kwargs)
```

This means caching can be enabled by simply configuring `langchain.llm_cache` before LLM instantiation.

#### 4.4.2 Enhanced LLM Function
```python
def get_llm_func(provider: str = "openai", model_name: str = "gpt-3.5-turbo", 
                enable_cache: bool = None, **kwargs) -> Callable[[str], str]:
    """
    Enhanced version of existing get_llm_func with caching support.
    
    Args:
        provider: LLM provider (openai, stub, etc.)
        model_name: Model name
        enable_cache: Override cache enable/disable (None = use config default)
        **kwargs: Additional LLM parameters
    
    Returns:
        LLM function (automatically cached if enabled)
    """
```

#### 4.4.3 Langchain Global Cache Setup
```python
def setup_langchain_cache(cache_config):
    """
    Configure Langchain's global cache based on configuration.
    Must be called before any LLM instantiation.
    
    Args:
        cache_config: Cache configuration dictionary
    """
    import langchain
    from langchain.cache import InMemoryCache, SQLiteCache
    
    if cache_config.backend == "memory":
        langchain.llm_cache = InMemoryCache()
    elif cache_config.backend == "sqlite":
        langchain.llm_cache = SQLiteCache(database_path=cache_config.sqlite_path)
    elif cache_config.backend == "redis":
        try:
            from langchain.cache import RedisCache
            import redis
            langchain.llm_cache = RedisCache(redis_=redis.from_url(cache_config.redis_url))
        except ImportError:
            # Fallback to SQLite if Redis not available
            langchain.llm_cache = SQLiteCache(database_path="./cache/fallback.db")
```

#### 4.4.4 Pipeline Integration
- Modify existing pipeline constructors to accept `enable_cache` parameter
- Automatic cache integration when LLM objects are created via [`get_llm_func`](common/utils.py:212)
- Backward compatibility with existing pipeline code
- Cache setup during application initialization

## 5. Implementation Architecture

### 5.1 Module Structure

```
rag_templates/llm/
├── __init__.py
├── cache_manager.py      # Main cache configuration and setup
├── langchain_cache.py    # Langchain cache integration
└── metrics.py           # Cache metrics and monitoring
```

### 5.2 Core Components

#### 5.2.1 Cache Manager
```python
class LangchainCacheManager:
    """Manages Langchain cache configuration and lifecycle."""
    
    def __init__(self, config: CacheConfig):
        self.config = config
        self.metrics = CacheMetrics()
    
    def setup_cache(self) -> None:
        """Configure Langchain's global cache."""
        
    def get_cache_stats(self) -> Dict[str, Any]:
        """Get cache performance statistics."""
        
    def clear_cache(self) -> None:
        """Clear all cached entries."""
```

#### 5.2.2 Configuration Model
```python
@dataclass
class CacheConfig:
    """Cache configuration model."""
    enabled: bool = True
    backend: str = "sqlite"  # memory, sqlite, redis
    ttl_seconds: int = 3600
    sqlite_path: str = "./cache/llm_cache.db"
    redis_url: str = "redis://localhost:6379/0"
    normalize_prompts: bool = False
    max_cache_size: int = 1000
    
    @classmethod
    def from_env(cls) -> 'CacheConfig':
        """Create configuration from environment variables."""
```

#### 5.2.3 LLM Function Enhancement
```python
def get_cached_llm_func(provider: str = "openai", 
                       model_name: str = "gpt-3.5-turbo",
                       enable_cache: bool = None,
                       **kwargs) -> Callable[[str], str]:
    """
    Enhanced version of get_llm_func with caching support.
    
    Args:
        provider: LLM provider (openai, stub, etc.)
        model_name: Model name
        enable_cache: Override cache enable/disable
        **kwargs: Additional LLM parameters
    
    Returns:
        Cached LLM function
    """
```

## 6. Pseudocode Implementation

### 6.1 Cache Setup Pseudocode

```python
def setup_langchain_cache():
    """
    PSEUDOCODE: Setup Langchain cache based on configuration
    """
    # Load configuration from environment/config file
    config = CacheConfig.from_env()
    
    if not config.enabled:
        return None
    
    # Import langchain cache modules
    import langchain
    from langchain.cache import InMemoryCache, SQLiteCache, RedisCache
    
    # Configure cache backend
    if config.backend == "memory":
        cache = InMemoryCache()
    elif config.backend == "sqlite":
        # Ensure cache directory exists
        cache_dir = Path(config.sqlite_path).parent
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache = SQLiteCache(database_path=config.sqlite_path)
    elif config.backend == "redis":
        try:
            import redis
            redis_client = redis.from_url(config.redis_url)
            cache = RedisCache(redis_=redis_client)
        except ImportError:
            log_warning("Redis not available, falling back to SQLite")
            cache = SQLiteCache(database_path="./cache/fallback.db")
    
    # Set global cache
    langchain.llm_cache = cache
    
    # Configure TTL if supported
    if hasattr(cache, 'ttl') and config.ttl_seconds:
        cache.ttl = config.ttl_seconds
    
    log_info(f"Langchain cache configured: {config.backend}")
    return cache
```

### 6.2 Enhanced LLM Function Pseudocode

```python
def enhanced_get_llm_func(provider="openai", model_name="gpt-3.5-turbo", 
                         enable_cache=None, **kwargs):
    """
    PSEUDOCODE: Enhanced version of existing get_llm_func with caching
    """
    # Load cache configuration
    cache_config = load_cache_config()
    
    # Determine if caching should be enabled
    cache_enabled = enable_cache if enable_cache is not None else cache_config.enabled
    
    # Setup Langchain cache if enabled and not already configured
    if cache_enabled and not is_langchain_cache_configured():
        setup_langchain_cache(cache_config)
    
    # Use existing get_llm_func logic with minimal changes
    # The existing ChatOpenAI instances will automatically use the cache
    global _llm_instance, _current_llm_key
    
    llm_key = f"{provider}_{model_name}_{cache_enabled}"
    
    if _llm_instance is None or _current_llm_key != llm_key:
        if provider == "openai":
            # Existing logic from common/utils.py:247-256
            from langchain_openai import ChatOpenAI
            api_key = get_openai_api_key()
            _llm_instance = ChatOpenAI(model_name=model_name, openai_api_key=api_key, **kwargs)
            _current_llm_key = llm_key
        elif provider == "stub":
            # Existing stub logic from common/utils.py:258-269
            _llm_instance = create_stub_llm(model_name, **kwargs)
            _current_llm_key = llm_key
    
    # Return existing query_llm function
    # Cache hits/misses are handled transparently by Langchain
    def query_llm(prompt: str) -> str:
        response = _llm_instance.invoke(prompt)
        if hasattr(response, 'content'):
            return str(response.content)
        return str(response)
    
    return query_llm
```

### 6.3 Pipeline Integration Pseudocode

```python
def create_rag_pipeline_with_cache(pipeline_type, enable_cache=True, **kwargs):
    """
    PSEUDOCODE: Create RAG pipeline with optional caching
    """
    # Initialize cache if enabled (global setup)
    if enable_cache:
        cache_config = load_cache_config()
        setup_langchain_cache(cache_config)
    
    # Get configuration
    config = load_pipeline_config(pipeline_type)
    
    # Use enhanced get_llm_func (existing interface, automatic caching)
    llm_func = get_llm_func(
        provider=config.llm_provider,
        model_name=config.llm_model,
        temperature=config.temperature,
        enable_cache=enable_cache  # New parameter
    )
    
    # Create pipeline with LLM function (no changes to existing pipeline code)
    pipeline = create_pipeline(
        pipeline_type=pipeline_type,
        llm_func=llm_func,
        **kwargs
    )
    
    return pipeline
```

### 6.4 Application Initialization Pseudocode

```python
def initialize_application_with_cache():
    """
    PSEUDOCODE: Application startup with cache initialization
    """
    # Load configuration early in application lifecycle
    cache_config = load_cache_config()
    
    # Setup Langchain cache globally before any LLM usage
    if cache_config.enabled:
        setup_langchain_cache(cache_config)
        log_info(f"LLM caching enabled with {cache_config.backend} backend")
    else:
        log_info("LLM caching disabled")
    
    # All subsequent get_llm_func() calls will automatically use cache
    # No changes needed to existing pipeline code
```

## 7. Migration Strategy

### 7.1 Phase 1: Langchain Cache Integration
1. Create new [`rag_templates/llm/`](rag_templates/llm/) module
2. Implement [`CacheConfig`](rag_templates/llm/cache_manager.py:1) and [`LangchainCacheManager`](rag_templates/llm/cache_manager.py:1)
3. Add cache setup utilities for Langchain global cache
4. Add configuration support to [`rag_templates/config/manager.py`](rag_templates/config/manager.py:1)

### 7.2 Phase 2: Integration with Existing Patterns
1. Enhance [`common/utils.py`](common/utils.py:212) `get_llm_func` to support `enable_cache` parameter
2. Add cache initialization to application startup
3. Update pipeline constructors to accept cache parameters
4. Add cache configuration to default configs

### 7.3 Phase 3: Testing and Validation
1. Create comprehensive test suite for cache functionality
2. Performance testing with 1000+ document datasets
3. Cache hit rate validation
4. Integration testing with existing RAG pipelines

### 7.4 Phase 4: Deprecation of Custom Cache
1. Mark [`iris_rag/llm/cache.py`](iris_rag/llm/cache.py:1) as deprecated
2. Provide migration guide for any direct usage
3. Remove custom cache in future release

## 8. Testing Requirements

### 8.1 Unit Tests
```python
def test_cache_configuration():
    """Test cache configuration from environment variables."""
    
def test_langchain_cache_setup():
    """Test Langchain cache backend setup."""
    
def test_cached_llm_function():
    """Test cached LLM function behavior with existing get_llm_func."""
    
def test_cache_hit_miss_metrics():
    """Test cache statistics tracking via Langchain."""
    
def test_langchain_cache_integration():
    """Test integration with existing ChatOpenAI instances."""
```

### 8.2 Integration Tests
```python
def test_rag_pipeline_with_cache():
    """Test RAG pipeline with caching enabled using existing patterns."""
    
def test_cache_persistence():
    """Test cache persistence across restarts (SQLite)."""
    
def test_cache_performance():
    """Test cache performance with 1000+ documents."""
    
def test_existing_pipeline_compatibility():
    """Test that existing pipelines work unchanged with caching enabled."""
```

### 8.3 Performance Tests
- Cache hit rate measurement
- Response time comparison (cached vs uncached)
- Memory usage with different cache backends
- Concurrent access testing

## 9. Monitoring and Metrics

### 9.1 Cache Metrics
```python
@dataclass
class CacheMetrics:
    """Cache performance metrics."""
    hits: int = 0
    misses: int = 0
    total_requests: int = 0
    hit_rate: float = 0.0
    avg_response_time_cached: float = 0.0
    avg_response_time_uncached: float = 0.0
```

### 9.2 Logging
- Cache setup and configuration
- Cache hit/miss events (debug level)
- Cache performance statistics (info level)
- Cache errors and fallbacks (warning level)

## 10. Configuration Examples

### 10.1 Development Configuration
```yaml
llm_cache:
  enabled: true
  backend: "memory"
  ttl_seconds: 1800  # 30 minutes
  max_cache_size: 500
```

### 10.2 Production Configuration
```yaml
llm_cache:
  enabled: true
  backend: "sqlite"
  ttl_seconds: 7200  # 2 hours
  sqlite_path: "/app/cache/llm_cache.db"
```

### 10.3 Redis Configuration
```yaml
llm_cache:
  enabled: true
  backend: "redis"
  ttl_seconds: 3600
  redis_url: "redis://redis-server:6379/1"
```

## 11. Error Handling

### 11.1 Cache Backend Failures
- Graceful fallback to no caching if backend fails
- Automatic retry with exponential backoff
- Fallback to alternative backends (Redis → SQLite → Memory)

### 11.2 Configuration Errors
- Validation of cache configuration on startup
- Clear error messages for misconfiguration
- Safe defaults for missing configuration

## 12. Performance Considerations

### 12.1 Cache Key Optimization
- Efficient hashing of prompts and parameters
- Optional prompt normalization for better hit rates
- Configurable key generation strategies

### 12.2 Memory Management
- LRU eviction for memory cache
- Configurable cache size limits
- Automatic cleanup of expired entries

### 12.3 Concurrency
- Thread-safe cache operations
- Minimal locking overhead
- Async support where applicable

## 13. Security Considerations

### 13.1 Cache Content
- No sensitive data in cache keys
- Optional encryption for cached responses
- Secure handling of API keys

### 13.2 Access Control
- Cache isolation between different applications
- Secure Redis configuration if used
- File permissions for SQLite cache

## 14. Success Criteria

### 14.1 Functional Requirements
- ✅ Successful integration with Langchain caching
- ✅ Backward compatibility with existing [`get_llm_func`](common/utils.py:212) pattern
- ✅ Configuration-driven cache backend selection
- ✅ Cache hit rate > 30% in typical usage scenarios

### 14.2 Performance Requirements
- ✅ < 10ms overhead for cache hits
- ✅ < 50ms overhead for cache misses
- ✅ Memory usage < 100MB for 1000 cached responses
- ✅ 90%+ reduction in API calls for repeated queries

### 14.3 Quality Requirements
- ✅ 100% test coverage for cache functionality
- ✅ Zero breaking changes to existing pipeline code
- ✅ Comprehensive documentation and examples
- ✅ Production-ready error handling and monitoring