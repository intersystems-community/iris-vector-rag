# ColBERT Pipeline Modular Design

## Overview

This document defines the modular system design for the ColBERT pipeline resurrection, ensuring each component stays under 500 lines while maintaining clean separation of concerns and testability.

## Design Principles

1. **Single Responsibility**: Each module has one clear purpose
2. **Dependency Injection**: All dependencies are injected, not hard-coded
3. **Interface Segregation**: Small, focused interfaces
4. **Open/Closed**: Open for extension, closed for modification
5. **Testability**: Every module can be unit tested in isolation
6. **Security by Design**: Security is built into every layer

## Module Architecture Overview

```
iris_rag/pipelines/colbert/
├── __init__.py (50 lines)
├── pipeline.py (450 lines) - Main orchestrator
├── encoders/
│   ├── __init__.py (30 lines)
│   ├── query_encoder.py (400 lines)
│   ├── doc_encoder.py (420 lines)
│   └── encoder_factory.py (280 lines)
├── retrieval/
│   ├── __init__.py (25 lines)
│   ├── maxsim_scorer.py (350 lines)
│   ├── hnsw_retriever.py (480 lines)
│   └── fallback_retriever.py (380 lines)
├── security/
│   ├── __init__.py (20 lines)
│   ├── input_validator.py (250 lines)
│   ├── resource_limiter.py (200 lines)
│   └── error_handler.py (180 lines)
└── config/
    ├── __init__.py (15 lines)
    ├── colbert_config.py (280 lines)
    └── schema.py (150 lines)

Total: 12 modules, 4,010 lines (avg 334 lines/module)
```

## Core Module Specifications

### 1. Main Pipeline Module

**File**: [`iris_rag/pipelines/colbert/pipeline.py`](iris_rag/pipelines/colbert/pipeline.py) (450 lines)

```python
"""
ColBERT Pipeline - Main orchestrator implementing RAGPipeline interface.

Responsibilities:
- Implements RAGPipeline abstract interface
- Orchestrates query processing workflow
- Coordinates security validation
- Manages component lifecycle
- Handles error propagation and logging

Dependencies:
- encoders.encoder_factory
- retrieval.hnsw_retriever
- retrieval.fallback_retriever
- retrieval.maxsim_scorer
- security.input_validator
- security.resource_limiter
- security.error_handler
- config.colbert_config
"""

# Line breakdown:
# - Imports and setup: 50 lines
# - Class definition and __init__: 80 lines
# - load_documents method: 100 lines
# - query method (main orchestration): 150 lines
# - Helper methods: 70 lines
```

**Key Methods**:
- `__init__(connection_manager, config_manager, vector_store, llm_func)` (40 lines)
- `load_documents(documents_path, **kwargs)` (100 lines)
- `query(query_text, top_k, **kwargs)` (150 lines)
- `_orchestrate_retrieval(query_embeddings)` (60 lines)
- `_generate_answer(query, documents)` (50 lines)

### 2. Security Layer Modules

#### Input Validator Module

**File**: [`iris_rag/pipelines/colbert/security/input_validator.py`](iris_rag/pipelines/colbert/security/input_validator.py) (250 lines)

```python
"""
Input validation and sanitization for ColBERT pipeline.

Responsibilities:
- Query text sanitization
- Parameter validation and bounding
- SQL injection prevention
- XSS protection
- Input length and complexity limits

Security Features:
- Character whitelist validation
- SQL injection pattern detection
- Parameter boundary enforcement
- Malicious payload detection
"""

# Line breakdown:
# - Imports and constants: 30 lines
# - ValidationResult dataclass: 20 lines
# - InputValidator class definition: 40 lines
# - validate_query method: 60 lines
# - validate_parameters method: 50 lines
# - validate_document_input method: 40 lines
# - Helper validation functions: 10 lines
```

**Key Components**:
- `ValidationResult` dataclass (20 lines)
- `InputValidator` class (200 lines)
- Security pattern constants (30 lines)

#### Resource Limiter Module  

**File**: [`iris_rag/pipelines/colbert/security/resource_limiter.py`](iris_rag/pipelines/colbert/security/resource_limiter.py) (200 lines)

```python
"""
Resource usage control and DoS prevention.

Responsibilities:
- Memory usage monitoring
- Token count enforcement
- Batch size control
- Timeout management
- Circuit breaker pattern

Features:
- Real-time memory tracking
- Progressive degradation
- Configurable limits
- Performance metrics
"""

# Line breakdown:
# - Imports and setup: 25 lines
# - ResourceLimiter class: 40 lines
# - Memory monitoring: 50 lines
# - Token limiting: 40 lines
# - Batch processing: 35 lines
# - Helper utilities: 10 lines
```

#### Error Handler Module

**File**: [`iris_rag/pipelines/colbert/security/error_handler.py`](iris_rag/pipelines/colbert/security/error_handler.py) (180 lines)

```python
"""
Secure error handling without information leakage.

Responsibilities:
- Exception sanitization
- Safe error logging
- Security event tracking
- Debug information masking
- Error response standardization

Security Features:
- Information leakage prevention
- Stack trace sanitization
- Sensitive data masking
- Security audit logging
"""

# Line breakdown:
# - Imports and exceptions: 30 lines
# - ErrorResponse dataclass: 25 lines
# - ErrorHandler class: 45 lines
# - Error sanitization: 40 lines
# - Security logging: 30 lines
# - Helper functions: 10 lines
```

### 3. Encoder Layer Modules

#### Query Encoder Module

**File**: [`iris_rag/pipelines/colbert/encoders/query_encoder.py`](iris_rag/pipelines/colbert/encoders/query_encoder.py) (400 lines)

```python
"""
Query tokenization and embedding generation for ColBERT.

Responsibilities:
- Query text tokenization
- Token-level embedding generation
- Embedding normalization and validation
- Model loading and management
- Memory-efficient processing

Features:
- Support for multiple ColBERT models
- Graceful fallback to mock implementation
- Memory usage monitoring
- Batch processing optimization
"""

# Line breakdown:
# - Imports and constants: 40 lines
# - QueryEncoder class definition: 60 lines
# - Model initialization: 80 lines
# - encode method (main functionality): 120 lines
# - tokenize method: 40 lines
# - Validation and normalization: 50 lines
# - Helper utilities: 10 lines
```

**Key Features**:
- Real ColBERT model support with fallback
- Token-level embedding generation
- Memory usage monitoring
- Input validation and sanitization
- Configurable model parameters

#### Document Encoder Module

**File**: [`iris_rag/pipelines/colbert/encoders/doc_encoder.py`](iris_rag/pipelines/colbert/encoders/doc_encoder.py) (420 lines)

```python
"""
Document tokenization and embedding generation for ColBERT indexing.

Responsibilities:
- Document text preprocessing
- Token-level embedding generation
- Batch processing for efficiency
- Embedding storage preparation
- Memory management

Features:
- Efficient batch processing
- Chunking for large documents
- Memory usage optimization
- Progress tracking
- Error recovery
"""

# Line breakdown:
# - Imports and setup: 40 lines
# - DocEncoder class definition: 60 lines
# - Model initialization: 80 lines
# - encode_document method: 100 lines
# - encode_batch method: 80 lines
# - Document preprocessing: 50 lines
# - Helper utilities: 10 lines
```

#### Encoder Factory Module

**File**: [`iris_rag/pipelines/colbert/encoders/encoder_factory.py`](iris_rag/pipelines/colbert/encoders/encoder_factory.py) (280 lines)

```python
"""
Factory for creating and configuring encoder instances.

Responsibilities:
- Encoder instance creation
- Configuration validation
- Dependency injection
- Model loading coordination
- Error handling for missing dependencies

Features:
- Lazy loading of heavy models
- Configuration-driven setup
- Graceful degradation
- Mock mode for testing
"""

# Line breakdown:
# - Imports and constants: 30 lines
# - EncoderFactory class: 50 lines
# - create_query_encoder method: 70 lines
# - create_doc_encoder method: 70 lines
# - Configuration validation: 50 lines
# - Helper utilities: 10 lines
```

### 4. Retrieval Layer Modules

#### MaxSim Scorer Module

**File**: [`iris_rag/pipelines/colbert/retrieval/maxsim_scorer.py`](iris_rag/pipelines/colbert/retrieval/maxsim_scorer.py) (350 lines)

```python
"""
MaxSim scoring implementation for ColBERT token-level retrieval.

Responsibilities:
- MaxSim score computation
- Vectorized similarity calculations
- Memory-efficient batch processing
- GPU acceleration support
- Performance optimization

Algorithm:
MaxSim(q, d) = (1/|q|) * Σ(i=1 to |q|) max(j=1 to |d|) cosine_sim(q_i, d_j)

Features:
- NumPy vectorized operations
- Memory usage monitoring
- Batch processing for multiple documents
- GPU support when available
"""

# Line breakdown:
# - Imports and setup: 30 lines
# - MaxSimScorer class: 50 lines
# - compute_maxsim method: 100 lines
# - compute_batch_maxsim method: 80 lines
# - Vectorized operations: 70 lines
# - Helper utilities: 20 lines
```

#### HNSW Retriever Module

**File**: [`iris_rag/pipelines/colbert/retrieval/hnsw_retriever.py`](iris_rag/pipelines/colbert/retrieval/hnsw_retriever.py) (480 lines)

```python
"""
HNSW-accelerated retrieval strategy for ColBERT.

Responsibilities:
- HNSW index interaction
- Candidate document retrieval
- Token-level similarity search
- Index health monitoring
- Performance optimization

Features:
- HNSW index utilization
- Parameterized query construction
- Candidate pool management
- Error handling and fallback
- Performance metrics collection
"""

# Line breakdown:
# - Imports and setup: 40 lines
# - HNSWRetriever class: 60 lines
# - retrieve_candidates method: 150 lines
# - HNSW query construction: 80 lines
# - Result processing: 70 lines
# - Index health checking: 60 lines
# - Helper utilities: 20 lines
```

#### Fallback Retriever Module

**File**: [`iris_rag/pipelines/colbert/retrieval/fallback_retriever.py`](iris_rag/pipelines/colbert/retrieval/fallback_retriever.py) (380 lines)

```python
"""
Fallback retrieval strategy using batch processing.

Responsibilities:
- Batch loading of document embeddings
- In-memory similarity computation
- Memory management and monitoring
- Timeout handling
- Progressive degradation

Features:
- Memory-efficient batch processing
- Timeout controls
- Progress monitoring
- Error recovery
- Performance metrics
"""

# Line breakdown:
# - Imports and setup: 35 lines
# - FallbackRetriever class: 55 lines
# - retrieve_candidates method: 120 lines
# - Batch loading: 80 lines
# - Similarity computation: 70 lines
# - Helper utilities: 20 lines
```

### 5. Configuration Layer Modules

#### ColBERT Config Module

**File**: [`iris_rag/pipelines/colbert/config/colbert_config.py`](iris_rag/pipelines/colbert/config/colbert_config.py) (280 lines)

```python
"""
ColBERT-specific configuration management.

Responsibilities:
- Configuration parameter definition
- Validation and type checking
- Default value management
- Environment variable integration
- Configuration serialization

Features:
- Dataclass-based configuration
- Comprehensive validation
- Environment variable override
- JSON serialization support
- Migration support
"""

# Line breakdown:
# - Imports and constants: 30 lines
# - ColBERTConfig dataclass: 100 lines
# - Validation methods: 80 lines
# - Serialization methods: 40 lines
# - Factory methods: 20 lines
# - Helper utilities: 10 lines
```

#### Schema Validation Module

**File**: [`iris_rag/pipelines/colbert/config/schema.py`](iris_rag/pipelines/colbert/config/schema.py) (150 lines)

```python
"""
Configuration schema validation using JSON Schema.

Responsibilities:
- JSON Schema definition
- Configuration validation
- Error message generation
- Type coercion
- Migration assistance

Features:
- Comprehensive schema definition
- Detailed validation errors
- Type conversion
- Default value injection
- Schema versioning support
"""

# Line breakdown:
# - Imports and schema definition: 60 lines
# - ConfigSchema class: 40 lines
# - Validation logic: 40 lines
# - Helper utilities: 10 lines
```

## Module Dependencies

### Dependency Graph

```
pipeline.py
├── encoders/encoder_factory.py
│   ├── encoders/query_encoder.py
│   └── encoders/doc_encoder.py
├── retrieval/hnsw_retriever.py
│   └── retrieval/maxsim_scorer.py
├── retrieval/fallback_retriever.py
│   └── retrieval/maxsim_scorer.py
├── security/input_validator.py
├── security/resource_limiter.py
├── security/error_handler.py
└── config/colbert_config.py
    └── config/schema.py
```

### Dependency Injection Pattern

```python
# Main pipeline receives all dependencies via constructor
class ColBERTPipeline:
    def __init__(self, connection_manager, config_manager, vector_store, llm_func):
        # Load configuration
        self.config = ColBERTConfig.from_config_manager(config_manager)
        
        # Initialize security layer
        self.input_validator = InputValidator(self.config.security_settings)
        self.resource_limiter = ResourceLimiter(self.config.resource_limits)
        self.error_handler = ErrorHandler(self.config.debug_mode)
        
        # Initialize encoders
        self.encoder_factory = EncoderFactory(self.config.encoder_settings)
        self.query_encoder = self.encoder_factory.create_query_encoder()
        self.doc_encoder = self.encoder_factory.create_doc_encoder()
        
        # Initialize retrieval
        self.maxsim_scorer = MaxSimScorer(self.config.use_gpu)
        self.hnsw_retriever = HNSWRetriever(connection_manager, self.config.hnsw_settings)
        self.fallback_retriever = FallbackRetriever(connection_manager, self.config.fallback_settings)
```

## Interface Compliance

### RAGPipeline Interface Compliance

Each module strictly adheres to the interface contracts:

```python
# All modules implement defined interfaces
class ColBERTPipeline(RAGPipeline):  # ✅ Implements abstract base class
class InputValidator(SecurityInterface):  # ✅ Security contract
class QueryEncoder(EncoderInterface):  # ✅ Encoder contract
class HNSWRetriever(RetrieverInterface):  # ✅ Retrieval contract
```

### Error Handling Compliance

```python
# Consistent error handling across all modules
try:
    result = component.execute(input)
except SecurityError as e:
    return self.error_handler.sanitize_error(e)
except ResourceError as e:
    return self.error_handler.sanitize_error(e)
except Exception as e:
    return self.error_handler.sanitize_error(e)
```

## Testing Strategy

### Unit Test Structure

Each module includes comprehensive unit tests:

```
tests/pipelines/colbert/
├── test_pipeline.py (200 lines)
├── encoders/
│   ├── test_query_encoder.py (150 lines)
│   ├── test_doc_encoder.py (150 lines)
│   └── test_encoder_factory.py (100 lines)
├── retrieval/
│   ├── test_maxsim_scorer.py (120 lines)
│   ├── test_hnsw_retriever.py (180 lines)
│   └── test_fallback_retriever.py (150 lines)
├── security/
│   ├── test_input_validator.py (100 lines)
│   ├── test_resource_limiter.py (80 lines)
│   └── test_error_handler.py (70 lines)
└── config/
    ├── test_colbert_config.py (80 lines)
    └── test_schema.py (60 lines)
```

### Test Coverage Targets

- **Unit Tests**: 100% line coverage per module
- **Integration Tests**: End-to-end pipeline functionality
- **Security Tests**: Vulnerability scanning and penetration testing
- **Performance Tests**: Response time and memory usage validation

## Development Guidelines

### Code Quality Standards

1. **Line Limits**: Strict 500-line maximum per module
2. **Complexity**: Maximum cyclomatic complexity of 10 per function
3. **Documentation**: Comprehensive docstrings and type hints
4. **Linting**: Pass flake8, mypy, and black formatting
5. **Security**: Pass security scanning with bandit

### Module Development Order

1. **Phase 1**: Configuration and security modules (foundation)
2. **Phase 2**: Encoder modules (core functionality)  
3. **Phase 3**: Retrieval modules (search engine)
4. **Phase 4**: Main pipeline (orchestration)
5. **Phase 5**: Integration and testing

### Performance Targets

| Module | Target Response Time | Memory Limit |
|--------|---------------------|--------------|
| InputValidator | <1ms | <1MB |
| QueryEncoder | <100ms | <50MB |
| DocEncoder | <500ms | <100MB |
| MaxSimScorer | <50ms | <25MB |
| HNSWRetriever | <200ms | <75MB |
| FallbackRetriever | <5s | <200MB |
| Overall Pipeline | 0.70s-34.36s | <1GB |

## Security Considerations

### Module-Level Security

Each module implements security controls:

```python
# Example: Query encoder with security integration
class QueryEncoder:
    def encode(self, query_text: str) -> List[List[float]]:
        # 1. Input validation
        validation_result = self.input_validator.validate_query(query_text)
        if not validation_result.is_valid:
            raise SecurityError("Invalid query input")
            
        # 2. Resource checking
        if not self.resource_limiter.check_memory_usage():
            raise ResourceError("Memory limit exceeded")
            
        # 3. Safe processing
        try:
            embeddings = self._encode_tokens(validation_result.sanitized_input)
            return embeddings
        except Exception as e:
            # 4. Secure error handling
            raise self.error_handler.sanitize_error(e)
```

### Security Validation Checklist

- ✅ Input validation in every public method
- ✅ Resource limits enforced
- ✅ No SQL injection vulnerabilities
- ✅ No information leakage in errors
- ✅ Secure logging without sensitive data
- ✅ Memory cleanup after operations

## Deployment Considerations

### Module Loading Order

1. Configuration modules load first
2. Security modules initialize
3. Encoder modules prepare
4. Retrieval modules connect to database
5. Main pipeline orchestrates

### Graceful Degradation

Each module supports graceful degradation:

```python
# Example: Encoder factory with fallback
class EncoderFactory:
    def create_query_encoder(self):
        try:
            return RealQueryEncoder(self.config)
        except ImportError:
            logger.warning("Falling back to mock encoder")
            return MockQueryEncoder(self.config)
        except Exception as e:
            logger.error(f"Encoder creation failed: {e}")
            return MockQueryEncoder(self.config)
```

## Monitoring and Observability

### Performance Monitoring

Each module exports metrics:

```python
# Example: Built-in performance monitoring
class MaxSimScorer:
    def compute_maxsim(self, query_embeddings, doc_embeddings):
        with self.performance_monitor.timer("maxsim_computation"):
            with self.resource_limiter.memory_monitor():
                result = self._compute_similarity_matrix(query_embeddings, doc_embeddings)
                self.performance_monitor.record_metric("maxsim_score", result)
                return result
```

### Health Checks

Each module provides health check endpoints:

```python
# Module health check interface
class ModuleHealthCheck:
    def check_health(self) -> Dict[str, Any]:
        return {
            "status": "healthy",
            "memory_usage": self._get_memory_usage(),
            "performance_metrics": self._get_performance_summary(),
            "error_rate": self._get_error_rate()
        }
```

## Summary

This modular design provides:

1. **Strict Line Limits**: All modules under 500 lines
2. **Clean Separation**: Single responsibility per module
3. **Security Integration**: Security built into every layer
4. **Testability**: Comprehensive unit testing strategy
5. **Performance**: Optimized for 0.70s-34.36s response times
6. **Maintainability**: Clear interfaces and documentation
7. **Scalability**: Modular design supports future enhancements

The design ensures the ColBERT pipeline can be implemented as a collection of focused, secure, and maintainable modules that integrate seamlessly with the existing RAG evaluation system.