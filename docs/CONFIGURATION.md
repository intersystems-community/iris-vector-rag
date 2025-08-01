# RAG System Configuration Guide

This document provides comprehensive configuration guidance for the RAG templates project, covering all aspects of system configuration from basic setup to advanced reconciliation framework settings.

## Overview

The RAG system uses a hierarchical configuration approach with support for:
- **🚀 Quick Start Configuration**: Template-based configuration with intelligent profiles (NEW!)
- **Multiple Configuration Files**: Main config, pipeline-specific configs, and specialized configurations
- **Environment Variable Overrides**: Runtime configuration overrides with `RAG_` prefix
- **Pipeline-Specific Settings**: Configuration for different RAG techniques (Basic, ColBERT, CRAG, HyDE, GraphRAG, HybridIFind, NodeRAG)
- **Reconciliation Framework**: Automated drift detection and healing capabilities
- **CLI Configuration**: Command-line interface for system management

## Quick Start Configuration System

### 🎯 Profile-Based Configuration

The Quick Start system provides intelligent configuration profiles optimized for different use cases:

| Profile | Documents | Memory | Use Case | Configuration Features |
|---------|-----------|--------|----------|----------------------|
| **Minimal** | 50 | 2GB | Development, Testing | Basic RAG, Local setup, Minimal resources |
| **Standard** | 500 | 4GB | Production, Demos | Multiple techniques, MCP server, Docker integration |
| **Extended** | 5000 | 8GB | Enterprise, Scale | Full stack, Monitoring, Advanced features |

### 🔧 Template Inheritance System

Quick Start uses a hierarchical template system:

```
base_config.yaml           # Core system defaults
    ↓
quick_start.yaml           # Quick Start framework settings
    ↓
quick_start_minimal.yaml   # Minimal profile optimizations
quick_start_standard.yaml  # Standard profile optimizations
quick_start_extended.yaml  # Extended profile optimizations
```

### 🌍 Environment Variable Injection

Templates support dynamic environment variable injection:

```yaml
database:
  iris:
    host: ${IRIS_HOST:-localhost}
    port: ${IRIS_PORT:-1972}
    username: ${IRIS_USERNAME:-demo}
    password: ${IRIS_PASSWORD:-demo}
```

### 📋 Schema Validation

All Quick Start configurations are validated against JSON schemas:
- **Type validation**: Ensures correct data types
- **Range validation**: Validates numeric ranges and constraints
- **Required fields**: Enforces mandatory configuration sections
- **Custom rules**: Profile-specific validation rules

### 🚀 Quick Start Commands

```bash
# Generate configuration for a profile
make quick-start-minimal    # Generates minimal profile config
make quick-start-standard   # Generates standard profile config
make quick-start-extended   # Generates extended profile config

# Interactive configuration wizard
make quick-start           # Interactive setup with profile selection

# Check configuration status
make quick-start-status    # Validate current configuration

# Custom profile configuration
make quick-start-custom PROFILE=my-profile
```

### 📁 Quick Start Configuration Files

Quick Start configurations are stored in:
- **Templates**: [`quick_start/config/templates/`](../quick_start/config/templates/)
- **Schemas**: [`quick_start/config/schemas/`](../quick_start/config/schemas/)
- **Generated configs**: Created in project root during setup

## Configuration Files

### Primary Configuration Files

1. **[`config/config.yaml`](../config/config.yaml)** - Main configuration file with core system settings
2. **[`config/default.yaml`](../config/default.yaml)** - Default configuration values and fallbacks
3. **[`config/pipelines.yaml`](../config/pipelines.yaml)** - Dynamic pipeline definitions and framework dependencies
4. **[`config/colbert_reconciliation_example.yaml`](../config/colbert_reconciliation_example.yaml)** - Complete reconciliation framework example
5. **[`config/basic_rag_example.yaml`](../config/basic_rag_example.yaml)** - Basic RAG pipeline configuration example
6. **[`config/cache_config.yaml`](../config/cache_config.yaml)** - LLM caching configuration
7. **[`config/monitoring.json`](../config/monitoring.json)** - System monitoring and alerting settings

### Configuration Loading Priority

The system loads configurations in the following order (later sources override earlier ones):

1. **Default values** (hardcoded in [`ConfigurationManager`](../iris_rag/config/manager.py))
2. **[`config/default.yaml`](../config/default.yaml)** (if exists)
3. **Main configuration file** (specified via `--config` or default [`config/config.yaml`](../config/config.yaml))
4. **Environment variables** (with `RAG_` prefix)

## Configuration Management Classes

### ConfigurationManager

The [`ConfigurationManager`](../iris_rag/config/manager.py) class provides centralized configuration access:

- **[`get(key_string, default)`](../iris_rag/config/manager.py:113)** - Retrieve configuration values using colon-delimited keys
- **[`get_reconciliation_config()`](../iris_rag/config/manager.py:191)** - Global reconciliation settings
- **[`get_desired_embedding_state(pipeline_type)`](../iris_rag/config/manager.py:234)** - Pipeline-specific desired state
- **[`get_target_state_config(environment)`](../iris_rag/config/manager.py:307)** - Environment-specific target states
- **[`get_embedding_config()`](../iris_rag/config/manager.py:171)** - Embedding model configuration
- **[`get_vector_index_config()`](../iris_rag/config/manager.py:136)** - Vector index settings

### PipelineConfigService

The [`PipelineConfigService`](../iris_rag/config/pipeline_config_service.py) handles dynamic pipeline loading:

- **[`load_pipeline_definitions(config_file_path)`](../iris_rag/config/pipeline_config_service.py:31)** - Load pipeline definitions from YAML
- **[`validate_pipeline_definition(definition)`](../iris_rag/config/pipeline_config_service.py:89)** - Validate pipeline configuration schema

## Core Configuration Sections

### 1. Database Configuration

```yaml
database:
  db_host: "localhost"         # Database host address
  db_port: 1972                # Database port number
  db_user: "SuperUser"         # Database username
  db_password: "SYS"           # Database password
  db_namespace: "USER"         # Database namespace

# Alternative IRIS-specific format (from default.yaml)
database:
  iris:
    driver: "iris._DBAPI"
    host: "localhost"
    port: 1972
    namespace: "USER"
    username: "_SYSTEM"
    password: "SYS"
    connection_timeout: 30
    max_retries: 3
    retry_delay: 1
```

### 2. Embedding Configuration

```yaml
# Main embedding configuration
embedding_model:
  name: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 384

# Extended embedding configuration
embeddings:
  backend: "sentence_transformers"
  model: "sentence-transformers/all-MiniLM-L6-v2"
  dimension: 384
  batch_size: 32
  cache_embeddings: true
```

### 3. Storage Backend Configuration

```yaml
storage:
  backends:
    iris:
      type: "iris"
      connection_type: "dbapi"
      schema: "RAG"
      table_prefix: ""
      vector_dimension: 384

# Storage table configuration
storage:
  document_table: "SourceDocuments"
  chunk_table: "DocumentChunks"
  embedding_table: "DocumentEmbeddings"
  vector_column: "embedding_vector"
```

### 4. Pipeline Configuration

```yaml
# Basic pipeline settings
pipelines:
  basic:
    chunk_size: 1000
    chunk_overlap: 200
    default_top_k: 5
    embedding_batch_size: 32
  colbert:
    chunk_size: 1000
    chunk_overlap: 200
    default_top_k: 5
  crag:
    chunk_size: 1000
    chunk_overlap: 200
    default_top_k: 5

# ColBERT-specific configuration
colbert:
  document_encoder_model: "fjmgAI/reason-colBERT-150M-GTE-ModernColBERT"
  candidate_pool_size: 100
```

### 5. Vector Search Configuration

```yaml
vector_search:
  hnsw:
    ef_construction: 200
    m: 16
    ef_search: 100
  similarity_metric: "cosine"
```

### 6. Logging Configuration

```yaml
logging:
  log_level: "INFO"
  log_format: "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
  file: "logs/iris_rag.log"
  max_file_size: "10MB"
  backup_count: 5
```

### 7. Testing Configuration

```yaml
testing:
  min_docs_e2e: 1000         # Minimum documents required for E2E tests

# RAGAS evaluation configuration
ragas:
  llm:
    model: "gpt-4o-mini"
    temperature: 0
    max_tokens: 2048
  embeddings:
    model: "text-embedding-3-small"
```

## Dynamic Pipeline Configuration

The [`config/pipelines.yaml`](../config/pipelines.yaml) file defines available RAG pipelines:

```yaml
pipelines:
  - name: "BasicRAG"
    module: "iris_rag.pipelines.basic"
    class: "BasicRAGPipeline"
    enabled: true
    params:
      top_k: 5
      chunk_size: 1000
      similarity_threshold: 0.7

  - name: "ColBERTRAG"
    module: "iris_rag.pipelines.colbert"
    class: "ColBERTRAGPipeline"
    enabled: true
    params:
      top_k: 10
      max_query_length: 512
      doc_maxlen: 180

# Framework dependencies (shared across all pipelines)
framework:
  llm:
    model: "gpt-4o-mini"
    temperature: 0
    max_tokens: 1024
  embeddings:
    model: "text-embedding-3-small"
    dimension: 1536
```

## Reconciliation Framework Configuration

### Global Reconciliation Settings

```yaml
reconciliation:
  enabled: true                    # Enable/disable reconciliation framework
  mode: "progressive"              # progressive | complete | emergency
  interval_hours: 24               # Reconciliation execution interval
  
  performance:
    max_concurrent_pipelines: 3    # Maximum pipelines to reconcile simultaneously
    batch_size_documents: 100      # Document processing batch size
    batch_size_embeddings: 50      # Embedding generation batch size
    memory_limit_gb: 8             # Memory limit for reconciliation operations
    cpu_limit_percent: 70          # CPU usage limit percentage
    
  error_handling:
    max_retries: 3                 # Maximum retry attempts for failed operations
    retry_delay_seconds: 30        # Delay between retry attempts
    rollback_on_failure: true      # Rollback changes on failure
    
  monitoring:
    enable_progress_tracking: true # Enable real-time progress tracking
    log_level: "INFO"              # Logging level for reconciliation operations
    alert_on_failures: true        # Send alerts on reconciliation failures
    
  pipeline_overrides:
    colbert:
      batch_size_embeddings: 16
      memory_limit_gb: 12
    graphrag:
      max_retries: 5
```

### Pipeline-Specific Reconciliation Configuration

#### ColBERT Configuration

```yaml
colbert:
  # Basic settings
  target_document_count: 1000
  model_name: "fjmgAI/reason-colBERT-150M-GTE-ModernColBERT"
  token_dimension: 768
  
  # Validation settings
  validation:
    diversity_threshold: 0.7                    # Minimum diversity score (0.0-1.0)
    mock_detection_enabled: true                # Enable detection of mock/dummy embeddings
    min_embedding_quality_score: 0.8            # Minimum quality score (0.0-1.0)
    
  # Completeness requirements
  completeness:
    require_all_docs: true                      # Require embeddings for all documents
    require_token_embeddings: true              # Require token-level embeddings (ColBERT-specific)
    min_completeness_percent: 95.0              # Minimum completeness percentage
    max_missing_documents: 50                   # Maximum allowed missing documents
    
  # Remediation settings
  remediation:
    auto_heal_missing_embeddings: true          # Automatically generate missing embeddings
    auto_migrate_schema: false                  # Automatically migrate schema changes
    embedding_generation_batch_size: 32        # Batch size for embedding generation
    max_remediation_time_minutes: 120          # Maximum time for remediation operations
    backup_before_remediation: true            # Create backup before remediation
```

### Target States for Different Environments

```yaml
target_states:
  development:
    document_count: 1000
    pipelines:
      basic:
        required_embeddings: 
          document_level: 1000
        schema_version: "2.1"
        embedding_model: "all-MiniLM-L6-v2"
        vector_dimensions: 384
      colbert:
        required_embeddings:
          document_level: 1000
          token_level: 1000
        schema_version: "2.1"
        embedding_model: "fjmgAI/reason-colBERT-150M-GTE-ModernColBERT"
        vector_dimensions: 768
        
  production:
    document_count: 50000
    pipelines:
      basic:
        required_embeddings:
          document_level: 50000
        schema_version: "2.1"
        embedding_model: "all-MiniLM-L6-v2"
        vector_dimensions: 384
      colbert:
        required_embeddings:
          document_level: 50000
          token_level: 50000
        schema_version: "2.1"
        embedding_model: "fjmgAI/reason-colBERT-150M-GTE-ModernColBERT"
        vector_dimensions: 768
```

## LLM Caching Configuration

```yaml
llm_cache:
  enabled: true
  backend: "iris"                  # 'memory' or 'iris'
  ttl_seconds: 3600               # Cache TTL (1 hour)
  normalize_prompts: false
  
  iris:
    table_name: "llm_cache"
    schema: "RAG"
    auto_cleanup: true
    cleanup_interval: 86400       # 24 hours
    
  key_generation:
    include_temperature: true
    include_max_tokens: true
    include_model_name: true
    hash_algorithm: "sha256"
    
  monitoring:
    enabled: true
    track_stats: true
    metrics_interval: 300
```

## Monitoring Configuration

```yaml
# From config/monitoring.json
{
  "performance_thresholds": {
    "vector_query_max_ms": 100,
    "ingestion_rate_min_docs_per_sec": 10,
    "memory_usage_max_percent": 85,
    "response_time_p95_max_ms": 500
  },
  "alert_settings": {
    "enable_alerts": true,
    "critical_threshold_breaches": 3,
    "alert_cooldown_minutes": 15
  },
  "health_check_schedule": {
    "interval_minutes": 15,
    "full_check_interval_hours": 6,
    "enable_continuous_monitoring": true
  }
}
```

## Environment Variable Support

All configuration values can be overridden using environment variables with the `RAG_` prefix and double underscores (`__`) for nested keys:

```bash
# Database configuration
export RAG_DATABASE__DB_HOST="production-host"
export RAG_DATABASE__DB_PORT=1972

# Embedding configuration
export RAG_EMBEDDING_MODEL__DIMENSION=768
export RAG_EMBEDDINGS__MODEL="text-embedding-3-large"

# ColBERT configuration
export RAG_COLBERT__TARGET_DOCUMENT_COUNT=2000
export RAG_COLBERT__VALIDATION__DIVERSITY_THRESHOLD=0.8

# Reconciliation configuration
export RAG_RECONCILIATION__PERFORMANCE__MEMORY_LIMIT_GB=16
export RAG_RECONCILIATION__ENABLED=true

# Pipeline configuration
export RAG_PIPELINES__BASIC__DEFAULT_TOP_K=10

# Cache configuration
export LLM_CACHE_ENABLED=true
export LLM_CACHE_BACKEND=iris
export LLM_CACHE_TTL=7200
```

## CLI Configuration and Usage

### Installation & Setup

The CLI is available through multiple entry points:

#### Method 1: Python Module (Recommended)
```bash
python -m iris_rag.cli --help
python -m iris_rag.cli run --pipeline colbert
```

#### Method 2: Standalone Script
```bash
./ragctl --help
./ragctl run --pipeline colbert
```

### Global CLI Options

All commands support these global options:

- `-c, --config PATH`: Path to configuration file
- `--log-level [DEBUG|INFO|WARNING|ERROR]`: Set logging level (default: INFO)

### CLI Commands

#### 1. `run` - Execute Reconciliation

```bash
python -m iris_rag.cli run [OPTIONS]
./ragctl run [OPTIONS]
```

**Options:**
- `-p, --pipeline [basic|colbert|noderag|graphrag|hyde|crag|hybrid_ifind|sql_rag]`: Pipeline type to reconcile (default: colbert)
- `-f, --force`: Force reconciliation even if no drift detected
- `-n, --dry-run`: Analyze drift without executing reconciliation actions

**Examples:**
```bash
# Basic reconciliation
./ragctl run --pipeline colbert

# Force reconciliation regardless of drift
./ragctl run --pipeline basic --force

# Dry-run analysis (no actions executed)
./ragctl run --pipeline noderag --dry-run

# With custom configuration
./ragctl run --config config/production.yaml --pipeline graphrag
```

#### 2. `status` - Display System Status

```bash
python -m iris_rag.cli status [OPTIONS]
./ragctl status [OPTIONS]
```

**Options:**
- `-p, --pipeline [basic|colbert|noderag|graphrag|hyde|crag|hybrid_ifind|sql_rag]`: Pipeline type to check status for (default: colbert)

#### 3. `daemon` - Continuous Reconciliation

```bash
python -m iris_rag.cli daemon [OPTIONS]
./ragctl daemon [OPTIONS]
```

**Options:**
- `-p, --pipeline [basic|colbert|noderag|graphrag|hyde|crag|hybrid_ifind|sql_rag]`: Pipeline type to monitor (default: colbert)
- `-i, --interval INTEGER`: Reconciliation interval in seconds (default: 3600 = 1 hour)
- `--max-iterations INTEGER`: Maximum iterations (0 = infinite, default: 0)

## Configuration Usage Examples

### Basic Configuration Usage

```python
from iris_rag.config.manager import ConfigurationManager
from iris_rag.controllers.reconciliation import ReconciliationController

# Load configuration
config_manager = ConfigurationManager('config/config.yaml')

# Create reconciliation controller
controller = ReconciliationController(config_manager)

# Reconcile ColBERT pipeline
result = controller.reconcile(pipeline_type="colbert")
```

### Advanced Configuration Access

```python
# Get reconciliation settings
reconciliation_config = config_manager.get_reconciliation_config()
print(f"Reconciliation enabled: {reconciliation_config['enabled']}")

# Get ColBERT desired state
colbert_config = config_manager.get_desired_embedding_state("colbert")
print(f"Target documents: {colbert_config['target_document_count']}")

# Get environment-specific target state
target_state = config_manager.get_target_state_config("production")
print(f"Production document count: {target_state['document_count']}")

# Get embedding configuration
embedding_config = config_manager.get_embedding_config()
print(f"Model: {embedding_config['model']}, Dimension: {embedding_config['dimension']}")
```

### Pipeline Setup with Configuration

```python
from iris_rag import setup_pipeline

# Setup pipeline with default configuration
setup_result = setup_pipeline("colbert")

# Setup pipeline with custom configuration
setup_result = setup_pipeline("basic", config_path="config/production.yaml")

# Setup pipeline with external connection
setup_result = setup_pipeline("graphrag", external_connection=my_connection)
```

## Production Usage

### Recommended Daemon Setup

For production environments, run the daemon with appropriate settings:

```bash
# Production daemon with 30-minute intervals
./ragctl daemon \
  --pipeline colbert \
  --interval 1800 \
  --config config/production.yaml \
  --log-level INFO
```

### Monitoring Integration

The CLI exit codes can be integrated with monitoring systems:

```bash
#!/bin/bash
# Health check script
./ragctl status --pipeline colbert
exit_code=$?

case $exit_code in
  0) echo "HEALTHY: No drift detected" ;;
  1) echo "WARNING: Non-critical drift detected" ;;
  2) echo "CRITICAL: Critical issues detected" ;;
  *) echo "ERROR: Command failed" ;;
esac

exit $exit_code
```

### Automation Examples

**Cron job for regular reconciliation:**
```bash
# Run reconciliation every 6 hours
0 */6 * * * /path/to/ragctl run --pipeline colbert --config /path/to/config.yaml
```

**Systemd service for daemon mode:**
```ini
[Unit]
Description=RAG Reconciliation Daemon
After=network.target

[Service]
Type=simple
User=raguser
WorkingDirectory=/path/to/rag-templates
ExecStart=/path/to/ragctl daemon --pipeline colbert --interval 3600
Restart=always
RestartSec=30

[Install]
WantedBy=multi-user.target
```

## Configuration Validation

The configuration is validated when loaded by the [`ConfigurationManager`](../iris_rag/config/manager.py). Invalid configurations will raise a [`ConfigValidationError`](../iris_rag/config/manager.py:6).

## Best Practices

1. **Start with defaults**: Use [`config/default.yaml`](../config/default.yaml) as a foundation and override specific values in your main config
2. **Environment-specific configs**: Use different target states for development vs. production
3. **Environment variables**: Use environment variables for deployment-specific overrides and sensitive data
4. **Gradual rollout**: Start with `mode: "progressive"` for safer reconciliation
5. **Monitor resources**: Adjust `memory_limit_gb` and `cpu_limit_percent` based on system capacity
6. **Regular validation**: Use `./ragctl status` to monitor system health
7. **Backup before changes**: Enable `backup_before_remediation` for safety
8. **Use appropriate cache settings**: Configure LLM caching based on your usage patterns
9. **Monitor performance**: Set appropriate thresholds in monitoring configuration

## Troubleshooting

### Common Configuration Issues

**Configuration file not found:**
```bash
Error initializing configuration: Configuration file not found: /path/to/config.yaml
```
*Solution*: Verify the configuration file path and permissions.

**Database connection errors:**
```bash
Error during reconciliation: Failed to connect to IRIS database
```
*Solution*: Check database connection settings and network connectivity.

**Environment variable format errors:**
```bash
Invalid environment variable format: RAG_INVALID_KEY
```
*Solution*: Ensure environment variables use the correct `RAG_` prefix and `__` delimiters.

### Debug Mode

Enable debug logging for detailed troubleshooting:

```bash
./ragctl run --log-level DEBUG --pipeline colbert
```

## Related Documentation

- [System Architecture](ARCHITECTURE.md)
- [API Reference](API_REFERENCE.md)
- [CLI Usage Guide](CLI_RECONCILIATION_USAGE.md)
- [ColBERT Reconciliation Configuration](COLBERT_RECONCILIATION_CONFIGURATION.md)
- [Comprehensive Generalized Reconciliation Design](design/COMPREHENSIVE_GENERALIZED_RECONCILIATION_DESIGN.md)
- [ConfigurationManager Implementation](../iris_rag/config/manager.py)
- [PipelineConfigService Implementation](../iris_rag/config/pipeline_config_service.py)