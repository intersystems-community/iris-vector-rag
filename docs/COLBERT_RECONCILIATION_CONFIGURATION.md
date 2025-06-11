# ColBERT Reconciliation Configuration Schema

This document describes the configuration schema for the Generalized Reconciliation Framework, specifically for ColBERT token embeddings desired state management.

## Overview

The reconciliation framework uses a declarative configuration approach to define the "desired state" of ColBERT token embeddings and other RAG pipeline components. The [`ReconciliationController`](../iris_rag/controllers/reconciliation.py) reads this configuration to determine what the system should look like and automatically reconciles any drift from the desired state.

## Configuration Files

### Primary Configuration Files

1. **[`config/config.yaml`](../config/config.yaml)** - Main configuration file with commented examples
2. **[`config/colbert_reconciliation_example.yaml`](../config/colbert_reconciliation_example.yaml)** - Complete example showing all available options

### Configuration Manager Integration

The [`ConfigurationManager`](../iris_rag/config/manager.py) provides helper methods for accessing reconciliation configuration:

- [`get_reconciliation_config()`](../iris_rag/config/manager.py:191) - Global reconciliation settings
- [`get_desired_embedding_state(pipeline_type)`](../iris_rag/config/manager.py:220) - Pipeline-specific desired state
- [`get_target_state_config(environment)`](../iris_rag/config/manager.py:290) - Environment-specific target states

## Configuration Schema

### 1. Global Reconciliation Settings

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
```

### 2. ColBERT Desired State Configuration

```yaml
colbert:
  # Basic settings
  target_document_count: 1000                                    # Target number of documents
  model_name: "fjmgAI/reason-colBERT-150M-GTE-ModernColBERT"    # ColBERT model name
  token_dimension: 768                                           # Token embedding dimensions
  
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

### 3. Target States for Different Environments

```yaml
target_states:
  development:
    document_count: 1000
    pipelines:
      colbert:
        required_embeddings:
          document_level: 1000      # Required document-level embeddings
          token_level: 1000         # Required token-level embeddings
        schema_version: "2.1"
        embedding_model: "fjmgAI/reason-colBERT-150M-GTE-ModernColBERT"
        vector_dimensions: 768
        
  production:
    document_count: 50000
    pipelines:
      colbert:
        required_embeddings:
          document_level: 50000
          token_level: 50000
        schema_version: "2.1"
        embedding_model: "fjmgAI/reason-colBERT-150M-GTE-ModernColBERT"
        vector_dimensions: 768
```

## Environment Variable Support

All configuration values can be overridden using environment variables with the `RAG_` prefix:

```bash
# Override ColBERT target document count
export RAG_COLBERT__TARGET_DOCUMENT_COUNT=2000

# Override reconciliation memory limit
export RAG_RECONCILIATION__PERFORMANCE__MEMORY_LIMIT_GB=16

# Override validation diversity threshold
export RAG_COLBERT__VALIDATION__DIVERSITY_THRESHOLD=0.8
```

## Usage Examples

### Basic Usage

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
```

## Configuration Validation

The configuration is validated when loaded by the [`ConfigurationManager`](../iris_rag/config/manager.py). Invalid configurations will raise a [`ConfigValidationError`](../iris_rag/config/manager.py:6).

## Pipeline-Specific Configurations

While this document focuses on ColBERT, the schema supports other pipeline types:

- **BasicRAG**: Uses `vector_dimensions` instead of `token_dimension`, `require_token_embeddings: false`
- **NodeRAG**: Includes chunk hierarchy validation settings
- **GraphRAG**: Includes entity extraction and relationship validation settings

## Best Practices

1. **Start with defaults**: Use the example configuration file as a starting point
2. **Environment-specific configs**: Use different target states for development vs. production
3. **Environment variables**: Use environment variables for deployment-specific overrides
4. **Gradual rollout**: Start with `mode: "progressive"` for safer reconciliation
5. **Monitor resources**: Adjust `memory_limit_gb` and `cpu_limit_percent` based on system capacity

## Related Documentation

- [Comprehensive Generalized Reconciliation Design](../COMPREHENSIVE_GENERALIZED_RECONCILIATION_DESIGN.md)
- [ReconciliationController Implementation](../iris_rag/controllers/reconciliation.py)
- [ConfigurationManager Implementation](../iris_rag/config/manager.py)
- [TDD Plan for Generalized Reconciliation](../tests/TDD_PLAN_GENERALIZED_RECONCILIATION.md)