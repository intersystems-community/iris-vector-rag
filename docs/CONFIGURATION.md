# RAG System Configuration Guide

This document provides comprehensive configuration guidance for the RAG system, including CLI usage, reconciliation framework settings, and pipeline-specific configurations.

## Overview

The RAG system uses a declarative configuration approach with support for:
- **CLI Configuration**: Command-line interface for reconciliation operations
- **Reconciliation Framework**: Automated drift detection and healing
- **Pipeline-Specific Settings**: Configuration for different RAG techniques
- **Environment Variables**: Runtime configuration overrides

## Configuration Files

### Primary Configuration Files

1. **[`config/config.yaml`](../config/config.yaml)** - Main configuration file with commented examples
2. **[`config/colbert_reconciliation_example.yaml`](../config/colbert_reconciliation_example.yaml)** - Complete example showing all available options

### Configuration Manager Integration

The [`ConfigurationManager`](../iris_rag/config/manager.py) provides helper methods for accessing configuration:

- [`get_reconciliation_config()`](../iris_rag/config/manager.py:191) - Global reconciliation settings
- [`get_desired_embedding_state(pipeline_type)`](../iris_rag/config/manager.py:220) - Pipeline-specific desired state
- [`get_target_state_config(environment)`](../iris_rag/config/manager.py:290) - Environment-specific target states

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

Performs the complete reconciliation cycle: observe current state, analyze drift, execute healing actions, and verify convergence.

**Syntax:**
```bash
python -m iris_rag.cli run [OPTIONS]
./ragctl run [OPTIONS]
```

**Options:**
- `-p, --pipeline [basic|colbert|noderag|graphrag|hyde|crag|hybrid_ifind]`: Pipeline type to reconcile (default: colbert)
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

Shows the current state of the system, desired state configuration, and any detected drift issues without executing reconciliation actions.

**Syntax:**
```bash
python -m iris_rag.cli status [OPTIONS]
./ragctl status [OPTIONS]
```

**Options:**
- `-p, --pipeline [basic|colbert|noderag|graphrag|hyde|crag|hybrid_ifind]`: Pipeline type to check status for (default: colbert)
- `-s, --since TEXT`: Filter status since time (e.g., "24h", "2023-01-01") - *placeholder, not yet implemented*

#### 3. `daemon` - Continuous Reconciliation

Continuously monitors the system and performs reconciliation at regular intervals. Useful for production environments requiring automatic healing.

**Syntax:**
```bash
python -m iris_rag.cli daemon [OPTIONS]
./ragctl daemon [OPTIONS]
```

**Options:**
- `-p, --pipeline [basic|colbert|noderag|graphrag|hyde|crag|hybrid_ifind]`: Pipeline type to monitor (default: colbert)
- `-i, --interval INTEGER`: Reconciliation interval in seconds (default: 3600 = 1 hour)
- `--max-iterations INTEGER`: Maximum iterations (0 = infinite, default: 0)

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

### 2. Pipeline-Specific Configuration

#### ColBERT Configuration

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

#### Other Pipeline Types

- **BasicRAG**: Uses `vector_dimensions` instead of `token_dimension`, `require_token_embeddings: false`
- **NodeRAG**: Includes chunk hierarchy validation settings
- **GraphRAG**: Includes entity extraction and relationship validation settings
- **HyDE**: Includes hypothetical document generation settings
- **CRAG**: Includes corrective retrieval settings
- **Hybrid iFind**: Includes IRIS iFind integration settings

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

# CLI configuration via environment
export RAG_CLI_CONFIG_PATH=/path/to/config.yaml
export RAG_CLI_LOG_LEVEL=DEBUG
```

## Usage Examples

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

## Error Handling and Troubleshooting

### Common Configuration Issues

**Configuration errors:**
```bash
Error initializing configuration: Configuration file not found: /path/to/config.yaml
```
*Solution*: Verify the configuration file path and permissions.

**Database connection errors:**
```bash
Error during reconciliation: Failed to connect to IRIS database
```
*Solution*: Check database connection settings and network connectivity.

**Permission errors:**
```bash
Error: Permission denied accessing configuration file
```
*Solution*: Ensure proper file permissions and user access rights.

### Debug Mode

Enable debug logging for detailed troubleshooting:

```bash
./ragctl run --log-level DEBUG --pipeline colbert
```

### Performance Issues

If reconciliation is slow:
1. Check database performance and connection latency
2. Adjust batch sizes in configuration
3. Monitor memory usage during operations
4. Consider running during off-peak hours

### Memory Issues

For large datasets:
1. Increase memory limits in configuration
2. Reduce batch sizes
3. Monitor system resources during reconciliation
4. Consider running reconciliation in stages

## Configuration Validation

The configuration is validated when loaded by the [`ConfigurationManager`](../iris_rag/config/manager.py). Invalid configurations will raise a [`ConfigValidationError`](../iris_rag/config/manager.py:6).

## Best Practices

1. **Start with defaults**: Use the example configuration file as a starting point
2. **Environment-specific configs**: Use different target states for development vs. production
3. **Environment variables**: Use environment variables for deployment-specific overrides
4. **Gradual rollout**: Start with `mode: "progressive"` for safer reconciliation
5. **Monitor resources**: Adjust `memory_limit_gb` and `cpu_limit_percent` based on system capacity
6. **Regular validation**: Use `./ragctl status` to monitor system health
7. **Backup before changes**: Enable `backup_before_remediation` for safety

## Related Documentation

- [System Architecture](ARCHITECTURE.md)
- [Comprehensive Generalized Reconciliation Design](../COMPREHENSIVE_GENERALIZED_RECONCILIATION_DESIGN.md)
- [ReconciliationController Implementation](../iris_rag/controllers/reconciliation.py)
- [ConfigurationManager Implementation](../iris_rag/config/manager.py)