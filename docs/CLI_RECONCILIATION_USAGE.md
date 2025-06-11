# RAG Reconciliation CLI Usage Guide

This document provides comprehensive usage instructions for the RAG Reconciliation CLI, which implements GitOps-style commands for managing RAG pipeline state reconciliation.

## Overview

The RAG Reconciliation CLI provides three main commands for managing data integrity across all RAG pipeline implementations:

- **`run`**: Execute reconciliation with optional force and dry-run modes
- **`status`**: Display current system status and drift analysis  
- **`daemon`**: Run continuous reconciliation in daemon mode

## Installation & Setup

The CLI is available through multiple entry points:

### Method 1: Python Module (Recommended)
```bash
python -m iris_rag.cli --help
python -m iris_rag.cli run --pipeline colbert
```

### Method 2: Standalone Script
```bash
./ragctl --help
./ragctl run --pipeline colbert
```

## Global Options

All commands support these global options:

- `-c, --config PATH`: Path to configuration file
- `--log-level [DEBUG|INFO|WARNING|ERROR]`: Set logging level (default: INFO)

## Commands

### 1. `run` - Execute Reconciliation

Performs the complete reconciliation cycle: observe current state, analyze drift, execute healing actions, and verify convergence.

#### Syntax
```bash
python -m iris_rag.cli run [OPTIONS]
./ragctl run [OPTIONS]
```

#### Options
- `-p, --pipeline [basic|colbert|noderag|graphrag|hyde|crag|hybrid_ifind]`: Pipeline type to reconcile (default: colbert)
- `-f, --force`: Force reconciliation even if no drift detected
- `-n, --dry-run`: Analyze drift without executing reconciliation actions

#### Examples

**Basic reconciliation:**
```bash
./ragctl run --pipeline colbert
```

**Force reconciliation regardless of drift:**
```bash
./ragctl run --pipeline basic --force
```

**Dry-run analysis (no actions executed):**
```bash
./ragctl run --pipeline noderag --dry-run
```

**With custom configuration:**
```bash
./ragctl run --config config/production.yaml --pipeline graphrag
```

#### Output

The `run` command provides a comprehensive reconciliation summary including:

- Reconciliation ID and execution time
- Current vs desired state comparison
- Drift analysis with detected issues
- Actions taken during reconciliation
- Convergence verification results

Example output:
```
============================================================
RECONCILIATION SUMMARY
============================================================
Reconciliation ID: 12345678-1234-5678-9abc-123456789abc
Success: ✓
Execution Time: 2.34s

Current State:
  Documents: 933
  Token Embeddings: 164,368

Desired State:
  Target Documents: 1,000
  Embedding Model: fjmgAI/reason-colBERT-150M-GTE-ModernColBERT
  Vector Dimensions: 768

Drift Analysis:
  Drift Detected: Yes
  Issues Found: 1

  Issues:
    ⚠ insufficient_documents: Only 933 documents found, need 1000
      Affected: 67 items

Actions Taken: 1
  • log_issue: Logged drift issue: insufficient_documents

Convergence:
  Converged: ✓
============================================================
```

#### Exit Codes
- `0`: Success, no drift or successfully reconciled
- `1`: Reconciliation failed
- `2`: Reconciliation completed but convergence not achieved

### 2. `status` - Display System Status

Shows the current state of the system, desired state configuration, and any detected drift issues without executing reconciliation actions.

#### Syntax
```bash
python -m iris_rag.cli status [OPTIONS]
./ragctl status [OPTIONS]
```

#### Options
- `-p, --pipeline [basic|colbert|noderag|graphrag|hyde|crag|hybrid_ifind]`: Pipeline type to check status for (default: colbert)
- `-s, --since TEXT`: Filter status since time (e.g., "24h", "2023-01-01") - *placeholder, not yet implemented*

#### Examples

**Check status for default pipeline:**
```bash
./ragctl status
```

**Check status for specific pipeline:**
```bash
./ragctl status --pipeline noderag
```

**Check status with time filter (placeholder):**
```bash
./ragctl status --since 24h
```

#### Output

The `status` command provides detailed system information:

```
============================================================
SYSTEM STATUS
============================================================
Current State:
  Documents: 933
  Token Embeddings: 164,368
  Avg Embedding Size: 0.00
  Observed At: 2025-06-11 08:22:22

Quality Assessment:
  Mock Embeddings Detected: No
  Diversity Score: 1.000
  Missing Embeddings: 0
  Corrupted Embeddings: 0

Desired State:
  Target Documents: 1,000
  Embedding Model: fjmgAI/reason-colBERT-150M-GTE-ModernColBERT
  Vector Dimensions: 768
  Diversity Threshold: 0.700

Completeness Requirements:
  Require All Docs: Yes
  Require Token Embeddings: Yes
  Min Quality Score: 0.800

Drift Analysis:
  Drift Detected: Yes
  Analysis Time: 2025-06-11 08:22:22

  Detected Issues (1):
    ⚠ insufficient_documents (medium)
      Only 933 documents found, need 1000
      Affected: 67 items
      Recommended: Ingest additional documents
============================================================
```

#### Exit Codes
- `0`: No drift detected
- `1`: Non-critical drift detected
- `2`: Critical issues detected

### 3. `daemon` - Continuous Reconciliation

Continuously monitors the system and performs reconciliation at regular intervals. Useful for production environments requiring automatic healing.

#### Syntax
```bash
python -m iris_rag.cli daemon [OPTIONS]
./ragctl daemon [OPTIONS]
```

#### Options
- `-p, --pipeline [basic|colbert|noderag|graphrag|hyde|crag|hybrid_ifind]`: Pipeline type to monitor (default: colbert)
- `-i, --interval INTEGER`: Reconciliation interval in seconds (default: 3600 = 1 hour)
- `--max-iterations INTEGER`: Maximum iterations (0 = infinite, default: 0)

#### Examples

**Run daemon with default settings:**
```bash
./ragctl daemon --pipeline colbert
```

**Run daemon with custom interval:**
```bash
./ragctl daemon --interval 1800 --pipeline basic
```

**Run daemon with limited iterations:**
```bash
./ragctl daemon --max-iterations 10 --interval 600
```

#### Output

The daemon provides continuous monitoring output:

```
Starting reconciliation daemon for colbert pipeline
Interval: 3600 seconds (1.0h)
Max iterations: infinite
Press Ctrl+C to stop gracefully

[2025-06-11 08:22:00] Starting reconciliation iteration 1
  ✓ Reconciliation completed - 1 issues addressed
  Execution time: 2.34s
  Waiting 1.0h until next reconciliation...

[2025-06-11 09:22:00] Starting reconciliation iteration 2
  ✓ No drift detected - system healthy
  Execution time: 0.45s
  Waiting 1.0h until next reconciliation...
```

#### Signal Handling

The daemon supports graceful shutdown:
- **Ctrl+C (SIGINT)**: Initiates graceful shutdown
- **SIGTERM**: Initiates graceful shutdown

## Configuration

### Configuration File

Use the `-c, --config` option to specify a custom configuration file:

```bash
./ragctl run --config config/production.yaml --pipeline colbert
```

### Environment Variables

Configuration can be overridden using environment variables with the `RAG_` prefix:

```bash
export RAG_COLBERT__TARGET_DOCUMENT_COUNT=2000
export RAG_RECONCILIATION__PERFORMANCE__MEMORY_LIMIT_GB=16
./ragctl run --pipeline colbert
```

### Pipeline Types

The CLI supports all RAG pipeline implementations:

- **`basic`**: Standard vector similarity retrieval
- **`colbert`**: Token-level embeddings with MaxSim
- **`noderag`**: Node-based hierarchical retrieval  
- **`graphrag`**: Graph-based entity relationships
- **`hyde`**: Hypothetical document generation
- **`crag`**: Corrective retrieval augmentation
- **`hybrid_ifind`**: IRIS iFind integration

## Error Handling

### Common Issues

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

## Troubleshooting

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

### Network Issues

For connection problems:
1. Verify database connectivity
2. Check firewall settings
3. Validate connection credentials
4. Test with basic database tools first

## Support

For additional support:
- Check the configuration documentation in [`config/`](../config/)
- Review the reconciliation controller implementation in [`iris_rag/controllers/reconciliation.py`](../iris_rag/controllers/reconciliation.py)
- Examine example configurations in [`config/colbert_reconciliation_example.yaml`](../config/colbert_reconciliation_example.yaml)