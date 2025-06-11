# Infrastructure Optimization for DBAPI Test System

This document describes the infrastructure optimization features implemented for the comprehensive DBAPI test system to support persistent IRIS instances and dramatically improve test iteration speed.

## Overview

The infrastructure optimization provides three modes for managing IRIS containers during testing:

1. **Reuse Mode**: Check if IRIS container is already running and reuse it (skip container setup)
2. **Clean Mode**: Force fresh container setup (current behavior, default)
3. **Incremental Mode**: Reuse container but optionally clear/reset specific data

## Command-Line Flags

### Shell Script Flags (`scripts/run_comprehensive_dbapi_test.sh`)

- `--reuse-iris` - Reuse existing IRIS container if available
- `--clean-iris` - Force fresh container setup (default)
- `--reset-data` - Clear data but keep schema when reusing

### Examples

```bash
# Default behavior - fresh container
./scripts/run_comprehensive_dbapi_test.sh

# Reuse existing container (fastest for iteration)
./scripts/run_comprehensive_dbapi_test.sh --reuse-iris

# Reuse container but reset data
./scripts/run_comprehensive_dbapi_test.sh --reuse-iris --reset-data

# Force fresh container (explicit)
./scripts/run_comprehensive_dbapi_test.sh --clean-iris

# Combine with other options
./scripts/run_comprehensive_dbapi_test.sh --reuse-iris --documents 500 --verbose
```

## Makefile Targets

### Standard Targets
- `make test-dbapi-comprehensive` - Default behavior (fresh container, 1000 docs)
- `make test-dbapi-comprehensive-large` - Fresh container with 5000 docs
- `make test-dbapi-comprehensive-quick` - Fresh container with 500 docs

### Infrastructure Optimization Targets
- `make test-dbapi-comprehensive-reuse` - Reuse container (faster iteration)
- `make test-dbapi-comprehensive-reuse-reset` - Reuse container with data reset
- `make test-dbapi-comprehensive-clean` - Explicit fresh container

### Development Targets (Optimized for Fast Iteration)
- `make test-dbapi-dev` - Reuse container, 500 docs (~5-10 minutes)
- `make test-dbapi-dev-reset` - Reuse container with data reset, 500 docs (~8-12 minutes)

## Performance Benefits

### Time Savings

| Mode | Container Setup | Data Loading | Total Time Saved |
|------|----------------|--------------|------------------|
| Fresh Container | ~3-5 minutes | ~5-10 minutes | Baseline |
| Reuse Container | ~10 seconds | ~5-10 minutes | ~3-5 minutes |
| Reuse + Existing Data | ~10 seconds | ~0 seconds | ~8-15 minutes |
| Reuse + Reset Data | ~10 seconds | ~5-10 minutes | ~3-5 minutes |

### Typical Development Workflow

1. **Initial Setup**: `make test-dbapi-comprehensive` (full fresh setup)
2. **Code Changes**: `make test-dbapi-dev` (fast iteration with existing data)
3. **Data Changes**: `make test-dbapi-dev-reset` (fast iteration with fresh data)
4. **Final Validation**: `make test-dbapi-comprehensive-clean` (clean validation)

## How It Works

### Container Detection

The system checks for existing IRIS containers using:
```bash
docker-compose ps iris_db --format json
```

### Health Verification

Before reusing a container, the system verifies:
1. Container is running
2. Container is healthy
3. IRIS connection is working
4. Database schema exists

### Data Management

When `--reset-data` is used, the system:
1. Preserves database schema
2. Clears data from all RAG tables:
   - `RAG.SourceDocuments`
   - `RAG.Entities`
   - `RAG.Relationships`
   - `RAG.KnowledgeGraphNodes`
   - `RAG.KnowledgeGraphEdges`
   - `RAG.ChunkedDocuments`

### Fallback Behavior

If container reuse fails for any reason, the system automatically falls back to fresh container setup.

## Environment Variables

The optimization modes are controlled by environment variables:

- `IRIS_REUSE_MODE` - Set to "true" to enable reuse mode
- `IRIS_CLEAN_MODE` - Set to "true" to force clean mode
- `IRIS_RESET_DATA` - Set to "true" to reset data when reusing

## Implementation Details

### Shell Script Changes

The `run_comprehensive_dbapi_test.sh` script now includes:
- Container status checking functions
- Health verification functions
- Data reset capabilities
- Conditional cleanup logic

### Python Test Script Changes

The `test_comprehensive_dbapi_rag_system.py` script now includes:
- Container status checking methods
- Data existence verification
- Conditional data loading
- Conditional cleanup

### Key Methods Added

```python
def check_iris_container_running(self) -> bool
def check_iris_container_healthy(self) -> bool
def test_iris_connection_simple(self) -> bool
def reset_iris_data_tables(self) -> bool
def check_existing_data(self) -> Dict[str, int]
```

## Best Practices

### For Development
1. Use `make test-dbapi-dev` for rapid iteration
2. Use `make test-dbapi-dev-reset` when testing data-related changes
3. Periodically run `make test-dbapi-comprehensive-clean` for validation

### For CI/CD
1. Always use clean mode for production validation
2. Consider reuse mode for development branches to speed up testing
3. Use explicit `--clean-iris` flag to ensure reproducible results

### For Debugging
1. Use `--verbose` flag with any mode for detailed logging
2. Check container logs: `docker-compose logs iris_db`
3. Manually inspect container: `docker-compose exec iris_db iris session iris`

## Troubleshooting

### Container Reuse Fails
- Check if container is actually running: `docker-compose ps`
- Verify container health: `docker-compose ps iris_db`
- Check IRIS logs: `docker-compose logs iris_db`

### Data Reset Fails
- Verify database connection is working
- Check if tables exist in RAG schema
- Manually connect and verify: `docker-compose exec iris_db iris session iris`

### Performance Not Improved
- Ensure you're using reuse mode: `--reuse-iris`
- Check if data already exists when expected
- Verify container is actually being reused (check logs)

## Future Enhancements

Potential improvements for the infrastructure optimization:

1. **Selective Data Reset**: Reset only specific tables instead of all
2. **Data Snapshots**: Save/restore data snapshots for different test scenarios
3. **Multi-Container Support**: Support for multiple IRIS instances
4. **Automated Health Monitoring**: Continuous health checking during tests
5. **Performance Metrics**: Detailed timing and resource usage tracking