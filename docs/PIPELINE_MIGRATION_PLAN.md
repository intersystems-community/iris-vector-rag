# RAG Pipeline Migration Plan

## Overview

This document outlines the plan to migrate all 7 RAG pipelines to use a centralized connection management system that will support JDBC, ODBC, and the upcoming dbapi.

## Current Status

### Pipelines to Migrate:
1. **BasicRAG** - Currently has JDBC version, needs refactoring
2. **CRAG** - Using ODBC directly
3. **HyDE** - Using ODBC directly  
4. **ColBERT** - Using ODBC directly
5. **NodeRAG** - Using ODBC directly
6. **GraphRAG** - Using ODBC directly
7. **Hybrid iFIND RAG** - Using ODBC directly

### Connection Issues:
- JDBC connection currently failing with "Access Denied" error
- ODBC doesn't support vector functions properly
- Need to prepare for upcoming dbapi support

## Migration Strategy

### Phase 1: Centralized Connection Management (Current)
1. Create base pipeline class with connection abstraction
2. Create simplified connection manager for immediate use
3. Refactor pipelines to use centralized connection

### Phase 2: JDBC Integration (When Access Fixed)
1. Fix JDBC authentication/access issues
2. Update connection manager to use JDBC by default
3. Run performance benchmarks

### Phase 3: dbapi Integration (Future)
1. Add dbapi support to connection manager
2. Update pipelines if needed
3. Run comprehensive tests

## Implementation Steps

### Step 1: Create Migration Script
Create an automated script to migrate all pipelines to use the new architecture.

### Step 2: Update Each Pipeline
For each pipeline:
1. Inherit from BaseRAGPipeline
2. Remove direct connection handling
3. Use connection manager for all queries
4. Maintain backward compatibility

### Step 3: Create Production Scripts
1. Deployment scripts for different environments
2. Configuration management
3. Health check scripts

### Step 4: Validation
1. Test each migrated pipeline
2. Compare results with original versions
3. Performance benchmarking

## Benefits

1. **Centralized Management**: Single point for connection configuration
2. **Easy Switching**: Switch between JDBC/ODBC/dbapi with environment variable
3. **Better Security**: Credentials managed in one place
4. **Future-Proof**: Ready for new connection types
5. **Consistent Interface**: All pipelines use same connection pattern

## Configuration

Pipelines will support configuration via environment variables:
- `RAG_CONNECTION_TYPE`: jdbc, odbc, or dbapi (default: odbc for now)
- `IRIS_HOST`: Database host
- `IRIS_PORT`: Database port
- `IRIS_NAMESPACE`: IRIS namespace
- `IRIS_USERNAME`: Database username
- `IRIS_PASSWORD`: Database password

## Next Steps

1. Implement migration script
2. Migrate all pipelines
3. Create deployment scripts
4. Run validation tests