# Vector Data Migration Scripts

This directory contains comprehensive scripts for migrating vector data from `VECTOR(DOUBLE)` to `VECTOR(FLOAT)` format.

## Quick Start

### Option 1: Complete Automated Migration (Recommended)
```bash
# Dry run to preview the complete migration
python scripts/complete_vector_float_migration.py --strategy in-place --dry-run --verbose

# Execute the complete migration
python scripts/complete_vector_float_migration.py --strategy in-place --verbose
```

### Option 2: Manual Step-by-Step Migration
```bash
# 1. Migrate database data
python scripts/migrate_vector_data_double_to_float.py --dry-run --verbose
python scripts/migrate_vector_data_double_to_float.py --verbose

# 2. Verify migration
python scripts/verify_vector_data_migration.py --verbose

# 3. Quick verification
python scripts/verify_vector_data_migration.py --quick
```

## Migration Scripts

### 1. `complete_vector_float_migration.py` - Complete Migration Orchestrator
**Purpose**: Unified interface for the entire migration process

**Features**:
- Orchestrates all migration steps automatically
- Supports both in-place and re-ingestion strategies
- Comprehensive error handling and reporting
- Built-in verification and testing

**Usage**:
```bash
# In-place migration (recommended for production)
python scripts/complete_vector_float_migration.py --strategy in-place

# Re-ingestion migration (safer for development)
python scripts/complete_vector_float_migration.py --strategy reingest --data-source sample
```

### 2. `migrate_vector_data_double_to_float.py` - In-Place Data Migration
**Purpose**: Convert existing `VECTOR(DOUBLE)` data to `VECTOR(FLOAT)` in-place

**Features**:
- Safe ALTER TABLE operations with backup columns
- Automatic dimension detection
- Comprehensive error handling and rollback
- Detailed progress monitoring

**Tables Migrated**:
- `RAG.SourceDocuments.embedding` (384 dimensions)
- `RAG.DocumentChunks.chunk_embedding` (384 dimensions)
- `RAG.Entities.embedding` (384 dimensions)
- `RAG.KnowledgeGraphNodes.embedding` (384 dimensions)
- `RAG.DocumentTokenEmbeddings.token_embedding` (128 dimensions)

**Usage**:
```bash
# Preview changes
python scripts/migrate_vector_data_double_to_float.py --dry-run --verbose

# Execute migration
python scripts/migrate_vector_data_double_to_float.py --verbose
```

### 3. `verify_vector_data_migration.py` - Migration Verification
**Purpose**: Comprehensive verification of migration results

**Verification Checks**:
- Schema verification (column types)
- Data integrity (row counts, dimensions)
- Functionality tests (TO_VECTOR, vector operations)
- Performance validation

**Usage**:
```bash
# Comprehensive verification
python scripts/verify_vector_data_migration.py --verbose

# Quick check
python scripts/verify_vector_data_migration.py --quick
```

### 4. `reingest_data_with_vector_float.py` - Safe Re-ingestion
**Purpose**: Alternative migration via data backup, clearing, and re-ingestion

**Features**:
- Complete data backup before clearing
- Safe table clearing with foreign key handling
- Re-ingestion using updated VECTOR(FLOAT) code
- Comprehensive verification

**Usage**:
```bash
# Re-ingest with sample data
python scripts/reingest_data_with_vector_float.py --data-source sample --verbose

# Re-ingest with full dataset
python scripts/reingest_data_with_vector_float.py --data-source full --verbose
```

## Migration Strategies

### Strategy 1: In-Place Migration (Recommended)
**Best for**: Production systems, large datasets, minimal downtime

**Process**:
1. Create backup columns for each vector column
2. Copy existing data to backup columns
3. ALTER TABLE to change column type to VECTOR(FLOAT)
4. Convert data using CAST operations
5. Verify conversion success
6. Keep backup columns for safety

**Advantages**:
- Preserves existing data
- Faster execution
- No re-processing required
- Minimal downtime

**Considerations**:
- Requires ALTER TABLE permissions
- Database must support VECTOR type conversion

### Strategy 2: Re-ingestion (Alternative)
**Best for**: Development systems, testing, when in-place migration is risky

**Process**:
1. Backup all vector table data to JSON files
2. Clear vector tables (respecting foreign keys)
3. Re-run data ingestion with updated VECTOR(FLOAT) code
4. Verify re-ingestion results

**Advantages**:
- Clean migration
- Good for testing new ingestion code
- Easier rollback from backups

**Considerations**:
- Requires data re-processing time
- Needs access to original data sources
- Longer downtime

## Error Handling and Rollback

### Automatic Rollback (In-Place Migration)
```bash
# Rollback using backup directory
python scripts/migrate_vector_data_double_to_float.py --rollback <backup_directory>
```

### Manual Rollback (Re-ingestion)
```bash
# Restore from JSON backup files
# (Detailed instructions in reingestion report)
```

## Monitoring and Reports

All scripts generate detailed reports:
- **Migration Reports**: JSON and Markdown summaries
- **Verification Reports**: Comprehensive test results
- **Error Logs**: Detailed error information for troubleshooting

Report files are automatically timestamped and saved in the current directory.

## Prerequisites

### Required Python Packages
- `sqlalchemy` (for database operations)
- `numpy` (for vector operations, optional)

### Database Requirements
- IRIS database with vector support
- Appropriate permissions for ALTER TABLE operations
- Existing vector data in tables

### Environment Setup
```bash
# Ensure IRIS connector is available
python -c "from common.iris_connector import get_iris_connection; print('✓ IRIS connector available')"

# Test database connection
python scripts/verify_vector_data_migration.py --quick
```

## Troubleshooting

### Common Issues

1. **Connection Errors**
   - Verify IRIS database is running
   - Check connection parameters in `common/iris_connector.py`

2. **Permission Errors**
   - Ensure database user has ALTER TABLE permissions
   - Verify schema access rights

3. **Type Conversion Errors**
   - Check if VECTOR(FLOAT) is supported in your IRIS version
   - Verify vector data format compatibility

4. **Memory Issues (Large Datasets)**
   - Use batch processing options
   - Consider re-ingestion strategy for very large datasets

### Getting Help

1. Check the detailed logs generated by each script
2. Review the migration reports for specific error details
3. Use `--verbose` flag for detailed debugging information
4. Refer to the main documentation in `docs/VECTOR_FLOAT_MIGRATION_SUMMARY.md`

## Testing

After migration, run these tests to verify functionality:

```bash
# Basic RAG pipeline test
python tests/test_basic_rag_pipeline.py

# HNSW integration test
python tests/test_hnsw_integration.py

# Vector similarity test
python tests/test_vector_similarity_search.py
```

## Performance Monitoring

Monitor these metrics after migration:
- Storage space reduction (~50% expected)
- Query performance improvements
- Memory usage reduction
- HNSW index efficiency

```sql
-- Check storage usage
SELECT TABLE_NAME, DATA_LENGTH, INDEX_LENGTH
FROM INFORMATION_SCHEMA.TABLES
WHERE TABLE_SCHEMA = 'RAG';
```

## Important: Schema Display Limitation

**CRITICAL UNDERSTANDING**: After migration, vector columns may still show as `VARCHAR` in schema introspection. This is **NORMAL and EXPECTED** due to IRIS Python driver limitations.

### Core Principle: Driver Limitation, Not Migration Failure

The InterSystems IRIS Python driver does NOT natively support the VECTOR data type:

- ✅ **Vector columns ARE correctly VECTOR(FLOAT) in the database**
- ✅ **All vector operations work perfectly (VECTOR_COSINE, etc.)**
- ✅ **HNSW indexes function correctly**
- ✅ **Storage space reduction is achieved**
- ❌ **Schema shows VARCHAR due to driver limitation**
- ❌ **Python returns vector data as strings**

### Verification

Use this script to verify the migration is functionally complete:

```bash
python scripts/vector_schema_limitation_explanation.py --verbose
```

### Documentation

See [`VECTOR_SCHEMA_LIMITATION_SOLUTION.md`](../VECTOR_SCHEMA_LIMITATION_SOLUTION.md) for complete explanation.

**Bottom Line**: If vector operations work and HNSW indexes function, the migration is successful regardless of schema display.