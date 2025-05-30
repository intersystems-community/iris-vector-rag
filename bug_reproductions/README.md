# IRIS SQL Vector Search Bug Reproductions

This directory contains standalone, reproducible test scripts that demonstrate critical bugs and limitations in InterSystems IRIS SQL vector search functionality. Each script is completely self-contained and can be run independently to reproduce specific issues.

## Prerequisites

- InterSystems IRIS instance with vector search enabled
- Python 3.7+
- InterSystems IRIS Python driver: `pip install intersystems-iris`

## Bug Reports

### 1. Parameter Binding Issue (`bug1_parameter_binding.py`)

**Critical Issue**: IRIS SQL does not support parameterized queries with vector functions.

**Impact**: Forces developers to use string interpolation, creating SQL injection vulnerabilities.

**Error**: `User defined function (VECTOR_COSINE) arguments number mismatch`

```bash
python bug_reproductions/bug1_parameter_binding.py
```

**Key Findings**:
- Cannot use `?` placeholders with VECTOR_COSINE or TO_VECTOR
- Must use string interpolation (security risk)
- Regular parameter binding works fine for non-vector operations

### 2. HNSW Index Creation on VARCHAR (`bug2_hnsw_varchar.py`)

**Issue**: HNSW indexes cannot be created on VARCHAR columns storing vector data.

**Impact**: Performance degradation for existing systems using VARCHAR for embeddings.

**Error**: `functional indices can only be defined on one vector property`

```bash
python bug_reproductions/bug2_hnsw_varchar.py
```

**Key Findings**:
- HNSW requires native VECTOR column type
- Cannot create HNSW on VARCHAR even with TO_VECTOR conversion
- Forces complete schema migration for existing systems

### 3. Missing VECTOR Type Driver Support (`bug3_vector_driver_support.py`)

**Issue**: Python driver lacks native support for VECTOR data type.

**Impact**: Complex workarounds required for vector operations.

```bash
python bug_reproductions/bug3_vector_driver_support.py
```

**Key Findings**:
- VECTOR columns retrieved as strings, not arrays
- Cannot bind Python lists/arrays to VECTOR parameters
- Must use TO_VECTOR() with string interpolation
- No type metadata for VECTOR columns

### 4. Stored Procedure Limitations (`bug4_stored_procedures.py`)

**Issue**: Vector functions have significant limitations in stored procedures.

**Impact**: Cannot optimize vector operations using stored procedures.

```bash
python bug_reproductions/bug4_stored_procedures.py
```

**Key Findings**:
- Dynamic dimensions often fail in procedures
- Parameter binding problematic with vector functions
- Performance overhead compared to inline SQL
- Limited optimization capabilities

## Running All Tests

To run all bug reproduction scripts:

```bash
# Update connection details in each script first
for script in bug_reproductions/bug*.py; do
    echo "Running $script..."
    python "$script"
    echo -e "\n\n"
done
```

## Connection Configuration

Before running any script, update the connection details at the top of each file:

```python
CONNECTION_STRING = "localhost:1972/USER"  # Your IRIS host:port/namespace
USERNAME = "_SYSTEM"                       # Your username
PASSWORD = "SYS"                          # Your password
NAMESPACE = "USER"                        # Your namespace
```

## Expected vs Actual Behavior

### Expected (Industry Standard)
1. **Parameter Binding**: Should work with all SQL functions including vector operations
2. **HNSW Indexes**: Should support VARCHAR columns with appropriate conversion functions
3. **Driver Support**: Should handle VECTOR type natively like other SQL types
4. **Stored Procedures**: Should support vector functions without limitations

### Actual (Current IRIS Behavior)
1. **Parameter Binding**: Fails with vector functions, forcing string interpolation
2. **HNSW Indexes**: Only works with native VECTOR columns
3. **Driver Support**: Returns VECTOR as strings, no native binding support
4. **Stored Procedures**: Limited support with various restrictions

## Workarounds

Each script includes practical workarounds, but these have significant drawbacks:

1. **String Interpolation**: SQL injection risk
2. **Schema Migration**: Expensive for existing systems
3. **Manual Parsing**: Performance overhead and complexity
4. **Inline SQL**: Cannot leverage stored procedure benefits

## Recommendations for IRIS Team

1. **Priority 1**: Fix parameter binding for vector functions (security critical)
2. **Priority 2**: Add native VECTOR type support to Python driver
3. **Priority 3**: Support HNSW on VARCHAR columns with TO_VECTOR
4. **Priority 4**: Full vector function support in stored procedures

## Additional Resources

- [IRIS Vector Search Documentation](docs/iris_sql_vector_limitations_bug_report.md)
- [Vector Search Technical Details](docs/VECTOR_SEARCH_TECHNICAL_DETAILS.md)
- [Complete Bug Report](docs/IRIS_SQL_VECTOR_LIMITATIONS.md)

## Support

These scripts are provided to help the IRIS development team reproduce and fix these issues. For questions or additional test cases, please refer to the main project documentation.