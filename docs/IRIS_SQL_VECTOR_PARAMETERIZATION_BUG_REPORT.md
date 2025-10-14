# IRIS SQL Vector Operations Driver Auto-Parameterization Bug Report

## Executive Summary

InterSystems IRIS database drivers exhibit problematic auto-parameterization behavior that breaks vector search functionality when using `TO_VECTOR()` functions and `TOP` clauses. This is a **critical driver-level bug** that prevents proper vector database operations.

## Problem Description

### Core Issue
IRIS Python DBAPI and JDBC drivers automatically convert embedded literals in SQL strings to parameter markers (`:%qpar(n)`) during `cursor.execute()`, even when no parameter list is provided. This auto-parameterization occurs **after** SQL generation but **before** compilation, causing IRIS SQL parser errors for constructs that cannot accept parameter markers.

### Affected SQL Constructs

1. **`TOP` clauses**: `SELECT TOP 5` → `SELECT TOP :%qpar(1)` (Invalid)
2. **`TO_VECTOR()` dimension parameters**: `TO_VECTOR(?, 'FLOAT', 384)` → `TO_VECTOR(?, 'FLOAT', :%qpar(2))` (Invalid)
3. **`TO_VECTOR()` data type literals**: `TO_VECTOR(?, 'FLOAT', 384)` → `TO_VECTOR(?, :%qpar(1), :%qpar(2))` (Invalid)

### Error Manifestation
```
[SQLCODE: <-1>:<Invalid SQL statement>]
[Location: <Prepare>]
[%msg: < ) expected, : found ^SELECT TOP :%qpar(1) doc_id , text_content , VECTOR_COSINE ( embedding , TO_VECTOR ( :%qpar(2) , :%qpar>]
```

## Historical Context

### Git History Analysis
- **Original issue**: Vector operations failing due to auto-parameterization
- **Commit `e7b5632ad`**: Merge conflict between `DOUBLE` and `FLOAT` data types with dimension parameters
- **Commit `81183a1c8`**: Jonathan Zhou reverted to `FLOAT` with comment "(currently not working)"

### Timeline of Attempted Fixes
1. **Initial approach**: Used `DOUBLE` data type in `TO_VECTOR()`
2. **Dimension fix**: Added missing `{embedding_dim}` parameter
3. **Data type revert**: Changed back to `FLOAT` due to compilation failures
4. **Current state**: Still failing due to driver auto-parameterization

## Technical Analysis

### Driver Behavior
```python
# Input SQL (what we generate):
sql = "SELECT TOP 5 doc_id, VECTOR_COSINE(embedding, TO_VECTOR(?, 'FLOAT', 384)) AS score FROM table"

# What the driver converts it to automatically:
# "SELECT TOP :%qpar(1) doc_id, VECTOR_COSINE(embedding, TO_VECTOR(:%qpar(2), :%qpar(3), :%qpar(4))) AS score FROM table"

# Parameters passed: ["[0.1,0.2,0.3]"]
# But IRIS expects exactly 1 parameter, gets 4 parameter placeholders
```

### Root Cause
1. **Driver over-parameterization**: Converts ALL numeric and string literals to parameters
2. **SQL parser restrictions**: IRIS cannot accept parameters in `TOP` and `TO_VECTOR` dimension/type positions
3. **Parameter mismatch**: Driver creates more parameter placeholders than parameters provided

## Current Workarounds

### String Concatenation Approach
```python
# Avoid f-strings and direct formatting that triggers parameterization
def format_vector_search_sql():
    top_k_str = str(top_k)
    embedding_dim_str = str(embedding_dim)
    
    # Use string concatenation instead of f-strings
    sql_parts = [
        "SELECT TOP ", top_k_str, " ", id_column,
        ", VECTOR_COSINE(", vector_column, ", TO_VECTOR('", 
        vector_string, "', 'FLOAT', ", embedding_dim_str, ")) AS score"
    ]
    return "".join(sql_parts)
```

### Limitations of Current Workarounds
- **Still fails**: Driver auto-parameterization occurs regardless of generation method
- **Security concerns**: Forces use of string interpolation instead of safe parameterization
- **Maintenance burden**: Requires constant vigilance to avoid triggering auto-parameterization

## Impact Assessment

### Functional Impact
- **Vector search broken**: Core RAG functionality non-functional
- **All pipelines affected**: BasicRAG, CRAG, and other vector-dependent pipelines fail
- **Development blocked**: Cannot reliably test or deploy vector features

### Business Impact
- **Production unusable**: Vector database features cannot be deployed
- **Development velocity**: Significant time spent on driver workarounds
- **Technical debt**: Accumulating unsafe SQL generation patterns

## Required Driver Fixes

### Option 1: Selective Auto-Parameterization (Preferred)
Modify drivers to **not** auto-parameterize literals in specific contexts:
- `TOP` and `LIMIT` clauses
- `TO_VECTOR()` dimension and data type parameters
- `FETCH FIRST` clauses

### Option 2: Auto-Parameterization Configuration
Add driver configuration option to:
- Disable auto-parameterization globally
- Enable/disable per connection
- Whitelist/blacklist specific SQL constructs

### Option 3: Enhanced SQL Parser
Update IRIS SQL parser to accept parameter markers in:
- `TOP` clauses: `SELECT TOP ? ...`
- `TO_VECTOR()` parameters: `TO_VECTOR(?, ?, ?)`

## Recommended Actions

### Immediate (Application Level)
1. ✅ **Implement string concatenation workaround** (high security risk)
2. ✅ **Document driver limitations** (this document)
3. 🔄 **Comprehensive testing** of workaround approaches

### Short Term (InterSystems Engagement)
1. 📋 **Submit formal bug report** to InterSystems
2. 📋 **Request driver configuration options** for auto-parameterization
3. 📋 **Provide reproducible test cases** and error scenarios

### Long Term (Strategic)
1. 📋 **Evaluate alternative database drivers** if fixes unavailable
2. 📋 **Consider IRIS SQL generation abstraction layer**
3. 📋 **Monitor InterSystems roadmap** for vector operation improvements

## Test Cases for Bug Report

### Minimal Reproduction
```python
import iris

conn = iris.connect(connection_string)
cursor = conn.cursor()

# This should work but fails due to auto-parameterization:
sql = "SELECT TOP 5 doc_id FROM table"
cursor.execute(sql)  # Fails: TOP :%qpar(1)

# This should work but fails:
sql = "SELECT VECTOR_COSINE(embedding, TO_VECTOR(?, 'FLOAT', 384)) FROM table"
cursor.execute(sql, ["[0.1,0.2]"])  # Fails: TO_VECTOR(:%qpar(1), :%qpar(2), :%qpar(3))
```

### Expected vs Actual Behavior
- **Expected**: Literals remain as literals, only explicit `?` becomes parameter
- **Actual**: All literals become parameters, breaking SQL syntax

## Contact Information

**Reporter**: [Your Information]  
**Project**: RAG Templates Vector Database Implementation  
**IRIS Version**: [Version Information]  
**Driver Version**: [Python DBAPI/JDBC Version]  
**Date**: 2025-09-14

## Appendix: Related Files
- `common/vector_sql_utils.py`: Core affected utility functions
- `iris_rag/storage/vector_store_iris.py`: Vector storage implementation
- `scripts/test_vector_sql_fix.py`: Test suite demonstrating issues
- Git commits: `81183a1c8`, `e7b5632ad` - Historical fix attempts