# IRIS Connection Architecture Guide

## Overview

The RAG Templates framework uses a **dual-path connection architecture** for InterSystems IRIS database connections. This document explains the two connection systems, when to use each, and how to troubleshoot connection issues.

## 🏗️ Architecture Summary

```
┌─────────────────────────────────────────────────────────────┐
│                    IRIS Connection Systems                  │
├─────────────────────────────────────────────────────────────┤
│                                                             │
│  ┌─────────────────────┐    ┌─────────────────────────────┐ │
│  │   DBAPI System      │    │    JDBC System             │ │
│  │  (iris_dbapi_       │    │  (iris_connection_          │ │
│  │   connector)        │    │   manager)                  │ │
│  │                     │    │                             │ │
│  │ ✓ Pure DBAPI        │    │ ✓ DBAPI → JDBC fallback    │ │
│  │ ✓ Fast queries      │    │ ✓ Reliable DDL operations   │ │
│  │ ✓ Low overhead      │    │ ✓ Schema management         │ │
│  │ ✓ RAG operations    │    │ ✓ Administrative tasks      │ │
│  └─────────────────────┘    └─────────────────────────────┘ │
│                                                             │
└─────────────────────────────────────────────────────────────┘
```

## 📋 Connection Systems Comparison

| Aspect | DBAPI System | JDBC System |
|--------|--------------|-------------|
| **Module** | `common.iris_dbapi_connector` | `common.iris_connection_manager` |
| **Primary Use** | RAG queries & data operations | Schema management & DDL |
| **Connection Type** | Pure DBAPI (intersystems-irispython) | DBAPI with JDBC fallback |
| **Performance** | Optimized for high-frequency queries | Reliable for administrative operations |
| **Error Handling** | Simple success/failure | Smart fallback with detailed logging |
| **Used By** | Core RAG pipelines, vector search | Schema manager, utilities, demos |

## 🎯 When to Use Which System

### Use **DBAPI System** (`iris_dbapi_connector`) for:
- ✅ **Core RAG operations** (vector search, document retrieval)
- ✅ **High-frequency queries** (embeddings, similarity search)
- ✅ **Performance-critical paths** (real-time RAG queries)
- ✅ **Simple connection needs** (just need a working DBAPI connection)

### Use **JDBC System** (`iris_connection_manager`) for:
- ✅ **Schema management** (table creation, migrations)
- ✅ **Administrative operations** (data utilities, maintenance)
- ✅ **Development tools** (demos, testing, validation)
- ✅ **Fallback reliability** (when DBAPI environment is uncertain)

## 🔧 Import Patterns

### DBAPI System Usage
```python
# For core RAG operations
from common.iris_dbapi_connector import get_iris_dbapi_connection

conn = get_iris_dbapi_connection()
if conn:
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM RAG.SourceDocuments LIMIT 5")
    results = cursor.fetchall()
    cursor.close()
    conn.close()
```

### JDBC System Usage
```python
# For schema management and utilities
from common.iris_connection_manager import get_iris_connection

conn = get_iris_connection()  # Prefers DBAPI, falls back to JDBC
cursor = conn.cursor()
cursor.execute("CREATE TABLE IF NOT EXISTS RAG.NewTable (...)")
conn.commit()
cursor.close()
conn.close()
```

## 🔍 Connection Flow Details

### DBAPI System Flow
```
1. Import intersystems_iris.dbapi
2. Check for _DBAPI submodule with connect()
3. If not found, fallback to import iris
4. Return DBAPI connection or None
```

### JDBC System Flow
```
1. Check environment compatibility
2. Try intersystems_iris.dbapi import
3. Attempt DBAPI connection
4. If DBAPI fails → Fall back to JDBC
5. Return connection with type tracking
```

## ⚠️ Common Issues & Solutions

### Issue: "JDBC fallback" warnings
**Symptom:** Logs show "Falling back to JDBC connection"
**Cause:** DBAPI connection failed in `iris_connection_manager`
**Solution:** This is normal behavior for schema utilities - JDBC is reliable for DDL operations

### Issue: "Circular import" errors
**Symptom:** "partially initialized module 'intersystems_iris' has no attribute 'dbapi'"
**Cause:** Multiple modules importing IRIS packages simultaneously
**Solution:** Use the appropriate connection system for your use case

### Issue: "No connect method" errors
**Symptom:** "module 'intersystems_iris.dbapi' has no attribute 'connect'"
**Cause:** Wrong IRIS module version or installation
**Solution:** Ensure `intersystems-irispython` package is properly installed

## 🎪 Environment Requirements

### Package Installation
```bash
# Required for DBAPI connections
pip install intersystems-irispython

# Alternative for UV users
uv add intersystems-irispython
```

### Environment Variables
```bash
# Connection parameters (used by both systems)
export IRIS_HOST="localhost"
export IRIS_PORT="1972"
export IRIS_NAMESPACE="USER"
export IRIS_USER="_SYSTEM"
export IRIS_PASSWORD="SYS"
```

## 🔬 Debugging Connection Issues

### Enable Debug Logging
```python
import logging
logging.basicConfig(level=logging.DEBUG)

# This will show detailed connection attempts
from common.iris_connection_manager import get_iris_connection
conn = get_iris_connection()
```

### Test Connection Systems Individually
```python
# Test DBAPI system
from common.iris_dbapi_connector import get_iris_dbapi_connection
dbapi_conn = get_iris_dbapi_connection()
print(f"DBAPI: {'✅' if dbapi_conn else '❌'}")

# Test JDBC system  
from common.iris_connection_manager import IRISConnectionManager
manager = IRISConnectionManager()
jdbc_conn = manager.get_connection()
print(f"JDBC: {manager._connection_type}")
```

## 📊 System Usage Mapping

### Files Using DBAPI System (13 files)
- `iris_rag/core/connection.py` - Core RAG connections
- `iris_rag/storage/vector_store_iris.py` - Vector operations
- `iris_rag/pipelines/*.py` - RAG pipeline implementations
- `data/loader_fixed.py` - Document loading

### Files Using JDBC System (76 files)
- `scripts/utilities/schema_managed_data_utils.py` - Schema management
- `examples/demo_chat_app.py` - Demo applications
- `tests/test_*.py` - Test infrastructure
- `scripts/populate_*.py` - Data population utilities

## 🛣️ Future Roadmap

### Planned Improvements
1. **Unified Connection API** - Single interface for both systems
2. **Better Error Messages** - Clearer indication of which system failed
3. **Connection Health Checks** - Automated diagnostics
4. **Performance Monitoring** - Connection pool metrics

### Refactoring Considerations
- **Risk Assessment** - 524 files potentially affected
- **Backward Compatibility** - Maintain existing APIs during transition
- **Performance Impact** - Ensure unified system doesn't degrade performance
- **Testing Coverage** - Comprehensive tests for unified connection layer

## 💡 Best Practices

1. **Use DBAPI for RAG operations** - Faster and more direct
2. **Use JDBC system for utilities** - More reliable fallback behavior
3. **Handle connection failures gracefully** - Both systems can fail
4. **Log connection types** - Help with debugging
5. **Test in your environment** - IRIS package availability varies

## 🆘 Getting Help

If you encounter connection issues:

1. **Check the logs** - Look for specific error messages
2. **Verify IRIS installation** - Ensure `intersystems-irispython` is available
3. **Test connection manually** - Use the debugging examples above
4. **Check environment variables** - Ensure IRIS_* variables are set
5. **Try both systems** - See which one works in your environment

---

*This architecture evolved to handle the diverse connection needs of a comprehensive RAG framework. While it adds complexity, it provides reliability and performance optimization for different use cases.*