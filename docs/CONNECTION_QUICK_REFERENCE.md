# IRIS Connection Quick Reference

## 🚀 Which Connection System Should I Use?

### ⚡ Need to do RAG queries, vector search, or data operations?
```python
from common.iris_dbapi_connector import get_iris_dbapi_connection
conn = get_iris_dbapi_connection()
```
**Use DBAPI System** - Fast, direct, optimized for queries

### 🔧 Need to do schema changes, utilities, or admin tasks?  
```python
from common.iris_connection_manager import get_iris_connection
conn = get_iris_connection()
```
**Use JDBC System** - Reliable fallback, good for DDL operations

## 🎯 Quick Decision Matrix

| Task | Use | Import |
|------|-----|--------|
| Vector search | DBAPI | `from common.iris_dbapi_connector import get_iris_dbapi_connection` |
| Document retrieval | DBAPI | `from common.iris_dbapi_connector import get_iris_dbapi_connection` |
| Schema management | JDBC | `from common.iris_connection_manager import get_iris_connection` |
| Data utilities | JDBC | `from common.iris_connection_manager import get_iris_connection` |
| Demo apps | JDBC | `from common.iris_connection_manager import get_iris_connection` |
| Tests | JDBC | `from common.iris_connection_manager import get_iris_connection` |

## ⚠️ Common Messages You'll See

### ✅ Normal (Expected)
- `"Successfully connected to IRIS using DBAPI interface"` - DBAPI working
- `"Falling back to JDBC connection"` - JDBC system's normal fallback behavior  
- `"✓ Connected using JDBC"` - JDBC system working properly

### ⚠️ Investigate Further
- `"Failed to import 'intersystems_iris.dbapi' module"` - Package installation issue
- `"All connection methods failed"` - Neither DBAPI nor JDBC working

## 🔍 Quick Debug

```python
# Test both systems quickly
import logging
logging.basicConfig(level=logging.INFO)

# Test DBAPI
from common.iris_dbapi_connector import get_iris_dbapi_connection
dbapi_conn = get_iris_dbapi_connection()
print(f"DBAPI: {'✅ Working' if dbapi_conn else '❌ Failed'}")

# Test JDBC  
from common.iris_connection_manager import get_iris_connection
jdbc_conn = get_iris_connection()
print(f"JDBC: ✅ Working")
```

📖 **Full details:** [IRIS Connection Architecture Guide](IRIS_CONNECTION_ARCHITECTURE.md)