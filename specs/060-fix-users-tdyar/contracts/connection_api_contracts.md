# Contract: IRIS Connection API (Bug 1)

This feature does not add external HTTP/GraphQL APIs. Contracts here define internal Python interfaces impacted by the fix.

## Module: `iris_vector_rag/common/iris_dbapi_connector.py`

### Behavior Contract

- Must establish connections using `iris.createConnection()` or `iris.dbapi.connect()`.
- Must never call `iris.connect()`.
- Connection errors must include host:port/namespace context.

### Example Usage

```python
from iris_vector_rag.common.iris_dbapi_connector import create_iris_connection

conn = create_iris_connection(
    host="localhost",
    port=1972,
    namespace="USER",
    username="SuperUser",
    password="SYS",
)
```
