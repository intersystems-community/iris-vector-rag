# Immediate Architecture Fixes Guide

## Critical Issues to Address

Based on the architectural review, these are the **immediate fixes** needed to resolve the benchmark script issues and align with the user's DBAPI preference.

## Fix 1: Benchmark Script Connection Strategy

### Problem
[`scripts/run_rag_benchmarks.py`](scripts/run_rag_benchmarks.py:67) imports JDBC connector but user prefers DBAPI as default.

### Solution
```python
# Replace line 67 in scripts/run_rag_benchmarks.py
# OLD:
from common.iris_connector import get_iris_connection

# NEW:
from common.iris_dbapi_connector import get_iris_dbapi_connection
```

### Implementation
```python
# Update setup_database_connection function (around line 400)
def setup_database_connection(args) -> Optional[Any]:
    """Set up and verify the database connection using DBAPI (user preference)."""
    logger.info("Establishing DBAPI connection to IRIS database...")
    
    if args.use_mock:
        # Keep existing mock logic
        pass
    else:
        # Use DBAPI connector as default
        iris_conn = get_iris_dbapi_connection()
    
    return iris_conn
```

## Fix 2: Core Pipeline Import Inconsistencies

### Problem
Mixed JDBC/DBAPI imports across core pipelines cause type mismatches.

### Solution: Standardize to DBAPI

#### Fix [`core_pipelines/basic_rag_pipeline.py`](core_pipelines/basic_rag_pipeline.py:20)
```python
# Replace line 20:
# OLD:
from common.iris_connector_jdbc import get_iris_connection

# NEW:
from common.iris_dbapi_connector import get_iris_dbapi_connection
```

#### Update Type Hints (line 25)
```python
# OLD:
def __init__(self, iris_connector: IRISConnection, ...)

# NEW:
from typing import Any
def __init__(self, iris_connector: Any, ...)  # Support both during transition
```

## Fix 3: Create Unified Connector Interface

### Create [`common/connector_interface.py`](common/connector_interface.py)
```python
"""
Unified interface for IRIS database connectors.
Supports both DBAPI and JDBC implementations.
"""

from abc import ABC, abstractmethod
from typing import Any, List, Optional

class IRISConnectorInterface(ABC):
    """Abstract interface for IRIS database connectors."""
    
    @abstractmethod
    def cursor(self):
        """Return a database cursor."""
        pass
    
    @abstractmethod
    def execute_query(self, query: str, params: Optional[List] = None):
        """Execute a query with optional parameters."""
        pass
    
    @abstractmethod
    def close(self):
        """Close the database connection."""
        pass
    
    @abstractmethod
    def commit(self):
        """Commit the current transaction."""
        pass

class DBAPIConnectorWrapper(IRISConnectorInterface):
    """Wrapper for DBAPI connections to implement standard interface."""
    
    def __init__(self, connection):
        self.connection = connection
    
    def cursor(self):
        return self.connection.cursor()
    
    def execute_query(self, query: str, params: Optional[List] = None):
        cursor = self.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        return cursor
    
    def close(self):
        self.connection.close()
    
    def commit(self):
        self.connection.commit()

class JDBCConnectorWrapper(IRISConnectorInterface):
    """Wrapper for JDBC connections to implement standard interface."""
    
    def __init__(self, connection):
        self.connection = connection
    
    def cursor(self):
        return self.connection.cursor()
    
    def execute_query(self, query: str, params: Optional[List] = None):
        cursor = self.cursor()
        if params:
            cursor.execute(query, params)
        else:
            cursor.execute(query)
        return cursor
    
    def close(self):
        self.connection.close()
    
    def commit(self):
        self.connection.commit()
```

## Fix 4: Update Connection Factory

### Create [`common/connection_factory.py`](common/connection_factory.py)
```python
"""
Factory for creating standardized database connections.
Implements user preference for DBAPI as default.
"""

import os
import logging
from typing import Dict, Any, Optional
from .connector_interface import IRISConnectorInterface, DBAPIConnectorWrapper, JDBCConnectorWrapper

logger = logging.getLogger(__name__)

class ConnectionFactory:
    """Factory for creating IRIS database connections."""
    
    @staticmethod
    def create_connection(connection_type: str = "dbapi", **config) -> IRISConnectorInterface:
        """
        Create an IRIS database connection.
        
        Args:
            connection_type: "dbapi" (default) or "jdbc"
            **config: Connection configuration parameters
            
        Returns:
            IRISConnectorInterface implementation
        """
        if connection_type == "dbapi":
            return ConnectionFactory._create_dbapi_connection(**config)
        elif connection_type == "jdbc":
            return ConnectionFactory._create_jdbc_connection(**config)
        else:
            raise ValueError(f"Unknown connection type: {connection_type}")
    
    @staticmethod
    def _create_dbapi_connection(**config) -> IRISConnectorInterface:
        """Create DBAPI connection (user preference)."""
        from .iris_dbapi_connector import get_iris_dbapi_connection
        
        # Set environment variables if provided in config
        if config:
            for key, value in config.items():
                env_key = f"IRIS_{key.upper()}"
                os.environ[env_key] = str(value)
        
        connection = get_iris_dbapi_connection()
        return DBAPIConnectorWrapper(connection)
    
    @staticmethod
    def _create_jdbc_connection(**config) -> IRISConnectorInterface:
        """Create JDBC connection (enterprise/legacy)."""
        from .iris_connector import get_real_iris_connection
        
        connection = get_real_iris_connection(config)
        return JDBCConnectorWrapper(connection)
    
    @staticmethod
    def from_environment() -> IRISConnectorInterface:
        """Create connection using environment variables with DBAPI default."""
        return ConnectionFactory.create_connection("dbapi")
```

## Fix 5: Update Benchmark Script to Use Factory

### Update [`scripts/run_rag_benchmarks.py`](scripts/run_rag_benchmarks.py:400)
```python
# Replace setup_database_connection function
def setup_database_connection(args) -> Optional[Any]:
    """Set up and verify the database connection using factory pattern."""
    logger.info("Establishing connection to IRIS database...")
    
    try:
        from common.connection_factory import ConnectionFactory
        
        if args.use_mock:
            # Keep existing mock logic
            logger.info("Using mock IRIS connection as requested by --use-mock flag.")
            # ... existing mock code ...
        else:
            # Use factory with DBAPI as default (user preference)
            connection_config = {}
            
            # Build config from command line args if provided
            if hasattr(args, 'iris_host') and args.iris_host:
                connection_config.update({
                    'host': args.iris_host,
                    'port': args.iris_port,
                    'namespace': args.iris_namespace,
                    'user': args.iris_user,
                    'password': args.iris_password
                })
            
            # Create DBAPI connection (user preference)
            iris_conn = ConnectionFactory.create_connection("dbapi", **connection_config)
            
        return iris_conn
        
    except Exception as e:
        logger.error(f"Failed to create database connection: {e}")
        return None
```

## Fix 6: Update Core Pipeline Constructors

### Standardize Constructor Pattern
```python
# Update all core_pipelines/*.py files to use this pattern:

from typing import Union, Any, Callable, List
from common.connector_interface import IRISConnectorInterface

class StandardRAGPipeline:
    def __init__(self, 
                 iris_connector: Union[IRISConnectorInterface, Any],  # Support both during transition
                 embedding_func: Callable[[List[str]], List[List[float]]],
                 llm_func: Callable[[str], str],
                 schema: str = "RAG",
                 **kwargs):
        self.iris_connector = iris_connector
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        self.schema = schema
        
        # Ensure we have a standardized interface
        if not isinstance(iris_connector, IRISConnectorInterface):
            logger.warning("Using non-standard connector interface")
```

## Implementation Order

### Phase 1: Immediate (Day 1)
1. ✅ Create `common/connector_interface.py`
2. ✅ Create `common/connection_factory.py`
3. ✅ Update `scripts/run_rag_benchmarks.py` to use factory

### Phase 2: Core Fixes (Day 2)
1. Update `core_pipelines/basic_rag_pipeline.py` imports
2. Update `core_pipelines/hyde_pipeline.py` imports
3. Standardize remaining core pipeline imports

### Phase 3: Testing (Day 3)
1. Test benchmark script with DBAPI connection
2. Verify all pipelines work with new interface
3. Run integration tests

## Validation Commands

### Test DBAPI Connection
```bash
python -c "
from common.connection_factory import ConnectionFactory
conn = ConnectionFactory.create_connection('dbapi')
print('DBAPI connection successful')
conn.close()
"
```

### Test Benchmark Script
```bash
python scripts/run_rag_benchmarks.py \
    --techniques basic_rag hyde \
    --num-docs 100 \
    --num-queries 2 \
    --use-mock
```

### Test Pipeline Integration
```bash
python -c "
from common.connection_factory import ConnectionFactory
from core_pipelines.basic_rag_pipeline import BasicRAGPipeline
from common.utils import get_embedding_func, get_llm_func

conn = ConnectionFactory.create_connection('dbapi')
embed_func = get_embedding_func('stub')
llm_func = get_llm_func('stub')

pipeline = BasicRAGPipeline(conn, embed_func, llm_func)
result = pipeline.run('test query')
print('Pipeline integration successful')
"
```

## Expected Outcomes

After implementing these fixes:

1. ✅ **Benchmark script runs without connection errors**
2. ✅ **DBAPI is used as default connection method** (user preference)
3. ✅ **All pipelines use consistent connection interface**
4. ✅ **JDBC remains available for enterprise scenarios**
5. ✅ **Type safety improved with standardized interfaces**
6. ✅ **Easier testing with mock implementations**

## Rollback Plan

If issues arise, rollback order:
1. Revert benchmark script changes
2. Revert core pipeline imports
3. Remove new interface files
4. Return to original JDBC-based approach

## Success Metrics

- [ ] Benchmark script executes without import errors
- [ ] All 6 core pipelines instantiate successfully
- [ ] DBAPI connection works end-to-end
- [ ] JDBC connection still available as fallback
- [ ] No regression in existing functionality