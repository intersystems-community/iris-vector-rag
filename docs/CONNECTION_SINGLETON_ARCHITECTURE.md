# Connection Singleton Architecture

## Problem Statement

The RAG pipeline integration tests were experiencing transaction isolation issues where:
1. Documents were successfully inserted into the database
2. Vector search operations returned 0 results
3. This caused all pipeline queries to fail

## Root Cause Analysis

The issue was caused by multiple database connections being created:
- Test setup used `get_iris_connection()` → Connection A
- IRISVectorStore created its own connection manager → Connection B
- Documents inserted via Connection B were not visible to queries via Connection A

## Solution: Connection Singleton Pattern

### Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    Connection Singleton Architecture             │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  ┌─────────────────┐    ┌─────────────────┐    ┌──────────────┐ │
│  │   Test Setup    │    │ IRISVectorStore │    │ SchemaManager│ │
│  │                 │    │                 │    │              │ │
│  │ get_shared_iris │    │ get_shared_iris │    │ shared conn  │ │
│  │ _connection()   │    │ _connection()   │    │ via wrapper  │ │
│  └─────────┬───────┘    └─────────┬───────┘    └──────┬───────┘ │
│            │                      │                   │         │
│            └──────────────────────┼───────────────────┘         │
│                                   │                             │
│                    ┌──────────────▼──────────────┐              │
│                    │    ConnectionSingleton      │              │
│                    │                             │              │
│                    │  - Thread-safe singleton   │              │
│                    │  - Single IRIS connection  │              │
│                    │  - Connection reset support│              │
│                    │  - Config management       │              │
│                    └──────────────┬──────────────┘              │
│                                   │                             │
│                    ┌──────────────▼──────────────┐              │
│                    │     IRIS Database           │              │
│                    │                             │              │
│                    │  - Consistent transactions  │              │
│                    │  - No isolation issues     │              │
│                    │  - Shared connection state │              │
│                    └─────────────────────────────┘              │
└─────────────────────────────────────────────────────────────────┘
```

### Key Components

#### 1. ConnectionSingleton Class
- **Location**: `common/connection_singleton.py`
- **Purpose**: Ensures all components use the same database connection
- **Features**:
  - Thread-safe singleton pattern
  - Connection reset capability for testing
  - Configuration management
  - Lazy connection initialization

#### 2. IRISVectorStore Modifications
- **Location**: `iris_rag/storage/vector_store_iris.py`
- **Changes**:
  - Uses `get_shared_iris_connection()` when no connection manager provided
  - Maintains backward compatibility with explicit connection managers
  - Creates mock connection manager for interface consistency

#### 3. Test Integration
- **Location**: `tests/test_all_pipelines_chunking_integration.py`
- **Changes**:
  - Uses `get_shared_iris_connection()` in test setup
  - Calls `reset_shared_connection()` for clean test state
  - Ensures all components use the same connection

### Connection Flow

```
Test Execution Flow:
1. reset_shared_connection() → Clean slate
2. get_shared_iris_connection() → Test gets Connection X
3. IRISVectorStore() → Uses same Connection X
4. SchemaManager() → Uses same Connection X via wrapper
5. All operations → Same transaction context
6. Vector search → Sees inserted documents ✓
```

### Benefits

1. **Transaction Consistency**: All operations use the same connection
2. **Data Visibility**: Inserted documents are immediately visible to queries
3. **Test Isolation**: Connection reset ensures clean test state
4. **Backward Compatibility**: Existing code continues to work
5. **Thread Safety**: Singleton pattern with proper locking
6. **Performance**: Reduced connection overhead

### Implementation Details

#### Connection Singleton
```python
class ConnectionSingleton:
    _instance = None
    _lock = threading.Lock()
    _connection_manager = None
    _connection = None
    
    def get_connection(self, config=None):
        if self._connection is None:
            with self._lock:
                if self._connection is None:
                    self._connection_manager = IRISConnectionManager(config)
                    self._connection = self._connection_manager.get_connection()
        return self._connection
```

#### IRISVectorStore Integration
```python
def __init__(self, config_manager, connection_manager=None, ...):
    if connection_manager:
        self.connection_manager = connection_manager
        self._connection = self.connection_manager.get_connection()
    else:
        # Use shared connection singleton
        self._connection = get_shared_iris_connection(...)
        self.connection_manager = MockConnectionManager(self._connection)
```

#### Test Setup
```python
@pytest.fixture(autouse=True)
def setup_test_environment(self):
    reset_shared_connection()  # Clean state
    self.connection = get_shared_iris_connection()  # Shared connection
    self.connection_manager = ConnectionManager(self.connection)
```

### Migration Strategy

1. **Phase 1**: Implement connection singleton (✓ Complete)
2. **Phase 2**: Update IRISVectorStore to use singleton (✓ Complete)
3. **Phase 3**: Update tests to use shared connection (✓ Complete)
4. **Phase 4**: Validate transaction consistency (In Progress)
5. **Phase 5**: Roll out to all pipeline components

### Testing Validation

The solution addresses the critical issue where:
- **Before**: Documents inserted but not visible → 0 search results
- **After**: Documents inserted and immediately visible → Successful queries

### Future Considerations

1. **Connection Pooling**: Consider connection pooling for production use
2. **Configuration Management**: Centralized connection configuration
3. **Monitoring**: Connection health and performance monitoring
4. **Error Handling**: Robust connection failure recovery

## Conclusion

The Connection Singleton Architecture resolves the transaction isolation issue by ensuring all components use the same database connection, enabling proper data visibility and transaction consistency across the RAG pipeline integration tests.