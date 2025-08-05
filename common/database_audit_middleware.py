#!/usr/bin/env python3
"""
Database Audit Middleware

This middleware intercepts and logs all database operations to provide
a complete audit trail of real SQL commands vs mocked operations.

Integrates with IRIS connection managers to capture actual database activity.
"""

import logging
import time
import inspect
from typing import Any, List, Dict, Optional, Callable, Union
from functools import wraps

from .sql_audit_logger import get_sql_audit_logger, log_sql_execution

logger = logging.getLogger(__name__)


class AuditableCursor:
    """
    Wrapper for database cursor that logs all SQL operations.
    
    This class intercepts execute(), fetchall(), fetchone(), etc. to provide
    complete audit trail of actual database operations.
    """
    
    def __init__(self, original_cursor, connection_type: str = "unknown"):
        self.original_cursor = original_cursor
        self.connection_type = connection_type
        self.audit_logger = get_sql_audit_logger()
        
        # Track current operation for correlation
        self._current_operation_id = None
        self._current_sql = None
        self._current_params = None
        self._operation_start_time = None
    
    def execute(self, sql: str, parameters: Any = None) -> Any:
        """Execute SQL with audit logging."""
        self._operation_start_time = time.time()
        self._current_sql = sql
        self._current_params = parameters or []
        
        # Log the SQL operation start
        self._current_operation_id = self.audit_logger.log_sql_operation(
            sql_statement=sql,
            parameters=self._current_params,
        )
        
        logger.debug(f"🔴 REAL SQL EXECUTION [{self._current_operation_id}]: {sql[:100]}...")
        
        try:
            # Execute the actual SQL - handle both parameterized and non-parameterized calls
            if parameters is None:
                result = self.original_cursor.execute(sql)
            else:
                result = self.original_cursor.execute(sql, parameters)
            
            # Log successful execution
            execution_time = (time.time() - self._operation_start_time) * 1000
            self.audit_logger.log_sql_operation(
                sql_statement=sql,
                parameters=self._current_params,
                execution_time_ms=execution_time,
                rows_affected=getattr(self.original_cursor, 'rowcount', None)
            )
            
            return result
            
        except Exception as e:
            # Log failed execution
            execution_time = (time.time() - self._operation_start_time) * 1000
            self.audit_logger.log_sql_operation(
                sql_statement=sql,
                parameters=self._current_params,
                execution_time_ms=execution_time,
                error=str(e)
            )
            
            logger.error(f"❌ SQL EXECUTION FAILED [{self._current_operation_id}]: {e}")
            raise
    
    def fetchall(self) -> List[Any]:
        """Fetch all results with audit logging."""
        try:
            results = self.original_cursor.fetchall()
            
            # Update the operation log with result count
            if self._current_operation_id:
                execution_time = (time.time() - self._operation_start_time) * 1000
                self.audit_logger.log_sql_operation(
                    sql_statement=self._current_sql,
                    parameters=self._current_params,
                    execution_time_ms=execution_time,
                    result_count=len(results) if results else 0
                )
            
            logger.debug(f"🔴 REAL SQL FETCHALL [{self._current_operation_id}]: {len(results) if results else 0} rows")
            return results
            
        except Exception as e:
            logger.error(f"❌ SQL FETCHALL FAILED [{self._current_operation_id}]: {e}")
            raise
    
    def fetchone(self) -> Any:
        """Fetch one result with audit logging."""
        try:
            result = self.original_cursor.fetchone()
            
            # Update the operation log
            if self._current_operation_id:
                execution_time = (time.time() - self._operation_start_time) * 1000
                self.audit_logger.log_sql_operation(
                    sql_statement=self._current_sql,
                    parameters=self._current_params,
                    execution_time_ms=execution_time,
                    result_count=1 if result else 0
                )
            
            logger.debug(f"🔴 REAL SQL FETCHONE [{self._current_operation_id}]: {'1 row' if result else 'no rows'}")
            return result
            
        except Exception as e:
            logger.error(f"❌ SQL FETCHONE FAILED [{self._current_operation_id}]: {e}")
            raise
    
    def fetchmany(self, size: int = None) -> List[Any]:
        """Fetch many results with audit logging."""
        try:
            results = self.original_cursor.fetchmany(size)
            
            # Update the operation log
            if self._current_operation_id:
                execution_time = (time.time() - self._operation_start_time) * 1000
                self.audit_logger.log_sql_operation(
                    sql_statement=self._current_sql,
                    parameters=self._current_params,
                    execution_time_ms=execution_time,
                    result_count=len(results) if results else 0
                )
            
            logger.debug(f"🔴 REAL SQL FETCHMANY [{self._current_operation_id}]: {len(results) if results else 0} rows")
            return results
            
        except Exception as e:
            logger.error(f"❌ SQL FETCHMANY FAILED [{self._current_operation_id}]: {e}")
            raise
    
    def close(self):
        """Close cursor with audit logging."""
        logger.debug(f"🔴 REAL SQL CURSOR CLOSE [{self._current_operation_id}]")
        return self.original_cursor.close()
    
    def __getattr__(self, name):
        """Delegate other methods to the original cursor."""
        return getattr(self.original_cursor, name)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


class AuditableConnection:
    """
    Wrapper for database connection that provides auditable cursors.
    """
    
    def __init__(self, original_connection, connection_type: str = "IRIS"):
        self.original_connection = original_connection
        self.connection_type = connection_type
        
        logger.info(f"🔴 REAL DATABASE CONNECTION CREATED: {connection_type}")
    
    def cursor(self) -> AuditableCursor:
        """Create an auditable cursor."""
        original_cursor = self.original_connection.cursor()
        return AuditableCursor(original_cursor, self.connection_type)
    
    def commit(self):
        """Commit transaction with audit logging."""
        logger.info(f"🔴 REAL DATABASE COMMIT: {self.connection_type}")
        return self.original_connection.commit()
    
    def rollback(self):
        """Rollback transaction with audit logging."""
        logger.warning(f"🔴 REAL DATABASE ROLLBACK: {self.connection_type}")
        return self.original_connection.rollback()
    
    def close(self):
        """Close connection with audit logging."""
        logger.info(f"🔴 REAL DATABASE CONNECTION CLOSED: {self.connection_type}")
        return self.original_connection.close()
    
    def __getattr__(self, name):
        """Delegate other methods to the original connection."""
        return getattr(self.original_connection, name)
    
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def audit_database_connection(connection_factory: Callable, connection_type: str = "IRIS"):
    """
    Decorator to wrap connection factory functions with audit logging.
    
    Usage:
        @audit_database_connection
        def get_iris_connection():
            return iris.connect(...)
    """
    @wraps(connection_factory)
    def wrapper(*args, **kwargs):
        # Get the original connection
        original_connection = connection_factory(*args, **kwargs)
        
        # Wrap it with auditing
        auditable_connection = AuditableConnection(original_connection, connection_type)
        
        return auditable_connection
    
    return wrapper


def patch_iris_connection_manager():
    """
    Monkey patch the IRIS connection manager to add audit logging.
    
    This should be called at the start of tests to ensure all database
    operations are logged.
    """
    try:
        # Patch the main connection function used by ConnectionManager
        from common.iris_dbapi_connector import get_iris_dbapi_connection as original_dbapi_connection
        
        # Create auditable version for DBAPI
        @audit_database_connection  
        def auditable_dbapi_connection(*args, **kwargs):
            return original_dbapi_connection(*args, **kwargs)
        
        # Monkey patch the DBAPI connector module (used by ConnectionManager)
        import common.iris_dbapi_connector
        common.iris_dbapi_connector.get_iris_dbapi_connection = auditable_dbapi_connection
        
        # Also patch the general connection manager for backward compatibility
        from common.iris_connection_manager import get_iris_connection as original_get_iris_connection
        
        # Create auditable version
        @audit_database_connection
        def auditable_get_iris_connection(*args, **kwargs):
            return original_get_iris_connection(*args, **kwargs)
        
        # Monkey patch the module
        import common.iris_connection_manager
        common.iris_connection_manager.get_iris_connection = auditable_get_iris_connection
        
        logger.info("✅ IRIS connection manager patched for SQL audit logging")
        
    except ImportError as e:
        logger.warning(f"Could not patch IRIS connection manager: {e}")


def mock_operation_tracker(original_method: Callable):
    """
    Decorator to track mocked database operations.
    
    This helps distinguish between real and mocked operations in tests.
    """
    @wraps(original_method)
    def wrapper(*args, **kwargs):
        # Get the mock call info
        method_name = original_method.__name__
        
        # Log the mocked operation
        audit_logger = get_sql_audit_logger()
        operation_id = audit_logger.log_sql_operation(
            sql_statement=f"MOCKED_{method_name.upper()}",
            parameters=list(args[1:]) if len(args) > 1 else [],
            execution_time_ms=0.001,  # Mocks are fast
            result_count=len(kwargs.get('return_value', [])) if 'return_value' in kwargs else None
        )
        
        logger.debug(f"🟡 MOCKED OPERATION [{operation_id}]: {method_name}")
        
        # Call the original mocked method
        return original_method(*args, **kwargs)
    
    return wrapper


class DatabaseOperationCounter:
    """
    Utility to count and categorize database operations during test execution.
    """
    
    def __init__(self):
        self.reset()
    
    def reset(self):
        """Reset all counters."""
        self.real_operations = 0
        self.mocked_operations = 0
        self.operation_details = []
    
    def count_operations(self, test_name: str = None) -> Dict[str, Any]:
        """
        Count operations from the audit logger for a specific test.
        
        Returns:
            Dictionary with operation counts and analysis
        """
        audit_logger = get_sql_audit_logger()
        
        if test_name:
            operations = audit_logger.get_operations_by_test(test_name)
        else:
            operations = audit_logger.operations
        
        real_ops = [op for op in operations if op.execution_context == 'real_database']
        mocked_ops = [op for op in operations if op.execution_context == 'mocked']
        
        return {
            "total_operations": len(operations),
            "real_database_operations": len(real_ops),
            "mocked_operations": len(mocked_ops),
            "real_operations_detail": [
                {
                    "operation_id": op.operation_id,
                    "sql": op.sql_statement[:100] + "..." if len(op.sql_statement) > 100 else op.sql_statement,
                    "execution_time_ms": op.execution_time_ms,
                    "result_count": op.result_count
                }
                for op in real_ops
            ],
            "mocked_operations_detail": [
                {
                    "operation_id": op.operation_id,
                    "sql": op.sql_statement,
                    "test_name": op.test_name
                }
                for op in mocked_ops
            ],
            "test_isolation_score": len(real_ops) / max(len(mocked_ops), 1)  # Higher is better
        }


# Global instance for easy access
operation_counter = DatabaseOperationCounter()


if __name__ == "__main__":
    # Test the audit middleware
    print("Testing Database Audit Middleware...")
    
    # Simulate database operations
    audit_logger = get_sql_audit_logger()
    
    with audit_logger.set_context('real_database', 'BasicRAG'):
        audit_logger.log_sql_operation(
            "SELECT * FROM RAG.SourceDocuments WHERE doc_id = ?",
            ["test_doc_1"],
            execution_time_ms=15.3,
            result_count=1
        )
    
    with audit_logger.set_context('mocked', test_name='test_basic_functionality'):
        audit_logger.log_sql_operation(
            "MOCKED_EXECUTE",
            ["SELECT * FROM RAG.SourceDocuments"],
            execution_time_ms=0.001,
            result_count=3
        )
    
    # Generate analysis
    counter = DatabaseOperationCounter()
    analysis = counter.count_operations()
    
    print(f"Analysis: {analysis}")
    print(f"Real vs Mock ratio: {analysis['test_isolation_score']:.2f}")