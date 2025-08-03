#!/usr/bin/env python3
"""
SQL Audit Trail Logger

This module provides comprehensive SQL operation logging to track real database
commands vs mocked operations, enabling correlation with IRIS audit logs.

This addresses the critical testing anti-pattern of "mocking to success"
without real database validation.
"""

import logging
import time
import json
import hashlib
import threading
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)


@dataclass
class SQLOperation:
    """Record of a SQL operation for audit trail."""
    operation_id: str
    timestamp: datetime
    sql_statement: str
    parameters: List[Any]
    operation_type: str  # 'SELECT', 'INSERT', 'UPDATE', 'DELETE', etc.
    execution_context: str  # 'real_database', 'mocked', 'test'
    pipeline_name: Optional[str]
    test_name: Optional[str]
    execution_time_ms: Optional[float]
    rows_affected: Optional[int]
    result_count: Optional[int]
    error: Optional[str]
    stack_trace: Optional[str]


class SQLAuditLogger:
    """
    Comprehensive SQL audit trail logger.
    
    Tracks all SQL operations to distinguish real database commands from mocks
    and enable correlation with IRIS audit logs.
    """
    
    def __init__(self, log_file_path: str = "sql_audit_trail.jsonl"):
        self.log_file_path = log_file_path
        self.operations: List[SQLOperation] = []
        self._lock = threading.Lock()
        self._current_context = threading.local()
        
        # Setup file logger for audit trail
        self._setup_file_logger()
        
        logger.info(f"SQL Audit Logger initialized - logging to {log_file_path}")
    
    def _setup_file_logger(self):
        """Setup dedicated file logger for SQL audit trail."""
        self.file_logger = logging.getLogger('sql_audit_trail')
        self.file_logger.setLevel(logging.INFO)
        
        # Remove any existing handlers
        self.file_logger.handlers.clear()
        
        # Add file handler for audit trail
        file_handler = logging.FileHandler(self.log_file_path, mode='a')
        file_handler.setLevel(logging.INFO)
        
        # JSON formatter for structured logging
        formatter = logging.Formatter('%(message)s')
        file_handler.setFormatter(formatter)
        
        self.file_logger.addHandler(file_handler)
        self.file_logger.propagate = False
    
    @contextmanager
    def set_context(self, context: str, pipeline_name: str = None, test_name: str = None):
        """
        Set execution context for SQL operations.
        
        Args:
            context: 'real_database', 'mocked', 'test'
            pipeline_name: Name of the pipeline executing the operation
            test_name: Name of the test if in test context
        """
        # Store previous context
        previous_context = getattr(self._current_context, 'context', None)
        previous_pipeline = getattr(self._current_context, 'pipeline_name', None)
        previous_test = getattr(self._current_context, 'test_name', None)
        
        # Set new context
        self._current_context.context = context
        self._current_context.pipeline_name = pipeline_name
        self._current_context.test_name = test_name
        
        try:
            yield
        finally:
            # Restore previous context
            self._current_context.context = previous_context
            self._current_context.pipeline_name = previous_pipeline
            self._current_context.test_name = previous_test
    
    def log_sql_operation(self, 
                         sql_statement: str,
                         parameters: List[Any] = None,
                         execution_time_ms: float = None,
                         rows_affected: int = None,
                         result_count: int = None,
                         error: str = None) -> str:
        """
        Log a SQL operation to the audit trail.
        
        Returns:
            operation_id: Unique identifier for this operation
        """
        import traceback
        
        # Generate unique operation ID
        timestamp = datetime.utcnow()
        content_hash = hashlib.md5(f"{sql_statement}{parameters}".encode()).hexdigest()[:8]
        operation_id = f"sql_{timestamp.strftime('%Y%m%d_%H%M%S')}_{content_hash}"
        
        # Get current context
        context = getattr(self._current_context, 'context', 'unknown')
        pipeline_name = getattr(self._current_context, 'pipeline_name', None)
        test_name = getattr(self._current_context, 'test_name', None)
        
        # Determine operation type
        operation_type = sql_statement.strip().split()[0].upper() if sql_statement else 'UNKNOWN'
        
        # Create operation record
        operation = SQLOperation(
            operation_id=operation_id,
            timestamp=timestamp,
            sql_statement=sql_statement,
            parameters=parameters or [],
            operation_type=operation_type,
            execution_context=context,
            pipeline_name=pipeline_name,
            test_name=test_name,
            execution_time_ms=execution_time_ms,
            rows_affected=rows_affected,
            result_count=result_count,
            error=error,
            stack_trace=traceback.format_stack() if error else None
        )
        
        # Store in memory and log to file
        with self._lock:
            self.operations.append(operation)
            
            # Log as JSON for easy parsing
            log_entry = asdict(operation)
            log_entry['timestamp'] = timestamp.isoformat()
            self.file_logger.info(json.dumps(log_entry))
        
        # Console log for immediate visibility
        context_emoji = {
            'real_database': '🔴',
            'mocked': '🟡', 
            'test': '🔵',
            'unknown': '⚫'
        }.get(context, '❓')
        
        logger.info(f"{context_emoji} SQL AUDIT [{operation_id}] {context.upper()}: {operation_type} "
                   f"({pipeline_name or 'unknown'}) - {sql_statement[:100]}...")
        
        if error:
            logger.error(f"❌ SQL ERROR [{operation_id}]: {error}")
        
        return operation_id
    
    def get_operations_by_context(self, context: str) -> List[SQLOperation]:
        """Get all operations for a specific context."""
        with self._lock:
            return [op for op in self.operations if op.execution_context == context]
    
    def get_operations_by_pipeline(self, pipeline_name: str) -> List[SQLOperation]:
        """Get all operations for a specific pipeline."""
        with self._lock:
            return [op for op in self.operations if op.pipeline_name == pipeline_name]
    
    def get_operations_by_test(self, test_name: str) -> List[SQLOperation]:
        """Get all operations for a specific test."""
        with self._lock:
            return [op for op in self.operations if op.test_name == test_name]
    
    def generate_audit_report(self) -> Dict[str, Any]:
        """Generate comprehensive audit report."""
        with self._lock:
            total_ops = len(self.operations)
            
            if total_ops == 0:
                return {"status": "no_operations", "total_operations": 0}
            
            # Group by context
            by_context = {}
            for op in self.operations:
                context = op.execution_context
                if context not in by_context:
                    by_context[context] = {
                        "count": 0,
                        "operations": [],
                        "pipelines": set(),
                        "tests": set(),
                        "errors": 0
                    }
                
                by_context[context]["count"] += 1
                by_context[context]["operations"].append(op.operation_id)
                if op.pipeline_name:
                    by_context[context]["pipelines"].add(op.pipeline_name)
                if op.test_name:
                    by_context[context]["tests"].add(op.test_name)
                if op.error:
                    by_context[context]["errors"] += 1
            
            # Convert sets to lists for JSON serialization
            for context_data in by_context.values():
                context_data["pipelines"] = list(context_data["pipelines"])
                context_data["tests"] = list(context_data["tests"])
            
            real_db_ops = by_context.get('real_database', {}).get('count', 0)
            mocked_ops = by_context.get('mocked', {}).get('count', 0)
            
            return {
                "status": "report_generated",
                "total_operations": total_ops,
                "real_database_operations": real_db_ops,
                "mocked_operations": mocked_ops,
                "real_vs_mock_ratio": real_db_ops / max(mocked_ops, 1),
                "by_context": by_context,
                "log_file": self.log_file_path,
                "generated_at": datetime.utcnow().isoformat()
            }
    
    def clear_audit_trail(self):
        """Clear the audit trail (for testing)."""
        with self._lock:
            self.operations.clear()
        
        logger.info("SQL audit trail cleared")


# Global singleton instance
_sql_audit_logger: Optional[SQLAuditLogger] = None


def get_sql_audit_logger() -> SQLAuditLogger:
    """Get the global SQL audit logger instance."""
    global _sql_audit_logger
    if _sql_audit_logger is None:
        _sql_audit_logger = SQLAuditLogger()
    return _sql_audit_logger


def log_sql_execution(sql: str, params: List[Any] = None, **kwargs) -> str:
    """Convenience function to log SQL execution."""
    audit_logger = get_sql_audit_logger()
    return audit_logger.log_sql_operation(sql, params, **kwargs)


@contextmanager
def sql_audit_context(context: str, pipeline_name: str = None, test_name: str = None):
    """Context manager for SQL audit logging."""
    audit_logger = get_sql_audit_logger()
    with audit_logger.set_context(context, pipeline_name, test_name):
        yield audit_logger


# Decorators for automatic SQL audit logging
def audit_real_database(pipeline_name: str = None):
    """Decorator to mark functions as using real database operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with sql_audit_context('real_database', pipeline_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def audit_mocked_database(test_name: str = None):
    """Decorator to mark functions as using mocked database operations."""
    def decorator(func):
        def wrapper(*args, **kwargs):
            with sql_audit_context('mocked', test_name=test_name):
                return func(*args, **kwargs)
        return wrapper
    return decorator


if __name__ == "__main__":
    # Test the audit logger
    audit_logger = get_sql_audit_logger()
    
    # Simulate different contexts
    with audit_logger.set_context('real_database', 'BasicRAG', None):
        audit_logger.log_sql_operation(
            "SELECT * FROM RAG.SourceDocuments WHERE embedding IS NOT NULL",
            [],
            execution_time_ms=45.2,
            result_count=1000
        )
    
    with audit_logger.set_context('mocked', 'HybridIFind', 'test_ifind_working_path'):
        audit_logger.log_sql_operation(
            "SELECT doc_id, text_content, score FROM RAG.SourceDocuments WHERE $FIND(text_content, ?)",
            ["diabetes"],
            execution_time_ms=0.1,
            result_count=3
        )
    
    # Generate report
    report = audit_logger.generate_audit_report()
    print(json.dumps(report, indent=2))