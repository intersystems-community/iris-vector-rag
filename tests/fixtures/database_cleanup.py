"""
Database cleanup handler for test isolation.

Provides reliable cleanup of test data even after test failures.
Implements T016 from Feature 028.
"""

import time
from typing import Optional

from tests.fixtures.database_state import TestDatabaseState, TestStateRegistry


class DatabaseCleanupHandler:
    """
    Handles cleanup of test data from database.

    Ensures test isolation by removing all data associated with a test run.
    Performance target: <100ms per cleanup (NFR-002).
    """

    def __init__(self, connection, test_run_id: str):
        """
        Initialize cleanup handler.

        Args:
            connection: IRIS database connection
            test_run_id: Unique identifier for this test run
        """
        self.connection = connection
        self.test_run_id = test_run_id

    def cleanup(self) -> None:
        """
        Remove all test data associated with this test run.

        Deletes from all RAG tables where test_run_id matches.
        Idempotent - safe to call multiple times.

        Raises:
            Exception: If cleanup operations fail
        """
        cursor = self.connection.cursor()

        try:
            # Delete in order to respect foreign key constraints
            # Relationships reference Entities
            # DocumentChunks reference SourceDocuments
            # Entities are standalone
            # SourceDocuments are standalone

            cleanup_tables = [
                'RAG.Relationships',
                'RAG.DocumentChunks',
                'RAG.Entities',
                'RAG.SourceDocuments'
            ]

            for table in cleanup_tables:
                delete_sql = f"DELETE FROM {table} WHERE test_run_id = ?"
                cursor.execute(delete_sql, [self.test_run_id])

            self.connection.commit()

        except Exception as e:
            self.connection.rollback()
            raise Exception(f"Cleanup failed for test_run_id {self.test_run_id}: {str(e)}")

    def cleanup_timed(self) -> float:
        """
        Execute cleanup and return execution time.

        Returns:
            Execution time in milliseconds

        Raises:
            Exception: If cleanup fails or exceeds 100ms limit
        """
        start = time.time()
        self.cleanup()
        duration_ms = (time.time() - start) * 1000

        if duration_ms >= 100.0:
            raise Exception(
                f"Cleanup took {duration_ms:.2f}ms, exceeds 100ms limit (NFR-002)"
            )

        return duration_ms

    def verify_cleanup(self) -> bool:
        """
        Verify all test data has been removed.

        Returns:
            True if no data remains for this test_run_id, False otherwise
        """
        cursor = self.connection.cursor()

        tables_to_check = [
            'RAG.SourceDocuments',
            'RAG.DocumentChunks',
            'RAG.Entities',
            'RAG.Relationships'
        ]

        for table in tables_to_check:
            cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE test_run_id = ?", [self.test_run_id])
            count = cursor.fetchone()[0]

            if count > 0:
                return False

        return True

    def get_remaining_count(self) -> int:
        """
        Get count of remaining test data across all tables.

        Returns:
            Total number of rows with this test_run_id
        """
        cursor = self.connection.cursor()
        total = 0

        tables_to_check = [
            'RAG.SourceDocuments',
            'RAG.DocumentChunks',
            'RAG.Entities',
            'RAG.Relationships'
        ]

        for table in tables_to_check:
            cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE test_run_id = ?", [self.test_run_id])
            total += cursor.fetchone()[0]

        return total


class CleanupRegistry:
    """
    Global registry for tracking cleanup handlers.

    Ensures all test data is cleaned up even if tests fail.
    """

    _instance: Optional['CleanupRegistry'] = None
    _handlers: dict = {}

    def __new__(cls) -> 'CleanupRegistry':
        """Singleton pattern for global registry."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._handlers = {}
        return cls._instance

    def register_handler(self, test_run_id: str, handler: DatabaseCleanupHandler) -> None:
        """
        Register a cleanup handler.

        Args:
            test_run_id: ID of test run
            handler: DatabaseCleanupHandler instance
        """
        self._handlers[test_run_id] = handler

    def execute_cleanup(self, test_run_id: str) -> None:
        """
        Execute cleanup for a specific test run.

        Args:
            test_run_id: ID of test run to clean up
        """
        if test_run_id in self._handlers:
            handler = self._handlers[test_run_id]
            handler.cleanup()
            del self._handlers[test_run_id]

    def execute_all_cleanups(self) -> None:
        """Execute all registered cleanups (emergency cleanup)."""
        for test_run_id in list(self._handlers.keys()):
            self.execute_cleanup(test_run_id)

    def __repr__(self) -> str:
        """String representation for debugging."""
        return f"CleanupRegistry({len(self._handlers)} active handlers)"
