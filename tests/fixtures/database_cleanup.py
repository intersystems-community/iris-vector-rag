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
        Falls back to doc_id prefix pattern if test_run_id column doesn't exist.
        Idempotent - safe to call multiple times.

        Raises:
            Exception: If cleanup operations fail
        """
        cursor = self.connection.cursor()

        try:
            # Delete in order to respect foreign key constraints
            # EntityRelationships reference Entities
            # DocumentChunks reference SourceDocuments
            # Entities are standalone
            # SourceDocuments are standalone

            cleanup_tables = [
                'RAG.EntityRelationships',
                'RAG.DocumentChunks',
                'RAG.Entities',
                'RAG.SourceDocuments'
            ]

            for table in cleanup_tables:
                # Try test_run_id column first, fall back to doc_id prefix pattern
                try:
                    delete_sql = f"DELETE FROM {table} WHERE test_run_id = ?"
                    cursor.execute(delete_sql, [self.test_run_id])
                    # If we got here, column exists - no fallback needed
                except Exception as column_error:
                    error_str = str(column_error)

                    # Skip if table doesn't exist (check for "Table 'X' not found" specifically)
                    if 'Table' in error_str and 'not found' in error_str:
                        continue

                    # If test_run_id column doesn't exist, use doc_id prefix pattern
                    if 'test_run_id' in error_str or 'Field' in error_str:
                        try:
                            # Fallback strategy: use doc_id LIKE pattern for SourceDocuments/DocumentChunks
                            # entity_id LIKE pattern for Entities
                            # relationship_id LIKE pattern for EntityRelationships
                            if table == 'RAG.SourceDocuments':
                                delete_sql = f"DELETE FROM {table} WHERE doc_id LIKE ?"
                                cursor.execute(delete_sql, [f"{self.test_run_id}%"])
                            elif table == 'RAG.DocumentChunks':
                                delete_sql = f"DELETE FROM {table} WHERE chunk_id LIKE ?"
                                cursor.execute(delete_sql, [f"{self.test_run_id}%"])
                            elif table == 'RAG.Entities':
                                delete_sql = f"DELETE FROM {table} WHERE entity_id LIKE ?"
                                cursor.execute(delete_sql, [f"{self.test_run_id}%"])
                            elif table == 'RAG.EntityRelationships':
                                delete_sql = f"DELETE FROM {table} WHERE relationship_id LIKE ?"
                                cursor.execute(delete_sql, [f"{self.test_run_id}%"])
                        except Exception as fallback_error:
                            # Skip if table doesn't exist in fallback
                            if 'Table' in str(fallback_error) and 'not found' in str(fallback_error):
                                continue
                            raise
                    else:
                        raise

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
            'RAG.EntityRelationships'
        ]

        for table in tables_to_check:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE test_run_id = ?", [self.test_run_id])
                count = cursor.fetchone()[0]
            except Exception as column_error:
                error_str = str(column_error)

                # Skip if table doesn't exist (check for "Table 'X' not found" specifically)
                if 'Table' in error_str and 'not found' in error_str:
                    continue

                # Fallback to ID prefix pattern if test_run_id column doesn't exist
                if 'test_run_id' in error_str or 'Field' in error_str:
                    try:
                        if table == 'RAG.SourceDocuments':
                            cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE doc_id LIKE ?", [f"{self.test_run_id}%"])
                        elif table == 'RAG.DocumentChunks':
                            cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE chunk_id LIKE ?", [f"{self.test_run_id}%"])
                        elif table == 'RAG.Entities':
                            cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE entity_id LIKE ?", [f"{self.test_run_id}%"])
                        elif table == 'RAG.EntityRelationships':
                            cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE relationship_id LIKE ?", [f"{self.test_run_id}%"])
                        count = cursor.fetchone()[0]
                    except Exception as fallback_error:
                        # Skip if table doesn't exist in fallback
                        if 'Table' in str(fallback_error) and 'not found' in str(fallback_error):
                            continue
                        raise
                else:
                    raise

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
            'RAG.EntityRelationships'
        ]

        for table in tables_to_check:
            try:
                cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE test_run_id = ?", [self.test_run_id])
                total += cursor.fetchone()[0]
            except Exception as column_error:
                error_str = str(column_error)

                # Skip if table doesn't exist (check for "Table 'X' not found" specifically)
                if 'Table' in error_str and 'not found' in error_str:
                    continue

                # Fallback to ID prefix pattern if test_run_id column doesn't exist
                if 'test_run_id' in error_str or 'Field' in error_str:
                    try:
                        if table == 'RAG.SourceDocuments':
                            cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE doc_id LIKE ?", [f"{self.test_run_id}%"])
                        elif table == 'RAG.DocumentChunks':
                            cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE chunk_id LIKE ?", [f"{self.test_run_id}%"])
                        elif table == 'RAG.Entities':
                            cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE entity_id LIKE ?", [f"{self.test_run_id}%"])
                        elif table == 'RAG.EntityRelationships':
                            cursor.execute(f"SELECT COUNT(*) FROM {table} WHERE relationship_id LIKE ?", [f"{self.test_run_id}%"])
                        total += cursor.fetchone()[0]
                    except Exception as fallback_error:
                        # Skip if table doesn't exist in fallback
                        if 'Table' in str(fallback_error) and 'not found' in str(fallback_error):
                            continue
                        raise
                else:
                    raise

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
