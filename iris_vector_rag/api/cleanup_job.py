#!/usr/bin/env python3
"""
Database Cleanup Job for RAG API.

Implements FR-031: Automated cleanup of old logs and expired data.
Runs as a scheduled job to maintain database performance.
"""

import logging
import sys
from datetime import datetime, timedelta
from pathlib import Path

from common.connection_pool import IRISConnectionPool
from iris_vector_rag.api.main import load_config


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


class DatabaseCleanupJob:
    """
    Automated database cleanup job.

    Implements FR-031: Clean up old logs and expired data.
    """

    def __init__(self, connection_pool: IRISConnectionPool, retention_days: int = 30):
        """
        Initialize cleanup job.

        Args:
            connection_pool: IRIS connection pool
            retention_days: Number of days to retain logs
        """
        self.connection_pool = connection_pool
        self.retention_days = retention_days

    def cleanup_request_logs(self) -> int:
        """
        Clean up old API request logs.

        Returns:
            Number of rows deleted
        """
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)

        with self.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            # Delete old logs
            query = """
                DELETE FROM api_request_logs
                WHERE timestamp < ?
            """

            cursor.execute(query, (cutoff_date,))
            rows_deleted = cursor.rowcount

            conn.commit()

            logger.info(f"Deleted {rows_deleted} old request logs (older than {cutoff_date})")

            return rows_deleted

    def cleanup_rate_limit_history(self) -> int:
        """
        Clean up old rate limit history.

        Returns:
            Number of rows deleted
        """
        cutoff_date = datetime.utcnow() - timedelta(days=7)  # Keep 7 days for rate limit history

        with self.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            query = """
                DELETE FROM rate_limit_history
                WHERE window_end < ?
            """

            cursor.execute(query, (cutoff_date,))
            rows_deleted = cursor.rowcount

            conn.commit()

            logger.info(f"Deleted {rows_deleted} old rate limit records (older than {cutoff_date})")

            return rows_deleted

    def cleanup_websocket_sessions(self) -> int:
        """
        Clean up inactive WebSocket sessions.

        Returns:
            Number of rows deleted
        """
        cutoff_date = datetime.utcnow() - timedelta(hours=24)

        with self.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            # Delete inactive sessions older than 24 hours
            query = """
                DELETE FROM websocket_sessions
                WHERE is_active = 0
                AND last_activity_at < ?
            """

            cursor.execute(query, (cutoff_date,))
            rows_deleted = cursor.rowcount

            conn.commit()

            logger.info(f"Deleted {rows_deleted} old WebSocket sessions (inactive since {cutoff_date})")

            return rows_deleted

    def cleanup_expired_api_keys(self) -> int:
        """
        Deactivate expired API keys.

        Returns:
            Number of keys deactivated
        """
        now = datetime.utcnow()

        with self.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            # Deactivate expired keys
            query = """
                UPDATE api_keys
                SET is_active = 0
                WHERE expires_at IS NOT NULL
                AND expires_at < ?
                AND is_active = 1
            """

            cursor.execute(query, (now,))
            rows_updated = cursor.rowcount

            conn.commit()

            logger.info(f"Deactivated {rows_updated} expired API keys")

            return rows_updated

    def cleanup_old_upload_operations(self) -> int:
        """
        Clean up old completed/failed upload operations.

        Returns:
            Number of rows deleted
        """
        cutoff_date = datetime.utcnow() - timedelta(days=self.retention_days)

        with self.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            # Delete old completed/failed operations
            query = """
                DELETE FROM document_upload_operations
                WHERE status IN ('completed', 'failed')
                AND created_at < ?
            """

            cursor.execute(query, (cutoff_date,))
            rows_deleted = cursor.rowcount

            conn.commit()

            logger.info(f"Deleted {rows_deleted} old upload operations (older than {cutoff_date})")

            return rows_deleted

    def update_cleanup_status(self, job_name: str, status: str, rows_deleted: int, error_message: str = None):
        """
        Update cleanup job status.

        Args:
            job_name: Name of cleanup job
            status: Job status (success, failed, running)
            rows_deleted: Number of rows deleted
            error_message: Error message if failed
        """
        with self.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            query = """
                UPDATE cleanup_job_status
                SET last_run_at = ?,
                    next_run_at = ?,
                    rows_deleted = ?,
                    status = ?,
                    error_message = ?
                WHERE job_name = ?
            """

            next_run = datetime.utcnow() + timedelta(days=1)

            cursor.execute(query, (
                datetime.utcnow(),
                next_run,
                rows_deleted,
                status,
                error_message,
                job_name
            ))

            conn.commit()

    def run_all_cleanups(self):
        """
        Run all cleanup jobs.

        Returns:
            Dictionary with cleanup statistics
        """
        logger.info("Starting database cleanup jobs...")

        stats = {}

        # Request logs cleanup
        try:
            self.update_cleanup_status('api_request_logs_cleanup', 'running', 0)
            rows = self.cleanup_request_logs()
            stats['request_logs'] = rows
            self.update_cleanup_status('api_request_logs_cleanup', 'success', rows)
        except Exception as e:
            logger.error(f"Request logs cleanup failed: {e}")
            self.update_cleanup_status('api_request_logs_cleanup', 'failed', 0, str(e))
            stats['request_logs'] = 0

        # Rate limit history cleanup
        try:
            self.update_cleanup_status('rate_limit_history_cleanup', 'running', 0)
            rows = self.cleanup_rate_limit_history()
            stats['rate_limit_history'] = rows
            self.update_cleanup_status('rate_limit_history_cleanup', 'success', rows)
        except Exception as e:
            logger.error(f"Rate limit history cleanup failed: {e}")
            self.update_cleanup_status('rate_limit_history_cleanup', 'failed', 0, str(e))
            stats['rate_limit_history'] = 0

        # WebSocket sessions cleanup
        try:
            self.update_cleanup_status('websocket_sessions_cleanup', 'running', 0)
            rows = self.cleanup_websocket_sessions()
            stats['websocket_sessions'] = rows
            self.update_cleanup_status('websocket_sessions_cleanup', 'success', rows)
        except Exception as e:
            logger.error(f"WebSocket sessions cleanup failed: {e}")
            self.update_cleanup_status('websocket_sessions_cleanup', 'failed', 0, str(e))
            stats['websocket_sessions'] = 0

        # Expired API keys
        try:
            rows = self.cleanup_expired_api_keys()
            stats['expired_api_keys'] = rows
        except Exception as e:
            logger.error(f"Expired API keys cleanup failed: {e}")
            stats['expired_api_keys'] = 0

        # Upload operations cleanup
        try:
            rows = self.cleanup_old_upload_operations()
            stats['upload_operations'] = rows
        except Exception as e:
            logger.error(f"Upload operations cleanup failed: {e}")
            stats['upload_operations'] = 0

        logger.info("Database cleanup completed")
        logger.info(f"Cleanup statistics: {stats}")

        return stats


def main():
    """Main entry point for cleanup job."""
    logger.info("RAG API Database Cleanup Job")
    logger.info("=" * 80)

    try:
        # Load configuration
        config = load_config()

        # Get retention days from config
        retention_days = config.get('logging', {}).get('retention_days', 30)

        # Initialize connection pool
        db_config = config.get('database', {})

        pool = IRISConnectionPool(
            host=db_config.get('host', 'localhost'),
            port=db_config.get('port', 1972),
            namespace=db_config.get('namespace', 'USER'),
            username=db_config.get('username', 'demo'),
            password=db_config.get('password', 'demo'),
            pool_size=5,
            max_overflow=2
        )

        # Run cleanup
        cleanup_job = DatabaseCleanupJob(pool, retention_days)
        stats = cleanup_job.run_all_cleanups()

        # Display results
        print("\nCleanup Results:")
        print("=" * 80)
        print(f"Request logs deleted:       {stats.get('request_logs', 0)}")
        print(f"Rate limit records deleted: {stats.get('rate_limit_history', 0)}")
        print(f"WebSocket sessions deleted: {stats.get('websocket_sessions', 0)}")
        print(f"Expired API keys:           {stats.get('expired_api_keys', 0)}")
        print(f"Upload operations deleted:  {stats.get('upload_operations', 0)}")
        print("=" * 80)

        total_deleted = sum(stats.values())
        print(f"\nTotal rows cleaned: {total_deleted}")

        pool.dispose()

        sys.exit(0)

    except Exception as e:
        logger.error(f"Cleanup job failed: {e}", exc_info=True)
        sys.exit(1)


if __name__ == '__main__':
    main()
