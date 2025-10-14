"""
IRIS Cache Backend for LLM Caching

This module provides an IRIS database backend for the LLM caching layer,
leveraging the existing IRIS infrastructure for persistent cache storage.
"""

import json
import hashlib
import logging
from typing import Any, Dict, Optional
from datetime import datetime, timedelta

from common.llm_cache_config import CacheConfig

logger = logging.getLogger(__name__)


class IRISCacheBackend:
    """IRIS database backend for LLM response caching."""

    def __init__(
        self,
        iris_connector,
        table_name: str = "llm_cache",
        ttl_seconds: int = 3600,
        schema: str = "USER",
    ):
        """
        Initialize IRIS cache backend.

        Args:
            iris_connector: IRIS database connector instance (DBAPI, JDBC, or SQLAlchemy)
            table_name: Name of the cache table
            ttl_seconds: Default time-to-live for cache entries
            schema: Database schema name
        """
        self.iris_connector = iris_connector
        self.table_name = table_name
        self.ttl_seconds = ttl_seconds
        self.schema = schema
        self.full_table_name = f"{schema}.{table_name}" if schema else table_name

        # Detect connection type and set up appropriate interface
        self._setup_connection_interface()

        # Cache statistics
        self.stats = {"hits": 0, "misses": 0, "sets": 0, "deletes": 0, "errors": 0}

        # Ensure table exists
        self.setup_table()

    def _setup_connection_interface(self):
        """Setup the appropriate interface based on connection type."""
        # Check if it's a SQLAlchemy connection
        if hasattr(self.iris_connector, "execute") and hasattr(
            self.iris_connector, "commit"
        ):
            self.connection_type = "sqlalchemy"
            logger.debug("Using SQLAlchemy connection interface for cache")
        # Check if it's a DBAPI/JDBC connection
        elif hasattr(self.iris_connector, "cursor"):
            self.connection_type = "dbapi"
            logger.debug("Using DBAPI/JDBC connection interface for cache")
        else:
            # Try to detect other connection types
            logger.warning(
                f"Unknown connection type: {type(self.iris_connector)}. Attempting DBAPI interface."
            )
            self.connection_type = "dbapi"

    def _get_cursor(self):
        """Get a cursor appropriate for the connection type."""
        # Validate connection handle before proceeding
        if self.iris_connector is None:
            raise ConnectionError("Cannot get cursor: _handle is NULL")

        if self.connection_type == "sqlalchemy":
            # For SQLAlchemy connections, we use the connection directly
            return self.iris_connector
        else:
            # For DBAPI/JDBC connections
            try:
                cursor = self.iris_connector.cursor()
                if cursor is None:
                    raise ConnectionError("Failed to create cursor: cursor is NULL")
                return cursor
            except AttributeError as e:
                raise ConnectionError(
                    f"Connection object does not support cursor(): {e}"
                )

    def _execute_sql(self, cursor, sql, params=None):
        """Execute SQL with appropriate method based on connection type."""
        if self.connection_type == "sqlalchemy":
            if params:
                return cursor.execute(sql, params)
            else:
                return cursor.execute(sql)
        else:
            if params:
                return cursor.execute(sql, params)
            else:
                return cursor.execute(sql)

    def _commit_transaction(self):
        """Commit transaction based on connection type."""
        if self.connection_type == "sqlalchemy":
            # SQLAlchemy connections handle commits differently
            pass  # SQLAlchemy autocommits by default
        else:
            self.iris_connector.commit()

    def _close_cursor(self, cursor):
        """Close cursor if needed based on connection type."""
        if self.connection_type == "sqlalchemy":
            # SQLAlchemy connections don't need cursor closing
            pass
        else:
            cursor.close()

    def setup_table(self) -> None:
        """Create the cache table if it doesn't exist."""
        # Validate connection handle before proceeding
        if self.iris_connector is None:
            error_msg = "Failed to setup IRIS cache table: _handle is NULL"
            logger.error(error_msg)
            self.stats["errors"] += 1
            raise ConnectionError(error_msg)

        try:
            cursor = self._get_cursor()

            # Validate cursor was created successfully
            if cursor is None:
                error_msg = "Failed to setup IRIS cache table: cursor is NULL"
                logger.error(error_msg)
                self.stats["errors"] += 1
                raise ConnectionError(error_msg)

            # Create table with proper IRIS SQL syntax
            create_table_sql = f"""
            CREATE TABLE IF NOT EXISTS {self.full_table_name} (
                cache_key VARCHAR(255) PRIMARY KEY,
                cache_value LONGVARCHAR,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                expires_at TIMESTAMP,
                model_name VARCHAR(100),
                prompt_hash VARCHAR(64)
            )
            """

            self._execute_sql(cursor, create_table_sql)

            # Create indexes with existence check
            self._create_index_if_not_exists(
                cursor, f"idx_{self.table_name}_expires", "expires_at"
            )
            self._create_index_if_not_exists(
                cursor, f"idx_{self.table_name}_model", "model_name"
            )

            self._commit_transaction()
            self._close_cursor(cursor)

            logger.info(
                f"IRIS cache table {self.full_table_name} initialized successfully"
            )

        except Exception as e:
            logger.error(f"Failed to setup IRIS cache table: {e}")
            self.stats["errors"] += 1
            raise

    def _create_index_if_not_exists(
        self, cursor, index_name: str, column_name: str
    ) -> None:
        """
        Create an index only if it doesn't already exist.

        Args:
            cursor: Database cursor
            index_name: Name of the index to create
            column_name: Column to index
        """
        try:
            # Check if index exists using IRIS system catalog
            # Try multiple approaches as IRIS may have different system table configurations
            index_exists = False

            # Method 1: Try %Dictionary.CompiledIndex (IRIS-specific)
            try:
                check_index_sql = """
                SELECT COUNT(*) FROM %Dictionary.CompiledIndex
                WHERE ParentClass = ? AND Name = ?
                """
                self._execute_sql(
                    cursor, check_index_sql, (self.full_table_name, index_name)
                )
                result = cursor.fetchone()
                if result and result[0] > 0:
                    index_exists = True
            except Exception:
                # Method 1 failed, try Method 2
                pass

            # Method 2: Try INFORMATION_SCHEMA.INDEXES if Method 1 failed
            if not index_exists:
                try:
                    check_index_sql = """
                    SELECT COUNT(*) FROM INFORMATION_SCHEMA.INDEXES
                    WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ? AND INDEX_NAME = ?
                    """
                    self._execute_sql(
                        cursor,
                        check_index_sql,
                        (self.schema, self.table_name, index_name),
                    )
                    result = cursor.fetchone()
                    if result and result[0] > 0:
                        index_exists = True
                except Exception:
                    # Method 2 failed, try Method 3
                    pass

            # Method 3: Try a simple CREATE INDEX and catch the error if it exists
            if not index_exists:
                try:
                    create_index_sql = f"""
                    CREATE INDEX {index_name} ON {self.full_table_name} ({column_name})
                    """
                    self._execute_sql(cursor, create_index_sql)
                    logger.debug(
                        f"Created index {index_name} on {self.full_table_name}({column_name})"
                    )
                except Exception as e:
                    # Check if error is about index already existing
                    error_msg = str(e).lower()
                    if any(
                        keyword in error_msg
                        for keyword in ["already exists", "duplicate", "exists"]
                    ):
                        logger.debug(f"Index {index_name} already exists")
                    else:
                        # Re-raise if it's a different error
                        raise
            else:
                logger.debug(f"Index {index_name} already exists")

        except Exception as e:
            logger.error(f"Error creating index {index_name}: {e}")
            # Don't raise - index creation is not critical for basic functionality

    def _serialize_cache_value(self, value: Any) -> str:
        """
        Robustly serialize cache value to JSON string, handling ChatGeneration objects.

        Args:
            value: Value to serialize

        Returns:
            JSON string representation of the value
        """
        try:
            # If it's already a string (pre-serialized), return as-is
            if isinstance(value, str):
                # Test if it's valid JSON
                try:
                    json.loads(value)
                    return value
                except json.JSONDecodeError:
                    # Not JSON, treat as plain string
                    return json.dumps(value)

            # Handle basic types
            if isinstance(value, (dict, list, int, float, bool)) or value is None:
                return json.dumps(value)

            # Handle LangChain Generation objects that might slip through
            try:
                from langchain_core.outputs import Generation, ChatGeneration
                from langchain_core.messages import BaseMessage
                from langchain_core.load import dumpd

                if isinstance(value, (Generation, ChatGeneration)):
                    logger.warning(
                        f"Received unserialized {type(value).__name__} object in IRIS cache backend"
                    )

                    # Use the same robust serialization as in _update_sync
                    try:
                        serialized = dumpd(value)

                        # Handle nested BaseMessage objects
                        if hasattr(value, "message") and isinstance(
                            value.message, BaseMessage
                        ):
                            if "message" in serialized and isinstance(
                                serialized["message"], BaseMessage
                            ):
                                serialized["message"] = dumpd(value.message)

                        return json.dumps(serialized)

                    except Exception as dumpd_error:
                        logger.warning(
                            f"dumpd failed for {type(value).__name__}: {dumpd_error}"
                        )

                        # Fallback to manual extraction
                        fallback_data = {
                            "text": getattr(value, "text", str(value)),
                            "type": type(value).__name__,
                            "generation_info": getattr(value, "generation_info", None),
                        }

                        if hasattr(value, "message"):
                            message = value.message
                            if hasattr(message, "content"):
                                fallback_data["message"] = {
                                    "content": getattr(message, "content", ""),
                                    "type": type(message).__name__,
                                }
                            else:
                                fallback_data["message"] = str(message)

                        return json.dumps(fallback_data)

            except ImportError:
                # LangChain not available, continue with other methods
                pass

            # Try to serialize as-is
            try:
                return json.dumps(value)
            except (TypeError, ValueError) as e:
                logger.warning(
                    f"Direct JSON serialization failed for {type(value)}: {e}"
                )

                # Try to convert to dict if possible
                if hasattr(value, "dict"):
                    try:
                        return json.dumps(value.dict())
                    except Exception as dict_error:
                        logger.warning(f"Dict serialization failed: {dict_error}")

                # Last resort: string representation
                logger.warning(f"Using string fallback for {type(value)}")
                return json.dumps(
                    {
                        "text": str(value),
                        "type": type(value).__name__,
                        "error": "serialization_fallback",
                    }
                )

        except Exception as e:
            logger.error(f"Cache value serialization failed completely: {e}")
            # Ultimate fallback
            return json.dumps(
                {"text": "Serialization failed", "type": "unknown", "error": str(e)}
            )

    def _generate_cache_key(
        self, prompt: str, model_name: str = "default", **kwargs
    ) -> str:
        """
        Generate a cache key from prompt and parameters.

        Args:
            prompt: The input prompt
            model_name: LLM model name
            **kwargs: Additional parameters (temperature, max_tokens, etc.)

        Returns:
            SHA256 hash as cache key
        """
        # Create deterministic key from prompt and parameters
        cache_data = {
            "prompt": prompt.strip() if isinstance(prompt, str) else str(prompt),
            "model": model_name,
            **kwargs,
        }

        # Sort keys for deterministic hashing
        cache_str = json.dumps(cache_data, sort_keys=True)
        return hashlib.sha256(cache_str.encode()).hexdigest()

    def get(self, cache_key: str) -> Optional[Any]:
        """
        Retrieve value from cache.

        Args:
            cache_key: Cache key to retrieve

        Returns:
            Cached value or None if not found/expired
        """
        # Validate connection handle before proceeding
        if self.iris_connector is None:
            logger.error("Cannot retrieve from cache: _handle is NULL")
            self.stats["errors"] += 1
            return None

        try:
            cursor = self._get_cursor()

            # Query with expiration check using IRIS SQL syntax
            select_sql = f"""
            SELECT cache_value
            FROM {self.full_table_name}
            WHERE cache_key = ?
            AND (expires_at IS NULL OR expires_at > CURRENT_TIMESTAMP)
            """

            self._execute_sql(cursor, select_sql, (cache_key,))

            if self.connection_type == "sqlalchemy":
                result = cursor.fetchone()
            else:
                result = cursor.fetchone()

            self._close_cursor(cursor)

            if result:
                self.stats["hits"] += 1
                try:
                    # Parse JSON response
                    return json.loads(result[0])
                except json.JSONDecodeError:
                    # Return as string if not JSON
                    return result[0]
            else:
                self.stats["misses"] += 1
                return None

        except Exception as e:
            logger.error(f"Error retrieving from IRIS cache: {e}")
            self.stats["errors"] += 1
            return None

    def set(
        self,
        cache_key: str,
        value: Any,
        ttl: Optional[int] = None,
        model_name: str = "default",
        prompt_hash: Optional[str] = None,
    ) -> None:
        """
        Store value in cache.

        Args:
            cache_key: Cache key
            value: Value to cache
            ttl: Time-to-live in seconds (uses default if None)
            model_name: LLM model name for analytics
            prompt_hash: Hash of the original prompt
        """
        # Validate connection handle before proceeding
        if self.iris_connector is None:
            logger.error("Cannot store to cache: _handle is NULL")
            self.stats["errors"] += 1
            return

        try:
            cursor = self._get_cursor()

            # Calculate expiration time
            ttl_to_use = ttl or self.ttl_seconds
            expires_at = datetime.now() + timedelta(seconds=ttl_to_use)

            # Convert datetime to string format for IRIS compatibility
            expires_at_str = expires_at.strftime("%Y-%m-%d %H:%M:%S")

            # Robust serialization with ChatGeneration handling
            cache_value = self._serialize_cache_value(value)

            # Add logging to debug the SQL statement
            logger.debug(
                f"Attempting to store cache entry for key: {cache_key[:16]}..."
            )

            # Use simpler INSERT OR REPLACE pattern for IRIS compatibility
            # First try to delete existing entry, then insert new one
            delete_sql = f"DELETE FROM {self.full_table_name} WHERE cache_key = ?"
            self._execute_sql(cursor, delete_sql, (cache_key,))

            # Now insert the new entry
            insert_sql = f"""
            INSERT INTO {self.full_table_name}
            (cache_key, cache_value, created_at, expires_at, model_name, prompt_hash)
            VALUES (?, ?, CURRENT_TIMESTAMP, ?, ?, ?)
            """

            logger.debug(f"Executing INSERT SQL: {insert_sql}")
            logger.debug(
                f"With parameters: cache_key={cache_key[:16]}..., cache_value_len={len(cache_value)}, expires_at={expires_at_str}, model_name={model_name}, prompt_hash={prompt_hash}"
            )

            self._execute_sql(
                cursor,
                insert_sql,
                (cache_key, cache_value, expires_at_str, model_name, prompt_hash),
            )

            self._commit_transaction()
            self._close_cursor(cursor)

            self.stats["sets"] += 1
            logger.debug(f"Cached value for key: {cache_key[:16]}...")

        except Exception as e:
            logger.error(f"Error storing to IRIS cache: {e}")
            self.stats["errors"] += 1
            # Don't raise - graceful fallback

    def delete(self, cache_key: str) -> None:
        """
        Delete value from cache.

        Args:
            cache_key: Cache key to delete
        """
        try:
            cursor = self._get_cursor()

            delete_sql = f"DELETE FROM {self.full_table_name} WHERE cache_key = ?"
            self._execute_sql(cursor, delete_sql, (cache_key,))

            self._commit_transaction()
            self._close_cursor(cursor)

            self.stats["deletes"] += 1

        except Exception as e:
            logger.error(f"Error deleting from IRIS cache: {e}")
            self.stats["errors"] += 1

    def clear(self) -> None:
        """Clear all cache entries."""
        try:
            cursor = self._get_cursor()

            delete_all_sql = f"DELETE FROM {self.full_table_name}"
            self._execute_sql(cursor, delete_all_sql)

            self._commit_transaction()
            self._close_cursor(cursor)

            logger.info("IRIS cache cleared")

        except Exception as e:
            logger.error(f"Error clearing IRIS cache: {e}")
            self.stats["errors"] += 1

    def cleanup_expired(self, batch_size: int = 1000) -> int:
        """
        Remove expired cache entries.

        Args:
            batch_size: Number of entries to delete in each batch

        Returns:
            Number of entries deleted
        """
        try:
            cursor = self._get_cursor()

            # Delete expired entries using IRIS SQL syntax
            delete_expired_sql = f"""
            DELETE FROM {self.full_table_name}
            WHERE expires_at IS NOT NULL
            AND expires_at < CURRENT_TIMESTAMP
            """

            result = self._execute_sql(cursor, delete_expired_sql)

            # Get row count based on connection type
            if self.connection_type == "sqlalchemy":
                deleted_count = result.rowcount if hasattr(result, "rowcount") else 0
            else:
                deleted_count = cursor.rowcount if hasattr(cursor, "rowcount") else 0

            self._commit_transaction()
            self._close_cursor(cursor)

            if deleted_count > 0:
                logger.info(f"Cleaned up {deleted_count} expired cache entries")

            return deleted_count

        except Exception as e:
            logger.error(f"Error cleaning up expired cache entries: {e}")
            self.stats["errors"] += 1
            return 0

    def get_stats(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_requests = self.stats["hits"] + self.stats["misses"]
        hit_rate = self.stats["hits"] / total_requests if total_requests > 0 else 0.0

        return {
            **self.stats,
            "total_requests": total_requests,
            "hit_rate": hit_rate,
            "table_name": self.full_table_name,
        }

    def get_cache_info(self) -> Dict[str, Any]:
        """Get detailed cache information from database."""
        try:
            cursor = self._get_cursor()

            # Get cache statistics using IRIS SQL
            stats_sql = f"""
            SELECT
                COUNT(*) as total_entries,
                COUNT(CASE WHEN expires_at > CURRENT_TIMESTAMP OR expires_at IS NULL THEN 1 END) as active_entries,
                COUNT(CASE WHEN expires_at <= CURRENT_TIMESTAMP THEN 1 END) as expired_entries,
                MIN(created_at) as oldest_entry,
                MAX(created_at) as newest_entry
            FROM {self.full_table_name}
            """

            self._execute_sql(cursor, stats_sql)
            result = cursor.fetchone()

            if result:
                info = {
                    "total_entries": result[0],
                    "active_entries": result[1],
                    "expired_entries": result[2],
                    "oldest_entry": result[3],
                    "newest_entry": result[4],
                }
            else:
                info = {
                    "total_entries": 0,
                    "active_entries": 0,
                    "expired_entries": 0,
                    "oldest_entry": None,
                    "newest_entry": None,
                }

            self._close_cursor(cursor)
            return info

        except Exception as e:
            logger.error(f"Error getting cache info: {e}")
            return {"error": str(e)}


def create_iris_cache_backend(config: CacheConfig, iris_connector) -> IRISCacheBackend:
    """
    Factory function to create IRIS cache backend.

    Args:
        config: Cache configuration
        iris_connector: IRIS database connector

    Returns:
        IRISCacheBackend instance
    """
    return IRISCacheBackend(
        iris_connector=iris_connector,
        table_name=config.table_name,
        ttl_seconds=config.ttl_seconds,
        schema=config.iris_schema,
    )
