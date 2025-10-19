"""
Request/Response Logging Middleware for RAG API.

Implements FR-029 to FR-031: Structured logging with request tracking.
Logs all requests to database and exports metrics for monitoring.
"""

import logging
import time
import uuid
from datetime import datetime
from typing import Optional

from fastapi import Request
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from iris_rag.api.models.request import APIRequestLog, HTTPMethod
from iris_rag.api.models.auth import ApiKey


logger = logging.getLogger(__name__)


class RequestLoggingMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for request/response logging.

    Implements FR-029: Log all requests to database with execution metrics.
    Implements FR-030: Include request_id for tracing.
    """

    def __init__(self, app, connection_pool):
        """
        Initialize logging middleware.

        Args:
            app: FastAPI application
            connection_pool: IRISConnectionPool for database logging
        """
        super().__init__(app)
        self.connection_pool = connection_pool

    def generate_request_id(self) -> uuid.UUID:
        """
        Generate unique request identifier.

        Returns:
            UUID for request tracking (FR-030)
        """
        return uuid.uuid4()

    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process request with logging.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware in chain

        Returns:
            HTTP response with X-Request-ID header
        """
        # Generate request ID (FR-030)
        request_id = self.generate_request_id()

        # Check for client-provided request ID
        client_request_id = request.headers.get("X-Request-ID")
        if client_request_id:
            try:
                request_id = uuid.UUID(client_request_id)
            except ValueError:
                # Invalid UUID, use generated one
                logger.warning(f"Invalid X-Request-ID from client: {client_request_id}")

        # Attach request_id to request state for downstream use
        request.state.request_id = request_id

        # Get authenticated API key (if available)
        api_key: Optional[ApiKey] = getattr(request.state, 'api_key', None)

        # Record start time
        start_time = time.time()
        start_datetime = datetime.utcnow()

        # Process request
        try:
            response = await call_next(request)

            # Calculate execution time
            execution_time_ms = int((time.time() - start_time) * 1000)

            # Log successful request
            self._log_request(
                request_id=request_id,
                api_key=api_key,
                method=request.method,
                endpoint=str(request.url.path),
                query_params=dict(request.query_params),
                status_code=response.status_code,
                execution_time_ms=execution_time_ms,
                timestamp=start_datetime,
                client_ip=request.client.host if request.client else "unknown",
                user_agent=request.headers.get("User-Agent", "unknown")
            )

            # Add response headers (FR-030, FR-031)
            response.headers["X-Request-ID"] = str(request_id)
            response.headers["X-Execution-Time-Ms"] = str(execution_time_ms)

            return response

        except Exception as e:
            # Calculate execution time
            execution_time_ms = int((time.time() - start_time) * 1000)

            # Log failed request
            status_code = getattr(e, 'status_code', 500)

            self._log_request(
                request_id=request_id,
                api_key=api_key,
                method=request.method,
                endpoint=str(request.url.path),
                query_params=dict(request.query_params),
                status_code=status_code,
                execution_time_ms=execution_time_ms,
                timestamp=start_datetime,
                client_ip=request.client.host if request.client else "unknown",
                user_agent=request.headers.get("User-Agent", "unknown"),
                error_message=str(e)
            )

            # Re-raise exception
            raise

    def _log_request(
        self,
        request_id: uuid.UUID,
        api_key: Optional[ApiKey],
        method: str,
        endpoint: str,
        query_params: dict,
        status_code: int,
        execution_time_ms: int,
        timestamp: datetime,
        client_ip: str,
        user_agent: str,
        error_message: Optional[str] = None
    ):
        """
        Log request to database and structured logger.

        Args:
            request_id: Unique request identifier
            api_key: Authenticated API key (if available)
            method: HTTP method
            endpoint: Request endpoint path
            query_params: Query parameters
            status_code: HTTP status code
            execution_time_ms: Request execution time
            timestamp: Request timestamp
            client_ip: Client IP address
            user_agent: User agent string
            error_message: Error message (if failed)
        """
        # Create log entry
        log_entry = APIRequestLog(
            request_id=request_id,
            api_key_id=api_key.key_id if api_key else None,
            method=HTTPMethod(method),
            endpoint=endpoint,
            query_params=query_params if query_params else None,
            status_code=status_code,
            execution_time_ms=execution_time_ms,
            timestamp=timestamp,
            client_ip=client_ip,
            user_agent=user_agent,
            error_message=error_message
        )

        # Log to structured logger (FR-030)
        log_level = logging.INFO if status_code < 400 else logging.WARNING
        logger.log(
            log_level,
            f"{method} {endpoint} - {status_code} - {execution_time_ms}ms",
            extra={
                "request_id": str(request_id),
                "api_key_id": str(api_key.key_id) if api_key else None,
                "api_key_name": api_key.name if api_key else None,
                "method": method,
                "endpoint": endpoint,
                "status_code": status_code,
                "execution_time_ms": execution_time_ms,
                "client_ip": client_ip,
                "user_agent": user_agent,
                "error_message": error_message
            }
        )

        # Log to database (FR-029)
        try:
            with self.connection_pool.get_connection() as conn:
                cursor = conn.cursor()

                query = """
                    INSERT INTO api_request_logs (
                        request_id, api_key_id, method, endpoint, query_params,
                        status_code, execution_time_ms, timestamp, client_ip,
                        user_agent, error_message
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """

                cursor.execute(query, (
                    str(log_entry.request_id),
                    str(log_entry.api_key_id) if log_entry.api_key_id else None,
                    log_entry.method.value,
                    log_entry.endpoint,
                    str(log_entry.query_params) if log_entry.query_params else None,
                    log_entry.status_code,
                    log_entry.execution_time_ms,
                    log_entry.timestamp,
                    log_entry.client_ip,
                    log_entry.user_agent,
                    log_entry.error_message
                ))

                conn.commit()

        except Exception as e:
            # Don't fail request if logging fails
            logger.error(f"Failed to log request to database: {e}")


class MetricsExporter:
    """
    Export API metrics for monitoring.

    Implements FR-031: Export metrics for external monitoring systems.
    """

    def __init__(self, connection_pool):
        """
        Initialize metrics exporter.

        Args:
            connection_pool: IRISConnectionPool for database queries
        """
        self.connection_pool = connection_pool

    def get_request_metrics(
        self,
        start_time: datetime,
        end_time: datetime
    ) -> dict:
        """
        Get request metrics for time window.

        Args:
            start_time: Window start time
            end_time: Window end time

        Returns:
            Dictionary of metrics
        """
        with self.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            # Total requests
            cursor.execute("""
                SELECT COUNT(*) FROM api_request_logs
                WHERE timestamp >= ? AND timestamp <= ?
            """, (start_time, end_time))
            total_requests = cursor.fetchone()[0]

            # Requests by status
            cursor.execute("""
                SELECT status_code, COUNT(*)
                FROM api_request_logs
                WHERE timestamp >= ? AND timestamp <= ?
                GROUP BY status_code
            """, (start_time, end_time))
            status_counts = {row[0]: row[1] for row in cursor.fetchall()}

            # Average execution time
            cursor.execute("""
                SELECT AVG(execution_time_ms),
                       MIN(execution_time_ms),
                       MAX(execution_time_ms)
                FROM api_request_logs
                WHERE timestamp >= ? AND timestamp <= ?
            """, (start_time, end_time))
            exec_time_stats = cursor.fetchone()

            # Top endpoints
            cursor.execute("""
                SELECT endpoint, COUNT(*) as count
                FROM api_request_logs
                WHERE timestamp >= ? AND timestamp <= ?
                GROUP BY endpoint
                ORDER BY count DESC
                LIMIT 10
            """, (start_time, end_time))
            top_endpoints = [
                {"endpoint": row[0], "count": row[1]}
                for row in cursor.fetchall()
            ]

            # Error rate
            error_count = sum(
                count for status, count in status_counts.items()
                if status >= 400
            )
            error_rate = (error_count / total_requests * 100) if total_requests > 0 else 0

            return {
                "total_requests": total_requests,
                "status_counts": status_counts,
                "avg_execution_time_ms": exec_time_stats[0] if exec_time_stats[0] else 0,
                "min_execution_time_ms": exec_time_stats[1] if exec_time_stats[1] else 0,
                "max_execution_time_ms": exec_time_stats[2] if exec_time_stats[2] else 0,
                "error_rate_percentage": round(error_rate, 2),
                "top_endpoints": top_endpoints,
                "window_start": start_time.isoformat(),
                "window_end": end_time.isoformat()
            }
