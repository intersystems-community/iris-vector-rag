"""
API Key Authentication Middleware for RAG API.

Implements FR-010 to FR-012: Secure API key authentication with header validation.
Uses base64-encoded 'id:secret' format in Authorization header.
"""

import base64
import logging
from typing import Optional, Tuple
from uuid import UUID

import bcrypt
from fastapi import Request, HTTPException
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.responses import Response

from iris_vector_rag.api.models.auth import ApiKey, Permission
from iris_vector_rag.api.models.errors import ErrorResponse, ErrorType, ErrorInfo, ErrorDetails


logger = logging.getLogger(__name__)


class ApiKeyAuth:
    """
    API key authentication service.

    Implements FR-010: API key authentication with base64-encoded credentials.
    Format: Authorization: ApiKey <base64(id:secret)>
    """

    def __init__(self, connection_pool):
        """
        Initialize authentication service.

        Args:
            connection_pool: IRISConnectionPool for database queries
        """
        self.connection_pool = connection_pool
        self.security = HTTPBearer(scheme_name="ApiKey")

    def parse_api_key(self, credentials: str) -> Tuple[str, str]:
        """
        Parse base64-encoded API key credentials.

        Args:
            credentials: Base64-encoded 'id:secret' string

        Returns:
            Tuple of (key_id, key_secret)

        Raises:
            HTTPException: If credentials are malformed
        """
        try:
            # Decode base64
            decoded = base64.b64decode(credentials).decode('utf-8')

            # Split on first colon
            if ':' not in decoded:
                raise ValueError("Missing colon separator")

            key_id, key_secret = decoded.split(':', 1)

            # Validate UUID format
            try:
                UUID(key_id)
            except ValueError:
                raise ValueError("Invalid UUID format for key_id")

            return key_id, key_secret

        except Exception as e:
            logger.warning(f"Failed to parse API key: {e}")
            raise HTTPException(
                status_code=401,
                detail=ErrorResponse(
                    error=ErrorInfo(
                        type=ErrorType.AUTHENTICATION_ERROR,
                        reason="Malformed API key credentials",
                        details=ErrorDetails(
                            message="API key must be base64-encoded 'id:secret' format"
                        )
                    )
                ).model_dump()
            )

    def verify_api_key(self, key_id: str, key_secret: str) -> ApiKey:
        """
        Verify API key against database.

        Args:
            key_id: API key UUID
            key_secret: Plain-text secret to verify

        Returns:
            ApiKey object if valid

        Raises:
            HTTPException: If key is invalid, expired, or inactive
        """
        with self.connection_pool.get_connection() as conn:
            cursor = conn.cursor()

            # Query API key from database
            query = """
                SELECT
                    key_id, key_secret_hash, name, permissions, rate_limit_tier,
                    requests_per_minute, requests_per_hour, created_at,
                    expires_at, is_active, owner_email, description
                FROM api_keys
                WHERE key_id = ?
            """

            cursor.execute(query, (key_id,))
            row = cursor.fetchone()

            if not row:
                logger.warning(f"API key not found: {key_id}")
                raise HTTPException(
                    status_code=401,
                    detail=ErrorResponse(
                        error=ErrorInfo(
                            type=ErrorType.INVALID_API_KEY,
                            reason="Invalid API key",
                            details=ErrorDetails(
                                message="API key not found or has been revoked",
                                key_id=key_id
                            )
                        )
                    ).model_dump()
                )

            # Parse row into ApiKey object
            api_key = ApiKey(
                key_id=UUID(row[0]),
                key_secret_hash=row[1],
                name=row[2],
                permissions=[Permission(p) for p in row[3].split(',')],
                rate_limit_tier=row[4],
                requests_per_minute=row[5],
                requests_per_hour=row[6],
                created_at=row[7],
                expires_at=row[8],
                is_active=row[9],
                owner_email=row[10],
                description=row[11]
            )

            # Check if key is active (FR-011)
            if not api_key.is_active:
                logger.warning(f"Inactive API key used: {key_id}")
                raise HTTPException(
                    status_code=401,
                    detail=ErrorResponse(
                        error=ErrorInfo(
                            type=ErrorType.INVALID_API_KEY,
                            reason="API key is inactive",
                            details=ErrorDetails(
                                message="This API key has been deactivated",
                                key_id=key_id
                            )
                        )
                    ).model_dump()
                )

            # Check if key is expired (FR-011)
            if api_key.expires_at and api_key.expires_at < api_key.created_at.now():
                logger.warning(f"Expired API key used: {key_id}")
                raise HTTPException(
                    status_code=401,
                    detail=ErrorResponse(
                        error=ErrorInfo(
                            type=ErrorType.EXPIRED_API_KEY,
                            reason="API key has expired",
                            details=ErrorDetails(
                                message="This API key expired and must be renewed",
                                key_id=key_id,
                                expired_at=api_key.expires_at.isoformat()
                            )
                        )
                    ).model_dump()
                )

            # Verify secret with bcrypt (FR-010)
            if not bcrypt.checkpw(key_secret.encode('utf-8'),
                                 api_key.key_secret_hash.encode('utf-8')):
                logger.warning(f"Invalid secret for API key: {key_id}")
                raise HTTPException(
                    status_code=401,
                    detail=ErrorResponse(
                        error=ErrorInfo(
                            type=ErrorType.INVALID_API_KEY,
                            reason="Invalid API key credentials",
                            details=ErrorDetails(
                                message="API key secret does not match",
                                key_id=key_id
                            )
                        )
                    ).model_dump()
                )

            logger.info(f"API key authenticated: {key_id} ({api_key.name})")
            return api_key

    def check_permission(self, api_key: ApiKey, required_permission: Permission):
        """
        Check if API key has required permission.

        Args:
            api_key: Authenticated API key
            required_permission: Permission required for operation

        Raises:
            HTTPException: If permission is missing (FR-011)
        """
        if required_permission not in api_key.permissions:
            logger.warning(
                f"Permission denied: {api_key.key_id} lacks {required_permission}"
            )
            raise HTTPException(
                status_code=403,
                detail=ErrorResponse(
                    error=ErrorInfo(
                        type=ErrorType.AUTHORIZATION_ERROR,
                        reason="Insufficient permissions for this operation",
                        details=ErrorDetails(
                            required_permissions=[required_permission.value],
                            current_permissions=[p.value for p in api_key.permissions],
                            message=f"This operation requires '{required_permission.value}' permission"
                        )
                    )
                ).model_dump()
            )

    async def __call__(
        self,
        request: Request,
        required_permission: Optional[Permission] = None
    ) -> ApiKey:
        """
        Authenticate request and check permissions.

        Args:
            request: FastAPI request object
            required_permission: Optional permission to check

        Returns:
            Authenticated ApiKey object

        Raises:
            HTTPException: If authentication or authorization fails
        """
        # Extract Authorization header
        auth_header = request.headers.get("Authorization")

        if not auth_header:
            raise HTTPException(
                status_code=401,
                detail=ErrorResponse(
                    error=ErrorInfo(
                        type=ErrorType.AUTHENTICATION_ERROR,
                        reason="Missing Authorization header",
                        details=ErrorDetails(
                            message="API requests must include 'Authorization: ApiKey <base64(id:secret)>' header"
                        )
                    )
                ).model_dump()
            )

        # Check format: "ApiKey <credentials>"
        parts = auth_header.split(' ', 1)
        if len(parts) != 2 or parts[0] != "ApiKey":
            raise HTTPException(
                status_code=401,
                detail=ErrorResponse(
                    error=ErrorInfo(
                        type=ErrorType.AUTHENTICATION_ERROR,
                        reason="Invalid Authorization header format",
                        details=ErrorDetails(
                            message="Authorization header must be 'ApiKey <base64(id:secret)>'"
                        )
                    )
                ).model_dump()
            )

        # Parse and verify credentials
        key_id, key_secret = self.parse_api_key(parts[1])
        api_key = self.verify_api_key(key_id, key_secret)

        # Check permission if required
        if required_permission:
            self.check_permission(api_key, required_permission)

        # Attach to request state for downstream use
        request.state.api_key = api_key

        return api_key


class AuthenticationMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for API key authentication.

    Implements FR-010: Authenticate all requests except health endpoint.
    """

    def __init__(self, app, connection_pool):
        """
        Initialize authentication middleware.

        Args:
            app: FastAPI application
            connection_pool: IRISConnectionPool for database queries
        """
        super().__init__(app)
        self.auth_service = ApiKeyAuth(connection_pool)

        # Endpoints that don't require authentication
        self.public_endpoints = {
            "/api/v1/health",
            "/api/v1/health/",
            "/docs",
            "/docs/",
            "/redoc",
            "/redoc/",
            "/openapi.json"
        }

    async def dispatch(self, request: Request, call_next) -> Response:
        """
        Process request with authentication.

        Args:
            request: Incoming HTTP request
            call_next: Next middleware in chain

        Returns:
            HTTP response
        """
        # Skip authentication for public endpoints
        if request.url.path in self.public_endpoints:
            return await call_next(request)

        # Authenticate request (will raise HTTPException if fails)
        try:
            await self.auth_service(request)
        except HTTPException as e:
            # Log authentication failure
            logger.warning(
                f"Authentication failed: {request.method} {request.url.path} - {e.detail}"
            )
            raise

        # Continue to next middleware
        response = await call_next(request)
        return response
