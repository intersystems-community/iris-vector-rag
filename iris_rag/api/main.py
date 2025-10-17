"""
Main FastAPI Application for RAG API.

Production-grade REST API server implementing all functional requirements.
Integrates middleware, services, routes, and WebSocket support.
"""

import logging
import sys
from contextlib import asynccontextmanager
from typing import AsyncIterator

import redis
import yaml
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from common.connection_pool import IRISConnectionPool
from iris_rag.api.middleware import (
    AuthenticationMiddleware,
    RateLimitMiddleware,
    RequestLoggingMiddleware
)
from iris_rag.api.services import (
    PipelineManager,
    AuthService,
    DocumentService
)
from iris_rag.api.routes import (
    create_query_router,
    create_pipeline_router,
    create_document_router,
    create_health_router
)
from iris_rag.api.websocket import (
    ConnectionManager,
    QueryStreamingHandler,
    DocumentUploadProgressHandler,
    create_websocket_router
)
from iris_rag.api.models.errors import ErrorResponse, ErrorType, ErrorInfo, ErrorDetails


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)

logger = logging.getLogger(__name__)


class AppState:
    """
    Application state container.

    Holds all service instances and connections.
    """

    def __init__(self):
        self.config: dict = {}
        self.connection_pool: IRISConnectionPool = None
        self.redis_client: redis.Redis = None
        self.pipeline_manager: PipelineManager = None
        self.auth_service: AuthService = None
        self.document_service: DocumentService = None
        self.connection_manager: ConnectionManager = None
        self.query_handler: QueryStreamingHandler = None
        self.upload_handler: DocumentUploadProgressHandler = None


# Global app state
app_state = AppState()


def load_config(config_path: str = "config/api_config.yaml") -> dict:
    """
    Load configuration from YAML file.

    Args:
        config_path: Path to configuration file

    Returns:
        Configuration dictionary
    """
    try:
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f)

        logger.info(f"Loaded configuration from {config_path}")
        return config

    except FileNotFoundError:
        logger.warning(f"Config file not found: {config_path}, using defaults")
        return get_default_config()

    except Exception as e:
        logger.error(f"Failed to load config: {e}")
        raise


def get_default_config() -> dict:
    """
    Get default configuration.

    Returns:
        Default configuration dictionary
    """
    return {
        "server": {
            "host": "0.0.0.0",
            "port": 8000,
            "workers": 4,
            "reload": False
        },
        "cors": {
            "enabled": True,
            "allowed_origins": ["http://localhost:3000", "http://localhost:8501"],
            "allow_credentials": True,
            "allow_methods": ["*"],
            "allow_headers": ["*"]
        },
        "database": {
            "host": "localhost",
            "port": 1972,
            "namespace": "USER",
            "username": "demo",
            "password": "demo",
            "pool_size": 20,
            "max_overflow": 10,
            "pool_recycle": 3600
        },
        "redis": {
            "enabled": False,
            "host": "localhost",
            "port": 6379,
            "db": 0
        },
        "pipelines": {
            "enabled": ["basic", "basic_rerank", "crag", "graphrag", "pylate_colbert"]
        },
        "auth": {
            "bcrypt_rounds": 12
        },
        "rate_limiting": {
            "max_concurrent_per_key": 10
        },
        "websocket": {
            "max_connections_per_key": 10,
            "heartbeat_interval": 30,
            "idle_timeout": 300
        }
    }


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """
    Application lifespan manager.

    Handles startup and shutdown operations.

    Args:
        app: FastAPI application

    Yields:
        None during application runtime
    """
    # Startup
    logger.info("Starting RAG API server...")

    try:
        # Load configuration
        app_state.config = load_config()

        # Initialize IRIS connection pool
        db_config = app_state.config.get("database", {})

        app_state.connection_pool = IRISConnectionPool(
            host=db_config.get("host", "localhost"),
            port=db_config.get("port", 1972),
            namespace=db_config.get("namespace", "USER"),
            username=db_config.get("username", "demo"),
            password=db_config.get("password", "demo"),
            pool_size=db_config.get("pool_size", 20),
            max_overflow=db_config.get("max_overflow", 10),
            pool_recycle=db_config.get("pool_recycle", 3600)
        )

        logger.info("IRIS connection pool initialized")

        # Initialize Redis (optional)
        redis_config = app_state.config.get("redis", {})

        if redis_config.get("enabled", False):
            app_state.redis_client = redis.Redis(
                host=redis_config.get("host", "localhost"),
                port=redis_config.get("port", 6379),
                db=redis_config.get("db", 0),
                decode_responses=True
            )

            # Test connection
            app_state.redis_client.ping()
            logger.info("Redis connection established")

        else:
            logger.info("Redis disabled - rate limiting will be in-memory")

        # Initialize services
        app_state.pipeline_manager = PipelineManager(app_state.config)
        app_state.auth_service = AuthService(
            app_state.connection_pool,
            bcrypt_rounds=app_state.config.get("auth", {}).get("bcrypt_rounds", 12)
        )
        app_state.document_service = DocumentService(
            app_state.connection_pool,
            app_state.pipeline_manager
        )

        logger.info("Services initialized")

        # Initialize WebSocket components
        from iris_rag.api.middleware.auth import ApiKeyAuth

        auth_middleware = ApiKeyAuth(app_state.connection_pool)

        ws_config = app_state.config.get("websocket", {})

        app_state.connection_manager = ConnectionManager(
            auth_service=auth_middleware,
            max_connections_per_key=ws_config.get("max_connections_per_key", 10),
            heartbeat_interval=ws_config.get("heartbeat_interval", 30),
            idle_timeout=ws_config.get("idle_timeout", 300)
        )

        app_state.query_handler = QueryStreamingHandler(
            app_state.connection_manager,
            app_state.pipeline_manager
        )

        app_state.upload_handler = DocumentUploadProgressHandler(
            app_state.connection_manager,
            app_state.document_service
        )

        logger.info("WebSocket handlers initialized")

        logger.info("RAG API server started successfully")

        yield

    except Exception as e:
        logger.error(f"Startup failed: {e}", exc_info=True)
        raise

    finally:
        # Shutdown
        logger.info("Shutting down RAG API server...")

        # Close connections
        if app_state.connection_pool:
            app_state.connection_pool.dispose()
            logger.info("IRIS connection pool closed")

        if app_state.redis_client:
            app_state.redis_client.close()
            logger.info("Redis connection closed")

        logger.info("RAG API server shut down")


def create_app() -> FastAPI:
    """
    Create and configure FastAPI application.

    Returns:
        Configured FastAPI application
    """
    app = FastAPI(
        title="RAG API",
        description="""
        Production-grade REST API for Retrieval-Augmented Generation (RAG) pipelines.

        **Features:**
        - Multiple RAG pipelines (BasicRAG, CRAG, GraphRAG, etc.)
        - API key authentication with bcrypt
        - Rate limiting (60-1000 requests/minute based on tier)
        - Request/response logging with tracing
        - WebSocket streaming for real-time progress
        - Async document upload with validation
        - Health monitoring for all components
        - 100% LangChain & RAGAS compatible responses

        **Authentication:**
        All endpoints (except /health and /pipelines) require API key authentication:
        ```
        Authorization: ApiKey <base64(id:secret)>
        ```

        **Rate Limits:**
        - Basic tier: 60 requests/minute
        - Premium tier: 100 requests/minute
        - Enterprise tier: 1000 requests/minute

        **Elasticsearch-Inspired Design:**
        - POST /{pipeline}/_search for queries
        - Structured error responses with actionable guidance
        - Adaptive request concurrency
        - Response headers for debugging (X-Request-ID, X-Execution-Time-Ms)
        """,
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json"
    )

    # Add CORS middleware
    cors_config = app_state.config.get("cors", {})

    if cors_config.get("enabled", True):
        app.add_middleware(
            CORSMiddleware,
            allow_origins=cors_config.get("allowed_origins", ["*"]),
            allow_credentials=cors_config.get("allow_credentials", True),
            allow_methods=cors_config.get("allow_methods", ["*"]),
            allow_headers=cors_config.get("allow_headers", ["*"])
        )

    # Add custom middleware (order matters!)
    # 1. Request logging (outermost - logs all requests)
    app.add_middleware(
        RequestLoggingMiddleware,
        connection_pool=app_state.connection_pool
    )

    # 2. Rate limiting (after logging)
    if app_state.redis_client:
        app.add_middleware(
            RateLimitMiddleware,
            redis_client=app_state.redis_client,
            max_concurrent_per_key=app_state.config.get("rate_limiting", {}).get(
                "max_concurrent_per_key", 10
            )
        )

    # 3. Authentication (innermost - runs last)
    app.add_middleware(
        AuthenticationMiddleware,
        connection_pool=app_state.connection_pool
    )

    # Add exception handlers
    @app.exception_handler(Exception)
    async def global_exception_handler(request, exc):
        """Handle uncaught exceptions."""
        logger.error(f"Unhandled exception: {exc}", exc_info=True)

        return JSONResponse(
            status_code=500,
            content=ErrorResponse(
                error=ErrorInfo(
                    type=ErrorType.INTERNAL_SERVER_ERROR,
                    reason="An unexpected error occurred",
                    details=ErrorDetails(
                        message="Please try again or contact support."
                    )
                )
            ).model_dump()
        )

    # Register routers
    from iris_rag.api.middleware.auth import ApiKeyAuth

    auth_middleware = ApiKeyAuth(app_state.connection_pool)

    # Query routes
    query_router = create_query_router(
        app_state.pipeline_manager,
        auth_middleware
    )
    app.include_router(query_router)

    # Pipeline routes
    pipeline_router = create_pipeline_router(app_state.pipeline_manager)
    app.include_router(pipeline_router)

    # Document routes
    document_router = create_document_router(
        app_state.document_service,
        auth_middleware
    )
    app.include_router(document_router)

    # Health routes
    health_router = create_health_router(
        app_state.pipeline_manager,
        app_state.connection_pool,
        app_state.redis_client
    )
    app.include_router(health_router)

    # WebSocket routes
    websocket_router = create_websocket_router(
        app_state.connection_manager,
        app_state.query_handler,
        app_state.upload_handler
    )
    app.include_router(websocket_router)

    return app


# Create application instance
app = create_app()


if __name__ == "__main__":
    import uvicorn

    config = load_config()
    server_config = config.get("server", {})

    uvicorn.run(
        "iris_rag.api.main:app",
        host=server_config.get("host", "0.0.0.0"),
        port=server_config.get("port", 8000),
        workers=server_config.get("workers", 4),
        reload=server_config.get("reload", False),
        log_level="info"
    )
