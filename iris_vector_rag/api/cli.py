#!/usr/bin/env python3
"""
CLI for RAG API Management.

Provides commands for running the server, managing API keys, and database setup.
"""

import sys
import logging
import argparse
import getpass
from pathlib import Path
from uuid import UUID

import uvicorn
import yaml

from iris_vector_rag.api.main import load_config, create_app
from iris_vector_rag.api.models.auth import ApiKeyCreateRequest, Permission, RateLimitTier
from common.connection_pool import IRISConnectionPool


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


def get_connection_pool(config_path: str = "config/api_config.yaml") -> IRISConnectionPool:
    """Get IRIS connection pool from config."""
    config = load_config(config_path)
    db_config = config.get("database", {})

    return IRISConnectionPool(
        host=db_config.get("host", "localhost"),
        port=db_config.get("port", 1972),
        namespace=db_config.get("namespace", "USER"),
        username=db_config.get("username", "demo"),
        password=db_config.get("password", "demo"),
        pool_size=5,
        max_overflow=2
    )


def cmd_run(args):
    """Run the API server."""
    logger.info("Starting RAG API server...")

    config = load_config(args.config)
    server_config = config.get("server", {})

    uvicorn.run(
        "iris_rag.api.main:app",
        host=args.host or server_config.get("host", "0.0.0.0"),
        port=args.port or server_config.get("port", 8000),
        workers=args.workers or server_config.get("workers", 4),
        reload=args.reload,
        log_level=args.log_level
    )


def cmd_create_key(args):
    """Create a new API key."""
    from iris_vector_rag.api.services.auth_service import AuthService

    logger.info("Creating API key...")

    # Parse permissions
    permissions = [Permission(p) for p in args.permissions]

    # Parse rate limit tier
    tier = RateLimitTier(args.tier)

    # Create request
    request = ApiKeyCreateRequest(
        name=args.name,
        permissions=permissions,
        rate_limit_tier=tier,
        description=args.description,
        expires_in_days=args.expires_in_days
    )

    # Get connection pool
    pool = get_connection_pool(args.config)

    try:
        # Create service
        auth_service = AuthService(pool)

        # Create API key
        response = auth_service.create_api_key(request, args.owner_email)

        # Display response
        print("\n" + "=" * 80)
        print("API Key Created Successfully!")
        print("=" * 80)
        print(f"Key ID:      {response.key_id}")
        print(f"Key Secret:  {response.key_secret}")
        print(f"Name:        {response.name}")
        print(f"Permissions: {', '.join([p.value for p in response.permissions])}")
        print(f"Tier:        {response.rate_limit_tier.value}")
        print(f"Created:     {response.created_at}")
        if response.expires_at:
            print(f"Expires:     {response.expires_at}")
        print("\n" + "!" * 80)
        print("IMPORTANT: Save the Key Secret - it will not be shown again!")
        print("!" * 80)
        print("\nBase64-encoded credentials for Authorization header:")
        import base64
        credentials = f"{response.key_id}:{response.key_secret}"
        encoded = base64.b64encode(credentials.encode()).decode()
        print(f"Authorization: ApiKey {encoded}")
        print("=" * 80 + "\n")

    finally:
        pool.dispose()


def cmd_list_keys(args):
    """List API keys."""
    from iris_vector_rag.api.services.auth_service import AuthService

    logger.info("Listing API keys...")

    pool = get_connection_pool(args.config)

    try:
        auth_service = AuthService(pool)
        api_keys = auth_service.list_api_keys(owner_email=args.owner_email)

        print("\n" + "=" * 100)
        print(f"API Keys ({len(api_keys)} total)")
        print("=" * 100)
        print(f"{'ID':<38} {'Name':<20} {'Tier':<12} {'Permissions':<20} {'Active':<8} {'Expires'}")
        print("-" * 100)

        for key in api_keys:
            expires = key.expires_at.strftime("%Y-%m-%d") if key.expires_at else "Never"
            perms = ",".join([p.value for p in key.permissions])
            active = "Yes" if key.is_active else "No"

            print(f"{str(key.key_id):<38} {key.name:<20} {key.rate_limit_tier.value:<12} {perms:<20} {active:<8} {expires}")

        print("=" * 100 + "\n")

    finally:
        pool.dispose()


def cmd_revoke_key(args):
    """Revoke an API key."""
    from iris_vector_rag.api.services.auth_service import AuthService

    logger.info(f"Revoking API key: {args.key_id}")

    pool = get_connection_pool(args.config)

    try:
        auth_service = AuthService(pool)
        success = auth_service.revoke_api_key(UUID(args.key_id))

        if success:
            print(f"\nAPI key {args.key_id} has been revoked.")
        else:
            print(f"\nAPI key {args.key_id} not found.")

    finally:
        pool.dispose()


def cmd_setup_db(args):
    """Setup database tables."""
    logger.info("Setting up database tables...")

    pool = get_connection_pool(args.config)

    try:
        with pool.get_connection() as conn:
            cursor = conn.cursor()

            # Read SQL schema file
            schema_path = Path(__file__).parent / "schema.sql"

            if not schema_path.exists():
                logger.error(f"Schema file not found: {schema_path}")
                sys.exit(1)

            with open(schema_path, 'r') as f:
                schema_sql = f.read()

            # Execute schema
            for statement in schema_sql.split(';'):
                statement = statement.strip()
                if statement:
                    cursor.execute(statement)

            conn.commit()

            logger.info("Database tables created successfully")

    except Exception as e:
        logger.error(f"Database setup failed: {e}")
        sys.exit(1)

    finally:
        pool.dispose()


def cmd_health(args):
    """Check API health."""
    import requests

    url = f"http://{args.host}:{args.port}/api/v1/health"

    logger.info(f"Checking health: {url}")

    try:
        response = requests.get(url, timeout=5)
        health = response.json()

        print("\n" + "=" * 80)
        print(f"Overall Status: {health['status'].upper()}")
        print("=" * 80)

        for component_name, component in health['components'].items():
            status = component['status'].upper()
            response_time = component.get('response_time_ms', 'N/A')

            print(f"{component_name:<30} {status:<12} {response_time}ms")

        print("=" * 80 + "\n")

        sys.exit(0 if health['status'] == 'healthy' else 1)

    except Exception as e:
        logger.error(f"Health check failed: {e}")
        sys.exit(1)


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="RAG API Management CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run the API server
  python -m iris_rag.api.cli run

  # Run with custom settings
  python -m iris_rag.api.cli run --host 0.0.0.0 --port 8000 --workers 4

  # Create an API key
  python -m iris_rag.api.cli create-key --name "My Key" --owner-email user@example.com

  # Create enterprise key with all permissions
  python -m iris_rag.api.cli create-key --name "Admin Key" --owner-email admin@example.com \\
    --permissions read write admin --tier enterprise

  # List all API keys
  python -m iris_rag.api.cli list-keys

  # Revoke an API key
  python -m iris_rag.api.cli revoke-key --key-id 7c9e6679-7425-40de-944b-e07fc1f90ae7

  # Setup database tables
  python -m iris_rag.api.cli setup-db

  # Check API health
  python -m iris_rag.api.cli health
        """
    )

    parser.add_argument(
        '--config',
        default='config/api_config.yaml',
        help='Path to configuration file (default: config/api_config.yaml)'
    )

    subparsers = parser.add_subparsers(dest='command', help='Command to execute')

    # Run command
    run_parser = subparsers.add_parser('run', help='Run the API server')
    run_parser.add_argument('--host', help='Host to bind to')
    run_parser.add_argument('--port', type=int, help='Port to bind to')
    run_parser.add_argument('--workers', type=int, help='Number of worker processes')
    run_parser.add_argument('--reload', action='store_true', help='Enable auto-reload')
    run_parser.add_argument('--log-level', default='info', help='Log level')
    run_parser.set_defaults(func=cmd_run)

    # Create key command
    create_key_parser = subparsers.add_parser('create-key', help='Create a new API key')
    create_key_parser.add_argument('--name', required=True, help='Key name')
    create_key_parser.add_argument('--owner-email', required=True, help='Owner email address')
    create_key_parser.add_argument(
        '--permissions',
        nargs='+',
        default=['read'],
        choices=['read', 'write', 'admin'],
        help='Permissions (default: read)'
    )
    create_key_parser.add_argument(
        '--tier',
        default='basic',
        choices=['basic', 'premium', 'enterprise'],
        help='Rate limit tier (default: basic)'
    )
    create_key_parser.add_argument('--description', help='Key description')
    create_key_parser.add_argument(
        '--expires-in-days',
        type=int,
        help='Expiration in days (default: 365)'
    )
    create_key_parser.set_defaults(func=cmd_create_key)

    # List keys command
    list_keys_parser = subparsers.add_parser('list-keys', help='List API keys')
    list_keys_parser.add_argument('--owner-email', help='Filter by owner email')
    list_keys_parser.set_defaults(func=cmd_list_keys)

    # Revoke key command
    revoke_key_parser = subparsers.add_parser('revoke-key', help='Revoke an API key')
    revoke_key_parser.add_argument('--key-id', required=True, help='API key ID to revoke')
    revoke_key_parser.set_defaults(func=cmd_revoke_key)

    # Setup DB command
    setup_db_parser = subparsers.add_parser('setup-db', help='Setup database tables')
    setup_db_parser.set_defaults(func=cmd_setup_db)

    # Health command
    health_parser = subparsers.add_parser('health', help='Check API health')
    health_parser.add_argument('--host', default='localhost', help='API host')
    health_parser.add_argument('--port', type=int, default=8000, help='API port')
    health_parser.set_defaults(func=cmd_health)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == '__main__':
    main()
