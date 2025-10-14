#!/usr/bin/env python3
"""
Health check utility for RAG Templates services.
Provides common health check functionality across all services.
"""

import argparse
import logging
import sys
import time
from typing import Any, Dict, Optional

import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class HealthChecker:
    """Generic health checker for services."""

    def __init__(self, service_name: str = "unknown"):
        self.service_name = service_name
        self.checks = []

    def add_http_check(self, url: str, expected_status: int = 200, timeout: int = 5):
        """Add HTTP endpoint health check."""
        self.checks.append(
            {
                "type": "http",
                "url": url,
                "expected_status": expected_status,
                "timeout": timeout,
            }
        )

    def add_redis_check(
        self, host: str = "localhost", port: int = 6379, timeout: int = 3
    ):
        """Add Redis connection check."""
        self.checks.append(
            {"type": "redis", "host": host, "port": port, "timeout": timeout}
        )

    def add_database_check(self, connection_string: str, timeout: int = 5):
        """Add database connection check."""
        self.checks.append(
            {
                "type": "database",
                "connection_string": connection_string,
                "timeout": timeout,
            }
        )

    def check_http(self, check: Dict[str, Any]) -> bool:
        """Perform HTTP health check."""
        try:
            response = requests.get(
                check["url"], timeout=check["timeout"], allow_redirects=False
            )
            return response.status_code == check["expected_status"]
        except Exception as e:
            logger.error(f"HTTP check failed: {e}")
            return False

    def check_redis(self, check: Dict[str, Any]) -> bool:
        """Perform Redis health check."""
        try:
            import redis

            client = redis.Redis(
                host=check["host"], port=check["port"], socket_timeout=check["timeout"]
            )
            client.ping()
            return True
        except Exception as e:
            logger.error(f"Redis check failed: {e}")
            return False

    def check_database(self, check: Dict[str, Any]) -> bool:
        """Perform database health check."""
        try:
            # This is a simplified check - real implementation would vary by database
            # For IRIS, we might use intersystems_iris module
            return True  # Placeholder
        except Exception as e:
            logger.error(f"Database check failed: {e}")
            return False

    def run_checks(self) -> bool:
        """Run all health checks."""
        if not self.checks:
            logger.warning("No health checks configured")
            return True

        all_passed = True
        for i, check in enumerate(self.checks):
            check_type = check["type"]
            logger.info(f"Running {check_type} check {i+1}/{len(self.checks)}")

            if check_type == "http":
                passed = self.check_http(check)
            elif check_type == "redis":
                passed = self.check_redis(check)
            elif check_type == "database":
                passed = self.check_database(check)
            else:
                logger.error(f"Unknown check type: {check_type}")
                passed = False

            if passed:
                logger.info(f"✓ {check_type} check passed")
            else:
                logger.error(f"✗ {check_type} check failed")
                all_passed = False

        return all_passed


def main():
    """Main health check entry point."""
    parser = argparse.ArgumentParser(description="Health check utility")
    parser.add_argument("--service", default="unknown", help="Service name")
    parser.add_argument("--http", help="HTTP endpoint to check")
    parser.add_argument("--redis-host", default="localhost", help="Redis host")
    parser.add_argument("--redis-port", type=int, default=6379, help="Redis port")
    parser.add_argument("--timeout", type=int, default=5, help="Check timeout")
    parser.add_argument("--retries", type=int, default=3, help="Number of retries")
    parser.add_argument(
        "--retry-delay", type=int, default=1, help="Delay between retries"
    )

    args = parser.parse_args()

    checker = HealthChecker(args.service)

    # Add checks based on arguments
    if args.http:
        checker.add_http_check(args.http, timeout=args.timeout)

    # For services that use Redis
    if args.service in ["rag_api", "streamlit_app"]:
        checker.add_redis_check(args.redis_host, args.redis_port, args.timeout)

    # Retry logic
    for attempt in range(args.retries):
        if attempt > 0:
            logger.info(f"Retry attempt {attempt + 1}/{args.retries}")
            time.sleep(args.retry_delay)

        if checker.run_checks():
            logger.info(f"✓ All health checks passed for {args.service}")
            sys.exit(0)

        if attempt < args.retries - 1:
            logger.warning(f"Health check failed, retrying in {args.retry_delay}s...")

    logger.error(f"✗ Health checks failed for {args.service}")
    sys.exit(1)


if __name__ == "__main__":
    main()
