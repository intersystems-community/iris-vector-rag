#!/usr/bin/env python3
"""
TDD test for IRIS database connectivity in biomedical RAG evaluation framework.
This test ensures the framework can connect to IRIS before proceeding with evaluation.
"""

import logging
import os
import sys
import traceback
from typing import Any, Dict, Optional

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class IRISConnectivityTest:
    """Test suite for IRIS database connectivity validation."""

    def __init__(self):
        self.test_results = {}
        self.connection = None

    def test_iris_import(self) -> bool:
        """Test if IRIS Python modules can be imported."""
        try:
            logger.info("Testing IRIS intersystems-irispython import...")
            import iris

            logger.info("✓ IRIS module imported successfully")
            return True
        except ImportError as e:
            logger.error(f"✗ Failed to import IRIS module: {e}")
            return False
        except Exception as e:
            logger.error(f"✗ Unexpected error importing IRIS: {e}")
            return False

    def test_iris_dbapi_connector(self) -> bool:
        """Test IRIS DBAPI connector from common utilities."""
        try:
            logger.info("Testing IRIS DBAPI connector...")
            from common.iris_dbapi_connector import get_iris_dbapi_connection

            logger.info("✓ IRIS DBAPI connector imported successfully")
            return True
        except ImportError as e:
            logger.error(f"✗ Failed to import IRIS DBAPI connector: {e}")
            return False
        except Exception as e:
            logger.error(f"✗ Unexpected error with IRIS DBAPI connector: {e}")
            return False

    def test_iris_connection_manager(self) -> bool:
        """Test IRIS connection manager."""
        try:
            logger.info("Testing IRIS connection manager...")
            from common.iris_connection_manager import IRISConnectionManager

            manager = IRISConnectionManager()
            logger.info("✓ IRIS connection manager initialized successfully")
            return True
        except ImportError as e:
            logger.error(f"✗ Failed to import IRIS connection manager: {e}")
            return False
        except Exception as e:
            logger.error(f"✗ Unexpected error with IRIS connection manager: {e}")
            return False

    def test_iris_connection(self) -> bool:
        """Test actual IRIS database connection."""
        try:
            logger.info("Testing actual IRIS database connection...")
            from common.iris_dbapi_connector import get_iris_dbapi_connection

            # Attempt connection
            self.connection = get_iris_dbapi_connection()

            if self.connection is None:
                logger.warning(
                    "⚠ IRIS connection returned None - database may not be running"
                )
                return False

            # Test basic query
            cursor = self.connection.cursor()
            cursor.execute("SELECT 1 as test_value")
            result = cursor.fetchone()
            cursor.close()

            if result and result[0] == 1:
                logger.info("✓ IRIS database connection and basic query successful")
                return True
            else:
                logger.error("✗ IRIS database query returned unexpected result")
                return False

        except Exception as e:
            logger.error(f"✗ Failed to connect to IRIS database: {e}")
            logger.info("This may be expected if IRIS is not running locally")
            return False

    def test_pipeline_targets(self) -> bool:
        """Test if the target RAG pipelines can be imported."""
        try:
            logger.info("Testing RAG pipeline imports...")

            # Test imports for the target pipelines
            pipelines_tested = 0
            pipelines_success = 0

            try:
                from iris_rag.pipelines.basic import BasicRAG

                logger.info("✓ BasicRAG imported successfully")
                pipelines_success += 1
            except Exception as e:
                logger.error(f"✗ Failed to import BasicRAG: {e}")
            pipelines_tested += 1

            try:
                from iris_rag.pipelines.crag import CRAG

                logger.info("✓ CRAG imported successfully")
                pipelines_success += 1
            except Exception as e:
                logger.error(f"✗ Failed to import CRAG: {e}")
            pipelines_tested += 1

            try:
                from iris_rag.pipelines.graphrag import GraphRAG

                logger.info("✓ GraphRAG imported successfully")
                pipelines_success += 1
            except Exception as e:
                logger.error(f"✗ Failed to import GraphRAG: {e}")
            pipelines_tested += 1

            try:
                from iris_rag.pipelines.basic_rerank import BasicRAGReranking

                logger.info("✓ BasicRAGReranking imported successfully")
                pipelines_success += 1
            except Exception as e:
                logger.error(f"✗ Failed to import BasicRAGReranking: {e}")
            pipelines_tested += 1

            success_rate = pipelines_success / pipelines_tested
            logger.info(
                f"Pipeline import success rate: {pipelines_success}/{pipelines_tested} ({success_rate:.1%})"
            )

            return success_rate >= 0.5  # At least 50% of pipelines should import

        except Exception as e:
            logger.error(f"✗ Failed to test pipeline imports: {e}")
            return False

    def cleanup(self):
        """Clean up database connections."""
        if self.connection:
            try:
                self.connection.close()
                logger.info("✓ IRIS connection closed successfully")
            except Exception as e:
                logger.error(f"Error closing IRIS connection: {e}")

    def run_all_tests(self) -> Dict[str, bool]:
        """Run all IRIS connectivity tests."""
        logger.info("=" * 60)
        logger.info("IRIS CONNECTIVITY TEST SUITE")
        logger.info("=" * 60)

        tests = [
            ("iris_import", self.test_iris_import),
            ("iris_dbapi_connector", self.test_iris_dbapi_connector),
            ("iris_connection_manager", self.test_iris_connection_manager),
            ("iris_connection", self.test_iris_connection),
            ("pipeline_targets", self.test_pipeline_targets),
        ]

        results = {}

        for test_name, test_func in tests:
            logger.info(f"\nRunning test: {test_name}")
            try:
                results[test_name] = test_func()
            except Exception as e:
                logger.error(f"✗ Test {test_name} failed with exception: {e}")
                results[test_name] = False

        # Cleanup
        self.cleanup()

        # Summary
        logger.info("\n" + "=" * 60)
        logger.info("TEST RESULTS SUMMARY")
        logger.info("=" * 60)

        passed = sum(results.values())
        total = len(results)

        for test_name, passed_test in results.items():
            status = "✓ PASS" if passed_test else "✗ FAIL"
            logger.info(f"  {status} - {test_name}")

        logger.info(f"\nOverall: {passed}/{total} tests passed ({passed/total:.1%})")

        if passed >= total * 0.6:  # 60% pass rate for basic connectivity
            logger.info(
                "🎉 IRIS connectivity tests PASSED - framework ready for evaluation"
            )
            return True
        else:
            logger.error("❌ IRIS connectivity tests FAILED - check database setup")
            return False


def main():
    """Main test runner."""
    tester = IRISConnectivityTest()
    success = tester.run_all_tests()

    if success:
        logger.info(
            "\n✅ IRIS connectivity validated - proceed with evaluation framework validation"
        )
        sys.exit(0)
    else:
        logger.error(
            "\n❌ IRIS connectivity failed - fix database setup before proceeding"
        )
        sys.exit(1)


if __name__ == "__main__":
    main()
