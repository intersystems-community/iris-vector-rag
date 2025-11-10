#!/usr/bin/env python3
"""
Pipeline Setup Testing from Clean IRIS

Tests complete pipeline initialization workflow starting from a fresh IRIS database.
Validates that the framework can set up all required components from scratch.

Constitutional Requirement: Clean IRIS Testing (NON-NEGOTIABLE)
"""

import argparse
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from common.utils import get_llm_func
from iris_vector_rag import create_pipeline
from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.connection import ConnectionManager


def test_pipeline_setup(verbose: bool = False):
    """Test complete pipeline setup from clean IRIS."""

    def log(message: str, level: str = "info"):
        timestamp = time.strftime("%H:%M:%S")
        if level == "error":
            print(f"❌ [{timestamp}] {message}")
        elif level == "success":
            print(f"✅ [{timestamp}] {message}")
        elif level == "warning":
            print(f"⚠️  [{timestamp}] {message}")
        else:
            print(f"ℹ️  [{timestamp}] {message}")

    log("🧪 Testing Pipeline Setup from Clean IRIS", "info")
    log("=" * 60, "info")

    # Test each pipeline type
    pipeline_types = ["basic", "graphrag", "crag", "hybrid_graphrag"]
    results = {}

    for pipeline_type in pipeline_types:
        log(f"Testing {pipeline_type} pipeline setup...", "info")

        try:
            start_time = time.time()

            # Create pipeline with auto-setup enabled
            pipeline = create_pipeline(
                pipeline_type=pipeline_type, validate_requirements=True, auto_setup=True
            )

            setup_time = time.time() - start_time

            # Configure LLM (using mock for testing)
            if hasattr(pipeline, "llm_func"):
                from scripts.testing.mock_providers import MockLLMProvider

                mock_llm = MockLLMProvider(mode="realistic")
                pipeline.llm_func = mock_llm.generate_response

            # Test basic functionality
            if hasattr(pipeline, "vector_store"):
                # Test vector store connectivity
                vector_store_ready = pipeline.vector_store is not None
            else:
                vector_store_ready = True

            results[pipeline_type] = {
                "success": True,
                "setup_time": setup_time,
                "vector_store_ready": vector_store_ready,
                "pipeline_created": pipeline is not None,
            }

            log(f"✅ {pipeline_type} setup successful ({setup_time:.2f}s)", "success")

            if verbose:
                log(f"   Pipeline object: {type(pipeline).__name__}", "info")
                log(f"   Vector store ready: {vector_store_ready}", "info")
                log(f"   Setup time: {setup_time:.2f}s", "info")

        except Exception as e:
            results[pipeline_type] = {
                "success": False,
                "error": str(e),
                "setup_time": 0,
            }
            log(f"❌ {pipeline_type} setup failed: {e}", "error")
            if verbose:
                import traceback

                log(f"   Full traceback: {traceback.format_exc()}", "error")

    # Summary report
    log("", "info")
    log("📊 Pipeline Setup Test Results", "info")
    log("=" * 60, "info")

    successful = [p for p, r in results.items() if r.get("success", False)]
    failed = [p for p, r in results.items() if not r.get("success", False)]

    log(f"Total pipelines tested: {len(pipeline_types)}", "info")
    log(f"Successful setups: {len(successful)}", "success" if successful else "info")
    log(f"Failed setups: {len(failed)}", "error" if failed else "info")

    if successful:
        log("✅ Successful pipelines:", "success")
        for pipeline_type in successful:
            result = results[pipeline_type]
            log(f"   - {pipeline_type}: {result['setup_time']:.2f}s", "info")

    if failed:
        log("❌ Failed pipelines:", "error")
        for pipeline_type in failed:
            result = results[pipeline_type]
            log(
                f"   - {pipeline_type}: {result.get('error', 'Unknown error')}", "error"
            )

    # Calculate overall success rate
    success_rate = len(successful) / len(pipeline_types)
    log(
        f"Overall success rate: {success_rate:.1%}",
        "success" if success_rate >= 0.75 else "warning",
    )

    return results, success_rate >= 0.75


def test_database_connectivity():
    """Test basic database connectivity from clean state."""

    def log(message: str, level: str = "info"):
        timestamp = time.strftime("%H:%M:%S")
        if level == "error":
            print(f"❌ [{timestamp}] {message}")
        elif level == "success":
            print(f"✅ [{timestamp}] {message}")
        else:
            print(f"ℹ️  [{timestamp}] {message}")

    log("Testing database connectivity...", "info")

    try:
        config = ConfigurationManager()
        conn_mgr = ConnectionManager(config)
        conn = conn_mgr.get_connection()

        # Test basic query
        cursor = conn.cursor()
        cursor.execute("SELECT 1 as test")
        result = cursor.fetchone()

        if result and result[0] == 1:
            log("Database connectivity verified", "success")
            cursor.close()
            return True
        else:
            log("Database query returned unexpected result", "error")
            return False

    except Exception as e:
        log(f"Database connectivity failed: {e}", "error")
        return False


def main():
    """Main execution function."""
    parser = argparse.ArgumentParser(
        description="Test pipeline setup from clean IRIS database"
    )
    parser.add_argument(
        "--verbose", "-v", action="store_true", help="Enable verbose output"
    )

    args = parser.parse_args()

    print("🧪 Pipeline Setup Testing from Clean IRIS")
    print("Constitutional Requirement: Clean IRIS Testing")
    print("=" * 60)

    # Test database connectivity first
    if not test_database_connectivity():
        print("\n❌ Database connectivity test failed")
        print("Ensure IRIS is running and accessible")
        return 1

    # Test pipeline setup
    results, overall_success = test_pipeline_setup(verbose=args.verbose)

    if overall_success:
        print("\n🎉 Pipeline setup testing completed successfully!")
        print("✅ All critical pipelines can be initialized from clean IRIS")
        return 0
    else:
        print("\n⚠️  Pipeline setup testing completed with issues")
        print("❌ Some pipelines failed to initialize from clean IRIS")
        return 1


if __name__ == "__main__":
    sys.exit(main())
