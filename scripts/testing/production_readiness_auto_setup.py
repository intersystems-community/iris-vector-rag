#!/usr/bin/env python3
"""
Production Readiness Test with Auto-Setup

This script validates production readiness by testing all RAG pipelines
with the auto_setup feature, which automatically resolves missing requirements.

This approach uses the config-driven state resolution system.
"""

import logging
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import iris_vector_rag
from common.utils import get_llm_func


def setup_logging():
    """Setup logging configuration."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def test_pipeline_with_auto_setup(
    pipeline_type: str, test_query: str = "What is diabetes?"
) -> Dict[str, Any]:
    """
    Test a pipeline with auto_setup enabled.

    Args:
        pipeline_type: Type of pipeline to test
        test_query: Query to test with

    Returns:
        Dictionary with test results
    """
    result = {
        "pipeline_type": pipeline_type,
        "success": False,
        "execution_time": 0.0,
        "error": None,
        "response": None,
        "auto_setup_worked": False,
    }

    start_time = time.time()

    try:
        print(f"🧪 Testing {pipeline_type} pipeline with auto_setup...")

        # Create pipeline with auto_setup=True
        pipeline = iris_rag.create_pipeline(
            pipeline_type=pipeline_type,
            auto_setup=True,  # This should automatically resolve missing requirements
            validate_requirements=True,
            llm_func=get_llm_func(),
        )

        result["auto_setup_worked"] = True
        print(f"✅ {pipeline_type} pipeline created successfully with auto_setup")

        # Test query
        response = pipeline.query(test_query, top_k=3)
        result["response"] = response
        result["success"] = True

        print(f"✅ {pipeline_type} query executed successfully")
        print(f"   Response length: {len(str(response))} characters")

    except Exception as e:
        result["error"] = str(e)
        print(f"❌ {pipeline_type} pipeline failed: {str(e)}")

    finally:
        result["execution_time"] = time.time() - start_time

    return result


def run_production_readiness_test() -> Dict[str, Any]:
    """
    Run comprehensive production readiness test with auto_setup.

    Returns:
        Dictionary with all test results
    """
    print("🚀 Production Readiness Test with Auto-Setup")
    print("=" * 60)
    print("Testing all RAG pipelines with config-driven state resolution")
    print("")

    # Define pipelines to test
    pipelines_to_test = [
        "basic",
        "crag",
        "basic_rerank",  # Correct name for reranking pipeline
        "graphrag",
        "hybrid_graphrag",
    ]

    test_queries = [
        "What is diabetes?",
        "How does insulin work?",
        "What are the symptoms of heart disease?",
    ]

    results = {
        "start_time": time.time(),
        "pipeline_results": {},
        "summary": {},
        "success_rate": 0.0,
    }

    successful_pipelines = 0
    total_pipelines = len(pipelines_to_test)

    for pipeline_type in pipelines_to_test:
        # Test with first query
        test_result = test_pipeline_with_auto_setup(pipeline_type, test_queries[0])
        results["pipeline_results"][pipeline_type] = test_result

        if test_result["success"]:
            successful_pipelines += 1

        print("")  # Add spacing between tests

    # Calculate summary
    results["summary"] = {
        "total_pipelines": total_pipelines,
        "successful_pipelines": successful_pipelines,
        "failed_pipelines": total_pipelines - successful_pipelines,
        "success_rate": (successful_pipelines / total_pipelines) * 100,
        "total_execution_time": time.time() - results["start_time"],
    }

    # Print summary
    print("=" * 60)
    print("📊 PRODUCTION READINESS SUMMARY")
    print("=" * 60)
    print(f"Total Pipelines Tested: {total_pipelines}")
    print(f"Successful: {successful_pipelines} ✅")
    print(f"Failed: {total_pipelines - successful_pipelines} ❌")
    print(f"Success Rate: {results['summary']['success_rate']:.1f}%")
    print(f"Total Execution Time: {results['summary']['total_execution_time']:.2f}s")
    print("")

    # Detail results
    for pipeline_type, result in results["pipeline_results"].items():
        status = "✅ PASS" if result["success"] else "❌ FAIL"
        auto_setup_status = "✅" if result["auto_setup_worked"] else "❌"
        print(
            f"{status} {pipeline_type:15} (auto_setup: {auto_setup_status}, {result['execution_time']:.1f}s)"
        )
        if result["error"]:
            print(f"      Error: {result['error'][:100]}...")

    print("")

    # Final assessment
    if results["summary"]["success_rate"] >= 80:
        print("🎉 PRODUCTION READY!")
        print("Framework demonstrates excellent stability with auto_setup feature.")
        assessment = "READY"
    elif results["summary"]["success_rate"] >= 60:
        print("⚠️ MOSTLY READY")
        print("Framework is largely functional but some pipelines need attention.")
        assessment = "MOSTLY_READY"
    else:
        print("🔧 NEEDS WORK")
        print("Significant issues need resolution before production release.")
        assessment = "NEEDS_WORK"

    results["assessment"] = assessment

    return results


def main():
    """Main execution function."""
    setup_logging()

    try:
        results = run_production_readiness_test()

        # Exit with appropriate code
        if results["assessment"] == "READY":
            return 0
        elif results["assessment"] == "MOSTLY_READY":
            return 1
        else:
            return 2

    except Exception as e:
        print(f"❌ Production readiness test failed with error: {str(e)}")
        return 3


if __name__ == "__main__":
    sys.exit(main())
