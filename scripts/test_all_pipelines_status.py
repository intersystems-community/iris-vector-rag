#!/usr/bin/env python
"""
Comprehensive pipeline status test script.
Tests all available RAG pipeline types and reports their status.
"""

import os
import sys
import json
import traceback
from datetime import datetime
from typing import Dict, Any, List

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from iris_rag import create_pipeline
from common.utils import get_llm_func, get_embedding_func

# List of all pipeline types to test
PIPELINE_TYPES = [
    "basic",
    "hyde",
    "crag",
    "colbert",
    "noderag",
    "graphrag",
    "hybrid_ifind"
]

# Mapping of pipeline types to their full names
PIPELINE_NAMES = {
    "basic": "BasicRAG",
    "hyde": "HyDERAG",
    "crag": "CRAG",
    "colbert": "ColBERTRAG",
    "noderag": "NodeRAG",
    "graphrag": "GraphRAG",
    "hybrid_ifind": "HybridIFindRAG"
}

def test_pipeline(pipeline_type: str) -> Dict[str, Any]:
    """
    Test a single pipeline type.
    
    Args:
        pipeline_type: The type of pipeline to test
        
    Returns:
        Dictionary containing test results
    """
    result = {
        "pipeline_type": pipeline_type,
        "pipeline_name": PIPELINE_NAMES.get(pipeline_type, pipeline_type),
        "status": "unknown",
        "initialization": {"success": False, "error": None},
        "query_test": {"success": False, "response": None, "error": None},
        "notes": []
    }
    
    print(f"\n{'='*60}")
    print(f"Testing {result['pipeline_name']} ({pipeline_type})")
    print(f"{'='*60}")
    
    try:
        # Step 1: Try to initialize the pipeline
        print(f"1. Initializing {pipeline_type} pipeline...")
        
        # Get LLM and embedding functions
        llm_func = get_llm_func(provider="stub", model_name="test-model")
        embedding_func = get_embedding_func(provider="stub", mock=True)
        
        # Create pipeline with validation disabled for initial testing
        pipeline = create_pipeline(
            pipeline_type=pipeline_type,
            llm_func=llm_func,
            embedding_func=embedding_func,
            validate_requirements=False,  # Disable validation to test raw functionality
            auto_setup=False
        )
        
        result["initialization"]["success"] = True
        print(f"   ✓ Pipeline initialized successfully")
        
        # Step 2: Run a simple query
        print(f"2. Running test query...")
        test_query = "What is IRIS?"
        
        try:
            response = pipeline.query(test_query)
            
            if response:
                result["query_test"]["success"] = True
                result["query_test"]["response"] = str(response)[:200] + "..." if len(str(response)) > 200 else str(response)
                print(f"   ✓ Query executed successfully")
                print(f"   Response preview: {result['query_test']['response']}")
                result["status"] = "working"
            else:
                result["query_test"]["error"] = "Empty response received"
                result["notes"].append("Pipeline returns empty responses")
                result["status"] = "partial"
                print(f"   ⚠ Query returned empty response")
                
        except Exception as e:
            result["query_test"]["error"] = str(e)
            result["query_test"]["traceback"] = traceback.format_exc()
            result["notes"].append(f"Query failed: {str(e)}")
            result["status"] = "initialization_only"
            print(f"   ✗ Query failed: {str(e)}")
            
    except Exception as e:
        result["initialization"]["error"] = str(e)
        result["initialization"]["traceback"] = traceback.format_exc()
        result["notes"].append(f"Initialization failed: {str(e)}")
        result["status"] = "broken"
        print(f"   ✗ Initialization failed: {str(e)}")
    
    # Add additional checks
    if result["status"] == "working":
        print(f"3. Additional checks...")
        try:
            # Test if pipeline has required methods
            required_methods = ["ingest", "query", "clear"]
            missing_methods = []
            
            for method in required_methods:
                if not hasattr(pipeline, method):
                    missing_methods.append(method)
            
            if missing_methods:
                result["notes"].append(f"Missing methods: {', '.join(missing_methods)}")
                result["status"] = "partial"
                print(f"   ⚠ Missing methods: {', '.join(missing_methods)}")
            else:
                print(f"   ✓ All required methods present")
                
        except Exception as e:
            result["notes"].append(f"Additional checks failed: {str(e)}")
            print(f"   ⚠ Additional checks failed: {str(e)}")
    
    print(f"\nFinal Status: {result['status'].upper()}")
    
    return result

def generate_summary_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate a summary report from all test results."""
    summary = {
        "test_timestamp": datetime.now().isoformat(),
        "total_pipelines": len(results),
        "status_counts": {},
        "working_pipelines": [],
        "broken_pipelines": [],
        "partial_pipelines": [],
        "details_by_pipeline": {}
    }
    
    for result in results:
        status = result["status"]
        pipeline_name = result["pipeline_name"]
        
        # Count statuses
        summary["status_counts"][status] = summary["status_counts"].get(status, 0) + 1
        
        # Categorize pipelines
        if status == "working":
            summary["working_pipelines"].append(pipeline_name)
        elif status == "broken":
            summary["broken_pipelines"].append(pipeline_name)
        elif status in ["partial", "initialization_only"]:
            summary["partial_pipelines"].append(pipeline_name)
        
        # Store detailed info
        summary["details_by_pipeline"][pipeline_name] = {
            "status": status,
            "notes": result["notes"],
            "initialization_error": result["initialization"]["error"],
            "query_error": result["query_test"]["error"]
        }
    
    return summary

def main():
    """Main test function."""
    print("RAG Pipeline Status Test")
    print("=" * 80)
    print(f"Testing {len(PIPELINE_TYPES)} pipeline types")
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Test all pipelines
    results = []
    for pipeline_type in PIPELINE_TYPES:
        try:
            result = test_pipeline(pipeline_type)
            results.append(result)
        except Exception as e:
            print(f"\nCritical error testing {pipeline_type}: {str(e)}")
            results.append({
                "pipeline_type": pipeline_type,
                "pipeline_name": PIPELINE_NAMES.get(pipeline_type, pipeline_type),
                "status": "error",
                "initialization": {"success": False, "error": str(e)},
                "query_test": {"success": False, "response": None, "error": None},
                "notes": [f"Critical error: {str(e)}"]
            })
    
    # Generate summary
    print("\n" + "=" * 80)
    print("SUMMARY REPORT")
    print("=" * 80)
    
    summary = generate_summary_report(results)
    
    print(f"\nTotal Pipelines Tested: {summary['total_pipelines']}")
    print(f"\nStatus Breakdown:")
    for status, count in summary["status_counts"].items():
        print(f"  - {status}: {count}")
    
    print(f"\nWorking Pipelines ({len(summary['working_pipelines'])}):")
    for pipeline in summary["working_pipelines"]:
        print(f"  ✓ {pipeline}")
    
    print(f"\nBroken Pipelines ({len(summary['broken_pipelines'])}):")
    for pipeline in summary["broken_pipelines"]:
        print(f"  ✗ {pipeline}")
    
    print(f"\nPartial/Limited Functionality ({len(summary['partial_pipelines'])}):")
    for pipeline in summary["partial_pipelines"]:
        print(f"  ⚠ {pipeline}")
    
    # Save detailed results
    output_dir = os.path.join(project_root, "outputs", "pipeline_status_tests")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    detailed_file = os.path.join(output_dir, f"pipeline_test_detailed_{timestamp}.json")
    with open(detailed_file, 'w') as f:
        json.dump({
            "test_metadata": {
                "timestamp": datetime.now().isoformat(),
                "pipeline_types": PIPELINE_TYPES,
                "test_query": "What is IRIS?"
            },
            "results": results,
            "summary": summary
        }, f, indent=2)
    
    # Save summary report
    summary_file = os.path.join(output_dir, f"pipeline_test_summary_{timestamp}.json")
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nDetailed results saved to: {detailed_file}")
    print(f"Summary report saved to: {summary_file}")
    
    # Return exit code based on results
    if summary["broken_pipelines"]:
        return 1  # Exit with error if any pipelines are broken
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)