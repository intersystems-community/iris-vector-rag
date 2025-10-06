#!/usr/bin/env python
"""
Test script to examine the actual interfaces of RAG pipelines.
This helps understand what methods each pipeline provides.
"""

import inspect
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, project_root)

from common.utils import get_embedding_func, get_llm_func
from iris_rag.config.manager import ConfigurationManager

# Pipeline imports
from iris_rag.pipelines.basic import BasicRAGPipeline
from iris_rag.pipelines.crag import CRAGPipeline
from iris_rag.pipelines.graphrag import GraphRAGPipeline
from iris_rag.pipelines.hybrid_ifind import HybridIFindRAGPipeline
from iris_rag.pipelines.hyde import HyDERAGPipeline
from iris_rag.pipelines.noderag import NodeRAGPipeline

PIPELINE_CLASSES = {
    "basic": BasicRAGPipeline,
    "hyde": HyDERAGPipeline,
    "crag": CRAGPipeline,
    "colbert": ColBERTRAGPipeline,
    "noderag": NodeRAGPipeline,
    "graphrag": GraphRAGPipeline,
    "hybrid_ifind": HybridIFindRAGPipeline,
}


def analyze_pipeline_interface(pipeline_name: str, pipeline_class) -> Dict[str, Any]:
    """
    Analyze the interface of a pipeline class.

    Args:
        pipeline_name: Name of the pipeline
        pipeline_class: The pipeline class to analyze

    Returns:
        Dictionary with interface analysis
    """
    result = {
        "pipeline_name": pipeline_name,
        "class_name": pipeline_class.__name__,
        "methods": {},
        "properties": [],
        "base_classes": [base.__name__ for base in pipeline_class.__bases__],
        "module": pipeline_class.__module__,
    }

    # Get all methods and properties
    for name, obj in inspect.getmembers(pipeline_class):
        if name.startswith("_"):
            continue  # Skip private methods

        if inspect.ismethod(obj) or inspect.isfunction(obj):
            # Get method signature
            try:
                sig = inspect.signature(obj)
                params = []
                for param_name, param in sig.parameters.items():
                    if param_name == "self":
                        continue
                    param_info = {
                        "name": param_name,
                        "default": (
                            str(param.default)
                            if param.default != inspect.Parameter.empty
                            else None
                        ),
                        "annotation": (
                            str(param.annotation)
                            if param.annotation != inspect.Parameter.empty
                            else None
                        ),
                    }
                    params.append(param_info)

                result["methods"][name] = {
                    "parameters": params,
                    "docstring": inspect.getdoc(obj) or "No documentation",
                    "is_abstract": inspect.isabstract(obj),
                }
            except Exception as e:
                result["methods"][name] = {"error": f"Could not inspect: {str(e)}"}
        elif isinstance(obj, property):
            result["properties"].append(name)

    return result


def test_pipeline_execution(pipeline_name: str, pipeline_class) -> Dict[str, Any]:
    """
    Test creating and using a pipeline instance.

    Args:
        pipeline_name: Name of the pipeline
        pipeline_class: The pipeline class to test

    Returns:
        Dictionary with test results
    """
    result = {
        "pipeline_name": pipeline_name,
        "initialization": {"success": False, "error": None},
        "execute_test": {"success": False, "response": None, "error": None},
        "required_init_params": [],
    }

    try:
        # Create minimal mocks
        config_manager = ConfigurationManager()

        # Mock connection manager
        mock_connection = MagicMock()

        connection_manager = MagicMock()
        connection_manager.return_value = mock_connection

        # Get stub functions
        llm_func = get_llm_func(provider="stub", model_name="test-model")
        embedding_func = get_embedding_func(provider="stub", mock=True)

        # Check initialization parameters
        init_sig = inspect.signature(pipeline_class.__init__)
        for param_name, param in init_sig.parameters.items():
            if param_name == "self":
                continue
            if param.default == inspect.Parameter.empty:
                result["required_init_params"].append(param_name)

        # Try to create instance
        if pipeline_name in ["crag", "colbert"]:
            # These need embedding_func
            pipeline = pipeline_class(
                iris_connector=connection_manager,
                config_manager=config_manager,
                llm_func=llm_func,
                embedding_func=embedding_func,
            )
        else:
            pipeline = pipeline_class(
                iris_connector=connection_manager,
                config_manager=config_manager,
                llm_func=llm_func,
            )

        result["initialization"]["success"] = True

        # Test execute method if it exists
        if hasattr(pipeline, "execute"):
            try:
                # Mock the vector store methods
                if hasattr(pipeline, "vector_store"):
                    pipeline.vector_store.search = lambda query, top_k=5: []
                    pipeline.vector_store.add_documents = lambda docs: None

                response = pipeline.query("What is IRIS?")
                result["execute_test"]["success"] = True
                result["execute_test"]["response"] = (
                    str(response)[:200] + "..."
                    if len(str(response)) > 200
                    else str(response)
                )
            except Exception as e:
                result["execute_test"]["error"] = str(e)
        else:
            result["execute_test"]["error"] = "No execute method found"

    except Exception as e:
        result["initialization"]["error"] = str(e)

    return result


def main():
    """Main analysis function."""
    print("RAG Pipeline Interface Analysis")
    print("=" * 80)
    print(f"Analyzing {len(PIPELINE_CLASSES)} pipeline types")
    print(f"Analysis started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Analyze all pipelines
    interface_results = {}
    execution_results = {}

    for pipeline_name, pipeline_class in PIPELINE_CLASSES.items():
        print(f"\n{'='*60}")
        print(f"Analyzing {pipeline_name} ({pipeline_class.__name__})")
        print(f"{'='*60}")

        # Analyze interface
        interface_results[pipeline_name] = analyze_pipeline_interface(
            pipeline_name, pipeline_class
        )

        # Test execution
        execution_results[pipeline_name] = test_pipeline_execution(
            pipeline_name, pipeline_class
        )

        # Print summary
        interface = interface_results[pipeline_name]
        execution = execution_results[pipeline_name]

        print(f"\nBase Classes: {', '.join(interface['base_classes'])}")
        print(f"Module: {interface['module']}")
        print(f"\nKey Methods Found:")

        important_methods = [
            "execute",
            "query",
            "load_documents",
            "ingest",
            "retrieve",
            "clear",
        ]
        for method in important_methods:
            if method in interface["methods"]:
                method_info = interface["methods"][method]
                if "parameters" in method_info:
                    params = [p["name"] for p in method_info["parameters"]]
                    print(f"  ✓ {method}({', '.join(params)})")
                else:
                    print(f"  ✓ {method}")
            else:
                print(f"  ✗ {method} (not found)")

        print(
            f"\nInitialization: {'✓ Success' if execution['initialization']['success'] else '✗ Failed'}"
        )
        if execution["initialization"]["error"]:
            print(f"  Error: {execution['initialization']['error']}")

        print(
            f"Execute Test: {'✓ Success' if execution['execute_test']['success'] else '✗ Failed'}"
        )
        if execution["execute_test"]["error"]:
            print(f"  Error: {execution['execute_test']['error']}")

    # Save detailed results
    output_dir = os.path.join(project_root, "outputs", "pipeline_status_tests")
    os.makedirs(output_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save analysis results
    analysis_file = os.path.join(
        output_dir, f"pipeline_interface_analysis_{timestamp}.json"
    )
    with open(analysis_file, "w") as f:
        json.dump(
            {
                "analysis_metadata": {
                    "timestamp": datetime.now().isoformat(),
                    "pipeline_count": len(PIPELINE_CLASSES),
                },
                "interface_analysis": interface_results,
                "execution_tests": execution_results,
            },
            f,
            indent=2,
        )

    print(f"\n{'='*80}")
    print("SUMMARY")
    print(f"{'='*80}")

    # Summary statistics
    working_pipelines = [
        name
        for name, result in execution_results.items()
        if result["initialization"]["success"] and result["execute_test"]["success"]
    ]
    initialized_only = [
        name
        for name, result in execution_results.items()
        if result["initialization"]["success"] and not result["execute_test"]["success"]
    ]
    broken_pipelines = [
        name
        for name, result in execution_results.items()
        if not result["initialization"]["success"]
    ]

    print(
        f"\nWorking Pipelines ({len(working_pipelines)}): {', '.join(working_pipelines) or 'None'}"
    )
    print(
        f"Initialized Only ({len(initialized_only)}): {', '.join(initialized_only) or 'None'}"
    )
    print(
        f"Broken Pipelines ({len(broken_pipelines)}): {', '.join(broken_pipelines) or 'None'}"
    )

    print(f"\nDetailed analysis saved to: {analysis_file}")

    # Create a simplified status report
    status_report = {
        "timestamp": datetime.now().isoformat(),
        "summary": {
            "total": len(PIPELINE_CLASSES),
            "working": len(working_pipelines),
            "initialized_only": len(initialized_only),
            "broken": len(broken_pipelines),
        },
        "pipelines": {},
    }

    for pipeline_name in PIPELINE_CLASSES:
        interface = interface_results[pipeline_name]
        execution = execution_results[pipeline_name]

        # Determine status
        if execution["initialization"]["success"]:
            if execution["execute_test"]["success"]:
                status = "working"
            else:
                status = "initialized_only"
        else:
            status = "broken"

        # Find available methods
        available_methods = []
        for method in ["execute", "query", "load_documents", "ingest", "retrieve"]:
            if method in interface["methods"]:
                available_methods.append(method)

        status_report["pipelines"][pipeline_name] = {
            "status": status,
            "available_methods": available_methods,
            "initialization_error": execution["initialization"]["error"],
            "execute_error": execution["execute_test"]["error"],
        }

    # Save status report
    status_file = os.path.join(output_dir, f"pipeline_status_report_{timestamp}.json")
    with open(status_file, "w") as f:
        json.dump(status_report, f, indent=2)

    print(f"Status report saved to: {status_file}")

    return 0 if broken_pipelines else 1


if __name__ == "__main__":
    sys.exit(main())
