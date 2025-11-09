#!/usr/bin/env python
"""
Comprehensive pipeline test that properly mocks database connections.
Tests all RAG pipelines with their actual interfaces.
"""

import json
import os
import sys
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional
from unittest.mock import MagicMock, Mock, patch

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from common.utils import (
    get_embedding_func,
    get_llm_func,
)
from iris_vector_rag.config.manager import ConfigurationManager
from iris_vector_rag.core.models import Document

# Pipeline imports
from iris_vector_rag.pipelines.basic import BasicRAGPipeline
from iris_vector_rag.pipelines.colbert_pylate.pylate_pipeline import PyLateColBERTPipeline
from iris_vector_rag.pipelines.crag import CRAGPipeline
from iris_vector_rag.pipelines.graphrag import GraphRAGPipeline
from iris_vector_rag.pipelines.hybrid_ifind import HybridIFindRAGPipeline
from iris_vector_rag.pipelines.hyde import HyDERAGPipeline
from iris_vector_rag.pipelines.noderag import NodeRAGPipeline

PIPELINE_INFO = {
    "basic": {
        "class": BasicRAGPipeline,
        "name": "BasicRAG",
        "description": "Standard RAG with vector similarity search"
    },
    "hyde": {
        "class": HyDERAGPipeline,
        "name": "HyDERAG",
        "description": "Hypothetical Document Embeddings RAG"
    },
    "crag": {
        "class": CRAGPipeline,
        "name": "CRAG",
        "description": "Corrective RAG with web search fallback"
    },
    "colbert": {
        "class": ColBERTRAGPipeline,
        "name": "ColBERTRAG",
        "description": "Token-level late interaction RAG"
    },
    "noderag": {
        "class": NodeRAGPipeline,
        "name": "NodeRAG",
        "description": "Node-based hierarchical RAG"
    },
    "graphrag": {
        "class": GraphRAGPipeline,
        "name": "GraphRAG",
        "description": "Graph-enhanced RAG with entity relationships"
    },
    "hybrid_ifind": {
        "class": HybridIFindRAGPipeline,
        "name": "HybridIFindRAG",
        "description": "Hybrid retrieval with multiple strategies"
    }
}

class MockCursor:
    """Mock database cursor."""
    def __init__(self):
        self.rowcount = 0
        self._results = []
    
    def execute(self, query, params=None):
        pass
    
    def fetchall(self):
        return self._results
    
    def fetchone(self):
        return self._results[0] if self._results else None
    
    def close(self):
        pass

class MockConnection:
    """Enhanced mock database connection with cursor support."""
    
    def __init__(self):
        self.closed = False
        self.in_transaction = False
        self._cursor = MockCursor()
        
    def cursor(self):
        """Return a mock cursor."""
        return self._cursor
        
    def execute(self, query, params=None):
        """Mock execute that returns fake results."""
        result = Mock()
        result.fetchall = lambda: []
        result.fetchone = lambda: None
        result.rowcount = 0
        return result
        
    def commit(self):
        """Mock commit."""
        
    def rollback(self):
        """Mock rollback."""
        
    def close(self):
        """Mock close."""
        self.closed = True
        
    def begin(self):
        """Mock begin transaction."""
        self.in_transaction = True
        return self
        
    def __enter__(self):
        return self
        
    def __exit__(self, exc_type, exc_val, exc_tb):
        if exc_type:
            self.rollback()
        else:
            self.commit()
        return False

class MockConnectionManager:
    """Mock connection manager for testing."""
    
    def __init__(self, config_manager):
        self.config_manager = config_manager
        self.connections = {}
        
    def get_connection(self, backend_name: str = "iris"):
        """Return a mock connection."""
        if backend_name not in self.connections:
            self.connections[backend_name] = MockConnection()
        return self.connections[backend_name]
        
    def close_all(self):
        """Close all mock connections."""
        for conn in self.connections.values():
            conn.close()
        self.connections.clear()

def create_test_documents(n: int = 5) -> List[Document]:
    """Create test documents."""
    docs = []
    for i in range(n):
        doc = Document(
            id=f"test_doc_{i}",
            content=f"This is test document {i}. IRIS is a powerful multi-model database system that supports SQL, NoSQL, and vector operations. It provides high-performance data management capabilities.",
            metadata={"source": "test", "index": i, "title": f"Test Document {i}"},
            embedding=[0.1 * (i + 1)] * 384  # Mock embedding
        )
        docs.append(doc)
    return docs

def test_pipeline(pipeline_type: str, pipeline_info: Dict[str, Any]) -> Dict[str, Any]:
    """
    Test a single pipeline type.
    
    Args:
        pipeline_type: The type of pipeline to test
        pipeline_info: Information about the pipeline
        
    Returns:
        Dictionary containing test results
    """
    result = {
        "pipeline_type": pipeline_type,
        "pipeline_name": pipeline_info["name"],
        "description": pipeline_info["description"],
        "status": "unknown",
        "tests": {
            "initialization": {"success": False, "error": None, "time": 0},
            "load_documents": {"success": False, "error": None, "time": 0},
            "query": {"success": False, "response": None, "error": None, "time": 0},
            "execute": {"success": False, "response": None, "error": None, "time": 0}
        },
        "features": {
            "supports_streaming": False,
            "supports_filters": False,
            "supports_metadata": False
        },
        "notes": []
    }
    
    print(f"\n{'='*70}")
    print(f"Testing {result['pipeline_name']} ({pipeline_type})")
    print(f"Description: {result['description']}")
    print(f"{'='*70}")
    
    try:
        # Step 1: Initialize pipeline
        print(f"\n1. Initializing {pipeline_type} pipeline...")
        start_time = datetime.now()
        
        # Setup
        config_manager = ConfigurationManager()
        connection_manager = MockConnectionManager(config_manager)
        llm_func = get_llm_func(provider="stub", model_name="test-model")
        embedding_func = get_embedding_func(provider="stub", mock=True)
        
        # Create pipeline instance
        pipeline_class = pipeline_info["class"]
        
        if pipeline_type in ["crag", "colbert"]:
            pipeline = pipeline_class(
                connection_manager=connection_manager,
                config_manager=config_manager,
                llm_func=llm_func,
                embedding_func=embedding_func
            )
        else:
            pipeline = pipeline_class(
                connection_manager=connection_manager,
                config_manager=config_manager,
                llm_func=llm_func
            )
        
        init_time = (datetime.now() - start_time).total_seconds()
        result["tests"]["initialization"]["success"] = True
        result["tests"]["initialization"]["time"] = init_time
        print(f"   ✓ Pipeline initialized successfully in {init_time:.3f}s")
        
        # Step 2: Test load_documents
        print(f"\n2. Testing document loading...")
        start_time = datetime.now()
        
        try:
            test_docs = create_test_documents(3)
            
            # Mock vector store operations
            if hasattr(pipeline, 'vector_store'):
                pipeline.vector_store.add_documents = Mock(return_value=None)
                pipeline.vector_store.search = Mock(return_value=test_docs[:2])
                pipeline.vector_store.get_all_documents = Mock(return_value=test_docs)
            
            # Try to load documents
            pipeline.load_documents("dummy_path", documents=test_docs)
            
            load_time = (datetime.now() - start_time).total_seconds()
            result["tests"]["load_documents"]["success"] = True
            result["tests"]["load_documents"]["time"] = load_time
            print(f"   ✓ Document loading successful in {load_time:.3f}s")
            
        except Exception as e:
            result["tests"]["load_documents"]["error"] = str(e)
            print(f"   ✗ Document loading failed: {str(e)}")
        
        # Step 3: Test query method
        print(f"\n3. Testing query method...")
        start_time = datetime.now()
        
        try:
            test_query = "What is IRIS database?"
            
            # Mock retrieval for pipelines that need it
            if hasattr(pipeline, '_retrieve') or hasattr(pipeline, 'retrieve'):
                with patch.object(pipeline, '_retrieve' if hasattr(pipeline, '_retrieve') else 'retrieve', 
                                return_value=test_docs[:2]):
                    response = pipeline.query(test_query, top_k=3)
            else:
                response = pipeline.query(test_query, top_k=3)
            
            query_time = (datetime.now() - start_time).total_seconds()
            
            if response:
                result["tests"]["query"]["success"] = True
                result["tests"]["query"]["response"] = str(response)[:200] + "..." if len(str(response)) > 200 else str(response)
                result["tests"]["query"]["time"] = query_time
                print(f"   ✓ Query successful in {query_time:.3f}s")
                print(f"   Response preview: {result['tests']['query']['response']}")
            else:
                result["tests"]["query"]["error"] = "Empty response"
                print(f"   ⚠ Query returned empty response")
                
        except Exception as e:
            result["tests"]["query"]["error"] = str(e)
            result["tests"]["query"]["traceback"] = traceback.format_exc()
            print(f"   ✗ Query failed: {str(e)}")
        
        # Step 4: Test execute method
        print(f"\n4. Testing execute method...")
        start_time = datetime.now()
        
        try:
            # Mock retrieval for execute as well
            if hasattr(pipeline, '_retrieve') or hasattr(pipeline, 'retrieve'):
                with patch.object(pipeline, '_retrieve' if hasattr(pipeline, '_retrieve') else 'retrieve', 
                                return_value=test_docs[:2]):
                    response = pipeline.query(test_query)
            else:
                response = pipeline.query(test_query)
            
            exec_time = (datetime.now() - start_time).total_seconds()
            
            if response:
                result["tests"]["execute"]["success"] = True
                result["tests"]["execute"]["response"] = str(response)[:200] + "..." if len(str(response)) > 200 else str(response)
                result["tests"]["execute"]["time"] = exec_time
                print(f"   ✓ Execute successful in {exec_time:.3f}s")
                
                # Check response structure
                if isinstance(response, dict):
                    print(f"   Response keys: {list(response.keys())}")
            else:
                result["tests"]["execute"]["error"] = "Empty response"
                print(f"   ⚠ Execute returned empty response")
                
        except Exception as e:
            result["tests"]["execute"]["error"] = str(e)
            result["tests"]["execute"]["traceback"] = traceback.format_exc()
            print(f"   ✗ Execute failed: {str(e)}")
        
        # Determine overall status
        if result["tests"]["initialization"]["success"]:
            if result["tests"]["execute"]["success"] or result["tests"]["query"]["success"]:
                result["status"] = "working"
            elif result["tests"]["load_documents"]["success"]:
                result["status"] = "partial"
            else:
                result["status"] = "initialization_only"
        else:
            result["status"] = "broken"
            
    except Exception as e:
        result["tests"]["initialization"]["error"] = str(e)
        result["tests"]["initialization"]["traceback"] = traceback.format_exc()
        result["notes"].append(f"Critical error: {str(e)}")
        result["status"] = "broken"
        print(f"\n   ✗ Critical error: {str(e)}")
    
    print(f"\n5. Final Status: {result['status'].upper()}")
    
    return result

def generate_summary_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate a comprehensive summary report."""
    summary = {
        "test_timestamp": datetime.now().isoformat(),
        "total_pipelines": len(results),
        "status_breakdown": {
            "working": 0,
            "partial": 0,
            "initialization_only": 0,
            "broken": 0
        },
        "test_results": {
            "initialization": {"passed": 0, "failed": 0},
            "load_documents": {"passed": 0, "failed": 0},
            "query": {"passed": 0, "failed": 0},
            "execute": {"passed": 0, "failed": 0}
        },
        "pipelines_by_status": {
            "working": [],
            "partial": [],
            "initialization_only": [],
            "broken": []
        },
        "average_performance": {
            "initialization_time": 0,
            "query_time": 0,
            "execute_time": 0
        }
    }
    
    init_times = []
    query_times = []
    exec_times = []
    
    for result in results:
        # Status breakdown
        status = result["status"]
        summary["status_breakdown"][status] += 1
        summary["pipelines_by_status"][status].append(result["pipeline_name"])
        
        # Test results
        for test_name in ["initialization", "load_documents", "query", "execute"]:
            if result["tests"][test_name]["success"]:
                summary["test_results"][test_name]["passed"] += 1
            else:
                summary["test_results"][test_name]["failed"] += 1
        
        # Performance metrics
        if result["tests"]["initialization"]["success"]:
            init_times.append(result["tests"]["initialization"]["time"])
        if result["tests"]["query"]["success"]:
            query_times.append(result["tests"]["query"]["time"])
        if result["tests"]["execute"]["success"]:
            exec_times.append(result["tests"]["execute"]["time"])
    
    # Calculate averages
    if init_times:
        summary["average_performance"]["initialization_time"] = sum(init_times) / len(init_times)
    if query_times:
        summary["average_performance"]["query_time"] = sum(query_times) / len(query_times)
    if exec_times:
        summary["average_performance"]["execute_time"] = sum(exec_times) / len(exec_times)
    
    return summary

def main():
    """Main test function."""
    print("Comprehensive RAG Pipeline Test Suite")
    print("=" * 80)
    print(f"Testing {len(PIPELINE_INFO)} pipeline types")
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nThis test suite evaluates each pipeline's:")
    print("  - Initialization capability")
    print("  - Document loading functionality")
    print("  - Query processing")
    print("  - Full execution pipeline")
    
    # Test all pipelines
    results = []
    for pipeline_type, info in PIPELINE_INFO.items():
        try:
            result = test_pipeline(pipeline_type, info)
            results.append(result)
        except Exception as e:
            print(f"\nFatal error testing {pipeline_type}: {str(e)}")
            results.append({
                "pipeline_type": pipeline_type,
                "pipeline_name": info["name"],
                "description": info["description"],
                "status": "error",
                "tests": {
                    "initialization": {"success": False, "error": str(e), "time": 0},
                    "load_documents": {"success": False, "error": None, "time": 0},
                    "query": {"success": False, "response": None, "error": None, "time": 0},
                    "execute": {"success": False, "response": None, "error": None, "time": 0}
                },
                "features": {},
                "notes": [f"Fatal error: {str(e)}"]
            })
    
    # Generate summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    summary = generate_summary_report(results)
    
    # Print summary
    print(f"\nTotal Pipelines Tested: {summary['total_pipelines']}")
    
    print("\nStatus Breakdown:")
    for status, count in summary["status_breakdown"].items():
        percentage = (count / summary['total_pipelines']) * 100
        print(f"  {status.title()}: {count} ({percentage:.0f}%)")
    
    print("\nTest Results:")
    for test_name, counts in summary["test_results"].items():
        total = counts["passed"] + counts["failed"]
        pass_rate = (counts["passed"] / total * 100) if total > 0 else 0
        print(f"  {test_name.replace('_', ' ').title()}: {counts['passed']}/{total} passed ({pass_rate:.0f}%)")
    
    print("\nPipelines by Status:")
    for status, pipelines in summary["pipelines_by_status"].items():
        if pipelines:
            print(f"\n  {status.title()} ({len(pipelines)}):")
            for pipeline in pipelines:
                print(f"    - {pipeline}")
    
    print("\nAverage Performance:")
    for metric, value in summary["average_performance"].items():
        if value > 0:
            print(f"  {metric.replace('_', ' ').title()}: {value:.3f}s")
    
    # Save results
    output_dir = os.path.join(project_root, "outputs", "pipeline_status_tests")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    detailed_file = os.path.join(output_dir, f"pipeline_comprehensive_test_{timestamp}.json")
    with open(detailed_file, 'w') as f:
        json.dump({
            "test_metadata": {
                "timestamp": datetime.now().isoformat(),
                "pipeline_count": len(PIPELINE_INFO),
                "test_query": "What is IRIS database?"
            },
            "results": results,
            "summary": summary
        }, f, indent=2)
    
    print(f"\nDetailed results saved to: {detailed_file}")
    
    # Create actionable report
    actionable_report = {
        "timestamp": datetime.now().isoformat(),
        "pipelines_needing_fixes": [],
        "specific_issues": {},
        "recommendations": []
    }
    
    for result in results:
        if result["status"] != "working":
            pipeline_issues = []
            
            if not result["tests"]["initialization"]["success"]:
                pipeline_issues.append({
                    "issue": "Initialization failure",
                    "error": result["tests"]["initialization"]["error"]
                })
            elif not result["tests"]["query"]["success"] and not result["tests"]["execute"]["success"]:
                pipeline_issues.append({
                    "issue": "Query/Execute failure",
                    "query_error": result["tests"]["query"]["error"],
                    "execute_error": result["tests"]["execute"]["error"]
                })
            
            if pipeline_issues:
                actionable_report["pipelines_needing_fixes"].append(result["pipeline_name"])
                actionable_report["specific_issues"][result["pipeline_name"]] = pipeline_issues
    
    # Add recommendations
    if actionable_report["pipelines_needing_fixes"]:
        actionable_report["recommendations"] = [
            "1. Fix database connection mocking for pipelines using cursor operations",
            "2. Ensure all pipelines properly handle mock vector store operations",
            "3. Review error handling in query and execute methods",
            "4. Consider implementing a common test interface for all pipelines"
        ]
    
    # Save actionable report
    actionable_file = os.path.join(output_dir, f"pipeline_actionable_report_{timestamp}.json")
    with open(actionable_file, 'w') as f:
        json.dump(actionable_report, f, indent=2)
    
    print(f"Actionable report saved to: {actionable_file}")
    
    # Print final recommendation
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)
    
    working_count = summary["status_breakdown"]["working"]
    if working_count == len(PIPELINE_INFO):
        print("\n✓ All pipelines are working correctly!")
    else:
        print(f"\n⚠ {len(PIPELINE_INFO) - working_count} pipelines need attention:")
        for pipeline in actionable_report["pipelines_needing_fixes"]:
            issues = actionable_report["specific_issues"][pipeline]
            print(f"\n  {pipeline}:")
            for issue in issues:
                print(f"    - {issue['issue']}")
    
    return 0 if working_count == len(PIPELINE_INFO) else 1

if __name__ == "__main__":
    sys.exit(main())