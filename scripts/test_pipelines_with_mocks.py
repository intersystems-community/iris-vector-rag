#!/usr/bin/env python
"""
Pipeline status test with mock connections.
Tests all available RAG pipeline types without requiring a real IRIS connection.
"""

import os
import sys
import json
import traceback
from datetime import datetime
from typing import Dict, Any, List, Optional
from unittest.mock import Mock, MagicMock, patch

# Add the project root to the path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, project_root)

from common.utils import get_llm_func, get_embedding_func, 
from iris_rag.core.models import Document

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

class MockConnection:
    """Mock database connection for testing."""
    
    def __init__(self):
        self.closed = False
        self.in_transaction = False
        
    def execute(self, query, params=None):
        """Mock execute that returns fake results."""
        result = Mock()
        result.fetchall = lambda: []
        result.fetchone = lambda: None
        result.rowcount = 0
        return result
        
    def commit(self):
        """Mock commit."""
        pass
        
    def rollback(self):
        """Mock rollback."""
        pass
        
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

def create_mock_documents(n: int = 5) -> List[Document]:
    """Create mock documents for testing."""
    docs = []
    for i in range(n):
        doc = Document(
            id=f"doc_{i}",
            content=f"This is test document {i}. IRIS is a database system.",
            metadata={"source": "test", "index": i},
            embedding=[0.1 * i] * 384  # Mock embedding
        )
        docs.append(doc)
    return docs

def test_pipeline_with_mocks(pipeline_type: str) -> Dict[str, Any]:
    """
    Test a single pipeline type with mock connections.
    
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
        "ingestion_test": {"success": False, "error": None},
        "query_test": {"success": False, "response": None, "error": None},
        "method_check": {"missing": [], "present": []},
        "notes": []
    }
    
    print(f"\n{'='*60}")
    print(f"Testing {result['pipeline_name']} ({pipeline_type})")
    print(f"{'='*60}")
    
    try:
        # Step 1: Import and setup mocks
        print(f"1. Setting up mocks and importing pipeline...")
        
        from iris_rag.config.manager import ConfigurationManager
        
        # Create mock config manager
        mock_config = ConfigurationManager()
        mock_conn_manager = MockConnectionManager(mock_config)
        
        # Get stub functions
        llm_func = get_llm_func(provider="stub", model_name="test-model")
        embedding_func = get_embedding_func(provider="stub", mock=True)
        
        # Import the specific pipeline class
        if pipeline_type == "basic":
            from iris_rag.pipelines.basic import BasicRAGPipeline
            pipeline_class = BasicRAGPipeline
        elif pipeline_type == "hyde":
            from iris_rag.pipelines.hyde import HyDERAGPipeline
            pipeline_class = HyDERAGPipeline
        elif pipeline_type == "crag":
            from iris_rag.pipelines.crag import CRAGPipeline
            pipeline_class = CRAGPipeline
        elif pipeline_type == "colbert":
            from iris_rag.pipelines.colbert import ColBERTRAGPipeline
            pipeline_class = ColBERTRAGPipeline
        elif pipeline_type == "noderag":
            from iris_rag.pipelines.noderag import NodeRAGPipeline
            pipeline_class = NodeRAGPipeline
        elif pipeline_type == "graphrag":
            from iris_rag.pipelines.graphrag import GraphRAGPipeline
            pipeline_class = GraphRAGPipeline
        elif pipeline_type == "hybrid_ifind":
            from iris_rag.pipelines.hybrid_ifind import HybridIFindRAGPipeline
            pipeline_class = HybridIFindRAGPipeline
        else:
            raise ValueError(f"Unknown pipeline type: {pipeline_type}")
        
        # Create pipeline instance with mocks
        print(f"2. Initializing {pipeline_type} pipeline...")
        
        # Special handling for pipelines that need embedding_func
        if pipeline_type in ["crag", "colbert"]:
            pipeline = pipeline_class(
                connection_manager=mock_conn_manager,
                config_manager=mock_config,
                llm_func=llm_func,
                embedding_func=embedding_func
            )
        else:
            pipeline = pipeline_class(
                connection_manager=mock_conn_manager,
                config_manager=mock_config,
                llm_func=llm_func
            )
        
        result["initialization"]["success"] = True
        print(f"   ✓ Pipeline initialized successfully")
        
        # Step 3: Check methods
        print(f"3. Checking pipeline methods...")
        required_methods = ["ingest", "query", "clear", "get_documents"]
        optional_methods = ["setup", "validate", "get_stats"]
        
        for method in required_methods:
            if hasattr(pipeline, method):
                result["method_check"]["present"].append(method)
            else:
                result["method_check"]["missing"].append(method)
        
        if result["method_check"]["missing"]:
            print(f"   ⚠ Missing required methods: {', '.join(result['method_check']['missing'])}")
            result["notes"].append(f"Missing methods: {', '.join(result['method_check']['missing'])}")
        else:
            print(f"   ✓ All required methods present")
        
        # Step 4: Test ingestion (mock)
        print(f"4. Testing document ingestion...")
        try:
            mock_docs = create_mock_documents(3)
            
            # Mock the actual database operations
            with patch.object(pipeline, '_store_documents', return_value=None):
                with patch.object(pipeline, '_store_embeddings', return_value=None):
                    pipeline.ingest(mock_docs)
            
            result["ingestion_test"]["success"] = True
            print(f"   ✓ Ingestion test passed (mocked)")
            
        except Exception as e:
            result["ingestion_test"]["error"] = str(e)
            result["notes"].append(f"Ingestion failed: {str(e)}")
            print(f"   ✗ Ingestion failed: {str(e)}")
        
        # Step 5: Test query
        print(f"5. Testing query functionality...")
        test_query = "What is IRIS?"
        
        try:
            # Mock the retrieval to return test documents
            mock_retrieved = create_mock_documents(2)
            
            with patch.object(pipeline, 'retrieve', return_value=mock_retrieved):
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
    
    # Determine final status
    if result["status"] == "unknown":
        if result["initialization"]["success"]:
            if result["query_test"]["success"]:
                result["status"] = "working"
            else:
                result["status"] = "initialization_only"
        else:
            result["status"] = "broken"
    
    print(f"\nFinal Status: {result['status'].upper()}")
    
    return result

def generate_summary_report(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Generate a summary report from all test results."""
    summary = {
        "test_timestamp": datetime.now().isoformat(),
        "test_type": "mock_connection_test",
        "total_pipelines": len(results),
        "status_counts": {},
        "working_pipelines": [],
        "broken_pipelines": [],
        "partial_pipelines": [],
        "initialization_only": [],
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
        elif status == "partial":
            summary["partial_pipelines"].append(pipeline_name)
        elif status == "initialization_only":
            summary["initialization_only"].append(pipeline_name)
        
        # Store detailed info
        summary["details_by_pipeline"][pipeline_name] = {
            "status": status,
            "notes": result["notes"],
            "initialization_error": result["initialization"]["error"],
            "query_error": result["query_test"]["error"],
            "missing_methods": result["method_check"]["missing"]
        }
    
    return summary

def main():
    """Main test function."""
    print("RAG Pipeline Status Test (with Mock Connections)")
    print("=" * 80)
    print(f"Testing {len(PIPELINE_TYPES)} pipeline types")
    print(f"Test started at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("\nNote: This test uses mock database connections to test pipeline logic")
    print("without requiring a real IRIS database connection.")
    
    # Test all pipelines
    results = []
    for pipeline_type in PIPELINE_TYPES:
        try:
            result = test_pipeline_with_mocks(pipeline_type)
            results.append(result)
        except Exception as e:
            print(f"\nCritical error testing {pipeline_type}: {str(e)}")
            results.append({
                "pipeline_type": pipeline_type,
                "pipeline_name": PIPELINE_NAMES.get(pipeline_type, pipeline_type),
                "status": "error",
                "initialization": {"success": False, "error": str(e)},
                "ingestion_test": {"success": False, "error": None},
                "query_test": {"success": False, "response": None, "error": None},
                "method_check": {"missing": [], "present": []},
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
    
    print(f"\nInitialization Only ({len(summary['initialization_only'])}):")
    for pipeline in summary["initialization_only"]:
        print(f"  ⚠ {pipeline} (initialized but query failed)")
    
    print(f"\nBroken Pipelines ({len(summary['broken_pipelines'])}):")
    for pipeline in summary["broken_pipelines"]:
        print(f"  ✗ {pipeline}")
    
    print(f"\nPartial Functionality ({len(summary['partial_pipelines'])}):")
    for pipeline in summary["partial_pipelines"]:
        print(f"  ⚠ {pipeline}")
    
    # Save detailed results
    output_dir = os.path.join(project_root, "outputs", "pipeline_status_tests")
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed results
    detailed_file = os.path.join(output_dir, f"pipeline_mock_test_detailed_{timestamp}.json")
    with open(detailed_file, 'w') as f:
        json.dump({
            "test_metadata": {
                "timestamp": datetime.now().isoformat(),
                "test_type": "mock_connection_test",
                "pipeline_types": PIPELINE_TYPES,
                "test_query": "What is IRIS?"
            },
            "results": results,
            "summary": summary
        }, f, indent=2)
    
    # Save summary report
    summary_file = os.path.join(output_dir, f"pipeline_mock_test_summary_{timestamp}.json") 
    with open(summary_file, 'w') as f:
        json.dump(summary, f, indent=2)
    
    print(f"\nDetailed results saved to: {detailed_file}")
    print(f"Summary report saved to: {summary_file}")
    
    # Print detailed errors for broken pipelines
    if summary["broken_pipelines"]:
        print("\n" + "=" * 80)
        print("DETAILED ERROR INFORMATION")
        print("=" * 80)
        for pipeline_name in summary["broken_pipelines"]:
            details = summary["details_by_pipeline"][pipeline_name]
            print(f"\n{pipeline_name}:")
            if details["initialization_error"]:
                print(f"  Initialization Error: {details['initialization_error']}")
            for note in details["notes"]:
                print(f"  Note: {note}")
    
    # Return exit code based on results
    if summary["broken_pipelines"]:
        return 1  # Exit with error if any pipelines are broken
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)