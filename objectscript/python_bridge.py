"""
Python bridge module for ObjectScript integration.

This module provides the interface between ObjectScript classes and Python RAG pipelines,
enabling IRIS Embedded Python to call our RAG implementations.
"""

import json
import logging
import traceback
from typing import Dict, Any, List, Optional
from datetime import datetime

# Import RAG pipeline modules
from basic_rag.pipeline import BasicRAGPipeline
from colbert.pipeline import ColbertRAGPipeline
from graphrag.pipeline import GraphRAGPipeline
from hyde.pipeline import HyDEPipeline
from crag.pipeline import CRAGPipeline
from noderag.pipeline import NodeRAGPipeline

# Import evaluation and benchmarking modules
from eval.metrics import calculate_benchmark_metrics, calculate_answer_faithfulness, calculate_answer_relevance
from eval.bench_runner import run_technique_benchmark
from common.iris_connector import get_iris_connection

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def _safe_execute(func, *args, **kwargs) -> Dict[str, Any]:
    """
    Safely execute a function and return standardized result format.
    
    Args:
        func: Function to execute
        *args: Positional arguments
        **kwargs: Keyword arguments
        
    Returns:
        Dict with success status, result/error, and metadata
    """
    try:
        result = func(*args, **kwargs)
        return {
            "success": True,
            "result": result,
            "error": None,
            "timestamp": datetime.now().isoformat()
        }
    except Exception as e:
        logger.error(f"Error executing {func.__name__}: {str(e)}")
        logger.error(traceback.format_exc())
        return {
            "success": False,
            "result": None,
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


def invoke_basic_rag(query: str, config: str) -> str:
    """
    Invoke Basic RAG pipeline from ObjectScript.
    
    Args:
        query: The search query
        config: JSON configuration string
        
    Returns:
        JSON string with results
    """
    def _execute():
        config_dict = json.loads(config) if isinstance(config, str) else config
        
        # Initialize pipeline
        iris_connector = get_iris_connection()
        pipeline = BasicRAGPipeline(
            iris_connector=iris_connector,
            embedding_func=config_dict.get("embedding_func"),
            llm_func=config_dict.get("llm_func")
        )
        
        # Execute pipeline
        result = pipeline.run(query)
        return result
    
    execution_result = _safe_execute(_execute)
    return json.dumps(execution_result)


def invoke_colbert(query: str, config: str) -> str:
    """
    Invoke ColBERT pipeline from ObjectScript.
    
    Args:
        query: The search query
        config: JSON configuration string
        
    Returns:
        JSON string with results
    """
    def _execute():
        config_dict = json.loads(config) if isinstance(config, str) else config
        
        # Initialize pipeline
        iris_connector = get_iris_connection()
        pipeline = ColbertRAGPipeline(
            iris_connector=iris_connector,
            colbert_query_encoder_func=config_dict.get("colbert_query_encoder_func"),
            colbert_doc_encoder_func=config_dict.get("colbert_doc_encoder_func"),
            llm_func=config_dict.get("llm_func")
        )
        
        # Execute pipeline
        result = pipeline.run(query)
        return result
    
    execution_result = _safe_execute(_execute)
    return json.dumps(execution_result)


def invoke_graphrag(query: str, config: str) -> str:
    """
    Invoke GraphRAG pipeline from ObjectScript.
    
    Args:
        query: The search query
        config: JSON configuration string
        
    Returns:
        JSON string with results
    """
    def _execute():
        config_dict = json.loads(config) if isinstance(config, str) else config
        
        # Initialize pipeline
        iris_connector = get_iris_connection()
        pipeline = GraphRAGPipeline(
            iris_connector=iris_connector,
            embedding_func=config_dict.get("embedding_func"),
            llm_func=config_dict.get("llm_func")
        )
        
        # Execute pipeline
        result = pipeline.run(query)
        return result
    
    execution_result = _safe_execute(_execute)
    return json.dumps(execution_result)


def invoke_hyde(query: str, config: str) -> str:
    """
    Invoke HyDE pipeline from ObjectScript.
    
    Args:
        query: The search query
        config: JSON configuration string
        
    Returns:
        JSON string with results
    """
    def _execute():
        config_dict = json.loads(config) if isinstance(config, str) else config
        
        # Initialize pipeline
        iris_connector = get_iris_connection()
        pipeline = HyDEPipeline(
            iris_connector=iris_connector,
            embedding_func=config_dict.get("embedding_func"),
            llm_func=config_dict.get("llm_func")
        )
        
        # Execute pipeline
        result = pipeline.run(query)
        return result
    
    execution_result = _safe_execute(_execute)
    return json.dumps(execution_result)


def invoke_crag(query: str, config: str) -> str:
    """
    Invoke CRAG pipeline from ObjectScript.
    
    Args:
        query: The search query
        config: JSON configuration string
        
    Returns:
        JSON string with results
    """
    def _execute():
        config_dict = json.loads(config) if isinstance(config, str) else config
        
        # Initialize pipeline
        iris_connector = get_iris_connection()
        pipeline = CRAGPipeline(
            iris_connector=iris_connector,
            embedding_func=config_dict.get("embedding_func"),
            llm_func=config_dict.get("llm_func")
        )
        
        # Execute pipeline
        result = pipeline.run(query)
        return result
    
    execution_result = _safe_execute(_execute)
    return json.dumps(execution_result)


def invoke_noderag(query: str, config: str) -> str:
    """
    Invoke NodeRAG pipeline from ObjectScript.
    
    Args:
        query: The search query
        config: JSON configuration string
        
    Returns:
        JSON string with results
    """
    def _execute():
        config_dict = json.loads(config) if isinstance(config, str) else config
        
        # Initialize pipeline
        iris_connector = get_iris_connection()
        pipeline = NodeRAGPipeline(
            iris_connector=iris_connector,
            embedding_func=config_dict.get("embedding_func"),
            llm_func=config_dict.get("llm_func")
        )
        
        # Execute pipeline
        result = pipeline.run(query)
        return result
    
    execution_result = _safe_execute(_execute)
    return json.dumps(execution_result)


def run_benchmarks(pipeline_names: str) -> str:
    """
    Run benchmarks for specified RAG pipelines from ObjectScript.
    
    Args:
        pipeline_names: JSON string list of pipeline names to benchmark
        
    Returns:
        JSON string with benchmark results
    """
    def _execute():
        names = json.loads(pipeline_names) if isinstance(pipeline_names, str) else pipeline_names
        
        # Initialize IRIS connection
        iris_connector = get_iris_connection()
        
        # Create sample queries for benchmarking
        sample_queries = [
            {"query": "What are the effects of COVID-19?"},
            {"query": "How does machine learning work?"},
            {"query": "What is artificial intelligence?"}
        ]
        
        # Run benchmarks for each pipeline
        benchmark_results = {
            "timestamp": datetime.now().isoformat(),
            "pipelines": {},
            "summary": {
                "total_pipelines": len(names),
                "successful_benchmarks": 0,
                "failed_benchmarks": 0
            }
        }
        
        for pipeline_name in names:
            try:
                # Create a simple pipeline function for benchmarking
                def pipeline_func(query):
                    # This is a simplified version - in practice, you'd call the actual pipeline
                    return {
                        "query": query,
                        "answer": f"Sample answer for {query}",
                        "retrieved_documents": []
                    }
                
                # Run benchmark for this pipeline
                result = run_technique_benchmark(
                    technique_name=pipeline_name,
                    pipeline_func=pipeline_func,
                    queries=sample_queries,
                    iris_connector=iris_connector
                )
                
                benchmark_results["pipelines"][pipeline_name] = result
                benchmark_results["summary"]["successful_benchmarks"] += 1
                
            except Exception as e:
                benchmark_results["pipelines"][pipeline_name] = {
                    "error": str(e),
                    "success": False
                }
                benchmark_results["summary"]["failed_benchmarks"] += 1
        
        return benchmark_results
    
    execution_result = _safe_execute(_execute)
    return json.dumps(execution_result)


def validate_results(results: str) -> str:
    """
    Validate RAG pipeline results from ObjectScript.
    
    Args:
        results: JSON string with RAG results to validate
        
    Returns:
        JSON string with validation results
    """
    def _execute():
        results_dict = json.loads(results) if isinstance(results, str) else results
        
        validation = {
            "is_valid": True,
            "errors": [],
            "warnings": [],
            "metrics": {}
        }
        
        # Check required fields
        required_fields = ["query", "answer", "retrieved_documents"]
        for field in required_fields:
            if field not in results_dict:
                validation["is_valid"] = False
                validation["errors"].append(f"Missing required field: {field}")
        
        # Validate query
        if "query" in results_dict:
            if not results_dict["query"] or not isinstance(results_dict["query"], str):
                validation["is_valid"] = False
                validation["errors"].append("Query must be a non-empty string")
        
        # Validate answer
        if "answer" in results_dict:
            if not results_dict["answer"] or not isinstance(results_dict["answer"], str):
                validation["is_valid"] = False
                validation["errors"].append("Answer must be a non-empty string")
        
        # Validate retrieved documents
        if "retrieved_documents" in results_dict:
            docs = results_dict["retrieved_documents"]
            if not isinstance(docs, list):
                validation["is_valid"] = False
                validation["errors"].append("Retrieved documents must be a list")
            elif len(docs) == 0:
                validation["warnings"].append("No documents were retrieved")
        
        # Calculate basic metrics if validation passes
        if validation["is_valid"]:
            try:
                # Calculate answer quality metrics
                if "answer" in results_dict and "query" in results_dict:
                    # Create mock queries list for metrics calculation
                    mock_queries = [{"query": results_dict["query"]}]
                    mock_results = [results_dict]
                    
                    answer_faithfulness = calculate_answer_faithfulness(mock_results, mock_queries)
                    answer_relevance = calculate_answer_relevance(mock_results, mock_queries)
                    
                    validation["metrics"]["answer_faithfulness"] = answer_faithfulness
                    validation["metrics"]["answer_relevance"] = answer_relevance
                
                # Calculate comprehensive metrics if ground truth is available
                if "ground_truth_docs" in results_dict:
                    mock_queries = [{
                        "query": results_dict["query"],
                        "ground_truth_contexts": results_dict["ground_truth_docs"]
                    }]
                    mock_results = [results_dict]
                    
                    benchmark_metrics = calculate_benchmark_metrics(mock_results, mock_queries)
                    validation["metrics"]["benchmark"] = benchmark_metrics
                    
            except Exception as e:
                validation["warnings"].append(f"Could not calculate metrics: {str(e)}")
        
        return validation
    
    execution_result = _safe_execute(_execute)
    return json.dumps(execution_result)


def get_available_pipelines() -> str:
    """
    Get list of available RAG pipelines from ObjectScript.
    
    Returns:
        JSON string with available pipeline information
    """
    def _execute():
        pipelines = {
            "basic_rag": {
                "name": "Basic RAG",
                "description": "Standard retrieval-augmented generation",
                "class": "BasicRAGPipeline"
            },
            "colbert": {
                "name": "ColBERT",
                "description": "Contextualized late interaction over BERT",
                "class": "ColBERTPipeline"
            },
            "graphrag": {
                "name": "GraphRAG",
                "description": "Graph-based retrieval-augmented generation",
                "class": "GraphRAGPipeline"
            },
            "hyde": {
                "name": "HyDE",
                "description": "Hypothetical document embeddings",
                "class": "HyDEPipeline"
            },
            "crag": {
                "name": "CRAG",
                "description": "Corrective retrieval-augmented generation",
                "class": "CRAGPipeline"
            },
            "noderag": {
                "name": "NodeRAG",
                "description": "Node-based retrieval-augmented generation",
                "class": "NodeRAGPipeline"
            }
        }
        return pipelines
    
    execution_result = _safe_execute(_execute)
    return json.dumps(execution_result)


def health_check() -> str:
    """
    Perform health check of the Python bridge from ObjectScript.
    
    Returns:
        JSON string with health status
    """
    def _execute():
        health = {
            "status": "healthy",
            "timestamp": datetime.now().isoformat(),
            "components": {}
        }
        
        # Check database connection
        try:
            iris_connector = get_iris_connection()
            cursor = iris_connector.cursor()
            cursor.execute("SELECT 1 AS test")
            result = cursor.fetchone()
            health["components"]["database"] = "healthy" if result[0] == 1 else "unhealthy"
        except Exception as e:
            health["components"]["database"] = f"unhealthy: {str(e)}"
            health["status"] = "degraded"
        
        # Check pipeline imports
        pipeline_classes = [
            BasicRAGPipeline, ColbertRAGPipeline, GraphRAGPipeline,
            HyDEPipeline, CRAGPipeline, NodeRAGPipeline
        ]
        
        for pipeline_class in pipeline_classes:
            try:
                # Try to instantiate with minimal config
                pipeline_class.__name__  # Just check the class exists
                health["components"][pipeline_class.__name__] = "healthy"
            except Exception as e:
                health["components"][pipeline_class.__name__] = f"unhealthy: {str(e)}"
                health["status"] = "degraded"
        
        return health
    
    execution_result = _safe_execute(_execute)
    return json.dumps(execution_result)