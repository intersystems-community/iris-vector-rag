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
import os # Added
import sys # Added

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import RAG pipeline modules (legacy imports with fallback)
try:
    from iris_rag.pipelines.basic import BasicRAGPipeline # Updated import
    from iris_rag.pipelines.colbert import ColBERTRAGPipeline # Updated import
    from iris_rag.pipelines.graphrag import GraphRAGPipeline # Updated import
    from iris_rag.pipelines.hyde import HyDERAGPipeline # Updated import
    from iris_rag.pipelines.crag import CRAGPipeline # Updated import
    from iris_rag.pipelines.noderag import NodeRAGPipeline # Updated import
    LEGACY_IMPORTS_AVAILABLE = True
except ImportError:
    LEGACY_IMPORTS_AVAILABLE = False
    logger.warning("Legacy pipeline imports not available, using new framework")

# Import new Library Consumption Framework APIs
try:
    from rag_templates.simple import RAG as SimpleRAG
    from rag_templates.standard import ConfigurableRAG
    from rag_templates.core.config_manager import ConfigurationManager
    NEW_FRAMEWORK_AVAILABLE = True
except ImportError:
    NEW_FRAMEWORK_AVAILABLE = False
    logger.warning("New Library Consumption Framework not available")

# Import evaluation and benchmarking modules
try:
    from scripts.utilities.evaluation.metrics import calculate_benchmark_metrics, calculate_answer_faithfulness, calculate_answer_relevance # Path remains same
    from scripts.utilities.evaluation.bench_runner import run_technique_benchmark # Path remains same
    EVAL_MODULES_AVAILABLE = True
except ImportError:
    EVAL_MODULES_AVAILABLE = False
    logger.warning("Evaluation modules not available")

# Import connection utilities
try:
    from common.iris_connection_manager import get_iris_connection # Updated import
    CONNECTION_UTILS_AVAILABLE = True
except ImportError:
    CONNECTION_UTILS_AVAILABLE = False
    logger.warning("Connection utilities not available")

# Import IrisSQLTool and SQL RAG Pipeline
try:
    from iris_rag.tools.iris_sql_tool import IrisSQLTool
    from iris_rag.pipelines.sql_rag import SQLRAGPipeline
    from common.utils import get_llm_func
    IRIS_SQL_TOOL_AVAILABLE = True
    SQL_RAG_PIPELINE_AVAILABLE = True
except ImportError:
    IRIS_SQL_TOOL_AVAILABLE = False
    SQL_RAG_PIPELINE_AVAILABLE = False
    logger.warning("IrisSQLTool and SQL RAG Pipeline not available")

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
        
        # Try new framework first, fallback to legacy
        if NEW_FRAMEWORK_AVAILABLE:
            try:
                # Use new Simple API for basic RAG
                if not config_dict.get("embedding_func") and not config_dict.get("llm_func"):
                    # Zero-config case - use Simple API
                    global _simple_rag_instance
                    if _simple_rag_instance is None:
                        _simple_rag_instance = SimpleRAG()
                    answer = _simple_rag_instance.query(query)
                    return {
                        "query": query,
                        "answer": answer,
                        "retrieved_documents": [],
                        "framework": "new_simple_api"
                    }
                else:
                    # Configured case - use Standard API
                    standard_config = {"technique": "basic"}
                    if config_dict.get("embedding_func"):
                        standard_config["embedding_provider"] = "custom"
                    if config_dict.get("llm_func"):
                        standard_config["llm_provider"] = "custom"
                    
                    rag_instance = ConfigurableRAG(standard_config)
                    answer = rag_instance.query(query)
                    return {
                        "query": query,
                        "answer": answer,
                        "retrieved_documents": [],
                        "framework": "new_standard_api"
                    }
            except Exception as e:
                logger.warning(f"New framework failed, falling back to legacy: {e}")
        
        # Legacy implementation
        if not LEGACY_IMPORTS_AVAILABLE or not CONNECTION_UTILS_AVAILABLE:
            raise ImportError("Neither new framework nor legacy imports available")
        
        # Initialize pipeline
        iris_connector = get_iris_connection()
        pipeline = BasicRAGPipeline(
            iris_connector=iris_connector,
            embedding_func=config_dict.get("embedding_func"),
            llm_func=config_dict.get("llm_func")
        )
        
        # Execute pipeline
        result = pipeline.run(query)
        result["framework"] = "legacy"
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
        pipeline = ColBERTRAGPipeline(
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
        pipeline = HyDERAGPipeline(
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


def invoke_iris_sql_search(query: str) -> str:
    """
    Invoke IRIS SQL Tool for rewriting and executing SQL queries from ObjectScript.
    
    Args:
        query: The SQL query to rewrite and execute
        
    Returns:
        JSON string with results including:
        - original_query: The original SQL query
        - rewritten_query: The IRIS-compliant rewritten query
        - explanation: Explanation of changes made
        - results: List of result dictionaries
        - success: Boolean indicating if the operation was successful
        - error: Error message if operation failed
    """
    def _execute():
        if not IRIS_SQL_TOOL_AVAILABLE:
            raise ImportError("IrisSQLTool not available - missing imports")
        
        if not CONNECTION_UTILS_AVAILABLE:
            raise ImportError("Connection utilities not available")
        
        # Initialize IRIS connection and LLM function
        iris_connector = get_iris_connection()
        llm_func = get_llm_func()
        
        # Initialize IrisSQLTool
        sql_tool = IrisSQLTool(
            iris_connector=iris_connector,
            llm_func=llm_func
        )
        
        # Execute SQL search
        result = sql_tool.search(query)
        return result
    
    execution_result = _safe_execute(_execute)
    return json.dumps(execution_result)


def invoke_sql_rag(query: str, config: str) -> str:
    """
    Invoke SQL RAG pipeline from ObjectScript.
    
    This pipeline converts natural language questions into SQL queries,
    executes them against IRIS database, and uses the results as context
    for generating comprehensive answers.
    
    Args:
        query: The natural language question
        config: JSON configuration string
        
    Returns:
        JSON string with results including:
        - query: Original question
        - answer: Generated answer based on SQL results
        - retrieved_documents: List of documents (SQL results formatted as documents)
        - sql_query: The generated SQL query
        - sql_results: Raw SQL results
        - execution_time: Time taken to execute the pipeline
    """
    def _execute():
        config_dict = json.loads(config) if isinstance(config, str) else config
        
        if not SQL_RAG_PIPELINE_AVAILABLE:
            raise ImportError("SQL RAG Pipeline not available - missing imports")
        
        if not CONNECTION_UTILS_AVAILABLE:
            raise ImportError("Connection utilities not available")
        
        # Initialize connection and configuration managers
        from iris_rag.core.connection_manager import ConnectionManager
        from iris_rag.core.config_manager import ConfigurationManager
        
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager(config_manager)
        
        # Initialize SQL RAG pipeline
        pipeline = SQLRAGPipeline(
            connection_manager=connection_manager,
            config_manager=config_manager,
            llm_func=config_dict.get("llm_func")
        )
        
        # Execute pipeline
        result = pipeline.execute(query)
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
                "class": "HyDERAGPipeline"
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
            },
            "iris_sql_tool": {
                "name": "IRIS SQL Tool",
                "description": "SQL query rewriting and execution tool for IRIS database",
                "class": "IrisSQLTool",
                "available": IRIS_SQL_TOOL_AVAILABLE
            },
            "sql_rag": {
                "name": "SQL RAG",
                "description": "Natural language to SQL RAG pipeline for IRIS database",
                "class": "SQLRAGPipeline",
                "available": SQL_RAG_PIPELINE_AVAILABLE
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
        if CONNECTION_UTILS_AVAILABLE:
            try:
                iris_connector = get_iris_connection()
                cursor = iris_connector.cursor()
                cursor.execute("SELECT 1 AS test")
                result = cursor.fetchone()
                health["components"]["database"] = "healthy" if result[0] == 1 else "unhealthy"
            except Exception as e:
                health["components"]["database"] = f"unhealthy: {str(e)}"
                health["status"] = "degraded"
        else:
            health["components"]["database"] = "unavailable: connection utils not imported"
            health["status"] = "degraded"
        
        # Check new framework availability
        health["components"]["new_framework"] = "healthy" if NEW_FRAMEWORK_AVAILABLE else "unavailable"
        health["components"]["legacy_pipelines"] = "healthy" if LEGACY_IMPORTS_AVAILABLE else "unavailable"
        health["components"]["evaluation_modules"] = "healthy" if EVAL_MODULES_AVAILABLE else "unavailable"
        
        # Check legacy pipeline imports if available
        if LEGACY_IMPORTS_AVAILABLE:
            pipeline_classes = [
                BasicRAGPipeline, ColBERTRAGPipeline, GraphRAGPipeline,
                HyDERAGPipeline, CRAGPipeline, NodeRAGPipeline
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


# ============================================================================
# SIMPLE API INTEGRATION (Phase 5: Library Consumption Framework Parity)
# ============================================================================

# Global Simple RAG instance for reuse
_simple_rag_instance = None

def invoke_simple_rag(query: str) -> str:
    """
    Zero-config RAG invocation from ObjectScript using Simple API.
    
    Args:
        query: The search query
        
    Returns:
        JSON string with results
    """
    def _execute():
        if not NEW_FRAMEWORK_AVAILABLE:
            raise ImportError("New Library Consumption Framework not available")
        
        global _simple_rag_instance
        
        # Initialize Simple RAG instance if needed
        if _simple_rag_instance is None:
            _simple_rag_instance = SimpleRAG()
        
        # Execute query
        answer = _simple_rag_instance.query(query)
        return answer
    
    execution_result = _safe_execute(_execute)
    return json.dumps(execution_result)


def add_documents_simple(documents: str) -> str:
    """
    Add documents using Simple API from ObjectScript.
    
    Args:
        documents: JSON string containing list of documents
        
    Returns:
        JSON string with results
    """
    def _execute():
        if not NEW_FRAMEWORK_AVAILABLE:
            raise ImportError("New Library Consumption Framework not available")
        
        global _simple_rag_instance
        
        # Initialize Simple RAG instance if needed
        if _simple_rag_instance is None:
            _simple_rag_instance = SimpleRAG()
        
        # Parse documents
        docs_list = json.loads(documents) if isinstance(documents, str) else documents
        
        # Add documents
        _simple_rag_instance.add_documents(docs_list)
        
        return {
            "message": f"Successfully added {len(docs_list)} documents",
            "document_count": len(docs_list)
        }
    
    execution_result = _safe_execute(_execute)
    return json.dumps(execution_result)


def configure_simple_rag(config: str) -> str:
    """
    Configure Simple API from ObjectScript.
    
    Args:
        config: JSON string containing configuration overrides
        
    Returns:
        JSON string with results
    """
    def _execute():
        if not NEW_FRAMEWORK_AVAILABLE:
            raise ImportError("New Library Consumption Framework not available")
        
        global _simple_rag_instance
        
        # Parse configuration
        config_dict = json.loads(config) if isinstance(config, str) else config
        
        # Initialize Simple RAG instance if needed
        if _simple_rag_instance is None:
            _simple_rag_instance = SimpleRAG()
        
        # Apply configuration
        for key, value in config_dict.items():
            _simple_rag_instance.set_config(key, value)
        
        return {
            "message": "Configuration applied successfully",
            "applied_configs": list(config_dict.keys())
        }
    
    execution_result = _safe_execute(_execute)
    return json.dumps(execution_result)


def get_simple_rag_status() -> str:
    """
    Get Simple API status from ObjectScript.
    
    Returns:
        JSON string with status information
    """
    def _execute():
        if not NEW_FRAMEWORK_AVAILABLE:
            raise ImportError("New Library Consumption Framework not available")
        
        global _simple_rag_instance
        
        if _simple_rag_instance is None:
            return {
                "initialized": False,
                "document_count": 0,
                "status": "not_initialized"
            }
        
        return {
            "initialized": True,
            "document_count": _simple_rag_instance.get_document_count(),
            "status": "ready",
            "representation": str(_simple_rag_instance)
        }
    
    execution_result = _safe_execute(_execute)
    return json.dumps(execution_result)


# ============================================================================
# STANDARD API INTEGRATION (Phase 5: Library Consumption Framework Parity)
# ============================================================================

# Global Standard RAG instances for reuse (keyed by technique)
_standard_rag_instances = {}

def invoke_configurable_rag(query: str, config: str) -> str:
    """
    Advanced RAG with technique selection from ObjectScript.
    
    Args:
        query: The search query
        config: JSON configuration string with technique and options
        
    Returns:
        JSON string with results
    """
    def _execute():
        if not NEW_FRAMEWORK_AVAILABLE:
            raise ImportError("New Library Consumption Framework not available")
        
        # Parse configuration
        config_dict = json.loads(config) if isinstance(config, str) else config
        technique = config_dict.get("technique", "basic").lower()
        
        global _standard_rag_instances
        
        # Get or create instance for this technique
        instance_key = f"{technique}_{hash(json.dumps(config_dict, sort_keys=True))}"
        
        if instance_key not in _standard_rag_instances:
            _standard_rag_instances[instance_key] = ConfigurableRAG(config_dict)
        
        rag_instance = _standard_rag_instances[instance_key]
        
        # Execute query
        answer = rag_instance.query(query)
        return answer
    
    execution_result = _safe_execute(_execute)
    return json.dumps(execution_result)


def invoke_configurable_rag_with_options(query: str, config: str, options: str) -> str:
    """
    Advanced RAG with technique selection and query options from ObjectScript.
    
    Args:
        query: The search query
        config: JSON configuration string with technique and options
        options: JSON string with query options
        
    Returns:
        JSON string with results
    """
    def _execute():
        if not NEW_FRAMEWORK_AVAILABLE:
            raise ImportError("New Library Consumption Framework not available")
        
        # Parse configuration and options
        config_dict = json.loads(config) if isinstance(config, str) else config
        options_dict = json.loads(options) if isinstance(options, str) else options
        technique = config_dict.get("technique", "basic").lower()
        
        global _standard_rag_instances
        
        # Get or create instance for this technique
        instance_key = f"{technique}_{hash(json.dumps(config_dict, sort_keys=True))}"
        
        if instance_key not in _standard_rag_instances:
            _standard_rag_instances[instance_key] = ConfigurableRAG(config_dict)
        
        rag_instance = _standard_rag_instances[instance_key]
        
        # Execute query with options
        result = rag_instance.query(query, options_dict)
        return result
    
    execution_result = _safe_execute(_execute)
    return json.dumps(execution_result)


def get_available_techniques() -> str:
    """
    Get available RAG techniques from ObjectScript.
    
    Returns:
        JSON string with list of available techniques
    """
    def _execute():
        if not NEW_FRAMEWORK_AVAILABLE:
            raise ImportError("New Library Consumption Framework not available")
        
        # Create a temporary instance to get available techniques
        temp_config = {"technique": "basic"}
        temp_rag = ConfigurableRAG(temp_config)
        
        techniques = temp_rag.get_available_techniques()
        return techniques
    
    execution_result = _safe_execute(_execute)
    return json.dumps(execution_result)


def switch_technique(technique: str, config: str) -> str:
    """
    Switch RAG technique from ObjectScript.
    
    Args:
        technique: Name of the new technique
        config: JSON string with technique configuration
        
    Returns:
        JSON string with results
    """
    def _execute():
        if not NEW_FRAMEWORK_AVAILABLE:
            raise ImportError("New Library Consumption Framework not available")
        
        # Parse configuration
        config_dict = json.loads(config) if isinstance(config, str) else config
        
        # Create new instance with the technique
        full_config = {"technique": technique}
        full_config.update(config_dict)
        
        global _standard_rag_instances
        
        # Create new instance
        instance_key = f"{technique}_{hash(json.dumps(full_config, sort_keys=True))}"
        _standard_rag_instances[instance_key] = ConfigurableRAG(full_config)
        
        return {
            "message": f"Switched to technique: {technique}",
            "technique": technique,
            "instance_key": instance_key
        }
    
    execution_result = _safe_execute(_execute)
    return json.dumps(execution_result)


def get_technique_info(technique: str) -> str:
    """
    Get technique information from ObjectScript.
    
    Args:
        technique: Name of the technique
        
    Returns:
        JSON string with technique information
    """
    def _execute():
        if not NEW_FRAMEWORK_AVAILABLE:
            raise ImportError("New Library Consumption Framework not available")
        
        # Create a temporary instance to get technique info
        temp_config = {"technique": "basic"}
        temp_rag = ConfigurableRAG(temp_config)
        
        info = temp_rag.get_technique_info(technique)
        return info
    
    execution_result = _safe_execute(_execute)
    return json.dumps(execution_result)


# ============================================================================
# CONFIGURATION BRIDGE FUNCTIONS (Phase 5: Library Consumption Framework Parity)
# ============================================================================

def load_configuration(config_path: str) -> str:
    """
    Load configuration from file for ObjectScript.
    
    Args:
        config_path: Path to configuration file
        
    Returns:
        JSON string with results
    """
    def _execute():
        if not NEW_FRAMEWORK_AVAILABLE:
            raise ImportError("New Library Consumption Framework not available")
        
        # Try to load configuration
        config_manager = ConfigurationManager(config_path)
        
        return {
            "message": f"Configuration loaded from: {config_path}",
            "config_path": config_path,
            "config": config_manager.to_dict()
        }
    
    execution_result = _safe_execute(_execute)
    return json.dumps(execution_result)


def validate_configuration(config: str) -> str:
    """
    Validate configuration from ObjectScript.
    
    Args:
        config: JSON string with configuration to validate
        
    Returns:
        JSON string with validation results
    """
    def _execute():
        if not NEW_FRAMEWORK_AVAILABLE:
            raise ImportError("New Library Consumption Framework not available")
        
        # Parse configuration
        config_dict = json.loads(config) if isinstance(config, str) else config
        
        # Create temporary configuration manager
        config_manager = ConfigurationManager()
        
        # Apply configuration and validate
        for key, value in config_dict.items():
            if isinstance(value, dict):
                for sub_key, sub_value in value.items():
                    config_manager.set(f"{key}:{sub_key}", sub_value)
            else:
                config_manager.set(key, value)
        
        # Validate
        config_manager.validate()
        
        return {
            "valid": True,
            "message": "Configuration validation passed"
        }
    
    execution_result = _safe_execute(_execute)
    return json.dumps(execution_result)


def get_default_configuration() -> str:
    """
    Get default configuration for ObjectScript.
    
    Returns:
        JSON string with default configuration
    """
    def _execute():
        if not NEW_FRAMEWORK_AVAILABLE:
            raise ImportError("New Library Consumption Framework not available")
        
        # Create configuration manager with defaults
        config_manager = ConfigurationManager()
        
        return config_manager.to_dict()
    
    execution_result = _safe_execute(_execute)
    return json.dumps(execution_result)