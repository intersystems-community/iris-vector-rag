"""
Pipeline Integration Module

Manages the integration between Streamlit and RAG pipeline implementations.
Handles pipeline initialization, query execution, and result processing.
"""

import sys
import time
import logging
import traceback
from pathlib import Path
from typing import Dict, Any, List, Optional, Tuple
from dataclasses import dataclass

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

# Import RAG pipeline classes
try:
    from iris_rag.pipelines.basic import BasicRAGPipeline
    from iris_rag.pipelines.basic_rerank import BasicRAGRerankingPipeline
    from iris_rag.pipelines.crag import CRAGPipeline
    from iris_rag.pipelines.graphrag import GraphRAGPipeline
    from iris_rag.core.connection import ConnectionManager
    from iris_rag.config.manager import ConfigurationManager
    from iris_rag.core.models import Document
except ImportError as e:
    logging.warning(f"Failed to import RAG pipelines: {e}")
    # Create mock classes for development/testing
    BasicRAGPipeline = None
    BasicRAGRerankingPipeline = None
    CRAGPipeline = None
    GraphRAGPipeline = None
    ConnectionManager = None
    ConfigurationManager = None
    Document = None

logger = logging.getLogger(__name__)

@dataclass
class QueryResult:
    """Standardized query result structure."""
    success: bool
    pipeline: str
    query: str
    answer: str
    retrieved_documents: List[Dict[str, Any]]
    contexts: List[str]
    execution_time: float
    metadata: Dict[str, Any]
    error: Optional[str] = None

class PipelineManager:
    """Manages RAG pipeline instances and execution."""
    
    def __init__(self):
        self.pipelines: Dict[str, Any] = {}
        self.connection_manager = None
        self.config_manager = None
        self._initialize_managers()
    
    def _initialize_managers(self):
        """Initialize connection and configuration managers."""
        try:
            if ConnectionManager and ConfigurationManager:
                self.connection_manager = ConnectionManager()
                self.config_manager = ConfigurationManager()
                logger.info("Successfully initialized connection and configuration managers")
            else:
                logger.warning("Pipeline classes not available - running in mock mode")
        except Exception as e:
            logger.error(f"Failed to initialize managers: {e}")
            self.connection_manager = None
            self.config_manager = None
    
    def initialize_pipeline(self, pipeline_name: str, config: Dict[str, Any] = None) -> bool:
        """Initialize a specific pipeline."""
        try:
            if not self.connection_manager or not self.config_manager:
                # Create mock pipeline for testing
                self.pipelines[pipeline_name] = MockPipeline(pipeline_name)
                return True
            
            pipeline_classes = {
                "BasicRAG": BasicRAGPipeline,
                "BasicRerank": BasicRAGRerankingPipeline,
                "CRAG": CRAGPipeline,
                "GraphRAG": GraphRAGPipeline
            }
            
            pipeline_class = pipeline_classes.get(pipeline_name)
            if not pipeline_class:
                logger.error(f"Unknown pipeline: {pipeline_name}")
                return False
            
            # Initialize pipeline with managers
            pipeline = pipeline_class(
                connection_manager=self.connection_manager,
                config_manager=self.config_manager
            )
            
            self.pipelines[pipeline_name] = pipeline
            logger.info(f"Successfully initialized {pipeline_name} pipeline")
            return True
            
        except Exception as e:
            logger.error(f"Failed to initialize {pipeline_name} pipeline: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def execute_query(self, pipeline_name: str, query: str, config: Dict[str, Any] = None) -> QueryResult:
        """Execute a query using the specified pipeline."""
        start_time = time.time()
        
        try:
            # Initialize pipeline if not already done
            if pipeline_name not in self.pipelines:
                success = self.initialize_pipeline(pipeline_name, config)
                if not success:
                    return QueryResult(
                        success=False,
                        pipeline=pipeline_name,
                        query=query,
                        answer="",
                        retrieved_documents=[],
                        contexts=[],
                        execution_time=time.time() - start_time,
                        metadata={},
                        error=f"Failed to initialize {pipeline_name} pipeline"
                    )
            
            pipeline = self.pipelines[pipeline_name]
            
            # Prepare query parameters
            query_params = {
                "top_k": config.get("top_k", 5) if config else 5,
                "generate_answer": True
            }
            
            # Add pipeline-specific parameters
            if config:
                if pipeline_name == "BasicRerank" and "rerank_top_k" in config:
                    query_params["rerank_top_k"] = config["rerank_top_k"]
                elif pipeline_name == "CRAG" and "confidence_threshold" in config:
                    query_params["confidence_threshold"] = config["confidence_threshold"]
                elif pipeline_name == "GraphRAG" and "max_hops" in config:
                    query_params["max_hops"] = config["max_hops"]
            
            # Execute query
            result = pipeline.query(query, **query_params)
            
            execution_time = time.time() - start_time
            
            # Standardize result format
            standardized_result = self._standardize_result(
                result, pipeline_name, query, execution_time
            )
            
            return standardized_result
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"Query execution failed for {pipeline_name}: {e}")
            logger.error(traceback.format_exc())
            
            return QueryResult(
                success=False,
                pipeline=pipeline_name,
                query=query,
                answer="",
                retrieved_documents=[],
                contexts=[],
                execution_time=execution_time,
                metadata={},
                error=str(e)
            )
    
    def _standardize_result(self, result: Dict[str, Any], pipeline_name: str, 
                           query: str, execution_time: float) -> QueryResult:
        """Standardize pipeline result into QueryResult format."""
        
        # Extract retrieved documents
        retrieved_docs = []
        if "retrieved_documents" in result:
            for doc in result["retrieved_documents"]:
                if hasattr(doc, '__dict__'):
                    # Document object
                    doc_dict = {
                        "content": getattr(doc, 'content', ''),
                        "title": getattr(doc, 'title', ''),
                        "metadata": getattr(doc, 'metadata', {}),
                        "id": getattr(doc, 'id', '')
                    }
                else:
                    # Already a dictionary
                    doc_dict = doc
                retrieved_docs.append(doc_dict)
        
        # Extract contexts
        contexts = result.get("contexts", [])
        if not contexts and retrieved_docs:
            contexts = [doc.get("content", "") for doc in retrieved_docs]
        
        return QueryResult(
            success=True,
            pipeline=pipeline_name,
            query=query,
            answer=result.get("answer", ""),
            retrieved_documents=retrieved_docs,
            contexts=contexts,
            execution_time=execution_time,
            metadata=result.get("metadata", {})
        )
    
    def get_pipeline_status(self, pipeline_name: str) -> Dict[str, Any]:
        """Get status information for a pipeline."""
        if pipeline_name not in self.pipelines:
            return {
                "initialized": False,
                "available": pipeline_name in ["BasicRAG", "BasicRerank", "CRAG", "GraphRAG"],
                "error": "Pipeline not initialized"
            }
        
        return {
            "initialized": True,
            "available": True,
            "type": type(self.pipelines[pipeline_name]).__name__,
            "ready": True
        }
    
    def get_all_pipeline_status(self) -> Dict[str, Dict[str, Any]]:
        """Get status for all available pipelines."""
        available_pipelines = ["BasicRAG", "BasicRerank", "CRAG", "GraphRAG"]
        
        status = {}
        for pipeline_name in available_pipelines:
            status[pipeline_name] = self.get_pipeline_status(pipeline_name)
        
        return status
    
    def clear_pipeline_cache(self, pipeline_name: str = None):
        """Clear cache for specific pipeline or all pipelines."""
        if pipeline_name:
            if pipeline_name in self.pipelines:
                # Try to clear pipeline-specific cache if available
                pipeline = self.pipelines[pipeline_name]
                if hasattr(pipeline, 'clear'):
                    pipeline.clear()
        else:
            # Clear all pipelines
            for pipeline in self.pipelines.values():
                if hasattr(pipeline, 'clear'):
                    pipeline.clear()

class MockPipeline:
    """Mock pipeline for testing when actual pipelines are not available."""
    
    def __init__(self, pipeline_name: str):
        self.pipeline_name = pipeline_name
    
    def query(self, query: str, **kwargs) -> Dict[str, Any]:
        """Mock query method that returns fake results."""
        time.sleep(0.5)  # Simulate processing time
        
        mock_docs = [
            {
                "content": f"This is a mock document result for '{query}' using {self.pipeline_name}. "
                          f"In a real implementation, this would contain relevant biomedical content.",
                "title": f"Mock Document 1 - {self.pipeline_name}",
                "metadata": {"source": "mock", "score": 0.95},
                "id": "mock_doc_1"
            },
            {
                "content": f"Another mock document discussing the topic of '{query}'. "
                          f"The {self.pipeline_name} pipeline would retrieve and process real content.",
                "title": f"Mock Document 2 - {self.pipeline_name}",
                "metadata": {"source": "mock", "score": 0.87},
                "id": "mock_doc_2"
            }
        ]
        
        mock_answer = (
            f"Mock answer generated by {self.pipeline_name} pipeline for the query: '{query}'. "
            f"This demonstrates the expected response format. In a real implementation, "
            f"this would be a comprehensive answer based on retrieved biomedical literature."
        )
        
        return {
            "query": query,
            "answer": mock_answer,
            "retrieved_documents": mock_docs,
            "contexts": [doc["content"] for doc in mock_docs],
            "metadata": {
                "pipeline": self.pipeline_name,
                "top_k": kwargs.get("top_k", 5),
                "mock": True
            }
        }

# Global pipeline manager instance
_pipeline_manager = None

def get_pipeline_manager() -> PipelineManager:
    """Get the global pipeline manager instance."""
    global _pipeline_manager
    if _pipeline_manager is None:
        _pipeline_manager = PipelineManager()
    return _pipeline_manager

def execute_pipeline_query(pipeline_name: str, query: str, config: Dict[str, Any] = None) -> QueryResult:
    """Execute a query using the specified pipeline."""
    manager = get_pipeline_manager()
    return manager.execute_query(pipeline_name, query, config)

def get_pipeline_status(pipeline_name: str = None) -> Dict[str, Any]:
    """Get pipeline status information."""
    manager = get_pipeline_manager()
    if pipeline_name:
        return manager.get_pipeline_status(pipeline_name)
    else:
        return manager.get_all_pipeline_status()

def initialize_all_pipelines() -> Dict[str, bool]:
    """Initialize all available pipelines."""
    manager = get_pipeline_manager()
    available_pipelines = ["BasicRAG", "BasicRerank", "CRAG", "GraphRAG"]
    
    results = {}
    for pipeline_name in available_pipelines:
        results[pipeline_name] = manager.initialize_pipeline(pipeline_name)
    
    return results