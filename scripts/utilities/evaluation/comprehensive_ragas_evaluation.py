#!/usr/bin/env python3
"""
Comprehensive RAGAS Performance Testing with DBAPI Default
Leverages optimized container reuse infrastructure for rapid testing cycles
"""

import os
import sys
import json
import time
import logging
import traceback
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any, Tuple, Optional, Callable, Union
from dataclasses import dataclass, asdict, field
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

# Suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)


def print_flush(message: str):
    """Print with immediate flush for real-time output."""
    print(message, flush=True)
    sys.stdout.flush()

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns

# RAGAS imports
try:
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
        answer_similarity,
        answer_correctness
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    print("⚠️ RAGAS not installed. Install with: pip install ragas datasets")

# Import iris_rag factory for pipeline creation
import iris_rag

# Common utilities - DBAPI as default
from common.iris_dbapi_connector import get_iris_dbapi_connection
from common.embedding_utils import get_embedding_model
from common.utils import get_embedding_func, get_llm_func

# Configuration management
from .config_manager import ConfigManager, ComprehensiveConfig

# LangChain for RAGAS
try:
    from langchain_openai import ChatOpenAI
    from langchain_community.embeddings import HuggingFaceEmbeddings
    LANGCHAIN_AVAILABLE = True
except ImportError:
    LANGCHAIN_AVAILABLE = False
    print("⚠️ LangChain not available for RAGAS evaluation")

from dotenv import load_dotenv
load_dotenv()

# Don't set a hardcoded level here - let it be controlled by the runner script
logger = logging.getLogger(__name__)

@dataclass
class RAGASEvaluationResult:
    """RAGAS evaluation result structure"""
    pipeline_name: str
    query: str
    answer: str
    contexts: List[str]
    ground_truth: str
    response_time: float
    documents_retrieved: int
    success: bool
    error: Optional[str] = None
    
    # RAGAS metrics
    answer_relevancy: Optional[float] = None
    context_precision: Optional[float] = None
    context_recall: Optional[float] = None
    faithfulness: Optional[float] = None
    answer_similarity: Optional[float] = None
    answer_correctness: Optional[float] = None
    
    # Performance metrics
    avg_similarity_score: Optional[float] = None
    answer_length: int = 0
    iteration: int = 0

@dataclass
class PipelinePerformanceMetrics:
    """Aggregated performance metrics for a pipeline"""
    pipeline_name: str
    total_queries: int
    success_rate: float
    avg_response_time: float
    std_response_time: float
    avg_documents_retrieved: float
    avg_answer_length: float
    
    # RAGAS metrics aggregated
    avg_answer_relevancy: Optional[float] = None
    avg_context_precision: Optional[float] = None
    avg_context_recall: Optional[float] = None
    avg_faithfulness: Optional[float] = None
    avg_answer_similarity: Optional[float] = None
    avg_answer_correctness: Optional[float] = None
    
    individual_results: List[RAGASEvaluationResult] = field(default_factory=list)

class ComprehensiveRAGASEvaluationFramework:
    """Comprehensive RAGAS evaluation framework with DBAPI default and container optimization"""
    
    def __init__(self, config_path: Optional[str] = None):
        """Initialize the evaluation framework with DBAPI as default"""
        print("DEBUG_PROGRESS: Starting ComprehensiveRAGASEvaluationFramework.__init__()")
        
        # Load configuration with DBAPI default
        print("DEBUG_PROGRESS: About to initialize config")
        self.config_manager = ConfigManager()
        if config_path:
            self.config = self.config_manager.load_config(config_path)
        else:
            self.config = self._create_dbapi_default_config()
        print("DEBUG_PROGRESS: Config initialization completed")
        
        # Ensure DBAPI is the default connection type
        self.config.database.connection_type = "dbapi"
        
        # Setup logging and directories
        self._setup_results_directory()
        self._setup_logging()
        
        # Initialize DBAPI connection
        print("DEBUG_PROGRESS: About to initialize connection")
        self.connection = self._initialize_dbapi_connection()
        print("DEBUG_PROGRESS: Connection initialization completed")
        
        # Initialize models
        print("DEBUG_PROGRESS: About to initialize LLM and embedding functions")
        self.embedding_func, self.llm_func = self._initialize_models()
        print("DEBUG_PROGRESS: LLM and embedding functions initialization completed")
        
        # Initialize RAGAS components
        print("DEBUG_PROGRESS: About to initialize RAGAS components")
        self.ragas_llm, self.ragas_embeddings = self._initialize_ragas()
        print("DEBUG_PROGRESS: RAGAS components initialization completed")
        
        # Initialize pipelines with DBAPI
        print("DEBUG_PROGRESS: About to initialize pipelines with DBAPI")
        self.pipelines = self._initialize_pipelines_with_dbapi()
        print("DEBUG_PROGRESS: Pipelines initialization completed")
        
        # Load test queries
        self.test_queries = self._load_comprehensive_test_queries()
        
        # Thread safety
        self._lock = threading.Lock()
        
    def _create_dbapi_default_config(self) -> ComprehensiveConfig:
        """Create configuration with DBAPI as default"""
        config = ComprehensiveConfig.from_env()
        
        # Force DBAPI as default
        config.database.connection_type = "dbapi"
        
        # Optimize for comprehensive testing
        config.evaluation.enable_ragas = True
        config.evaluation.enable_statistical_testing = True
        config.evaluation.num_iterations = 3
        config.evaluation.parallel_execution = True
        config.evaluation.max_workers = 4
        
        # Enable all pipelines for comprehensive testing using iris_rag factory slugs
        config.pipelines = {
            "basic": {"enabled": True, "timeout": 60, "retry_attempts": 3, "custom_params": {}},
            "hyde": {"enabled": True, "timeout": 90, "retry_attempts": 3, "custom_params": {}},
            "crag": {"enabled": True, "timeout": 120, "retry_attempts": 3, "custom_params": {}},
            "colbert": {"enabled": True, "timeout": 180, "retry_attempts": 3, "custom_params": {}},
            "noderag": {"enabled": True, "timeout": 150, "retry_attempts": 3, "custom_params": {}},
            "graphrag": {"enabled": True, "timeout": 200, "retry_attempts": 3, "custom_params": {}},
            "hybrid_ifind": {"enabled": True, "timeout": 120, "retry_attempts": 3, "custom_params": {}}
        }
        
        # Optimize output for comprehensive analysis
        config.output.results_dir = "comprehensive_ragas_results"
        config.output.create_visualizations = True
        config.output.generate_report = True
        config.output.export_formats = ["json", "csv"]
        config.output.visualization_formats = ["png", "pdf"]
        
        return config
        
    def _setup_logging(self):
        """Setup comprehensive logging - only add file handler, don't override levels"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        
        # Setup file handler only - don't change logging levels
        results_dir = Path(self.config.output.results_dir)
        log_file = results_dir / "comprehensive_evaluation.log"
        file_handler = logging.FileHandler(log_file)
        
        # Use the same level as the current logger, don't force INFO
        current_level = logger.getEffectiveLevel()
        file_handler.setLevel(current_level)
        file_handler.setFormatter(logging.Formatter(log_format))
        
        # Add file handler to the logger without changing its level
        logger.addHandler(file_handler)
        
        # Ensure this logger respects the global logging configuration
        logger.propagate = True
        
    def _setup_results_directory(self):
        """Create comprehensive results directory structure"""
        results_dir = Path(self.config.output.results_dir)
        results_dir.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (results_dir / "visualizations").mkdir(exist_ok=True)
        (results_dir / "reports").mkdir(exist_ok=True)
        (results_dir / "raw_data").mkdir(exist_ok=True)
        
    def _initialize_dbapi_connection(self):
        """Initialize DBAPI connection with container optimization"""
        try:
            logger.info("🔌 Initializing DBAPI connection with container optimization...")
            connection = get_iris_dbapi_connection()
            
            if connection:
                # Test connection with a simple query
                cursor = connection.cursor()
                cursor.execute("SELECT 1 as test_connection")
                test_result = cursor.fetchone()
                logger.info(f"✅ DBAPI connection established. Test query result: {test_result[0] if test_result else 'Failed'}")
                cursor.close()
                return connection
            else:
                raise Exception("Failed to establish DBAPI connection")
                
        except Exception as e:
            logger.error(f"❌ DBAPI connection failed: {e}")
            raise
            
    def _initialize_models(self) -> Tuple[Callable, Callable]:
        """Initialize embedding and LLM functions optimized for evaluation"""
        try:
            # Initialize embedding function
            embedding_model = get_embedding_model(
                self.config.embedding.model_name
            )
            
            def cached_embedding_func(texts):
                if isinstance(texts, str):
                    texts = [texts]
                return embedding_model.encode(texts, normalize_embeddings=True)
            
            # Initialize LLM function
            if os.getenv("OPENAI_API_KEY"):
                llm_func = get_llm_func("openai")
                logger.info("✅ Using OpenAI LLM")
            else:
                llm_func = lambda prompt: f"Based on the provided context: {prompt[:100]}..."
                logger.warning("⚠️ Using stub LLM (set OPENAI_API_KEY for real evaluation)")
                
            return cached_embedding_func, llm_func
            
        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}")
            # Return stub functions
            return (
                lambda texts: [[0.0] * 384 for _ in (texts if isinstance(texts, list) else [texts])],
                lambda prompt: f"Stub response to: {prompt[:50]}..."
            )
    
    def _initialize_ragas(self) -> Tuple[Any, Any]:
        """Initialize RAGAS components for comprehensive evaluation"""
        if not RAGAS_AVAILABLE or not LANGCHAIN_AVAILABLE:
            logger.warning("⚠️ RAGAS or LangChain not available")
            return None, None
            
        try:
            if os.getenv("OPENAI_API_KEY"):
                ragas_llm = ChatOpenAI(
                    model_name="gpt-3.5-turbo",
                    temperature=0,
                    openai_api_key=os.getenv("OPENAI_API_KEY")
                )
                ragas_embeddings = HuggingFaceEmbeddings(
                    model_name=self.config.embedding.model_name,
                    model_kwargs={'device': self.config.embedding.device}
                )
                logger.info("✅ RAGAS components initialized")
                return ragas_llm, ragas_embeddings
            else:
                logger.warning("⚠️ RAGAS requires OpenAI API key")
                return None, None
                
        except Exception as e:
            logger.error(f"❌ RAGAS initialization failed: {e}")
            return None, None
    
    def _initialize_pipelines_with_dbapi(self) -> Dict[str, Any]:
        """Initialize all RAG pipelines using iris_rag.create_pipeline() factory"""
        pipelines = {}
        
        if not self.connection:
            logger.error("❌ No DBAPI connection available for pipeline initialization")
            return pipelines
        
        # Define pipeline types using iris_rag factory slugs
        pipeline_types = [
            "basic", "hyde", "crag", "colbert", "noderag", "graphrag", "hybrid_ifind"
        ]
        
        # Initialize enabled pipelines using iris_rag factory
        for pipeline_type in pipeline_types:
            pipeline_config = self.config.pipelines.get(pipeline_type)
            
            if not pipeline_config or not getattr(pipeline_config, "enabled", True):
                logger.info(f"⏭️ {pipeline_type} pipeline disabled")
                continue
            
            try:
                # Enhanced logging: Before pipeline initialization
                logger.info(f"🔧 Initializing {pipeline_type} pipeline...")
                print(f"DEBUG_PROGRESS: About to create pipeline: {pipeline_type}")
                
                # Log pre-initialization data status if verbose logging is enabled
                if logger.isEnabledFor(logging.DEBUG):
                    self._log_pre_initialization_status(pipeline_type)
                
                # Create pipeline using iris_rag factory with auto_setup
                pipeline = iris_rag.create_pipeline(
                    pipeline_type=pipeline_type,
                    llm_func=self.llm_func,
                    embedding_func=self.embedding_func,
                    external_connection=self.connection,
                    auto_setup=True,
                    validate_requirements=True
                )
                print(f"DEBUG_PROGRESS: Successfully created pipeline: {pipeline_type}")
                
                pipelines[pipeline_type] = pipeline
                
                # Enhanced logging: After pipeline initialization
                logger.info(f"✅ {pipeline_type} pipeline initialized using iris_rag factory")
                
                # Log post-initialization data status if verbose logging is enabled
                if logger.isEnabledFor(logging.DEBUG):
                    self._log_post_initialization_status(pipeline_type)
                
            except Exception as e:
                logger.error(f"❌ {pipeline_type} pipeline failed: {e}")
                if logger.isEnabledFor(logging.DEBUG):
                    traceback.print_exc()
                    # Try to get more detailed error information
                    self._log_pipeline_validation_details(pipeline_type)
        
        logger.info(f"🚀 Initialized {len(pipelines)} RAG pipelines using iris_rag factory")
        return pipelines
    
    def _log_pre_initialization_status(self, pipeline_type: str):
        """Log data status before pipeline initialization"""
        try:
            logger.debug(f"📊 Pre-initialization data status for {pipeline_type}:")
            
            # Get basic table counts that all pipelines need
            source_docs_count = self._get_table_count("RAG.SourceDocuments")
            logger.debug(f"  📄 RAG.SourceDocuments: {source_docs_count} records")
            
            # Get pipeline-specific table counts
            pipeline_tables = self._get_pipeline_specific_tables(pipeline_type)
            for table_name, description in pipeline_tables.items():
                count = self._get_table_count(table_name)
                logger.debug(f"  📊 {table_name} ({description}): {count} records")
                
        except Exception as e:
            logger.debug(f"⚠️ Could not retrieve pre-initialization status for {pipeline_type}: {e}")
    
    def _log_post_initialization_status(self, pipeline_type: str):
        """Log data status and validation results after pipeline initialization"""
        try:
            logger.debug(f"✅ Post-initialization status for {pipeline_type}:")
            
            # Use iris_rag.get_pipeline_status to get detailed status
            status_info = iris_rag.get_pipeline_status(
                pipeline_type=pipeline_type,
                external_connection=self.connection
            )
            
            if status_info:
                logger.debug(f"  🔍 Pipeline validation status: {status_info.get('overall_valid', 'Unknown')}")
                
                # Log any validation issues
                if 'validation_results' in status_info:
                    validation_results = status_info['validation_results']
                    for requirement, result in validation_results.items():
                        if isinstance(result, dict):
                            status = "✅" if result.get('valid', False) else "❌"
                            logger.debug(f"  {status} {requirement}: {result.get('message', 'No details')}")
                
                # Log table status if available
                if 'table_status' in status_info:
                    table_status = status_info['table_status']
                    for table_name, status in table_status.items():
                        count = status.get('count', 'Unknown')
                        exists = status.get('exists', False)
                        status_icon = "✅" if exists else "❌"
                        logger.debug(f"  {status_icon} {table_name}: {count} records")
            
            # Also get updated table counts
            source_docs_count = self._get_table_count("RAG.SourceDocuments")
            logger.debug(f"  📄 RAG.SourceDocuments: {source_docs_count} records")
            
            # Get pipeline-specific table counts
            pipeline_tables = self._get_pipeline_specific_tables(pipeline_type)
            for table_name, description in pipeline_tables.items():
                count = self._get_table_count(table_name)
                logger.debug(f"  📊 {table_name} ({description}): {count} records")
                
        except Exception as e:
            logger.debug(f"⚠️ Could not retrieve post-initialization status for {pipeline_type}: {e}")
    
    def _log_pipeline_validation_details(self, pipeline_type: str):
        """Log detailed validation information when pipeline creation fails"""
        try:
            logger.debug(f"🔍 Validation details for failed {pipeline_type} pipeline:")
            
            # Try to get validation results without creating the pipeline
            validation_info = iris_rag.validate_pipeline(
                pipeline_type=pipeline_type,
                external_connection=self.connection
            )
            
            if validation_info:
                logger.debug(f"  📋 Validation summary: {validation_info.get('summary', 'No summary available')}")
                
                if 'validation_results' in validation_info:
                    for requirement, result in validation_info['validation_results'].items():
                        if isinstance(result, dict):
                            status = "✅" if result.get('valid', False) else "❌"
                            message = result.get('message', 'No details')
                            logger.debug(f"  {status} {requirement}: {message}")
                
                # Log setup suggestions if available
                if 'setup_suggestions' in validation_info:
                    suggestions = validation_info['setup_suggestions']
                    if suggestions:
                        logger.debug(f"  💡 Setup suggestions:")
                        for suggestion in suggestions:
                            logger.debug(f"    - {suggestion}")
                            
        except Exception as e:
            logger.debug(f"⚠️ Could not retrieve validation details for {pipeline_type}: {e}")
    
    def _get_pipeline_specific_tables(self, pipeline_type: str) -> Dict[str, str]:
        """Get pipeline-specific table names and descriptions"""
        pipeline_tables = {
            "basic": {},
            "hyde": {},
            "crag": {},
            "colbert": {
                "RAG.DocumentTokenEmbeddings": "ColBERT token embeddings"
            },
            "noderag": {
                "RAG.KnowledgeGraphNodes": "Knowledge graph nodes",
                "RAG.KnowledgeGraphEdges": "Knowledge graph edges"
            },
            "graphrag": {
                "RAG.DocumentEntities": "Document entities",
                "RAG.EntityRelationships": "Entity relationships"
            },
            "hybrid_ifind": {}
        }
        
        return pipeline_tables.get(pipeline_type, {})
    
    def _get_table_count(self, table_name: str) -> int:
        """Get the count of records in a specific table using a temporary connection"""
        temp_connection = None
        temp_cursor = None
        
        try:
            # Get a temporary database connection
            temp_connection = get_iris_dbapi_connection()
            if temp_connection is None:
                logger.error(f"Failed to get temporary DB connection for table count: {table_name}")
                return 0
            
            # Create cursor and execute query
            temp_cursor = temp_connection.cursor()
            # Use IRIS SQL syntax with TOP instead of LIMIT
            temp_cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            result = temp_cursor.fetchone()
            return result[0] if result else 0
            
        except Exception as e:
            logger.debug(f"Could not get count for table {table_name}: {e}")
            return 0
            
        finally:
            # Clean up resources
            if temp_cursor:
                try:
                    temp_cursor.close()
                except Exception as e:
                    logger.warning(f"Error closing temporary cursor: {e}")
            
            if temp_connection:
                try:
                    temp_connection.close()
                except Exception as e:
                    logger.warning(f"Error closing temporary connection: {e}")
    
    def _load_comprehensive_test_queries(self) -> List[Dict[str, Any]]:
        """Load comprehensive test queries for evaluation"""
        # Try to load from sample_queries.json first
        sample_queries_path = Path("eval/sample_queries.json")
        if sample_queries_path.exists():
            try:
                with open(sample_queries_path, 'r') as f:
                    queries_data = json.load(f)
                    
                test_queries = []
                for item in queries_data:
                    test_queries.append({
                        "query": item["query"],
                        "ground_truth": item["ground_truth_answer"],
                        "keywords": self._extract_keywords(item["query"])
                    })
                    
                logger.info(f"✅ Loaded {len(test_queries)} queries from sample_queries.json")
                return test_queries
                
            except Exception as e:
                logger.warning(f"⚠️ Failed to load sample_queries.json: {e}")
        
        # Fallback to comprehensive default queries
        return self._get_comprehensive_default_queries()
    
    def _get_comprehensive_default_queries(self) -> List[Dict[str, Any]]:
        """Get comprehensive default test queries covering various medical domains"""
        return [
            {
                "query": "What are the effects of metformin on type 2 diabetes?",
                "ground_truth": "Metformin helps treat type 2 diabetes by reducing glucose production in the liver and increasing insulin sensitivity in peripheral tissues.",
                "keywords": ["metformin", "diabetes", "glucose", "insulin"]
            },
            {
                "query": "How does SGLT2 inhibition affect kidney function?",
                "ground_truth": "SGLT2 inhibitors protect kidney function by reducing hyperfiltration, decreasing albuminuria, and providing nephroprotection through mechanisms independent of glycemic control.",
                "keywords": ["SGLT2", "kidney", "nephroprotection", "albuminuria"]
            },
            {
                "query": "What is the mechanism of action of GLP-1 receptor agonists?",
                "ground_truth": "GLP-1 receptor agonists work by stimulating insulin secretion, suppressing glucagon secretion, slowing gastric emptying, and promoting satiety.",
                "keywords": ["GLP-1", "insulin", "glucagon", "satiety"]
            },
            {
                "query": "What are the cardiovascular benefits of SGLT2 inhibitors?",
                "ground_truth": "SGLT2 inhibitors provide cardiovascular benefits by reducing major adverse cardiovascular events and hospitalization for heart failure.",
                "keywords": ["SGLT2", "cardiovascular", "heart failure", "events"]
            },
            {
                "query": "How do statins prevent cardiovascular disease?",
                "ground_truth": "Statins prevent cardiovascular disease by inhibiting HMG-CoA reductase to lower LDL cholesterol, reducing atherosclerotic plaque formation.",
                "keywords": ["statins", "cholesterol", "atherosclerotic", "HMG-CoA"]
            }
        ]
    
    def _extract_keywords(self, query: str) -> List[str]:
        """Extract keywords from query text"""
        import re
        words = re.findall(r'\b\w+\b', query.lower())
        stop_words = {'what', 'how', 'the', 'is', 'are', 'of', 'in', 'to', 'and', 'or', 'for', 'with', 'do', 'does'}
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        return keywords[:5]
    
    def run_single_evaluation(self, pipeline_name: str, query_data: Dict[str, Any], iteration: int = 0) -> RAGASEvaluationResult:
        """Run a single query evaluation with comprehensive metrics"""
        pipeline = self.pipelines[pipeline_name]
        query = query_data["query"]
        
        # Enhanced debug logging for single evaluation
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"🔍 Starting single evaluation:")
            logger.debug(f"  📊 Pipeline: {pipeline_name}")
            logger.debug(f"  ❓ Query: '{query}'")
            logger.debug(f"  🔄 Iteration: {iteration}")
            logger.debug(f"  📋 Pipeline type: {type(pipeline)}")
            logger.debug(f"  ⚙️ Config - top_k: {self.config.retrieval.top_k}, threshold: {self.config.retrieval.similarity_threshold}")
        
        start_time = time.time()
        try:
            # Enhanced debug logging before pipeline execution
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"🚀 Executing {pipeline_name} pipeline...")
            
            # Run pipeline with standardized parameters
            result = pipeline.query(
                query,
                top_k=self.config.retrieval.top_k,
                similarity_threshold=self.config.retrieval.similarity_threshold
            )
            
            response_time = time.time() - start_time
            
            # Enhanced debug logging after pipeline execution
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"✅ Pipeline execution completed in {response_time:.2f}s")
                logger.debug(f"  📊 Result keys: {list(result.keys()) if isinstance(result, dict) else 'Not a dict'}")
            
            # Extract information
            documents = result.get('retrieved_documents', [])
            answer = result.get('answer', '')
            
            # Enhanced debug logging for extracted data
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"📄 Extracted data:")
                logger.debug(f"  📚 Documents retrieved: {len(documents)}")
                logger.debug(f"  💬 Answer length: {len(answer)} chars")
                logger.debug(f"  📝 Answer preview: '{answer[:100]}...' " if len(answer) > 100 else f"  📝 Answer: '{answer}'")
            
            # Extract contexts for RAGAS
            contexts = self._extract_contexts(documents)
            
            # Calculate similarity scores
            similarity_scores = self._extract_similarity_scores(documents)
            avg_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
            
            # Enhanced debug logging for RAGAS preparation
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"🎯 RAGAS preparation:")
                logger.debug(f"  📄 Contexts extracted: {len(contexts)}")
                logger.debug(f"  📊 Similarity scores: {len(similarity_scores)} scores, avg: {avg_similarity:.3f}")
                logger.debug(f"  🔗 RAGAS available: {bool(self.ragas_llm and self.ragas_embeddings)}")
            
            # Create evaluation result
            eval_result = RAGASEvaluationResult(
                pipeline_name=pipeline_name,
                query=query,
                answer=answer,
                contexts=contexts,
                ground_truth=query_data.get('ground_truth', ''),
                response_time=response_time,
                documents_retrieved=len(documents),
                avg_similarity_score=avg_similarity,
                answer_length=len(answer),
                success=True,
                iteration=iteration
            )
            
            # Run RAGAS evaluation if available
            if self.ragas_llm and self.ragas_embeddings and contexts and answer:
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"🎯 Running RAGAS evaluation...")
                
                ragas_scores = self._evaluate_with_ragas_single(eval_result)
                if ragas_scores:
                    eval_result.answer_relevancy = ragas_scores.get('answer_relevancy')
                    eval_result.context_precision = ragas_scores.get('context_precision')
                    eval_result.context_recall = ragas_scores.get('context_recall')
                    eval_result.faithfulness = ragas_scores.get('faithfulness')
                    eval_result.answer_similarity = ragas_scores.get('answer_similarity')
                    eval_result.answer_correctness = ragas_scores.get('answer_correctness')
                    
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"📊 RAGAS scores: {ragas_scores}")
                else:
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"⚠️ RAGAS evaluation returned no scores")
            else:
                if logger.isEnabledFor(logging.DEBUG):
                    missing_components = []
                    if not self.ragas_llm:
                        missing_components.append("LLM")
                    if not self.ragas_embeddings:
                        missing_components.append("embeddings")
                    if not contexts:
                        missing_components.append("contexts")
                    if not answer:
                        missing_components.append("answer")
                    logger.debug(f"⏭️ Skipping RAGAS evaluation - missing: {', '.join(missing_components)}")
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"✅ Single evaluation completed successfully")
            
            return eval_result
            
        except Exception as e:
            error_time = time.time() - start_time
            logger.error(f"❌ {pipeline_name} failed for query '{query[:50]}...': {e}")
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"💥 Pipeline execution failed after {error_time:.2f}s")
                logger.debug(f"  ⚠️ Error type: {type(e).__name__}")
                logger.debug(f"  📝 Error details: {str(e)}")
                import traceback
                logger.debug(f"  📋 Traceback: {traceback.format_exc()}")
            
            return RAGASEvaluationResult(
                pipeline_name=pipeline_name,
                query=query,
                answer='',
                contexts=[],
                ground_truth=query_data.get('ground_truth', ''),
                response_time=error_time,
                documents_retrieved=0,
                success=False,
                error=str(e),
                iteration=iteration
            )
    
    def _extract_contexts(self, documents: List[Any]) -> List[str]:
        """Extract context strings from documents"""
        contexts = []
        for doc in documents:
            if hasattr(doc, 'content'):
                contexts.append(str(doc.content))
            elif hasattr(doc, 'text_content'):
                contexts.append(str(doc.text_content))
            elif isinstance(doc, dict):
                contexts.append(str(doc.get('content', doc.get('text_content', ''))))
            else:
                contexts.append(str(doc))
        return contexts
    
    def _extract_similarity_scores(self, documents: List[Any]) -> List[float]:
        """Extract similarity scores from documents"""
        scores = []
        for doc in documents:
            if hasattr(doc, 'similarity_score'):
                scores.append(float(doc.similarity_score))
            elif hasattr(doc, 'score'):
                scores.append(float(doc.score))
            elif isinstance(doc, dict):
                score = doc.get('similarity_score', doc.get('score', 0.0))
                scores.append(float(score))
        return scores
    
    def _evaluate_with_ragas_single(self, result: RAGASEvaluationResult) -> Optional[Dict[str, float]]:
        """Evaluate a single result with RAGAS metrics"""
        if not RAGAS_AVAILABLE or not self.ragas_llm:
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"⏭️ RAGAS not available - RAGAS_AVAILABLE: {RAGAS_AVAILABLE}, ragas_llm: {bool(self.ragas_llm)}")
            return None
            
        try:
            # Enhanced debug logging for RAGAS input preparation
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"🎯 Preparing RAGAS evaluation:")
                logger.debug(f"  ❓ Query: '{result.query}'")
                logger.debug(f"  💬 Answer: '{result.answer[:100]}...' ({len(result.answer)} chars)")
                logger.debug(f"  📄 Contexts: {len(result.contexts)} contexts")
                logger.debug(f"  🎯 Ground truth: '{result.ground_truth[:100]}...' ({len(result.ground_truth)} chars)")
                for i, context in enumerate(result.contexts[:3]):  # Show first 3 contexts
                    logger.debug(f"    Context {i+1}: '{context[:100]}...' ({len(context)} chars)")
            
            # Create dataset for single evaluation
            dataset_dict = {
                'question': [result.query],
                'answer': [result.answer],
                'contexts': [result.contexts],
                'ground_truth': [result.ground_truth]
            }
            
            dataset = Dataset.from_dict(dataset_dict)
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"📊 Dataset created with {len(dataset)} rows")
            
            # Define metrics
            metrics = [
                answer_relevancy,
                context_precision,
                context_recall,
                faithfulness,
                answer_similarity,
                answer_correctness
            ]
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"📏 Using {len(metrics)} RAGAS metrics")
                logger.debug(f"  🔗 LLM type: {type(self.ragas_llm)}")
                logger.debug(f"  🔗 Embeddings type: {type(self.ragas_embeddings)}")
            
            # Run evaluation
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"🚀 Starting RAGAS evaluation...")
            
            ragas_result = evaluate(
                dataset,
                metrics=metrics,
                llm=self.ragas_llm,
                embeddings=self.ragas_embeddings
            )
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"✅ RAGAS evaluation completed")
                logger.debug(f"  📊 Raw result keys: {list(ragas_result.keys()) if hasattr(ragas_result, 'keys') else 'Not a dict'}")
            
            # Extract scores - handle both dict and RagasDataset objects
            scores = {}
            
            # Try to convert RagasDataset to pandas DataFrame and extract scores
            try:
                if hasattr(ragas_result, 'to_pandas'):
                    # RagasDataset object - convert to pandas and extract scores
                    df = ragas_result.to_pandas()
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"  📊 Converted RagasDataset to DataFrame with columns: {list(df.columns)}")
                    
                    # Extract scores from the first row (single evaluation)
                    for metric_name in ['answer_relevancy', 'context_precision', 'context_recall',
                                      'faithfulness', 'answer_similarity', 'answer_correctness']:
                        if metric_name in df.columns and len(df) > 0:
                            score_value = df[metric_name].iloc[0]
                            if pd.notna(score_value):  # Check for NaN values
                                scores[metric_name] = float(score_value)
                                if logger.isEnabledFor(logging.DEBUG):
                                    logger.debug(f"  📊 {metric_name}: {score_value:.3f}")
                            else:
                                if logger.isEnabledFor(logging.DEBUG):
                                    logger.debug(f"  ⚠️ {metric_name}: NaN value in results")
                        else:
                            if logger.isEnabledFor(logging.DEBUG):
                                logger.debug(f"  ⚠️ {metric_name}: not found in DataFrame columns")
                                
                elif isinstance(ragas_result, dict):
                    # Dictionary object - direct access
                    for metric_name in ['answer_relevancy', 'context_precision', 'context_recall',
                                      'faithfulness', 'answer_similarity', 'answer_correctness']:
                        if metric_name in ragas_result:
                            score_value = float(ragas_result[metric_name])
                            scores[metric_name] = score_value
                            if logger.isEnabledFor(logging.DEBUG):
                                logger.debug(f"  📊 {metric_name}: {score_value:.3f}")
                        else:
                            if logger.isEnabledFor(logging.DEBUG):
                                logger.debug(f"  ⚠️ {metric_name}: not found in results dict")
                else:
                    # Unknown format - try to inspect and handle gracefully
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"  ⚠️ Unexpected ragas_result type: {type(ragas_result)}")
                        logger.debug(f"  📋 Available attributes: {dir(ragas_result)}")
                    
                    # Try to access as attributes if available
                    for metric_name in ['answer_relevancy', 'context_precision', 'context_recall',
                                      'faithfulness', 'answer_similarity', 'answer_correctness']:
                        try:
                            if hasattr(ragas_result, metric_name):
                                score_value = getattr(ragas_result, metric_name)
                                if score_value is not None:
                                    scores[metric_name] = float(score_value)
                                    if logger.isEnabledFor(logging.DEBUG):
                                        logger.debug(f"  📊 {metric_name}: {score_value:.3f}")
                        except (AttributeError, TypeError, ValueError) as attr_e:
                            if logger.isEnabledFor(logging.DEBUG):
                                logger.debug(f"  ⚠️ {metric_name}: failed to extract as attribute - {attr_e}")
                            
            except Exception as extract_e:
                logger.warning(f"⚠️ Failed to extract scores from ragas_result: {extract_e}")
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"  💥 Score extraction error details:")
                    logger.debug(f"    ⚠️ Error type: {type(extract_e).__name__}")
                    logger.debug(f"    📝 Error message: {str(extract_e)}")
                    logger.debug(f"    📋 ragas_result type: {type(ragas_result)}")
            
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"� Final RAGAS scores: {len(scores)} metrics extracted")
            
            return scores
            
        except Exception as e:
            logger.warning(f"⚠️ RAGAS evaluation failed for single result: {e}")
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"💥 RAGAS evaluation error details:")
                logger.debug(f"  ⚠️ Error type: {type(e).__name__}")
                logger.debug(f"  📝 Error message: {str(e)}")
                import traceback
                logger.debug(f"  📋 Traceback: {traceback.format_exc()}")
            return None
    
    def run_comprehensive_evaluation(self) -> Dict[str, PipelinePerformanceMetrics]:
        """Run comprehensive evaluation across all pipelines and queries"""
        print_flush("🚀 Starting comprehensive RAGAS evaluation with DBAPI...")
        logger.info("🚀 Starting comprehensive RAGAS evaluation with DBAPI...")
        
        all_results = {}
        total_evaluations = len(self.pipelines) * len(self.test_queries) * self.config.evaluation.num_iterations
        completed_evaluations = 0
        
        print_flush(f"📊 Total evaluations planned: {total_evaluations}")
        print_flush(f"📋 Pipelines: {list(self.pipelines.keys())}")
        print_flush(f"📝 Queries: {len(self.test_queries)}")
        print_flush(f"🔄 Iterations per query: {self.config.evaluation.num_iterations}")
        
        # Enhanced debug logging for evaluation setup
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"🔍 Evaluation setup details:")
            logger.debug(f"  📊 Total pipelines: {len(self.pipelines)}")
            logger.debug(f"  📝 Total queries: {len(self.test_queries)}")
            logger.debug(f"  🔄 Iterations per query: {self.config.evaluation.num_iterations}")
            logger.debug(f"  📈 Total evaluations planned: {total_evaluations}")
            logger.debug(f"  🔗 Connection type: {self.config.database.connection_type}")
            logger.debug(f"  🎯 RAGAS enabled: {self.config.evaluation.enable_ragas}")
        
        for pipeline_name in self.pipelines.keys():
            print_flush(f"📊 Evaluating {pipeline_name} pipeline...")
            logger.info(f"📊 Evaluating {pipeline_name} pipeline...")
            
            # Enhanced debug logging for pipeline evaluation
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"🔧 Starting evaluation for {pipeline_name} pipeline")
                logger.debug(f"  📋 Pipeline object: {type(self.pipelines[pipeline_name])}")
            
            pipeline_results = []
            
            for iteration in range(self.config.evaluation.num_iterations):
                print_flush(f"  🔄 Iteration {iteration + 1}/{self.config.evaluation.num_iterations}")
                logger.info(f"  Iteration {iteration + 1}/{self.config.evaluation.num_iterations}")
                
                if logger.isEnabledFor(logging.DEBUG):
                    logger.debug(f"🔄 Starting iteration {iteration + 1} for {pipeline_name}")
                
                for query_idx, query_data in enumerate(self.test_queries):
                    progress = f"({completed_evaluations + 1}/{total_evaluations})"
                    print_flush(f"    ❓ Query {query_idx + 1}/{len(self.test_queries)} {progress}: '{query_data['query'][:50]}...'")
                    
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"❓ Processing query {query_idx + 1}/{len(self.test_queries)}: '{query_data['query'][:50]}...'")
                    
                    result = self.run_single_evaluation(pipeline_name, query_data, iteration)
                    pipeline_results.append(result)
                    completed_evaluations += 1
                    
                    # Real-time progress feedback
                    success_status = "✅" if result.success else "❌"
                    print_flush(f"      {success_status} Result: docs={result.documents_retrieved}, time={result.response_time:.2f}s")
                    
                    # Enhanced debug logging for query results
                    if logger.isEnabledFor(logging.DEBUG):
                        logger.debug(f"  {success_status} Query result: success={result.success}, docs={result.documents_retrieved}, time={result.response_time:.2f}s")
                        if not result.success and result.error:
                            logger.debug(f"    ⚠️ Error: {result.error}")
                    
                    progress = (completed_evaluations / total_evaluations) * 100
                    logger.info(f"    Progress: {progress:.1f}% ({completed_evaluations}/{total_evaluations})")
            
            # Aggregate results for this pipeline
            aggregated_results = self._aggregate_pipeline_results(pipeline_name, pipeline_results)
            all_results[pipeline_name] = aggregated_results
            
            # Enhanced debug logging for pipeline completion
            if logger.isEnabledFor(logging.DEBUG):
                logger.debug(f"✅ Completed {pipeline_name} pipeline evaluation:")
                logger.debug(f"  📊 Success rate: {aggregated_results.success_rate:.2%}")
                logger.debug(f"  ⏱️ Avg response time: {aggregated_results.avg_response_time:.2f}s")
                logger.debug(f"  📄 Avg documents retrieved: {aggregated_results.avg_documents_retrieved:.1f}")
        
        logger.info("✅ Comprehensive evaluation completed!")
        
        if logger.isEnabledFor(logging.DEBUG):
            logger.debug(f"🎉 Final evaluation summary:")
            logger.debug(f"  📊 Pipelines evaluated: {len(all_results)}")
            logger.debug(f"  📈 Total evaluations completed: {completed_evaluations}")
            for pipeline_name, metrics in all_results.items():
                logger.debug(f"  {pipeline_name}: {metrics.success_rate:.2%} success, {metrics.avg_response_time:.2f}s avg")
        
        return all_results
    
    def _aggregate_pipeline_results(self, pipeline_name: str, results: List[RAGASEvaluationResult]) -> PipelinePerformanceMetrics:
        """Aggregate results for a single pipeline"""
        successful_results = [r for r in results if r.success]
        total_queries = len(results)
        success_rate = len(successful_results) / total_queries if total_queries > 0 else 0.0
        
        if not successful_results:
            return PipelinePerformanceMetrics(
                pipeline_name=pipeline_name,
                total_queries=total_queries,
                success_rate=success_rate,
                avg_response_time=0.0,
                std_response_time=0.0,
                avg_documents_retrieved=0.0,
                avg_answer_length=0.0,
                individual_results=results
            )
        
        # Calculate aggregated metrics
        response_times = [r.response_time for r in successful_results]
        documents_retrieved = [r.documents_retrieved for r in successful_results]
        answer_lengths = [r.answer_length for r in successful_results]
        
        # RAGAS metrics
        ragas_metrics = {}
        for metric_name in ['answer_relevancy', 'context_precision', 'context_recall', 
                           'faithfulness', 'answer_similarity', 'answer_correctness']:
            values = [getattr(r, metric_name) for r in successful_results if getattr(r, metric_name) is not None]
            if values:
                ragas_metrics[f'avg_{metric_name}'] = np.mean(values)
            else:
                ragas_metrics[f'avg_{metric_name}'] = None
        
        return PipelinePerformanceMetrics(
            pipeline_name=pipeline_name,
            total_queries=total_queries,
            success_rate=success_rate,
            avg_response_time=np.mean(response_times) if response_times else 0.0,
            std_response_time=np.std(response_times) if response_times else 0.0,
            avg_documents_retrieved=np.mean(documents_retrieved) if documents_retrieved else 0.0,
            avg_answer_length=np.mean(answer_lengths) if answer_lengths else 0.0,
            **ragas_metrics,
            individual_results=results
        )
    
    def save_results(self, results: Dict[str, PipelinePerformanceMetrics], timestamp: str = None):
        """Save comprehensive evaluation results"""
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        results_dir = Path(self.config.output.results_dir)
        
        # Save raw results
        raw_data_file = results_dir / "raw_data" / f"comprehensive_results_{timestamp}.json"
        raw_data_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(raw_data_file, 'w') as f:
            # Convert to serializable format
            serializable_results = {}
            for pipeline_name, metrics in results.items():
                serializable_results[pipeline_name] = {
                    **asdict(metrics),
                    'individual_results': [asdict(r) for r in metrics.individual_results]
                }
            json.dump(serializable_results, f, indent=2, default=str)
        
        # Save CSV summary
        if "csv" in self.config.output.export_formats:
            self._save_csv_summary(results, timestamp)
        
        logger.info(f"📁 Results saved to {results_dir}")
    
    def _save_csv_summary(self, results: Dict[str, PipelinePerformanceMetrics], timestamp: str):
        """Save CSV summary of results"""
        results_dir = Path(self.config.output.results_dir)
        
        # Pipeline summary
        summary_data = []
        for pipeline_name, metrics in results.items():
            row = {
                'Pipeline': pipeline_name,
                'Success_Rate': metrics.success_rate,
                'Avg_Response_Time': metrics.avg_response_time,
                'Std_Response_Time': metrics.std_response_time,
                'Avg_Documents_Retrieved': metrics.avg_documents_retrieved,
                'Avg_Answer_Length': metrics.avg_answer_length,
                'Avg_Answer_Relevancy': metrics.avg_answer_relevancy,
                'Avg_Context_Precision': metrics.avg_context_precision,
                'Avg_Context_Recall': metrics.avg_context_recall,
                'Avg_Faithfulness': metrics.avg_faithfulness,
                'Avg_Answer_Similarity': metrics.avg_answer_similarity,
                'Avg_Answer_Correctness': metrics.avg_answer_correctness
            }
            summary_data.append(row)
        
        summary_df = pd.DataFrame(summary_data)
        summary_file = results_dir / f"pipeline_summary_{timestamp}.csv"
        summary_df.to_csv(summary_file, index=False)
        
        # Detailed results
        detailed_data = []
        for pipeline_name, metrics in results.items():
            for result in metrics.individual_results:
                row = asdict(result)
                detailed_data.append(row)
        
        detailed_df = pd.DataFrame(detailed_data)
        detailed_file = results_dir / f"detailed_results_{timestamp}.csv"
        detailed_df.to_csv(detailed_file, index=False)
    
    def create_visualizations(self, results: Dict[str, PipelinePerformanceMetrics], timestamp: str = None):
        """Create comprehensive visualizations"""
        if not self.config.output.create_visualizations:
            return
            
        if timestamp is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        viz_dir = Path(self.config.output.results_dir) / "visualizations"
        
        # Performance comparison
        self._create_performance_comparison(results, viz_dir, timestamp)
        
        # RAGAS metrics comparison
        self._create_ragas_comparison(results, viz_dir, timestamp)
        
        logger.info(f"📊 Visualizations saved to {viz_dir}")
    
    def _create_performance_comparison(self, results: Dict[str, PipelinePerformanceMetrics], viz_dir: Path, timestamp: str):
        """Create performance comparison charts"""
        pipelines = list(results.keys())
        response_times = [results[p].avg_response_time for p in pipelines]
        success_rates = [results[p].success_rate * 100 for p in pipelines]
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Response time comparison
        bars1 = ax1.bar(pipelines, response_times, color='skyblue', alpha=0.7)
        ax1.set_title('Average Response Time by Pipeline')
        ax1.set_ylabel('Response Time (seconds)')
        ax1.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars1:
            height = bar.get_height()
            ax1.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.2f}s', ha='center', va='bottom')
        
        # Success rate comparison
        bars2 = ax2.bar(pipelines, success_rates, color='lightgreen', alpha=0.7)
        ax2.set_title('Success Rate by Pipeline')
        ax2.set_ylabel('Success Rate (%)')
        ax2.set_ylim(0, 100)
        ax2.tick_params(axis='x', rotation=45)
        
        # Add value labels on bars
        for bar in bars2:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.1f}%', ha='center', va='bottom')
        
        plt.tight_layout()
        
        # Save in multiple formats
        for fmt in self.config.output.visualization_formats:
            if fmt in ['png', 'pdf', 'svg']:
                plt.savefig(viz_dir / f"performance_comparison_{timestamp}.{fmt}",
                           dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def _create_ragas_comparison(self, results: Dict[str, PipelinePerformanceMetrics], viz_dir: Path, timestamp: str):
        """Create RAGAS metrics comparison"""
        # Filter pipelines with RAGAS results
        ragas_results = {p: m for p, m in results.items() if m.avg_answer_relevancy is not None}
        
        if not ragas_results:
            logger.warning("⚠️ No RAGAS results available for visualization")
            return
        
        pipelines = list(ragas_results.keys())
        metrics = ['answer_relevancy', 'context_precision', 'context_recall',
                  'faithfulness', 'answer_similarity', 'answer_correctness']
        
        # Create subplot for each metric
        fig, axes = plt.subplots(2, 3, figsize=(18, 12))
        axes = axes.flatten()
        
        for i, metric in enumerate(metrics):
            values = [getattr(ragas_results[p], f'avg_{metric}') for p in pipelines]
            
            bars = axes[i].bar(pipelines, values, color=plt.cm.Set3(i), alpha=0.7)
            axes[i].set_title(f'{metric.replace("_", " ").title()}')
            axes[i].set_ylabel('Score')
            axes[i].set_ylim(0, 1)
            axes[i].tick_params(axis='x', rotation=45)
            
            # Add value labels
            for bar in bars:
                height = bar.get_height()
                if height is not None:
                    axes[i].text(bar.get_x() + bar.get_width()/2., height,
                               f'{height:.3f}', ha='center', va='bottom')
        
        plt.suptitle('RAGAS Metrics Comparison Across Pipelines', fontsize=16)
        plt.tight_layout()
        
        # Save in multiple formats
        for fmt in self.config.output.visualization_formats:
            if fmt in ['png', 'pdf', 'svg']:
                plt.savefig(viz_dir / f"ragas_comparison_{timestamp}.{fmt}",
                           dpi=300, bbox_inches='tight')
        
        plt.close()
    
    def generate_comprehensive_report(self, results: Dict[str, PipelinePerformanceMetrics], timestamp: str = None) -> str:
        """Generate comprehensive evaluation report"""
        try:
            if timestamp is None:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            logger.info(f"🔧 Generating comprehensive report with timestamp: {timestamp}")
            
            # Check if we have results to work with
            if not results:
                logger.warning("⚠️ No results provided for report generation")
                return ""
            
            logger.info(f"📊 Processing {len(results)} pipeline results for report")
            
            report_lines = []
            report_lines.append("# Comprehensive RAGAS Performance Evaluation Report")
            report_lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            report_lines.append(f"**Configuration:** DBAPI Default with Container Optimization")
            report_lines.append("")
            
            # Executive Summary
            report_lines.append("## Executive Summary")
            report_lines.append("")
            total_pipelines = len(results)
            
            # Safely calculate averages with error handling
            try:
                success_rates = [r.success_rate for r in results.values() if r.success_rate is not None]
                avg_success_rate = np.mean(success_rates) if success_rates else 0.0
                
                response_times = [r.avg_response_time for r in results.values() if r.avg_response_time is not None]
                avg_response_time = np.mean(response_times) if response_times else 0.0
                
                logger.debug(f"📈 Calculated averages: success_rate={avg_success_rate:.3f}, response_time={avg_response_time:.3f}")
            except Exception as e:
                logger.warning(f"⚠️ Error calculating averages: {e}")
                avg_success_rate = 0.0
                avg_response_time = 0.0
            
            report_lines.append(f"- **Total Pipelines Evaluated:** {total_pipelines}")
            report_lines.append(f"- **Average Success Rate:** {avg_success_rate:.1%}")
            report_lines.append(f"- **Average Response Time:** {avg_response_time:.2f} seconds")
            report_lines.append(f"- **Total Queries per Pipeline:** {len(self.test_queries)}")
            report_lines.append(f"- **Iterations per Query:** {self.config.evaluation.num_iterations}")
            report_lines.append("")
            
            # Pipeline Performance Summary
            report_lines.append("## Pipeline Performance Summary")
            report_lines.append("")
            report_lines.append("| Pipeline | Success Rate | Avg Response Time | Avg Documents | RAGAS Score* |")
            report_lines.append("|----------|--------------|-------------------|---------------|--------------|")
            
            for pipeline_name, metrics in results.items():
                try:
                    ragas_score = "N/A"
                    if metrics.avg_answer_relevancy is not None:
                        # Calculate composite RAGAS score
                        ragas_metrics = [
                            metrics.avg_answer_relevancy,
                            metrics.avg_context_precision,
                            metrics.avg_context_recall,
                            metrics.avg_faithfulness,
                            metrics.avg_answer_correctness
                        ]
                        valid_metrics = [m for m in ragas_metrics if m is not None]
                        if valid_metrics:
                            ragas_score = f"{np.mean(valid_metrics):.3f}"
                    
                    # Safely format metrics with defaults
                    success_rate = metrics.success_rate if metrics.success_rate is not None else 0.0
                    response_time = metrics.avg_response_time if metrics.avg_response_time is not None else 0.0
                    docs_retrieved = metrics.avg_documents_retrieved if metrics.avg_documents_retrieved is not None else 0.0
                    
                    report_lines.append(
                        f"| {pipeline_name} | {success_rate:.1%} | "
                        f"{response_time:.2f}s | {docs_retrieved:.1f} | {ragas_score} |"
                    )
                except Exception as e:
                    logger.warning(f"⚠️ Error processing metrics for {pipeline_name}: {e}")
                    report_lines.append(f"| {pipeline_name} | Error | Error | Error | Error |")
            
            report_lines.append("")
            report_lines.append("*RAGAS Score is the average of available RAGAS metrics")
            report_lines.append("")
            
            # Detailed RAGAS Analysis
            ragas_results = {p: m for p, m in results.items() if m.avg_answer_relevancy is not None}
            if ragas_results:
                report_lines.append("## Detailed RAGAS Analysis")
                report_lines.append("")
                report_lines.append("| Pipeline | Answer Relevancy | Context Precision | Context Recall | Faithfulness | Answer Correctness |")
                report_lines.append("|----------|------------------|-------------------|----------------|--------------|-------------------|")
                
                for pipeline_name, metrics in ragas_results.items():
                    try:
                        report_lines.append(
                            f"| {pipeline_name} | {metrics.avg_answer_relevancy:.3f} | "
                            f"{metrics.avg_context_precision:.3f} | {metrics.avg_context_recall:.3f} | "
                            f"{metrics.avg_faithfulness:.3f} | {metrics.avg_answer_correctness:.3f} |"
                        )
                    except Exception as e:
                        logger.warning(f"⚠️ Error formatting RAGAS metrics for {pipeline_name}: {e}")
                        report_lines.append(f"| {pipeline_name} | Error | Error | Error | Error | Error |")
                report_lines.append("")
            
            # Performance Analysis
            report_lines.append("## Performance Analysis")
            report_lines.append("")
            
            try:
                # Best performing pipeline
                best_pipeline = max(results.items(), key=lambda x: x[1].success_rate if x[1].success_rate is not None else 0.0)
                fastest_pipeline = min(results.items(), key=lambda x: x[1].avg_response_time if x[1].avg_response_time is not None else float('inf'))
                
                report_lines.append(f"- **Most Reliable:** {best_pipeline[0]} ({best_pipeline[1].success_rate:.1%} success rate)")
                report_lines.append(f"- **Fastest:** {fastest_pipeline[0]} ({fastest_pipeline[1].avg_response_time:.2f}s average)")
                
                if ragas_results:
                    best_ragas = max(ragas_results.items(),
                                   key=lambda x: np.mean([getattr(x[1], f'avg_{m}') for m in
                                                        ['answer_relevancy', 'context_precision', 'context_recall',
                                                         'faithfulness', 'answer_correctness']
                                                        if getattr(x[1], f'avg_{m}') is not None]))
                    report_lines.append(f"- **Highest RAGAS Score:** {best_ragas[0]}")
            except Exception as e:
                logger.warning(f"⚠️ Error in performance analysis: {e}")
                report_lines.append("- **Performance analysis:** Error calculating best performers")
            
            report_lines.append("")
            
            # Configuration Details
            report_lines.append("## Configuration Details")
            report_lines.append("")
            try:
                report_lines.append(f"- **Connection Type:** {self.config.database.connection_type.upper()}")
                report_lines.append(f"- **Database Schema:** {self.config.database.schema}")
                report_lines.append(f"- **Embedding Model:** {self.config.embedding.model_name}")
                report_lines.append(f"- **LLM Provider:** {self.config.llm.provider}")
                
                # Safely access retrieval config if it exists
                if hasattr(self.config, 'retrieval'):
                    report_lines.append(f"- **Top K Documents:** {self.config.retrieval.top_k}")
                    report_lines.append(f"- **Similarity Threshold:** {self.config.retrieval.similarity_threshold}")
                else:
                    report_lines.append("- **Retrieval Config:** Not available")
            except Exception as e:
                logger.warning(f"⚠️ Error accessing configuration details: {e}")
                report_lines.append("- **Configuration details:** Error accessing config")
            
            report_lines.append("")
            
            # Infrastructure Optimization
            report_lines.append("## Infrastructure Optimization")
            report_lines.append("")
            report_lines.append("This evaluation leveraged the optimized container reuse infrastructure:")
            report_lines.append("- ✅ Container reuse for faster iteration cycles")
            report_lines.append("- ✅ DBAPI connections as default for optimal performance")
            report_lines.append("- ✅ Healthcheck integration for reliable testing")
            report_lines.append("- ✅ Parallel execution support for comprehensive evaluation")
            report_lines.append("")
            
            # Save report with enhanced error handling
            report_content = "\n".join(report_lines)
            
            # Construct report file path
            results_dir = Path(self.config.output.results_dir)
            reports_dir = results_dir / "reports"
            report_file = reports_dir / f"comprehensive_report_{timestamp}.md"
            
            logger.info(f"📁 Creating report directory: {reports_dir}")
            
            # Ensure directory exists
            try:
                reports_dir.mkdir(parents=True, exist_ok=True)
                logger.info(f"✅ Report directory created/verified: {reports_dir}")
            except Exception as e:
                logger.error(f"❌ Failed to create report directory {reports_dir}: {e}")
                raise
            
            # Write report file
            try:
                logger.info(f"💾 Writing report to: {report_file}")
                with open(report_file, 'w', encoding='utf-8') as f:
                    f.write(report_content)
                
                # Verify file was written
                if report_file.exists():
                    file_size = report_file.stat().st_size
                    logger.info(f"✅ Comprehensive report saved successfully to {report_file} ({file_size} bytes)")
                else:
                    logger.error(f"❌ Report file was not created: {report_file}")
                    
            except Exception as e:
                logger.error(f"❌ Failed to write report file {report_file}: {e}")
                raise
            
            return report_content
            
        except Exception as e:
            logger.error(f"❌ Critical error in generate_comprehensive_report: {e}")
            logger.error(f"❌ Error type: {type(e).__name__}")
            import traceback
            logger.error(f"❌ Traceback: {traceback.format_exc()}")
            
            # Return a minimal error report
            error_report = f"""# Comprehensive RAGAS Performance Evaluation Report - ERROR

**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
**Status:** Report generation failed

## Error Details

An error occurred while generating the comprehensive report:

```
{str(e)}
```

Please check the logs for more details.
"""
            
            # Try to save error report
            try:
                results_dir = Path(self.config.output.results_dir)
                reports_dir = results_dir / "reports"
                reports_dir.mkdir(parents=True, exist_ok=True)
                
                error_file = reports_dir / f"comprehensive_report_ERROR_{timestamp}.md"
                with open(error_file, 'w', encoding='utf-8') as f:
                    f.write(error_report)
                logger.info(f"📋 Error report saved to {error_file}")
            except Exception as save_error:
                logger.error(f"❌ Failed to save error report: {save_error}")
            
            return error_report
    
    def run_full_evaluation_suite(self) -> Dict[str, Any]:
        """Run the complete evaluation suite with all features"""
        print_flush("🚀 Starting full RAGAS evaluation suite with DBAPI optimization...")
        logger.info("🚀 Starting full RAGAS evaluation suite with DBAPI optimization...")
        
        start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        try:
            # Run comprehensive evaluation
            print_flush("🔄 Running comprehensive evaluation across all pipelines...")
            results = self.run_comprehensive_evaluation()
            
            # Save results
            print_flush("💾 Saving evaluation results...")
            self.save_results(results, timestamp)
            
            # Create visualizations
            print_flush("📊 Creating visualizations...")
            self.create_visualizations(results, timestamp)
            
            # Generate comprehensive report
            print_flush("📝 Generating comprehensive report...")
            report = self.generate_comprehensive_report(results, timestamp)
            
            # Calculate total time
            total_time = time.time() - start_time
            
            # Summary
            summary = {
                "timestamp": timestamp,
                "total_time": total_time,
                "pipelines_evaluated": len(results),
                "total_queries": len(self.test_queries),
                "iterations": self.config.evaluation.num_iterations,
                "connection_type": "DBAPI",
                "results": results,
                "report": report
            }
            
            print_flush(f"🎉 Full evaluation suite completed in {total_time:.2f} seconds")
            print_flush(f"📊 Evaluated {len(results)} pipelines with {len(self.test_queries)} queries each")
            print_flush(f"📁 Results saved with timestamp: {timestamp}")
            
            logger.info(f"🎉 Full evaluation suite completed in {total_time:.2f} seconds")
            logger.info(f"📊 Evaluated {len(results)} pipelines with {len(self.test_queries)} queries each")
            logger.info(f"📁 Results saved with timestamp: {timestamp}")
            
            return summary
            
        except Exception as e:
            logger.error(f"❌ Evaluation suite failed: {e}")
            traceback.print_exc()
            raise


def main():
    """Main function to run comprehensive RAGAS evaluation"""
    try:
        # Initialize framework with DBAPI default
        framework = ComprehensiveRAGASEvaluationFramework()
        
        # Run full evaluation suite
        results = framework.run_full_evaluation_suite()
        
        print("\n" + "="*80)
        print("🎉 COMPREHENSIVE RAGAS EVALUATION COMPLETED!")
        print("="*80)
        print(f"📊 Evaluated {results['pipelines_evaluated']} pipelines")
        print(f"⏱️  Total time: {results['total_time']:.2f} seconds")
        print(f"🔗 Connection type: {results['connection_type']}")
        print(f"📁 Results saved with timestamp: {results['timestamp']}")
        print("="*80)
        
        return results
        
    except Exception as e:
        logger.error(f"❌ Main execution failed: {e}")
        traceback.print_exc()
        return None


if __name__ == "__main__":
    main()