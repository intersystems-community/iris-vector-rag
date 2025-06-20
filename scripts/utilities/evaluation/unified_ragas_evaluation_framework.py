#!/usr/bin/env python3
"""
Unified RAGAS-based Evaluation Framework
Consolidates all scattered testing code with consistent imports and comprehensive evaluation
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
from enum import Enum

# Suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Add project root to path
# Correctly navigate three levels up from scripts/utilities/evaluation to the workspace root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.iris_connection_manager import IRISConnectionManager
from .config_manager import ConfigManager # This is scripts.utilities.evaluation.config_manager
from iris_rag.config.manager import ConfigurationManager as IrisConfigManager # This is iris_rag.config.manager

# Import RAG pipeline classes
from iris_rag.pipelines.basic import BasicRAGPipeline
from iris_rag.pipelines.colbert import ColBERTRAGPipeline # Corrected class name
from iris_rag.pipelines.crag import CRAGPipeline
from iris_rag.pipelines.graphrag import GraphRAGPipeline
from iris_rag.pipelines.hybrid_ifind import HybridIFindRAGPipeline
from iris_rag.pipelines.hyde import HyDERAGPipeline # Corrected class name
from iris_rag.pipelines.noderag import NodeRAGPipeline
from iris_rag.embeddings.manager import EmbeddingManager # Added for NodeRAG

# Visualization imports
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
from plotly.subplots import make_subplots

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

# Statistical analysis
try:
    from scipy import stats
    from scipy.stats import ttest_ind, mannwhitneyu
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("⚠️ SciPy not available for statistical testing")

# Core pipeline imports - FIXED PATHS
try:
    from core_pipelines.basic_rag_pipeline import BasicRAGPipeline
except ImportError:
    BasicRAGPipeline = None
    
try:
    from core_pipelines.hyde_pipeline import HyDEPipeline
except ImportError:
    HyDEPipeline = None
    
try:
    from core_pipelines.crag_pipeline import CRAGPipeline
except ImportError:
    CRAGPipeline = None
    
try:
    from core_pipelines.colbert_pipeline import ColBERTPipeline
except ImportError:
    ColBERTPipeline = None
    
try:
    from core_pipelines.noderag_pipeline import NodeRAGPipeline
except ImportError:
    NodeRAGPipeline = None
    
try:
    from core_pipelines.graphrag_pipeline import GraphRAGPipeline
except ImportError:
    GraphRAGPipeline = None

# Common utilities - FIXED PATHS
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

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ConnectionType(Enum):
    """Connection type enumeration"""
    DBAPI = "dbapi"
    JDBC = "jdbc"

class ChunkingMethod(Enum):
    """Chunking method enumeration"""
    FIXED_SIZE = "fixed_size"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"
    SENTENCE = "sentence"

@dataclass
class EvaluationConfig:
    """Configuration for evaluation parameters"""
    # Pipeline parameters
    top_k: int = 10
    similarity_threshold: float = 0.1
    
    # Chunking parameters
    chunk_size: int = 512
    chunk_overlap: int = 50
    chunking_method: ChunkingMethod = ChunkingMethod.FIXED_SIZE
    
    # Connection parameters
    connection_type: ConnectionType = ConnectionType.DBAPI
    
    # Evaluation parameters
    enable_ragas: bool = True
    enable_statistical_testing: bool = True
    num_iterations: int = 3
    
    # Output parameters
    save_results: bool = True
    create_visualizations: bool = True
    results_dir: str = "eval_results"

@dataclass
class QueryResult:
    """Standardized query result structure"""
    query: str
    answer: str
    contexts: List[str]
    ground_truth: str
    keywords: List[str]
    response_time: float
    documents_retrieved: int
    avg_similarity_score: float
    answer_length: int
    success: bool
    error: Optional[str] = None
    pipeline_name: str = ""
    iteration: int = 0

@dataclass
class PipelineMetrics:
    """Aggregated metrics for a pipeline"""
    pipeline_name: str
    success_rate: float
    avg_response_time: float
    avg_documents_retrieved: float
    avg_similarity_score: float
    avg_answer_length: float
    ragas_scores: Optional[Dict[str, float]] = None
    individual_results: List[QueryResult] = field(default_factory=list)

class UnifiedRAGASEvaluationFramework:
    """Unified evaluation framework with RAGAS integration"""
    
    def __init__(self, config: Union[EvaluationConfig, ComprehensiveConfig, str, Path] = None):
        """Initialize the evaluation framework"""
        # Handle different config types
        if isinstance(config, (str, Path)):
            # Load from file
            config_manager = ConfigManager()
            self.comprehensive_config = config_manager.load_config(config)
        elif isinstance(config, ComprehensiveConfig):
            self.comprehensive_config = config
        elif isinstance(config, EvaluationConfig):
            # Convert old config to new format
            self.comprehensive_config = ComprehensiveConfig()
            self.comprehensive_config.evaluation = config
        else:
            # Load from environment
            config_manager = ConfigManager()
            self.comprehensive_config = config_manager.load_config()
        
        # Extract legacy config for backward compatibility
        self.config = self.comprehensive_config.evaluation

        # Create results directory first
        self._setup_results_directory() # Uses self.comprehensive_config.output.results_dir
        
        # Setup logging
        self._setup_logging() # Uses self.comprehensive_config.output.results_dir
        
        # Initialize ConfigManager (can be useful for other operations if needed)
        # The main config object (self.comprehensive_config) is already loaded and passed in.
        self.config_manager = ConfigManager()

        # Note: The comprehensive_config is passed in directly to __init__.
        # The logic for deciding which config file to load (dev, specific, or default)
        # is handled by the calling script (run_unified_evaluation.py) before this class is instantiated.
        # self.comprehensive_config is already set from the __init__ parameter.

        # Extract legacy evaluation-specific config for backward compatibility
        # This assumes self.comprehensive_config is correctly populated by the caller.
        self.config = self.comprehensive_config.evaluation

        # Initialize ConnectionManager and database connection
        self.db_connection_manager: Optional[IRISConnectionManager] = None
        self.db_connection: Optional[Any] = None
        self._initialize_db_connection_and_manager() # Uses self.comprehensive_config.database

        # Initialize embedding and LLM functions
        self.embedding_func, self.llm_func = self._initialize_models() # Uses self.comprehensive_config
        
        # Initialize RAGAS components
        self.ragas_llm, self.ragas_embeddings = self._initialize_ragas() # Uses self.comprehensive_config
        
        # Initialize pipelines
        self.pipelines = self._initialize_pipelines() # Uses self.db_connection_manager, self.config_manager, etc.
        
        # Load test queries
        self.test_queries = self._load_test_queries() # Uses self.comprehensive_config

    def _setup_logging(self):
        """Setup logging configuration"""
        log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        logging.basicConfig(
            level=logging.INFO,
            format=log_format,
            handlers=[
                logging.StreamHandler(),
                logging.FileHandler(f"{self.comprehensive_config.output.results_dir}/evaluation.log")
            ]
        )
        
    def _setup_results_directory(self):
        """Create results directory if it doesn't exist"""
        Path(self.comprehensive_config.output.results_dir).mkdir(parents=True, exist_ok=True)

    def _initialize_db_connection_and_manager(self) -> None:
        """Initialize IRISConnectionManager and the database connection."""
        db_conf_obj = self.comprehensive_config.database
        try:
            db_params_dict = asdict(db_conf_obj)
            prefer_dbapi_flag = (db_conf_obj.connection_type.lower() == "dbapi")

            self.db_connection_manager = IRISConnectionManager(prefer_dbapi=prefer_dbapi_flag)
            self.db_connection = self.db_connection_manager.get_connection(config=db_params_dict)
            
            connection_type_used = self.db_connection_manager.get_connection_type()
            if self.db_connection:
                if connection_type_used == "DBAPI":
                    logger.info("✅ DBAPI connection initialized and stored via IRISConnectionManager")
                elif connection_type_used == "JDBC":
                    logger.info("✅ JDBC connection initialized and stored via IRISConnectionManager")
                else:
                    logger.warning(f"⚠️ Connection established (type: {connection_type_used}), stored.")
            else:
                logger.error(f"❌ Failed to establish database connection via IRISConnectionManager.")
        except Exception as e:
            logger.error(f"❌ Database connection and manager initialization failed: {e}", exc_info=True)
            self.db_connection_manager = None
            self.db_connection = None
        
    def _initialize_models(self) -> Tuple[Callable, Callable]:
        """Initialize embedding and LLM functions"""
        try:
            # Initialize embedding function
            embedding_model = get_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
            embedding_func = lambda texts: embedding_model.encode(texts)
            
            # Initialize LLM function
            if os.getenv("OPENAI_API_KEY"):
                llm_func = get_llm_func("openai")
                logger.info("✅ Using OpenAI LLM")
            else:
                llm_func = lambda prompt: f"Based on the provided context: {prompt[:100]}..."
                logger.warning("⚠️ Using stub LLM (set OPENAI_API_KEY for real evaluation)")
                
            return embedding_func, llm_func
            
        except Exception as e:
            logger.error(f"❌ Model initialization failed: {e}")
            # Return stub functions
            return (
                lambda texts: [[0.0] * 384 for _ in texts],
                lambda prompt: f"Stub response to: {prompt[:50]}..."
            )
    
    def _initialize_ragas(self) -> Tuple[Any, Any]:
        """Initialize RAGAS components"""
        if not RAGAS_AVAILABLE or not LANGCHAIN_AVAILABLE:
            return None, None
            
        try:
            if os.getenv("OPENAI_API_KEY"):
                ragas_llm = ChatOpenAI(
                    model_name="gpt-3.5-turbo",
                    temperature=0,
                    openai_api_key=os.getenv("OPENAI_API_KEY")
                )
                ragas_embeddings = HuggingFaceEmbeddings(
                    model_name='sentence-transformers/all-MiniLM-L6-v2',
                    model_kwargs={'device': 'cpu'}
                )
                logger.info("✅ RAGAS components initialized with OpenAI")
                return ragas_llm, ragas_embeddings
            else:
                logger.warning("⚠️ RAGAS requires OpenAI API key")
                return None, None
                
        except Exception as e:
            logger.error(f"❌ RAGAS initialization failed: {e}")
            return None, None
    
    def _initialize_pipelines(self) -> Dict[str, Any]:
        """Initialize all RAG pipelines with standardized parameters"""
        pipelines = {}
        
        if not self.db_connection_manager or not self.db_connection:
            logger.error("❌ Database connection manager or connection not initialized. Cannot initialize pipelines.")
            return pipelines
        
        # ConfigManager instance is self.config_manager
        # ComprehensiveConfig instance is self.comprehensive_config

        # Define available pipeline configurations
        # Note: Ensure class names BasicRAGPipeline, CRAGPipeline, etc. are correctly imported
        available_pipelines = {
            "BasicRAG": (BasicRAGPipeline, {
                "connection_manager": self.db_connection_manager,
                "config_manager": IrisConfigManager(),
                "llm_func": self.llm_func
            }),
            "HyDE": (HyDERAGPipeline, {
                "connection_manager": self.db_connection_manager,
                "config_manager": IrisConfigManager(),
                "llm_func": self.llm_func
            }),
            "CRAG": (CRAGPipeline, {
                "connection_manager": self.db_connection_manager,
                "config_manager": IrisConfigManager(),
                "llm_func": self.llm_func,
                "embedding_func": self.embedding_func
            }),
            "ColBERT": (ColBERTRAGPipeline, {
                "connection_manager": self.db_connection_manager,
                "config_manager": IrisConfigManager(),
                "colbert_query_encoder": self.embedding_func,
                "llm_func": self.llm_func
            }),
            "NodeRAG": (NodeRAGPipeline, {
                "connection_manager": self.db_connection_manager,
                "config_manager": IrisConfigManager(),
                "embedding_manager": EmbeddingManager(IrisConfigManager()),
                "llm_func": self.llm_func
            }),
            "GraphRAG": (GraphRAGPipeline, {
                "connection_manager": self.db_connection_manager,
                "config_manager": IrisConfigManager(),
                "llm_func": self.llm_func
            }),
            "HybridIFind": (HybridIFindRAGPipeline, {
                "connection_manager": self.db_connection_manager,
                "config_manager": IrisConfigManager(),
                "llm_func": self.llm_func
            })
        }
        
        # Initialize only enabled pipelines
        for name, (pipeline_class, kwargs) in available_pipelines.items():
            # Check if pipeline is enabled in configuration
            pipeline_config = self.comprehensive_config.pipelines.get(name)
            if pipeline_config and not pipeline_config.enabled:
                logger.info(f"⏭️ {name} pipeline disabled in configuration")
                continue
                
            # Check if pipeline class is available
            if pipeline_class is None:
                logger.warning(f"⚠️ {name} pipeline class not available (import failed)")
                continue
            
            try:
                # Add custom parameters if specified
                if pipeline_config and pipeline_config.custom_params:
                    kwargs.update(pipeline_config.custom_params)
                
                pipelines[name] = pipeline_class(**kwargs)
                logger.info(f"✅ {name} pipeline initialized")
            except Exception as e:
                logger.error(f"❌ {name} pipeline failed: {e}")
                if logger.isEnabledFor(logging.DEBUG):
                    traceback.print_exc()
        
        logger.info(f"🚀 Initialized {len(pipelines)} RAG pipelines")
        return pipelines
    
    def _load_test_queries(self) -> List[Dict[str, Any]]:
        """Load test queries from configuration"""
        # Try to load from sample_queries.json first
        sample_queries_path = Path("eval/sample_queries.json")
        if sample_queries_path.exists():
            try:
                with open(sample_queries_path, 'r') as f:
                    queries_data = json.load(f)
                    
                # Convert to expected format
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
        
        # Fallback to default queries
        return self._get_default_queries()
    
    def _get_default_queries(self) -> List[Dict[str, Any]]:
        """Get default test queries"""
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
        # Simple keyword extraction - can be enhanced with NLP
        import re
        words = re.findall(r'\b\w+\b', query.lower())
        # Filter out common stop words
        stop_words = {'what', 'how', 'the', 'is', 'are', 'of', 'in', 'to', 'and', 'or', 'for', 'with'}
        keywords = [word for word in words if word not in stop_words and len(word) > 3]
        return keywords[:5]  # Return top 5 keywords
    
    def run_single_query(self, pipeline_name: str, query_data: Dict[str, Any], iteration: int = 0) -> QueryResult:
        """Run a single query and collect standardized metrics"""
        pipeline = self.pipelines[pipeline_name]
        query = query_data["query"]
        
        start_time = time.time()
        try:
            # Run pipeline with standardized parameters
            result = pipeline.run(
                query,
                top_k=self.comprehensive_config.retrieval.top_k,
                similarity_threshold=self.comprehensive_config.retrieval.similarity_threshold
            )
            
            response_time = time.time() - start_time
            
            # Extract standardized information
            documents = result.get('retrieved_documents', [])
            answer = result.get('answer', '')
            
            # Extract contexts for RAGAS - prefer pre-extracted contexts
            contexts = result.get('contexts')
            if contexts is None:
                # Fall back to extracting from documents if contexts not provided
                contexts = self._extract_contexts(documents)
            
            # Calculate similarity scores
            similarity_scores = self._extract_similarity_scores(documents)
            avg_similarity = np.mean(similarity_scores) if similarity_scores else 0.0
            
            return QueryResult(
                query=query,
                answer=answer,
                contexts=contexts,
                ground_truth=query_data.get('ground_truth', ''),
                keywords=query_data.get('keywords', []),
                response_time=response_time,
                documents_retrieved=len(documents),
                avg_similarity_score=avg_similarity,
                answer_length=len(answer),
                success=True,
                pipeline_name=pipeline_name,
                iteration=iteration
            )
            
        except Exception as e:
            logger.error(f"❌ {pipeline_name} failed for query '{query[:50]}...': {e}")
            return QueryResult(
                query=query,
                answer='',
                contexts=[],
                ground_truth=query_data.get('ground_truth', ''),
                keywords=query_data.get('keywords', []),
                response_time=time.time() - start_time,
                documents_retrieved=0,
                avg_similarity_score=0.0,
                answer_length=0,
                success=False,
                error=str(e),
                pipeline_name=pipeline_name,
                iteration=iteration
            )
    
    def _extract_contexts(self, documents: List[Any]) -> List[str]:
        """Extract context texts from documents"""
        contexts = []
        for doc in documents:
            if isinstance(doc, dict):
                text = doc.get('text', '') or doc.get('content', '') or doc.get('chunk_text', '')
            elif hasattr(doc, 'text'):
                text = doc.text
            elif hasattr(doc, 'content'):
                text = doc.content
            else:
                text = str(doc)
            if text:
                contexts.append(text)
        return contexts
    
    def _extract_similarity_scores(self, documents: List[Any]) -> List[float]:
        """Extract similarity scores from documents"""
        scores = []
        for doc in documents:
            if isinstance(doc, dict) and 'score' in doc:
                scores.append(doc['score'])
            elif hasattr(doc, 'score'):
                scores.append(doc.score)
        return scores
    
    def evaluate_with_ragas(self, results: List[QueryResult]) -> Optional[Dict[str, float]]:
        """Evaluate results using RAGAS metrics"""
        if not RAGAS_AVAILABLE or not self.ragas_llm or not self.ragas_embeddings:
            logger.warning("⚠️ RAGAS not available, skipping quality evaluation")
            return None
        
        # Filter valid results
        valid_results = [r for r in results if r.success and r.answer and r.contexts]
        
        if not valid_results:
            logger.warning("⚠️ No valid results for RAGAS evaluation")
            return None
        
        try:
            # Prepare data for RAGAS
            data = {
                'question': [r.query for r in valid_results],
                'answer': [r.answer for r in valid_results],
                'contexts': [r.contexts for r in valid_results],
                'ground_truth': [r.ground_truth for r in valid_results]
            }
            
            dataset = Dataset.from_dict(data)
            
            # Select metrics based on available data
            metrics = [answer_relevancy, faithfulness]
            if all(r.ground_truth for r in valid_results):
                metrics.extend([answer_similarity, answer_correctness])
            if all(r.contexts for r in valid_results):
                metrics.extend([context_precision])
            
            # Run RAGAS evaluation
            logger.info("🔍 Running RAGAS evaluation...")
            ragas_results = evaluate(
                dataset,
                metrics=metrics,
                llm=self.ragas_llm,
                embeddings=self.ragas_embeddings
            )
            
            return ragas_results
            
        except Exception as e:
            logger.error(f"❌ RAGAS evaluation failed: {e}")
            traceback.print_exc()
            return None
    
    def run_comprehensive_evaluation(self) -> Dict[str, PipelineMetrics]:
        """Run comprehensive evaluation with multiple iterations"""
        logger.info("🚀 Starting comprehensive RAG evaluation...")
        
        all_results = {}
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        for pipeline_name in self.pipelines.keys():
            logger.info(f"\n📊 Evaluating {pipeline_name}...")
            
            pipeline_results = []
            
            # Run multiple iterations for statistical significance
            for iteration in range(self.config.num_iterations):
                logger.info(f"  Iteration {iteration + 1}/{self.config.num_iterations}")
                
                for i, query_data in enumerate(self.test_queries):
                    logger.info(f"    Query {i+1}/{len(self.test_queries)}: {query_data['query'][:50]}...")
                    
                    result = self.run_single_query(pipeline_name, query_data, iteration)
                    pipeline_results.append(result)
                    
                    time.sleep(0.1)  # Brief pause between queries
            
            # Calculate aggregate metrics
            successful_results = [r for r in pipeline_results if r.success]
            
            if successful_results:
                # Performance metrics
                success_rate = len(successful_results) / len(pipeline_results)
                avg_response_time = np.mean([r.response_time for r in successful_results])
                avg_documents = np.mean([r.documents_retrieved for r in successful_results])
                avg_similarity = np.mean([r.avg_similarity_score for r in successful_results])
                avg_answer_length = np.mean([r.answer_length for r in successful_results])
                
                # RAGAS evaluation
                ragas_scores = None
                if self.config.enable_ragas:
                    ragas_scores = self.evaluate_with_ragas(successful_results)
                
                metrics = PipelineMetrics(
                    pipeline_name=pipeline_name,
                    success_rate=success_rate,
                    avg_response_time=avg_response_time,
                    avg_documents_retrieved=avg_documents,
                    avg_similarity_score=avg_similarity,
                    avg_answer_length=avg_answer_length,
                    ragas_scores=ragas_scores,
                    individual_results=pipeline_results
                )
                
                all_results[pipeline_name] = metrics
                
                logger.info(f"✅ {pipeline_name}: {len(successful_results)}/{len(pipeline_results)} successful")
                if ragas_scores:
                    logger.info(f"   RAGAS Scores: {ragas_scores}")
            else:
                logger.error(f"❌ {pipeline_name}: No successful queries")
                all_results[pipeline_name] = PipelineMetrics(
                    pipeline_name=pipeline_name,
                    success_rate=0,
                    avg_response_time=0,
                    avg_documents_retrieved=0,
                    avg_similarity_score=0,
                    avg_answer_length=0,
                    individual_results=pipeline_results
                )
        
        # Save results
        if self.comprehensive_config.output.save_results:
            self._save_results(all_results, timestamp)
        
        # Create visualizations
        if self.comprehensive_config.output.create_visualizations:
            self._create_visualizations(all_results, timestamp)
        
        # Perform statistical analysis
        if self.comprehensive_config.evaluation.enable_statistical_testing and SCIPY_AVAILABLE:
            self._perform_statistical_analysis(all_results, timestamp)
        
        return all_results
    
    def _save_results(self, results: Dict[str, PipelineMetrics], timestamp: str):
        """Save evaluation results to JSON"""
        results_file = f"{self.comprehensive_config.output.results_dir}/evaluation_results_{timestamp}.json"
        
        # Convert to serializable format
        serializable_results = {}
        for name, metrics in results.items():
            data = asdict(metrics)
            # individual_results are already converted to dicts by asdict(metrics)
            # No need for: data['individual_results'] = [asdict(r) for r in data['individual_results']]
            
            # Convert RAGAS results to serializable format
            if data.get('ragas_scores') is not None: # Use .get for safety
                # Ensure values are float or handle other potential types if necessary
                serializable_ragas_scores = {}
                for k, v in data['ragas_scores'].items():
                    try:
                        serializable_ragas_scores[k] = float(v)
                    except (ValueError, TypeError):
                        logger.warning(f"Could not convert RAGAS score {k}={v} to float. Storing as string.")
                        serializable_ragas_scores[k] = str(v) # Store as string if not floatable
                data['ragas_scores'] = serializable_ragas_scores
            serializable_results[name] = data
        
        with open(results_file, 'w') as f:
            json.dump(serializable_results, f, indent=2, default=str)
        
        logger.info(f"💾 Results saved to {results_file}")
    
    def _create_visualizations(self, results: Dict[str, PipelineMetrics], timestamp: str):
        """Create comprehensive visualizations"""
        # Performance comparison
        self._create_performance_comparison(results, timestamp)
        
        # RAGAS comparison
        if any(metrics.ragas_scores for metrics in results.values()):
            self._create_ragas_comparison(results, timestamp)
        
        # Spider chart
        self._create_spider_chart(results, timestamp)
        
        logger.info(f"📊 Visualizations created with timestamp: {timestamp}")
    
    def _create_performance_comparison(self, results: Dict[str, PipelineMetrics], timestamp: str):
        """Create performance comparison charts"""
        techniques = list(results.keys())
        response_times = [results[t].avg_response_time for t in techniques]
        documents_retrieved = [results[t].avg_documents_retrieved for t in techniques]
        similarity_scores = [results[t].avg_similarity_score for t in techniques]
        success_rates = [results[t].success_rate for t in techniques]
        
        fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
        
        # Response Time
        bars1 = ax1.bar(techniques, response_times, color='skyblue', alpha=0.8)
        ax1.set_title('Average Response Time', fontsize=16, fontweight='bold')
        ax1.set_ylabel('Seconds', fontsize=12)
        ax1.tick_params(axis='x', rotation=45)
        
        # Documents Retrieved
        bars2 = ax2.bar(techniques, documents_retrieved, color='lightgreen', alpha=0.8)
        ax2.set_title('Average Documents Retrieved', fontsize=16, fontweight='bold')
        ax2.set_ylabel('Number of Documents', fontsize=12)
        ax2.tick_params(axis='x', rotation=45)
        
        # Similarity Scores
        bars3 = ax3.bar(techniques, similarity_scores, color='orange', alpha=0.8)
        ax3.set_title('Average Similarity Score', fontsize=16, fontweight='bold')
        ax3.set_ylabel('Similarity Score', fontsize=12)
        ax3.tick_params(axis='x', rotation=45)
        
        # Success Rate
        bars4 = ax4.bar(techniques, success_rates, color='lightcoral', alpha=0.8)
        ax4.set_title('Success Rate', fontsize=16, fontweight='bold')
        ax4.set_ylabel('Success Rate', fontsize=12)
        ax4.tick_params(axis='x', rotation=45)
        
        plt.tight_layout()
        plt.savefig(f"{self.comprehensive_config.output.results_dir}/performance_comparison_{timestamp}.png",
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_ragas_comparison(self, results: Dict[str, PipelineMetrics], timestamp: str):
        """Create RAGAS metrics comparison"""
        techniques = []
        ragas_data = {}
        
        for name, metrics in results.items():
            if metrics.ragas_scores:
                techniques.append(name)
                for metric, score in metrics.ragas_scores.items():
                    if metric not in ragas_data:
                        ragas_data[metric] = []
                    ragas_data[metric].append(score)
        
        if not techniques:
            return
        
        # Create grouped bar chart
        x = np.arange(len(techniques))
        width = 0.15
        
        fig, ax = plt.subplots(figsize=(14, 8))
        
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4', '#FFEAA7', '#DDA0DD']
        
        for i, (metric, scores) in enumerate(ragas_data.items()):
            ax.bar(x + i * width, scores, width, label=metric, color=colors[i % len(colors)])
        
        ax.set_xlabel('RAG Techniques')
        ax.set_ylabel('RAGAS Scores')
        ax.set_title('RAGAS Metrics Comparison')
        ax.set_xticks(x + width * (len(ragas_data) - 1) / 2)
        ax.set_xticklabels(techniques, rotation=45)
        ax.legend()
        ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(f"{self.comprehensive_config.output.results_dir}/ragas_comparison_{timestamp}.png",
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _create_spider_chart(self, results: Dict[str, PipelineMetrics], timestamp: str):
        """Create comprehensive spider chart"""
        techniques = list(results.keys())
        
        # Normalize metrics to 0-1 scale
        metrics_data = {
            'Success Rate': [results[t].success_rate for t in techniques],
            'Response Time': [1 / (1 + results[t].avg_response_time) for t in techniques],  # Inverse for better visualization
            'Documents Retrieved': [min(results[t].avg_documents_retrieved / 10, 1) for t in techniques],
            'Similarity Score': [results[t].avg_similarity_score for t in techniques],
        }
        
        # Add RAGAS metrics if available
        if any(results[t].ragas_scores for t in techniques):
            for metric in ['answer_relevancy', 'faithfulness', 'context_precision']:
                scores = []
                for t in techniques:
                    if results[t].ragas_scores and metric in results[t].ragas_scores:
                        scores.append(results[t].ragas_scores[metric])
                    else:
                        scores.append(0)
                if any(scores):
                    metrics_data[metric.replace('_', ' ').title()] = scores
        
        # Create spider chart
        angles = np.linspace(0, 2 * np.pi, len(metrics_data), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        fig, ax = plt.subplots(figsize=(12, 12), subplot_kw=dict(projection='polar'))
        
        colors = plt.cm.Set3(np.linspace(0, 1, len(techniques)))
        
        for i, technique in enumerate(techniques):
            values = [metrics_data[metric][i] for metric in metrics_data.keys()]
            values += values[:1]  # Complete the circle
            
            ax.plot(angles, values, 'o-', linewidth=2, label=technique, color=colors[i])
            ax.fill(angles, values, alpha=0.25, color=colors[i])
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics_data.keys())
        ax.set_ylim(0, 1)
        ax.set_title('RAG Techniques Comparison - Spider Chart', size=16, fontweight='bold', pad=20)
        ax.legend(loc='upper right', bbox_to_anchor=(1.3, 1.0))
        ax.grid(True)
        
        plt.tight_layout()
        plt.savefig(f"{self.comprehensive_config.output.results_dir}/spider_chart_{timestamp}.png",
                   dpi=300, bbox_inches='tight')
        plt.close()
    
    def _perform_statistical_analysis(self, results: Dict[str, PipelineMetrics], timestamp: str):
        """Perform statistical significance testing"""
        if not SCIPY_AVAILABLE:
            logger.warning("⚠️ SciPy not available for statistical testing")
            return
        
        logger.info("📊 Performing statistical analysis...")
        
        analysis_results = {}
        techniques = list(results.keys())
        
        # Compare response times
        response_time_data = {}
        for name, metrics in results.items():
            if metrics.individual_results:
                response_times = [r.response_time for r in metrics.individual_results if r.success]
                if response_times:
                    response_time_data[name] = response_times
        
        # Pairwise comparisons
        comparisons = []
        for i, tech1 in enumerate(techniques):
            for tech2 in techniques[i+1:]:
                if tech1 in response_time_data and tech2 in response_time_data:
                    data1 = response_time_data[tech1]
                    data2 = response_time_data[tech2]
                    
                    # Perform t-test
                    try:
                        t_stat, p_value = ttest_ind(data1, data2)
                        comparisons.append({
                            'technique1': tech1,
                            'technique2': tech2,
                            'metric': 'response_time',
                            't_statistic': t_stat,
                            'p_value': p_value,
                            'significant': p_value < 0.05
                        })
                    except Exception as e:
                        logger.warning(f"Statistical test failed for {tech1} vs {tech2}: {e}")
        
        analysis_results['pairwise_comparisons'] = comparisons
        
        # Save statistical analysis
        stats_file = f"{self.comprehensive_config.output.results_dir}/statistical_analysis_{timestamp}.json"
        with open(stats_file, 'w') as f:
            json.dump(analysis_results, f, indent=2, default=str)
        
        logger.info(f"📊 Statistical analysis saved to {stats_file}")
    
    def generate_report(self, results: Dict[str, PipelineMetrics], timestamp: str) -> str:
        """Generate comprehensive evaluation report"""
        report_lines = [
            "# RAG Evaluation Report",
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "## Configuration",
            f"- Top K: {self.comprehensive_config.retrieval.top_k}",
            f"- Similarity Threshold: {self.comprehensive_config.retrieval.similarity_threshold}",
            f"- Connection Type: {self.comprehensive_config.database.connection_type if isinstance(self.comprehensive_config.database.connection_type, str) else self.comprehensive_config.database.connection_type.value}",
            f"- Chunking Method: {self.comprehensive_config.chunking.method if isinstance(self.comprehensive_config.chunking.method, str) else self.comprehensive_config.chunking.method.value}",
            f"- Number of Iterations: {self.comprehensive_config.evaluation.num_iterations}",
            "",
            "## Results Summary",
            ""
        ]
        
        # Add results table
        report_lines.append("| Technique | Success Rate | Avg Response Time | Avg Documents | Avg Similarity |")
        report_lines.append("|-----------|--------------|-------------------|---------------|----------------|")
        
        for name, metrics in results.items():
            report_lines.append(
                f"| {name} | {metrics.success_rate:.2%} | {metrics.avg_response_time:.3f}s | "
                f"{metrics.avg_documents_retrieved:.1f} | {metrics.avg_similarity_score:.3f} |"
            )
        
        # Add RAGAS results if available
        ragas_techniques = [name for name, metrics in results.items() if metrics.ragas_scores]
        if ragas_techniques:
            report_lines.extend([
                "",
                "## RAGAS Quality Metrics",
                ""
            ])
            
            # Create RAGAS table
            all_metrics = set()
            for name in ragas_techniques:
                all_metrics.update(results[name].ragas_scores.keys())
            
            header = "| Technique |" + "".join(f" {metric} |" for metric in sorted(all_metrics))
            separator = "|-----------|" + "".join("----------|" for _ in all_metrics)
            
            report_lines.append(header)
            report_lines.append(separator)
            
            for name in ragas_techniques:
                row = f"| {name} |"
                for metric in sorted(all_metrics):
                    score = results[name].ragas_scores.get(metric, 0)
                    row += f" {score:.3f} |"
                report_lines.append(row)
        
        # Add recommendations
        report_lines.extend([
            "",
            "## Recommendations",
            ""
        ])
        
        # Find best performing technique
        best_success = max(results.values(), key=lambda x: x.success_rate)
        fastest = min([m for m in results.values() if m.success_rate > 0],
                     key=lambda x: x.avg_response_time, default=None)
        
        if best_success:
            report_lines.append(f"- **Highest Success Rate**: {best_success.pipeline_name} ({best_success.success_rate:.2%})")
        
        if fastest:
            report_lines.append(f"- **Fastest Response**: {fastest.pipeline_name} ({fastest.avg_response_time:.3f}s)")
        
        # Add quality recommendations if RAGAS available
        if ragas_techniques:
            best_quality = max(
                [results[name] for name in ragas_techniques],
                key=lambda x: sum(x.ragas_scores.values()) / len(x.ragas_scores)
            )
            avg_quality = sum(best_quality.ragas_scores.values()) / len(best_quality.ragas_scores)
            report_lines.append(f"- **Best Overall Quality**: {best_quality.pipeline_name} (avg RAGAS: {avg_quality:.3f})")
        
        report_content = "\n".join(report_lines)
        
        # Save report
        report_file = f"{self.comprehensive_config.output.results_dir}/evaluation_report_{timestamp}.md"
        with open(report_file, 'w') as f:
            f.write(report_content)
        
        logger.info(f"📄 Report saved to {report_file}")
        return report_content


def main():
    """Main execution function"""
    print("🚀 Starting Unified RAGAS Evaluation Framework")
    
    # Create configuration
    config = EvaluationConfig(
        top_k=10,
        similarity_threshold=0.1,
        connection_type=ConnectionType.DBAPI,
        enable_ragas=True,
        enable_statistical_testing=True,
        num_iterations=3,
        save_results=True,
        create_visualizations=True
    )
    
    # Initialize framework
    framework = UnifiedRAGASEvaluationFramework(config)
    
    # Run evaluation
    results = framework.run_comprehensive_evaluation()
    
    # Generate report
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report = framework.generate_report(results, timestamp)
    
    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)
    print(report)
    
    return results


if __name__ == "__main__":
    main()