#!/usr/bin/env python3
"""
Comprehensive RAGAS Evaluation Script

Performs a comprehensive RAGAS evaluation on RAG pipelines using dynamic pipeline loading.
This script validates the environment, dataset completeness, initializes RAGAS framework,
loads evaluation queries, executes pipeline evaluations, calculates RAGAS metrics,
and generates comprehensive evaluation reports.

Dynamic Pipeline Loading:
- Pipelines are loaded from config/pipelines.yaml configuration
- Use --pipelines ALL to evaluate all enabled pipelines
- Use --pipelines <name1> <name2> to evaluate specific pipelines by name
- Pipeline names must match those defined in config/pipelines.yaml

Framework Dependencies:
- llm_func: LLM function for answer generation (automatically injected)
- embedding_func: Embedding function for vector operations (automatically injected)
- vector_store: IRIS vector store instance (automatically injected)
- config_manager: Configuration manager instance (automatically injected)
"""

import os
import sys
import json
import time
import logging
import argparse
import langchain
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Load environment variables from .env file
from dotenv import load_dotenv
load_dotenv()

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
sys.path.insert(0, project_root)

# Import RAGAS components
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision, 
    context_recall,
    faithfulness,
    answer_similarity,
    answer_correctness
)

# Import datasets
from datasets import Dataset

# Import LangChain components
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Import cache management
from common.llm_cache_manager import LangchainIRISCacheWrapper, setup_langchain_cache
from common.llm_cache_config import load_cache_config

# Import dynamic pipeline loading services
from iris_rag.config.pipeline_config_service import PipelineConfigService
from iris_rag.utils.module_loader import ModuleLoader
from iris_rag.pipelines.factory import PipelineFactory
from iris_rag.pipelines.registry import PipelineRegistry

# Import connection and config managers
from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager

# Constants
QUERIES_FILE_PATH = "eval/sample_queries.json"
NOT_APPLICABLE_GROUND_TRUTH = "N/A - No ground truth available for this evaluation"
MIN_REQUIRED_DOCUMENTS = 933

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def setup_iris_cache(connection_manager: ConnectionManager, config_manager: ConfigurationManager) -> None:
    """
    Set up IRIS-backed Langchain cache for improved performance.
    
    Args:
        connection_manager: Database connection manager
        config_manager: Configuration manager
    """
    try:
        # Load cache configuration
        cache_config = load_cache_config()
        
        if not cache_config.enabled:
            logger.info("LLM caching is disabled in configuration")
            return
        
        # Get IRIS connection
        iris_connector = connection_manager.get_connection("iris")
        
        # Create IRIS cache backend
        from common.llm_cache_iris import create_iris_cache_backend
        iris_cache_backend = create_iris_cache_backend(cache_config, iris_connector)
        
        # Create Langchain-compatible wrapper
        iris_cache_wrapper = LangchainIRISCacheWrapper(iris_cache_backend)
        
        # Set global Langchain cache
        langchain.llm_cache = iris_cache_wrapper
        
        logger.info("✅ IRIS-backed Langchain cache configured successfully")
        
    except Exception as e:
        logger.warning(f"Failed to setup IRIS cache, continuing without cache: {e}")
        # Continue without cache rather than failing the entire evaluation


def validate_openai_api_key() -> None:
    """
    Validate that the OpenAI API key is set in environment variables.
    
    Raises:
        SystemExit: If the API key is not set or empty
    """
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key or api_key.strip() == "":
        logger.error("OPENAI_API_KEY environment variable is not set or empty")
        sys.exit(1)
    logger.info("✅ OpenAI API key validation passed")


def validate_dataset_completeness(iris_connector) -> None:
    """
    Validate that the dataset has sufficient documents and embeddings.
    
    Args:
        iris_connector: Database connection to IRIS
        
    Raises:
        SystemExit: If dataset requirements are not met
    """
    cursor = iris_connector.cursor()
    
    try:
        # Check total document count
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
        total_docs = cursor.fetchone()[0]
        
        # Check documents with embeddings (main document embeddings, not token embeddings)
        cursor.execute("""
            SELECT COUNT(*) 
            FROM RAG.SourceDocuments 
            WHERE embedding IS NOT NULL AND embedding != ''
        """)
        docs_with_embeddings = cursor.fetchone()[0]
        
        logger.info(f"Total documents: {total_docs}")
        logger.info(f"Documents with embeddings: {docs_with_embeddings}")
        
        if total_docs < MIN_REQUIRED_DOCUMENTS:
            logger.error(f"Insufficient documents: {total_docs} < {MIN_REQUIRED_DOCUMENTS}")
            sys.exit(1)
            
        if docs_with_embeddings < total_docs:
            missing_embeddings = total_docs - docs_with_embeddings
            logger.error(f"Missing embeddings for {missing_embeddings} documents")
            sys.exit(1)
            
        logger.info("✅ Dataset completeness validation passed")
        
    except Exception as e:
        logger.error(f"Dataset validation failed: {e}")
        sys.exit(1)
    finally:
        cursor.close()


def initialize_ragas_framework(config_manager: ConfigurationManager) -> Tuple[ChatOpenAI, OpenAIEmbeddings, List]:
    """
    Initialize RAGAS framework components.
    
    Args:
        config_manager: Configuration manager for accessing RAGAS settings
    
    Returns:
        Tuple of (LLM, embeddings, metrics) for RAGAS evaluation
    """
    # Get RAGAS configuration using the correct method
    llm_model = config_manager.get('ragas:llm:model', 'gpt-4o-mini')
    llm_temperature = config_manager.get('ragas:llm:temperature', 0)
    llm_max_tokens = config_manager.get('ragas:llm:max_tokens', 2048)
    embeddings_model = config_manager.get('ragas:embeddings:model', 'text-embedding-3-small')
    
    # Initialize LLM for RAGAS evaluation with increased max_tokens
    llm = ChatOpenAI(
        model=llm_model,
        temperature=llm_temperature,
        max_tokens=llm_max_tokens  # Increased from 1000 to prevent LLMDidNotFinishException
    )
    
    embeddings = OpenAIEmbeddings(
        model=embeddings_model
    )
    
    # Define RAGAS metrics
    metrics = [
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
        answer_similarity,
        answer_correctness
    ]
    
    logger.info(f"✅ RAGAS framework initialized with max_tokens={llm_max_tokens}")
    return llm, embeddings, metrics


def load_evaluation_queries() -> List[Dict[str, Any]]:
    """
    Load evaluation queries from the sample queries file.
    
    Returns:
        List of query dictionaries
        
    Raises:
        SystemExit: If queries file cannot be loaded
    """
    try:
        with open(QUERIES_FILE_PATH, 'r') as f:
            queries = json.load(f)
        
        logger.info(f"✅ Loaded {len(queries)} evaluation queries")
        return queries
        
    except Exception as e:
        logger.error(f"Failed to load evaluation queries from {QUERIES_FILE_PATH}: {e}")
        sys.exit(1)

def get_pipelines_to_evaluate(connection_manager: ConnectionManager,
                             config_manager: ConfigurationManager,
                             target_pipelines: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Get pipeline instances to evaluate using dynamic loading.
    
    Args:
        connection_manager: Database connection manager
        config_manager: Configuration manager
        target_pipelines: Specific pipelines to evaluate (None for all enabled)
        
    Returns:
        Dictionary mapping pipeline names to their instances
    """
    try:
        # Create LLM function for pipelines
        def create_llm_function():
            from langchain_openai import ChatOpenAI
            llm = ChatOpenAI(
                model="gpt-4o-mini",
                temperature=0,
                max_tokens=1024
            )
            return lambda prompt: llm.invoke(prompt).content

        llm_func = create_llm_function()
        
        # Create embedding function
        from langchain_openai import OpenAIEmbeddings
        embedding_func = OpenAIEmbeddings(model="text-embedding-3-small")
        
        # Create vector store (this will be passed to pipelines)
        from iris_rag.storage.vector_store_iris import IRISVectorStore
        vector_store = IRISVectorStore(connection_manager, config_manager)
        
        # Setup framework dependencies to match current pipeline constructor signatures
        framework_dependencies = {
            "connection_manager": connection_manager,
            "config_manager": config_manager,
            "llm_func": llm_func,
            "vector_store": vector_store
        }
        
        # Initialize dynamic loading services
        config_service = PipelineConfigService()
        module_loader = ModuleLoader()
        pipeline_factory = PipelineFactory(config_service, module_loader, framework_dependencies)
        pipeline_registry = PipelineRegistry(pipeline_factory)
        
        # Register all pipelines
        pipeline_registry.register_pipelines()
        
        # Get pipelines to evaluate
        if target_pipelines is None or (len(target_pipelines) == 1 and target_pipelines[0] == "ALL"):
            # Get all registered pipelines
            pipeline_names = pipeline_registry.list_pipeline_names()
            pipelines_to_evaluate = {}
            for name in pipeline_names:
                pipeline = pipeline_registry.get_pipeline(name)
                if pipeline:
                    pipelines_to_evaluate[name] = pipeline
        else:
            # Get specific pipelines
            pipelines_to_evaluate = {}
            for pipeline_name in target_pipelines:
                if pipeline_registry.is_pipeline_registered(pipeline_name):
                    pipeline = pipeline_registry.get_pipeline(pipeline_name)
                    if pipeline:
                        pipelines_to_evaluate[pipeline_name] = pipeline
                else:
                    logger.warning(f"Pipeline '{pipeline_name}' not found in registry")
            
            if not pipelines_to_evaluate:
                available_pipelines = pipeline_registry.list_pipeline_names()
                logger.warning(f"No specified target pipelines found. Available: {available_pipelines}")
        
        logger.info(f"Selected {len(pipelines_to_evaluate)} pipelines for evaluation: {list(pipelines_to_evaluate.keys())}")
        return pipelines_to_evaluate
        
    except Exception as e:
        logger.error(f"Failed to initialize pipelines: {e}")
        return {}

def execute_pipeline_evaluations(queries: List[Dict[str, Any]],
                                connection_manager: ConnectionManager,
                                config_manager: ConfigurationManager,
                                target_pipelines: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Execute evaluations on all RAG pipelines using dynamic loading.
    
    Args:
        queries: List of evaluation queries
        connection_manager: Database connection manager
        config_manager: Configuration manager
        target_pipelines: Specific pipelines to evaluate (None for all enabled)
        
    Returns:
        Dictionary mapping pipeline names to their evaluation results
    """
    # Get pipelines to evaluate using dynamic loading
    pipelines_to_evaluate = get_pipelines_to_evaluate(
        connection_manager, config_manager, target_pipelines
    )
    
    if not pipelines_to_evaluate:
        logger.warning("No pipelines available for evaluation")
        return {}
    
    all_results = {}
    
    for pipeline_name, pipeline_instance in pipelines_to_evaluate.items():
        logger.info(f"🔄 Evaluating {pipeline_name} pipeline...")
        
        try:
            # Evaluate pipeline
            results = evaluate_single_pipeline(pipeline_instance, queries)
            all_results[pipeline_name] = results
            
            logger.info(f"✅ {pipeline_name} evaluation completed")
            
        except Exception as e:
            logger.error(f"❌ Failed to evaluate {pipeline_name}: {e}")
            all_results[pipeline_name] = []
    
    return all_results


def evaluate_single_pipeline(pipeline: Any, queries: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Evaluate a single RAG pipeline with the given queries.
    
    Args:
        pipeline: RAG pipeline instance
        queries: List of evaluation queries
        
    Returns:
        List of evaluation results for each query
    """
    results = []
    
    for query_data in queries:
        query_text = query_data.get('query', query_data.get('query_text', ''))
        
        try:
            start_time = time.time()
            
            # Execute pipeline query using standardized interface
            pipeline_response = pipeline.execute(query_text)
            
            execution_time = time.time() - start_time
            
            
            # Standardize result format - prioritize retrieved_documents over contexts
            if isinstance(pipeline_response, dict):
                answer = pipeline_response.get('answer', str(pipeline_response))
                
                # PRIORITY 1: Extract contexts from retrieved_documents (reliable source)
                retrieved_documents = pipeline_response.get('retrieved_documents', [])
                context_strings = []
                
                if retrieved_documents:
                    for doc in retrieved_documents:
                        if hasattr(doc, 'content'):
                            # Document object with content attribute
                            if doc.content and doc.content.strip():
                                context_strings.append(str(doc.content))
                        elif hasattr(doc, 'page_content'):
                            # Document object with page_content attribute
                            if doc.page_content and doc.page_content.strip():
                                context_strings.append(str(doc.page_content))
                        elif isinstance(doc, dict):
                            # Dictionary format document
                            content_val = doc.get('content', doc.get('text', doc.get('page_content', '')))
                            if content_val and str(content_val).strip():
                                context_strings.append(str(content_val))
                        elif isinstance(doc, str):
                            # String content directly
                            if doc.strip():
                                context_strings.append(doc)
                
                # FALLBACK: Use contexts field only if retrieved_documents didn't provide content
                if not context_strings:
                    contexts_field = pipeline_response.get('contexts', [])
                    for ctx in contexts_field:
                        if isinstance(ctx, str) and ctx.strip():
                            context_strings.append(ctx)
                        elif hasattr(ctx, 'content') and ctx.content and ctx.content.strip():
                            context_strings.append(str(ctx.content))
                        elif hasattr(ctx, 'page_content') and ctx.page_content and ctx.page_content.strip():
                            context_strings.append(str(ctx.page_content))
            else:
                answer = str(pipeline_response)
                context_strings = []
            
            # Format for RAGAS
            result_data = {
                'question': query_text,
                'answer': answer,
                'contexts': context_strings,
                'ground_truth': query_data.get('ground_truth_answer', NOT_APPLICABLE_GROUND_TRUTH),
                'execution_time': execution_time,
                'success': True,
                'error': None
            }
            
        except Exception as e:
            logger.error(f"Error executing query '{query_text}': {e}")
            result_data = {
                'question': query_text,
                'answer': f"Error: {str(e)}",
                'contexts': [],
                'ground_truth': query_data.get('ground_truth_answer', NOT_APPLICABLE_GROUND_TRUTH),
                'execution_time': 0,
                'success': False,
                'error': str(e)
            }
        
        results.append(result_data)
    
    return results


def calculate_ragas_metrics(pipeline_results: Dict[str, List[Dict[str, Any]]],
                          ragas_llm: ChatOpenAI,
                          ragas_embeddings: OpenAIEmbeddings,
                          ragas_metrics: List) -> Dict[str, Dict[str, Any]]:
    """
    Calculate RAGAS metrics for all pipeline results.
    
    Args:
        pipeline_results: Results from all pipeline evaluations
        ragas_llm: LLM for RAGAS evaluation
        ragas_embeddings: Embeddings for RAGAS evaluation
        ragas_metrics: List of RAGAS metrics to calculate
        
    Returns:
        Dictionary mapping pipeline names to their RAGAS scores
    """
    logger.info("Starting RAGAS metrics calculation...")
    logger.info(f"Processing {len(pipeline_results)} pipelines for RAGAS evaluation")
    
    ragas_results = {}
    
    for pipeline_name, results in pipeline_results.items():
        logger.info(f"📊 Calculating RAGAS metrics for {pipeline_name}")
        
        try:
            # Filter successful results
            successful_results = [r for r in results if r['success']]
            
            if not successful_results:
                logger.warning(f"⚠️ No successful results for {pipeline_name}")
                ragas_results[pipeline_name] = {
                    'error': 'No successful results',
                    'answer_relevancy': None,
                    'context_precision': None,
                    'context_recall': None,
                    'faithfulness': None,
                    'answer_similarity': None,
                    'answer_correctness': None
                }
                continue
            
            # Prepare data for RAGAS evaluation
            questions = []
            answers = []
            contexts = []
            ground_truths = []
            
            for pipeline_item_response in successful_results:
                questions.append(pipeline_item_response.get('question', ''))
                
                # Extract and validate answer field
                raw_answer = pipeline_item_response.get('answer', '')
                processed_answer = ""
                
                if isinstance(raw_answer, str):
                    # Check if answer contains document objects or is empty list string
                    if raw_answer == "[]" or "Document(" in raw_answer:
                        logger.warning(f"Pipeline {pipeline_name}: Answer field contains document objects or empty list string: {raw_answer[:100]}...")
                        processed_answer = ""
                    else:
                        processed_answer = raw_answer
                else:
                    logger.warning(f"Pipeline {pipeline_name}: Answer field is not a string, type: {type(raw_answer)}")
                    processed_answer = ""
                
                answers.append(processed_answer)
                
                # Extract contexts from retrieved documents (prioritize retrieved_documents)
                retrieved_docs = pipeline_item_response.get('retrieved_documents', [])
                pipeline_contexts = []
                
                # First try to get contexts from the contexts field if retrieved_documents is empty
                if not retrieved_docs:
                    existing_contexts = pipeline_item_response.get('contexts', [])
                    if existing_contexts and ("Document(" in str(raw_answer)):
                        logger.warning(f"Pipeline {pipeline_name}: No retrieved_documents but answer contains document objects - this indicates a pipeline issue")
                    
                    # Process existing contexts
                    for ctx in existing_contexts:
                        if isinstance(ctx, str):
                            if ctx == "[Error Reading Streamed dict content]":
                                logger.warning(f"Pipeline {pipeline_name}: Invalid context string found: {ctx}")
                                pipeline_contexts.append("")
                            else:
                                pipeline_contexts.append(ctx)
                        else:
                            logger.warning(f"Pipeline {pipeline_name}: Unknown context type: {type(ctx)}")
                            pipeline_contexts.append("")
                else:
                    # Process retrieved documents
                    for doc in retrieved_docs:
                        if hasattr(doc, 'page_content'):
                            page_content = doc.page_content
                            if page_content == "[Error Reading Streamed dict content]" or not isinstance(page_content, str):
                                logger.warning(f"Pipeline {pipeline_name}: Invalid page_content found: {page_content}")
                                pipeline_contexts.append("")
                            else:
                                pipeline_contexts.append(page_content)
                        elif isinstance(doc, str):
                            if doc == "[Error Reading Streamed dict content]":
                                logger.warning(f"Pipeline {pipeline_name}: Invalid context string found: {doc}")
                                pipeline_contexts.append("")
                            else:
                                pipeline_contexts.append(doc)
                        else:
                            logger.warning(f"Pipeline {pipeline_name}: Unknown document type: {type(doc)}")
                            pipeline_contexts.append("")
                
                contexts.append(pipeline_contexts)
                ground_truths.append(pipeline_item_response.get('ground_truth', ''))
                
                # Log data preparation summary for debugging
                logger.debug(f"Pipeline {pipeline_name}: Prepared answer length: {len(processed_answer)}, contexts count: {len(pipeline_contexts)}")
            
            if not questions:
                logger.warning("No questions found for RAGAS evaluation")
                return {}
            
            logger.info(f"Prepared {len(questions)} questions for RAGAS evaluation")
            logger.info(f"Answer statistics: {len([a for a in answers if a])} non-empty answers out of {len(answers)} total")
            logger.info(f"Context statistics: {len([c for c in contexts if c])} non-empty context lists out of {len(contexts)} total")
            
            # Log sample data for debugging
            for i, (q, a, c) in enumerate(zip(questions[:2], answers[:2], contexts[:2])):
                logger.debug(f"Sample {i+1}: Question: {q[:50]}..., Answer: {a[:50]}..., Contexts: {len(c)} items")
            
            # Validate that we have answers for RAGAS
            if not answers or all(not answer.strip() for answer in answers):
                logger.warning(f"⚠️ No valid answers for {pipeline_name}")
                ragas_results[pipeline_name] = {
                    'error': 'No valid answers found',
                    'answer_relevancy': None,
                    'context_precision': None,
                    'context_recall': None,
                    'faithfulness': None,
                    'answer_similarity': None,
                    'answer_correctness': None
                }
                continue
            
            
            # Create RAGAS dataset
            dataset = Dataset.from_dict({
                'question': questions,
                'response': answers,  # Changed from 'answer' to 'response' for RAGAS compatibility
                'contexts': contexts,
                'ground_truth': ground_truths
            })
            
            # Run RAGAS evaluation
            logger.info(f"🔄 Running RAGAS evaluation for {pipeline_name}...")
            evaluation_result = evaluate(
                dataset=dataset,
                metrics=ragas_metrics,
                llm=ragas_llm,
                embeddings=ragas_embeddings
            )
            
            # Extract and store scores - use safe access method for EvaluationResult
            def safe_get_metric(result, metric_name):
                """Safely extract metric from EvaluationResult object and convert to scalar"""
                try:
                    value = result[metric_name]
                    # Handle case where RAGAS returns a list of values - take the mean
                    if isinstance(value, list):
                        return sum(value) / len(value) if value else None
                    return value
                except (KeyError, TypeError):
                    return None
            
            ragas_results[pipeline_name] = {
                'answer_relevancy': safe_get_metric(evaluation_result, 'answer_relevancy'),
                'context_precision': safe_get_metric(evaluation_result, 'context_precision'),
                'context_recall': safe_get_metric(evaluation_result, 'context_recall'),
                'faithfulness': safe_get_metric(evaluation_result, 'faithfulness'),
                'answer_similarity': safe_get_metric(evaluation_result, 'answer_similarity'),
                'answer_correctness': safe_get_metric(evaluation_result, 'answer_correctness')
            }
            
            logger.info(f"✅ RAGAS metrics calculated for {pipeline_name}")
            
        except Exception as e:
            logger.error(f"❌ Error calculating RAGAS metrics for {pipeline_name}: {e}")
            ragas_results[pipeline_name] = {
                'error': str(e),
                'answer_relevancy': None,
                'context_precision': None,
                'context_recall': None,
                'context_recall': None,
                'faithfulness': None,
                'answer_similarity': None,
                'answer_correctness': None
            }
            
    return ragas_results


def generate_evaluation_report(pipeline_results: Dict[str, List[Dict[str, Any]]],
                             ragas_results: Dict[str, Dict[str, Any]],
                             evaluation_duration: float) -> str:
    """
    Generate comprehensive evaluation report.
    
    Args:
        pipeline_results: Results from pipeline evaluations
        ragas_results: RAGAS metric results
        evaluation_duration: Total evaluation time
        
    Returns:
        Path to the generated report directory
    """
    # Create timestamped output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_dir = f"comprehensive_ragas_results_{timestamp}"
    os.makedirs(report_dir, exist_ok=True)
    
    # Save raw results
    raw_results = {
        'timestamp': timestamp,
        'evaluation_duration': evaluation_duration,
        'pipeline_results': pipeline_results,
        'ragas_results': ragas_results
    }
    
    raw_results_file = os.path.join(report_dir, 'raw_results.json')
    with open(raw_results_file, 'w') as f:
        json.dump(raw_results, f, indent=2, default=str)
    
    # Generate summary report
    summary_file = os.path.join(report_dir, 'evaluation_summary.md')
    with open(summary_file, 'w') as f:
        f.write(f"# Comprehensive RAGAS Evaluation Report\n")
        f.write(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"**Duration:** {evaluation_duration:.2f} seconds\n\n")
        
        f.write(f"## Pipeline Performance Summary\n\n")
        f.write(f"| Pipeline | Success Rate | Avg Time (s) |\n")
        f.write(f"|----------|--------------|-------------|\n")
        
        for pipeline_name, results in pipeline_results.items():
            if results:
                successful = [r for r in results if r['success']]
                success_rate = len(successful) / len(results)
                avg_time = sum(r['execution_time'] for r in successful) / len(successful) if successful else 0
                f.write(f"| {pipeline_name} | {success_rate:.1%} | {avg_time:.2f} |\n")
        
        f.write(f"\n## RAGAS Quality Metrics\n\n")
        f.write(f"| Pipeline | Answer Relevancy | Context Precision | Context Recall | Faithfulness | Answer Similarity | Answer Correctness |\n")
        f.write(f"|----------|------------------|-------------------|----------------|--------------|-------------------|--------------------|\n")
        
        for pipeline_name, metrics in ragas_results.items():
            if 'error' not in metrics:
                # Helper function to format metric values
                def format_metric(value):
                    import math
                    if value is None:
                        return "NaN"
                    elif isinstance(value, float) and math.isnan(value):
                        return "NaN"
                    else:
                        return f"{value:.3f}"
                
                f.write(f"| {pipeline_name} | "
                       f"{format_metric(metrics.get('answer_relevancy'))} | "
                       f"{format_metric(metrics.get('context_precision'))} | "
                       f"{format_metric(metrics.get('context_recall'))} | "
                       f"{format_metric(metrics.get('faithfulness'))} | "
                       f"{format_metric(metrics.get('answer_similarity'))} | "
                       f"{format_metric(metrics.get('answer_correctness'))} |\n")
            else:
                f.write(f"| {pipeline_name} | ERROR | ERROR | ERROR | ERROR | ERROR | ERROR |\n")
    
    logger.info(f"📁 Evaluation report generated: {report_dir}")
    return report_dir


def execute_ragas_evaluation(num_queries: Optional[int] = None,
                           target_pipelines: Optional[List[str]] = None) -> Dict[str, Any]:
    """
    Main function to execute comprehensive RAGAS evaluation.
    
    Args:
        num_queries: Number of queries to run (None for all)
        target_pipelines: Specific pipelines to target (None for all)
        
    Returns:
        Complete evaluation results
    """
    logger.info("🚀 Starting Comprehensive RAGAS Evaluation")
    start_time = time.time()
    
    # Step 1: Validate environment
    validate_openai_api_key()
    
    # Step 2: Initialize managers
    connection_manager = ConnectionManager()
    config_manager = ConfigurationManager()
    iris_connector = connection_manager.get_connection("iris")
    
    # Step 2.5: Setup IRIS cache for improved performance
    setup_iris_cache(connection_manager, config_manager)
    
    # Step 3: Validate dataset
    validate_dataset_completeness(iris_connector)
    
    # Step 4: Initialize RAGAS framework
    ragas_llm, ragas_embeddings, ragas_metrics = initialize_ragas_framework(config_manager)
    
    # Step 5: Load evaluation queries
    queries = load_evaluation_queries()
    if num_queries:
        queries = queries[:num_queries]
    
    # Step 6: Execute pipeline evaluations
    pipeline_results = execute_pipeline_evaluations(
        queries,
        connection_manager,
        config_manager,
        target_pipelines=target_pipelines
    )
    
    # Step 7: Calculate RAGAS metrics
    ragas_results = calculate_ragas_metrics(pipeline_results, ragas_llm, ragas_embeddings, ragas_metrics)
    
    # Step 8: Generate evaluation report
    evaluation_duration = time.time() - start_time
    report_dir = generate_evaluation_report(pipeline_results, ragas_results, evaluation_duration)
    
    logger.info(f"✅ Comprehensive RAGAS evaluation completed in {evaluation_duration:.2f} seconds")
    logger.info(f"📁 Results saved to: {report_dir}")
    
    return {
        'pipeline_results': pipeline_results,
        'ragas_results': ragas_results,
        'evaluation_duration': evaluation_duration,
        'report_directory': report_dir
    }


def main():
    """Main execution function with argument parsing."""
    parser = argparse.ArgumentParser(description='Execute comprehensive RAGAS evaluation')
    parser.add_argument('--num-queries', type=int, help='Number of queries to run (default: all)')
    parser.add_argument('--pipelines', nargs='+',
                       help='Specific pipelines to evaluate (use "ALL" for all enabled pipelines, or specify names from config/pipelines.yaml)')
    
    args = parser.parse_args()
    
    try:
        results = execute_ragas_evaluation(
            num_queries=args.num_queries,
            target_pipelines=args.pipelines
        )
        
        print("\n" + "="*80)
        print("🎉 COMPREHENSIVE RAGAS EVALUATION COMPLETED!")
        print("="*80)
        print(f"📁 Results directory: {results['report_directory']}")
        print(f"⏱️  Total duration: {results['evaluation_duration']:.2f} seconds")
        print("="*80)
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()