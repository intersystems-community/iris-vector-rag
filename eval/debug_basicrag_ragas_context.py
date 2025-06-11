#!/usr/bin/env python3
"""
RAGAS Context Debug Test Harness

A reusable test harness for debugging RAGAS context handling in RAG pipelines.
This follows TDD principles and can be used to verify context extraction and
RAGAS metric calculation for any pipeline.

Usage:
    python eval/debug_basicrag_ragas_context.py --pipeline BasicRAG --queries 3
    python eval/debug_basicrag_ragas_context.py --pipeline HyDE --queries 5
    python eval/debug_basicrag_ragas_context.py --help
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path

# Load environment variables
from dotenv import load_dotenv
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# Import RAGAS components
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision, 
    context_recall,
    faithfulness
)
from datasets import Dataset
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

# Import framework components
from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager
from iris_rag.pipelines.factory import PipelineFactory
from iris_rag.pipelines.registry import PipelineRegistry
from iris_rag.config.pipeline_config_service import PipelineConfigService
from iris_rag.utils.module_loader import ModuleLoader
from iris_rag.storage.vector_store_iris import IRISVectorStore

# Import cache management
from common.llm_cache_manager import setup_langchain_cache
from common.llm_cache_config import load_cache_config

logger = logging.getLogger(__name__)


class RAGASContextDebugHarness:
    """
    Reusable test harness for debugging RAGAS context handling in RAG pipelines.
    
    This class provides a standardized way to:
    1. Initialize any RAG pipeline
    2. Execute it with test queries
    3. Verify context extraction and execution_time handling
    4. Calculate RAGAS metrics
    5. Provide detailed debugging output
    
    The harness expects pipelines to return results with:
    - 'contexts': List of strings (actual document content for RAGAS)
    - 'execution_time': Float (pipeline execution time)
    - 'answer': String (generated answer)
    - 'query': String (original query)
    """
    
    def __init__(self, config_path: Optional[str] = None):
        """
        Initialize the debug harness.
        
        Args:
            config_path: Optional path to configuration file
        """
        self.setup_logging()
        self._setup_ragas_debug_environment()
        self.config_manager = ConfigurationManager(config_path)
        self.connection_manager = ConnectionManager(self.config_manager)
        
        # Initialize pipeline configuration service and module loader
        self.config_service = PipelineConfigService()
        self.module_loader = ModuleLoader()
        
        # Create framework dependencies for pipeline factory
        self.framework_dependencies = self._create_framework_dependencies()
        
        # Initialize pipeline factory with correct signature
        self.pipeline_factory = PipelineFactory(
            self.config_service,
            self.module_loader,
            self.framework_dependencies
        )
        
        # Initialize RAGAS components
        self.ragas_llm = None
        self.ragas_embeddings = None
        self.ragas_metrics = None
        
    def setup_logging(self):
        """Set up logging configuration with enhanced debugging."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        
        # Set up enhanced logging for debugging RAGAS issues
        # This will be further enhanced when RAGAS evaluation starts
        logger.info("Enhanced logging configured for RAGAS debugging")
    
    def _setup_ragas_debug_environment(self):
        """Set up environment variables for RAGAS debugging early."""
        # Set environment variables that might help with RAGAS debugging
        os.environ['RAGAS_LOGGING_LEVEL'] = 'DEBUG'
        os.environ['RAGAS_DEBUG'] = '1'
        
        # Set OpenAI debugging if available
        os.environ['OPENAI_LOG'] = 'debug'
        
        logger.info("RAGAS debug environment variables configured")
    
    def _create_framework_dependencies(self) -> Dict[str, Any]:
        """
        Create framework dependencies dictionary for pipeline factory.
        
        Returns:
            Dictionary containing framework dependencies
        """
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
        
        # Create vector store
        vector_store = IRISVectorStore(self.connection_manager, self.config_manager)
        
        # Return framework dependencies dictionary
        return {
            "connection_manager": self.connection_manager,
            "config_manager": self.config_manager,
            "llm_func": llm_func,
            "embedding_func": embedding_func,
            "vector_store": vector_store
        }
        
    def initialize_ragas_framework(self) -> Tuple[Any, Any, List[Any]]:
        """
        Initialize RAGAS framework components.
        
        Returns:
            Tuple of (ragas_llm, ragas_embeddings, ragas_metrics)
        """
        logger.info("Initializing RAGAS framework...")
        
        # Initialize LLM for RAGAS evaluation
        self.ragas_llm = ChatOpenAI(
            model="gpt-3.5-turbo",
            temperature=0,
            timeout=60
        )
        
        # Initialize embeddings for RAGAS evaluation
        self.ragas_embeddings = OpenAIEmbeddings(
            model="text-embedding-ada-002"
        )
        
        # Define RAGAS metrics to evaluate
        self.ragas_metrics = [
            context_precision,
            context_recall,
            faithfulness,
            answer_relevancy
        ]
        
        return self.ragas_llm, self.ragas_embeddings, self.ragas_metrics
    
    def load_test_queries(self, num_queries: int = 3) -> List[Dict[str, str]]:
        """
        Load test queries for evaluation.
        
        Args:
            num_queries: Number of queries to load for testing
            
        Returns:
            List of query dictionaries
        """
        queries_file = project_root / "eval" / "sample_queries.json"
        
        if not queries_file.exists():
            # Create sample queries if file doesn't exist
            sample_queries = [
                {
                    "query": "What are the main causes of diabetes?",
                    "expected_answer": "The main causes of diabetes include genetic factors, lifestyle factors, and autoimmune responses.",
                    "category": "medical"
                },
                {
                    "query": "How does machine learning work?",
                    "expected_answer": "Machine learning works by training algorithms on data to make predictions or decisions.",
                    "category": "technology"
                },
                {
                    "query": "What is the role of mitochondria in cells?",
                    "expected_answer": "Mitochondria are the powerhouses of cells, producing ATP through cellular respiration.",
                    "category": "biology"
                }
            ]
            
            with open(queries_file, 'w') as f:
                json.dump(sample_queries, f, indent=2)
            
            logger.info(f"Created sample queries file at {queries_file}")
        
        with open(queries_file, 'r') as f:
            all_queries = json.load(f)
        
        # Return the requested number of queries
        return all_queries[:num_queries]
    
    def get_pipeline(self, pipeline_name: str):
        """
        Get a pipeline instance by name.
        
        Args:
            pipeline_name: Name of the pipeline to instantiate
            
        Returns:
            Pipeline instance
        """
        logger.info(f"Instantiating {pipeline_name} pipeline...")
        
        # Get pipeline from factory
        pipeline = self.pipeline_factory.create_pipeline(pipeline_name)
        
        if pipeline is None:
            raise ValueError(f"Pipeline '{pipeline_name}' not found or failed to instantiate")
        
        return pipeline
    
    def execute_pipeline_with_debug(self, pipeline, queries: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        """
        Execute pipeline with detailed debugging information.
        
        Args:
            pipeline: Pipeline instance to execute
            queries: List of query dictionaries
            
        Returns:
            List of execution results with debug information
        """
        results = []
        
        for i, query_data in enumerate(queries):
            query = query_data['query']
            expected_answer = query_data.get('expected_answer', '')
            
            logger.info(f"Processing query {i+1}/{len(queries)}: {query}")
            
            try:
                # Execute pipeline
                start_time = time.time()
                result = pipeline.execute(query)
                fallback_execution_time = time.time() - start_time
                
                # Validate pipeline response format
                result = self._validate_pipeline_response(result, query)
                
                # Extract contexts - handle different result formats
                contexts = self._extract_contexts(result)
                
                # Get execution time from pipeline result or use fallback
                pipeline_execution_time = result.get('execution_time', fallback_execution_time)
                if pipeline_execution_time is None:
                    logger.warning(f"Pipeline returned None for execution_time, using fallback: {fallback_execution_time:.3f}s")
                    pipeline_execution_time = fallback_execution_time
                
                # Create evaluation result
                eval_result = {
                    'query': query,
                    'answer': result.get('answer', ''),
                    'contexts': contexts,
                    'ground_truth': expected_answer,
                    'execution_time': pipeline_execution_time,
                    'debug_info': {
                        'raw_result_keys': list(result.keys()),
                        'contexts_count': len(contexts),
                        'contexts_total_length': sum(len(ctx) for ctx in contexts),
                        'answer_length': len(result.get('answer', '')),
                        'pipeline_execution_time': pipeline_execution_time,
                        'fallback_execution_time': fallback_execution_time
                    }
                }
                
                results.append(eval_result)
                
                # Log debug information
                self._log_debug_info(i+1, eval_result)
                
            except Exception as e:
                logger.error(f"Error processing query {i+1}: {e}")
                # Add failed result for completeness
                results.append({
                    'query': query,
                    'answer': '',
                    'contexts': [],
                    'ground_truth': expected_answer,
                    'execution_time': 0.0,
                    'error': str(e)
                })
        
        return results
    
    def _extract_contexts(self, result: Dict[str, Any]) -> List[str]:
        """
        Extract contexts from pipeline result, handling different formats.
        
        Args:
            result: Pipeline execution result
            
        Returns:
            List of context strings
        """
        # First, try the 'contexts' field directly (preferred for RAGAS)
        if 'contexts' in result:
            contexts_data = result['contexts']
            if isinstance(contexts_data, list):
                # Validate that all items are strings
                contexts = []
                for item in contexts_data:
                    if isinstance(item, str):
                        contexts.append(item)
                    else:
                        logger.warning(f"Non-string context found in 'contexts' field: {type(item)}")
                        contexts.append(str(item))
                logger.debug(f"Extracted {len(contexts)} contexts from 'contexts' field")
                return contexts
            elif isinstance(contexts_data, str):
                return [contexts_data]
        
        # Fallback: try other possible context keys for backward compatibility
        fallback_keys = ['retrieved_documents', 'documents', 'chunks']
        
        for key in fallback_keys:
            if key in result:
                contexts_data = result[key]
                logger.info(f"Using fallback context extraction from '{key}' field")
                
                # Handle different context formats
                if isinstance(contexts_data, list):
                    contexts = []
                    for item in contexts_data:
                        if isinstance(item, str):
                            contexts.append(item)
                        elif isinstance(item, dict):
                            # Try common text keys for Document objects
                            text_keys = ['page_content', 'content', 'text', 'chunk_text']
                            for text_key in text_keys:
                                if text_key in item:
                                    contexts.append(item[text_key])
                                    break
                            else:
                                # Fallback to string representation
                                logger.warning(f"No recognized text field in document object: {list(item.keys())}")
                                contexts.append(str(item))
                        elif hasattr(item, 'page_content'):
                            # Handle Document objects with page_content attribute
                            contexts.append(item.page_content)
                        else:
                            contexts.append(str(item))
                    return contexts
                elif isinstance(contexts_data, str):
                    return [contexts_data]
        
        logger.warning(f"No contexts found in result keys: {list(result.keys())}")
        return []
    
    def _validate_pipeline_response(self, result: Dict[str, Any], query: str) -> Dict[str, Any]:
        """
        Validate and normalize pipeline response format.
        
        Args:
            result: Raw pipeline response
            query: Original query for context
            
        Returns:
            Validated and normalized response
        """
        validation_issues = []
        
        # Check required fields
        required_fields = ['answer', 'contexts', 'execution_time']
        for field in required_fields:
            if field not in result:
                validation_issues.append(f"Missing required field: '{field}'")
        
        # Validate contexts field specifically
        if 'contexts' in result:
            contexts = result['contexts']
            if not isinstance(contexts, list):
                validation_issues.append(f"'contexts' field must be a list, got {type(contexts)}")
            elif contexts:
                non_string_contexts = [i for i, ctx in enumerate(contexts) if not isinstance(ctx, str)]
                if non_string_contexts:
                    validation_issues.append(f"'contexts' must contain only strings, found non-strings at indices: {non_string_contexts}")
        
        # Validate execution_time field
        if 'execution_time' in result:
            exec_time = result['execution_time']
            if not isinstance(exec_time, (int, float)):
                validation_issues.append(f"'execution_time' must be numeric, got {type(exec_time)}")
            elif exec_time < 0:
                validation_issues.append(f"'execution_time' must be non-negative, got {exec_time}")
        
        # Log validation issues
        if validation_issues:
            logger.warning(f"Pipeline response validation issues for query '{query[:50]}...': {validation_issues}")
        else:
            logger.debug(f"Pipeline response validation passed for query '{query[:50]}...'")
        
        return result

    def _log_debug_info(self, query_num: int, eval_result: Dict[str, Any]):
        """Log detailed debug information for a query result."""
        debug_info = eval_result['debug_info']
        
        logger.info(f"Query {query_num} Debug Info:")
        logger.info(f"  Pipeline execution time: {debug_info['pipeline_execution_time']:.3f}s")
        logger.info(f"  Fallback execution time: {debug_info['fallback_execution_time']:.3f}s")
        logger.info(f"  Result keys: {debug_info['raw_result_keys']}")
        logger.info(f"  Contexts count: {debug_info['contexts_count']}")
        logger.info(f"  Total context length: {debug_info['contexts_total_length']} chars")
        logger.info(f"  Answer length: {debug_info['answer_length']} chars")
        
        # Show sample context if available
        if eval_result['contexts']:
            sample_context = eval_result['contexts'][0][:200]
            logger.info(f"  Sample context: {sample_context}...")
        else:
            logger.warning(f"  No contexts available for query {query_num}")
    
    def _log_ragas_input_dataset(self, dataset_dict: Dict[str, List[Any]]) -> None:
        """
        Log the RAGAS input dataset in detail for debugging.
        
        Args:
            dataset_dict: Dictionary containing the dataset to be passed to RAGAS
        """
        logger.info("="*80)
        logger.info("RAGAS INPUT DATASET DETAILED LOG")
        logger.info("="*80)
        
        # Log dataset structure
        logger.info(f"Dataset structure:")
        for key, values in dataset_dict.items():
            logger.info(f"  {key}: {len(values)} items (type: {type(values)})")
        
        # Log each item in detail
        num_items = len(dataset_dict.get('question', []))
        logger.info(f"\nDataset contains {num_items} items:")
        
        for i in range(num_items):
            logger.info(f"\n--- ITEM {i+1} ---")
            
            # Log question
            question = dataset_dict['question'][i] if i < len(dataset_dict.get('question', [])) else 'N/A'
            logger.info(f"Question: {question}")
            
            # Log answer
            answer = dataset_dict['answer'][i] if i < len(dataset_dict.get('answer', [])) else 'N/A'
            logger.info(f"Answer: {answer[:200]}{'...' if len(str(answer)) > 200 else ''}")
            
            # Log ground truth
            ground_truth = dataset_dict['ground_truth'][i] if i < len(dataset_dict.get('ground_truth', [])) else 'N/A'
            logger.info(f"Ground Truth: {ground_truth[:200]}{'...' if len(str(ground_truth)) > 200 else ''}")
            
            # Log contexts in detail
            contexts = dataset_dict['contexts'][i] if i < len(dataset_dict.get('contexts', [])) else []
            logger.info(f"Contexts: {len(contexts)} items")
            for j, context in enumerate(contexts):
                logger.info(f"  Context {j+1}: {context[:150]}{'...' if len(str(context)) > 150 else ''}")
                logger.info(f"    Length: {len(str(context))} chars")
                logger.info(f"    Type: {type(context)}")
        
        logger.info("="*80)
        logger.info("END RAGAS INPUT DATASET LOG")
        logger.info("="*80)

    def _enable_verbose_ragas_logging(self) -> None:
        """
        Enable verbose logging for RAGAS to get more detailed error information.
        """
        # Set RAGAS logging level to DEBUG
        ragas_logger = logging.getLogger('ragas')
        ragas_logger.setLevel(logging.DEBUG)
        
        # Create a detailed handler for RAGAS logs if not already present
        if not any(isinstance(h, logging.StreamHandler) for h in ragas_logger.handlers):
            handler = logging.StreamHandler()
            handler.setLevel(logging.DEBUG)
            formatter = logging.Formatter(
                '%(asctime)s - RAGAS-%(name)s - %(levelname)s - %(message)s'
            )
            handler.setFormatter(formatter)
            ragas_logger.addHandler(handler)
        
        # Also enable debug logging for specific RAGAS components
        for component in ['ragas.metrics', 'ragas.metrics._context_recall', 'ragas.metrics._context_precision']:
            comp_logger = logging.getLogger(component)
            comp_logger.setLevel(logging.DEBUG)
        
        # Set environment variable for RAGAS debugging if supported
        os.environ['RAGAS_LOGGING_LEVEL'] = 'DEBUG'
        os.environ['RAGAS_DEBUG'] = '1'
        
        logger.info("Enabled verbose RAGAS logging (DEBUG level)")

    def _calculate_ragas_metrics(self, ragas_result) -> Dict[str, float]:
        """
        Safely extract RAGAS metric scores from EvaluationResult.
        
        Args:
            ragas_result: RAGAS EvaluationResult object
            
        Returns:
            Dictionary of metric scores with None for failed metrics
        """
        # Define expected RAGAS metric names
        expected_metrics = [
            'context_precision',
            'context_recall',
            'faithfulness',
            'answer_relevancy',
            'answer_correctness',
            'answer_similarity'
        ]
        
        scores = {}
        
        logger.info("Extracting RAGAS metric scores...")
        logger.info(f"RAGAS result type: {type(ragas_result)}")
        
        # Log available attributes/keys for debugging
        if hasattr(ragas_result, 'keys'):
            available_keys = list(ragas_result.keys())
            logger.info(f"Available keys in RAGAS result: {available_keys}")
        elif hasattr(ragas_result, '__dict__'):
            available_attrs = list(ragas_result.__dict__.keys())
            logger.info(f"Available attributes in RAGAS result: {available_attrs}")
        
        # Try to extract each metric score safely
        for metric_name in expected_metrics:
            try:
                # Try different ways to access the metric score
                score = None
                
                # Method 1: Try dictionary-style access
                if hasattr(ragas_result, 'keys') and metric_name in ragas_result:
                    score = ragas_result[metric_name]
                    logger.info(f"RAGAS metric '{metric_name}': {score} (dict access)")
                
                # Method 2: Try attribute access
                elif hasattr(ragas_result, metric_name):
                    score = getattr(ragas_result, metric_name)
                    logger.info(f"RAGAS metric '{metric_name}': {score} (attr access)")
                
                # Method 3: Try to_pandas() and extract from DataFrame
                elif hasattr(ragas_result, 'to_pandas'):
                    try:
                        df = ragas_result.to_pandas()
                        if metric_name in df.columns:
                            # Get mean score if multiple rows
                            score = df[metric_name].mean() if len(df) > 1 else df[metric_name].iloc[0]
                            logger.info(f"RAGAS metric '{metric_name}': {score} (pandas access)")
                    except Exception as pandas_e:
                        logger.warning(f"Failed to extract '{metric_name}' via pandas: {pandas_e}")
                
                # Validate the score
                if score is not None:
                    # Check for NaN or invalid values
                    import math
                    if isinstance(score, (int, float)) and not math.isnan(score):
                        scores[metric_name] = float(score)
                    else:
                        logger.warning(f"RAGAS metric '{metric_name}': Invalid score (NaN or None)")
                        scores[metric_name] = None
                else:
                    logger.warning(f"RAGAS metric '{metric_name}': Score not available")
                    scores[metric_name] = None
                    
            except KeyError as ke:
                logger.warning(f"RAGAS metric '{metric_name}': KeyError - {ke}")
                scores[metric_name] = None
            except Exception as e:
                logger.error(f"RAGAS metric '{metric_name}': Unexpected error - {e}")
                scores[metric_name] = None
        
        # Log summary of extracted scores
        successful_metrics = [k for k, v in scores.items() if v is not None]
        failed_metrics = [k for k, v in scores.items() if v is None]
        
        logger.info(f"Successfully extracted {len(successful_metrics)} RAGAS metrics: {successful_metrics}")
        if failed_metrics:
            logger.warning(f"Failed to extract {len(failed_metrics)} RAGAS metrics: {failed_metrics}")
        
        return scores

    def calculate_ragas_metrics(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """
        Calculate RAGAS metrics for the results with enhanced debugging.
        
        Args:
            results: List of evaluation results
            
        Returns:
            Dictionary of RAGAS scores
        """
        if not results or not self.ragas_llm:
            logger.error("No results or RAGAS not initialized")
            return {}
        
        # Filter out failed results
        valid_results = [r for r in results if 'error' not in r and r['contexts']]
        
        if not valid_results:
            logger.error("No valid results with contexts for RAGAS evaluation")
            return {}
        
        logger.info(f"Calculating RAGAS metrics for {len(valid_results)} valid results...")
        
        # Enable verbose RAGAS logging
        self._enable_verbose_ragas_logging()
        
        try:
            # Create RAGAS dataset dictionary
            dataset_dict = {
                'question': [r['query'] for r in valid_results],
                'answer': [r['answer'] for r in valid_results],
                'contexts': [r['contexts'] for r in valid_results],
                'ground_truth': [r['ground_truth'] for r in valid_results]
            }
            
            # Log the dataset in detail BEFORE passing to RAGAS
            logger.info("Logging RAGAS input dataset before evaluation...")
            self._log_ragas_input_dataset(dataset_dict)
            
            # Create RAGAS Dataset object
            dataset = Dataset.from_dict(dataset_dict)
            
            logger.info("Created RAGAS Dataset object successfully")
            logger.info(f"Dataset features: {dataset.features}")
            logger.info(f"Dataset num_rows: {dataset.num_rows}")
            
            # Log additional debugging info before evaluation
            logger.info("About to call ragas.evaluate with:")
            logger.info(f"  Metrics: {[m.__name__ if hasattr(m, '__name__') else str(m) for m in self.ragas_metrics]}")
            logger.info(f"  LLM: {type(self.ragas_llm).__name__}")
            logger.info(f"  Embeddings: {type(self.ragas_embeddings).__name__}")
            
            # Evaluate with RAGAS
            logger.info("Starting RAGAS evaluation...")
            ragas_result = evaluate(
                dataset=dataset,
                metrics=self.ragas_metrics,
                llm=self.ragas_llm,
                embeddings=self.ragas_embeddings
            )
            
            logger.info("RAGAS evaluation completed successfully")
            logger.info(f"RAGAS result type: {type(ragas_result)}")
            logger.info(f"RAGAS result keys: {list(ragas_result.keys()) if hasattr(ragas_result, 'keys') else 'N/A'}")
            
            # Use the safe extraction method instead of dict(ragas_result)
            return self._calculate_ragas_metrics(ragas_result)
            
        except Exception as e:
            logger.error(f"Error calculating RAGAS metrics: {e}")
            logger.error(f"Exception type: {type(e).__name__}")
            logger.error(f"Exception args: {e.args}")
            
            # Log additional context for debugging
            import traceback
            logger.error(f"Full traceback:\n{traceback.format_exc()}")
            
            return {}
    
    def run_debug_session(self, pipeline_name: str, num_queries: int = 3) -> Dict[str, Any]:
        """
        Run a complete debug session for a pipeline.
        
        Args:
            pipeline_name: Name of the pipeline to debug
            num_queries: Number of test queries to use
            
        Returns:
            Complete debug session results
        """
        logger.info(f"Starting RAGAS context debug session for {pipeline_name}")
        
        # Initialize RAGAS
        self.initialize_ragas_framework()
        
        # Load test queries
        queries = self.load_test_queries(num_queries)
        
        # Get pipeline
        pipeline = self.get_pipeline(pipeline_name)
        
        # Execute pipeline with debug info
        results = self.execute_pipeline_with_debug(pipeline, queries)
        
        # Calculate RAGAS metrics
        ragas_scores = self.calculate_ragas_metrics(results)
        
        # Compile debug session results
        session_results = {
            'pipeline_name': pipeline_name,
            'timestamp': datetime.now().isoformat(),
            'num_queries': len(queries),
            'successful_executions': len([r for r in results if 'error' not in r]),
            'results_with_contexts': len([r for r in results if r.get('contexts')]),
            'execution_results': results,
            'ragas_scores': ragas_scores
        }
        
        # Print summary
        self._print_debug_summary(session_results)
        
        return session_results
    
    def _print_debug_summary(self, session_results: Dict[str, Any]):
        """Print a comprehensive debug summary."""
        print("\n" + "="*60)
        print(f"RAGAS CONTEXT DEBUG SUMMARY - {session_results['pipeline_name']}")
        print("="*60)
        
        print(f"Timestamp: {session_results['timestamp']}")
        print(f"Queries processed: {session_results['num_queries']}")
        print(f"Successful executions: {session_results['successful_executions']}")
        print(f"Results with contexts: {session_results['results_with_contexts']}")
        
        # RAGAS scores with safe handling of None values
        ragas_scores = session_results['ragas_scores']
        if ragas_scores:
            print(f"\nRAGAS Scores:")
            
            # Helper function to format score safely
            def format_score(score):
                if score is None:
                    return "N/A"
                elif isinstance(score, (int, float)):
                    return f"{score:.4f}"
                else:
                    return str(score)
            
            print(f"  Context Precision: {format_score(ragas_scores.get('context_precision'))}")
            print(f"  Context Recall: {format_score(ragas_scores.get('context_recall'))}")
            print(f"  Faithfulness: {format_score(ragas_scores.get('faithfulness'))}")
            print(f"  Answer Relevancy: {format_score(ragas_scores.get('answer_relevancy'))}")
            
            # Show additional metrics if available
            if 'answer_correctness' in ragas_scores:
                print(f"  Answer Correctness: {format_score(ragas_scores.get('answer_correctness'))}")
            if 'answer_similarity' in ragas_scores:
                print(f"  Answer Similarity: {format_score(ragas_scores.get('answer_similarity'))}")
            
            # Summary of metric status
            successful_metrics = [k for k, v in ragas_scores.items() if v is not None]
            failed_metrics = [k for k, v in ragas_scores.items() if v is None]
            
            print(f"\nMetric Status:")
            print(f"  Successful: {len(successful_metrics)} metrics")
            print(f"  Failed: {len(failed_metrics)} metrics")
            if failed_metrics:
                print(f"  Failed metrics: {', '.join(failed_metrics)}")
        else:
            print(f"\nRAGAS Scores: No valid results for evaluation")
        
        # Context analysis
        print(f"\nContext Analysis:")
        for i, result in enumerate(session_results['execution_results'][:2]):  # Show first 2
            if 'error' not in result:
                print(f"  Query {i+1}: {result['query'][:50]}...")
                print(f"    Contexts: {len(result['contexts'])}")
                print(f"    Answer length: {len(result['answer'])} chars")
                if result['contexts']:
                    print(f"    Sample context: {result['contexts'][0][:100]}...")
        
        print("="*60)


def main():
    """Main entry point for the debug harness."""
    parser = argparse.ArgumentParser(description="RAGAS Context Debug Test Harness")
    parser.add_argument("--pipeline", default="BasicRAG", help="Pipeline name to debug")
    parser.add_argument("--queries", type=int, default=3, help="Number of test queries")
    parser.add_argument("--config", help="Path to configuration file")
    parser.add_argument("--output", help="Path to save debug results (JSON)")
    
    args = parser.parse_args()
    
    # Create and run debug harness
    harness = RAGASContextDebugHarness(args.config)
    results = harness.run_debug_session(args.pipeline, args.queries)
    
    # Save results if requested
    if args.output:
        with open(args.output, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        print(f"\nResults saved to: {args.output}")


if __name__ == "__main__":
    main()