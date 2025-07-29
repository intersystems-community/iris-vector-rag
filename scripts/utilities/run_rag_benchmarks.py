#!/usr/bin/env python3
"""
Comprehensive RAG Benchmarking Script

This script runs benchmarks for multiple RAG techniques against real PMC data,
measures performance metrics, and generates detailed reports with visualizations.

Usage:
    python scripts/run_rag_benchmarks.py --techniques basic_rag hyde crag colbert noderag graphrag
                                        --dataset medical
                                        --num-docs 1000
                                        --num-queries 10
                                        --output-dir benchmark_results/my_benchmark
"""

import os
import sys
import json
import time
import logging
import argparse
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Add the parent directory to the Python path to allow importing from common, eval, etc.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

# Try to import dotenv, but make it optional
try:
    from dotenv import load_dotenv
    # Load environment variables from .env file if present
    load_dotenv()
except ImportError:
    # Define a no-op function if dotenv is not available
    def load_dotenv():
        print("python-dotenv not installed. Environment variables from .env will not be loaded.")

# Try to import numpy, but provide fallback if not available
try:
    import numpy as np
except ImportError:
    logger = logging.getLogger("rag_benchmarks")
    logger.warning("numpy not installed. Some functionality may be limited.")
    # Define a minimal numpy-like percentile function for latency calculations
    class NumpyFallback:
        @staticmethod
        def percentile(data, percentile):
            if not data:
                return 0
            sorted_data = sorted(data)
            index = int(len(sorted_data) * percentile / 100)
            return sorted_data[min(index, len(sorted_data) - 1)]
    np = NumpyFallback()

# Call load_dotenv (either the real one or our no-op version)
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("rag_benchmarks")

# Import IRIS connector and utility functions
try:
    from common.iris_dbapi_connector import get_iris_dbapi_connection
    import common.embedding_utils as embedding_utils_module
    from common.db_init import initialize_database
except ImportError as e:
    logger = logging.getLogger("rag_benchmarks")
    logger.warning(f"Error importing from common module: {e}")
    logger.warning("Defining mock implementations for required functions")
    
    # Mock implementations
    def get_iris_connection(use_mock=True, use_testcontainer=False):
        logger.warning("Using mock IRIS connection due to import error")
        return None
        
    # Mock for common.embedding_utils module if it failed to import
    class MockEmbeddingUtils:
        def get_embedding_func(self, provider="stub"):
            logger.warning("Using stub embedding_utils.get_embedding_func due to import error")
            def stub_embedding_func(text):
                return [0.0] * 384  # Return a vector of zeros
            return stub_embedding_func
    embedding_utils_module = MockEmbeddingUtils() # Assign to the alias used in the try block
        
    def initialize_database(conn, force_recreate=False):
        logger.warning("Mock database initialization (no-op)")
        return True

# Import evaluation modules
try:
    from scripts.utilities.evaluation.bench_runner import run_all_techniques_benchmark, load_benchmark_results
    from scripts.utilities.evaluation.comparative import generate_combined_report
    from scripts.utilities.evaluation.metrics import (
        calculate_context_recall,
        calculate_precision_at_k,
        calculate_answer_faithfulness,
        calculate_answer_relevance,
        calculate_throughput
    )

    # Import or define calculate_latency_percentiles based on numpy availability
    try:
        from scripts.utilities.evaluation.metrics import calculate_latency_percentiles
    except (ImportError, AttributeError):
        # Define a fallback if the imported function requires numpy and it's not available
        def calculate_latency_percentiles(latencies: List[float]) -> Dict[str, float]:
            """
            Calculate P50, P95, P99 latency percentiles.
            
            Args:
                latencies: List of latency measurements in milliseconds
                
            Returns:
                Dictionary with keys 'p50', 'p95', 'p99' and their values
            """
            if not latencies:
                return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
            
            sorted_latencies = sorted(latencies)
            
            # Calculate percentiles using our numpy fallback
            p50 = np.percentile(sorted_latencies, 50)
            p95 = np.percentile(sorted_latencies, 95)
            p99 = np.percentile(sorted_latencies, 99)
            
            return {
                "p50": float(p50),
                "p95": float(p95),
                "p99": float(p99)
            }
except ImportError as e:
    logger = logging.getLogger("rag_benchmarks")
    logger.warning(f"Error importing evaluation modules: {e}")
    logger.warning("Defining mock implementations for evaluation functions")
    
    # Mock implementations for evaluation functions
    def run_all_techniques_benchmark(queries, techniques, output_path=None):
        logger.warning("Using mock benchmark runner due to import error")
        return {tech: {"metrics": {"mock": True}} for tech in techniques}
        
    def load_benchmark_results(input_path):
        logger.warning("Using mock benchmark loader due to import error")
        return {}
        
    def generate_combined_report(benchmarks, output_dir=None, dataset_name="medical"):
        logger.warning("Using mock report generator due to import error")
        return {"json": "mock_results.json", "markdown": "mock_report.md", "charts": []}
        
    def calculate_context_recall(results, queries):
        return 0.5  # Mock value
        
    def calculate_precision_at_k(results, queries, k=5):
        return 0.5  # Mock value
        
    def calculate_answer_faithfulness(results, queries):
        return 0.5  # Mock value
        
    def calculate_answer_relevance(results, queries):
        return 0.5  # Mock value
        
    def calculate_throughput(num_queries, total_time_sec):
        return num_queries / max(1, total_time_sec)
        
    def calculate_latency_percentiles(latencies):
        if not latencies:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0}
        return {"p50": sum(latencies)/len(latencies), "p95": max(latencies), "p99": max(latencies)}

# Import pipeline classes
try:
    from iris_rag.pipelines import (
        BasicRAGPipeline,
        HyDERAGPipeline,
        ColBERTRAGPipeline,
        CRAGPipeline,
        NodeRAGPipeline,
        GraphRAGPipeline,
        create_colbert_semantic_encoder # Added for ColBERT wrapper
    )
except ImportError as e:
    logger = logging.getLogger("rag_benchmarks")
    logger.warning(f"Error importing pipeline classes: {e}")
    logger.warning("Using mock pipeline classes")
    
    # Define mock pipeline classes
    class MockPipeline:
        def __init__(self, iris_connector=None, embedding_func=None, llm_func=None):
            self.iris_connector = iris_connector
            self.embedding_func = embedding_func
            self.llm_func = llm_func
            
        def run(self, query, **kwargs):
            return {
                "query": query,
                "answer": f"Mock answer for: {query}",
                "retrieved_documents": []
            }
    
    # Use the same mock class for all pipelines
    BasicRAGPipeline = MockPipeline
    HyDERAGPipeline = MockPipeline
    ColBERTRAGPipeline = MockPipeline
    CRAGPipeline = MockPipeline
    NodeRAGPipeline = MockPipeline
    GraphRAGPipeline = MockPipeline

# Constants
MIN_DOCUMENT_COUNT = 1000
DEFAULT_TOP_K = 5
DEFAULT_QUERY_LIMIT = 10
DEFAULT_DATASET = "medical"
DEFAULT_LLM = "stub"
DEFAULT_TECHNIQUES = ["basic_rag", "hyde", "crag", "colbert", "noderag", "graphrag"]


def load_queries(dataset_type: str = DEFAULT_DATASET, query_limit: int = DEFAULT_QUERY_LIMIT) -> List[Dict[str, Any]]:
    """
    Load queries from sample_queries.json or create queries based on the specified dataset type.
    
    Args:
        dataset_type: Type of dataset queries to create (medical, multihop, etc.)
        query_limit: Maximum number of queries to return
        
    Returns:
        List of query dictionaries
    """
    try:
        with open('eval/sample_queries.json', 'r') as f:
            queries = json.load(f)
            logger.info(f"Loaded {len(queries)} queries from sample_queries.json")
            return queries[:query_limit]
    except (FileNotFoundError, json.JSONDecodeError):
        logger.warning("sample_queries.json not found or invalid, creating default queries")
        
        # Create dataset-specific queries
        if dataset_type == "multihop":
            # MultiHopQA inspired queries requiring multi-step reasoning
            queries = [
                {"query": "What symptoms can result from the mechanism that allows beta blockers to treat hypertension?"},
                {"query": "Which imaging techniques can detect the abnormalities caused by the gene mutation responsible for cystic fibrosis?"},
                {"query": "What immune cells are activated by the same pathway that's targeted by TNF inhibitors?"},
                {"query": "What side effects might be expected when using drugs that inhibit the enzymes responsible for serotonin metabolism?"},
                {"query": "What proteins are involved in both Alzheimer's disease pathology and regulation of calcium homeostasis?"}
            ]
        else:
            # Default medical queries about PMC articles
            queries = [
                {"query": "What are the mechanisms of cancer immunotherapy?"},
                {"query": "How effective are mRNA vaccines?"},
                {"query": "What is the relationship between diet and cardiovascular disease?"},
                {"query": "What are biomarkers for early detection of Alzheimer's disease?"},
                {"query": "How does the gut microbiome affect immune response?"}
            ]
            
        # Save for future use
        os.makedirs('eval', exist_ok=True)
        with open('eval/sample_queries.json', 'w') as f:
            json.dump(queries, f, indent=2)
        
        logger.info(f"Created and saved {len(queries)} default {dataset_type} queries")
        return queries[:query_limit]


def create_pipeline_wrappers(top_k: int = DEFAULT_TOP_K) -> Dict[str, Dict[str, Any]]:
    """
    Create wrapper functions for each RAG pipeline.
    
    Args:
        top_k: Number of documents to retrieve for each query
        
    Returns:
        Dictionary mapping technique names to their configuration
    """
    # Basic RAG wrapper
    def basic_rag_wrapper(query, iris_connector=None, embedding_func=None, llm_func=None, **kwargs):
        """Wrapper for BasicRAGPipeline."""
        pipeline = BasicRAGPipeline(iris_connector, embedding_func, llm_func)
        top_k = kwargs.get("top_k", DEFAULT_TOP_K)
        return pipeline.run(query, top_k=top_k)

    # HyDE wrapper
    def hyde_wrapper(query, iris_connector=None, embedding_func=None, llm_func=None, **kwargs):
        """Wrapper for HyDERAGPipeline."""
        pipeline = HyDERAGPipeline(iris_connector, embedding_func, llm_func)
        top_k = kwargs.get("top_k", DEFAULT_TOP_K)
        return pipeline.run(query, top_k=top_k)

    # ColBERT wrapper
    def colbert_wrapper(query, iris_connector=None, embedding_func=None, llm_func=None, **kwargs):
        """Wrapper for ColBERTRAGPipeline."""
        # For ColBERT, use the semantic encoder from the core pipeline
        
        # Pass the potentially stubbed embedding_func to the encoder factory
        # create_colbert_semantic_encoder is imported at the top from core_pipelines
        semantic_encoder = create_colbert_semantic_encoder(embedding_func_override=embedding_func)
        
        # Initialize ColBERTRAGPipeline with the created encoder
        pipeline = ColBERTRAGPipeline(
            iris_connector=iris_connector,
            colbert_query_encoder_func=semantic_encoder, # Use the returned encoder directly
            colbert_doc_encoder_func=semantic_encoder, # Use the same for doc encoding as per original ColBERTRAGPipeline
            llm_func=llm_func
        )
        
        top_k = kwargs.get("top_k", DEFAULT_TOP_K)
        return pipeline.run(query, top_k=top_k)

    # CRAG wrapper
    def crag_wrapper(query, iris_connector=None, embedding_func=None, llm_func=None, **kwargs):
        """Wrapper for CRAGPipeline."""
        pipeline = CRAGPipeline(iris_connector, embedding_func, llm_func)
        top_k = kwargs.get("top_k", DEFAULT_TOP_K)
        return pipeline.run(query, top_k=top_k)

    # NodeRAG wrapper
    def noderag_wrapper(query, iris_connector=None, embedding_func=None, llm_func=None, **kwargs):
        """Wrapper for NodeRAGPipeline."""
        pipeline = NodeRAGPipeline(iris_connector, embedding_func, llm_func)
        # NodeRAGPipeline.run expects 'top_k' for the final document count.
        # The 'top_k_seeds' logic is internal to its retrieval methods.
        # The wrapper should pass the 'top_k' value intended for the overall pipeline.
        actual_top_k = kwargs.get("top_k", DEFAULT_TOP_K)
        return pipeline.run(query, top_k=actual_top_k)

    # GraphRAG wrapper
    def graphrag_wrapper(query, iris_connector=None, embedding_func=None, llm_func=None, **kwargs):
        """Wrapper for GraphRAGPipeline."""
        pipeline = GraphRAGPipeline(iris_connector, embedding_func, llm_func)
        # GraphRAGPipeline.execute (which is its run method) calls self.query(query_text, top_k)
        # The wrapper should pass 'top_k'.
        actual_top_k = kwargs.get("top_k", DEFAULT_TOP_K)
        return pipeline.run(query, top_k=actual_top_k)

    # Return all wrappers in a dictionary
    return {
        "basic_rag": {
            "pipeline_func": basic_rag_wrapper,
            "top_k": top_k
        },
        "hyde": {
            "pipeline_func": hyde_wrapper,
            "top_k": top_k
        },
        "colbert": {
            "pipeline_func": colbert_wrapper,
            "top_k": top_k
        },
        "crag": {
            "pipeline_func": crag_wrapper,
            "top_k": top_k
        },
        "noderag": {
            "pipeline_func": noderag_wrapper,
            "top_k": top_k
        },
        "graphrag": {
            "pipeline_func": graphrag_wrapper,
            "top_k": top_k
        }
    }


def ensure_min_documents(conn, db_schema: str, min_count: int = MIN_DOCUMENT_COUNT) -> bool:
    """
    Ensure that the database has at least the minimum required documents.
    
    Args:
        conn: IRIS database connection
        db_schema: The database schema (e.g., "RAG")
        min_count: Minimum number of documents required
        
    Returns:
        Boolean indicating success
    """
    try:
        # Check current document count
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) FROM {db_schema}.SourceDocuments") # Use schema and correct table
            count_result = cursor.fetchone()
            current_count = int(count_result[0]) if count_result else 0
            logger.info(f"Current document count: {current_count}")
            
            # If we already have enough documents, we're done
            if current_count >= min_count:
                logger.info(f"✅ Found {current_count} documents (≥{min_count} required)")
                return True
            else:
                logger.error(f"❌ Insufficient documents: {current_count} < {min_count}")
                logger.info(f"Please load more documents using load_pmc_data.py --limit {min_count}")
                return False
    except Exception as e:
        logger.error(f"Error checking document count: {e}")
        return False


def setup_database_connection(args) -> Optional[Any]:
    """Set up and verify the database connection using factory pattern."""
    logger.info("Establishing connection to IRIS database...")
    
    try:
        from common.connection_factory import ConnectionFactory
        
        if args.use_mock:
            print("Mock not supported anymore")
        else:
            # Use factory with DBAPI as default (user preference)
            connection_config = {}
            
            # Build config from command line args if provided
            if hasattr(args, 'iris_host') and args.iris_host:
                connection_config.update({
                    'host': args.iris_host,
                    'port': args.iris_port,
                    'namespace': args.iris_namespace,
                    'user': args.iris_user,
                    'password': args.iris_password
                })
            
            # Create DBAPI connection (user preference)
            iris_conn = ConnectionFactory.create_connection("dbapi", **connection_config)
            
        # Validate connection
        if iris_conn is None:
            error_msg = """
ERROR: Failed to establish an IRIS connection. This benchmark requires an IRIS database.

To fix this issue:
1. Ensure IRIS database is installed and running
2. Check your connection credentials
3. Verify environment variables are set correctly:
   - IRIS_HOST (default: localhost)
   - IRIS_PORT (default: 1972)
   - IRIS_NAMESPACE (default: USER)
   - IRIS_USERNAME (default: SuperUser)
   - IRIS_PASSWORD (default: SYS)

See docs/BENCHMARK_SETUP.md for detailed setup instructions.
"""
            logger.error(error_msg)
            print(error_msg)
            return None
        
        logger.info("IRIS connection established successfully")
        return iris_conn
        
    except Exception as e:
        logger.error(f"Failed to create database connection: {e}")
        return None


def prepare_colbert_embeddings(iris_conn, args) -> bool:
    """
    Prepare ColBERT token embeddings if needed.
    
    Args:
        iris_conn: IRIS database connection
        args: Command line arguments
        
    Returns:
        Boolean indicating success
    """
    if 'colbert' in args.techniques and iris_conn:
        logger.info("Preparing for ColBERT: loading token embeddings...")
        try:
            from tests.utils import load_colbert_token_embeddings
            
            num_tokens = load_colbert_token_embeddings(
                connection=iris_conn,
                limit=args.num_docs,
                mock_colbert_encoder=args.use_mock
            )
            logger.info(f"ColBERT: Loaded {num_tokens} token embeddings (mock_encoder={args.use_mock}).")
            time.sleep(1)  # Brief pause after potential DB operations
            return True
        except Exception as e:
            logger.error(f"Error loading ColBERT token embeddings: {e}")
            return False
    return True  # Skip if ColBERT not in techniques


def initialize_embedding_and_llm(args) -> Tuple[Any, Any]:
    """
    Initialize embedding and LLM functions based on arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        Tuple of (embedding_func, llm_func)
    """
    logger.info(f"Initializing embedding function with provider: {args.embedding_provider}")
    # embedding_utils_module is imported and aliased at the top of the script
    # and includes a mock if the import fails.
    embedding_func = embedding_utils_module.get_embedding_func(provider=args.embedding_provider)
    
    logger.info(f"Initializing LLM function with provider: {args.llm}")
    # get_llm_func is defined later in this script.
    llm_func = get_llm_func(provider=args.llm, model_name=getattr(args, 'llm_model_name', 'gpt-3.5-turbo')) # Added model_name
    
    logger.info(f"Embedding and LLM functions initialized (LLM provider: {args.llm}, Embedding provider: {args.embedding_provider})")
    return embedding_func, llm_func


def get_llm_func(provider: str = "stub", model_name: str = "gpt-3.5-turbo") -> Any:
    """
    Get an LLM function based on the provider and model name.
    
    Args:
        provider: LLM provider (openai, stub, etc.)
        model_name: Name of the model to use
        
    Returns:
        LLM function
    """
    if provider == "stub":
        # Return a stub LLM function that returns a fixed response
        def stub_llm_func(prompt, **kwargs):
            return f"Stub LLM response for: {prompt[:50]}..."
        return stub_llm_func
    elif provider == "openai":
        # Import here to avoid dependency if not needed
        try:
            from openai import OpenAI
            client = OpenAI()
            
            def openai_llm_func(prompt, **kwargs):
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[{"role": "user", "content": prompt}],
                    temperature=kwargs.get("temperature", 0.7),
                    max_tokens=kwargs.get("max_tokens", 500)
                )
                return response.choices[0].message.content
            
            return openai_llm_func
        except ImportError:
            logger.warning("OpenAI package not installed, falling back to stub LLM")
            return get_llm_func(provider="stub")
    else:
        logger.warning(f"Unknown LLM provider: {provider}, falling back to stub LLM")
        return get_llm_func(provider="stub")


def run_benchmarks(args) -> Optional[str]:
    """
    Run the RAG benchmarks according to the specified arguments.
    
    Args:
        args: Command line arguments
        
    Returns:
        Path to the generated report or None if benchmarking failed
    """
    # Create output directory for results
    output_dir = args.output_dir
    if not output_dir:
        output_dir = os.path.join(
            "benchmark_results", 
            f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
    os.makedirs(output_dir, exist_ok=True)
    
    # Set up database connection
    iris_conn = setup_database_connection(args)
    if iris_conn is None:
        print("DEBUG: run_benchmarks returning None because iris_conn is None")
        return None
    
    try:
        print("DEBUG: Entered run_benchmarks try block")
        # Ensure we have enough documents
        if not args.use_mock:
            logger.info("Verifying document count requirement...")
            if not ensure_min_documents(iris_conn, db_schema=args.db_schema, min_count=args.num_docs):
                logger.critical(
                    f"Initial document check failed. The table {args.db_schema}.SourceDocuments might be missing or empty. "
                    f"Please ensure the database schema is correctly initialized by running: "
                    f"python run_db_init_local.py --force-recreate"
                )
                print("DEBUG: run_benchmarks returning None due to insufficient documents")
                return None
        else:
            logger.info("Skipping document count verification for mock run.")
        
        # Prepare ColBERT token embeddings if needed
        if not prepare_colbert_embeddings(iris_conn, args):
            logger.error("Failed to prepare ColBERT token embeddings.")
            print("DEBUG: run_benchmarks returning None due to ColBERT prep failure")
            return None
        
        # Initialize embedding and LLM functions
        try:
            embedding_func, llm_func = initialize_embedding_and_llm(args)
        except Exception:
            print("DEBUG: run_benchmarks returning None due to initialize_embedding_and_llm failure")
            return None
        
        # Load queries based on dataset type
        queries = load_queries(dataset_type=args.dataset, query_limit=args.num_queries)
        logger.info(f"Loaded {len(queries)} queries from {args.dataset} dataset")
        if not queries: # Explicit check, though load_queries might return empty list not None
            logger.error("No queries loaded (explicit check).")
            print("DEBUG: run_benchmarks returning None due to no queries (explicit check)")
            return None
        
        # Create pipeline wrappers
        pipeline_wrappers = create_pipeline_wrappers(top_k=args.top_k)
        
        # Filter techniques based on command line arguments
        techniques = {}
        for tech_name in args.techniques:
            if tech_name in pipeline_wrappers:
                # Add the connection and functions to each technique
                techniques[tech_name] = pipeline_wrappers[tech_name].copy()
                techniques[tech_name]["iris_connector"] = iris_conn
                techniques[tech_name]["embedding_func"] = embedding_func
                techniques[tech_name]["llm_func"] = llm_func
            else:
                logger.warning(f"Technique '{tech_name}' not found, skipping")
        
        if not techniques:
            logger.error("No valid techniques specified. Exiting.")
            print("DEBUG: run_benchmarks returning None due to no valid techniques")
            return None
        
        # Output path for benchmark results
        benchmark_output = os.path.join(output_dir, "benchmark_results.json")
        
        # Run the benchmarks
        logger.info(f"Running benchmarks for {len(techniques)} techniques...")
        logger.info(f"Using {len(queries)} queries from {args.dataset} dataset")
        
        results = run_all_techniques_benchmark(
            queries=queries,
            techniques=techniques,
            output_path=benchmark_output
        )
        logger.info(f"--- Intermediate results from run_all_techniques_benchmark: {results}")
        logger.info(f"Successfully ran benchmarks for {len(results)} techniques")
        
        # Generate the comparative analysis report
        logger.info("Generating comparative analysis report...")
        report_dir = os.path.join(output_dir, "reports")
        os.makedirs(report_dir, exist_ok=True)
        
        report_paths = generate_combined_report(
            benchmarks=results,
            output_dir=report_dir,
            dataset_name=args.dataset
        )
        
        logger.info("\nBenchmark Complete!")
        logger.info(f"Results saved to: {output_dir}")
        logger.info("\nGenerated Files:")
        
        for report_type, path in report_paths.items():
            if report_type == "charts":
                logger.info(f"- Generated {len(path)} charts in {report_dir}")
            else:
                logger.info(f"- {report_type}: {path}")
        
        return report_paths.get("markdown")
    
    except Exception as e:
        print(f"DEBUG: In run_benchmarks except block. Error: {e}") # DEBUG PRINT
        logger.error(f"Error running benchmarks: {e}", exc_info=True)
        import traceback
        tb_str = traceback.format_exc()
        print(f"DEBUG: Full traceback:\n{tb_str}") # DEBUG PRINT
        logger.error(f"Full traceback:\n{tb_str}")
        
        # Attempt to flush logs here to ensure this error gets written
        if logging.getLogger().handlers:
            file_handler = next((h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler)), None)
            if file_handler and hasattr(file_handler, 'flush'):
                try:
                    file_handler.flush()
                    logger.info("Log file handler flushed within run_benchmarks except block.")
                    import time
                    time.sleep(0.1) # Brief pause to allow I/O to complete
                except Exception as e_flush_except:
                    print(f"Warning: Could not flush file log handler in run_benchmarks except block: {e_flush_except}")
        return None
    
    finally:
        # Make sure to close the IRIS connection when done
        try:
            if iris_conn:
                iris_conn.close()
                logger.info("IRIS connection closed")
        except Exception as e:
            logger.error(f"Error closing IRIS connection: {e}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Comprehensive RAG Benchmarking with real IRIS database",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Database connection options
    db_group = parser.add_argument_group("Database Connection Options")
    db_group.add_argument("--use-testcontainer", action="store_true",
                        help="Use testcontainer instead of direct IRIS connection")
    db_group.add_argument("--use-mock", action="store_true",
                        help="Use mock IRIS connection and mock embeddings/LLM")
    db_group.add_argument("--iris-host", type=str,
                        help="IRIS host (if connecting to existing instance)")
    db_group.add_argument("--iris-port", type=int,
                        help="IRIS port (if connecting to existing instance)")
    db_group.add_argument("--iris-namespace", type=str,
                        help="IRIS namespace (if connecting to existing instance)")
    db_group.add_argument("--iris-user", type=str,
                        help="IRIS username (if connecting to existing instance)")
    db_group.add_argument("--iris-password", type=str,
                        help="IRIS password (if connecting to existing instance)")
    db_group.add_argument("--db-schema", type=str, default="RAG",
                        help="Database schema to use (default: RAG)")
    
    # Benchmark configuration
    bench_group = parser.add_argument_group("Benchmark Configuration")
    bench_group.add_argument("--techniques", nargs="+", default=DEFAULT_TECHNIQUES,
                           help="RAG techniques to benchmark")
    bench_group.add_argument("--dataset", choices=["medical", "multihop"], default=DEFAULT_DATASET,
                           help="Type of queries to use for benchmarking")
    bench_group.add_argument("--llm", choices=["gpt-3.5-turbo", "gpt-4", "stub"], default=DEFAULT_LLM,
                           help="LLM model to use for generating answers")
    bench_group.add_argument("--embedding-provider", type=str, default="stub",
                           help="Embedding provider to use (stub, openai, etc.)")
    bench_group.add_argument("--num-docs", type=int, default=MIN_DOCUMENT_COUNT,
                           help="Expected minimum document count for the benchmark run")
    bench_group.add_argument("--num-queries", type=int, default=DEFAULT_QUERY_LIMIT,
                           help="Maximum number of queries to run")
    bench_group.add_argument("--top-k", type=int, default=DEFAULT_TOP_K,
                           help="Number of documents to retrieve for each query")
    
    # Output options
    output_group = parser.add_argument_group("Output Options")
    output_group.add_argument("--output-dir", type=str,
                            help="Directory to save benchmark results (default: benchmark_results/timestamp)")
    output_group.add_argument("--verbose", action="store_true",
                            help="Enable verbose logging")
    
    return parser.parse_args()


def main():
    """Main entry point for the script."""
    # Configure basic logging first to catch early errors if parse_args fails
    # This will log to console by default if file handler setup fails.
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        # handlers will be added after log file is cleared
    )
    logger = logging.getLogger("rag_benchmarks_main") # Use a specific logger for main

    try:
        # Parse command line arguments
        args = parse_args()
        
        # Setup file logging (now that args are parsed, if output_dir is used for log path)
        # For simplicity, keeping log_file fixed as "benchmark_run.log"
        log_file = "benchmark_run.log"
        # This was moved to if __name__ == "__main__" block

        # Reconfigure logging to include FileHandler
        # Remove any existing handlers to avoid duplication if basicConfig was called before
        root_logger = logging.getLogger()
        for handler in root_logger.handlers[:]:
            root_logger.removeHandler(handler)
        
        logging.basicConfig(
            level=logging.DEBUG if args.verbose else logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[
                logging.FileHandler(log_file, mode='w'), # Ensure mode is 'w'
                logging.StreamHandler()
            ]
        )
        # Update our specific logger instance if needed, or just use root.
        logger = logging.getLogger("rag_benchmarks") # Main script logger

        logger.info(f"Arguments: {args}") # Log parsed arguments

        # Attempt to flush logs immediately after basicConfig and level setting
        if logging.getLogger().handlers:
            file_handler = next((h for h in logging.getLogger().handlers if isinstance(h, logging.FileHandler)), None)
            if file_handler and hasattr(file_handler, 'flush'):
                try:
                    file_handler.flush()
                except Exception as e_flush:
                    print(f"Warning: Could not flush file log handler during setup: {e_flush}")
        
        logger.info("Starting RAG benchmarking script (main flow)...")
        
        # Record start time
        start_time = time.time()
        
        # Run the benchmarks
        report_path = run_benchmarks(args)
        
        # Calculate duration
        end_time = time.time()
        duration = end_time - start_time
        minutes, seconds = divmod(duration, 60)
        
        # Print summary
        if report_path:
            logger.info(f"Benchmark completed in {int(minutes)} minutes and {seconds:.1f} seconds. Report: {report_path}")
            print(f"\nBenchmark completed in {int(minutes)} minutes and {seconds:.1f} seconds")
            print(f"Open this file to view the report: {report_path}")
        else:
            logger.error("Benchmark failed (main flow determined this). Check logs.")
            print("\nBenchmark failed. Check the logs for details.")
        
        return 0 if report_path else 1

    except Exception as e:
        logger.critical(f"--- Unhandled exception in main(): {e} ---", exc_info=True)
        import traceback
        logger.critical(f"--- Main() full traceback ---\n{traceback.format_exc()}")
        return 1 # Indicate failure
    finally:
        logging.shutdown() # Flushes and closes all handlers


if __name__ == "__main__":
    # Clear log file at the very beginning of script execution
    # This ensures it's cleared even if main() or arg parsing fails early.
    log_file_name = "benchmark_run.log"
    try:
        with open(log_file_name, 'w') as lf:
            lf.write("") # Truncate the file
        # print(f"Log file {log_file_name} cleared at script start.") # For debugging log clearing
    except IOError as e_io:
        print(f"Warning: Could not clear log file {log_file_name} at start: {e_io}")

    sys.exit(main())