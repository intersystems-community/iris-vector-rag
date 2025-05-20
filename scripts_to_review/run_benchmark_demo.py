#!/usr/bin/env python3
# run_benchmark_demo.py
# Demo script to run the benchmarking system with real data and IRIS connection

import os
import sys
import json
import logging
import time
import numpy as np
import argparse
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("benchmark_demo")

# Import IRIS connector
from common.iris_connector import get_iris_connection
from common.utils import get_embedding_func, get_llm_func

# Constants
MIN_DOCUMENT_COUNT = 1000

# Import the benchmarking and analysis modules
from eval.bench_runner import run_all_techniques_benchmark, load_benchmark_results
# from eval.comparative import generate_combined_report # Moved to main()

# Import pipeline classes
from basic_rag.pipeline import BasicRAGPipeline
from hyde.pipeline import HyDEPipeline
from colbert.pipeline import ColbertRAGPipeline
from crag.pipeline import CRAGPipeline
from noderag.pipeline import NodeRAGPipeline
from graphrag.pipeline import GraphRAGPipeline

# Load queries from sample_queries.json or create MultiHopQA queries
def load_queries(dataset_type="medical"):
    """
    Load queries from sample_queries.json or create queries based on the specified dataset type.
    
    Args:
        dataset_type: Type of dataset queries to create (medical, multihop, etc.)
        
    Returns:
        List of query dictionaries
    """
    try:
        with open('eval/sample_queries.json', 'r') as f:
            return json.load(f)
    except (FileNotFoundError, json.JSONDecodeError):
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
        return queries

# Create wrapper functions for each pipeline
def basic_rag_wrapper(query, iris_connector=None, embedding_func=None, llm_func=None, **kwargs):
    """Wrapper for BasicRAGPipeline."""
    pipeline = BasicRAGPipeline(iris_connector, embedding_func, llm_func)
    top_k = kwargs.get("top_k", 5)
    return pipeline.run(query, top_k=top_k)

def hyde_wrapper(query, iris_connector=None, embedding_func=None, llm_func=None, **kwargs):
    """Wrapper for HyDEPipeline."""
    pipeline = HyDEPipeline(iris_connector, embedding_func, llm_func)
    top_k = kwargs.get("top_k", 5)
    return pipeline.run(query, top_k=top_k)

def colbert_wrapper(query, iris_connector=None, embedding_func=None, llm_func=None, **kwargs):
    """Wrapper for ColbertRAGPipeline."""
    # For ColBERT, we need to create a query encoder that handles token-level embeddings
    from colbert.query_encoder import ColBERTQueryEncoder
    
    # Pass the potentially stubbed embedding_func to the encoder
    # The ColBERTQueryEncoder's __init__ will detect if it's a stub.
    query_encoder = ColBERTQueryEncoder(embedding_func=embedding_func) 
    
    # Initialize ColbertRAGPipeline with the needed encoders
    pipeline = ColbertRAGPipeline(
        iris_connector=iris_connector, 
        colbert_query_encoder_func=query_encoder.encode_query, # Uses the new method
        colbert_doc_encoder_func=query_encoder.encode_document, # Uses the new method
        llm_func=llm_func
    )
    
    top_k = kwargs.get("top_k", 5)
    return pipeline.run(query, top_k=top_k)

def crag_wrapper(query, iris_connector=None, embedding_func=None, llm_func=None, **kwargs):
    """Wrapper for CRAGPipeline."""
    pipeline = CRAGPipeline(iris_connector, embedding_func, llm_func)
    top_k = kwargs.get("top_k", 5)
    return pipeline.run(query, top_k=top_k)

def noderag_wrapper(query, iris_connector=None, embedding_func=None, llm_func=None, **kwargs):
    """Wrapper for NodeRAGPipeline."""
    pipeline = NodeRAGPipeline(iris_connector, embedding_func, llm_func)
    top_k_seeds = kwargs.get("top_k", 5) # The wrapper receives 'top_k' from run_all_techniques_benchmark
    return pipeline.run(query, top_k_seeds=top_k_seeds) # Pass as 'top_k_seeds'

def graphrag_wrapper(query, iris_connector=None, embedding_func=None, llm_func=None, **kwargs):
    """Wrapper for GraphRAGPipeline."""
    pipeline = GraphRAGPipeline(iris_connector, embedding_func, llm_func)
    top_n_start_nodes = kwargs.get("top_k", 5) # The wrapper receives 'top_k'
    return pipeline.run(query, top_n_start_nodes=top_n_start_nodes) # Pass as 'top_n_start_nodes'

def ensure_min_documents(conn, min_count=MIN_DOCUMENT_COUNT):
    """
    Ensure that the database has at least the minimum required documents.
    
    Args:
        conn: IRIS database connection
        min_count: Minimum number of documents required
        
    Returns:
        Boolean indicating success
    """
    try:
        # Check current document count
        with conn.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
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

def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="RAG Benchmarking with real IRIS database")
    
    parser.add_argument("--use-testcontainer", action="store_true",
                       help="Use testcontainer instead of direct IRIS connection")
    
    parser.add_argument("--techniques", nargs="+", 
                       default=["basic_rag", "hyde", "colbert", "crag", "noderag", "graphrag"],
                       help="RAG techniques to benchmark")
    
    parser.add_argument("--query-limit", type=int, default=5,
                       help="Maximum number of queries to run")
    
    parser.add_argument("--dataset", choices=["medical", "multihop"], default="medical",
                       help="Type of queries to use for benchmarking")
    
    parser.add_argument("--llm", choices=["gpt-3.5-turbo", "gpt-4", "stub"], default="stub",
                       help="LLM model to use for generating answers")
    parser.add_argument("--document-count", type=int, default=MIN_DOCUMENT_COUNT, 
                        help=f"Expected minimum document count for the benchmark run (default: {MIN_DOCUMENT_COUNT})")
    
    # Arguments for connecting to an existing IRIS instance (e.g., a testcontainer started by a parent script)
    parser.add_argument("--iris-host", type=str, help="IRIS host (if connecting to existing instance)")
    parser.add_argument("--iris-port", type=int, help="IRIS port (if connecting to existing instance)")
    parser.add_argument("--iris-namespace", type=str, help="IRIS namespace (if connecting to existing instance)")
    parser.add_argument("--iris-user", type=str, help="IRIS username (if connecting to existing instance)")
    parser.add_argument("--iris-password", type=str, help="IRIS password (if connecting to existing instance)")
    parser.add_argument("--use-mock", action="store_true", 
                        help="Use mock IRIS connection and mock ColBERT encoder.")
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_args()
    
    logger.info("Starting RAG benchmarking demo script...")
    
    # Create output directory for results
    output_dir = os.path.join(
        "benchmark_results", 
        f"benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    )
    os.makedirs(output_dir, exist_ok=True)
    
    # Get IRIS connection
    logger.info("Establishing connection to IRIS database...")
    
    iris_conn = None
    if args.use_mock:
        logger.info("Using mock IRIS connection as requested by --use-mock flag.")
        from common.iris_connector import get_mock_iris_connection
        iris_conn = get_mock_iris_connection()
    elif args.iris_host and args.iris_port and args.iris_namespace and args.iris_user and args.iris_password:
        logger.info(f"Attempting to connect to existing IRIS instance: {args.iris_host}:{args.iris_port}/{args.iris_namespace}")
        from common.iris_connector import get_real_iris_connection # Use the direct connection function
        iris_conn = get_real_iris_connection(config={
            "hostname": args.iris_host,
            "port": args.iris_port,
            "namespace": args.iris_namespace,
            "username": args.iris_user,
            "password": args.iris_password
        })
    elif args.use_testcontainer:
        logger.info("Using testcontainer as specified (no existing instance parameters provided)...")
        iris_conn = get_iris_connection(use_mock=False, use_testcontainer=True)
    else:
        # Use default direct connection (e.g. localhost)
        logger.info("Connecting to IRIS directly (default parameters)...")
        iris_conn = get_iris_connection(use_mock=False, use_testcontainer=False)
    
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

See BENCHMARK_SETUP.md for detailed setup instructions.
"""
        logger.error(error_msg)
        print(error_msg)
        return None
    
    logger.info("IRIS connection established successfully")
    
    # Ensure we have at least args.document_count documents in the database
    if not args.use_mock: # Skip document verification for mock runs
        logger.info("Verifying document count requirement...")
        # Use the document_count from args, which defaults to MIN_DOCUMENT_COUNT if not provided
        # but can be overridden by the parent script.
        if not ensure_min_documents(iris_conn, min_count=args.document_count):
            logger.warning(f"Initial document check failed. Attempting schema initialization within demo script.")
            try:
                from common.db_init import initialize_database
                initialize_database(iris_conn, force_recreate=False) 
                logger.info("Schema initialization attempted from demo script.")
                # Re-check documents with the required count
                if not ensure_min_documents(iris_conn, min_count=args.document_count):
                    logger.error(f"Insufficient documents in database after re-check. Need at least {args.document_count}.")
                    logger.error(f"Please load at least {args.document_count} documents.")
                    return None
            except Exception as e_init:
                logger.error(f"Error during schema initialization from demo script: {e_init}")
                logger.error(f"Insufficient documents in database. Need at least {args.document_count}.")
            logger.error(f"Please load at least {args.document_count} documents.")
            return None
    else:
        logger.info("Skipping document count verification for mock run.")

    # Load ColBERT token embeddings if 'colbert' is a technique
    # This needs to happen after potential schema init and document loading (for non-mock)
    # or after mock connection setup.
    if 'colbert' in args.techniques and iris_conn:
        logger.info("Preparing for ColBERT: loading token embeddings...")
        from tests.utils import load_colbert_token_embeddings
        # The mock_colbert_encoder flag in load_colbert_token_embeddings will use
        # a mock encoder if args.use_mock is True.
        # If not args.use_mock, it will attempt to use the real ColBERT encoder.
        # For a mock DB run, we still need to "load" mock tokens.
        num_tokens = load_colbert_token_embeddings(
            connection=iris_conn,
            limit=args.document_count, # Use the same document limit
            mock_colbert_encoder=args.use_mock # Critically, pass the mock flag here
        )
        logger.info(f"ColBERT: Loaded {num_tokens} token embeddings (mock_encoder={args.use_mock}).")
        time.sleep(1) # Brief pause after potential DB operations

    # Get embedding and LLM functions
    try:
        # If --use-mock is passed, it implies stubbing for LLM and general embeddings too for this demo script's context
        if args.use_mock or args.llm == "stub":
            logger.info("Using stub for LLM and general embedding function (due to --use-mock or --llm stub).")
            embedding_func = get_embedding_func(provider="stub")
            llm_func = get_llm_func(provider="stub")
        else:
            logger.info(f"Using real models for LLM ({args.llm}) and default embedding model.")
            embedding_func = get_embedding_func() # Default real model
            llm_func = get_llm_func(provider="openai", model_name=args.llm)
        
        logger.info(f"Embedding and LLM functions initialized (LLM mode: {args.llm}, Mock mode: {args.use_mock})")
    except Exception as e:
        logger.error(f"Failed to initialize embedding or LLM functions: {e}")
        return None
    
    # Load queries based on dataset type
    queries = load_queries(dataset_type=args.dataset)
    logger.info(f"Loaded {len(queries)} queries from {args.dataset} dataset")
    
    # Define all available techniques
    available_techniques = {
        "basic_rag": {
            "pipeline_func": basic_rag_wrapper,
            "iris_connector": iris_conn,
            "embedding_func": embedding_func,
            "llm_func": llm_func,
            "top_k": 5
        },
        "hyde": {
            "pipeline_func": hyde_wrapper,
            "iris_connector": iris_conn,
            "embedding_func": embedding_func,
            "llm_func": llm_func,
            "top_k": 5
        },
        "colbert": {
            "pipeline_func": colbert_wrapper,
            "iris_connector": iris_conn,
            "embedding_func": embedding_func,
            "llm_func": llm_func,
            "top_k": 5
        },
        "crag": {
            "pipeline_func": crag_wrapper,
            "iris_connector": iris_conn,
            "embedding_func": embedding_func,
            "llm_func": llm_func,
            "top_k": 5
        },
        "noderag": {
            "pipeline_func": noderag_wrapper,
            "iris_connector": iris_conn,
            "embedding_func": embedding_func,
            "llm_func": llm_func,
            "top_k": 5
        },
        "graphrag": {
            "pipeline_func": graphrag_wrapper,
            "iris_connector": iris_conn,
            "embedding_func": embedding_func,
            "llm_func": llm_func,
            "top_k": 5
        }
    }
    
    # Filter techniques based on command line arguments
    techniques = {}
    for tech_name in args.techniques:
        if tech_name in available_techniques:
            techniques[tech_name] = available_techniques[tech_name]
        else:
            logger.warning(f"Technique '{tech_name}' not found, skipping")
    
    if not techniques:
        logger.error("No valid techniques specified. Exiting.")
        return None
    
    # Output path for benchmark results
    benchmark_output = os.path.join(output_dir, "benchmark_results.json")
    
    try:
        # Run the benchmarks
        logger.info(f"Running benchmarks for {len(techniques)} techniques...")
        logger.info(f"Using {min(args.query_limit, len(queries))} queries from {args.dataset} dataset")
        
        # Use subset of queries based on limit
        results = run_all_techniques_benchmark(
            queries=queries[:args.query_limit],
            techniques=techniques,
            output_path=benchmark_output
        )
        logger.info(f"Successfully ran benchmarks for {len(results)} techniques")
        
    except Exception as e:
        logger.error(f"Error running benchmarks: {e}")
        return None
    
    finally:
        # Make sure to close the IRIS connection when done
        try:
            if iris_conn:
                iris_conn.close()
                logger.info("IRIS connection closed")
        except Exception as e:
            logger.error(f"Error closing IRIS connection: {e}")
    
    # Generate the comparative analysis report
    logger.info("Generating comparative analysis report...")
    report_dir = os.path.join(output_dir, "reports")
    markdown_report_path = None
    try:
        from eval.comparative import generate_combined_report # Import here
        report_paths = generate_combined_report(
            benchmarks=results,
            output_dir=report_dir,
            dataset_name=args.dataset
        )
        
        logger.info("\nBenchmark Complete!")
        # This specific print is for run_complete_benchmark.py to parse
        print(f"Results saved to: {output_dir}") 
        logger.info(f"Results saved to: {output_dir}") # Keep logging for human readability
        logger.info("\nGenerated Files:")
        
        for report_type, path in report_paths.items():
            if report_type == "charts":
                logger.info(f"- Generated {len(path)} charts in {report_dir}")
            else:
                logger.info(f"- {report_type}: {path}")
        markdown_report_path = report_paths.get("markdown")
        
    except ImportError as e_import:
        if 'matplotlib' in str(e_import).lower():
            logger.warning(f"Matplotlib not found, skipping chart generation: {e_import}")
            # Still print "Results saved to" because JSON results were generated
            print(f"Results saved to: {output_dir}")
            logger.info(f"Results saved to: {output_dir} (chart generation skipped)")
            # Try to find if markdown report was generated if it doesn't depend on charts
            # Assuming benchmark_results.json is the primary output.
            # The markdown report might also fail if it tries to embed charts.
            # For now, just indicate main results are in output_dir.
        else:
            logger.error(f"Error generating report (ImportError): {e_import}")
            # If it's another import error, also print results saved to allow parent to proceed
            print(f"Results saved to: {output_dir}")
            logger.info(f"Results saved to: {output_dir} (report generation may be incomplete)")

    except Exception as e_report:
        logger.error(f"Error generating report: {e_report}")
        # If report generation fails, still print that main results were saved.
        print(f"Results saved to: {output_dir}")
        logger.info(f"Results saved to: {output_dir} (report generation failed)")
    
    return markdown_report_path # This might be None if report generation failed


if __name__ == "__main__":
    start_time = time.time()
    report_path = main()
    end_time = time.time()
    
    duration = end_time - start_time
    minutes, seconds = divmod(duration, 60)
    
    if report_path:
        print(f"\nBenchmark completed in {int(minutes)} minutes and {seconds:.1f} seconds")
        print(f"Open this file to view the report: {report_path}")
    else:
        print("\nBenchmark failed. Check the logs for details.")
