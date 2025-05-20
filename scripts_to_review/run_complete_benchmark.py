#!/usr/bin/env python3
# run_complete_benchmark.py
# Script to execute the complete benchmarking process as defined in BENCHMARK_EXECUTION_PLAN.md

import os
import sys
import logging
import subprocess
import argparse
import time
import importlib.util
from datetime import datetime
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("benchmark_runner")

# Define constants
MIN_DOCUMENT_COUNT = 1000 # Default minimum for "real" benchmarks
# DEFAULT_TECHNIQUES and DEFAULT_DATASETS are now in benchmark_runner_utils.py with parse_args
BENCHMARK_DIR = "benchmark_results" # This can remain if used elsewhere, or also move.

# Import utilities
from benchmark_runner_utils import (
    parse_args, # We will need to modify parse_args to include the new flag
    check_dependencies,
    # setup_iris_testcontainer, # We might replace this or make it conditional
    load_test_data,
    verify_document_count,
    run_comparison_analysis,
    create_summary_report
)

# Functions like check_dependencies, setup_iris_testcontainer, load_test_data, 
# verify_document_count, run_comparison_analysis, and create_summary_report
# are now imported from benchmark_runner_utils.py and their definitions are removed from this file.

def run_benchmarks(args, iris_connection_params=None):
    """
    Run benchmarks for all specified techniques and datasets.
    If iris_connection_params is provided, run_benchmark_demo.py will use these.
    Otherwise, run_benchmark_demo.py will attempt its own connection (e.g. new testcontainer).
    """
    logger.info("Step 2: Running benchmarks...")
    os.makedirs(args.output_dir, exist_ok=True)
    dataset_results = {}

    for dataset in args.datasets:
        logger.info(f"Running benchmarks for {dataset} dataset...")
        dataset_log_dir = os.path.join(args.output_dir, dataset)
        os.makedirs(dataset_log_dir, exist_ok=True)

        # Ensure subprocess runs within the poetry environment
        cmd_prefix = ["poetry", "run"] if os.path.exists("pyproject.toml") else []
        
        # Use "python" directly with "poetry run" to ensure venv's Python is used
        python_executable = "python" if cmd_prefix else sys.executable

        cmd = cmd_prefix + [
            python_executable, "run_benchmark_demo.py",
            "--techniques", *args.techniques,
            "--dataset", dataset,
            "--query-limit", str(args.query_limit),
            "--llm", args.llm,
            "--document-count", str(args.document_count) # Pass document_count
        ]

        if iris_connection_params:
            cmd.extend([
                "--iris-host", iris_connection_params["host"],
                "--iris-port", str(iris_connection_params["port"]),
                "--iris-namespace", iris_connection_params["namespace"],
                "--iris-user", iris_connection_params["user"],
                "--iris-password", iris_connection_params["password"]
            ])
        # If no explicit connection params from parent, decide how subprocess connects
        elif args.use_mock: # If parent is using mock, subprocess should too
            cmd.append("--use-mock")
        elif args.use_testcontainer: # Else, if parent wants testcontainer for subprocess
             cmd.append("--use-testcontainer")
        # If none of the above, subprocess will use its defaults (likely direct connection or its own testcontainer logic if not overridden)


        logger.info(f"Executing: {' '.join(cmd)}")
        try:
            result = subprocess.run(cmd, capture_output=True, text=True, check=False)
            
            log_file_path = os.path.join(dataset_log_dir, f"benchmark_{dataset}.log")
            with open(log_file_path, 'w') as log_f:
                log_f.write("STDOUT:\n")
                log_f.write(result.stdout)
                if result.stderr:
                    log_f.write("\n\nSTDERR:\n")
                    log_f.write(result.stderr)
            logger.info(f"Full output for {dataset} saved to {log_file_path}")

            output_lines = result.stdout.splitlines()
            result_dir = None
            for line in output_lines:
                if "Results saved to:" in line:
                    result_dir = line.split("Results saved to:")[1].strip()
                    break
            
            if result_dir and os.path.exists(result_dir):
                logger.info(f"Benchmark for {dataset} dataset completed. Results in: {result_dir}")
                dataset_results[dataset] = result_dir
            else:
                logger.error(f"Could not find 'Results saved to:' line or directory for {dataset} dataset. Check logs at {log_file_path}.")
        
        except Exception as e:
            logger.error(f"Error running benchmark for {dataset} dataset: {str(e)}")
    
    return dataset_results

def main():
    banner = """
╔═══════════════════════════════════════════════════════════════╗
║                                                               ║
║   RAG Techniques Benchmark System                             ║
║   ------------------------------                              ║
║                                                               ║
║   Testing all RAG implementations against real PMC data       ║
║   and comparing results against published benchmarks.         ║
║                                                               ║
╚═══════════════════════════════════════════════════════════════╝
"""
    print(banner)
    start_time = time.time()
    args = parse_args()
    logger.info(f"Starting complete RAG benchmarking process. Output directory: {args.output_dir}")
    deps = check_dependencies()
    os.makedirs(args.output_dir, exist_ok=True)

    iris_conn = None
    iris_connection_params = None

    # Setup IRIS Connection
    if args.use_mock:
        logger.info("Using mock IRIS connection as requested.")
        from common.iris_connector import get_mock_iris_connection
        iris_conn = get_mock_iris_connection()
        if not iris_conn:
            logger.error("Failed to create mock IRIS connection.")
            return 1
    elif args.use_manual_docker: # New condition for manual docker setup
        logger.info("Setting up IRIS database using manual Docker container management.")
        from common.iris_connector import setup_docker_test_db # Import the new function
        # Default params for setup_docker_test_db can be used or overridden by more args if needed
        iris_conn = setup_docker_test_db(host_port=args.iris_port if hasattr(args, 'iris_port') else 1972) 
        if not iris_conn:
            logger.error("Failed to set up IRIS via manual Docker. Exiting.")
            return 1
        # Store the container name on the connection if not already done by setup_docker_test_db
        # This is for potential cleanup. setup_docker_test_db now stores _docker_container_name
        
        # Load data and verify (same logic as for testcontainer)
        if not args.skip_verification:
            from common.utils import get_embedding_func, DEFAULT_EMBEDDING_MODEL # Import for data loading
            embedding_model_name_for_load = DEFAULT_EMBEDDING_MODEL if args.llm != "stub" else "stub"
            embedding_func_for_load = get_embedding_func(model_name=embedding_model_name_for_load)

            logger.info(f"Loading {args.document_count} documents into the manually managed Dockerized IRIS...")
            loaded_count = load_test_data(iris_conn, embedding_func_for_load, args.document_count)
            logger.info("Pausing for 3 seconds to allow DB operations to settle...")
            time.sleep(3)
            if not verify_document_count(iris_conn, args.document_count):
                logger.error(f"Document count verification failed for manually managed Dockerized IRIS. Expected {args.document_count}, found 0 or less.")
                # No automatic re-setup here, direct failure.
                if hasattr(iris_conn, '_docker_container_name'):
                    container_to_stop = iris_conn._docker_container_name
                    logger.info(f"Attempting to stop container {container_to_stop} due to verification failure.")
                    subprocess.run(["docker", "stop", container_to_stop], check=False)
                    # --rm was used, so it should be removed on stop.
                return 1
            else:
                logger.info(f"Successfully verified documents in manually managed Dockerized IRIS.")
                
                # Build Knowledge Graph after document loading and verification
                logger.info("Building Knowledge Graph for manually managed Dockerized IRIS...")
                from tests.utils import build_knowledge_graph, load_colbert_token_embeddings
                # Note: build_knowledge_graph needs an embedding_func.
                # We can reuse embedding_func_for_load or get a new one.
                # For KG node/entity embeddings, a sentence-level embedder is fine.
                node_count, edge_count = build_knowledge_graph(iris_conn, embedding_func_for_load, limit=args.document_count)
                logger.info(f"Knowledge Graph built: {node_count} nodes, {edge_count} edges.")
                logger.info("Pausing for 3 seconds to allow KG DB operations to settle...")
                time.sleep(3)

                # Load ColBERT token embeddings if colbert technique is selected
                if 'colbert' in args.techniques:
                    logger.info("Loading ColBERT token embeddings for manually managed Dockerized IRIS...")
                    num_colbert_tokens = load_colbert_token_embeddings(
                        connection=iris_conn,
                        limit=args.document_count,
                        mock_colbert_encoder=args.use_mock # Pass the main mock flag
                    )
                    logger.info(f"Loaded {num_colbert_tokens} ColBERT token embeddings.")
                    logger.info("Pausing for 3 seconds to allow ColBERT DB operations to settle...")
                    time.sleep(3)

        # Extract connection parameters for subprocesses
        # For manual docker, we know the host is localhost and the mapped host_port
        iris_connection_params = {
            "host": "localhost",
            "port": str(args.iris_port if hasattr(args, 'iris_port') else 1972), # The host port we mapped to
            "namespace": "USER", # Default, or make configurable
            "user": "test",      # Default, or make configurable
            "password": "test"   # Default, or make configurable
        }
        logger.info(f"Manually managed Docker IRIS params for subprocess: {iris_connection_params}")

    elif args.use_testcontainer and deps["testcontainers"]:
        logger.info("Setting up IRIS testcontainer.")
        from benchmark_runner_utils import setup_iris_testcontainer # ensure it's imported if not at top
        iris_conn = setup_iris_testcontainer()
        if not iris_conn:
            logger.error("Failed to set up IRIS testcontainer. Exiting.")
            return 1
        
        # Load data into the testcontainer and verify
        if not args.skip_verification:
            from common.utils import get_embedding_func, DEFAULT_EMBEDDING_MODEL # Import for data loading
            embedding_model_name_for_load = DEFAULT_EMBEDDING_MODEL if args.llm != "stub" else "stub"
            embedding_func_for_load = get_embedding_func(model_name=embedding_model_name_for_load)

            logger.info(f"Loading {args.document_count} documents into the testcontainer...")
            loaded_count = load_test_data(iris_conn, embedding_func_for_load, args.document_count) # This will use the per-batch commit
            logger.info("Pausing for 3 seconds to allow DB operations to settle after initial load...")
            time.sleep(3)

            # Simple direct test of insert/commit/select on iris_conn
            logger.info("Performing simple direct DDL & DML test on iris_conn...")
            try:
                with iris_conn.cursor() as cur_ddl:
                    cur_ddl.execute("CREATE TABLE TestTable (id INT PRIMARY KEY, data VARCHAR(50))")
                iris_conn.commit()
                logger.info("Committed CREATE TABLE TestTable.")
                time.sleep(0.5)

                with iris_conn.cursor() as cur_insert:
                    cur_insert.execute("INSERT INTO TestTable (id, data) VALUES (1, 'test data')")
                iris_conn.commit()
                logger.info("Committed INSERT into TestTable.")
                time.sleep(0.5)

                with iris_conn.cursor() as cur_select:
                    cur_select.execute("SELECT COUNT(*) FROM TestTable")
                    count_res = cur_select.fetchone()
                    logger.info(f"Direct SELECT COUNT(*) from TestTable: {count_res[0] if count_res else 'None'}")
                    
                    cur_select.execute("SELECT id, data FROM TestTable WHERE id = 1")
                    row = cur_select.fetchone()
                    logger.info(f"Direct SELECT from TestTable (id=1): {row}")
                    
                    # Also check SourceDocuments count again after these operations
                    cur_select.execute("SELECT COUNT(*) FROM SourceDocuments")
                    source_doc_count_res = cur_select.fetchone()
                    logger.info(f"Direct SELECT COUNT(*) from SourceDocuments after TestTable ops: {source_doc_count_res[0] if source_doc_count_res else 'None'}")

            except Exception as e_simple_test:
                logger.error(f"Error during simple direct DDL/DML test: {e_simple_test}")

            # Direct check within main script before calling verify_document_count
            try:
                with iris_conn.cursor() as direct_cursor:
                    direct_cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
                    direct_count_result = direct_cursor.fetchone()
                    direct_db_count = int(direct_count_result[0]) if direct_count_result else -1
                    logger.info(f"Direct count check on iris_conn in main (after simple test): {direct_db_count} documents.")
            except Exception as e_direct_check:
                logger.error(f"Error during direct count check in main: {e_direct_check}")

            # Attempt verification with a new connection to the same container
            logger.info("Attempting verification with a new connection to the same running container.")
            verification_passed = False
            temp_conn_params_for_verify = None
            if hasattr(iris_conn, '_container') and iris_conn._container:
                try:
                    container_for_verify = iris_conn._container
                    temp_conn_params_for_verify = {
                        "host": container_for_verify.get_container_host_ip(),
                        "port": int(container_for_verify.get_exposed_port(container_for_verify.port)),
                        "namespace": container_for_verify.namespace,
                        "user": container_for_verify.username,
                        "password": container_for_verify.password
                    }
                    logger.info(f"Params for verification connection: {temp_conn_params_for_verify}")
                    
                    from common.iris_connector import get_real_iris_connection
                    verify_conn = get_real_iris_connection(config=temp_conn_params_for_verify)
                    
                    if verify_conn:
                        logger.info("Successfully created a new connection for verification.")
                        if verify_document_count(verify_conn, args.document_count):
                            logger.info(f"✅ Successfully verified {args.document_count} documents using a new connection.")
                            verification_passed = True
                        else:
                            logger.error(f"⚠️ Verification FAILED using a new connection. Expected at least {args.document_count}.")
                        verify_conn.close()
                    else:
                        logger.error("Failed to create a new connection for verification.")
                except Exception as e_verify_new_conn:
                    logger.error(f"Error during new connection verification attempt: {e_verify_new_conn}")
            else:
                logger.warning("Cannot attempt verification with new connection: _container attribute missing or None.")

            if not verification_passed:
                logger.warning("Verification with new connection failed or was not possible. Trying with original connection.")
                if not verify_document_count(iris_conn, args.document_count):
                    logger.error(f"Initial document count verification failed with original connection. Expected at least {args.document_count}. Loaded: {loaded_count}")
                else:
                    logger.info(f"Successfully verified {loaded_count} (expected >={args.document_count}) documents in the container with original connection.")
                    verification_passed = True # Mark as passed if original connection works

            if not verification_passed:
                 logger.error(f"DOCUMENT VERIFICATION FAILED. ABORTING BENCHMARK. Loaded: {loaded_count}, Expected: {args.document_count}")
                 if iris_conn and hasattr(iris_conn, 'close'): iris_conn.close()
                 return 1 # Exit if verification fails
            else: # Verification passed
                # Build Knowledge Graph after document loading and verification
                logger.info("Building Knowledge Graph for testcontainer IRIS...")
                from tests.utils import build_knowledge_graph, load_colbert_token_embeddings
                node_count, edge_count = build_knowledge_graph(iris_conn, embedding_func_for_load, limit=args.document_count)
                logger.info(f"Knowledge Graph built: {node_count} nodes, {edge_count} edges.")
                logger.info("Pausing for 3 seconds to allow KG DB operations to settle...")
                time.sleep(3)

                # Load ColBERT token embeddings if colbert technique is selected
                if 'colbert' in args.techniques:
                    logger.info("Loading ColBERT token embeddings for testcontainer IRIS...")
                    num_colbert_tokens = load_colbert_token_embeddings(
                        connection=iris_conn,
                        limit=args.document_count,
                        mock_colbert_encoder=args.use_mock # Pass the main mock flag
                    )
                    logger.info(f"Loaded {num_colbert_tokens} ColBERT token embeddings.")
                    logger.info("Pausing for 3 seconds to allow ColBERT DB operations to settle...")
                    time.sleep(3)

        # Extract connection parameters for subprocesses from the current iris_conn
        if hasattr(iris_conn, '_container'):
            container = iris_conn._container
            if container:
                try:
                    iris_connection_params = {
                        "host": container.get_container_host_ip(),
                        "port": str(container.get_exposed_port(container.port)), # Ensure port is string for subprocess
                        "namespace": container.namespace,
                        "user": container.username,
                        "password": container.password
                    }
                    logger.info(f"Testcontainer params for subprocess: {iris_connection_params}")
                except Exception as e:
                    logger.error(f"Error getting connection parameters from container: {e}")
                    iris_connection_params = None 
            else:
                logger.warning("iris_conn has _container attribute but it is None.")
                iris_connection_params = None
        else:
            logger.warning("Could not extract connection parameters from testcontainer object (no _container attribute). Subprocesses might fail to connect or will create their own testcontainer.")
            iris_connection_params = None

    else: # Direct connection
        logger.info("Using direct IRIS connection (or testcontainer setup failed/skipped).")
        from common.iris_connector import get_real_iris_connection
        iris_conn = get_real_iris_connection()
        if not iris_conn:
            logger.error("Failed to establish direct connection to IRIS. Exiting.")
            return 1
        if not args.skip_verification: # Verify count for direct connection
            if not verify_document_count(iris_conn, args.document_count): # Use MIN_DOCUMENT_COUNT for real DB if not specified
                logger.error(f"Insufficient documents in direct IRIS. Expected at least {args.document_count if args.document_count else MIN_DOCUMENT_COUNT}.")
                return 1
    
    if not iris_conn : # If connection failed and not using mock (mock case handled above)
        logger.error("IRIS connection could not be established. Exiting.")
        return 1

    # Run benchmarks using the established and potentially populated iris_conn
    dataset_results = run_benchmarks(args, iris_connection_params) # Pass params to subprocess
    
    if not dataset_results:
        logger.error("No benchmark results were generated. Exiting.")
    else:
        run_comparison_analysis(dataset_results, args.output_dir)
        create_summary_report(dataset_results, args.output_dir)

    end_time = time.time()
    duration = end_time - start_time
    minutes, seconds = divmod(duration, 60)
    logger.info(f"Complete benchmarking process finished in {int(minutes)}m {seconds:.1f}s")
    logger.info(f"Output directory: {args.output_dir}")

    if iris_conn and hasattr(iris_conn, 'close'):
        try:
            iris_conn.close()
            logger.info("IRIS connection closed.")
        except Exception as e:
            logger.warning(f"Error closing IRIS connection: {e}")
    
    return 0 if dataset_results else 1

if __name__ == "__main__":
    sys.exit(main())
