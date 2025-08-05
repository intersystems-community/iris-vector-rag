#!/usr/bin/env python3
"""
Master Zero-to-RAGAS Demonstration Script

This script orchestrates the complete RAG pipeline from database clearing
to RAGAS evaluation, providing detailed visibility into each step.
It leverages existing Makefile targets and Python utilities.
"""

import os
import sys
import json
import time
import logging
import argparse
import subprocess
from datetime import datetime
from pathlib import Path
from typing import Dict, Any, Optional, Tuple

# Add project root to path to allow importing project modules
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

# Attempt to import common modules, with fallbacks if not found
try:
    from common.iris_connection_manager import get_iris_connection, test_connection
except ImportError:
    print("Warning: Could not import from common.iris_connection_manager. Database state checks might be limited.")
    get_iris_connection = None
    test_connection = None

try:
    # This is a placeholder, actual RAGAS result parsing might need specific utility
    # from common.utils import parse_ragas_json_output
    pass # Replace with actual import if a utility exists
except ImportError:
    print("Warning: RAGAS parsing utility not found. Will attempt basic JSON parsing.")
    # Define a simple fallback parser if needed
    def parse_ragas_json_output(json_string: str) -> Dict[str, Any]:
        return json.loads(json_string)


# --- Configuration ---
DEFAULT_OUTPUT_DIR = project_root / "outputs" / "zero_to_ragas_demo"
DEFAULT_LOG_FILE = DEFAULT_OUTPUT_DIR / f"zero_to_ragas_demo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
RAGAS_RESULTS_FILE_PATTERN = "ragas_*.json" # Adjust if RAGAS output has a specific name

# --- Logger Setup ---
logger = logging.getLogger("ZeroToRAGASDemo")

def setup_logging(verbose: bool, log_file: Path = DEFAULT_LOG_FILE):
    """Configure logging for the script."""
    log_level = logging.DEBUG if verbose else logging.INFO
    log_file.parent.mkdir(parents=True, exist_ok=True)

    # Remove all handlers associated with the root logger object.
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Remove all handlers associated with this specific logger
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s [%(levelname)-8s] %(name)s: %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout)
        ]
    )
    logger.propagate = False # Prevent duplicate logging if root logger also configured

    # Set higher level for noisy libraries if needed
    logging.getLogger("matplotlib").setLevel(logging.WARNING)
    logging.getLogger("PIL").setLevel(logging.WARNING)

    logger.info(f"Logging initialized. Level: {logging.getLevelName(log_level)}. Log file: {log_file}")


# --- Helper Functions ---
def run_makefile_target(target: str, verbose: bool, capture_output: bool = False) -> Tuple[bool, str, str]:
    """
    Run a Makefile target using subprocess.

    Args:
        target: The Makefile target to run (e.g., "clear-rag-data").
        verbose: If True, show detailed command output.
        capture_output: If True, capture stdout and stderr.

    Returns:
        A tuple (success: bool, stdout: str, stderr: str).
    """
    command = ["make", target]
    logger.info(f"🚀 Executing Makefile target: '{' '.join(command)}'")
    start_time = time.time()

    try:
        process = subprocess.Popen(
            command,
            stdout=subprocess.PIPE if capture_output or not verbose else sys.stdout,
            stderr=subprocess.PIPE if capture_output or not verbose else sys.stderr,
            text=True,
            cwd=project_root
        )
        stdout, stderr = process.communicate()
        return_code = process.returncode
        
        duration = time.time() - start_time
        
        if return_code == 0:
            logger.info(f"✅ Makefile target '{target}' completed successfully in {duration:.2f}s.")
            if verbose and capture_output and stdout:
                 logger.debug(f"Stdout for '{target}':\n{stdout}")
            return True, stdout or "", stderr or ""
        else:
            logger.error(f"❌ Makefile target '{target}' failed with code {return_code} after {duration:.2f}s.")
            if stdout: logger.error(f"Stdout:\n{stdout}")
            if stderr: logger.error(f"Stderr:\n{stderr}")
            return False, stdout or "", stderr or ""
            
    except FileNotFoundError:
        logger.error("❌ 'make' command not found. Please ensure Make is installed and in your PATH.")
        return False, "", "Make command not found"
    except Exception as e:
        duration = time.time() - start_time
        logger.error(f"❌ Exception running Makefile target '{target}' after {duration:.2f}s: {e}")
        return False, "", str(e)

def show_database_state(step_name: str):
    """Show current database state with document counts."""
    logger.info(f"📊 Database State Check after: {step_name}")
    if not get_iris_connection or not test_connection:
        logger.warning("Database utilities not available. Skipping database state check.")
        return

    if not test_connection():
        logger.error("❌ Cannot connect to IRIS database. Skipping state check.")
        return

    try:
        with get_iris_connection() as conn:
            with conn.cursor() as cursor:
                # Check SourceDocuments
                cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
                source_doc_count = cursor.fetchone()[0]
                logger.info(f"   - SourceDocuments count: {source_doc_count}")

                # Check DocumentChunks
                cursor.execute("SELECT COUNT(*) FROM RAG.DocumentChunks")
                chunk_count = cursor.fetchone()[0]
                logger.info(f"   - DocumentChunks count: {chunk_count}")

                # Sample SourceDocument
                if source_doc_count > 0:
                    cursor.execute("SELECT TOP 1 DocumentID, SourceName FROM RAG.SourceDocuments")
                    sample_doc = cursor.fetchone()
                    logger.info(f"   - Sample SourceDocument: ID={sample_doc[0]}, Name='{sample_doc[1]}'")
                
                # Sample DocumentChunk
                if chunk_count > 0:
                    cursor.execute("SELECT TOP 1 ChunkID, SourceDocumentID, LENGTH(ChunkText) FROM RAG.DocumentChunks")
                    sample_chunk = cursor.fetchone()
                    logger.info(f"   - Sample DocumentChunk: ID={sample_chunk[0]}, SourceDocID={sample_chunk[1]}, TextLength={sample_chunk[2]}")

    except Exception as e:
        logger.error(f"❌ Error checking database state: {e}")

def show_embedding_stats():
    """Show embedding generation statistics (placeholder)."""
    # This would ideally query the database for vector counts or analyze logs
    logger.info("ℹ️  Embedding Statistics (Illustrative):")
    if get_iris_connection and test_connection and test_connection():
        try:
            with get_iris_connection() as conn:
                with conn.cursor() as cursor:
                    cursor.execute("SELECT COUNT(*) FROM RAG.DocumentChunks WHERE Embedding IS NOT NULL")
                    embedded_chunks = cursor.fetchone()[0]
                    logger.info(f"   - Number of chunks with embeddings: {embedded_chunks}")
                    # Could also check embedding dimensions if schema supports it
                    # cursor.execute("SELECT TOP 1 VECTOR_DIMENSION(Embedding) FROM RAG.DocumentChunks WHERE Embedding IS NOT NULL")
                    # dim = cursor.fetchone()
                    # if dim: logger.info(f"   - Embedding dimension: {dim[0]}")

        except Exception as e:
            logger.warning(f"Could not retrieve embedding stats from DB: {e}")
    else:
        logger.info("   - (Database connection not available for detailed stats)")
    logger.info("   - (Further stats would require specific logging or DB queries)")


def parse_and_display_ragas_results(ragas_output_dir: Path, verbose: bool):
    """Parse and display RAGAS evaluation results from JSON files."""
    logger.info("📊 Parsing RAGAS Evaluation Results...")
    
    ragas_files = list(ragas_output_dir.glob(RAGAS_RESULTS_FILE_PATTERN))
    if not ragas_files:
        logger.warning(f"No RAGAS result files found in {ragas_output_dir} matching '{RAGAS_RESULTS_FILE_PATTERN}'.")
        return

    latest_ragas_file = max(ragas_files, key=os.path.getctime)
    logger.info(f"Found RAGAS result file: {latest_ragas_file}")

    try:
        with open(latest_ragas_file, 'r') as f:
            content = f.read()
            if not content.strip():
                logger.error(f"RAGAS result file {latest_ragas_file} is empty.")
                return
            results_data = json.loads(content) # Using basic json.loads

        logger.info("✅ RAGAS Results Parsed Successfully.")

        # Display key metrics (customize based on actual RAGAS output structure)
        # This is a generic example; actual parsing depends on RAGAS output format.
        if isinstance(results_data, dict):
            for pipeline, metrics in results_data.items():
                if isinstance(metrics, dict): # Assuming one level of nesting for pipeline results
                    logger.info(f"  Pipeline: {pipeline}")
                    for metric_name, value in metrics.items():
                        if isinstance(value, (float, int)):
                            logger.info(f"    - {metric_name}: {value:.4f}")
                        elif isinstance(value, dict) and 'mean' in value : # Handle cases like RAGAS outputting dicts with 'mean'
                             logger.info(f"    - {metric_name} (mean): {value['mean']:.4f}")
                        else:
                            if verbose: logger.info(f"    - {metric_name}: {value}")
                else: # If results_data is flat or has unexpected structure
                     logger.info(f"  Metric: {pipeline}, Value: {metrics}")


        elif isinstance(results_data, list): # If RAGAS output is a list of results
            for i, item in enumerate(results_data):
                logger.info(f"  Result Set {i+1}:")
                if isinstance(item, dict):
                    for key, value in item.items():
                        if isinstance(value, (float, int)):
                            logger.info(f"    - {key}: {value:.4f}")
                        else:
                             if verbose: logger.info(f"    - {key}: {value}")
                else:
                    if verbose: logger.info(f"    - {item}")
        else:
            logger.warning("RAGAS results format not a dictionary or list. Displaying raw if verbose.")
            if verbose:
                logger.info(f"Raw RAGAS Data:\n{json.dumps(results_data, indent=2)}")


    except json.JSONDecodeError as e:
        logger.error(f"❌ Error decoding RAGAS JSON from {latest_ragas_file}: {e}")
        logger.error("Please ensure the RAGAS evaluation script outputs valid JSON.")
        if verbose:
            try:
                with open(latest_ragas_file, 'r') as f_err:
                    logger.debug(f"Content of problematic file:\n{f_err.read()}")
            except Exception as read_err:
                logger.debug(f"Could not read problematic file content: {read_err}")

    except Exception as e:
        logger.error(f"❌ Error processing RAGAS results from {latest_ragas_file}: {e}")


# --- Main Orchestration ---
def main(verbose: bool, quick_run: bool, output_dir: Path):
    """Main orchestration function for the zero-to-RAGAS demo."""
    
    setup_logging(verbose, output_dir / f"demo_run_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    logger.info("🌟 Starting Ultimate Zero-to-RAGAS Demonstration Script 🌟")
    logger.info(f"Verbose mode: {'Enabled' if verbose else 'Disabled'}")
    logger.info(f"Quick run: {'Enabled (fewer docs/iterations)' if quick_run else 'Disabled (full run)'}")
    logger.info(f"Output directory: {output_dir}")
    
    overall_success = True
    final_report_data = {
        "start_time": datetime.now().isoformat(),
        "steps": []
    }

    def record_step(name: str, success: bool, details: Optional[str] = None, error_msg: Optional[str] = None):
        nonlocal overall_success
        if not success:
            overall_success = False
        step_data = {"name": name, "status": "SUCCESS" if success else "FAILURE"}
        if details: step_data["details"] = details
        if error_msg: step_data["error"] = error_msg
        final_report_data["steps"].append(step_data)
        if not success:
            logger.error(f"Critical failure in step: {name}. Aborting further Makefile-dependent steps if applicable.")
            # Depending on severity, you might choose to exit or skip subsequent steps
            # For this demo, we'll log and continue to show as much as possible,
            # but mark overall_success as False.

    # --- Step 1: Database Clearing ---
    logger.info("\n--- STEP 1: Database Clearing ---")
    success, _, err_msg = run_makefile_target("clear-rag-data", verbose)
    record_step("Database Clearing", success, error_msg=err_msg if not success else None)
    if success: show_database_state("Database Clearing")
    else: logger.warning("Skipping DB state check due to clearing failure.")


    # --- Step 2: Document Loading ---
    if overall_success: # Only proceed if previous critical steps were okay
        logger.info("\n--- STEP 2: Document Loading ---")
        load_target = "load-data" if quick_run else "load-1000" # Use smaller dataset for quick run
        logger.info(f"Using Makefile target for loading: '{load_target}'")
        success, _, err_msg = run_makefile_target(load_target, verbose)
        record_step("Document Loading", success, details=f"Target: {load_target}", error_msg=err_msg if not success else None)
        if success: show_database_state("Document Loading")
        else: logger.warning("Skipping DB state check due to loading failure.")
    else:
        record_step("Document Loading", False, details="Skipped due to previous errors.")


    # --- Step 3: Data Verification & Embedding (Conceptual) ---
    # Actual embedding is part of pipeline setup, this is more of a check/conceptual step
    if overall_success:
        logger.info("\n--- STEP 3: Data Verification & Embedding Status ---")
        # This step is more conceptual in this script as embeddings are usually part of pipeline setup.
        # We'll check database state again and show illustrative embedding stats.
        show_database_state("Data Verification")
        # The `auto-setup-all` or specific pipeline setup targets in Makefile handle embeddings.
        # We can call one of those here, or assume it's part of RAG pipeline execution.
        # For simplicity, we'll assume `make auto-setup-all` prepares embeddings.
        logger.info("Running 'auto-setup-all' to ensure embeddings and pipeline readiness...")
        success_setup, _, err_msg_setup = run_makefile_target("auto-setup-all", verbose) # This can take time
        record_step("Pipeline Auto-Setup (includes embedding)", success_setup, error_msg=err_msg_setup if not success_setup else None)
        if success_setup:
            show_embedding_stats()
            show_database_state("Pipeline Auto-Setup")
        else:
            logger.warning("Skipping embedding/DB state check due to auto-setup failure.")
    else:
        record_step("Data Verification & Embedding Status", False, details="Skipped due to previous errors.")

    # --- Step 4: RAG Pipeline Execution (Example Query) ---
    # This is illustrative. A real demo might run a specific pipeline script.
    # Here, we'll use a Makefile target that implies a query, e.g., a test target.
    if overall_success:
        logger.info("\n--- STEP 4: RAG Pipeline Execution (Illustrative) ---")
        # We'll use 'test-pipeline PIPELINE=basic' as an example of running a RAG query.
        # This target should ideally output some results or logs.
        # For a real demo, you might have a dedicated script: `python scripts/run_sample_query.py basic "What is AI?"`
        logger.info("Illustrating RAG pipeline execution by running a test for 'basic' pipeline.")
        success_rag, stdout_rag, stderr_rag = run_makefile_target("test-pipeline PIPELINE=basic", verbose, capture_output=True)
        details_rag = "Ran 'make test-pipeline PIPELINE=basic'."
        if stdout_rag: details_rag += f"\nStdout:\n{stdout_rag[:500]}{'...' if len(stdout_rag) > 500 else ''}" # Show partial output
        record_step("RAG Pipeline Execution (Basic)", success_rag, details=details_rag, error_msg=stderr_rag if not success_rag else None)
        if success_rag:
            logger.info("Sample RAG execution completed. Check logs for output from 'test-pipeline'.")
        # Add more examples for other pipelines if needed
        # success_colbert, _, _ = run_makefile_target("test-pipeline PIPELINE=colbert", verbose)
        # record_step("RAG Pipeline Execution (ColBERT)", success_colbert)
    else:
        record_step("RAG Pipeline Execution", False, details="Skipped due to previous errors.")


    # --- Step 5: RAGAS Evaluation ---
    if overall_success:
        logger.info("\n--- STEP 5: RAGAS Evaluation ---")
        # Choose a RAGAS target. 'ragas-debug' is quicker. 'ragas-full' or 'eval-all-ragas-1000' are more comprehensive.
        ragas_target = "ragas-debug" if quick_run else "eval-all-ragas-1000" # or "ragas-full"
        logger.info(f"Using Makefile target for RAGAS: '{ragas_target}'")
        # RAGAS evaluation scripts should output JSON results to a known location or stdout.
        # For this demo, we assume it outputs to project_root / "outputs" / "reports" / "ragas_evaluations" / "runs" / ...
        # Or, if it prints JSON to stdout, we can capture it.
        # The `eval-all-ragas-1000` target in Makefile redirects output, so direct capture might not work.
        # We will rely on it creating files in a standard location.
        
        # Define where RAGAS results are expected (adjust if Makefile changes this)
        # This path is based on `eval-all-ragas-1000` target's redirection.
        # If using `ragas-debug` or `ragas-full` from the newer `scripts/run_ragas_evaluation.py`,
        # the output path might be different (e.g., `outputs/ragas_evaluations/runs/...`)
        
        # Let's assume the newer script's output path for flexibility
        ragas_output_location = project_root / "outputs" / "reports" / "ragas_evaluations" / "runs"
        if ragas_target == "eval-all-ragas-1000": # Older script might output elsewhere
             ragas_output_location = project_root / "comprehensive_ragas_results" # As per Makefile
        
        logger.info(f"Expecting RAGAS results in directory: {ragas_output_location}")
        ragas_output_location.mkdir(parents=True, exist_ok=True) # Ensure dir exists

        success_ragas, stdout_ragas, stderr_ragas = run_makefile_target(ragas_target, verbose, capture_output=True)
        record_step("RAGAS Evaluation", success_ragas, details=f"Target: {ragas_target}", error_msg=stderr_ragas if not success_ragas else None)
        
        if success_ragas:
            # If RAGAS script prints JSON to stdout and we captured it:
            if stdout_ragas and stdout_ragas.strip().startswith("{") and stdout_ragas.strip().endswith("}"):
                logger.info("Attempting to parse RAGAS JSON from stdout...")
                # Create a dummy file from stdout to parse
                dummy_ragas_file = output_dir / f"ragas_stdout_results_{int(time.time())}.json"
                with open(dummy_ragas_file, "w") as f:
                    f.write(stdout_ragas)
                parse_and_display_ragas_results(output_dir, verbose) # Parse from where we saved it
            else: # Otherwise, parse from the expected file location
                parse_and_display_ragas_results(ragas_output_location, verbose)
        else:
            logger.error("RAGAS evaluation failed. Skipping results parsing.")
            if stdout_ragas: logger.info(f"RAGAS stdout (if any):\n{stdout_ragas}")

    else:
        record_step("RAGAS Evaluation", False, details="Skipped due to previous errors.")

    # --- Step 6: Results Analysis and Reporting (Summarized by this script's logs) ---
    logger.info("\n--- STEP 6: Results Analysis and Reporting ---")
    logger.info("Comprehensive logging throughout this script serves as the detailed report.")
    logger.info(f"Final log file: {DEFAULT_LOG_FILE if DEFAULT_LOG_FILE.exists() else 'Check configured log path'}")
    
    final_report_data["end_time"] = datetime.now().isoformat()
    final_report_data["overall_status"] = "SUCCESS" if overall_success else "FAILURE"
    
    report_file_path = output_dir / f"master_demo_summary_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    try:
        with open(report_file_path, "w") as f:
            json.dump(final_report_data, f, indent=2)
        logger.info(f"📄 Summary report saved to: {report_file_path}")
    except Exception as e:
        logger.error(f"Failed to save summary report: {e}")

    if overall_success:
        logger.info("\n🎉🎉🎉 Zero-to-RAGAS Demonstration Completed Successfully! 🎉🎉🎉")
    else:
        logger.error("\n❌❌❌ Zero-to-RAGAS Demonstration Completed with Failures. ❌❌❌")
        logger.error("Please review the logs for details on failed steps.")

    sys.exit(0 if overall_success else 1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Master Zero-to-RAGAS Demonstration Script.")
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging and command output."
    )
    parser.add_argument(
        "--quick",
        action="store_true",
        help="Run a quicker version of the demo (e.g., fewer documents, fewer RAGAS iterations)."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=str(DEFAULT_OUTPUT_DIR),
        help=f"Directory to store logs and reports (default: {DEFAULT_OUTPUT_DIR})."
    )
    
    args = parser.parse_args()
    
    # Ensure output directory is a Path object
    output_dir_path = Path(args.output_dir)
    
    main(verbose=args.verbose, quick_run=args.quick, output_dir=output_dir_path)