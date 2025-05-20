#!/usr/bin/env python3
# benchmark_runner_utils.py
# Utility functions for the main benchmark runner script.

import os
import sys
import logging
import subprocess
import time
from datetime import datetime
from typing import Callable # Added import

logger = logging.getLogger("benchmark_runner_utils")

# Define constants used by parse_args if not already defined globally or passed
DEFAULT_TECHNIQUES = ["basic_rag", "hyde", "colbert", "crag", "noderag", "graphrag"]
DEFAULT_DATASETS = ["medical", "multihop"]
BENCHMARK_DIR = "benchmark_results"

def parse_args():
    """Parse command line arguments for the main benchmark runner."""
    import argparse # Ensure argparse is imported here
    parser = argparse.ArgumentParser(description="Run complete RAG benchmarking process")
    
    parser.add_argument("--use-testcontainer", action="store_true", default=True,
                       help="Use testcontainer instead of direct IRIS connection (default: True)")
    
    parser.add_argument("--use-mock", action="store_true", 
                       help="Use mock IRIS connection instead of real connection")
    
    parser.add_argument("--use-manual-docker", action="store_true",
                        help="Use manually managed Docker container for IRIS instead of testcontainers library")

    parser.add_argument("--iris-port", type=int, default=1972,
                        help="Host port to map to IRIS container, or port of existing IRIS instance (default: 1972)")
    
    parser.add_argument("--techniques", nargs="+", default=DEFAULT_TECHNIQUES,
                       help=f"RAG techniques to benchmark (default: {' '.join(DEFAULT_TECHNIQUES)})")
    
    parser.add_argument("--datasets", nargs="+", default=DEFAULT_DATASETS,
                       help=f"Query datasets to use (default: {' '.join(DEFAULT_DATASETS)})")
    
    parser.add_argument("--llm", choices=["gpt-3.5-turbo", "gpt-4", "stub"], default="stub",
                       help="LLM model to use (default: stub)")
    
    parser.add_argument("--query-limit", type=int, default=5,
                       help="Maximum number of queries to run per dataset (default: 5)")
    
    parser.add_argument("--skip-verification", action="store_true",
                       help="Skip IRIS database verification and data loading step")
    
    parser.add_argument("--document-count", type=int, default=50,
                        help="Number of PMC documents to load if using testcontainer (default: 50)")
    
    parser.add_argument("--output-dir", type=str, 
                        default=f"{BENCHMARK_DIR}/full_benchmark_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                        help="Directory to store all benchmark results")
    
    return parser.parse_args()

def check_dependencies():
    """
    Check for required dependencies and warn about missing ones.
    
    Returns:
        Dict of dependency status (True if available, False if missing)
    """
    dependencies = {
        "matplotlib": False,
        "testcontainers": False,
        "intersystems_iris": False
    }
    
    try:
        import matplotlib
        dependencies["matplotlib"] = True
        logger.info("√ Matplotlib is available for visualizations")
    except ImportError:
        logger.warning("✗ Matplotlib is not installed. Visualizations will not be generated.")
        logger.warning("  Install with: pip install matplotlib")
    
    try:
        from testcontainers.iris import IRISContainer
        dependencies["testcontainers"] = True
        logger.info("√ Testcontainers is available for IRIS database testing")
    except ImportError:
        logger.warning("✗ Testcontainers is not installed. Will use direct connection or mock.")
        logger.warning("  Install with: pip install testcontainers-iris")
    
    try:
        import intersystems_iris
        dependencies["intersystems_iris"] = True
        logger.info("√ InterSystems IRIS module is available for database connection")
    except ImportError:
        logger.warning("✗ InterSystems IRIS module is not installed. Direct connection will not work.")
        logger.warning("  Install with: pip install intersystems-iris-native")
    
    return dependencies

def setup_iris_testcontainer():
    """
    Set up an IRIS testcontainer and return connection.
    
    Returns:
        Connection to IRIS testcontainer, or None if setup fails
    """
    logger.info("Setting up IRIS testcontainer...")
    os.environ["TEST_IRIS"] = "true" # Ensure iris_connector uses testcontainer logic
    
    try:
        from common.iris_connector import get_iris_connection
        logger.info("Creating testcontainer via iris_connector...")
        connection = get_iris_connection(use_mock=False, use_testcontainer=True)
        
        if connection is None:
            logger.error("Failed to create IRIS testcontainer connection.")
            return None
        
        logger.info("Testing testcontainer connection...")
        with connection.cursor() as cursor:
            cursor.execute("SELECT 1")
            cursor.fetchone()
        logger.info("Connection test successful")
            
        logger.info("Initializing database schema in testcontainer...")
        from common.db_init import initialize_database
        initialize_database(connection, force_recreate=True)
        logger.info("Database schema initialized successfully in testcontainer.")
        
        return connection
        
    except Exception as e:
        logger.error(f"Error setting up IRIS testcontainer: {e}")
        return None

def load_test_data(connection, embedding_func: Callable, document_count=50):
    """
    Load test data into IRIS database.
    """
    logger.info(f"Loading {document_count} PMC documents into database...")
    try:
        from tests.utils import load_pmc_documents
        doc_count = load_pmc_documents(
            connection=connection,
            embedding_func=embedding_func, # Pass embedding_func
            limit=document_count,
            pmc_dir="data/pmc_oas_downloaded", 
            batch_size=20, 
            show_progress=True
        )
        logger.info(f"Successfully loaded {doc_count} PMC documents.")
        return doc_count
    except Exception as e:
        logger.error(f"Error loading test data: {e}")
        return 0

def verify_document_count(connection, expected_count):
    """
    Verify that the database contains at least expected_count documents.
    """
    logger.info(f"Verifying document count (expected at least: {expected_count})...")
    try:
        with connection.cursor() as cursor:
            cursor.execute("SELECT COUNT(*) FROM SourceDocuments")
            count_result = cursor.fetchone()
            current_count = int(count_result[0]) if count_result else 0
            logger.info(f"Current document count in DB: {current_count}")
            if current_count >= expected_count:
                logger.info(f"✅ Document count verification passed: {current_count} ≥ {expected_count}")
                return True
            else:
                logger.warning(f"⚠️ Document count verification failed: {current_count} < {expected_count}")
                return False
    except Exception as e:
        logger.error(f"Error verifying document count: {e}")
        return False

def run_comparison_analysis(dataset_results, output_dir):
    logger.info("Step 3: Generating comparative visualizations...")
    comparison_dir = os.path.join(output_dir, "comparison_reports")
    os.makedirs(comparison_dir, exist_ok=True)

    try:
        import matplotlib
    except ImportError:
        logger.warning("Matplotlib not available. Skipping visualization generation.")
        with open(os.path.join(comparison_dir, "visualization_skipped.txt"), 'w') as f:
            f.write("Visualization generation skipped: matplotlib not installed.")
        return

    logger.info("Running demo_benchmark_analysis.py for overall visualization generation...")
    cmd = [sys.executable, "demo_benchmark_analysis.py"] 
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=False)
        log_file_path = os.path.join(comparison_dir, "demo_benchmark_analysis.log")
        with open(log_file_path, 'w') as log_f:
            log_f.write("STDOUT:\n")
            log_f.write(result.stdout)
            if result.stderr:
                log_f.write("\n\nSTDERR:\n")
                log_f.write(result.stderr)
        logger.info(f"demo_benchmark_analysis.py output saved to {log_file_path}")
    except Exception as e:
        logger.error(f"Error running demo_benchmark_analysis.py: {str(e)}")

def create_summary_report(dataset_results, output_dir):
    logger.info("Creating benchmark summary report...")
    summary_file = os.path.join(output_dir, "benchmark_summary.md")
    with open(summary_file, 'w') as f:
        f.write("# RAG Techniques Benchmark Summary\n\n")
        f.write(f"Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
        f.write("## Datasets Benchmarked\n\n")
        for dataset in dataset_results:
            f.write(f"- **{dataset}**\n")
        f.write("\n")
        f.write("## Benchmark Results\n\n")
        for dataset, result_dir in dataset_results.items():
            f.write(f"### {dataset.capitalize()} Dataset\n\n")
            # Construct path to the specific dataset's report
            # Assuming run_benchmark_demo.py saves its report in output_dir/dataset/reports/benchmark_report.md
            # The result_dir from run_benchmarks is output_dir/dataset/benchmark_datetime
            # So, the report would be result_dir/reports/benchmark_report.md
            report_file = os.path.join(result_dir, "reports", "benchmark_report.md") 
            
            if os.path.exists(report_file):
                # Make the link relative to the main output_dir for the summary report
                relative_report_path = os.path.relpath(report_file, output_dir)
                f.write(f"Detailed report: [{relative_report_path}]({relative_report_path})\n\n")
                try:
                    with open(report_file, 'r') as report_f:
                        content = report_f.read()
                        best_section_start = content.find("## Best Performing Techniques")
                        if best_section_start != -1:
                            best_section_end = content.find("##", best_section_start + len("## Best Performing Techniques"))
                            best_section_end = best_section_end if best_section_end != -1 else len(content)
                            f.write(content[best_section_start:best_section_end].strip() + "\n\n")
                except Exception as e_read:
                    f.write(f"Could not read content from {report_file}: {e_read}\n\n")
            else:
                f.write(f"Detailed report not found at {report_file}.\n\n")
        f.write("## Next Steps\n\n")
        f.write("1. Review the detailed reports for each dataset.\n")
        f.write("2. Compare results with published benchmarks.\n")
        f.write("3. Identify optimization opportunities.\n")
    logger.info(f"Summary report created: {summary_file}")
    return summary_file
