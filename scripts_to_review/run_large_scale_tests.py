#!/usr/bin/env python
"""
Run large-scale tests with real data and real embeddings on 1000+ documents.

This script facilitates testing with large document collections by:
1. Efficiently loading and processing PMC documents in batches
2. Using testcontainers for database isolation
3. Collecting detailed performance metrics
4. Supporting real embedding models for accurate comparison

Usage examples:
    # Run with 1000 documents, all techniques
    python run_large_scale_tests.py --document-count 1000 --techniques all
    
    # Run only GraphRAG with 1000 documents
    python run_large_scale_tests.py --document-count 1000 --techniques graphrag
    
    # Run with real embeddings (default) and detailed reporting
    python run_large_scale_tests.py --document-count 1000 --verbose --report-file results.json
"""

import os
import sys
import time
import json
import logging
import argparse
import subprocess
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

def parse_args():
    """Parse command-line arguments"""
    parser = argparse.ArgumentParser(description="Run large-scale RAG tests with real data")
    
    parser.add_argument(
        "--techniques", 
        nargs="+", 
        choices=["all", "graphrag", "colbert", "hyde", "noderag", "crag", "basic"],
        default=["all"],
        help="RAG techniques to test"
    )
    
    parser.add_argument(
        "--document-count",
        type=int,
        default=1000,
        help="Total number of documents to process (default: 1000)"
    )
    
    parser.add_argument(
        "--batch-size",
        type=int,
        default=50,
        help="Batch size for document processing (default: 50)"
    )
    
    parser.add_argument(
        "--use-real-embeddings",
        action="store_true",
        default=True,  # Default to real embeddings for accurate results
        help="Use real embedding model (default: True)"
    )
    
    parser.add_argument(
        "--use-mock-embeddings",
        action="store_true",
        help="Use mock embedding model instead of real embeddings"
    )
    
    parser.add_argument(
        "--no-testcontainer",
        action="store_true",
        help="Don't use testcontainer (use real or mock connections instead)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Run with verbose output"
    )
    
    parser.add_argument(
        "--report-file",
        type=str,
        default="large_scale_results.json",
        help="File to save test results (default: large_scale_results.json)"
    )
    
    parser.add_argument(
        "--memory-limit",
        type=int,
        default=None,
        help="Memory limit for testcontainer in MB (default: auto)"
    )
    
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip document loading if documents already exist in database"
    )
    
    return parser.parse_args()

def ensure_pmc_documents(document_count: int) -> Tuple[int, str]:
    """
    Ensure we have enough PMC documents for the test.
    
    Args:
        document_count: Number of documents needed
        
    Returns:
        Tuple of (available_count, pmc_dir)
    """
    from pathlib import Path
    
    # Default PMC directory
    pmc_dir = "data/pmc_oas_downloaded"
    
    # Count available XML files
    xml_files = list(Path(pmc_dir).glob("**/*.xml"))
    available_count = len(xml_files)
    
    logger.info(f"Found {available_count} PMC XML files in {pmc_dir}")
    
    if available_count < document_count:
        logger.warning(f"Insufficient documents: requested {document_count}, found {available_count}")
        logger.warning("Proceeding with available documents - results may not be representative")
    
    return min(available_count, document_count), pmc_dir

def run_performance_test(args):
    """
    Run performance test with the specified techniques and document count.
    
    Args:
        args: Command-line arguments
    """
    start_time = time.time()
    
    # Set environment variables for the test
    if not args.no_testcontainer:
        os.environ["TEST_IRIS"] = "true"
    
    # Handle document count setting
    os.environ["TEST_DOCUMENT_COUNT"] = str(args.document_count)
    
    # Handle embedding model options
    if args.use_mock_embeddings:
        os.environ["USE_MOCK_EMBEDDINGS"] = "true"
        logger.info("Using mock embeddings")
    else:
        # Explicitly set to false to override any environment setting
        os.environ["USE_MOCK_EMBEDDINGS"] = "false"
        logger.info("Using real embeddings for accurate testing")
    
    # Handle memory limit for testcontainer
    if args.memory_limit:
        os.environ["TESTCONTAINER_MEMORY_LIMIT"] = str(args.memory_limit)
        logger.info(f"Setting testcontainer memory limit to {args.memory_limit}MB")
    
    # Build test patterns
    if "all" in args.techniques:
        test_pattern = "tests/test_*_with_testcontainer.py"
    else:
        patterns = []
        technique_map = {
            "graphrag": "graphrag",
            "colbert": "colbert",
            "hyde": "hyde",
            "noderag": "noderag",
            "crag": "crag",
            "basic": "basic_rag"
        }
        for technique in args.techniques:
            patterns.append(f"tests/test_{technique_map[technique]}_with_testcontainer.py")
        test_pattern = " ".join(patterns)
    
    # Verify environment variables for tracking performance metrics
    os.environ["COLLECT_PERFORMANCE_METRICS"] = "true"
    
    # Build pytest command - ensuring we use poetry run for correct environment
    verbose_flag = "-v" if args.verbose else ""
    command = f"poetry run pytest {test_pattern} {verbose_flag} --log-cli-level=INFO -xvs"
    
    logger.info(f"Running command: {command}")
    logger.info(f"Environment settings:")
    logger.info(f"  TEST_IRIS: {os.environ.get('TEST_IRIS', 'not set')}")
    logger.info(f"  TEST_DOCUMENT_COUNT: {os.environ.get('TEST_DOCUMENT_COUNT', 'not set')}")
    logger.info(f"  USE_MOCK_EMBEDDINGS: {os.environ.get('USE_MOCK_EMBEDDINGS', 'not set')}")
    
    # Execute pytest
    logger.info(f"Starting large-scale test with {args.document_count} documents...")
    result = subprocess.run(command, shell=True, capture_output=True, text=True)
    
    # Process test results
    elapsed_time = time.time() - start_time
    
    logger.info(f"Tests completed in {elapsed_time:.2f} seconds")
    logger.info(f"Return code: {result.returncode}")
    
    # Extract performance metrics from output if possible
    metrics = extract_performance_metrics(result.stdout)
    
    # Save full results
    save_test_results(args, result, metrics, elapsed_time)
    
    return result.returncode

def extract_performance_metrics(output: str) -> Dict[str, Any]:
    """
    Extract performance metrics from test output.
    
    Args:
        output: Test output string
        
    Returns:
        Dictionary with performance metrics
    """
    metrics = {
        "techniques": {}
    }
    
    # Try to extract metrics
    try:
        # Look for result markers in the output
        import re
        result_blocks = re.findall(r'RESULT_METRICS_START(.*?)RESULT_METRICS_END', output, re.DOTALL)
        
        for block in result_blocks:
            try:
                # Parse JSON metrics
                data = json.loads(block.strip())
                technique = data.get("technique", "unknown")
                metrics["techniques"][technique] = data
            except json.JSONDecodeError:
                logger.warning(f"Failed to parse metrics block: {block[:100]}...")
    except Exception as e:
        logger.error(f"Error extracting metrics: {e}")
    
    return metrics

def save_test_results(args, result, metrics, elapsed_time):
    """
    Save test results to a file.
    
    Args:
        args: Command-line arguments
        result: Subprocess result
        metrics: Extracted performance metrics
        elapsed_time: Total elapsed time
    """
    # Prepare results
    test_results = {
        "configuration": {
            "document_count": args.document_count,
            "batch_size": args.batch_size,
            "techniques": args.techniques,
            "use_real_embeddings": not args.use_mock_embeddings,
            "use_testcontainer": not args.no_testcontainer,
            "memory_limit": args.memory_limit
        },
        "execution": {
            "total_time_seconds": elapsed_time,
            "return_code": result.returncode,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        },
        "metrics": metrics
    }
    
    # Save to file
    try:
        with open(args.report_file, 'w') as f:
            json.dump(test_results, f, indent=2)
        logger.info(f"Results saved to {args.report_file}")
    except Exception as e:
        logger.error(f"Error saving results: {e}")
    
    # Display summary
    logger.info("Test Summary:")
    logger.info(f"  Document count: {args.document_count}")
    logger.info(f"  Total time: {elapsed_time:.2f}s")
    
    # Show metrics summary if available
    if metrics and "techniques" in metrics:
        for technique, data in metrics["techniques"].items():
            if "avg_time_per_query" in data:
                logger.info(f"  {technique}: {data['avg_time_per_query']:.3f}s per query")

def main():
    """Main function"""
    args = parse_args()
    
    # Warn about large document counts
    if args.document_count > 500:
        logger.warning(f"Processing {args.document_count} documents may take significant time and resources")
        logger.warning("Consider running with fewer documents first to establish performance baseline")
    
    if args.document_count >= 1000:
        logger.warning("LARGE-SCALE TEST: This will process 1000+ documents")
        logger.warning("Make sure you have sufficient disk space, memory, and time")
        
        # Check system resources
        try:
            import psutil
            mem = psutil.virtual_memory()
            logger.info(f"System memory: {mem.total / (1024**3):.1f} GB total, {mem.available / (1024**3):.1f} GB available")
            
            if mem.available < 4 * (1024**3):  # Less than 4GB available
                logger.warning("Low memory available - test may fail or system may become unresponsive")
        except ImportError:
            logger.info("Cannot check system resources - psutil not available")
    
    # Ensure we have enough PMC documents
    available_count, pmc_dir = ensure_pmc_documents(args.document_count)
    
    # Update document count based on availability
    if available_count < args.document_count:
        args.document_count = available_count
        logger.info(f"Adjusted document count to {args.document_count} based on availability")
    
    # Set batch size for document processing
    os.environ["BATCH_SIZE"] = str(args.batch_size)
    
    # Run the test
    return run_performance_test(args)

if __name__ == "__main__":
    try:
        sys.exit(main())
    except KeyboardInterrupt:
        logger.info("Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.critical(f"Unexpected error: {e}", exc_info=True)
        sys.exit(2)
