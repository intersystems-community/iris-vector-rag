#!/usr/bin/env python3
"""
Lightweight RAGAs Testing Resumption Specification Implementation

This script provides a targeted and efficient approach for RAGAs testing with:
- Command-line interface for flexible evaluation
- Cache management and status checking
- Multiple metric levels (core, extended, full)
- Modular pipeline evaluation
- Transparent cache usage via get_llm_func
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Suppress tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Core imports
import iris_rag
from common.utils import get_llm_func
from common.iris_connection_manager import get_iris_connection

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
    # Try to import context_relevancy, but don't fail if not available
    try:
        from ragas.metrics import context_relevancy
        CONTEXT_RELEVANCY_AVAILABLE = True
    except ImportError:
        CONTEXT_RELEVANCY_AVAILABLE = False
        context_relevancy = None
    
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    CONTEXT_RELEVANCY_AVAILABLE = False

# Cache management imports
try:
    from common.llm_cache_manager import LangchainCacheManager, load_cache_config
    CACHE_AVAILABLE = True
except ImportError:
    CACHE_AVAILABLE = False

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Available pipelines
AVAILABLE_PIPELINES = ['basic', 'hyde', 'crag', 'colbert', 'noderag', 'graphrag', 'hybrid_ifind']

# Metric level definitions
def get_metric_levels():
    """Get metric levels based on available metrics."""
    base_metrics = {
        'core': [answer_relevancy, faithfulness],
        'extended': [answer_relevancy, faithfulness, context_precision, context_recall],
        'full': [answer_relevancy, faithfulness, context_precision, context_recall,
                 answer_similarity, answer_correctness]
    }
    
    # Add context_relevancy to full if available
    if CONTEXT_RELEVANCY_AVAILABLE and context_relevancy is not None:
        base_metrics['full'].append(context_relevancy)
    
    return base_metrics

METRIC_LEVELS = get_metric_levels() if RAGAS_AVAILABLE else {}


def check_cache_status() -> Dict[str, Any]:
    """
    Check the current cache configuration and status.
    
    Returns:
        Dictionary with cache status information
    """
    print("🔍 [DEBUG] Starting cache status check...", flush=True)
    status = {
        'cache_available': CACHE_AVAILABLE,
        'cache_enabled': False,
        'cache_configured': False,
        'cache_backend': 'unknown',
        'cache_status': 'Not Configured',
        'details': []
    }
    
    if not CACHE_AVAILABLE:
        status['cache_status'] = 'Cache Manager Not Available'
        status['details'].append('Cache management modules not found')
        return status
    
    try:
        # Load cache configuration
        config = load_cache_config()
        status['cache_enabled'] = config.enabled
        status['cache_backend'] = config.backend
        
        if not config.enabled:
            status['cache_status'] = 'Disabled in Configuration'
            status['details'].append('Cache is disabled in config file')
            return status
        
        # Check backend-specific requirements
        if config.backend == 'iris':
            # Check IRIS connection requirements
            # Check for IRIS connection availability (either URL or individual params)
            iris_url = os.getenv('IRIS_CONNECTION_URL')
            iris_host = os.getenv('IRIS_HOST', 'localhost')
            iris_port = os.getenv('IRIS_PORT', '1972')
            
            # Test if we can get an IRIS connection for cache
            try:
                from common.iris_connection_manager import get_iris_connection
                test_connection = get_iris_connection()
                if test_connection:
                    status['details'].append('IRIS connection available for cache reuse')
                    connection_available = True
                else:
                    connection_available = False
            except Exception as e:
                connection_available = False
                status['details'].append(f'IRIS connection test failed: {e}')
            
            if not iris_url and not connection_available:
                status['cache_status'] = 'IRIS Backend - No Connection Available'
                status['details'].append('Neither IRIS_CONNECTION_URL nor reusable IRIS connection available')
                status['details'].append(f'Set IRIS_CONNECTION_URL or ensure IRIS_HOST={iris_host}, IRIS_PORT={iris_port} are correct')
                return status
            elif connection_available and not iris_url:
                status['details'].append('Will reuse existing RAG database connection for cache')
        
        # Try to initialize cache manager
        try:
            cache_manager = LangchainCacheManager(config)
            cache_instance = cache_manager.setup_cache()
            
            if cache_instance is not None:
                status['cache_configured'] = True
                status['cache_status'] = f'Configured and Ready ({config.backend})'
                status['details'].append('Cache successfully initialized')
            else:
                status['cache_status'] = 'Enabled but Failed to Initialize'
                status['details'].append('Cache setup returned None')
                
        except Exception as setup_error:
            status['cache_status'] = f'Setup Failed: {str(setup_error)}'
            status['details'].append(f'Cache initialization error: {setup_error}')
            
            # Provide specific guidance for common issues
            if 'IRIS_CONNECTION_URL' in str(setup_error):
                status['details'].append('Solution: Set IRIS_CONNECTION_URL, ensure IRIS connection parameters are correct, or use memory cache')
            elif 'connection' in str(setup_error).lower():
                status['details'].append('Solution: Check IRIS database connection settings (IRIS_HOST, IRIS_PORT, etc.)')
            else:
                status['details'].append('Solution: Verify IRIS database connectivity or switch to memory cache backend')
            
    except Exception as e:
        status['cache_status'] = f'Configuration Error: {str(e)}'
        status['details'].append(f'Failed to load cache config: {e}')
        logger.warning(f"Cache status check failed: {e}")
    
    return status


def clear_llm_cache() -> bool:
    """
    Clear the LLM cache if available.
    
    Returns:
        True if cache was cleared successfully, False otherwise
    """
    if not CACHE_AVAILABLE:
        logger.warning("Cache manager not available")
        return False
    
    try:
        # Clear langchain cache if configured
        import langchain
        if hasattr(langchain, 'llm_cache') and langchain.llm_cache is not None:
            if hasattr(langchain.llm_cache, 'clear'):
                langchain.llm_cache.clear()
                logger.info("LLM cache cleared successfully")
                return True
            else:
                logger.warning("Cache does not support clearing")
                return False
        else:
            logger.info("No active cache to clear")
            return True
            
    except Exception as e:
        logger.error(f"Failed to clear cache: {e}")
        return False


def disable_llm_cache() -> bool:
    """
    Disable LLM caching for the current session.
    
    Returns:
        True if cache was disabled successfully, False otherwise
    """
    try:
        import langchain
        langchain.llm_cache = None
        logger.info("LLM cache disabled for current session")
        return True
    except Exception as e:
        logger.error(f"Failed to disable cache: {e}")
        return False


def load_test_queries(query_file: str = "eval/sample_queries.json") -> List[Dict[str, Any]]:
    """
    Load test queries from JSON file.
    
    Args:
        query_file: Path to the query file
        
    Returns:
        List of query dictionaries
    """
    print(f"📂 [DEBUG] Loading test queries from: {query_file}", flush=True)
    query_path = Path(query_file)
    if not query_path.exists():
        # Try relative to project root
        query_path = Path(project_root) / query_file
    
    if not query_path.exists():
        logger.error(f"Query file not found: {query_file}")
        return []
    
    try:
        with open(query_path, 'r') as f:
            queries = json.load(f)
        logger.info(f"Loaded {len(queries)} test queries from {query_path}")
        return queries
    except Exception as e:
        logger.error(f"Failed to load queries from {query_file}: {e}")
        return []


def validate_document_count(min_docs: int = 100) -> Tuple[bool, int]:
    """
    Validate that sufficient documents are available in the database.
    
    Args:
        min_docs: Minimum number of documents required
        
    Returns:
        Tuple of (is_valid, actual_count)
    """
    try:
        conn = get_iris_connection()
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
        count = cursor.fetchone()[0]
        cursor.close()
        conn.close()
        
        is_valid = count >= min_docs
        logger.info(f"Document validation: {count} documents found (minimum: {min_docs})")
        return is_valid, count
        
    except Exception as e:
        logger.error(f"Failed to validate document count: {e}")
        return False, 0


def evaluate_pipeline(pipeline_name: str, queries: List[Dict[str, Any]],
                     metrics_level: str = 'core') -> List[Dict[str, Any]]:
    """
    Evaluate a single pipeline with the given queries.
    
    Args:
        pipeline_name: Name of the pipeline to evaluate
        queries: List of test queries
        metrics_level: Level of metrics to use ('core', 'extended', 'full')
        
    Returns:
        List of evaluation results
    """
    print(f"🚀 [DEBUG] Starting evaluation of {pipeline_name} pipeline with {len(queries)} queries", flush=True)
    logger.info(f"Evaluating {pipeline_name} pipeline with {len(queries)} queries")
    
    try:
        print(f"🔧 [DEBUG] Getting LLM function...", flush=True)
        # Get LLM function with transparent cache usage
        llm_func = get_llm_func()
        print(f"✅ [DEBUG] LLM function obtained", flush=True)
        
        # Create pipeline with auto-setup
        print(f"🏗️  [DEBUG] Creating {pipeline_name} pipeline...", flush=True)
        pipeline = iris_rag.create_pipeline(
            pipeline_name,
            llm_func=llm_func,
            external_connection=get_iris_connection(),
            auto_setup=True
        )
        print(f"✅ [DEBUG] Pipeline {pipeline_name} created successfully", flush=True)
        
        results = []
        print(f"🔄 [DEBUG] Starting query loop ({len(queries)} queries)", flush=True)
        for i, query_data in enumerate(queries):
            query = query_data['query']
            print(f"📝 [DEBUG] Processing query {i+1}/{len(queries)}: {query[:50]}...", flush=True)
            logger.info(f"Processing query {i+1}/{len(queries)}: {query[:50]}...")
            
            try:
                print(f"⏱️  [DEBUG] Running pipeline for query {i+1}...", flush=True)
                start_time = time.time()
                result = pipeline.run(query, top_k=5)
                response_time = time.time() - start_time
                print(f"✅ [DEBUG] Query {i+1} completed in {response_time:.2f}s", flush=True)
                
                # Extract contexts from retrieved documents
                contexts = []
                if 'retrieved_documents' in result:
                    contexts = [doc.page_content if hasattr(doc, 'page_content') else '' for doc in result['retrieved_documents']]
                
                query_result = {
                    'pipeline': pipeline_name,
                    'query': query,
                    'answer': result.get('answer', ''),
                    'contexts': contexts,
                    'ground_truth': query_data.get('ground_truth_answer', ''),
                    'response_time': response_time,
                    'documents_retrieved': len(contexts),
                    'success': True
                }
                
                results.append(query_result)
                
            except Exception as e:
                logger.error(f"Failed to process query {i+1}: {e}")
                results.append({
                    'pipeline': pipeline_name,
                    'query': query,
                    'answer': '',
                    'contexts': [],
                    'ground_truth': query_data.get('ground_truth_answer', ''),
                    'response_time': 0.0,
                    'documents_retrieved': 0,
                    'success': False,
                    'error': str(e)
                })
        
        logger.info(f"Completed evaluation of {pipeline_name}: {len(results)} results")
        return results
        
    except Exception as e:
        logger.error(f"Failed to evaluate pipeline {pipeline_name}: {e}")
        return []


def evaluate_with_ragas_simple(query_results: List[Dict[str, Any]],
                              metrics_level: str = 'core') -> Dict[str, Any]:
    """
    Evaluate query results using RAGAS metrics.
    
    Args:
        query_results: List of query results from pipeline evaluation
        metrics_level: Level of metrics to use ('core', 'extended', 'full')
        
    Returns:
        Dictionary with RAGAS evaluation results
    """
    print(f"📊 [DEBUG] Starting RAGAS evaluation with {len(query_results)} results", flush=True)
    if not RAGAS_AVAILABLE:
        print("⚠️  [DEBUG] RAGAS not available", flush=True)
        logger.warning("RAGAS not available, skipping metric evaluation")
        return {'ragas_available': False}
    
    if not query_results:
        logger.warning("No query results to evaluate")
        return {'ragas_available': True, 'results': None}
    
    # Filter successful results
    successful_results = [r for r in query_results if r.get('success', False)]
    if not successful_results:
        logger.warning("No successful query results to evaluate")
        return {'ragas_available': True, 'results': None}
    
    try:
        # Prepare data for RAGAS
        data = {
            'question': [r['query'] for r in successful_results],
            'answer': [r['answer'] for r in successful_results],
            'contexts': [r['contexts'] for r in successful_results],
            'ground_truth': [r['ground_truth'] for r in successful_results]
        }
        
        dataset = Dataset.from_dict(data)
        
        # Get metrics for the specified level
        metrics = METRIC_LEVELS.get(metrics_level, METRIC_LEVELS['core'])
        
        print(f"✅ [DEBUG] Using {len(metrics)} metrics: {[m.name for m in metrics]}", flush=True)
        logger.info(f"Running RAGAS evaluation with {len(metrics)} metrics at '{metrics_level}' level")
        
        # Run RAGAS evaluation with transparent cache usage
        print("🔧 [DEBUG] Getting LLM function for RAGAS...", flush=True)
        llm_func = get_llm_func()
        print("✅ [DEBUG] LLM function obtained for RAGAS", flush=True)
        
        # Note: RAGAS will use its own LLM configuration, but our get_llm_func
        # handles caching transparently
        print("🚀 [DEBUG] Starting RAGAS evaluate() call - this may take a while...", flush=True)
        ragas_result = evaluate(
            dataset,
            metrics=metrics
        )
        print("🎉 [DEBUG] RAGAS evaluate() completed successfully!", flush=True)
        
        logger.info("RAGAS evaluation completed successfully")
        return {
            'ragas_available': True,
            'metrics_level': metrics_level,
            'results': ragas_result,
            'num_queries': len(successful_results)
        }
        
    except Exception as e:
        print(f"❌ [DEBUG] RAGAS evaluation failed: {e}", flush=True)
        logger.error(f"RAGAS evaluation failed: {e}")
        return {
            'ragas_available': True,
            'error': str(e),
            'metrics_level': metrics_level
        }


def generate_simple_report(evaluation_results: Dict[str, Any], 
                          cache_status: Dict[str, Any]) -> str:
    """
    Generate a simple evaluation report.
    
    Args:
        evaluation_results: Results from pipeline evaluation
        cache_status: Cache status information
        
    Returns:
        Formatted report string
    """
    report_lines = [
        "=" * 60,
        "LIGHTWEIGHT RAGAS EVALUATION REPORT",
        "=" * 60,
        f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        "",
        "CACHE STATUS:",
        f"  Available: {cache_status.get('cache_available', False)}",
        f"  Enabled: {cache_status.get('cache_enabled', False)}",
        f"  Status: {cache_status.get('cache_status', 'Unknown')}",
        f"  Backend: {cache_status.get('cache_backend', 'Unknown')}",
        ""
    ]
    
    if 'pipelines' in evaluation_results:
        report_lines.extend([
            "PIPELINE EVALUATION RESULTS:",
            ""
        ])
        
        for pipeline_name, pipeline_results in evaluation_results['pipelines'].items():
            successful = len([r for r in pipeline_results if r.get('success', False)])
            total = len(pipeline_results)
            avg_time = sum(r.get('response_time', 0) for r in pipeline_results) / max(total, 1)
            
            report_lines.extend([
                f"  {pipeline_name.upper()}:",
                f"    Queries: {successful}/{total} successful",
                f"    Avg Response Time: {avg_time:.2f}s",
                ""
            ])
    
    if 'ragas_results' in evaluation_results:
        ragas_data = evaluation_results['ragas_results']
        if ragas_data.get('ragas_available') and 'results' in ragas_data:
            report_lines.extend([
                "RAGAS METRICS:",
                f"  Level: {ragas_data.get('metrics_level', 'unknown')}",
                f"  Queries Evaluated: {ragas_data.get('num_queries', 0)}",
                ""
            ])
    
    report_lines.extend([
        "=" * 60,
        ""
    ])
    
    return "\n".join(report_lines)


def run_lightweight_ragas_evaluation(args) -> Dict[str, Any]:
    """
    Main evaluation function that orchestrates the lightweight RAGAS evaluation.
    
    Args:
        args: Parsed command line arguments
        
    Returns:
        Dictionary with complete evaluation results
    """
    print("🚀 [DEBUG] === STARTING LIGHTWEIGHT RAGAS EVALUATION ===", flush=True)
    logger.info("Starting Lightweight RAGAs Evaluation")
    
    # Check cache status
    print("🔍 [DEBUG] Checking cache status...", flush=True)
    cache_status = check_cache_status()
    print(f"📋 [DEBUG] Cache status result: {cache_status['cache_status']}", flush=True)
    logger.info(f"Cache Status: {cache_status['cache_status']}")
    
    # Handle cache operations
    if args.clear_cache:
        print("🧹 [DEBUG] Clearing LLM cache...", flush=True)
        logger.info("Clearing LLM cache...")
        clear_llm_cache()
    
    if args.no_cache:
        print("🚫 [DEBUG] Disabling LLM cache for this session...", flush=True)
        logger.info("Disabling LLM cache for this session...")
        disable_llm_cache()
    
    # Validate document count
    print("🔢 [DEBUG] Validating document count...", flush=True)
    doc_valid, doc_count = validate_document_count(args.min_docs)
    if not doc_valid:
        print(f"❌ [DEBUG] Insufficient documents: {doc_count} < {args.min_docs}", flush=True)
        logger.error(f"Insufficient documents: {doc_count} < {args.min_docs}")
        return {'error': 'Insufficient documents', 'doc_count': doc_count}
    
    # Load test queries
    print("📂 [DEBUG] Loading test queries...", flush=True)
    queries = load_test_queries(args.queries_file)
    if not queries:
        print("❌ [DEBUG] No test queries available", flush=True)
        logger.error("No test queries available")
        return {'error': 'No test queries available'}
    
    # Limit queries if specified
    if args.max_queries and args.max_queries < len(queries):
        print(f"✂️  [DEBUG] Limiting queries from {len(queries)} to {args.max_queries}", flush=True)
        queries = queries[:args.max_queries]
        logger.info(f"Limited to {args.max_queries} queries")
    
    # Evaluate pipelines
    print("🏗️  [DEBUG] Setting up evaluation results structure...", flush=True)
    evaluation_results = {
        'pipelines': {},
        'cache_status': cache_status,
        'timestamp': datetime.now().isoformat(),
        'args': vars(args)
    }
    
    print(f"🔄 [DEBUG] Starting pipeline evaluation loop for {len(args.pipelines)} pipelines: {args.pipelines}", flush=True)
    for pipeline_name in args.pipelines:
        if pipeline_name not in AVAILABLE_PIPELINES:
            print(f"⚠️  [DEBUG] Unknown pipeline: {pipeline_name}", flush=True)
            logger.warning(f"Unknown pipeline: {pipeline_name}")
            continue
        
        print(f"🚀 [DEBUG] Evaluating pipeline: {pipeline_name}", flush=True)
        pipeline_results = evaluate_pipeline(pipeline_name, queries, args.metrics_level)
        evaluation_results['pipelines'][pipeline_name] = pipeline_results
        print(f"✅ [DEBUG] Completed evaluation of {pipeline_name}", flush=True)
    
    # Run RAGAS evaluation if enabled
    if not args.no_ragas and RAGAS_AVAILABLE:
        print("📊 [DEBUG] Starting RAGAS evaluation phase...", flush=True)
        # Combine all pipeline results for RAGAS evaluation
        print("🔗 [DEBUG] Combining all pipeline results for RAGAS...", flush=True)
        all_results = []
        for pipeline_results in evaluation_results['pipelines'].values():
            all_results.extend(pipeline_results)
        print(f"📋 [DEBUG] Combined {len(all_results)} total results from all pipelines", flush=True)
        
        if all_results:
            print("🚀 [DEBUG] Running RAGAS evaluation...", flush=True)
            ragas_results = evaluate_with_ragas_simple(all_results, args.metrics_level)
            evaluation_results['ragas_results'] = ragas_results
            print("✅ [DEBUG] RAGAS evaluation completed", flush=True)
        else:
            print("⚠️  [DEBUG] No results to evaluate with RAGAS", flush=True)
    elif args.no_ragas:
        print("🚫 [DEBUG] RAGAS evaluation skipped (--no-ragas flag)", flush=True)
    elif not RAGAS_AVAILABLE:
        print("⚠️  [DEBUG] RAGAS evaluation skipped (RAGAS not available)", flush=True)
    
    print("🎉 [DEBUG] === LIGHTWEIGHT RAGAS EVALUATION COMPLETED ===", flush=True)
    return evaluation_results


def main():
    """Main entry point for the lightweight RAGAs evaluation script."""
    print("🎬 [DEBUG] === MAIN FUNCTION STARTED ===", flush=True)
    parser = argparse.ArgumentParser(
        description="Lightweight RAGAs Testing Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s --pipelines basic hyde --metrics-level core
  %(prog)s --pipelines basic --no-ragas --max-queries 5
  %(prog)s --cache-check
  %(prog)s --clear-cache --pipelines colbert
        """
    )
    
    # Pipeline selection
    parser.add_argument(
        '--pipelines', 
        nargs='+', 
        default=['basic'],
        choices=AVAILABLE_PIPELINES,
        help='Pipelines to evaluate (default: basic)'
    )
    
    # Metrics configuration
    parser.add_argument(
        '--metrics-level',
        choices=['core', 'extended', 'full'],
        default='core',
        help='Level of RAGAS metrics to use (default: core)'
    )
    
    # Query configuration
    parser.add_argument(
        '--queries-file',
        default='eval/sample_queries.json',
        help='Path to test queries JSON file (default: eval/sample_queries.json)'
    )
    
    parser.add_argument(
        '--max-queries',
        type=int,
        help='Maximum number of queries to process'
    )
    
    parser.add_argument(
        '--min-docs',
        type=int,
        default=100,
        help='Minimum number of documents required (default: 100)'
    )
    
    # Cache management
    parser.add_argument(
        '--cache-check',
        action='store_true',
        help='Check cache status and exit'
    )
    
    parser.add_argument(
        '--clear-cache',
        action='store_true',
        help='Clear LLM cache before evaluation'
    )
    
    parser.add_argument(
        '--no-cache',
        action='store_true',
        help='Disable LLM cache for this evaluation'
    )
    
    # RAGAS configuration
    parser.add_argument(
        '--no-ragas',
        action='store_true',
        help='Skip RAGAS metric evaluation'
    )
    
    # Output configuration
    parser.add_argument(
        '--output-dir',
        default='.',
        help='Output directory for results (default: current directory)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Configure logging
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Handle cache check
    if args.cache_check:
        cache_status = check_cache_status()
        print(json.dumps(cache_status, indent=2))
        return
    
    # Run evaluation
    try:
        results = run_lightweight_ragas_evaluation(args)
        
        # Save results
        output_dir = Path(args.output_dir)
        output_dir.mkdir(exist_ok=True)
        
        # Save JSON results
        results_file = output_dir / 'ragas_results.json'
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        logger.info(f"Results saved to {results_file}")
        
        # Save cache summary
        cache_summary_file = output_dir / 'cache_summary.txt'
        with open(cache_summary_file, 'w') as f:
            cache_status = results.get('cache_status', {})
            f.write(f"Cache Status: {cache_status.get('cache_status', 'Unknown')}\n")
            f.write(f"Cache Backend: {cache_status.get('cache_backend', 'Unknown')}\n")
            f.write(f"Cache Enabled: {cache_status.get('cache_enabled', False)}\n")
        logger.info(f"Cache summary saved to {cache_summary_file}")
        
        # Generate and save simple report
        report = generate_simple_report(results, results.get('cache_status', {}))
        
        # Save evaluation log
        log_file = output_dir / 'evaluation_log.txt'
        with open(log_file, 'w') as f:
            f.write(report)
        logger.info(f"Evaluation log saved to {log_file}")
        
        # Print summary
        print(report)
        
        # Check for errors
        if 'error' in results:
            logger.error(f"Evaluation failed: {results['error']}")
            sys.exit(1)
        
        logger.info("Lightweight RAGAs evaluation completed successfully")
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)


if __name__ == '__main__':
    main()