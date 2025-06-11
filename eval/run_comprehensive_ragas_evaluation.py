#!/usr/bin/env python3
"""
Comprehensive RAGAS Evaluation Script with DBAPI Support

This script runs comprehensive RAGAS evaluations across multiple RAG pipelines
using the DBAPI connection interface for optimal performance.
"""

import os
import sys
import argparse
import logging
from pathlib import Path
from dotenv import load_dotenv

# Configure unbuffered output for real-time progress display
os.environ['PYTHONUNBUFFERED'] = '1'

# Load environment variables
load_dotenv()

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from eval.comprehensive_ragas_evaluation import ComprehensiveRAGASEvaluationFramework


def print_flush(message: str):
    """Print with immediate flush for real-time output."""
    print(message, flush=True)
    sys.stdout.flush()


def setup_logging(verbose: bool = False) -> logging.Logger:
    """Setup logging configuration with optional verbose mode."""
    level = logging.DEBUG if verbose else logging.INFO
    
    # Force immediate flushing
    class FlushingHandler(logging.StreamHandler):
        def emit(self, record):
            super().emit(record)
            self.flush()
            sys.stdout.flush()
    
    # Configure root logger with flushing handler
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            FlushingHandler(sys.stdout)
        ],
        force=True
    )
    
    # Set specific logger levels
    loggers_to_configure = [
        'comprehensive_ragas_evaluation',
        'eval.comprehensive_ragas_evaluation', 
        '__main__',
        'iris_rag',
        'eval'
    ]
    
    for logger_name in loggers_to_configure:
        logger = logging.getLogger(logger_name)
        logger.setLevel(level)
    
    # Get main logger for this script
    main_logger = logging.getLogger(__name__)
    
    if verbose:
        # Debug information about logging setup
        root_logger = logging.getLogger()
        main_logger.debug(f"🔍 DEBUG logging enabled - root logger level: {root_logger.level}")
        main_logger.debug(f"🔍 Configured loggers: {loggers_to_configure}")
        main_logger.debug(f"🔍 Handler levels: {[h.level for h in root_logger.handlers]}")
    
    return main_logger


def main():
    """Main execution function."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Run comprehensive RAGAS evaluation')
    parser.add_argument('--verbose', '-v', action='store_true', 
                       help='Enable verbose logging')
    parser.add_argument('--pipelines', nargs='+', 
                       default=['basic', 'hyde', 'crag', 'colbert', 'noderag', 'graphrag', 'hybrid_ifind'],
                       help='Pipelines to evaluate')
    parser.add_argument('--iterations', type=int, default=1,
                       help='Number of iterations per query')
    
    args = parser.parse_args()
    
    # Setup logging and get logger
    logger = setup_logging(args.verbose)
    
    print_flush("🚀 Starting Comprehensive RAGAS Evaluation with DBAPI")
    print_flush("📋 Configuration: eval/config/ragas_dbapi_config.json")
    print_flush("🔧 Initializing evaluation framework...")
    print_flush("🔍 Verbose logging enabled - detailed pipeline initialization diagnostics will be shown")
    
    logger.info("🚀 Starting Comprehensive RAGAS Evaluation with DBAPI")
    logger.info("📋 Configuration: eval/config/ragas_dbapi_config.json")
    logger.info("🔧 Initializing evaluation framework...")
    logger.info("🔍 Verbose logging enabled - detailed pipeline initialization diagnostics will be shown")
    
    try:
        print_flush("🔧 Initializing evaluation framework...")
        logger.info("🔧 Initializing evaluation framework...")
        
        # Create evaluation framework
        framework = ComprehensiveRAGASEvaluationFramework(
            config_path="eval/config/ragas_dbapi_config.json"
        )
        
        print_flush("✅ Framework initialized successfully")
        logger.info("✅ Framework initialized successfully")
        
        print_flush("🏃 Running comprehensive evaluation suite...")
        
        # Run the comprehensive evaluation
        results = framework.run_full_evaluation_suite()
        
        # Display comprehensive results
        print_flush("")
        print_flush("="*80)
        print_flush("🎉 COMPREHENSIVE RAGAS EVALUATION COMPLETED!")
        logger.info("")
        logger.info("="*80)
        logger.info("🎉 COMPREHENSIVE RAGAS EVALUATION COMPLETED!")
        logger.info("="*80)
        logger.info(f"📊 Evaluated {results['pipelines_evaluated']} pipelines")
        logger.info(f"📝 Processed {results['total_queries']} queries per pipeline")
        logger.info(f"🔄 Ran {results['iterations']} iterations per query")
        logger.info(f"⏱️  Total time: {results['total_time']:.2f} seconds")
        logger.info(f"🔗 Connection type: {results['connection_type']}")
        logger.info(f"📁 Results saved with timestamp: {results['timestamp']}")
        
        # Display pipeline performance summary
        logger.info("\n📈 Pipeline Performance Summary:")
        logger.info("-" * 80)
        for pipeline_name, metrics in results['results'].items():
            success_rate = metrics.success_rate * 100
            avg_time = metrics.avg_response_time
            avg_docs = metrics.avg_documents_retrieved
            
            ragas_info = ""
            if metrics.avg_answer_relevancy is not None:
                ragas_score = (
                    metrics.avg_answer_relevancy + 
                    metrics.avg_context_precision + 
                    metrics.avg_context_recall + 
                    metrics.avg_faithfulness + 
                    metrics.avg_answer_correctness
                ) / 5
                ragas_info = f" | RAGAS: {ragas_score:.3f}"
            
            logger.info(
                f"{pipeline_name:12} | Success: {success_rate:5.1f}% | "
                f"Time: {avg_time:5.2f}s | Docs: {avg_docs:4.1f}{ragas_info}"
            )
        
        logger.info("="*80)
        logger.info(f"📋 Full report: {Path(framework.config.output.results_dir) / 'reports'}")
        logger.info(f"📊 Visualizations: {Path(framework.config.output.results_dir) / 'visualizations'}")
        logger.info(f"📁 Raw data: {Path(framework.config.output.results_dir) / 'raw_data'}")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("\n⚠️ Evaluation interrupted by user")
        return 1
    except Exception as e:
        if args.verbose:
            logger.exception("❌ Detailed error information:")
        else:
            logger.error(f"❌ Evaluation failed: {str(e)}")
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)