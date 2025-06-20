#!/usr/bin/env python3
"""
Script to run the Unified RAGAS Evaluation Framework
Provides command-line interface for running comprehensive RAG evaluations
"""

import os
import sys
import argparse
import logging
from pathlib import Path

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from utilities.evaluation.unified_ragas_evaluation_framework import UnifiedRAGASEvaluationFramework
from utilities.evaluation.config_manager import ConfigManager

def setup_logging(level: str = "INFO"):
    """Setup logging configuration"""
    numeric_level = getattr(logging, level.upper(), None)
    if not isinstance(numeric_level, int):
        raise ValueError(f'Invalid log level: {level}')
    
    logging.basicConfig(
        level=numeric_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),
            logging.FileHandler('evaluation.log')
        ]
    )

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(
        description="Run Unified RAGAS Evaluation Framework",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with default configuration
  python scripts/run_unified_evaluation.py

  # Run with custom configuration file
  python scripts/run_unified_evaluation.py --config eval/config/dev_config.json

  # Run with specific pipelines only
  python scripts/run_unified_evaluation.py --pipelines BasicRAG,HyDE

  # Run in development mode (fast)
  python scripts/run_unified_evaluation.py --dev

  # Run with custom parameters
  python scripts/run_unified_evaluation.py --iterations 5 --top-k 15 --no-ragas
        """
    )
    
    # Configuration options
    parser.add_argument(
        '--config', '-c',
        type=str,
        help='Path to configuration file (JSON or YAML)'
    )
    
    parser.add_argument(
        '--dev',
        action='store_true',
        help='Use development configuration (faster, limited pipelines)'
    )
    
    # Pipeline selection
    parser.add_argument(
        '--pipelines',
        type=str,
        help='Comma-separated list of pipelines to run (e.g., BasicRAG,HyDE,CRAG)'
    )
    
    # Evaluation parameters
    parser.add_argument(
        '--iterations',
        type=int,
        help='Number of evaluation iterations'
    )
    
    parser.add_argument(
        '--top-k',
        type=int,
        help='Number of documents to retrieve (top-k)'
    )
    
    parser.add_argument(
        '--similarity-threshold',
        type=float,
        help='Similarity threshold for retrieval'
    )
    
    # Feature toggles
    parser.add_argument(
        '--no-ragas',
        action='store_true',
        help='Disable RAGAS evaluation'
    )
    
    parser.add_argument(
        '--no-stats',
        action='store_true',
        help='Disable statistical analysis'
    )
    
    parser.add_argument(
        '--no-viz',
        action='store_true',
        help='Disable visualization generation'
    )
    
    parser.add_argument(
        '--parallel',
        action='store_true',
        help='Enable parallel execution'
    )
    
    # Output options
    parser.add_argument(
        '--output-dir',
        type=str,
        default='eval_results',
        help='Output directory for results'
    )
    
    parser.add_argument(
        '--log-level',
        choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
        default='INFO',
        help='Logging level'
    )
    
    args = parser.parse_args()
    
    # Setup logging
    setup_logging(args.log_level)
    logger = logging.getLogger(__name__)
    
    try:
        # Load configuration
        config_manager = ConfigManager()
        
        if args.dev:
            # Use development configuration
            config_path = "eval/config/dev_config.json"
            logger.info("Using development configuration")
        elif args.config:
            # Use specified configuration file
            config_path = args.config
            logger.info(f"Using configuration file: {config_path}")
        else:
            # Use default configuration or environment
            config_path = None
            logger.info("Using default configuration")
        
        # Load base configuration
        if config_path and Path(config_path).exists():
            config = config_manager.load_config(config_path)
        else:
            config = config_manager.load_config()
        
        # Apply command-line overrides
        if args.pipelines:
            # Enable only specified pipelines
            pipeline_names = [name.strip() for name in args.pipelines.split(',')]
            for name in config.pipelines:
                config.pipelines[name].enabled = name in pipeline_names
            logger.info(f"Enabled pipelines: {pipeline_names}")
        
        if args.iterations is not None:
            config.evaluation.num_iterations = args.iterations
            logger.info(f"Set iterations to: {args.iterations}")
        
        if args.top_k is not None:
            config.retrieval.top_k = args.top_k
            logger.info(f"Set top-k to: {args.top_k}")
        
        if args.similarity_threshold is not None:
            config.retrieval.similarity_threshold = args.similarity_threshold
            logger.info(f"Set similarity threshold to: {args.similarity_threshold}")
        
        if args.no_ragas:
            config.evaluation.enable_ragas = False
            logger.info("RAGAS evaluation disabled")
        
        if args.no_stats:
            config.evaluation.enable_statistical_testing = False
            logger.info("Statistical analysis disabled")
        
        if args.no_viz:
            config.output.create_visualizations = False
            logger.info("Visualization generation disabled")
        
        if args.parallel:
            config.evaluation.parallel_execution = True
            logger.info("Parallel execution enabled")
        
        if args.output_dir:
            config.output.results_dir = args.output_dir
            logger.info(f"Output directory set to: {args.output_dir}")
        
        # Validate configuration
        if not config.validate():
            logger.error("Configuration validation failed")
            return 1
        
        # Initialize framework
        logger.info("Initializing Unified RAGAS Evaluation Framework...")
        framework = UnifiedRAGASEvaluationFramework(config)
        
        # Check if any pipelines are available
        if not framework.pipelines:
            logger.error("No pipelines available for evaluation")
            return 1
        
        enabled_pipelines = list(framework.pipelines.keys())
        logger.info(f"Available pipelines: {enabled_pipelines}")
        
        # Run evaluation
        logger.info("Starting comprehensive evaluation...")
        results = framework.run_comprehensive_evaluation()
        
        # Generate and display report
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report = framework.generate_report(results, timestamp)
        
        print("\n" + "="*80)
        print("EVALUATION COMPLETE")
        print("="*80)
        print(report)
        
        # Summary statistics
        successful_pipelines = [name for name, metrics in results.items() if metrics.success_rate > 0]
        total_pipelines = len(results)
        
        print(f"\nSummary:")
        print(f"- Total pipelines evaluated: {total_pipelines}")
        print(f"- Successful pipelines: {len(successful_pipelines)}")
        print(f"- Results saved to: {config.output.results_dir}")
        
        if successful_pipelines:
            best_pipeline = max(results.items(), key=lambda x: x[1].success_rate)
            print(f"- Best performing pipeline: {best_pipeline[0]} ({best_pipeline[1].success_rate:.2%} success rate)")
        
        return 0
        
    except KeyboardInterrupt:
        logger.info("Evaluation interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        if args.log_level == 'DEBUG':
            import traceback
            traceback.print_exc()
        return 1

if __name__ == "__main__":
    sys.exit(main())