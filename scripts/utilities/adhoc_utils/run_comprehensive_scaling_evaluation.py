#!/usr/bin/env python3
"""
Main Execution Script for Comprehensive Scaling and Evaluation
Runs the complete pipeline for testing all 7 RAG techniques across dataset sizes with RAGAS metrics
"""

import sys
import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any
import argparse

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from scripts.utilities.evaluation.comprehensive_scaling_orchestrator import ComprehensiveScalingOrchestrator
from scripts.utilities.evaluation.scaling_evaluation_framework import ScalingEvaluationFramework
from scripts.utilities.automated_dataset_scaling import AutomatedDatasetScaling
from common.iris_connector import get_iris_connection
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'scaling_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_prerequisites() -> Dict[str, Any]:
    """Check system prerequisites for running the evaluation"""
    logger.info("🔍 Checking prerequisites...")
    
    prerequisites = {
        'database_connection': False,
        'ragas_available': False,
        'openai_api_key': False,
        'document_count': 0,
        'ready': False
    }
    
    try:
        # Check database connection
        connection = get_iris_connection()
        cursor = connection.cursor()
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
        doc_count = cursor.fetchone()[0]
        cursor.close()
        connection.close()
        
        prerequisites['database_connection'] = True
        prerequisites['document_count'] = doc_count
        logger.info(f"✅ Database connection: {doc_count:,} documents available")
        
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return prerequisites
    
    # Check RAGAS availability
    try:
        from ragas import evaluate
        prerequisites['ragas_available'] = True
        logger.info("✅ RAGAS library available")
    except ImportError:
        logger.warning("⚠️ RAGAS not available - install with: pip install ragas datasets")
    
    # Check OpenAI API key
    if os.getenv("OPENAI_API_KEY"):
        prerequisites['openai_api_key'] = True
        logger.info("✅ OpenAI API key configured")
    else:
        logger.warning("⚠️ OpenAI API key not found - RAGAS evaluation will use stub LLM")
    
    # Overall readiness
    prerequisites['ready'] = (
        prerequisites['database_connection'] and 
        prerequisites['document_count'] > 0
    )
    
    return prerequisites

def print_evaluation_plan():
    """Print the comprehensive evaluation plan"""
    logger.info("\n" + "="*80)
    logger.info("📋 COMPREHENSIVE SCALING AND EVALUATION PLAN")
    logger.info("="*80)
    
    logger.info("\n🎯 OBJECTIVE:")
    logger.info("Test all 7 RAG techniques across increasing dataset sizes with comprehensive RAGAS metrics")
    
    logger.info("\n🔬 RAG TECHNIQUES TO EVALUATE:")
    techniques = [
        "1. BasicRAG - Reliable production baseline",
        "2. HyDE - Hypothetical document generation", 
        "3. CRAG - Corrective retrieval with enhanced coverage",
        "4. ColBERT - Token-level semantic matching",
        "5. NodeRAG - Maximum coverage specialist",
        "6. GraphRAG - Ultra-fast graph-based retrieval",
        "7. HybridIFindRAG - Multi-modal fusion approach"
    ]
    for technique in techniques:
        logger.info(f"   {technique}")
    
    logger.info("\n📊 DATASET SCALING STRATEGY:")
    sizes = [1000, 2500, 5000, 10000, 25000, 50000]
    logger.info(f"   Target sizes: {', '.join(f'{s:,}' for s in sizes)} documents")
    
    logger.info("\n📈 RAGAS METRICS:")
    metrics = [
        "• Answer Relevancy", "• Context Precision", "• Context Recall",
        "• Faithfulness", "• Answer Similarity", "• Answer Correctness", 
        "• Context Relevancy"
    ]
    for metric in metrics:
        logger.info(f"   {metric}")
    
    logger.info("\n⚡ PERFORMANCE METRICS:")
    perf_metrics = [
        "• Response Time", "• Documents Retrieved", "• Similarity Scores",
        "• Answer Length", "• Memory Usage", "• Success Rate"
    ]
    for metric in perf_metrics:
        logger.info(f"   {metric}")
    
    logger.info("\n📋 EVALUATION PROTOCOL:")
    protocol = [
        "1. Scale dataset to target size with performance monitoring",
        "2. Run all 7 techniques with standardized test queries",
        "3. Collect comprehensive RAGAS metrics for each technique",
        "4. Measure retrieval performance and system resource usage",
        "5. Generate comparative analysis and visualizations",
        "6. Provide technique selection recommendations"
    ]
    for step in protocol:
        logger.info(f"   {step}")
    
    logger.info("\n📊 DELIVERABLES:")
    deliverables = [
        "• Comprehensive JSON results for each dataset size",
        "• Performance vs scale visualizations",
        "• Quality vs scale analysis charts", 
        "• Technique comparison dashboard",
        "• Executive summary report with recommendations",
        "• Raw data for further analysis"
    ]
    for deliverable in deliverables:
        logger.info(f"   {deliverable}")
    
    logger.info("\n" + "="*80)

def run_evaluation_mode(mode: str) -> Dict[str, Any]:
    """Run evaluation in specified mode"""
    
    if mode == "current_size":
        logger.info("🎯 Running evaluation at current database size...")
        evaluator = ScalingEvaluationFramework()
        return evaluator.run_complete_scaling_evaluation()
    
    elif mode == "comprehensive":
        logger.info("🚀 Running comprehensive scaling and evaluation pipeline...")
        orchestrator = ComprehensiveScalingOrchestrator()
        return orchestrator.run_complete_pipeline()
    
    elif mode == "scaling_only":
        logger.info("📈 Running dataset scaling only...")
        scaler = AutomatedDatasetScaling()
        return scaler.run_automated_scaling()
    
    else:
        raise ValueError(f"Unknown evaluation mode: {mode}")

def main():
    """Main execution function"""
    parser = argparse.ArgumentParser(description="Comprehensive RAG Scaling and Evaluation")
    parser.add_argument(
        "--mode", 
        choices=["current_size", "comprehensive", "scaling_only"],
        default="current_size",
        help="Evaluation mode to run"
    )
    parser.add_argument(
        "--skip-checks", 
        action="store_true",
        help="Skip prerequisite checks"
    )
    
    args = parser.parse_args()
    
    logger.info("🚀 Starting Comprehensive RAG Scaling and Evaluation")
    logger.info(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    # Print evaluation plan
    print_evaluation_plan()
    
    # Check prerequisites
    if not args.skip_checks:
        prerequisites = check_prerequisites()
        
        if not prerequisites['ready']:
            logger.error("❌ Prerequisites not met. Cannot proceed with evaluation.")
            logger.error("💡 Ensure database is accessible and contains documents.")
            return 1
        
        logger.info(f"✅ Prerequisites met. Ready to evaluate with {prerequisites['document_count']:,} documents.")
        
        if not prerequisites['ragas_available']:
            logger.warning("⚠️ RAGAS not available - quality metrics will be limited")
        
        if not prerequisites['openai_api_key']:
            logger.warning("⚠️ OpenAI API key not configured - using stub LLM for evaluation")
    
    # Confirm execution
    if args.mode == "comprehensive":
        logger.info("\n⚠️ COMPREHENSIVE MODE will run scaling AND evaluation - this may take significant time")
        response = input("Continue? (y/N): ")
        if response.lower() != 'y':
            logger.info("❌ Evaluation cancelled by user")
            return 0
    
    # Run evaluation
    start_time = time.time()
    
    try:
        logger.info(f"\n🎯 Starting evaluation in '{args.mode}' mode...")
        results = run_evaluation_mode(args.mode)
        
        execution_time = time.time() - start_time
        
        logger.info(f"\n🎉 EVALUATION COMPLETE!")
        logger.info(f"⏱️ Total execution time: {execution_time:.1f} seconds ({execution_time/60:.1f} minutes)")
        
        # Summary of results
        if 'evaluation_results' in results:
            eval_results = results['evaluation_results']
            logger.info(f"📊 Evaluated {len(eval_results)} dataset sizes")
            
            for size_str, size_result in eval_results.items():
                techniques = size_result.get('techniques', {})
                successful = sum(1 for t in techniques.values() if t.get('success', False))
                logger.info(f"   {size_str} documents: {successful}/{len(techniques)} techniques successful")
        
        logger.info("\n📁 Generated files:")
        # List generated files
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        possible_files = [
            f"comprehensive_scaling_pipeline_{timestamp}.json",
            f"complete_scaling_evaluation_{timestamp}.json", 
            f"automated_scaling_results_{timestamp}.json",
            f"comprehensive_scaling_report_{timestamp}.md",
            f"scaling_evaluation_report_{timestamp}.md",
            f"performance_scaling_analysis_{timestamp}.png",
            f"quality_scaling_analysis_{timestamp}.png"
        ]
        
        for filename in possible_files:
            if os.path.exists(filename):
                logger.info(f"   ✅ {filename}")
        
        logger.info("\n🎯 NEXT STEPS:")
        logger.info("   1. Review the generated report and visualizations")
        logger.info("   2. Analyze technique performance characteristics")
        logger.info("   3. Select optimal techniques for your use case")
        logger.info("   4. Consider scaling optimizations based on results")
        
        return 0
        
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)