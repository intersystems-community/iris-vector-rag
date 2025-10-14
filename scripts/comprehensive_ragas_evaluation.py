#!/usr/bin/env python3
"""
Comprehensive RAGAS Evaluation Script - One-Stop Solution

This script:
1. Loads sample PMC documents into all pipelines
2. Runs RAGAS evaluation across all pipeline types
3. Generates visualization reports comparing performance
4. Handles data loading, evaluation, and reporting in one go

Usage:
    python scripts/comprehensive_ragas_evaluation.py
"""

import logging
import os
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from iris_rag import create_pipeline
from scripts.generate_ragas_evaluation import (
    EvaluationConfig,
    RAGASEvaluationOrchestrator,
)

logger = logging.getLogger(__name__)


def setup_logging():
    """Setup comprehensive logging."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )


def load_documents_into_pipelines(data_dir: str = "data/sample_10_docs"):
    """Load documents into all pipeline types that support document loading."""
    logger.info(f"🚀 Loading documents from {data_dir} into pipelines...")

    # Pipeline types that support document loading (all production pipelines)
    pipeline_types = ["basic", "basic_rerank", "crag", "graphrag"]

    successful_loads = []
    failed_loads = []

    for pipeline_type in pipeline_types:
        try:
            logger.info(f"📂 Loading documents into {pipeline_type} pipeline...")

            # Create pipeline without validation to avoid schema issues
            pipeline = create_pipeline(
                pipeline_type=pipeline_type, validate_requirements=False, auto_setup=False
            )

            # Load documents
            load_result = pipeline.load_documents(data_dir)

            if load_result:
                successful_loads.append(pipeline_type)
                logger.info(f"✅ Successfully loaded documents into {pipeline_type}")
            else:
                failed_loads.append(pipeline_type)
                logger.warning(
                    f"⚠️  Document loading returned False for {pipeline_type}"
                )

        except Exception as e:
            failed_loads.append(pipeline_type)
            logger.error(f"❌ Failed to load documents into {pipeline_type}: {e}")

    logger.info(f"📊 Document loading summary:")
    logger.info(f"   ✅ Successful: {successful_loads}")
    logger.info(f"   ❌ Failed: {failed_loads}")

    return successful_loads, failed_loads


def run_comprehensive_evaluation():
    """Run the comprehensive RAGAS evaluation."""
    logger.info("🎯 Starting Comprehensive RAGAS Evaluation with Data Loading")

    # Step 1: Load documents into pipelines
    successful_loads, failed_loads = load_documents_into_pipelines()

    if not successful_loads:
        logger.error(
            "❌ No pipelines successfully loaded documents. Cannot proceed with evaluation."
        )
        return False

    # Step 2: Configure evaluation for successfully loaded pipelines
    config = EvaluationConfig(
        num_queries=10,  # Use fewer queries for faster evaluation
        pipelines=successful_loads,  # Only evaluate pipelines with data
        output_dir=Path("outputs/reports/ragas_evaluations"),
        use_cache=True,
        parallel_execution=True,
        confidence_level=0.95,
        target_accuracy=0.80,
    )

    # Step 3: Run RAGAS evaluation
    logger.info("🚀 Starting RAGAS evaluation on loaded data...")

    try:
        orchestrator = RAGASEvaluationOrchestrator(config)
        results = orchestrator.run_comprehensive_evaluation()

        # Step 4: Print summary
        logger.info("🎉 Evaluation completed successfully!")

        # Find best performing pipeline
        pipeline_metrics = results.get("pipeline_metrics", {})
        if pipeline_metrics:
            best_pipeline = None
            best_score = 0

            for pipeline_name, metrics in pipeline_metrics.items():
                score = metrics.get("answer_correctness", {}).get("mean", 0)
                if score > best_score:
                    best_score = score
                    best_pipeline = pipeline_name

            logger.info(
                f"🏆 Best performing pipeline: {best_pipeline} (Score: {best_score:.3f})"
            )

        # Step 5: Provide report locations
        timestamp = results["metadata"]["timestamp"]
        html_report = f"outputs/reports/ragas_evaluations/ragas_report_{timestamp}.html"
        json_report = f"outputs/reports/ragas_evaluations/ragas_report_{timestamp}.json"

        logger.info(f"📄 Reports generated:")
        logger.info(f"   📊 HTML Report: {html_report}")
        logger.info(f"   📋 JSON Report: {json_report}")

        return True

    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return False


def main():
    """Main entry point."""
    setup_logging()

    logger.info("=" * 80)
    logger.info("🚀 COMPREHENSIVE RAGAS EVALUATION WITH DATA LOADING")
    logger.info("=" * 80)

    start_time = time.time()

    try:
        success = run_comprehensive_evaluation()

        end_time = time.time()
        duration = end_time - start_time

        logger.info("=" * 80)
        if success:
            logger.info(
                f"✅ EVALUATION COMPLETED SUCCESSFULLY in {duration:.1f} seconds"
            )
            logger.info("🎉 RAGAS visualization reports are ready!")
            logger.info(
                "📊 Open the HTML report to see pipeline performance comparison"
            )
        else:
            logger.error(f"❌ EVALUATION FAILED after {duration:.1f} seconds")
        logger.info("=" * 80)

        return 0 if success else 1

    except Exception as e:
        logger.error(f"💥 Fatal error: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
