#!/usr/bin/env python3
"""
Comprehensive RAGAS Evaluation Script for All Pipelines

This script runs comprehensive RAGAS evaluation on all 4 pipelines using the PMC dataset
to demonstrate >80% accuracy with full statistical analysis.

Requirements met:
1. RAGAS Metrics: Answer Correctness, Faithfulness, Context Precision, Context Recall, Answer Relevance
2. Test Data: Uses SAMPLE_QUERIES from PMC loader with ground truth answers
3. All 4 Pipelines: BasicRAG, BasicRerank, CRAG, GraphRAG
4. Statistical Analysis: Mean, std deviation, confidence intervals
5. Output Format: JSON and HTML reports with timestamp
6. Configuration: Environment variables for customization
"""

import hashlib
import json
import logging
import os
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from evaluation_framework.ragas_metrics_framework import (
    BiomedicalRAGASFramework,
    ComprehensiveRAGASResults,
    RAGASResult,
)
from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager

# Import existing infrastructure
from scripts.data_loaders.pmc_loader import SAMPLE_QUERIES

# Pipeline imports with error handling
try:
    from iris_rag.pipelines.basic import BasicRAGPipeline
    from iris_rag.pipelines.basic_rerank import BasicRAGRerankingPipeline
    from iris_rag.pipelines.crag import CRAGPipeline
    from iris_rag.pipelines.graphrag import GraphRAGPipeline

    # NEW: Import HybridGraphRAG for performance comparison
    from iris_rag.pipelines.hybrid_graphrag import HybridGraphRAGPipeline

    PIPELINES_AVAILABLE = True
    HYBRID_GRAPHRAG_AVAILABLE = True
except ImportError as e:
    logging.warning(f"Pipeline imports failed: {e}")
    PIPELINES_AVAILABLE = False
    HYBRID_GRAPHRAG_AVAILABLE = False

logger = logging.getLogger(__name__)


@dataclass
class EvaluationConfig:
    """Configuration for RAGAS evaluation."""

    num_queries: int
    pipelines: List[str]
    output_dir: Path
    use_cache: bool
    parallel_execution: bool
    confidence_level: float
    target_accuracy: float


@dataclass
class PipelineMetrics:
    """Metrics for a single pipeline."""

    answer_correctness: Dict[str, float]
    faithfulness: Dict[str, float]
    context_precision: Dict[str, float]
    context_recall: Dict[str, float]
    answer_relevance: Dict[str, float]


@dataclass
class ComparisonResult:
    """Results of pipeline comparison."""

    best_overall: str
    best_per_metric: Dict[str, str]
    statistical_significance: Dict[str, Any]


# Ground truth answers for SAMPLE_QUERIES
GROUND_TRUTH_ANSWERS = [
    "Type 2 diabetes treatment includes lifestyle modifications (diet, exercise), metformin as first-line medication, and additional medications like sulfonylureas, DPP-4 inhibitors, or insulin as needed.",
    "COVID-19 affects respiratory function by causing inflammation in the lungs, reducing oxygen exchange capacity, and potentially leading to acute respiratory distress syndrome (ARDS) with long-term lung damage.",
    "Effective hypertension management includes lifestyle changes (reduced sodium, regular exercise, weight management), ACE inhibitors, ARBs, diuretics, calcium channel blockers, and regular monitoring.",
    "Cardiovascular disease risk factors include high blood pressure, diabetes, smoking, obesity, family history, and age. Prevention involves healthy lifestyle, regular exercise, and managing risk factors.",
    "Immunotherapy works by enhancing or modifying the immune system's ability to recognize and attack cancer cells, including checkpoint inhibitors, CAR-T therapy, and monoclonal antibodies.",
    "Alzheimer's research focuses on amyloid plaques, tau proteins, neuroinflammation, genetic factors, and potential treatments including anti-amyloid drugs, lifestyle interventions, and neuroprotective strategies.",
    "Depression symptoms include persistent sadness, loss of interest, fatigue, sleep disturbances, appetite changes, and cognitive difficulties. Diagnosis follows DSM-5 criteria with clinical assessment.",
    "Chronic pain management strategies include medications (analgesics, anti-inflammatories), physical therapy, cognitive behavioral therapy, nerve blocks, and multimodal approaches.",
    "Antibiotic resistance develops through bacterial mutations, gene transfer, and selective pressure from antibiotic use, leading to mechanisms like enzyme production, target modification, and efflux pumps.",
    "Metabolic syndrome nutritional interventions include reduced refined carbohydrates, increased fiber intake, portion control, Mediterranean diet patterns, and regular meal timing.",
    "Vaccines stimulate immune responses by presenting antigens to activate B cells (antibody production) and T cells (cellular immunity), creating immunological memory for future protection.",
    "Breast cancer genetic factors include BRCA1/BRCA2 mutations, family history, hormonal influences, and genetic polymorphisms affecting DNA repair, hormone metabolism, and tumor suppression.",
    "Inflammation contributes to autoimmune diseases through molecular mimicry, tissue damage, cytokine release, and loss of immune tolerance, perpetuating chronic inflammatory cycles.",
    "Kidney disease biomarkers include serum creatinine, estimated GFR, albuminuria, cystatin C, and novel markers like NGAL and KIM-1 for early detection and monitoring.",
    "Effective heart valve surgery techniques include minimally invasive approaches, valve repair vs replacement, transcatheter procedures (TAVR), and robotic-assisted surgery with improved outcomes.",
]


class RAGASEvaluationOrchestrator:
    """
    Orchestrates comprehensive RAGAS evaluation across all pipelines.
    """

    def __init__(self, config: EvaluationConfig):
        self.config = config
        self.ragas_framework = BiomedicalRAGASFramework()
        self.pipelines = {}
        self.cache_dir = self.config.output_dir / "cache"
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Setup logging
        self._setup_logging()

        # Initialize pipelines
        self._initialize_pipelines()

        # Ensure knowledge graph/documents are loaded for consistent, comparable evaluation
        pmc_dir = (
            os.getenv("EVAL_DATASET_DIR")
            or os.getenv("EVAL_PMC_DIR")
            or os.getenv("APPLES_PMC_DIR")  # legacy support
            or "data/sample_10_docs"
        )
        try:
            self._ensure_graph_data(pmc_dir)
        except Exception as e:
            logger.warning(f"Dataset preparation skipped/failed ({pmc_dir}): {e}")

    def _setup_logging(self):
        """Setup comprehensive logging."""
        log_file = (
            self.config.output_dir
            / f"ragas_evaluation_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
        )

        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
            handlers=[logging.FileHandler(log_file), logging.StreamHandler()],
        )

        logger.info("RAGAS Evaluation Orchestrator initialized")
        logger.info(f"Configuration: {asdict(self.config)}")

    def _initialize_pipelines(self):
        """Initialize all available pipelines."""
        if not PIPELINES_AVAILABLE:
            raise RuntimeError(
                "Pipeline imports failed - cannot proceed with evaluation"
            )

        # Use real managers for IRIS-backed evaluation (consistent dataset on PMC data)
        conn_manager = ConnectionManager()
        config_manager = ConfigurationManager()

        pipeline_classes = {
            "BasicRAG": BasicRAGPipeline,
            "BasicRerank": BasicRAGRerankingPipeline,
            "CRAG": CRAGPipeline,
            "GraphRAG": GraphRAGPipeline,
        }

        # Add HybridGraphRAG if available
        if HYBRID_GRAPHRAG_AVAILABLE:
            pipeline_classes["HybridGraphRAG"] = HybridGraphRAGPipeline
            logger.info("HybridGraphRAG pipeline available for comparison testing")

        for pipeline_name in self.config.pipelines:
            if pipeline_name in pipeline_classes:
                try:
                    self.pipelines[pipeline_name] = pipeline_classes[pipeline_name](
                        conn_manager, config_manager
                    )
                    logger.info(f"Initialized {pipeline_name} pipeline")
                except Exception as e:
                    logger.error(f"Failed to initialize {pipeline_name}: {e}")
                    # Continue with other pipelines
            else:
                logger.warning(f"Unknown pipeline: {pipeline_name}")

        logger.info(
            f"Initialized {len(self.pipelines)} pipelines: {list(self.pipelines.keys())}"
        )

    def _ensure_graph_data(self, pmc_dir: str) -> None:
        """Ensure RAG.Documents and RAG.Entities are populated from PMC files for consistent evaluation."""
        try:
            # Prefer GraphRAG for loading, else fall back to HybridGraphRAG
            target_name = (
                "GraphRAG"
                if "GraphRAG" in self.pipelines
                else ("HybridGraphRAG" if "HybridGraphRAG" in self.pipelines else None)
            )
            if not target_name:
                logger.info(
                    "No GraphRAG/HybridGraphRAG pipeline available for dataset preparation; skipping."
                )
                return

            target = self.pipelines[target_name]

            # Check if knowledge graph already has entities
            entity_count = 0
            try:
                conn = target.connection_manager.get_connection()
                if conn:
                    cur = conn.cursor()
                    try:
                        cur.execute("SELECT COUNT(*) FROM RAG.Entities")
                        row = cur.fetchone()
                        entity_count = int(row[0]) if row else 0
                    finally:
                        cur.close()
            except Exception as e:
                logger.info(f"Could not query entity count (will attempt to load): {e}")

            if entity_count > 0:
                logger.info(
                    f"Knowledge graph already populated with {entity_count} entities; skipping document load."
                )
                return

            logger.info(f"Populating knowledge graph from PMC directory: {pmc_dir}")
            target.load_documents(pmc_dir)
            logger.info("Document load and entity extraction completed.")
        except Exception as e:
            logger.warning(f"Dataset preparation failed: {e}")

    def _get_cache_key(self, pipeline_name: str, query: str) -> str:
        """Generate cache key for query results."""
        combined = f"{pipeline_name}:{query}"
        return hashlib.md5(combined.encode()).hexdigest()

    def _load_cache(self, cache_key: str) -> Optional[Dict[str, Any]]:
        """Load cached results if available."""
        if not self.config.use_cache:
            return None

        cache_file = self.cache_dir / f"{cache_key}.json"
        if cache_file.exists():
            try:
                with open(cache_file, "r") as f:
                    return json.load(f)
            except Exception as e:
                logger.warning(f"Failed to load cache {cache_key}: {e}")
        return None

    def _save_cache(self, cache_key: str, result: Dict[str, Any]):
        """Save results to cache."""
        if not self.config.use_cache:
            return

        cache_file = self.cache_dir / f"{cache_key}.json"
        try:
            with open(cache_file, "w") as f:
                json.dump(result, f, indent=2, default=str)
        except Exception as e:
            logger.warning(f"Failed to save cache {cache_key}: {e}")

    def _execute_pipeline_query(self, pipeline_name: str, query: str) -> Dict[str, Any]:
        """Execute a single query on a pipeline."""
        cache_key = self._get_cache_key(pipeline_name, query)

        # Try cache first
        cached_result = self._load_cache(cache_key)
        if cached_result:
            logger.debug(f"Using cached result for {pipeline_name}: {query[:50]}...")
            return cached_result

        try:
            pipeline = self.pipelines[pipeline_name]
            start_time = time.time()

            # Execute pipeline query
            result = pipeline.query(query, generate_answer=True)

            execution_time = time.time() - start_time

            # Standardize result format
            standardized_result = {
                "pipeline_name": pipeline_name,
                "query": query,
                "answer": result.get("answer", ""),
                "contexts": result.get("contexts", []),
                "execution_time": execution_time,
                "success": True,
                "error": None,
            }

            # Cache the result
            self._save_cache(cache_key, standardized_result)

            return standardized_result

        except Exception as e:
            error_result = {
                "pipeline_name": pipeline_name,
                "query": query,
                "answer": "",
                "contexts": [],
                "execution_time": 0.0,
                "success": False,
                "error": str(e),
            }
            logger.error(
                f"Pipeline {pipeline_name} failed for query '{query[:50]}...': {e}"
            )
            return error_result

    def _execute_pipeline_queries(
        self, pipeline_name: str, queries: List[str]
    ) -> List[Dict[str, Any]]:
        """Execute all queries for a single pipeline."""
        logger.info(f"Executing {len(queries)} queries for {pipeline_name}")
        results = []

        if self.config.parallel_execution:
            with ThreadPoolExecutor(max_workers=4) as executor:
                future_to_query = {
                    executor.submit(
                        self._execute_pipeline_query, pipeline_name, query
                    ): query
                    for query in queries
                }

                for future in as_completed(future_to_query):
                    result = future.result()
                    results.append(result)
        else:
            for i, query in enumerate(queries):
                logger.info(
                    f"Processing query {i+1}/{len(queries)} for {pipeline_name}"
                )
                result = self._execute_pipeline_query(pipeline_name, query)
                results.append(result)

        successful_queries = sum(1 for r in results if r["success"])
        logger.info(
            f"{pipeline_name}: {successful_queries}/{len(queries)} queries successful"
        )

        return results

    def _calculate_metric_statistics(self, scores: List[float]) -> Dict[str, float]:
        """Calculate comprehensive statistics for a metric."""
        if not scores:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "confidence_95": [0.0, 0.0],
                "median": 0.0,
            }

        scores_array = np.array(scores)
        mean_score = float(np.mean(scores_array))
        std_score = float(np.std(scores_array))

        # Calculate confidence interval
        confidence_interval = self._calculate_confidence_interval(scores)

        return {
            "mean": mean_score,
            "std": std_score,
            "min": float(np.min(scores_array)),
            "max": float(np.max(scores_array)),
            "median": float(np.median(scores_array)),
            "confidence_95": confidence_interval,
        }

    def _calculate_confidence_interval(self, scores: List[float]) -> List[float]:
        """Calculate 95% confidence interval."""
        if len(scores) < 2:
            return [0.0, 0.0]

        try:
            # Use t-distribution for small samples
            from scipy.stats import t

            mean = np.mean(scores)
            sem = np.std(scores, ddof=1) / np.sqrt(len(scores))
            confidence = self.config.confidence_level
            alpha = 1 - confidence
            dof = len(scores) - 1
            t_val = t.ppf(1 - alpha / 2, dof)

            margin_error = t_val * sem
            return [float(mean - margin_error), float(mean + margin_error)]
        except Exception:
            # Fallback to normal approximation
            mean = np.mean(scores)
            std = np.std(scores)
            margin = 1.96 * std / np.sqrt(len(scores))
            return [float(mean - margin), float(mean + margin)]

    def _evaluate_pipeline_with_ragas(
        self, pipeline_name: str, pipeline_results: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Evaluate a pipeline using RAGAS metrics."""
        logger.info(f"Running RAGAS evaluation for {pipeline_name}")

        # Filter successful results
        successful_results = [
            r for r in pipeline_results if r["success"] and r["answer"]
        ]

        if not successful_results:
            logger.warning(f"No successful results for {pipeline_name}")
            return self._create_empty_metrics()

        # Prepare data for RAGAS evaluation
        queries = [r["query"] for r in successful_results]
        ground_truth_answers = GROUND_TRUTH_ANSWERS[: len(successful_results)]

        # Use RAGAS framework
        try:
            ragas_results = self.ragas_framework.evaluate_pipeline(
                successful_results, ground_truth_answers, pipeline_name
            )

            # Convert to our format
            metrics = {}
            for metric_name in [
                "answer_correctness",
                "faithfulness",
                "context_precision",
                "context_recall",
                "answer_relevancy",
            ]:
                metric_result = getattr(ragas_results, metric_name)
                if metric_result and metric_result.error_message is None:
                    metrics[metric_name] = self._calculate_metric_statistics(
                        metric_result.individual_scores
                    )
                else:
                    metrics[metric_name] = self._create_empty_metric_stats()

            return metrics

        except Exception as e:
            logger.error(f"RAGAS evaluation failed for {pipeline_name}: {e}")
            return self._create_empty_metrics()

    def _create_empty_metric_stats(self) -> Dict[str, float]:
        """Create empty metric statistics."""
        return {
            "mean": 0.0,
            "std": 0.0,
            "min": 0.0,
            "max": 0.0,
            "confidence_95": [0.0, 0.0],
            "median": 0.0,
        }

    def _create_empty_metrics(self) -> Dict[str, Any]:
        """Create empty metrics structure."""
        empty_stats = self._create_empty_metric_stats()
        return {
            "answer_correctness": empty_stats,
            "faithfulness": empty_stats,
            "context_precision": empty_stats,
            "context_recall": empty_stats,
            "answer_relevancy": empty_stats,
        }

    def _generate_comparative_analysis(
        self, results: Dict[str, Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Generate comparative analysis across pipelines."""
        if not results:
            return {"best_overall": "None", "best_per_metric": {}}

        # Find best pipeline per metric
        best_per_metric = {}
        metric_names = [
            "answer_correctness",
            "faithfulness",
            "context_precision",
            "context_recall",
            "answer_relevancy",
        ]

        for metric in metric_names:
            best_score = -1
            best_pipeline = None

            for pipeline_name, pipeline_metrics in results.items():
                score = pipeline_metrics.get(metric, {}).get("mean", 0)
                if score > best_score:
                    best_score = score
                    best_pipeline = pipeline_name

            best_per_metric[metric] = best_pipeline or "None"

        # Find best overall pipeline (weighted average)
        overall_scores = {}
        weights = {
            "answer_correctness": 0.3,
            "faithfulness": 0.25,
            "context_precision": 0.15,
            "context_recall": 0.15,
            "answer_relevancy": 0.15,
        }

        for pipeline_name, pipeline_metrics in results.items():
            weighted_score = 0
            total_weight = 0

            for metric, weight in weights.items():
                score = pipeline_metrics.get(metric, {}).get("mean", 0)
                weighted_score += score * weight
                total_weight += weight

            overall_scores[pipeline_name] = (
                weighted_score / total_weight if total_weight > 0 else 0
            )

        best_overall = (
            max(overall_scores.items(), key=lambda x: x[1])[0]
            if overall_scores
            else "None"
        )

        return {
            "best_overall": best_overall,
            "best_per_metric": best_per_metric,
            "overall_scores": overall_scores,
        }

    def _save_json_report(self, results: Dict[str, Any], timestamp: str):
        """Save JSON report."""
        json_file = self.config.output_dir / f"ragas_report_{timestamp}.json"

        with open(json_file, "w") as f:
            json.dump(results, f, indent=2, default=str)

        logger.info(f"JSON report saved: {json_file}")

    def _save_html_report(self, results: Dict[str, Any], timestamp: str):
        """Save HTML report."""
        html_file = self.config.output_dir / f"ragas_report_{timestamp}.html"

        html_content = self._generate_html_report(results)

        with open(html_file, "w") as f:
            f.write(html_content)

        logger.info(f"HTML report saved: {html_file}")

    def _generate_html_report(self, results: Dict[str, Any]) -> str:
        """Generate HTML report content."""
        metadata = results["metadata"]
        pipeline_metrics = results["pipeline_metrics"]
        comparative_analysis = results["comparative_analysis"]

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>RAGAS Evaluation Report - {metadata['timestamp']}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .header {{ background-color: #f0f0f0; padding: 20px; border-radius: 5px; }}
        .metric-table {{ width: 100%; border-collapse: collapse; margin: 20px 0; }}
        .metric-table th, .metric-table td {{ border: 1px solid #ddd; padding: 8px; text-align: center; }}
        .metric-table th {{ background-color: #4CAF50; color: white; }}
        .pipeline-section {{ margin: 30px 0; border: 1px solid #ddd; padding: 20px; border-radius: 5px; }}
        .success {{ color: green; font-weight: bold; }}
        .warning {{ color: orange; font-weight: bold; }}
        .error {{ color: red; font-weight: bold; }}
        .best {{ background-color: #d4edda; }}
    </style>
</head>
<body>
    <div class="header">
        <h1>RAGAS Evaluation Report</h1>
        <p><strong>Timestamp:</strong> {metadata['timestamp']}</p>
        <p><strong>Dataset:</strong> {metadata['dataset']}</p>
        <p><strong>Queries:</strong> {metadata['num_queries']}</p>
        <p><strong>Pipelines:</strong> {', '.join(metadata['pipelines_evaluated'])}</p>
    </div>
    
    <h2>Executive Summary</h2>
    <ul>
        <li><strong>Best Overall Pipeline:</strong> {comparative_analysis['best_overall']}</li>
        <li><strong>Target Accuracy (80%) Achievement:</strong> 
            {"<span class='success'>✓ ACHIEVED</span>" if any(
                pm.get('answer_correctness', {}).get('mean', 0) >= 0.8 
                for pm in pipeline_metrics.values()
            ) else "<span class='error'>✗ NOT ACHIEVED</span>"}
        </li>
    </ul>
    
    <h2>Pipeline Performance Comparison</h2>
    <table class="metric-table">
        <tr>
            <th>Pipeline</th>
            <th>Answer Correctness</th>
            <th>Faithfulness</th>
            <th>Context Precision</th>
            <th>Context Recall</th>
            <th>Answer Relevance</th>
        </tr>
"""

        for pipeline_name, metrics in pipeline_metrics.items():
            row_class = (
                "best" if pipeline_name == comparative_analysis["best_overall"] else ""
            )
            html += f"""
        <tr class="{row_class}">
            <td><strong>{pipeline_name}</strong></td>
            <td>{metrics.get('answer_correctness', {}).get('mean', 0):.3f} ± {metrics.get('answer_correctness', {}).get('std', 0):.3f}</td>
            <td>{metrics.get('faithfulness', {}).get('mean', 0):.3f} ± {metrics.get('faithfulness', {}).get('std', 0):.3f}</td>
            <td>{metrics.get('context_precision', {}).get('mean', 0):.3f} ± {metrics.get('context_precision', {}).get('std', 0):.3f}</td>
            <td>{metrics.get('context_recall', {}).get('mean', 0):.3f} ± {metrics.get('context_recall', {}).get('std', 0):.3f}</td>
            <td>{metrics.get('answer_relevancy', {}).get('mean', 0):.3f} ± {metrics.get('answer_relevancy', {}).get('std', 0):.3f}</td>
        </tr>
"""

        html += """
    </table>
    
    <h2>Detailed Pipeline Results</h2>
"""

        for pipeline_name, metrics in pipeline_metrics.items():
            html += f"""
    <div class="pipeline-section">
        <h3>{pipeline_name}</h3>
        <table class="metric-table">
            <tr><th>Metric</th><th>Mean</th><th>Std Dev</th><th>95% CI</th><th>Min</th><th>Max</th></tr>
"""
            for metric_name, metric_data in metrics.items():
                ci = metric_data.get("confidence_95", [0, 0])
                html += f"""
            <tr>
                <td>{metric_name.replace('_', ' ').title()}</td>
                <td>{metric_data.get('mean', 0):.4f}</td>
                <td>{metric_data.get('std', 0):.4f}</td>
                <td>[{ci[0]:.4f}, {ci[1]:.4f}]</td>
                <td>{metric_data.get('min', 0):.4f}</td>
                <td>{metric_data.get('max', 0):.4f}</td>
            </tr>
"""
            html += """
        </table>
    </div>
"""

        html += """
</body>
</html>
"""
        return html

    def run_comprehensive_evaluation(self) -> Dict[str, Any]:
        """Run the complete RAGAS evaluation pipeline."""
        logger.info("🚀 Starting Comprehensive RAGAS Evaluation")
        logger.info("=" * 80)

        start_time = time.time()
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Prepare queries
        queries = SAMPLE_QUERIES[: self.config.num_queries]
        logger.info(f"Using {len(queries)} queries for evaluation")

        # Execute all pipelines
        all_pipeline_results = {}
        pipeline_metrics = {}

        for pipeline_name in self.config.pipelines:
            if pipeline_name not in self.pipelines:
                logger.warning(f"Pipeline {pipeline_name} not available, skipping")
                continue

            logger.info(f"Evaluating pipeline: {pipeline_name}")

            # Execute queries
            pipeline_results = self._execute_pipeline_queries(pipeline_name, queries)
            all_pipeline_results[pipeline_name] = pipeline_results

            # Run RAGAS evaluation
            metrics = self._evaluate_pipeline_with_ragas(
                pipeline_name, pipeline_results
            )
            pipeline_metrics[pipeline_name] = metrics

            # Log primary metric
            answer_correctness = metrics.get("answer_correctness", {}).get("mean", 0)
            status = (
                "✅ PASS"
                if answer_correctness >= self.config.target_accuracy
                else "❌ FAIL"
            )
            logger.info(
                f"{pipeline_name} Answer Correctness: {answer_correctness:.3f} {status}"
            )

        # Generate comparative analysis
        comparative_analysis = self._generate_comparative_analysis(pipeline_metrics)

        # Calculate execution time
        total_time = time.time() - start_time

        # Compile final results
        results = {
            "metadata": {
                "timestamp": timestamp,
                "dataset": "PMC 10K documents",
                "num_queries": len(queries),
                "pipelines_evaluated": list(pipeline_metrics.keys()),
                "execution_time_minutes": total_time / 60,
                "target_accuracy": self.config.target_accuracy,
                "configuration": asdict(self.config),
            },
            "pipeline_metrics": pipeline_metrics,
            "comparative_analysis": comparative_analysis,
            "detailed_results": all_pipeline_results,
        }

        # Save reports
        self._save_json_report(results, timestamp)
        self._save_html_report(results, timestamp)

        # Print summary
        self._print_evaluation_summary(results)

        return results

    def _print_evaluation_summary(self, results: Dict[str, Any]):
        """Print evaluation summary to console."""
        metadata = results["metadata"]
        pipeline_metrics = results["pipeline_metrics"]
        comparative_analysis = results["comparative_analysis"]

        logger.info("\n" + "=" * 80)
        logger.info("🎯 RAGAS EVALUATION SUMMARY")
        logger.info("=" * 80)

        logger.info(f"📊 Dataset: {metadata['dataset']}")
        logger.info(f"🔍 Queries Evaluated: {metadata['num_queries']}")
        logger.info(
            f"⏱️  Total Execution Time: {metadata['execution_time_minutes']:.1f} minutes"
        )
        logger.info(f"🎯 Target Accuracy: {metadata['target_accuracy'] * 100}%")

        logger.info("\n📈 PIPELINE PERFORMANCE:")
        for pipeline_name, metrics in pipeline_metrics.items():
            answer_correctness = metrics.get("answer_correctness", {}).get("mean", 0)
            ci = metrics.get("answer_correctness", {}).get("confidence_95", [0, 0])

            status_icon = (
                "✅" if answer_correctness >= metadata["target_accuracy"] else "❌"
            )
            logger.info(
                f"  {status_icon} {pipeline_name}: {answer_correctness:.3f} (95% CI: [{ci[0]:.3f}, {ci[1]:.3f}])"
            )

        logger.info(
            f"\n🏆 BEST OVERALL PIPELINE: {comparative_analysis['best_overall']}"
        )

        # Check if target achieved
        target_achieved = any(
            pm.get("answer_correctness", {}).get("mean", 0)
            >= metadata["target_accuracy"]
            for pm in pipeline_metrics.values()
        )

        if target_achieved:
            logger.info("\n🎉 SUCCESS: Target accuracy (>80%) achieved!")
        else:
            logger.info(
                "\n⚠️  WARNING: Target accuracy (>80%) not achieved by any pipeline"
            )

        logger.info("=" * 80)


def get_evaluation_config() -> EvaluationConfig:
    """Get evaluation configuration from environment variables."""
    num_queries = int(os.getenv("RAGAS_NUM_QUERIES", len(SAMPLE_QUERIES)))

    # Include HybridGraphRAG in default pipeline comparison if available
    default_pipelines = "BasicRAG,BasicRerank,CRAG,GraphRAG"
    if HYBRID_GRAPHRAG_AVAILABLE:
        default_pipelines += ",HybridGraphRAG"

    pipelines_str = os.getenv("RAGAS_PIPELINES", default_pipelines)
    pipelines = [p.strip() for p in pipelines_str.split(",")]
    output_dir = Path(
        os.getenv("RAGAS_OUTPUT_DIR", "outputs/reports/ragas_evaluations")
    )
    use_cache = os.getenv("RAGAS_USE_CACHE", "true").lower() == "true"

    output_dir.mkdir(parents=True, exist_ok=True)

    return EvaluationConfig(
        num_queries=num_queries,
        pipelines=pipelines,
        output_dir=output_dir,
        use_cache=use_cache,
        parallel_execution=True,
        confidence_level=0.95,
        target_accuracy=0.80,
    )


def main():
    """Main entry point."""
    try:
        # Get configuration
        config = get_evaluation_config()

        # Create orchestrator
        orchestrator = RAGASEvaluationOrchestrator(config)

        # Run evaluation
        results = orchestrator.run_comprehensive_evaluation()

        # Return success code based on target achievement
        target_achieved = any(
            pm.get("answer_correctness", {}).get("mean", 0) >= config.target_accuracy
            for pm in results["pipeline_metrics"].values()
        )

        return 0 if target_achieved else 1

    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        import traceback

        traceback.print_exc()
        return 1


if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
