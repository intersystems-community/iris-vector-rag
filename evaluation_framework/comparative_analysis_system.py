"""
Comprehensive Comparative Analysis System for Biomedical RAG Pipeline Evaluation

This module integrates biomedical question generation, RAGAS metrics evaluation,
statistical analysis, and visualization to provide comprehensive pipeline comparison
with empirical evidence and actionable insights.

Key Features:
1. End-to-end pipeline comparison workflow
2. Integrated question generation and evaluation
3. Multi-dimensional performance analysis
4. Statistical significance testing
5. Interactive visualizations and reporting
6. Scalable architecture for 10K+ documents
7. Empirical evidence generation
"""

import json
import logging
import pickle
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

# Import our custom frameworks
from biomedical_question_generator import (
    BiomedicalQuestion,
    BiomedicalQuestionGenerator,
    QuestionGenerationConfig,
    create_biomedical_question_generator,
)
from plotly.subplots import make_subplots
from ragas_metrics_framework import (
    BiomedicalRAGASFramework,
    ComprehensiveRAGASResults,
    create_biomedical_ragas_framework,
)
from statistical_evaluation_methodology import (
    PowerAnalysisResult,
    StatisticalEvaluationFramework,
    StatisticalTestResult,
    create_statistical_framework,
)

# Pipeline imports (existing infrastructure)
from iris_rag.pipelines.basic import BasicRAGPipeline
from iris_rag.pipelines.basic_rerank import BasicRAGRerankingPipeline
from iris_rag.pipelines.colbert_pylate.pylate_pipeline import PyLateColBERTPipeline
from iris_rag.pipelines.crag import CRAGPipeline
from iris_rag.pipelines.graphrag import GraphRAGPipeline

logger = logging.getLogger(__name__)


@dataclass
class PipelineEvaluationConfig:
    """Configuration for pipeline evaluation experiment."""

    # Data configuration
    total_questions: int = 1000
    questions_per_pipeline_test: int = 100
    sample_documents: int = 10000

    # Question generation
    question_complexity_distribution: Dict[str, float] = field(
        default_factory=lambda: {"basic": 0.4, "intermediate": 0.4, "advanced": 0.2}
    )
    question_types_distribution: Dict[str, float] = field(
        default_factory=lambda: {
            "factual": 0.3,
            "analytical": 0.25,
            "procedural": 0.2,
            "comparative": 0.15,
            "causal": 0.1,
        }
    )

    # Evaluation configuration
    ragas_metrics: List[str] = field(
        default_factory=lambda: [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "context_utilization",
            "answer_correctness",
            "answer_similarity",
        ]
    )

    # Statistical configuration
    statistical_alpha: float = 0.05
    statistical_power_target: float = 0.8
    multiple_correction_method: str = "fdr_bh"

    # Computational configuration
    max_workers: int = 4
    batch_size: int = 32
    enable_caching: bool = True
    cache_dir: str = "cache/evaluation"

    # Output configuration
    output_dir: str = "outputs/comparative_analysis"
    generate_plots: bool = True
    generate_interactive_dashboard: bool = True


@dataclass
class PipelinePerformanceResult:
    """Complete performance evaluation for a single pipeline."""

    pipeline_name: str
    ragas_results: ComprehensiveRAGASResults
    execution_times: List[float]
    success_rate: float
    error_messages: List[str]
    sample_responses: List[Dict[str, Any]]

    def get_metric_value(self, metric_name: str) -> Optional[float]:
        """Get specific metric value."""
        metric_result = getattr(self.ragas_results, metric_name, None)
        return metric_result.value if metric_result else None

    def get_metric_confidence_interval(
        self, metric_name: str
    ) -> Optional[Tuple[float, float]]:
        """Get metric confidence interval."""
        metric_result = getattr(self.ragas_results, metric_name, None)
        return metric_result.confidence_interval if metric_result else None


@dataclass
class ComparativeAnalysisResults:
    """Complete comparative analysis results across all pipelines."""

    experiment_id: str
    timestamp: str
    config: PipelineEvaluationConfig

    # Pipeline results
    pipeline_results: Dict[str, PipelinePerformanceResult]

    # Questions and ground truth
    evaluation_questions: List[BiomedicalQuestion]

    # Statistical analysis
    statistical_comparisons: Dict[str, List[StatisticalTestResult]]
    power_analysis: Dict[str, PowerAnalysisResult]

    # Summary metrics
    overall_rankings: Dict[
        str, List[Tuple[str, float]]
    ]  # metric -> [(pipeline, score), ...]
    best_pipeline_per_metric: Dict[str, str]
    statistical_significance_summary: Dict[str, Dict[str, Any]]

    # Performance metadata
    total_evaluation_time: float
    total_questions_evaluated: int
    successful_evaluations_per_pipeline: Dict[str, int]


class ComparativeAnalysisSystem:
    """
    Comprehensive system for comparing biomedical RAG pipelines with statistical rigor.

    Orchestrates question generation, pipeline evaluation, RAGAS metrics calculation,
    statistical analysis, and visualization generation for complete pipeline comparison.
    """

    def __init__(self, config: Optional[PipelineEvaluationConfig] = None):
        """
        Initialize the comparative analysis system.

        Args:
            config: Configuration for evaluation experiment
        """
        self.config = config or PipelineEvaluationConfig()

        # Initialize sub-frameworks
        self.question_generator = None
        self.ragas_framework = None
        self.statistical_framework = None

        # Pipeline registry
        self.available_pipelines = {
            "BasicRAG": BasicRAGPipeline,
            "CRAG": CRAGPipeline,
            "GraphRAG": GraphRAGPipeline,
            "BasicRAGReranking": BasicRAGRerankingPipeline,
        }

        self.initialized_pipelines = {}

        # Setup output directories
        self.output_dir = Path(self.config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Setup cache
        if self.config.enable_caching:
            self.cache_dir = Path(self.config.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)

        self._initialize_frameworks()

        logger.info("Comparative Analysis System initialized successfully")

    def _initialize_frameworks(self):
        """Initialize all sub-frameworks."""
        logger.info("Initializing evaluation frameworks...")

        # Question generator
        question_config = QuestionGenerationConfig(
            total_questions=self.config.total_questions,
            question_types_distribution=self.config.question_types_distribution,
            complexity_distribution=self.config.question_complexity_distribution,
        )
        self.question_generator = BiomedicalQuestionGenerator(question_config)

        # RAGAS framework
        ragas_config = {
            "batch_size": self.config.batch_size,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }
        self.ragas_framework = create_biomedical_ragas_framework(ragas_config)

        # Statistical framework
        statistical_config = {
            "alpha": self.config.statistical_alpha,
            "power_target": self.config.statistical_power_target,
            "multiple_correction_method": self.config.multiple_correction_method,
        }
        self.statistical_framework = create_statistical_framework(statistical_config)

        logger.info("All frameworks initialized successfully")

    def run_comprehensive_comparison(
        self,
        documents: List[Dict[str, Any]],
        pipeline_names: Optional[List[str]] = None,
    ) -> ComparativeAnalysisResults:
        """
        Run comprehensive comparison of RAG pipelines.

        Args:
            documents: List of PMC documents for evaluation
            pipeline_names: List of pipeline names to evaluate (default: all available)

        Returns:
            Complete comparative analysis results
        """
        experiment_id = (
            f"biomedical_rag_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )
        logger.info(f"Starting comprehensive comparison: {experiment_id}")

        start_time = time.time()

        # Default to all available pipelines
        if pipeline_names is None:
            pipeline_names = list(self.available_pipelines.keys())

        # Initialize pipelines
        self._initialize_pipelines(pipeline_names)

        # Step 1: Generate evaluation questions
        logger.info("Generating biomedical evaluation questions...")
        evaluation_questions = self._generate_evaluation_questions(documents)

        # Step 2: Evaluate all pipelines
        logger.info("Evaluating all pipelines...")
        pipeline_results = self._evaluate_all_pipelines(evaluation_questions)

        # Step 3: Statistical analysis
        logger.info("Conducting statistical analysis...")
        statistical_results = self._conduct_statistical_analysis(pipeline_results)

        # Step 4: Generate summary metrics
        logger.info("Generating summary metrics...")
        summary_metrics = self._generate_summary_metrics(
            pipeline_results, statistical_results
        )

        # Create comprehensive results
        total_time = time.time() - start_time

        results = ComparativeAnalysisResults(
            experiment_id=experiment_id,
            timestamp=datetime.now().isoformat(),
            config=self.config,
            pipeline_results=pipeline_results,
            evaluation_questions=evaluation_questions,
            statistical_comparisons=statistical_results["comparisons"],
            power_analysis=statistical_results["power_analysis"],
            overall_rankings=summary_metrics["rankings"],
            best_pipeline_per_metric=summary_metrics["best_per_metric"],
            statistical_significance_summary=statistical_results[
                "significance_summary"
            ],
            total_evaluation_time=total_time,
            total_questions_evaluated=len(evaluation_questions),
            successful_evaluations_per_pipeline=summary_metrics["success_rates"],
        )

        # Step 5: Save results and generate reports
        self._save_results(results)

        if self.config.generate_plots:
            self._generate_visualizations(results)

        if self.config.generate_interactive_dashboard:
            self._generate_interactive_dashboard(results)

        logger.info(f"Comprehensive comparison completed in {total_time:.2f}s")
        return results

    def _initialize_pipelines(self, pipeline_names: List[str]):
        """Initialize specified RAG pipelines."""
        logger.info(f"Initializing {len(pipeline_names)} pipelines...")

        for pipeline_name in pipeline_names:
            if pipeline_name not in self.available_pipelines:
                logger.warning(f"Pipeline {pipeline_name} not available")
                continue

            try:
                # Initialize with appropriate configuration
                pipeline_class = self.available_pipelines[pipeline_name]

                # Basic configuration for all pipelines
                pipeline_config = {
                    "connection_manager": None,  # Will use defaults
                    "config_manager": None,  # Will use defaults
                    "llm_func": self._get_mock_llm_function(),  # Mock for evaluation
                    "vector_store": None,  # Will use defaults
                }

                pipeline = pipeline_class(**pipeline_config)
                self.initialized_pipelines[pipeline_name] = pipeline

                logger.info(f"Successfully initialized {pipeline_name}")

            except Exception as e:
                logger.error(f"Failed to initialize {pipeline_name}: {e}")

    def _get_mock_llm_function(self):
        """Get mock LLM function for evaluation."""

        def mock_llm(prompt: str) -> str:
            # Simple mock that generates relevant medical responses
            if "diabetes" in prompt.lower():
                return "Diabetes is a chronic condition affecting blood sugar regulation. Common symptoms include increased thirst, frequent urination, and fatigue. Treatment typically involves lifestyle modifications and medication management."
            elif "heart" in prompt.lower() or "cardiac" in prompt.lower():
                return "Cardiovascular conditions affect the heart and blood vessels. Risk factors include hypertension, high cholesterol, and diabetes. Prevention strategies include regular exercise and healthy diet."
            else:
                return "This is a medical condition that requires proper diagnosis and treatment by healthcare professionals. Symptoms and treatment options vary based on individual patient factors."

        return mock_llm

    def _generate_evaluation_questions(
        self, documents: List[Dict[str, Any]]
    ) -> List[BiomedicalQuestion]:
        """Generate biomedical questions for evaluation."""

        # Check cache first
        cache_file = (
            self.cache_dir / "evaluation_questions.pkl"
            if self.config.enable_caching
            else None
        )

        if cache_file and cache_file.exists():
            logger.info("Loading cached evaluation questions")
            with open(cache_file, "rb") as f:
                return pickle.load(f)

        # Filter to biomedical documents (sample for efficiency)
        sampled_docs = documents[: self.config.sample_documents]

        # Generate questions
        questions = self.question_generator.generate_questions_from_documents(
            sampled_docs
        )

        # Cache results
        if cache_file:
            with open(cache_file, "wb") as f:
                pickle.dump(questions, f)
            logger.info(f"Cached {len(questions)} evaluation questions")

        return questions

    def _evaluate_all_pipelines(
        self, evaluation_questions: List[BiomedicalQuestion]
    ) -> Dict[str, PipelinePerformanceResult]:
        """Evaluate all pipelines on the question set."""

        pipeline_results = {}

        # Use subset of questions for pipeline testing
        test_questions = evaluation_questions[: self.config.questions_per_pipeline_test]

        for pipeline_name, pipeline in self.initialized_pipelines.items():
            logger.info(f"Evaluating {pipeline_name}...")

            try:
                result = self._evaluate_single_pipeline(
                    pipeline, pipeline_name, test_questions
                )
                pipeline_results[pipeline_name] = result

                logger.info(
                    f"Completed evaluation of {pipeline_name}: "
                    f"Success rate: {result.success_rate:.2%}"
                )

            except Exception as e:
                logger.error(f"Failed to evaluate {pipeline_name}: {e}")
                # Create failed result
                pipeline_results[pipeline_name] = PipelinePerformanceResult(
                    pipeline_name=pipeline_name,
                    ragas_results=None,
                    execution_times=[],
                    success_rate=0.0,
                    error_messages=[str(e)],
                    sample_responses=[],
                )

        return pipeline_results

    def _evaluate_single_pipeline(
        self, pipeline: Any, pipeline_name: str, questions: List[BiomedicalQuestion]
    ) -> PipelinePerformanceResult:
        """Evaluate a single pipeline on the question set."""

        pipeline_responses = []
        execution_times = []
        error_messages = []
        ground_truth_answers = []

        # Execute pipeline on each question
        for question in questions:
            try:
                start_time = time.time()

                # Query pipeline
                response = pipeline.query(
                    question.question, top_k=5, generate_answer=True
                )

                execution_time = time.time() - start_time
                execution_times.append(execution_time)

                # Store response
                pipeline_responses.append(response)
                ground_truth_answers.append(question.ground_truth_answer)

            except Exception as e:
                error_messages.append(
                    f"Question '{question.question[:50]}...': {str(e)}"
                )
                logger.warning(f"Pipeline {pipeline_name} failed on question: {e}")

        # Calculate success rate
        success_rate = len(pipeline_responses) / len(questions)

        # Conduct RAGAS evaluation
        if pipeline_responses:
            ragas_results = self.ragas_framework.evaluate_pipeline(
                pipeline_responses, ground_truth_answers, pipeline_name
            )
        else:
            ragas_results = None

        # Sample responses for analysis
        sample_responses = pipeline_responses[: min(10, len(pipeline_responses))]

        return PipelinePerformanceResult(
            pipeline_name=pipeline_name,
            ragas_results=ragas_results,
            execution_times=execution_times,
            success_rate=success_rate,
            error_messages=error_messages,
            sample_responses=sample_responses,
        )

    def _conduct_statistical_analysis(
        self, pipeline_results: Dict[str, PipelinePerformanceResult]
    ) -> Dict[str, Any]:
        """Conduct comprehensive statistical analysis."""

        statistical_results = {
            "comparisons": {},
            "power_analysis": {},
            "significance_summary": {},
        }

        # Extract metric data for each pipeline
        for metric_name in self.config.ragas_metrics:
            metric_data = {}

            for pipeline_name, result in pipeline_results.items():
                if result.ragas_results:
                    metric_result = getattr(result.ragas_results, metric_name, None)
                    if metric_result and metric_result.individual_scores:
                        metric_data[pipeline_name] = metric_result.individual_scores

            if len(metric_data) >= 2:
                # Power analysis
                try:
                    power_result = self.statistical_framework.conduct_power_analysis(
                        metric_data, metric_name
                    )
                    statistical_results["power_analysis"][metric_name] = power_result
                except Exception as e:
                    logger.warning(f"Power analysis failed for {metric_name}: {e}")

                # Statistical comparisons
                try:
                    comparison_results = (
                        self.statistical_framework.compare_pipelines_statistical(
                            metric_data, metric_name, paired=False
                        )
                    )
                    statistical_results["comparisons"][metric_name] = comparison_results

                    # Summarize significance
                    significant_tests = [
                        r
                        for r in comparison_results
                        if (r.p_value_adjusted or r.p_value)
                        < self.config.statistical_alpha
                    ]

                    statistical_results["significance_summary"][metric_name] = {
                        "total_tests": len(comparison_results),
                        "significant_tests": len(significant_tests),
                        "largest_effect_size": max(
                            [abs(r.effect_size) for r in comparison_results],
                            default=0.0,
                        ),
                        "significant_comparisons": [
                            {
                                "comparison": f"{r.raw_data.get('group1_name', 'unknown')} vs {r.raw_data.get('group2_name', 'unknown')}",
                                "p_value": r.p_value_adjusted or r.p_value,
                                "effect_size": r.effect_size,
                            }
                            for r in significant_tests
                        ],
                    }

                except Exception as e:
                    logger.warning(
                        f"Statistical comparison failed for {metric_name}: {e}"
                    )

        return statistical_results

    def _generate_summary_metrics(
        self,
        pipeline_results: Dict[str, PipelinePerformanceResult],
        statistical_results: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Generate summary metrics and rankings."""

        summary = {"rankings": {}, "best_per_metric": {}, "success_rates": {}}

        # Calculate success rates
        for pipeline_name, result in pipeline_results.items():
            summary["success_rates"][pipeline_name] = (
                result.ragas_results.successful_evaluations
                if result.ragas_results
                else 0
            )

        # Generate rankings for each metric
        for metric_name in self.config.ragas_metrics:
            metric_scores = []

            for pipeline_name, result in pipeline_results.items():
                if result.ragas_results:
                    metric_value = result.get_metric_value(metric_name)
                    if metric_value is not None:
                        metric_scores.append((pipeline_name, metric_value))

            # Sort by score (descending)
            metric_scores.sort(key=lambda x: x[1], reverse=True)
            summary["rankings"][metric_name] = metric_scores

            # Best pipeline for this metric
            if metric_scores:
                summary["best_per_metric"][metric_name] = metric_scores[0][0]

        return summary

    def _save_results(self, results: ComparativeAnalysisResults):
        """Save comprehensive results to files."""

        # Create experiment directory
        experiment_dir = self.output_dir / results.experiment_id
        experiment_dir.mkdir(exist_ok=True)

        # Save main results
        results_file = experiment_dir / "comprehensive_results.json"

        # Convert to serializable format
        serializable_results = {
            "experiment_id": results.experiment_id,
            "timestamp": results.timestamp,
            "config": {
                "total_questions": results.config.total_questions,
                "questions_per_pipeline_test": results.config.questions_per_pipeline_test,
                "sample_documents": results.config.sample_documents,
                "statistical_alpha": results.config.statistical_alpha,
                "ragas_metrics": results.config.ragas_metrics,
            },
            "pipeline_results": {
                name: {
                    "pipeline_name": result.pipeline_name,
                    "success_rate": result.success_rate,
                    "avg_execution_time": (
                        np.mean(result.execution_times)
                        if result.execution_times
                        else 0.0
                    ),
                    "ragas_scores": (
                        result.ragas_results.to_dict() if result.ragas_results else {}
                    ),
                    "error_count": len(result.error_messages),
                }
                for name, result in results.pipeline_results.items()
            },
            "overall_rankings": results.overall_rankings,
            "best_pipeline_per_metric": results.best_pipeline_per_metric,
            "statistical_significance_summary": results.statistical_significance_summary,
            "total_evaluation_time": results.total_evaluation_time,
            "total_questions_evaluated": results.total_questions_evaluated,
        }

        with open(results_file, "w") as f:
            json.dump(serializable_results, f, indent=2, default=str)

        # Save detailed statistical results
        stats_file = experiment_dir / "statistical_analysis.json"
        with open(stats_file, "w") as f:
            json.dump(results.statistical_comparisons, f, indent=2, default=str)

        # Save evaluation questions
        questions_file = experiment_dir / "evaluation_questions.json"
        questions_data = [
            {
                "question_id": q.question_id,
                "question": q.question,
                "ground_truth_answer": q.ground_truth_answer,
                "question_type": q.question_type,
                "complexity_level": q.complexity_level,
                "medical_domain": q.medical_domain,
                "confidence_score": q.confidence_score,
            }
            for q in results.evaluation_questions
        ]
        with open(questions_file, "w") as f:
            json.dump(questions_data, f, indent=2)

        logger.info(f"Results saved to {experiment_dir}")

    def _generate_visualizations(self, results: ComparativeAnalysisResults):
        """Generate comprehensive visualizations."""

        experiment_dir = self.output_dir / results.experiment_id
        plots_dir = experiment_dir / "plots"
        plots_dir.mkdir(exist_ok=True)

        # Set style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

        # 1. RAGAS Metrics Comparison (Radar Chart)
        self._create_radar_chart(results, plots_dir)

        # 2. Statistical Significance Heatmap
        self._create_significance_heatmap(results, plots_dir)

        # 3. Performance Distribution Plots
        self._create_performance_distributions(results, plots_dir)

        # 4. Execution Time Analysis
        self._create_execution_time_analysis(results, plots_dir)

        # 5. Overall Performance Summary
        self._create_performance_summary(results, plots_dir)

        logger.info(f"Visualizations saved to {plots_dir}")

    def _create_radar_chart(self, results: ComparativeAnalysisResults, plots_dir: Path):
        """Create radar chart comparing all metrics across pipelines."""

        # Prepare data
        pipelines = list(results.pipeline_results.keys())
        metrics = self.config.ragas_metrics

        # Create subplot
        fig = make_subplots(rows=1, cols=1, specs=[[{"type": "scatterpolar"}]])

        colors = px.colors.qualitative.Set1

        for i, pipeline_name in enumerate(pipelines):
            result = results.pipeline_results[pipeline_name]
            if not result.ragas_results:
                continue

            values = []
            for metric in metrics:
                metric_value = result.get_metric_value(metric)
                values.append(metric_value if metric_value is not None else 0.0)

            # Add pipeline trace
            fig.add_trace(
                go.Scatterpolar(
                    r=values,
                    theta=metrics,
                    fill="toself",
                    name=pipeline_name,
                    line_color=colors[i % len(colors)],
                )
            )

        fig.update_layout(
            polar=dict(radialaxis=dict(visible=True, range=[0, 1])),
            showlegend=True,
            title="RAGAS Metrics Comparison Across Pipelines",
        )

        fig.write_html(plots_dir / "ragas_radar_chart.html")
        fig.write_image(plots_dir / "ragas_radar_chart.png")

    def _create_significance_heatmap(
        self, results: ComparativeAnalysisResults, plots_dir: Path
    ):
        """Create heatmap showing statistical significance between pipelines."""

        pipelines = list(results.pipeline_results.keys())
        metrics = self.config.ragas_metrics

        # Create significance matrix for each metric
        fig, axes = plt.subplots(2, 4, figsize=(20, 10))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            if idx >= len(axes):
                break

            # Create pairwise significance matrix
            n_pipelines = len(pipelines)
            sig_matrix = np.ones((n_pipelines, n_pipelines))

            if metric in results.statistical_comparisons:
                comparisons = results.statistical_comparisons[metric]

                for comparison in comparisons:
                    if hasattr(comparison, "raw_data"):
                        group1 = comparison.raw_data.get("group1_name")
                        group2 = comparison.raw_data.get("group2_name")

                        if group1 in pipelines and group2 in pipelines:
                            i = pipelines.index(group1)
                            j = pipelines.index(group2)

                            p_val = comparison.p_value_adjusted or comparison.p_value
                            sig_matrix[i, j] = p_val
                            sig_matrix[j, i] = p_val

            # Plot heatmap
            sns.heatmap(
                sig_matrix,
                annot=True,
                fmt=".3f",
                xticklabels=pipelines,
                yticklabels=pipelines,
                cmap="RdYlBu_r",
                vmin=0,
                vmax=0.1,
                ax=axes[idx],
            )
            axes[idx].set_title(f"{metric} - p-values")

        # Remove extra subplots
        for idx in range(len(metrics), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.savefig(
            plots_dir / "statistical_significance_heatmap.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _create_performance_distributions(
        self, results: ComparativeAnalysisResults, plots_dir: Path
    ):
        """Create distribution plots for each metric."""

        metrics = self.config.ragas_metrics
        pipelines = list(results.pipeline_results.keys())

        fig, axes = plt.subplots(3, 3, figsize=(18, 15))
        axes = axes.flatten()

        for idx, metric in enumerate(metrics):
            if idx >= len(axes):
                break

            # Collect data for violin plot
            plot_data = []

            for pipeline_name in pipelines:
                result = results.pipeline_results[pipeline_name]
                if result.ragas_results:
                    metric_result = getattr(result.ragas_results, metric, None)
                    if metric_result and metric_result.individual_scores:
                        for score in metric_result.individual_scores:
                            plot_data.append(
                                {
                                    "Pipeline": pipeline_name,
                                    "Score": score,
                                    "Metric": metric,
                                }
                            )

            if plot_data:
                df = pd.DataFrame(plot_data)
                sns.violinplot(data=df, x="Pipeline", y="Score", ax=axes[idx])
                axes[idx].set_title(f"{metric} Distribution")
                axes[idx].tick_params(axis="x", rotation=45)

        # Remove extra subplots
        for idx in range(len(metrics), len(axes)):
            fig.delaxes(axes[idx])

        plt.tight_layout()
        plt.savefig(
            plots_dir / "performance_distributions.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def _create_execution_time_analysis(
        self, results: ComparativeAnalysisResults, plots_dir: Path
    ):
        """Create execution time analysis plots."""

        pipelines = list(results.pipeline_results.keys())

        # Prepare data
        time_data = []
        for pipeline_name in pipelines:
            result = results.pipeline_results[pipeline_name]
            for time_val in result.execution_times:
                time_data.append(
                    {"Pipeline": pipeline_name, "Execution_Time": time_val}
                )

        if time_data:
            df = pd.DataFrame(time_data)

            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))

            # Box plot
            sns.boxplot(data=df, x="Pipeline", y="Execution_Time", ax=ax1)
            ax1.set_title("Execution Time Distribution by Pipeline")
            ax1.tick_params(axis="x", rotation=45)

            # Bar plot of mean times
            mean_times = df.groupby("Pipeline")["Execution_Time"].mean()
            mean_times.plot(kind="bar", ax=ax2)
            ax2.set_title("Average Execution Time by Pipeline")
            ax2.tick_params(axis="x", rotation=45)

            plt.tight_layout()
            plt.savefig(
                plots_dir / "execution_time_analysis.png", dpi=300, bbox_inches="tight"
            )
            plt.close()

    def _create_performance_summary(
        self, results: ComparativeAnalysisResults, plots_dir: Path
    ):
        """Create overall performance summary visualization."""

        pipelines = list(results.pipeline_results.keys())
        metrics = self.config.ragas_metrics

        # Create summary matrix
        summary_data = []

        for pipeline_name in pipelines:
            result = results.pipeline_results[pipeline_name]
            row_data = {"Pipeline": pipeline_name}

            for metric in metrics:
                metric_value = (
                    result.get_metric_value(metric) if result.ragas_results else 0.0
                )
                row_data[metric] = metric_value if metric_value is not None else 0.0

            # Add overall score
            row_data["Overall"] = (
                result.ragas_results.overall_score if result.ragas_results else 0.0
            )
            row_data["Success_Rate"] = result.success_rate

            summary_data.append(row_data)

        df = pd.DataFrame(summary_data)

        # Create clustered bar chart
        x = np.arange(len(pipelines))
        width = 0.1

        fig, ax = plt.subplots(figsize=(16, 8))

        for i, metric in enumerate(metrics + ["Overall"]):
            values = df[metric].values
            ax.bar(x + i * width, values, width, label=metric)

        ax.set_xlabel("Pipelines")
        ax.set_ylabel("Scores")
        ax.set_title("Comprehensive Performance Summary")
        ax.set_xticks(x + width * len(metrics) / 2)
        ax.set_xticklabels(pipelines)
        ax.legend(bbox_to_anchor=(1.05, 1), loc="upper left")
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        plt.savefig(plots_dir / "performance_summary.png", dpi=300, bbox_inches="tight")
        plt.close()

    def _generate_interactive_dashboard(self, results: ComparativeAnalysisResults):
        """Generate interactive dashboard using Plotly."""

        experiment_dir = self.output_dir / results.experiment_id

        # Create comprehensive dashboard HTML
        dashboard_html = self._create_dashboard_html(results)

        dashboard_file = experiment_dir / "interactive_dashboard.html"
        with open(dashboard_file, "w") as f:
            f.write(dashboard_html)

        logger.info(f"Interactive dashboard saved to {dashboard_file}")

    def _create_dashboard_html(self, results: ComparativeAnalysisResults) -> str:
        """Create HTML dashboard with all visualizations."""

        html_template = """
        <!DOCTYPE html>
        <html>
        <head>
            <title>Biomedical RAG Pipeline Comparison Dashboard</title>
            <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
            <style>
                body { font-family: Arial, sans-serif; margin: 20px; }
                .metric-card { 
                    border: 1px solid #ddd; 
                    border-radius: 8px; 
                    padding: 15px; 
                    margin: 10px 0; 
                    background-color: #f9f9f9; 
                }
                .highlight { 
                    background-color: #e7f3ff; 
                    padding: 10px; 
                    border-radius: 5px; 
                    margin: 10px 0; 
                }
                table { 
                    border-collapse: collapse; 
                    width: 100%; 
                    margin: 20px 0; 
                }
                th, td { 
                    border: 1px solid #ddd; 
                    padding: 8px; 
                    text-align: center; 
                }
                th { background-color: #f2f2f2; }
            </style>
        </head>
        <body>
            <h1>Biomedical RAG Pipeline Comparison Dashboard</h1>
            
            <div class="highlight">
                <h2>Executive Summary</h2>
                <p><strong>Experiment ID:</strong> {experiment_id}</p>
                <p><strong>Evaluation Date:</strong> {timestamp}</p>
                <p><strong>Total Questions Evaluated:</strong> {total_questions}</p>
                <p><strong>Total Evaluation Time:</strong> {total_time:.2f} seconds</p>
            </div>
            
            <div class="metric-card">
                <h2>Overall Pipeline Rankings</h2>
                {ranking_table}
            </div>
            
            <div class="metric-card">
                <h2>Best Pipeline Per Metric</h2>
                {best_per_metric_table}
            </div>
            
            <div class="metric-card">
                <h2>Statistical Significance Summary</h2>
                {significance_summary}
            </div>
            
            <div class="metric-card">
                <h2>Detailed RAGAS Metrics</h2>
                {detailed_metrics_table}
            </div>
            
        </body>
        </html>
        """

        # Generate tables and content
        ranking_table = self._generate_ranking_table(results)
        best_per_metric_table = self._generate_best_per_metric_table(results)
        significance_summary = self._generate_significance_summary_html(results)
        detailed_metrics_table = self._generate_detailed_metrics_table(results)

        return html_template.format(
            experiment_id=results.experiment_id,
            timestamp=results.timestamp,
            total_questions=results.total_questions_evaluated,
            total_time=results.total_evaluation_time,
            ranking_table=ranking_table,
            best_per_metric_table=best_per_metric_table,
            significance_summary=significance_summary,
            detailed_metrics_table=detailed_metrics_table,
        )

    def _generate_ranking_table(self, results: ComparativeAnalysisResults) -> str:
        """Generate ranking table HTML."""

        # Calculate overall rankings based on average performance
        pipeline_scores = {}

        for pipeline_name, result in results.pipeline_results.items():
            if result.ragas_results and result.ragas_results.overall_score:
                pipeline_scores[pipeline_name] = result.ragas_results.overall_score
            else:
                pipeline_scores[pipeline_name] = 0.0

        # Sort by score
        sorted_pipelines = sorted(
            pipeline_scores.items(), key=lambda x: x[1], reverse=True
        )

        html = "<table><tr><th>Rank</th><th>Pipeline</th><th>Overall Score</th></tr>"

        for rank, (pipeline, score) in enumerate(sorted_pipelines, 1):
            html += f"<tr><td>{rank}</td><td>{pipeline}</td><td>{score:.4f}</td></tr>"

        html += "</table>"
        return html

    def _generate_best_per_metric_table(
        self, results: ComparativeAnalysisResults
    ) -> str:
        """Generate best pipeline per metric table."""

        html = "<table><tr><th>Metric</th><th>Best Pipeline</th><th>Score</th></tr>"

        for metric, pipeline in results.best_pipeline_per_metric.items():
            result = results.pipeline_results[pipeline]
            score = result.get_metric_value(metric)
            html += f"<tr><td>{metric}</td><td>{pipeline}</td><td>{score:.4f if score else 'N/A'}</td></tr>"

        html += "</table>"
        return html

    def _generate_significance_summary_html(
        self, results: ComparativeAnalysisResults
    ) -> str:
        """Generate statistical significance summary."""

        html = "<ul>"

        for metric, summary in results.statistical_significance_summary.items():
            html += f"<li><strong>{metric}:</strong> "
            html += f"{summary['significant_tests']}/{summary['total_tests']} comparisons significant "
            html += f"(largest effect size: {summary['largest_effect_size']:.3f})</li>"

        html += "</ul>"
        return html

    def _generate_detailed_metrics_table(
        self, results: ComparativeAnalysisResults
    ) -> str:
        """Generate detailed metrics table."""

        metrics = self.config.ragas_metrics
        pipelines = list(results.pipeline_results.keys())

        html = "<table><tr><th>Pipeline</th>"
        for metric in metrics:
            html += f"<th>{metric}</th>"
        html += "<th>Success Rate</th></tr>"

        for pipeline_name in pipelines:
            result = results.pipeline_results[pipeline_name]
            html += f"<tr><td>{pipeline_name}</td>"

            for metric in metrics:
                score = result.get_metric_value(metric)
                html += f"<td>{score:.4f if score else 'N/A'}</td>"

            html += f"<td>{result.success_rate:.2%}</td></tr>"

        html += "</table>"
        return html


def create_comparative_analysis_system(
    config: Optional[PipelineEvaluationConfig] = None,
) -> ComparativeAnalysisSystem:
    """Factory function to create a configured comparative analysis system."""
    return ComparativeAnalysisSystem(config)


if __name__ == "__main__":
    # Example usage
    config = PipelineEvaluationConfig(
        total_questions=100,  # Reduced for testing
        questions_per_pipeline_test=20,
        sample_documents=100,
    )

    system = create_comparative_analysis_system(config)

    # Mock documents for testing
    sample_documents = [
        {
            "doc_id": "pmc_001",
            "title": "Diabetes Management in Clinical Practice",
            "content": "Type 2 diabetes is characterized by insulin resistance and relative insulin deficiency. Management includes lifestyle modifications, oral hypoglycemic agents, and insulin therapy when indicated.",
            "source": "PMC_database",
        },
        {
            "doc_id": "pmc_002",
            "title": "Cardiovascular Risk in Diabetic Patients",
            "content": "Patients with diabetes have increased cardiovascular risk. Preventive measures include blood pressure control, lipid management, and antiplatelet therapy.",
            "source": "PMC_database",
        },
    ]

    # Run comprehensive comparison
    results = system.run_comprehensive_comparison(
        documents=sample_documents, pipeline_names=["BasicRAG", "CRAG"]  # Test subset
    )

    print(f"Comparison completed: {results.experiment_id}")
    print(f"Total evaluation time: {results.total_evaluation_time:.2f}s")
    print(f"Questions evaluated: {results.total_questions_evaluated}")
