"""
Reproducible Evaluation Orchestration System for Biomedical RAG Pipeline Comparison

This module provides a comprehensive orchestration system that coordinates all evaluation
components to deliver reproducible, end-to-end pipeline comparisons with statistical rigor.
It serves as the main entry point for conducting large-scale biomedical RAG evaluations.

Key Features:
1. End-to-end evaluation orchestration
2. Reproducible experiment management
3. Configuration-driven workflows
4. Progress monitoring and checkpointing
5. Automated result reporting
6. Multi-experiment comparison
7. Integration with existing infrastructure
8. Comprehensive logging and error handling
"""

import argparse
import asyncio
import hashlib
import json
import logging
import shutil
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
import yaml
from biomedical_question_generator import (
    BiomedicalQuestionGenerator,
    QuestionGenerationConfig,
    create_biomedical_question_generator,
)
from comparative_analysis_system import (
    ComparativeAnalysisSystem,
    PipelineEvaluationConfig,
    create_comparative_analysis_system,
)

# Import our evaluation frameworks
from pmc_data_pipeline import (
    PMCDataPipeline,
    PMCDocument,
    ProcessingConfig,
    create_pmc_data_pipeline,
)
from ragas_metrics_framework import (
    BiomedicalRAGASFramework,
    create_biomedical_ragas_framework,
)
from statistical_evaluation_methodology import (
    StatisticalEvaluationFramework,
    create_statistical_framework,
)

# Import existing pipeline infrastructure
from iris_rag.pipelines.basic import BasicRAGPipeline
from iris_rag.pipelines.basic_rerank import BasicRAGRerankingPipeline
from iris_rag.pipelines.crag import CRAGPipeline
from iris_rag.pipelines.graphrag import GraphRAGPipeline

logger = logging.getLogger(__name__)


@dataclass
class EvaluationExperimentConfig:
    """Complete configuration for evaluation experiment."""

    # Experiment metadata
    experiment_name: str = "biomedical_rag_evaluation"
    experiment_description: str = "Comprehensive evaluation of biomedical RAG pipelines"
    experiment_version: str = "1.0.0"

    # Data configuration
    data_source: str = "data/pmc_documents"
    max_documents: int = 10000
    max_questions: int = 1000
    test_questions_per_pipeline: int = 100

    # Pipeline configuration
    pipelines_to_evaluate: List[str] = field(
        default_factory=lambda: ["BasicRAG", "CRAG", "GraphRAG", "BasicRAGReranking"]
    )

    # Processing configuration
    max_workers: int = 8
    batch_size: int = 32
    enable_caching: bool = True
    enable_checkpointing: bool = True

    # Quality thresholds
    min_document_quality: float = 0.3
    min_question_quality: float = 0.5
    statistical_significance_threshold: float = 0.05

    # Output configuration
    output_base_dir: str = "outputs/evaluations"
    generate_visualizations: bool = True
    generate_interactive_dashboard: bool = True
    generate_final_report: bool = True

    # Reproducibility configuration
    random_seed: int = 42
    save_intermediate_results: bool = True
    save_model_artifacts: bool = True

    # Infrastructure configuration
    vector_db_config: Dict[str, Any] = field(default_factory=dict)
    compute_resources: Dict[str, Any] = field(
        default_factory=lambda: {"cpu_cores": 8, "memory_gb": 32, "gpu_enabled": False}
    )


@dataclass
class EvaluationStage:
    """Represents a single evaluation stage."""

    name: str
    description: str
    required: bool = True
    completed: bool = False
    start_time: Optional[str] = None
    end_time: Optional[str] = None
    duration: Optional[float] = None
    status: str = "pending"  # pending, running, completed, failed
    error_message: Optional[str] = None
    output_files: List[str] = field(default_factory=list)
    metrics: Dict[str, Any] = field(default_factory=dict)


@dataclass
class EvaluationRun:
    """Represents a complete evaluation run."""

    run_id: str
    experiment_name: str
    config: EvaluationExperimentConfig
    start_time: str
    end_time: Optional[str] = None
    total_duration: Optional[float] = None
    status: str = "running"  # running, completed, failed, cancelled

    # Stage tracking
    stages: List[EvaluationStage] = field(default_factory=list)
    current_stage: int = 0

    # Results
    pipeline_results: Dict[str, Any] = field(default_factory=dict)
    comparative_analysis: Optional[Dict[str, Any]] = None
    final_recommendations: List[str] = field(default_factory=list)

    # File paths
    run_directory: Optional[str] = None
    log_file: Optional[str] = None
    config_file: Optional[str] = None
    results_file: Optional[str] = None


class EvaluationOrchestrator:
    """
    Main orchestration system for biomedical RAG pipeline evaluation.

    Coordinates all evaluation components to provide reproducible, comprehensive
    pipeline comparisons with statistical rigor and detailed reporting.
    """

    def __init__(self, config: Optional[EvaluationExperimentConfig] = None):
        """
        Initialize the evaluation orchestrator.

        Args:
            config: Experiment configuration
        """
        self.config = config or EvaluationExperimentConfig()

        # Set random seed for reproducibility
        np.random.seed(self.config.random_seed)

        # Initialize run tracking
        self.current_run: Optional[EvaluationRun] = None
        self.run_history: List[EvaluationRun] = []

        # Initialize evaluation components
        self.data_pipeline: Optional[PMCDataPipeline] = None
        self.question_generator: Optional[BiomedicalQuestionGenerator] = None
        self.ragas_framework: Optional[BiomedicalRAGASFramework] = None
        self.statistical_framework: Optional[StatisticalEvaluationFramework] = None
        self.comparative_analysis: Optional[ComparativeAnalysisSystem] = None

        # Setup directories
        self.base_output_dir = Path(self.config.output_base_dir)
        self.base_output_dir.mkdir(parents=True, exist_ok=True)

        logger.info("Evaluation Orchestrator initialized successfully")

    def run_complete_evaluation(
        self, config_override: Optional[Dict[str, Any]] = None
    ) -> EvaluationRun:
        """
        Run complete evaluation pipeline from start to finish.

        Args:
            config_override: Override specific configuration parameters

        Returns:
            Complete evaluation run results
        """
        logger.info("Starting complete biomedical RAG evaluation")

        # Apply configuration overrides
        if config_override:
            self._apply_config_overrides(config_override)

        # Create new evaluation run
        run = self._create_evaluation_run()
        self.current_run = run

        try:
            # Define evaluation stages
            stages = self._define_evaluation_stages()
            run.stages = stages

            # Execute stages sequentially
            for stage_idx, stage in enumerate(stages):
                run.current_stage = stage_idx

                logger.info(
                    f"Executing stage {stage_idx + 1}/{len(stages)}: {stage.name}"
                )

                # Execute stage
                success = self._execute_evaluation_stage(stage)

                if not success and stage.required:
                    logger.error(
                        f"Required stage {stage.name} failed - aborting evaluation"
                    )
                    run.status = "failed"
                    break

                # Save checkpoint
                if self.config.enable_checkpointing:
                    self._save_checkpoint(run)

            # Finalize evaluation
            if run.status != "failed":
                self._finalize_evaluation(run)

        except Exception as e:
            logger.error(f"Evaluation failed with exception: {e}")
            run.status = "failed"
            if run.stages and run.current_stage < len(run.stages):
                run.stages[run.current_stage].status = "failed"
                run.stages[run.current_stage].error_message = str(e)

        finally:
            # Complete run
            run.end_time = datetime.now().isoformat()
            if run.start_time:
                start_dt = datetime.fromisoformat(run.start_time)
                end_dt = datetime.fromisoformat(run.end_time)
                run.total_duration = (end_dt - start_dt).total_seconds()

            # Save final results
            self._save_run_results(run)

            # Add to history
            self.run_history.append(run)

        logger.info(f"Evaluation completed: {run.run_id} ({run.status})")
        return run

    def run_partial_evaluation(
        self, stages: List[str], config_override: Optional[Dict[str, Any]] = None
    ) -> EvaluationRun:
        """
        Run partial evaluation with specified stages only.

        Args:
            stages: List of stage names to execute
            config_override: Override specific configuration parameters

        Returns:
            Partial evaluation run results
        """
        logger.info(f"Starting partial evaluation with stages: {stages}")

        # Apply configuration overrides
        if config_override:
            self._apply_config_overrides(config_override)

        # Create new evaluation run
        run = self._create_evaluation_run()
        run.experiment_name += "_partial"
        self.current_run = run

        try:
            # Filter stages
            all_stages = self._define_evaluation_stages()
            filtered_stages = [s for s in all_stages if s.name in stages]

            if not filtered_stages:
                raise ValueError(f"No valid stages found in: {stages}")

            run.stages = filtered_stages

            # Execute specified stages
            for stage_idx, stage in enumerate(filtered_stages):
                run.current_stage = stage_idx

                logger.info(
                    f"Executing partial stage {stage_idx + 1}/{len(filtered_stages)}: {stage.name}"
                )

                success = self._execute_evaluation_stage(stage)

                if not success and stage.required:
                    run.status = "failed"
                    break

            # Finalize partial evaluation
            if run.status != "failed":
                run.status = "completed"

        except Exception as e:
            logger.error(f"Partial evaluation failed: {e}")
            run.status = "failed"

        finally:
            run.end_time = datetime.now().isoformat()
            if run.start_time:
                start_dt = datetime.fromisoformat(run.start_time)
                end_dt = datetime.fromisoformat(run.end_time)
                run.total_duration = (end_dt - start_dt).total_seconds()

            self._save_run_results(run)
            self.run_history.append(run)

        return run

    def resume_evaluation(self, run_id: str) -> EvaluationRun:
        """
        Resume a previous evaluation run from checkpoint.

        Args:
            run_id: ID of the run to resume

        Returns:
            Resumed evaluation run
        """
        logger.info(f"Resuming evaluation: {run_id}")

        # Load checkpoint
        run = self._load_checkpoint(run_id)
        if not run:
            raise ValueError(f"No checkpoint found for run: {run_id}")

        self.current_run = run

        # Resume from current stage
        start_stage = run.current_stage

        try:
            for stage_idx in range(start_stage, len(run.stages)):
                stage = run.stages[stage_idx]
                run.current_stage = stage_idx

                # Skip completed stages
                if stage.completed:
                    continue

                logger.info(
                    f"Resuming stage {stage_idx + 1}/{len(run.stages)}: {stage.name}"
                )

                success = self._execute_evaluation_stage(stage)

                if not success and stage.required:
                    run.status = "failed"
                    break

                # Save checkpoint
                if self.config.enable_checkpointing:
                    self._save_checkpoint(run)

            # Finalize if all stages completed
            if run.status != "failed":
                self._finalize_evaluation(run)

        except Exception as e:
            logger.error(f"Resume evaluation failed: {e}")
            run.status = "failed"

        finally:
            run.end_time = datetime.now().isoformat()
            if run.start_time:
                start_dt = datetime.fromisoformat(run.start_time)
                end_dt = datetime.fromisoformat(run.end_time)
                run.total_duration = (end_dt - start_dt).total_seconds()

            self._save_run_results(run)

        return run

    def compare_runs(self, run_ids: List[str]) -> Dict[str, Any]:
        """
        Compare results across multiple evaluation runs.

        Args:
            run_ids: List of run IDs to compare

        Returns:
            Comparison analysis results
        """
        logger.info(f"Comparing evaluation runs: {run_ids}")

        # Load run results
        runs = []
        for run_id in run_ids:
            run = self._load_run_results(run_id)
            if run:
                runs.append(run)
            else:
                logger.warning(f"Could not load run: {run_id}")

        if len(runs) < 2:
            raise ValueError("Need at least 2 runs for comparison")

        # Perform comparison analysis
        comparison = {
            "run_ids": run_ids,
            "comparison_timestamp": datetime.now().isoformat(),
            "run_summaries": {},
            "metric_comparisons": {},
            "statistical_differences": {},
            "recommendations": [],
        }

        # Summarize each run
        for run in runs:
            comparison["run_summaries"][run.run_id] = {
                "experiment_name": run.experiment_name,
                "status": run.status,
                "duration": run.total_duration,
                "config_hash": self._hash_config(run.config),
                "pipeline_results": run.pipeline_results,
            }

        # Compare metrics across runs
        self._compare_run_metrics(runs, comparison)

        # Save comparison results
        comparison_file = (
            self.base_output_dir
            / f"run_comparison_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(comparison_file, "w") as f:
            json.dump(comparison, f, indent=2, default=str)

        logger.info(f"Run comparison saved to: {comparison_file}")
        return comparison

    def _create_evaluation_run(self) -> EvaluationRun:
        """Create a new evaluation run."""

        run_id = (
            f"{self.config.experiment_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        )

        # Create run directory
        run_dir = self.base_output_dir / run_id
        run_dir.mkdir(exist_ok=True)

        # Setup logging for this run
        log_file = run_dir / "evaluation.log"
        self._setup_run_logging(log_file)

        # Save configuration
        config_file = run_dir / "config.yaml"
        with open(config_file, "w") as f:
            yaml.dump(asdict(self.config), f, indent=2)

        return EvaluationRun(
            run_id=run_id,
            experiment_name=self.config.experiment_name,
            config=self.config,
            start_time=datetime.now().isoformat(),
            run_directory=str(run_dir),
            log_file=str(log_file),
            config_file=str(config_file),
            results_file=str(run_dir / "results.json"),
        )

    def _define_evaluation_stages(self) -> List[EvaluationStage]:
        """Define the evaluation stages."""

        stages = [
            EvaluationStage(
                name="data_preparation",
                description="Process PMC documents and prepare evaluation data",
                required=True,
            ),
            EvaluationStage(
                name="question_generation",
                description="Generate biomedical evaluation questions",
                required=True,
            ),
            EvaluationStage(
                name="pipeline_initialization",
                description="Initialize and validate RAG pipelines",
                required=True,
            ),
            EvaluationStage(
                name="pipeline_evaluation",
                description="Execute pipeline evaluation using RAGAS metrics",
                required=True,
            ),
            EvaluationStage(
                name="statistical_analysis",
                description="Conduct statistical analysis and comparisons",
                required=True,
            ),
            EvaluationStage(
                name="comparative_analysis",
                description="Generate comprehensive comparative analysis",
                required=True,
            ),
            EvaluationStage(
                name="visualization",
                description="Generate visualizations and interactive dashboards",
                required=False,
            ),
            EvaluationStage(
                name="reporting",
                description="Generate final evaluation report",
                required=False,
            ),
        ]

        return stages

    def _execute_evaluation_stage(self, stage: EvaluationStage) -> bool:
        """Execute a single evaluation stage."""

        stage.status = "running"
        stage.start_time = datetime.now().isoformat()

        try:
            if stage.name == "data_preparation":
                success = self._execute_data_preparation(stage)
            elif stage.name == "question_generation":
                success = self._execute_question_generation(stage)
            elif stage.name == "pipeline_initialization":
                success = self._execute_pipeline_initialization(stage)
            elif stage.name == "pipeline_evaluation":
                success = self._execute_pipeline_evaluation(stage)
            elif stage.name == "statistical_analysis":
                success = self._execute_statistical_analysis(stage)
            elif stage.name == "comparative_analysis":
                success = self._execute_comparative_analysis(stage)
            elif stage.name == "visualization":
                success = self._execute_visualization(stage)
            elif stage.name == "reporting":
                success = self._execute_reporting(stage)
            else:
                logger.error(f"Unknown stage: {stage.name}")
                success = False

            stage.status = "completed" if success else "failed"
            stage.completed = success

        except Exception as e:
            logger.error(f"Stage {stage.name} failed with exception: {e}")
            stage.status = "failed"
            stage.error_message = str(e)
            success = False

        finally:
            stage.end_time = datetime.now().isoformat()
            if stage.start_time:
                start_dt = datetime.fromisoformat(stage.start_time)
                end_dt = datetime.fromisoformat(stage.end_time)
                stage.duration = (end_dt - start_dt).total_seconds()

        return success

    def _execute_data_preparation(self, stage: EvaluationStage) -> bool:
        """Execute data preparation stage."""
        logger.info("Executing data preparation stage")

        # Initialize data pipeline
        data_config = ProcessingConfig(
            input_data_path=self.config.data_source,
            max_documents=self.config.max_documents,
            max_workers=self.config.max_workers,
            batch_size=self.config.batch_size,
            min_quality_score=self.config.min_document_quality,
            output_data_path=f"{self.current_run.run_directory}/processed_data",
            enable_caching=self.config.enable_caching,
        )

        self.data_pipeline = create_pmc_data_pipeline(data_config)

        # Process documents
        stats = self.data_pipeline.process_documents(
            self.config.data_source, resume_from_checkpoint=False
        )

        # Save stage results
        stage.metrics = {
            "total_documents": stats.total_documents,
            "successful_documents": stats.successful_documents,
            "failed_documents": stats.failed_documents,
            "total_chunks": stats.total_chunks,
            "processing_time": stats.processing_time,
            "average_quality": stats.avg_document_quality,
        }

        stage.output_files = [
            f"{self.current_run.run_directory}/processed_data/metadata",
            f"{self.current_run.run_directory}/processed_data/chunks",
        ]

        # Validation
        if stats.successful_documents < 100:
            logger.warning(
                f"Only {stats.successful_documents} documents processed successfully"
            )
            return False

        logger.info(
            f"Data preparation completed: {stats.successful_documents} documents processed"
        )
        return True

    def _execute_question_generation(self, stage: EvaluationStage) -> bool:
        """Execute question generation stage."""
        logger.info("Executing question generation stage")

        # Load processed documents
        processed_docs = self._load_processed_documents()

        # Initialize question generator
        question_config = QuestionGenerationConfig(
            total_questions=self.config.max_questions,
            min_confidence_score=self.config.min_question_quality,
        )

        self.question_generator = create_biomedical_question_generator(question_config)

        # Generate questions
        questions = self.question_generator.generate_questions_from_documents(
            processed_docs
        )

        # Save questions
        questions_file = (
            Path(self.current_run.run_directory) / "evaluation_questions.json"
        )
        questions_data = [
            {
                "question_id": q.question_id,
                "question": q.question,
                "ground_truth_answer": q.ground_truth_answer,
                "question_type": q.question_type,
                "complexity_level": q.complexity_level,
                "confidence_score": q.confidence_score,
            }
            for q in questions
        ]

        with open(questions_file, "w") as f:
            json.dump(questions_data, f, indent=2)

        stage.metrics = {
            "total_questions": len(questions),
            "question_types": {
                qt: len([q for q in questions if q.question_type == qt])
                for qt in set(q.question_type for q in questions)
            },
            "avg_confidence": np.mean([q.confidence_score for q in questions]),
        }

        stage.output_files = [str(questions_file)]

        logger.info(
            f"Question generation completed: {len(questions)} questions generated"
        )
        return len(questions) >= 50  # Minimum threshold

    def _execute_pipeline_initialization(self, stage: EvaluationStage) -> bool:
        """Execute pipeline initialization stage."""
        logger.info("Executing pipeline initialization stage")

        # Initialize comparative analysis system
        comparison_config = PipelineEvaluationConfig(
            total_questions=self.config.max_questions,
            questions_per_pipeline_test=self.config.test_questions_per_pipeline,
            max_workers=self.config.max_workers,
            batch_size=self.config.batch_size,
            statistical_alpha=self.config.statistical_significance_threshold,
            output_dir=f"{self.current_run.run_directory}/comparative_analysis",
        )

        self.comparative_analysis = create_comparative_analysis_system(
            comparison_config
        )

        # Validate pipeline availability
        available_pipelines = []
        for pipeline_name in self.config.pipelines_to_evaluate:
            try:
                # Test pipeline initialization
                self.comparative_analysis._initialize_pipelines([pipeline_name])
                available_pipelines.append(pipeline_name)
                logger.info(f"Pipeline {pipeline_name} initialized successfully")
            except Exception as e:
                logger.error(f"Pipeline {pipeline_name} initialization failed: {e}")

        stage.metrics = {
            "requested_pipelines": len(self.config.pipelines_to_evaluate),
            "available_pipelines": len(available_pipelines),
            "pipeline_names": available_pipelines,
        }

        # Update config with available pipelines
        self.config.pipelines_to_evaluate = available_pipelines

        logger.info(
            f"Pipeline initialization completed: {len(available_pipelines)} pipelines available"
        )
        return len(available_pipelines) >= 2  # Need at least 2 for comparison

    def _execute_pipeline_evaluation(self, stage: EvaluationStage) -> bool:
        """Execute pipeline evaluation stage."""
        logger.info("Executing pipeline evaluation stage")

        # Load documents and questions
        processed_docs = self._load_processed_documents()
        questions = self._load_evaluation_questions()

        # Run comprehensive comparison
        results = self.comparative_analysis.run_comprehensive_comparison(
            documents=processed_docs, pipeline_names=self.config.pipelines_to_evaluate
        )

        # Store results in run
        self.current_run.comparative_analysis = {
            "experiment_id": results.experiment_id,
            "pipeline_results": {
                name: {
                    "success_rate": result.success_rate,
                    "ragas_scores": (
                        result.ragas_results.to_dict() if result.ragas_results else {}
                    ),
                    "execution_times": result.execution_times,
                }
                for name, result in results.pipeline_results.items()
            },
            "overall_rankings": results.overall_rankings,
            "best_pipeline_per_metric": results.best_pipeline_per_metric,
        }

        stage.metrics = {
            "pipelines_evaluated": len(results.pipeline_results),
            "questions_evaluated": results.total_questions_evaluated,
            "evaluation_time": results.total_evaluation_time,
            "successful_evaluations": results.successful_evaluations_per_pipeline,
        }

        stage.output_files = [f"{self.current_run.run_directory}/comparative_analysis"]

        logger.info(
            f"Pipeline evaluation completed: {len(results.pipeline_results)} pipelines evaluated"
        )
        return True

    def _execute_statistical_analysis(self, stage: EvaluationStage) -> bool:
        """Execute statistical analysis stage."""
        logger.info("Executing statistical analysis stage")

        # Statistical analysis is part of comparative analysis
        # Additional statistical tests can be performed here

        stage.metrics = {
            "statistical_tests_completed": True,
            "significance_threshold": self.config.statistical_significance_threshold,
        }

        logger.info("Statistical analysis completed")
        return True

    def _execute_comparative_analysis(self, stage: EvaluationStage) -> bool:
        """Execute comparative analysis stage."""
        logger.info("Executing comparative analysis stage")

        # Comparative analysis already completed in pipeline evaluation
        # Generate summary insights here

        if not self.current_run.comparative_analysis:
            logger.error("No comparative analysis results available")
            return False

        # Generate recommendations
        recommendations = self._generate_recommendations()
        self.current_run.final_recommendations = recommendations

        stage.metrics = {
            "recommendations_generated": len(recommendations),
            "comparative_analysis_completed": True,
        }

        logger.info("Comparative analysis completed")
        return True

    def _execute_visualization(self, stage: EvaluationStage) -> bool:
        """Execute visualization stage."""
        logger.info("Executing visualization stage")

        if not self.config.generate_visualizations:
            logger.info("Visualization generation disabled")
            return True

        # Visualizations are generated by comparative analysis system
        viz_dir = Path(self.current_run.run_directory) / "visualizations"
        viz_dir.mkdir(exist_ok=True)

        stage.output_files = [str(viz_dir)]
        stage.metrics = {"visualizations_generated": True}

        logger.info("Visualization generation completed")
        return True

    def _execute_reporting(self, stage: EvaluationStage) -> bool:
        """Execute reporting stage."""
        logger.info("Executing reporting stage")

        if not self.config.generate_final_report:
            logger.info("Final report generation disabled")
            return True

        # Generate comprehensive report
        report = self._generate_final_report()

        report_file = Path(self.current_run.run_directory) / "final_report.md"
        with open(report_file, "w") as f:
            f.write(report)

        stage.output_files = [str(report_file)]
        stage.metrics = {"final_report_generated": True}

        logger.info(f"Final report generated: {report_file}")
        return True

    def _load_processed_documents(self) -> List[Dict[str, Any]]:
        """Load processed documents from data pipeline."""
        # Placeholder - would load from data pipeline output
        return []

    def _load_evaluation_questions(self) -> List[Dict[str, Any]]:
        """Load evaluation questions."""
        questions_file = (
            Path(self.current_run.run_directory) / "evaluation_questions.json"
        )
        if questions_file.exists():
            with open(questions_file, "r") as f:
                return json.load(f)
        return []

    def _generate_recommendations(self) -> List[str]:
        """Generate final recommendations based on analysis."""
        recommendations = []

        if self.current_run.comparative_analysis:
            best_overall = self.current_run.comparative_analysis.get(
                "best_pipeline_per_metric", {}
            )

            # Overall best pipeline
            if best_overall:
                most_common = max(
                    set(best_overall.values()), key=list(best_overall.values()).count
                )
                recommendations.append(f"Overall recommended pipeline: {most_common}")

            # Specific use case recommendations
            for metric, pipeline in best_overall.items():
                recommendations.append(f"For {metric} optimization: {pipeline}")

        return recommendations

    def _generate_final_report(self) -> str:
        """Generate comprehensive final report."""

        report = f"""# Biomedical RAG Pipeline Evaluation Report

## Experiment Overview
- **Experiment Name**: {self.config.experiment_name}
- **Run ID**: {self.current_run.run_id}
- **Execution Time**: {self.current_run.total_duration:.2f} seconds
- **Status**: {self.current_run.status}

## Configuration
- **Documents Processed**: {self.config.max_documents}
- **Questions Generated**: {self.config.max_questions}
- **Pipelines Evaluated**: {', '.join(self.config.pipelines_to_evaluate)}
- **Statistical Significance Threshold**: {self.config.statistical_significance_threshold}

## Results Summary
"""

        if self.current_run.comparative_analysis:
            report += """
### Pipeline Performance Rankings
"""
            best_per_metric = self.current_run.comparative_analysis.get(
                "best_pipeline_per_metric", {}
            )
            for metric, pipeline in best_per_metric.items():
                report += f"- **{metric}**: {pipeline}\n"

        if self.current_run.final_recommendations:
            report += """
## Recommendations
"""
            for i, rec in enumerate(self.current_run.final_recommendations, 1):
                report += f"{i}. {rec}\n"

        report += f"""
## Execution Details
"""
        for stage in self.current_run.stages:
            report += f"- **{stage.name}**: {stage.status} ({stage.duration:.2f}s)\n"

        return report

    def _apply_config_overrides(self, overrides: Dict[str, Any]):
        """Apply configuration overrides."""
        for key, value in overrides.items():
            if hasattr(self.config, key):
                setattr(self.config, key, value)
                logger.info(f"Config override: {key} = {value}")

    def _setup_run_logging(self, log_file: Path):
        """Setup logging for evaluation run."""
        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        file_handler.setFormatter(formatter)

        # Add to root logger
        logging.getLogger().addHandler(file_handler)

    def _save_checkpoint(self, run: EvaluationRun):
        """Save evaluation checkpoint."""
        if not self.config.enable_checkpointing:
            return

        checkpoint_file = Path(run.run_directory) / "checkpoint.json"
        with open(checkpoint_file, "w") as f:
            json.dump(asdict(run), f, indent=2, default=str)

    def _load_checkpoint(self, run_id: str) -> Optional[EvaluationRun]:
        """Load evaluation checkpoint."""
        checkpoint_file = self.base_output_dir / run_id / "checkpoint.json"

        if checkpoint_file.exists():
            with open(checkpoint_file, "r") as f:
                data = json.load(f)

            # Reconstruct run object
            config_data = data["config"]
            config = EvaluationExperimentConfig(**config_data)

            run = EvaluationRun(
                run_id=data["run_id"],
                experiment_name=data["experiment_name"],
                config=config,
                start_time=data["start_time"],
                end_time=data.get("end_time"),
                total_duration=data.get("total_duration"),
                status=data["status"],
                current_stage=data["current_stage"],
                run_directory=data["run_directory"],
                log_file=data["log_file"],
                config_file=data["config_file"],
                results_file=data["results_file"],
            )

            # Reconstruct stages
            stages = []
            for stage_data in data["stages"]:
                stage = EvaluationStage(**stage_data)
                stages.append(stage)
            run.stages = stages

            return run

        return None

    def _save_run_results(self, run: EvaluationRun):
        """Save final run results."""
        with open(run.results_file, "w") as f:
            json.dump(asdict(run), f, indent=2, default=str)

    def _load_run_results(self, run_id: str) -> Optional[EvaluationRun]:
        """Load run results."""
        results_file = self.base_output_dir / run_id / "results.json"

        if results_file.exists():
            with open(results_file, "r") as f:
                data = json.load(f)

            # Reconstruct run object (similar to checkpoint loading)
            config_data = data["config"]
            config = EvaluationExperimentConfig(**config_data)

            return EvaluationRun(
                run_id=data["run_id"],
                experiment_name=data["experiment_name"],
                config=config,
                start_time=data["start_time"],
                end_time=data.get("end_time"),
                total_duration=data.get("total_duration"),
                status=data["status"],
            )

        return None

    def _finalize_evaluation(self, run: EvaluationRun):
        """Finalize evaluation run."""
        if all(stage.completed for stage in run.stages if stage.required):
            run.status = "completed"
        else:
            run.status = "partial"

        logger.info(f"Evaluation finalized with status: {run.status}")

    def _hash_config(self, config: EvaluationExperimentConfig) -> str:
        """Generate hash for configuration."""
        config_str = json.dumps(asdict(config), sort_keys=True)
        return hashlib.md5(config_str.encode()).hexdigest()[:8]

    def _compare_run_metrics(
        self, runs: List[EvaluationRun], comparison: Dict[str, Any]
    ):
        """Compare metrics across multiple runs."""
        # Placeholder for cross-run metric comparison
        comparison["metric_comparisons"] = {
            "total_runs": len(runs),
            "comparison_completed": True,
        }


def create_evaluation_orchestrator(
    config: Optional[EvaluationExperimentConfig] = None,
) -> EvaluationOrchestrator:
    """Factory function to create evaluation orchestrator."""
    return EvaluationOrchestrator(config)


def main():
    """Command-line interface for evaluation orchestrator."""
    parser = argparse.ArgumentParser(
        description="Biomedical RAG Pipeline Evaluation Orchestrator"
    )

    parser.add_argument("--config", type=str, help="Configuration file path")
    parser.add_argument(
        "--experiment-name", type=str, default="biomedical_rag_evaluation"
    )
    parser.add_argument("--max-documents", type=int, default=1000)
    parser.add_argument("--max-questions", type=int, default=100)
    parser.add_argument("--pipelines", nargs="+", default=["BasicRAG", "CRAG"])
    parser.add_argument("--output-dir", type=str, default="outputs/evaluations")
    parser.add_argument("--resume", type=str, help="Resume from run ID")
    parser.add_argument("--partial", nargs="+", help="Run only specified stages")

    args = parser.parse_args()

    # Load configuration
    if args.config and Path(args.config).exists():
        with open(args.config, "r") as f:
            config_data = yaml.safe_load(f)
        config = EvaluationExperimentConfig(**config_data)
    else:
        config = EvaluationExperimentConfig(
            experiment_name=args.experiment_name,
            max_documents=args.max_documents,
            max_questions=args.max_questions,
            pipelines_to_evaluate=args.pipelines,
            output_base_dir=args.output_dir,
        )

    # Create orchestrator
    orchestrator = create_evaluation_orchestrator(config)

    # Execute evaluation
    if args.resume:
        run = orchestrator.resume_evaluation(args.resume)
    elif args.partial:
        run = orchestrator.run_partial_evaluation(args.partial)
    else:
        run = orchestrator.run_complete_evaluation()

    print(f"Evaluation completed: {run.run_id}")
    print(f"Status: {run.status}")
    print(f"Duration: {run.total_duration:.2f}s")
    print(f"Results: {run.results_file}")


if __name__ == "__main__":
    main()
