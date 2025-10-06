"""
Comprehensive RAGAS Metrics Framework for Biomedical RAG Evaluation

This module implements all key RAGAS metrics with biomedical domain adaptations,
statistical rigor, and robust error handling for systematic pipeline comparison.

Key Features:
1. All 7 core RAGAS metrics implementation
2. Biomedical domain-specific adaptations
3. Statistical significance testing
4. Confidence interval calculation
5. Batch processing for scalability
6. Comprehensive error handling and fallbacks
"""

import json
import logging
import time
import warnings
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

# Statistical analysis
from scipy import stats
from scipy.stats import bootstrap

warnings.filterwarnings("ignore")

import torch

# NLP and ML libraries
from sentence_transformers import SentenceTransformer, util
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline

# SpaCy import with fallback
try:
    import spacy

    SPACY_AVAILABLE = True
except (ImportError, ValueError) as e:
    SPACY_AVAILABLE = False
    warnings.warn(f"SpaCy not available: {e}. Some NLP features will be disabled.")

# RAGAS specific imports
try:
    from ragas import evaluate
    from ragas.metrics import (
        answer_correctness,
        answer_relevancy,
        answer_similarity,
        context_precision,
        context_recall,
        context_utilization,
        faithfulness,
    )

    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False
    logging.warning("RAGAS library not available, using custom implementations")

logger = logging.getLogger(__name__)


@dataclass
class RAGASResult:
    """Structured RAGAS evaluation result with statistical measures."""

    metric_name: str
    value: float
    confidence_interval: Tuple[float, float]
    sample_size: int
    standard_error: float
    computation_time: float
    error_message: Optional[str] = None
    individual_scores: List[float] = field(default_factory=list)


@dataclass
class ComprehensiveRAGASResults:
    """Complete RAGAS evaluation results for a pipeline."""

    pipeline_name: str
    timestamp: str
    total_queries: int
    successful_evaluations: int

    # Core RAGAS metrics
    faithfulness: Optional[RAGASResult] = None
    answer_relevancy: Optional[RAGASResult] = None
    context_precision: Optional[RAGASResult] = None
    context_recall: Optional[RAGASResult] = None
    context_utilization: Optional[RAGASResult] = None
    answer_correctness: Optional[RAGASResult] = None
    answer_similarity: Optional[RAGASResult] = None

    # Aggregate scores
    overall_score: Optional[float] = None
    weighted_score: Optional[float] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        result = {
            "pipeline_name": self.pipeline_name,
            "timestamp": self.timestamp,
            "total_queries": self.total_queries,
            "successful_evaluations": self.successful_evaluations,
            "overall_score": self.overall_score,
            "weighted_score": self.weighted_score,
        }

        # Add metric results
        for metric_name in [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "context_utilization",
            "answer_correctness",
            "answer_similarity",
        ]:
            metric_result = getattr(self, metric_name)
            if metric_result:
                result[metric_name] = {
                    "value": metric_result.value,
                    "confidence_interval": metric_result.confidence_interval,
                    "sample_size": metric_result.sample_size,
                    "standard_error": metric_result.standard_error,
                    "computation_time": metric_result.computation_time,
                    "error_message": metric_result.error_message,
                }

        return result


class BiomedicalRAGASFramework:
    """
    Comprehensive RAGAS evaluation framework adapted for biomedical domain
    with statistical rigor and scalability for enterprise evaluation.
    """

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the RAGAS framework with biomedical adaptations.

        Args:
            config: Configuration dictionary for customizing evaluation
        """
        self.config = config or self._get_default_config()
        self.nlp = None
        self.embedding_model = None
        self.biomedical_qa_model = None
        self.entailment_model = None

        self._initialize_models()

        # Metric weights for biomedical domain
        self.metric_weights = {
            "faithfulness": 0.25,  # Critical for medical accuracy
            "answer_relevancy": 0.20,  # Important for user satisfaction
            "context_precision": 0.15,  # Efficiency metric
            "context_recall": 0.15,  # Completeness metric
            "answer_correctness": 0.20,  # Critical for medical accuracy
            "answer_similarity": 0.05,  # Less critical than correctness
        }

        logger.info("Biomedical RAGAS Framework initialized successfully")

    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration for biomedical RAGAS evaluation."""
        return {
            "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
            "biomedical_qa_model": "dmis-lab/biobert-base-cased-v1.1-squad",
            "entailment_model": "microsoft/deberta-large-mnli",
            "batch_size": 32,
            "max_context_length": 2048,
            "confidence_level": 0.95,
            "bootstrap_samples": 1000,
            "similarity_threshold": 0.7,
            "device": "cuda" if torch.cuda.is_available() else "cpu",
        }

    def _initialize_models(self):
        """Initialize all required models for RAGAS evaluation."""
        logger.info("Initializing biomedical NLP models for RAGAS evaluation...")

        try:
            # Embedding model for semantic similarity
            self.embedding_model = SentenceTransformer(
                self.config["embedding_model"], device=self.config["device"]
            )
            logger.info(f"Loaded embedding model: {self.config['embedding_model']}")

            # Biomedical QA model for answer validation
            self.biomedical_qa_model = pipeline(
                "question-answering",
                model=self.config["biomedical_qa_model"],
                device=0 if self.config["device"] == "cuda" else -1,
            )
            logger.info(
                f"Loaded biomedical QA model: {self.config['biomedical_qa_model']}"
            )

            # Entailment model for faithfulness assessment
            self.entailment_model = pipeline(
                "text-classification",
                model=self.config["entailment_model"],
                device=0 if self.config["device"] == "cuda" else -1,
            )
            logger.info(f"Loaded entailment model: {self.config['entailment_model']}")

            # SpaCy for text processing
            try:
                self.nlp = spacy.load("en_core_sci_sm")  # Biomedical spaCy model
            except OSError:
                self.nlp = spacy.load("en_core_web_sm")
                logger.warning("Using standard spaCy model instead of biomedical")

        except Exception as e:
            logger.error(f"Failed to initialize models: {e}")
            raise

    def evaluate_pipeline(
        self,
        pipeline_results: List[Dict[str, Any]],
        ground_truth_answers: List[str],
        pipeline_name: str,
    ) -> ComprehensiveRAGASResults:
        """
        Comprehensive RAGAS evaluation of a RAG pipeline.

        Args:
            pipeline_results: List of pipeline query results with 'query', 'answer', 'contexts'
            ground_truth_answers: List of ground truth answers
            pipeline_name: Name of the pipeline being evaluated

        Returns:
            Comprehensive RAGAS evaluation results
        """
        logger.info(f"Starting comprehensive RAGAS evaluation for {pipeline_name}")
        start_time = time.time()

        # Validate input data
        self._validate_evaluation_inputs(pipeline_results, ground_truth_answers)

        # Initialize results
        results = ComprehensiveRAGASResults(
            pipeline_name=pipeline_name,
            timestamp=time.strftime("%Y-%m-%d %H:%M:%S"),
            total_queries=len(pipeline_results),
            successful_evaluations=0,
        )

        # Prepare evaluation data
        evaluation_data = self._prepare_evaluation_data(
            pipeline_results, ground_truth_answers
        )

        # Evaluate each metric
        metrics_to_evaluate = [
            ("faithfulness", self._evaluate_faithfulness),
            ("answer_relevancy", self._evaluate_answer_relevancy),
            ("context_precision", self._evaluate_context_precision),
            ("context_recall", self._evaluate_context_recall),
            ("context_utilization", self._evaluate_context_utilization),
            ("answer_correctness", self._evaluate_answer_correctness),
            ("answer_similarity", self._evaluate_answer_similarity),
        ]

        successful_metrics = 0
        for metric_name, metric_func in metrics_to_evaluate:
            try:
                logger.info(f"Evaluating {metric_name}...")
                metric_result = metric_func(evaluation_data)
                setattr(results, metric_name, metric_result)
                if metric_result.error_message is None:
                    successful_metrics += 1
                logger.info(
                    f"{metric_name}: {metric_result.value:.4f} ± {metric_result.standard_error:.4f}"
                )
            except Exception as e:
                logger.error(f"Failed to evaluate {metric_name}: {e}")
                error_result = RAGASResult(
                    metric_name=metric_name,
                    value=0.0,
                    confidence_interval=(0.0, 0.0),
                    sample_size=0,
                    standard_error=0.0,
                    computation_time=0.0,
                    error_message=str(e),
                )
                setattr(results, metric_name, error_result)

        # Calculate aggregate scores
        results.overall_score = self._calculate_overall_score(results)
        results.weighted_score = self._calculate_weighted_score(results)
        results.successful_evaluations = successful_metrics

        total_time = time.time() - start_time
        logger.info(
            f"RAGAS evaluation completed in {total_time:.2f}s. "
            f"Overall score: {results.overall_score:.4f}"
        )

        return results

    def _validate_evaluation_inputs(
        self, pipeline_results: List[Dict[str, Any]], ground_truth_answers: List[str]
    ):
        """Validate inputs for RAGAS evaluation."""
        if len(pipeline_results) != len(ground_truth_answers):
            raise ValueError(
                "Mismatch between pipeline results and ground truth answers"
            )

        required_keys = ["query", "answer", "contexts"]
        for i, result in enumerate(pipeline_results):
            for key in required_keys:
                if key not in result:
                    raise ValueError(f"Missing key '{key}' in pipeline result {i}")
                if not result[key]:
                    logger.warning(f"Empty {key} in pipeline result {i}")

    def _prepare_evaluation_data(
        self, pipeline_results: List[Dict[str, Any]], ground_truth_answers: List[str]
    ) -> List[Dict[str, Any]]:
        """Prepare data for RAGAS evaluation."""
        evaluation_data = []

        for i, (result, ground_truth) in enumerate(
            zip(pipeline_results, ground_truth_answers)
        ):
            # Ensure contexts is a list of strings
            contexts = result["contexts"]
            if isinstance(contexts, str):
                contexts = [contexts]
            elif not isinstance(contexts, list):
                contexts = [str(contexts)]

            # Clean and validate data
            evaluation_item = {
                "question": str(result["query"]).strip(),
                "answer": str(result["answer"]).strip() if result["answer"] else "",
                "contexts": [str(ctx).strip() for ctx in contexts if str(ctx).strip()],
                "ground_truth": str(ground_truth).strip(),
                "index": i,
            }

            # Skip if critical data is missing
            if (
                evaluation_item["question"]
                and evaluation_item["answer"]
                and evaluation_item["contexts"]
                and evaluation_item["ground_truth"]
            ):
                evaluation_data.append(evaluation_item)
            else:
                logger.warning(f"Skipping evaluation item {i} due to missing data")

        logger.info(f"Prepared {len(evaluation_data)} valid evaluation items")
        return evaluation_data

    def _evaluate_faithfulness(
        self, evaluation_data: List[Dict[str, Any]]
    ) -> RAGASResult:
        """
        Evaluate faithfulness: How factually accurate is the answer based on the context.

        Uses entailment models to check if the answer is supported by the retrieved context.
        """
        start_time = time.time()
        scores = []

        for item in evaluation_data:
            try:
                answer = item["answer"]
                contexts = item["contexts"]

                # Combine contexts
                combined_context = " ".join(contexts)

                # Check entailment between answer and context
                entailment_result = self.entailment_model(
                    f"Premise: {combined_context} Hypothesis: {answer}"
                )

                # Extract faithfulness score
                if entailment_result[0]["label"] == "ENTAILMENT":
                    score = entailment_result[0]["score"]
                elif entailment_result[0]["label"] == "NEUTRAL":
                    score = 0.5
                else:  # CONTRADICTION
                    score = 1.0 - entailment_result[0]["score"]

                scores.append(score)

            except Exception as e:
                logger.warning(f"Failed to evaluate faithfulness for item: {e}")
                scores.append(0.0)

        return self._compute_metric_statistics(
            "faithfulness", scores, time.time() - start_time
        )

    def _evaluate_answer_relevancy(
        self, evaluation_data: List[Dict[str, Any]]
    ) -> RAGASResult:
        """
        Evaluate answer relevancy: How relevant is the answer to the question.

        Uses semantic similarity between question and answer embeddings.
        """
        start_time = time.time()
        scores = []

        # Batch encode questions and answers
        questions = [item["question"] for item in evaluation_data]
        answers = [item["answer"] for item in evaluation_data]

        try:
            question_embeddings = self.embedding_model.encode(
                questions, batch_size=self.config["batch_size"]
            )
            answer_embeddings = self.embedding_model.encode(
                answers, batch_size=self.config["batch_size"]
            )

            # Calculate cosine similarities
            similarities = util.cos_sim(question_embeddings, answer_embeddings)
            scores = [similarities[i][i].item() for i in range(len(questions))]

        except Exception as e:
            logger.error(
                f"Batch processing failed, falling back to individual processing: {e}"
            )
            for item in evaluation_data:
                try:
                    question_emb = self.embedding_model.encode([item["question"]])
                    answer_emb = self.embedding_model.encode([item["answer"]])
                    similarity = cosine_similarity(question_emb, answer_emb)[0][0]
                    scores.append(similarity)
                except Exception:
                    scores.append(0.0)

        return self._compute_metric_statistics(
            "answer_relevancy", scores, time.time() - start_time
        )

    def _evaluate_context_precision(
        self, evaluation_data: List[Dict[str, Any]]
    ) -> RAGASResult:
        """
        Evaluate context precision: What fraction of retrieved context is relevant.

        Measures how much of the retrieved context is actually useful for answering the question.
        """
        start_time = time.time()
        scores = []

        for item in evaluation_data:
            try:
                question = item["question"]
                contexts = item["contexts"]

                relevant_contexts = 0
                total_contexts = len(contexts)

                if total_contexts == 0:
                    scores.append(0.0)
                    continue

                for context in contexts:
                    # Use QA model to check if context can answer the question
                    try:
                        qa_result = self.biomedical_qa_model(
                            question=question, context=context
                        )

                        # Consider context relevant if QA confidence is high
                        if qa_result["score"] > self.config["similarity_threshold"]:
                            relevant_contexts += 1

                    except Exception:
                        # Fallback to embedding similarity
                        question_emb = self.embedding_model.encode([question])
                        context_emb = self.embedding_model.encode([context])
                        similarity = cosine_similarity(question_emb, context_emb)[0][0]

                        if similarity > self.config["similarity_threshold"]:
                            relevant_contexts += 1

                precision = relevant_contexts / total_contexts
                scores.append(precision)

            except Exception as e:
                logger.warning(f"Failed to evaluate context precision for item: {e}")
                scores.append(0.0)

        return self._compute_metric_statistics(
            "context_precision", scores, time.time() - start_time
        )

    def _evaluate_context_recall(
        self, evaluation_data: List[Dict[str, Any]]
    ) -> RAGASResult:
        """
        Evaluate context recall: How much of the relevant information was retrieved.

        Measures if the retrieved context contains information present in ground truth.
        """
        start_time = time.time()
        scores = []

        for item in evaluation_data:
            try:
                ground_truth = item["ground_truth"]
                contexts = item["contexts"]

                if not contexts:
                    scores.append(0.0)
                    continue

                # Combine all contexts
                combined_context = " ".join(contexts)

                # Use embedding similarity between ground truth and contexts
                gt_embedding = self.embedding_model.encode([ground_truth])
                context_embedding = self.embedding_model.encode([combined_context])

                similarity = cosine_similarity(gt_embedding, context_embedding)[0][0]
                scores.append(similarity)

            except Exception as e:
                logger.warning(f"Failed to evaluate context recall for item: {e}")
                scores.append(0.0)

        return self._compute_metric_statistics(
            "context_recall", scores, time.time() - start_time
        )

    def _evaluate_context_utilization(
        self, evaluation_data: List[Dict[str, Any]]
    ) -> RAGASResult:
        """
        Evaluate context utilization: How effectively was the retrieved context used.

        Measures how much of the provided context is reflected in the generated answer.
        """
        start_time = time.time()
        scores = []

        for item in evaluation_data:
            try:
                answer = item["answer"]
                contexts = item["contexts"]

                if not contexts:
                    scores.append(0.0)
                    continue

                # Calculate semantic overlap between answer and contexts
                combined_context = " ".join(contexts)

                answer_embedding = self.embedding_model.encode([answer])
                context_embedding = self.embedding_model.encode([combined_context])

                utilization = cosine_similarity(answer_embedding, context_embedding)[0][
                    0
                ]
                scores.append(utilization)

            except Exception as e:
                logger.warning(f"Failed to evaluate context utilization for item: {e}")
                scores.append(0.0)

        return self._compute_metric_statistics(
            "context_utilization", scores, time.time() - start_time
        )

    def _evaluate_answer_correctness(
        self, evaluation_data: List[Dict[str, Any]]
    ) -> RAGASResult:
        """
        Evaluate answer correctness: How factually correct is the answer.

        Combines semantic similarity with factual accuracy assessment.
        """
        start_time = time.time()
        scores = []

        # Batch processing for efficiency
        answers = [item["answer"] for item in evaluation_data]
        ground_truths = [item["ground_truth"] for item in evaluation_data]

        try:
            # Semantic similarity component
            answer_embeddings = self.embedding_model.encode(
                answers, batch_size=self.config["batch_size"]
            )
            gt_embeddings = self.embedding_model.encode(
                ground_truths, batch_size=self.config["batch_size"]
            )

            similarities = util.cos_sim(answer_embeddings, gt_embeddings)
            semantic_scores = [similarities[i][i].item() for i in range(len(answers))]

            # Factual accuracy component using entailment
            for i, (answer, gt) in enumerate(zip(answers, ground_truths)):
                try:
                    # Check if answer entails ground truth and vice versa
                    entailment1 = self.entailment_model(
                        f"Premise: {answer} Hypothesis: {gt}"
                    )
                    entailment2 = self.entailment_model(
                        f"Premise: {gt} Hypothesis: {answer}"
                    )

                    # Calculate factual score
                    factual_score = 0.0
                    if entailment1[0]["label"] == "ENTAILMENT":
                        factual_score += entailment1[0]["score"] * 0.5
                    if entailment2[0]["label"] == "ENTAILMENT":
                        factual_score += entailment2[0]["score"] * 0.5

                    # Combine semantic and factual scores
                    combined_score = 0.6 * semantic_scores[i] + 0.4 * factual_score
                    scores.append(combined_score)

                except Exception:
                    # Fallback to semantic similarity only
                    scores.append(semantic_scores[i])

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            # Fallback to individual processing
            for item in evaluation_data:
                try:
                    answer_emb = self.embedding_model.encode([item["answer"]])
                    gt_emb = self.embedding_model.encode([item["ground_truth"]])
                    similarity = cosine_similarity(answer_emb, gt_emb)[0][0]
                    scores.append(similarity)
                except Exception:
                    scores.append(0.0)

        return self._compute_metric_statistics(
            "answer_correctness", scores, time.time() - start_time
        )

    def _evaluate_answer_similarity(
        self, evaluation_data: List[Dict[str, Any]]
    ) -> RAGASResult:
        """
        Evaluate answer similarity: Semantic similarity between answer and ground truth.

        Pure semantic similarity measure using embeddings.
        """
        start_time = time.time()
        scores = []

        # Batch encode for efficiency
        answers = [item["answer"] for item in evaluation_data]
        ground_truths = [item["ground_truth"] for item in evaluation_data]

        try:
            answer_embeddings = self.embedding_model.encode(
                answers, batch_size=self.config["batch_size"]
            )
            gt_embeddings = self.embedding_model.encode(
                ground_truths, batch_size=self.config["batch_size"]
            )

            similarities = util.cos_sim(answer_embeddings, gt_embeddings)
            scores = [similarities[i][i].item() for i in range(len(answers))]

        except Exception as e:
            logger.error(f"Batch processing failed: {e}")
            for item in evaluation_data:
                try:
                    answer_emb = self.embedding_model.encode([item["answer"]])
                    gt_emb = self.embedding_model.encode([item["ground_truth"]])
                    similarity = cosine_similarity(answer_emb, gt_emb)[0][0]
                    scores.append(similarity)
                except Exception:
                    scores.append(0.0)

        return self._compute_metric_statistics(
            "answer_similarity", scores, time.time() - start_time
        )

    def _compute_metric_statistics(
        self, metric_name: str, scores: List[float], computation_time: float
    ) -> RAGASResult:
        """Compute comprehensive statistics for a metric."""
        if not scores:
            return RAGASResult(
                metric_name=metric_name,
                value=0.0,
                confidence_interval=(0.0, 0.0),
                sample_size=0,
                standard_error=0.0,
                computation_time=computation_time,
                error_message="No valid scores computed",
            )

        # Remove any NaN or infinite values
        valid_scores = [s for s in scores if np.isfinite(s)]

        if not valid_scores:
            return RAGASResult(
                metric_name=metric_name,
                value=0.0,
                confidence_interval=(0.0, 0.0),
                sample_size=0,
                standard_error=0.0,
                computation_time=computation_time,
                error_message="No valid finite scores",
            )

        # Basic statistics
        mean_score = np.mean(valid_scores)
        std_score = np.std(valid_scores, ddof=1) if len(valid_scores) > 1 else 0.0
        standard_error = (
            std_score / np.sqrt(len(valid_scores)) if len(valid_scores) > 0 else 0.0
        )

        # Confidence interval using bootstrap
        confidence_interval = self._compute_confidence_interval(valid_scores)

        return RAGASResult(
            metric_name=metric_name,
            value=mean_score,
            confidence_interval=confidence_interval,
            sample_size=len(valid_scores),
            standard_error=standard_error,
            computation_time=computation_time,
            individual_scores=valid_scores,
        )

    def _compute_confidence_interval(self, scores: List[float]) -> Tuple[float, float]:
        """Compute confidence interval using bootstrap resampling."""
        if len(scores) < 2:
            return (scores[0], scores[0]) if scores else (0.0, 0.0)

        try:
            # Bootstrap resampling
            data = np.array(scores)

            def bootstrap_mean(data):
                return np.mean(np.random.choice(data, size=len(data), replace=True))

            bootstrap_means = [
                bootstrap_mean(data) for _ in range(self.config["bootstrap_samples"])
            ]

            # Calculate confidence interval
            alpha = 1 - self.config["confidence_level"]
            lower_percentile = (alpha / 2) * 100
            upper_percentile = (1 - alpha / 2) * 100

            ci_lower = np.percentile(bootstrap_means, lower_percentile)
            ci_upper = np.percentile(bootstrap_means, upper_percentile)

            return (ci_lower, ci_upper)

        except Exception as e:
            logger.warning(f"Bootstrap confidence interval failed: {e}")
            # Fallback to normal approximation
            mean_val = np.mean(scores)
            se = np.std(scores) / np.sqrt(len(scores))
            z_score = stats.norm.ppf(1 - (1 - self.config["confidence_level"]) / 2)
            margin = z_score * se
            return (mean_val - margin, mean_val + margin)

    def _calculate_overall_score(self, results: ComprehensiveRAGASResults) -> float:
        """Calculate overall score as simple average of available metrics."""
        metric_values = []

        for metric_name in [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "context_utilization",
            "answer_correctness",
            "answer_similarity",
        ]:
            metric_result = getattr(results, metric_name)
            if metric_result and metric_result.error_message is None:
                metric_values.append(metric_result.value)

        return np.mean(metric_values) if metric_values else 0.0

    def _calculate_weighted_score(self, results: ComprehensiveRAGASResults) -> float:
        """Calculate weighted score using biomedical domain-specific weights."""
        weighted_sum = 0.0
        total_weight = 0.0

        for metric_name, weight in self.metric_weights.items():
            metric_result = getattr(results, metric_name)
            if metric_result and metric_result.error_message is None:
                weighted_sum += metric_result.value * weight
                total_weight += weight

        return weighted_sum / total_weight if total_weight > 0 else 0.0

    def compare_pipelines(
        self, pipeline_results: Dict[str, ComprehensiveRAGASResults]
    ) -> Dict[str, Any]:
        """
        Statistical comparison of multiple pipeline results.

        Args:
            pipeline_results: Dictionary mapping pipeline names to their RAGAS results

        Returns:
            Comprehensive comparison analysis with statistical significance tests
        """
        logger.info(f"Comparing {len(pipeline_results)} pipelines...")

        comparison = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "pipelines_compared": list(pipeline_results.keys()),
            "metric_comparisons": {},
            "overall_ranking": [],
            "statistical_significance": {},
            "best_pipeline_per_metric": {},
        }

        # Compare each metric across pipelines
        for metric_name in [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "context_utilization",
            "answer_correctness",
            "answer_similarity",
        ]:

            metric_data = {}
            individual_scores = {}

            for pipeline_name, results in pipeline_results.items():
                metric_result = getattr(results, metric_name)
                if metric_result and metric_result.error_message is None:
                    metric_data[pipeline_name] = metric_result.value
                    individual_scores[pipeline_name] = metric_result.individual_scores

            if len(metric_data) > 1:
                # Statistical comparison
                comparison["metric_comparisons"][metric_name] = metric_data

                # Find best pipeline for this metric
                best_pipeline = max(metric_data, key=metric_data.get)
                comparison["best_pipeline_per_metric"][metric_name] = {
                    "pipeline": best_pipeline,
                    "score": metric_data[best_pipeline],
                }

                # Statistical significance testing
                if len(individual_scores) >= 2:
                    significance_results = self._test_statistical_significance(
                        individual_scores
                    )
                    comparison["statistical_significance"][
                        metric_name
                    ] = significance_results

        # Overall ranking
        overall_scores = {}
        for pipeline_name, results in pipeline_results.items():
            if results.weighted_score is not None:
                overall_scores[pipeline_name] = results.weighted_score

        comparison["overall_ranking"] = sorted(
            overall_scores.items(), key=lambda x: x[1], reverse=True
        )

        return comparison

    def _test_statistical_significance(
        self, score_groups: Dict[str, List[float]]
    ) -> Dict[str, Any]:
        """Test statistical significance between pipeline performance."""
        pipeline_names = list(score_groups.keys())
        significance_results = {
            "test_type": "mannwhitneyu",  # Non-parametric test
            "pairwise_comparisons": {},
            "overall_p_value": None,
            "significant_differences": [],
        }

        # Pairwise comparisons
        for i in range(len(pipeline_names)):
            for j in range(i + 1, len(pipeline_names)):
                pipeline1, pipeline2 = pipeline_names[i], pipeline_names[j]
                scores1, scores2 = score_groups[pipeline1], score_groups[pipeline2]

                try:
                    # Mann-Whitney U test (non-parametric)
                    statistic, p_value = stats.mannwhitneyu(
                        scores1, scores2, alternative="two-sided"
                    )

                    comparison_key = f"{pipeline1}_vs_{pipeline2}"
                    significance_results["pairwise_comparisons"][comparison_key] = {
                        "statistic": statistic,
                        "p_value": p_value,
                        "significant": p_value < 0.05,
                        "effect_size": self._calculate_effect_size(scores1, scores2),
                    }

                    if p_value < 0.05:
                        better_pipeline = (
                            pipeline1
                            if np.mean(scores1) > np.mean(scores2)
                            else pipeline2
                        )
                        significance_results["significant_differences"].append(
                            {
                                "comparison": comparison_key,
                                "p_value": p_value,
                                "better_pipeline": better_pipeline,
                            }
                        )

                except Exception as e:
                    logger.warning(
                        f"Statistical test failed for {pipeline1} vs {pipeline2}: {e}"
                    )

        return significance_results

    def _calculate_effect_size(
        self, scores1: List[float], scores2: List[float]
    ) -> float:
        """Calculate Cohen's d effect size."""
        try:
            mean1, mean2 = np.mean(scores1), np.mean(scores2)
            pooled_std = np.sqrt(
                (
                    (len(scores1) - 1) * np.var(scores1, ddof=1)
                    + (len(scores2) - 1) * np.var(scores2, ddof=1)
                )
                / (len(scores1) + len(scores2) - 2)
            )
            return (mean1 - mean2) / pooled_std if pooled_std > 0 else 0.0
        except Exception:
            return 0.0

    def save_results(
        self,
        results: Union[ComprehensiveRAGASResults, Dict[str, Any]],
        output_path: str,
    ):
        """Save RAGAS evaluation results to file."""
        output_dir = Path(output_path)
        output_dir.mkdir(parents=True, exist_ok=True)

        if isinstance(results, ComprehensiveRAGASResults):
            # Single pipeline results
            filename = f"ragas_results_{results.pipeline_name}_{results.timestamp.replace(':', '-').replace(' ', '_')}.json"
            data = results.to_dict()
        else:
            # Comparison results
            filename = f"ragas_comparison_{results['timestamp'].replace(':', '-').replace(' ', '_')}.json"
            data = results

        filepath = output_dir / filename
        with open(filepath, "w") as f:
            json.dump(data, f, indent=2, default=str)

        logger.info(f"RAGAS results saved to {filepath}")


def create_biomedical_ragas_framework(
    config: Optional[Dict[str, Any]] = None,
) -> BiomedicalRAGASFramework:
    """Factory function to create a configured biomedical RAGAS framework."""
    return BiomedicalRAGASFramework(config)


if __name__ == "__main__":
    # Example usage
    framework = create_biomedical_ragas_framework()

    # Mock evaluation data
    sample_pipeline_results = [
        {
            "query": "What are the symptoms of diabetes?",
            "answer": "Common symptoms include increased thirst, frequent urination, and fatigue.",
            "contexts": [
                "Diabetes mellitus symptoms include polydipsia, polyuria, and general weakness."
            ],
        }
    ]

    sample_ground_truth = [
        "Diabetes symptoms include excessive thirst, frequent urination, fatigue, and blurred vision."
    ]

    results = framework.evaluate_pipeline(
        sample_pipeline_results, sample_ground_truth, "sample_pipeline"
    )

    framework.save_results(results, "output/ragas_results")
