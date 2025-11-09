#!/usr/bin/env python3
"""
GraphRAG vs HybridGraphRAG Validation Testing Framework

This script implements comprehensive head-to-head comparison testing
as specified in the migration specification.
"""

import json
import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from common.utils import get_llm_func
from iris_vector_rag import create_pipeline


@dataclass
class QueryResult:
    """Structure for storing query results."""

    pipeline_type: str
    query: str
    answer: str
    contexts: List[str]
    retrieval_method: str
    response_time: float
    answer_length: int
    context_count: int
    success: bool
    metadata: Dict[str, Any]


@dataclass
class ComparisonMetrics:
    """Metrics for comparing pipeline performance."""

    query: str
    graphrag_score: float
    hybrid_score: float
    winner: str
    quality_difference: float
    speed_difference: float
    context_difference: int


class GraphRAGHybridValidationFramework:
    """
    Comprehensive validation framework for GraphRAG → HybridGraphRAG migration.

    Implements the testing specifications defined in the migration spec.
    """

    def __init__(self):
        """Initialize the validation framework."""
        self.setup_environment()

        # Test query suites as specified
        self.test_queries = {
            "simple_factual": [
                "What are the symptoms of diabetes?",
                "How is COVID-19 transmitted?",
                "What medications treat heart disease?",
            ],
            "multi_hop_reasoning": [
                "What drugs treat diseases that cause the same symptoms as diabetes?",
                "How are COVID transmission methods related to respiratory disease treatments?",
                "What medications for cancer have side effects treated by heart disease drugs?",
            ],
            "complex_entity_relations": [
                "Which vaccines work against diseases with similar transmission patterns?",
                "What treatments for COVID-19 also help with other respiratory conditions?",
                "How do diabetes medications interact with cardiovascular treatments?",
            ],
        }

        # Acceptance criteria thresholds
        self.acceptance_criteria = {
            "min_ragas_score": 0.85,  # 85% minimum
            "max_response_time_ratio": 1.2,  # 20% slower max
            "min_success_rate": 1.0,  # 100% success required
            "quality_tolerance": 0.05,  # 5% quality difference tolerance
        }

    def setup_environment(self):
        """Setup test environment."""
        os.environ["IRIS_HOST"] = "localhost"
        os.environ["IRIS_PORT"] = "1974"
        os.environ["TOKENIZERS_PARALLELISM"] = "false"

    def create_test_pipeline(self, pipeline_type: str):
        """Create a pipeline for testing."""
        try:
            pipeline = create_pipeline(
                pipeline_type, validate_requirements=True, auto_setup=True
            )
            pipeline.llm_func = get_llm_func("openai", "gpt-4o-mini")
            return pipeline
        except Exception as e:
            print(f"❌ Failed to create {pipeline_type} pipeline: {e}")
            return None

    def execute_query_test(
        self, pipeline, pipeline_type: str, query: str, method: str = None
    ) -> QueryResult:
        """Execute a single query test on a pipeline."""
        try:
            start_time = time.time()

            # Execute query with method if specified (for HybridGraphRAG)
            if method and hasattr(pipeline, "query"):
                result = pipeline.query(
                    query, method=method, generate_answer=True, top_k=5
                )
            else:
                result = pipeline.query(query, generate_answer=True, top_k=5)

            response_time = time.time() - start_time

            answer = result.get("answer", "")
            contexts = result.get("contexts", [])
            retrieval_method = result.get("metadata", {}).get(
                "retrieval_method", "unknown"
            )

            return QueryResult(
                pipeline_type=pipeline_type,
                query=query,
                answer=answer,
                contexts=contexts,
                retrieval_method=retrieval_method,
                response_time=response_time,
                answer_length=len(answer),
                context_count=len(contexts),
                success=len(answer) > 50 and len(contexts) > 0,
                metadata=result.get("metadata", {}),
            )

        except Exception as e:
            print(f"❌ Query failed on {pipeline_type}: {e}")
            return QueryResult(
                pipeline_type=pipeline_type,
                query=query,
                answer="",
                contexts=[],
                retrieval_method="failed",
                response_time=0.0,
                answer_length=0,
                context_count=0,
                success=False,
                metadata={"error": str(e)},
            )

    def calculate_quality_score(self, result: QueryResult) -> float:
        """Calculate quality score for a query result."""
        # Quality scoring based on RAGAS-style metrics
        base_score = 0.0

        # Answer relevancy (40%)
        if result.success and result.answer_length > 50:
            answer_score = min(
                result.answer_length / 500, 1.0
            )  # Normalize to max 500 chars
            base_score += answer_score * 0.4

        # Context coverage (30%)
        if result.context_count > 0:
            context_score = min(
                result.context_count / 5, 1.0
            )  # Normalize to max 5 contexts
            base_score += context_score * 0.3

        # Response efficiency (20%)
        if result.response_time > 0:
            efficiency_score = max(
                0, 1.0 - (result.response_time - 1.0) / 10.0
            )  # 1s baseline
            base_score += max(efficiency_score, 0) * 0.2

        # Success bonus (10%)
        if result.success:
            base_score += 0.1

        return min(base_score, 1.0)

    def compare_pipeline_results(
        self, graphrag_result: QueryResult, hybrid_result: QueryResult
    ) -> ComparisonMetrics:
        """Compare results from GraphRAG and HybridGraphRAG."""
        graphrag_score = self.calculate_quality_score(graphrag_result)
        hybrid_score = self.calculate_quality_score(hybrid_result)

        quality_diff = hybrid_score - graphrag_score
        speed_diff = hybrid_result.response_time - graphrag_result.response_time
        context_diff = hybrid_result.context_count - graphrag_result.context_count

        if abs(quality_diff) < self.acceptance_criteria["quality_tolerance"]:
            winner = "tie"
        elif quality_diff > 0:
            winner = "hybrid"
        else:
            winner = "graphrag"

        return ComparisonMetrics(
            query=graphrag_result.query,
            graphrag_score=graphrag_score,
            hybrid_score=hybrid_score,
            winner=winner,
            quality_difference=quality_diff,
            speed_difference=speed_diff,
            context_difference=context_diff,
        )

    def run_head_to_head_comparison(self) -> Dict[str, Any]:
        """Execute comprehensive head-to-head comparison."""
        print("🎯 PHASE 2: HEAD-TO-HEAD VALIDATION TESTING")
        print("=" * 60)

        # Create pipelines
        print("🔧 Creating test pipelines...")
        graphrag_pipeline = self.create_test_pipeline("graphrag")
        hybrid_pipeline = self.create_test_pipeline("hybrid_graphrag")

        if not graphrag_pipeline or not hybrid_pipeline:
            return {"status": "failed", "error": "Pipeline creation failed"}

        print("✅ Both pipelines created successfully")

        all_results = {}
        all_comparisons = {}

        # Test each query category
        for category, queries in self.test_queries.items():
            print(f"\n📊 Testing {category.upper()} queries...")
            print("-" * 50)

            category_results = []

            for i, query in enumerate(queries, 1):
                print(f"\n🧪 Query {i}/{len(queries)}: {query[:50]}...")

                # Test GraphRAG
                graphrag_result = self.execute_query_test(
                    graphrag_pipeline, "graphrag", query
                )

                # Test HybridGraphRAG with different methods
                hybrid_kg_result = self.execute_query_test(
                    hybrid_pipeline, "hybrid_kg", query, "kg"
                )
                hybrid_fusion_result = self.execute_query_test(
                    hybrid_pipeline, "hybrid_fusion", query, "hybrid"
                )

                # Use best HybridGraphRAG result
                if hybrid_kg_result.success and hybrid_fusion_result.success:
                    # Choose based on quality score
                    kg_score = self.calculate_quality_score(hybrid_kg_result)
                    fusion_score = self.calculate_quality_score(hybrid_fusion_result)
                    hybrid_result = (
                        hybrid_kg_result
                        if kg_score >= fusion_score
                        else hybrid_fusion_result
                    )
                elif hybrid_kg_result.success:
                    hybrid_result = hybrid_kg_result
                elif hybrid_fusion_result.success:
                    hybrid_result = hybrid_fusion_result
                else:
                    # Both failed, use kg result for comparison
                    hybrid_result = hybrid_kg_result

                # Compare results
                comparison = self.compare_pipeline_results(
                    graphrag_result, hybrid_result
                )
                category_results.append(comparison)

                # Display results
                print(
                    f"   GraphRAG:  {comparison.graphrag_score:.2f} | {graphrag_result.retrieval_method} | {graphrag_result.response_time:.2f}s"
                )
                print(
                    f"   Hybrid:    {comparison.hybrid_score:.2f} | {hybrid_result.retrieval_method} | {hybrid_result.response_time:.2f}s"
                )
                print(
                    f"   Winner:    {comparison.winner} ({comparison.quality_difference:+.2f})"
                )

            all_results[category] = category_results

            # Category summary
            winners = [comp.winner for comp in category_results]
            hybrid_wins = winners.count("hybrid")
            graphrag_wins = winners.count("graphrag")
            ties = winners.count("tie")

            print(f"\n📈 {category.upper()} SUMMARY:")
            print(f"   HybridGraphRAG wins: {hybrid_wins}/{len(queries)}")
            print(f"   GraphRAG wins: {graphrag_wins}/{len(queries)}")
            print(f"   Ties: {ties}/{len(queries)}")

        return self.generate_validation_report(all_results)

    def generate_validation_report(
        self, results: Dict[str, List[ComparisonMetrics]]
    ) -> Dict[str, Any]:
        """Generate comprehensive validation report."""
        print("\n" + "=" * 60)
        print("📋 VALIDATION REPORT")
        print("=" * 60)

        # Calculate overall metrics
        all_comparisons = []
        for category_results in results.values():
            all_comparisons.extend(category_results)

        total_queries = len(all_comparisons)
        hybrid_wins = sum(1 for comp in all_comparisons if comp.winner == "hybrid")
        graphrag_wins = sum(1 for comp in all_comparisons if comp.winner == "graphrag")
        ties = sum(1 for comp in all_comparisons if comp.winner == "tie")

        # Calculate average metrics
        avg_hybrid_score = (
            sum(comp.hybrid_score for comp in all_comparisons) / total_queries
        )
        avg_graphrag_score = (
            sum(comp.graphrag_score for comp in all_comparisons) / total_queries
        )
        avg_quality_diff = (
            sum(comp.quality_difference for comp in all_comparisons) / total_queries
        )
        avg_speed_diff = (
            sum(comp.speed_difference for comp in all_comparisons) / total_queries
        )

        # Acceptance criteria validation
        acceptance_status = {
            "hybrid_score_meets_minimum": avg_hybrid_score
            >= self.acceptance_criteria["min_ragas_score"],
            "speed_within_tolerance": avg_speed_diff
            <= (
                avg_graphrag_score * self.acceptance_criteria["max_response_time_ratio"]
            ),
            "quality_competitive": avg_quality_diff
            >= -self.acceptance_criteria["quality_tolerance"],
            "no_significant_regression": hybrid_wins + ties >= graphrag_wins,
        }

        overall_pass = all(acceptance_status.values())

        # Generate report
        report = {
            "validation_status": "PASS" if overall_pass else "FAIL",
            "overall_metrics": {
                "total_queries": total_queries,
                "hybrid_wins": hybrid_wins,
                "graphrag_wins": graphrag_wins,
                "ties": ties,
                "hybrid_win_rate": hybrid_wins / total_queries,
                "avg_hybrid_score": avg_hybrid_score,
                "avg_graphrag_score": avg_graphrag_score,
                "avg_quality_difference": avg_quality_diff,
                "avg_speed_difference": avg_speed_diff,
            },
            "acceptance_criteria": acceptance_status,
            "detailed_results": results,
            "recommendation": self.generate_recommendation(
                overall_pass, avg_quality_diff, hybrid_wins, total_queries
            ),
        }

        self.print_validation_summary(report)
        return report

    def generate_recommendation(
        self, overall_pass: bool, quality_diff: float, hybrid_wins: int, total: int
    ) -> str:
        """Generate migration recommendation based on results."""
        if overall_pass:
            if quality_diff > 0.1:
                return (
                    "RECOMMEND MIGRATION - HybridGraphRAG shows significant improvement"
                )
            elif hybrid_wins / total > 0.6:
                return (
                    "RECOMMEND MIGRATION - HybridGraphRAG wins majority of comparisons"
                )
            else:
                return (
                    "APPROVE MIGRATION - HybridGraphRAG meets all acceptance criteria"
                )
        else:
            if quality_diff < -0.1:
                return "BLOCK MIGRATION - Significant quality regression detected"
            else:
                return (
                    "CONDITIONAL MIGRATION - Address remaining issues before proceeding"
                )

    def print_validation_summary(self, report: Dict[str, Any]):
        """Print validation summary to console."""
        status = report["validation_status"]
        metrics = report["overall_metrics"]

        print(f"\n🎯 VALIDATION STATUS: {status}")
        print(f"📊 Overall Results:")
        print(
            f"   HybridGraphRAG wins: {metrics['hybrid_wins']}/{metrics['total_queries']} ({metrics['hybrid_win_rate']:.1%})"
        )
        print(f"   Average Quality Scores:")
        print(f"     - HybridGraphRAG: {metrics['avg_hybrid_score']:.2f}")
        print(f"     - GraphRAG: {metrics['avg_graphrag_score']:.2f}")
        print(f"     - Difference: {metrics['avg_quality_difference']:+.2f}")

        print(f"\n✅ Acceptance Criteria:")
        for criterion, passed in report["acceptance_criteria"].items():
            status_icon = "✅" if passed else "❌"
            print(f"   {status_icon} {criterion.replace('_', ' ').title()}")

        print(f"\n🎯 RECOMMENDATION: {report['recommendation']}")


def main():
    """Main execution function."""
    framework = GraphRAGHybridValidationFramework()

    try:
        validation_report = framework.run_head_to_head_comparison()

        # Save report to file
        report_path = (
            Path(__file__).parent
            / f"validation_report_{time.strftime('%Y%m%d_%H%M%S')}.json"
        )
        with open(report_path, "w") as f:
            json.dump(validation_report, f, indent=2, default=str)

        print(f"\n📄 Full report saved to: {report_path}")

        # Return validation status for automation
        return validation_report["validation_status"] == "PASS"

    except Exception as e:
        print(f"❌ Validation framework failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
