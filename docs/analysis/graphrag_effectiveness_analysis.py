#!/usr/bin/env python3
"""
GraphRAG Effectiveness Analysis

Comprehensive analysis of GraphRAG entity/relation storage patterns and
performance comparison against BasicRAG across different query types.

Author: Data Science Team
Date: 2025-09-14
"""

import json
import logging
import os
import sys
from pathlib import Path
from typing import Any, Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# Add paths for database connectivity
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))
sys.path.insert(
    0,
    os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "..", "evaluation_framework"
    ),
)

# Database imports
from common.iris_dbapi_connector import get_iris_dbapi_connection  # noqa: E402

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


class GraphRAGAnalyzer:
    """Comprehensive analyzer for GraphRAG data and performance."""

    def __init__(self):
        """Initialize the analyzer with database connection."""
        self.conn = get_iris_dbapi_connection()
        if not self.conn:
            raise ValueError("Failed to connect to IRIS database")

        # Results storage
        self.entity_stats = {}
        self.relation_stats = {}
        self.performance_data = {}

        # Create output directory
        self.output_dir = Path("analysis/outputs")
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Set up plotting style
        plt.style.use("seaborn-v0_8")
        sns.set_palette("husl")

    def analyze_graph_data_distribution(self) -> Dict[str, Any]:
        """Analyze entity and relation distribution per document."""
        logger.info("🔍 Analyzing graph data distribution...")

        # Query entities per document
        cursor = self.conn.cursor()

        # Get entity counts per document
        entity_query = """
        SELECT
            source_document_id,
            COUNT(*) as entity_count,
            COUNT(DISTINCT entity_type) as unique_entity_types
        FROM RAG.GraphEntities
        GROUP BY source_document_id
        ORDER BY entity_count DESC
        """

        try:
            cursor.execute(entity_query)
            entity_results = cursor.fetchall()

            if entity_results:
                entity_df = pd.DataFrame(
                    entity_results, columns=["doc_id", "entity_count", "unique_types"]
                )

                # Get relation counts per document
                relation_query = """
                SELECT
                    source_document_id,
                    COUNT(*) as relation_count,
                    COUNT(DISTINCT relation_type) as unique_relation_types
                FROM RAG.GraphRelations
                GROUP BY source_document_id
                ORDER BY relation_count DESC
                """

                cursor.execute(relation_query)
                relation_results = cursor.fetchall()

                if relation_results:
                    relation_df = pd.DataFrame(
                        relation_results,
                        columns=["doc_id", "relation_count", "unique_rel_types"],
                    )

                    # Merge data
                    graph_stats = pd.merge(
                        entity_df, relation_df, on="doc_id", how="outer"
                    ).fillna(0)

                    # Calculate statistics
                    stats = {
                        "total_documents_with_entities": len(entity_df),
                        "total_documents_with_relations": len(relation_df),
                        "entity_stats": {
                            "mean_entities_per_doc": float(
                                entity_df["entity_count"].mean()
                            ),
                            "median_entities_per_doc": float(
                                entity_df["entity_count"].median()
                            ),
                            "std_entities_per_doc": float(
                                entity_df["entity_count"].std()
                            ),
                            "max_entities_per_doc": int(
                                entity_df["entity_count"].max()
                            ),
                            "min_entities_per_doc": int(
                                entity_df["entity_count"].min()
                            ),
                        },
                        "relation_stats": {
                            "mean_relations_per_doc": float(
                                relation_df["relation_count"].mean()
                            ),
                            "median_relations_per_doc": float(
                                relation_df["relation_count"].median()
                            ),
                            "std_relations_per_doc": float(
                                relation_df["relation_count"].std()
                            ),
                            "max_relations_per_doc": int(
                                relation_df["relation_count"].max()
                            ),
                            "min_relations_per_doc": int(
                                relation_df["relation_count"].min()
                            ),
                        },
                    }

                    # Create visualizations
                    self._create_distribution_plots(graph_stats)

                    # Save detailed data
                    graph_stats.to_csv(
                        self.output_dir / "graph_data_distribution.csv", index=False
                    )

                    logger.info(
                        f"✅ Found {len(graph_stats)} documents with graph data"
                    )
                    return stats

            logger.warning("⚠️  No graph data found in database")
            return {"message": "No graph entities or relations found"}

        except Exception as e:
            logger.error(f"❌ Error analyzing graph distribution: {e}")
            return {"error": str(e)}
        finally:
            cursor.close()

    def analyze_entity_relation_characteristics(self) -> Dict[str, Any]:
        """Analyze the types and characteristics of entities and relations."""
        logger.info("🔬 Analyzing entity and relation characteristics...")

        cursor = self.conn.cursor()

        try:
            # Analyze entity types
            entity_type_query = """
            SELECT
                entity_type,
                COUNT(*) as count,
                AVG(LENGTH(entity_text)) as avg_text_length
            FROM RAG.GraphEntities
            GROUP BY entity_type
            ORDER BY count DESC
            LIMIT 20
            """

            cursor.execute(entity_type_query)
            entity_type_results = cursor.fetchall()

            # Analyze relation types
            relation_type_query = """
            SELECT
                relation_type,
                COUNT(*) as count,
                AVG(confidence_score) as avg_confidence
            FROM RAG.GraphRelations
            GROUP BY relation_type
            ORDER BY count DESC
            LIMIT 20
            """

            cursor.execute(relation_type_query)
            relation_type_results = cursor.fetchall()

            # Sample entities for qualitative analysis
            entity_sample_query = """
            SELECT
                entity_type,
                entity_text,
                confidence_score
            FROM RAG.GraphEntities
            ORDER BY RANDOM()
            LIMIT 50
            """

            cursor.execute(entity_sample_query)
            entity_samples = cursor.fetchall()

            # Sample relations for qualitative analysis
            relation_sample_query = """
            SELECT
                r.relation_type,
                e1.entity_text as source_entity,
                e2.entity_text as target_entity,
                r.confidence_score
            FROM RAG.GraphRelations r
            JOIN RAG.GraphEntities e1 ON r.source_entity_id = e1.entity_id
            JOIN RAG.GraphEntities e2 ON r.target_entity_id = e2.entity_id
            ORDER BY RANDOM()
            LIMIT 30
            """

            cursor.execute(relation_sample_query)
            relation_samples = cursor.fetchall()

            # Compile results
            characteristics = {
                "entity_types": [
                    {"type": row[0], "count": row[1], "avg_length": row[2]}
                    for row in entity_type_results
                ],
                "relation_types": [
                    {"type": row[0], "count": row[1], "avg_confidence": row[2]}
                    for row in relation_type_results
                ],
                "entity_samples": [
                    {"type": row[0], "text": row[1], "confidence": row[2]}
                    for row in entity_samples
                ],
                "relation_samples": [
                    {
                        "type": row[0],
                        "source": row[1],
                        "target": row[2],
                        "confidence": row[3],
                    }
                    for row in relation_samples
                ],
            }

            # Create visualizations
            self._create_characteristic_plots(characteristics)

            # Save data
            with open(
                self.output_dir / "entity_relation_characteristics.json", "w"
            ) as f:
                json.dump(characteristics, f, indent=2, default=str)

            return characteristics

        except Exception as e:
            logger.error(f"❌ Error analyzing characteristics: {e}")
            return {"error": str(e)}
        finally:
            cursor.close()

    def analyze_performance_comparison(self) -> Dict[str, Any]:
        """Compare GraphRAG vs BasicRAG performance across evaluation results."""
        logger.info("📊 Analyzing GraphRAG vs BasicRAG performance...")

        # Look for recent evaluation results
        results_dir = Path("evaluation_framework/outputs/production_evaluation/results")

        if not results_dir.exists():
            logger.warning("⚠️  No evaluation results found")
            return {"message": "No evaluation results available for comparison"}

        # Find most recent evaluation
        eval_dirs = [d for d in results_dir.iterdir() if d.is_dir()]
        if not eval_dirs:
            logger.warning("⚠️  No evaluation directories found")
            return {"message": "No evaluation directories found"}

        latest_dir = max(eval_dirs, key=lambda d: d.stat().st_mtime)
        logger.info(f"📁 Using evaluation results from: {latest_dir.name}")

        # Load evaluation results
        results_file = latest_dir / "evaluation_results.json"
        if not results_file.exists():
            logger.warning("⚠️  evaluation_results.json not found")
            return {"message": "Evaluation results file not found"}

        try:
            with open(results_file, "r") as f:
                eval_results = json.load(f)

            # Extract pipeline results
            pipeline_results = eval_results.get("pipeline_results", {})

            if not pipeline_results:
                logger.warning("⚠️  No pipeline results found in evaluation data")
                return {"message": "No pipeline results found"}

            # Find GraphRAG and BasicRAG results
            graphrag_results = None
            basicrag_results = None

            for pipeline_name, results in pipeline_results.items():
                if "graphrag" in pipeline_name.lower():
                    graphrag_results = results
                elif (
                    "basicrag" in pipeline_name.lower()
                    and "rerank" not in pipeline_name.lower()
                ):
                    basicrag_results = results

            if not graphrag_results or not basicrag_results:
                logger.warning("⚠️  Could not find both GraphRAG and BasicRAG results")
                return {"message": "Missing GraphRAG or BasicRAG results"}

            # Compare metrics
            comparison = self._compare_pipeline_metrics(
                graphrag_results, basicrag_results
            )

            # Analyze query-specific performance
            query_analysis = self._analyze_query_specific_performance(
                latest_dir, graphrag_results, basicrag_results
            )

            performance_data = {
                "metric_comparison": comparison,
                "query_analysis": query_analysis,
                "evaluation_metadata": {
                    "evaluation_id": eval_results.get("experiment_id"),
                    "total_questions": eval_results.get("total_questions"),
                    "total_documents": eval_results.get("total_documents"),
                },
            }

            # Create performance visualizations
            self._create_performance_plots(performance_data)

            # Save results
            with open(self.output_dir / "performance_comparison.json", "w") as f:
                json.dump(performance_data, f, indent=2, default=str)

            return performance_data

        except Exception as e:
            logger.error(f"❌ Error analyzing performance: {e}")
            return {"error": str(e)}

    def _compare_pipeline_metrics(
        self, graphrag: Dict, basicrag: Dict
    ) -> Dict[str, Any]:
        """Compare metrics between GraphRAG and BasicRAG."""

        # Extract RAGAS metrics
        graphrag_metrics = graphrag.get("ragas_evaluation", {}).get("metrics", {})
        basicrag_metrics = basicrag.get("ragas_evaluation", {}).get("metrics", {})

        comparison = {}

        for metric in [
            "faithfulness",
            "answer_relevancy",
            "context_precision",
            "context_recall",
            "answer_similarity",
            "answer_correctness",
        ]:

            if metric in graphrag_metrics and metric in basicrag_metrics:
                graph_score = graphrag_metrics[metric]
                basic_score = basicrag_metrics[metric]

                comparison[metric] = {
                    "graphrag_score": graph_score,
                    "basicrag_score": basic_score,
                    "difference": graph_score - basic_score,
                    "percent_improvement": (
                        ((graph_score - basic_score) / basic_score * 100)
                        if basic_score > 0
                        else 0
                    ),
                }

        return comparison

    def _analyze_query_specific_performance(
        self, eval_dir: Path, graphrag: Dict, basicrag: Dict
    ) -> Dict[str, Any]:
        """Analyze performance on specific query types."""

        # Look for detailed question-level results
        detailed_files = list(eval_dir.glob("*detailed*results*.json"))

        if not detailed_files:
            return {"message": "No detailed results available for query analysis"}

        try:
            with open(detailed_files[0], "r") as f:
                detailed_results = json.load(f)

            # Analyze question categories or types if available
            query_analysis = {
                "total_questions_analyzed": len(detailed_results.get("questions", [])),
                "performance_by_question_length": self._analyze_by_question_length(
                    detailed_results
                ),
                "performance_trends": self._analyze_performance_trends(
                    detailed_results
                ),
            }

            return query_analysis

        except Exception as e:
            logger.warning(f"Could not analyze detailed results: {e}")
            return {"message": "Detailed query analysis not available"}

    def _analyze_by_question_length(self, detailed_results: Dict) -> Dict[str, Any]:
        """Analyze performance by question length categories."""
        # Placeholder for question length analysis
        return {"message": "Question length analysis would be implemented here"}

    def _analyze_performance_trends(self, detailed_results: Dict) -> Dict[str, Any]:
        """Analyze performance trends across different aspects."""
        # Placeholder for trend analysis
        return {"message": "Performance trends analysis would be implemented here"}

    def _create_distribution_plots(self, graph_stats: pd.DataFrame):
        """Create distribution visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))

        # Entity count distribution
        axes[0, 0].hist(
            graph_stats["entity_count"], bins=20, alpha=0.7, color="skyblue"
        )
        axes[0, 0].set_title("Distribution of Entities per Document")
        axes[0, 0].set_xlabel("Number of Entities")
        axes[0, 0].set_ylabel("Number of Documents")

        # Relation count distribution
        axes[0, 1].hist(
            graph_stats["relation_count"], bins=20, alpha=0.7, color="lightcoral"
        )
        axes[0, 1].set_title("Distribution of Relations per Document")
        axes[0, 1].set_xlabel("Number of Relations")
        axes[0, 1].set_ylabel("Number of Documents")

        # Entity vs Relation scatter
        axes[1, 0].scatter(
            graph_stats["entity_count"],
            graph_stats["relation_count"],
            alpha=0.6,
            color="green",
        )
        axes[1, 0].set_title("Entities vs Relations per Document")
        axes[1, 0].set_xlabel("Number of Entities")
        axes[1, 0].set_ylabel("Number of Relations")

        # Entity type diversity
        axes[1, 1].hist(graph_stats["unique_types"], bins=10, alpha=0.7, color="gold")
        axes[1, 1].set_title("Entity Type Diversity per Document")
        axes[1, 1].set_xlabel("Number of Unique Entity Types")
        axes[1, 1].set_ylabel("Number of Documents")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "graph_data_distributions.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _create_characteristic_plots(self, characteristics: Dict):
        """Create characteristic visualizations."""
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))

        # Entity types bar chart
        if characteristics["entity_types"]:
            entity_df = pd.DataFrame(characteristics["entity_types"])
            entity_df = entity_df.head(10)  # Top 10
            axes[0, 0].bar(range(len(entity_df)), entity_df["count"])
            axes[0, 0].set_title("Top Entity Types by Frequency")
            axes[0, 0].set_xlabel("Entity Type")
            axes[0, 0].set_ylabel("Count")
            axes[0, 0].set_xticks(range(len(entity_df)))
            axes[0, 0].set_xticklabels(entity_df["type"], rotation=45, ha="right")

        # Relation types bar chart
        if characteristics["relation_types"]:
            relation_df = pd.DataFrame(characteristics["relation_types"])
            relation_df = relation_df.head(10)  # Top 10
            axes[0, 1].bar(range(len(relation_df)), relation_df["count"])
            axes[0, 1].set_title("Top Relation Types by Frequency")
            axes[0, 1].set_xlabel("Relation Type")
            axes[0, 1].set_ylabel("Count")
            axes[0, 1].set_xticks(range(len(relation_df)))
            axes[0, 1].set_xticklabels(relation_df["type"], rotation=45, ha="right")

        # Entity confidence distribution
        if characteristics["entity_samples"]:
            confidences = [
                s["confidence"]
                for s in characteristics["entity_samples"]
                if s["confidence"]
            ]
            if confidences:
                axes[1, 0].hist(confidences, bins=15, alpha=0.7, color="lightblue")
                axes[1, 0].set_title("Entity Confidence Score Distribution")
                axes[1, 0].set_xlabel("Confidence Score")
                axes[1, 0].set_ylabel("Count")

        # Relation confidence distribution
        if characteristics["relation_samples"]:
            rel_confidences = [
                s["confidence"]
                for s in characteristics["relation_samples"]
                if s["confidence"]
            ]
            if rel_confidences:
                axes[1, 1].hist(rel_confidences, bins=15, alpha=0.7, color="lightgreen")
                axes[1, 1].set_title("Relation Confidence Score Distribution")
                axes[1, 1].set_xlabel("Confidence Score")
                axes[1, 1].set_ylabel("Count")

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "entity_relation_characteristics.png",
            dpi=300,
            bbox_inches="tight",
        )
        plt.close()

    def _create_performance_plots(self, performance_data: Dict):
        """Create performance comparison visualizations."""

        comparison = performance_data.get("metric_comparison", {})
        if not comparison:
            return

        # Metric comparison bar chart
        metrics = list(comparison.keys())
        graphrag_scores = [comparison[m]["graphrag_score"] for m in metrics]
        basicrag_scores = [comparison[m]["basicrag_score"] for m in metrics]

        x = np.arange(len(metrics))
        width = 0.35

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Score comparison
        ax1.bar(x - width / 2, graphrag_scores, width, label="GraphRAG", alpha=0.8)
        ax1.bar(x + width / 2, basicrag_scores, width, label="BasicRAG", alpha=0.8)
        ax1.set_title("GraphRAG vs BasicRAG Performance Comparison")
        ax1.set_xlabel("Metrics")
        ax1.set_ylabel("Score")
        ax1.set_xticks(x)
        ax1.set_xticklabels(metrics, rotation=45, ha="right")
        ax1.legend()
        ax1.set_ylim(0, 1)

        # Improvement percentages
        improvements = [comparison[m]["percent_improvement"] for m in metrics]
        colors = ["green" if imp > 0 else "red" for imp in improvements]

        ax2.bar(metrics, improvements, color=colors, alpha=0.7)
        ax2.set_title("GraphRAG Performance Improvement (%)")
        ax2.set_xlabel("Metrics")
        ax2.set_ylabel("Improvement (%)")
        ax2.set_xticklabels(metrics, rotation=45, ha="right")
        ax2.axhline(y=0, color="black", linestyle="-", alpha=0.3)

        plt.tight_layout()
        plt.savefig(
            self.output_dir / "performance_comparison.png", dpi=300, bbox_inches="tight"
        )
        plt.close()

    def generate_comprehensive_report(self) -> str:
        """Generate a comprehensive analysis report."""
        logger.info("📝 Generating comprehensive analysis report...")

        # Run all analyses
        distribution_results = self.analyze_graph_data_distribution()
        characteristics_results = self.analyze_entity_relation_characteristics()
        performance_results = self.analyze_performance_comparison()

        # Generate report
        report = f"""# GraphRAG Effectiveness Analysis Report

## Executive Summary

This report analyzes the effectiveness of GraphRAG implementation in the RAG Templates framework,
examining entity/relation storage patterns and performance comparisons against BasicRAG.

## 1. Graph Data Distribution Analysis

{self._format_distribution_summary(distribution_results)}

## 2. Entity and Relation Characteristics

{self._format_characteristics_summary(characteristics_results)}

## 3. Performance Comparison: GraphRAG vs BasicRAG

{self._format_performance_summary(performance_results)}

## Key Findings

{self._generate_key_findings(distribution_results, characteristics_results, performance_results)}

## Recommendations

{self._generate_recommendations(distribution_results, characteristics_results, performance_results)}

---
*Report generated on {pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')}*
*Analysis artifacts saved to: {self.output_dir.absolute()}*
"""

        # Save report
        report_path = self.output_dir / "graphrag_effectiveness_report.md"
        with open(report_path, "w") as f:
            f.write(report)

        logger.info(f"✅ Report saved to: {report_path}")
        return report

    def _format_distribution_summary(self, results: Dict) -> str:
        """Format distribution analysis summary."""
        if "error" in results or "message" in results:
            return f"**Status**: {results.get('error', results.get('message'))}"

        entity_stats = results.get("entity_stats", {})
        relation_stats = results.get("relation_stats", {})

        return f"""
**Documents with Graph Data**: {results.get('total_documents_with_entities', 0)} (entities), {results.get('total_documents_with_relations', 0)} (relations)

**Entity Statistics per Document**:
- Mean: {entity_stats.get('mean_entities_per_doc', 0):.1f}
- Median: {entity_stats.get('median_entities_per_doc', 0):.1f}
- Range: {entity_stats.get('min_entities_per_doc', 0)} - {entity_stats.get('max_entities_per_doc', 0)}

**Relation Statistics per Document**:
- Mean: {relation_stats.get('mean_relations_per_doc', 0):.1f}
- Median: {relation_stats.get('median_relations_per_doc', 0):.1f}
- Range: {relation_stats.get('min_relations_per_doc', 0)} - {relation_stats.get('max_relations_per_doc', 0)}
"""

    def _format_characteristics_summary(self, results: Dict) -> str:
        """Format characteristics analysis summary."""
        if "error" in results:
            return f"**Status**: {results['error']}"

        entity_types = results.get("entity_types", [])
        relation_types = results.get("relation_types", [])

        top_entities = ", ".join([et["type"] for et in entity_types[:5]])
        top_relations = ", ".join([rt["type"] for rt in relation_types[:5]])

        return f"""
**Entity Types Found**: {len(entity_types)} unique types
- Top 5: {top_entities}

**Relation Types Found**: {len(relation_types)} unique types
- Top 5: {top_relations}

**Sample Entity Examples**:
{self._format_entity_samples(results.get('entity_samples', [])[:3])}

**Sample Relation Examples**:
{self._format_relation_samples(results.get('relation_samples', [])[:3])}
"""

    def _format_entity_samples(self, samples: List[Dict]) -> str:
        """Format entity sample examples."""
        if not samples:
            return "- No samples available"

        formatted = []
        for sample in samples:
            formatted.append(
                f"- **{sample['type']}**: \"{sample['text']}\" (confidence: {sample['confidence']:.2f})"
            )

        return "\n".join(formatted)

    def _format_relation_samples(self, samples: List[Dict]) -> str:
        """Format relation sample examples."""
        if not samples:
            return "- No samples available"

        formatted = []
        for sample in samples:
            formatted.append(
                f"- **{sample['type']}**: \"{sample['source']}\" → \"{sample['target']}\" (confidence: {sample['confidence']:.2f})"
            )

        return "\n".join(formatted)

    def _format_performance_summary(self, results: Dict) -> str:
        """Format performance comparison summary."""
        if "error" in results or "message" in results:
            return f"**Status**: {results.get('error', results.get('message'))}"

        comparison = results.get("metric_comparison", {})
        if not comparison:
            return "**Status**: No performance comparison data available"

        summary = "**RAGAS Metrics Comparison**:\n\n"

        for metric, data in comparison.items():
            improvement = data["percent_improvement"]
            symbol = "📈" if improvement > 0 else "📉" if improvement < 0 else "➡️"

            summary += f"- **{metric.replace('_', ' ').title()}**: "
            summary += f"GraphRAG {data['graphrag_score']:.3f} vs BasicRAG {data['basicrag_score']:.3f} "
            summary += f"{symbol} {improvement:+.1f}%\n"

        return summary

    def _generate_key_findings(self, dist: Dict, char: Dict, perf: Dict) -> str:
        """Generate key findings section."""
        findings = []

        # Graph data findings
        if "entity_stats" in dist:
            avg_entities = dist["entity_stats"].get("mean_entities_per_doc", 0)
            findings.append(
                f"📊 **Graph Density**: Average of {avg_entities:.1f} entities extracted per document"
            )

        # Performance findings
        if "metric_comparison" in perf:
            improvements = [
                data["percent_improvement"]
                for data in perf["metric_comparison"].values()
            ]
            if improvements:
                avg_improvement = np.mean(improvements)
                if avg_improvement > 0:
                    findings.append(
                        f"🚀 **Performance**: GraphRAG shows average {avg_improvement:.1f}% improvement over BasicRAG"
                    )
                else:
                    findings.append(
                        f"⚠️ **Performance**: GraphRAG shows average {avg_improvement:.1f}% decline vs BasicRAG"
                    )

        # Entity diversity findings
        if "entity_types" in char and char["entity_types"]:
            entity_count = len(char["entity_types"])
            findings.append(
                f"🏷️ **Entity Diversity**: {entity_count} distinct entity types identified"
            )

        return "\n".join(findings) if findings else "- Analysis in progress..."

    def _generate_recommendations(self, dist: Dict, char: Dict, perf: Dict) -> str:
        """Generate recommendations section."""
        recommendations = []

        # Performance-based recommendations
        if "metric_comparison" in perf:
            improvements = [
                data["percent_improvement"]
                for data in perf["metric_comparison"].values()
            ]
            if improvements and np.mean(improvements) < 0:
                recommendations.append(
                    "🔧 **Optimize GraphRAG**: Current performance suggests need for parameter tuning"
                )
            elif improvements and np.mean(improvements) > 5:
                recommendations.append(
                    "✅ **Scale GraphRAG**: Strong performance warrants broader deployment"
                )

        # Data quality recommendations
        if "entity_stats" in dist:
            avg_entities = dist["entity_stats"].get("mean_entities_per_doc", 0)
            if avg_entities < 5:
                recommendations.append(
                    "📈 **Enhance Extraction**: Low entity count suggests improving NER models"
                )

        return (
            "\n".join(recommendations)
            if recommendations
            else "- Awaiting analysis completion for specific recommendations..."
        )


def main():
    """Main analysis execution."""
    logger.info("🚀 Starting GraphRAG Effectiveness Analysis")

    try:
        analyzer = GraphRAGAnalyzer()
        report = analyzer.generate_comprehensive_report()

        logger.info("✅ Analysis completed successfully!")
        print("\n" + "=" * 80)
        print("GRAPHRAG EFFECTIVENESS ANALYSIS COMPLETE")
        print("=" * 80)
        print(report)
        print("=" * 80)

    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        raise


if __name__ == "__main__":
    main()
