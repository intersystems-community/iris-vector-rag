#!/usr/bin/env python3
"""
PyLate ColBERT RAGAS Demo - Mock Evaluation

This is a demo version that shows the RAGAS evaluation flow without requiring
a live IRIS database connection. It uses mock results to demonstrate the
comparison and reporting capabilities.

Usage:
    python scripts/test_pylate_ragas_demo.py
"""

import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def generate_mock_results(pipeline_type: str, num_queries: int = 5) -> Dict[str, Any]:
    """Generate realistic mock results for a pipeline."""

    # Pipeline characteristics (realistic benchmarks)
    characteristics = {
        "basic": {
            "success_rate": 1.0,
            "avg_query_time": 1.2,
            "avg_answer_length": 235,
            "avg_contexts": 3.0
        },
        "basic_rerank": {
            "success_rate": 1.0,
            "avg_query_time": 1.5,
            "avg_answer_length": 245,
            "avg_contexts": 3.0
        },
        "pylate_colbert": {
            "success_rate": 1.0,
            "avg_query_time": 1.7,
            "avg_answer_length": 250,
            "avg_contexts": 3.0
        },
        "crag": {
            "success_rate": 1.0,
            "avg_query_time": 2.3,
            "avg_answer_length": 260,
            "avg_contexts": 3.2
        }
    }

    char = characteristics.get(pipeline_type, characteristics["basic"])

    # Generate query results
    results = []
    for i in range(num_queries):
        # Add some variance
        query_time = char["avg_query_time"] * random.uniform(0.8, 1.2)
        answer_length = int(char["avg_answer_length"] * random.uniform(0.9, 1.1))
        num_contexts = int(char["avg_contexts"])

        results.append({
            "query": f"Query {i+1}",
            "answer": "A" * answer_length,  # Mock answer
            "contexts": [f"Context {j}" for j in range(num_contexts)],
            "ground_truth": f"Ground truth {i+1}",
            "num_contexts": num_contexts,
            "answer_length": answer_length,
            "query_time": query_time,
            "success": True,
            "retrieved_docs": num_contexts
        })

    total_time = sum(r["query_time"] for r in results)
    successful = [r for r in results if r["success"]]

    return {
        "pipeline_type": pipeline_type,
        "total_queries": num_queries,
        "successful_queries": len(successful),
        "success_rate": len(successful) / num_queries,
        "avg_answer_length": sum(r["answer_length"] for r in successful) / len(successful),
        "avg_contexts_retrieved": sum(r["num_contexts"] for r in successful) / len(successful),
        "avg_query_time": sum(r["query_time"] for r in successful) / len(successful),
        "total_time": total_time,
        "queries_per_second": num_queries / total_time,
        "results": results
    }


def compare_pipelines(all_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Compare results across all pipelines."""
    logger.info("\n📊 Comparing Pipeline Performance...")

    comparison = {
        "timestamp": datetime.now().isoformat(),
        "pipelines_tested": len(all_results),
        "rankings": {}
    }

    # Rank by success rate
    sorted_by_success = sorted(all_results, key=lambda x: x.get("success_rate", 0), reverse=True)
    comparison["rankings"]["by_success_rate"] = [
        {"pipeline": r["pipeline_type"], "score": r.get("success_rate", 0)}
        for r in sorted_by_success
    ]

    # Rank by query speed
    sorted_by_speed = sorted(
        [r for r in all_results if r.get("avg_query_time", 0) > 0],
        key=lambda x: x.get("avg_query_time", float('inf'))
    )
    comparison["rankings"]["by_speed"] = [
        {"pipeline": r["pipeline_type"], "avg_time": r.get("avg_query_time", 0)}
        for r in sorted_by_speed
    ]

    # Rank by answer quality (length as proxy)
    sorted_by_quality = sorted(
        [r for r in all_results if r.get("avg_answer_length", 0) > 0],
        key=lambda x: x.get("avg_answer_length", 0),
        reverse=True
    )
    comparison["rankings"]["by_answer_quality"] = [
        {"pipeline": r["pipeline_type"], "avg_length": r.get("avg_answer_length", 0)}
        for r in sorted_by_quality
    ]

    return comparison


def generate_report(all_results: List[Dict[str, Any]], comparison: Dict[str, Any], output_dir: Path):
    """Generate HTML and JSON reports."""
    logger.info(f"\n📝 Generating reports in {output_dir}...")

    output_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    # Save JSON report
    json_file = output_dir / f"pylate_ragas_demo_{timestamp}.json"
    with open(json_file, "w") as f:
        json.dump({
            "results": all_results,
            "comparison": comparison,
            "note": "This is a DEMO with mock data. For real results, use test_pylate_ragas_comparison.py with IRIS database."
        }, f, indent=2)
    logger.info(f"  ✅ JSON report saved: {json_file}")

    # Generate HTML report
    html_file = output_dir / f"pylate_ragas_demo_{timestamp}.html"
    html_content = f"""
    <html>
    <head>
        <title>PyLate ColBERT RAGAS Demo - {timestamp}</title>
        <style>
            body {{ font-family: Arial, sans-serif; margin: 20px; background-color: #f5f5f5; }}
            .container {{ max-width: 1200px; margin: 0 auto; background: white; padding: 30px; border-radius: 10px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
            h1 {{ color: #2c3e50; border-bottom: 3px solid #3498db; padding-bottom: 10px; }}
            h2 {{ color: #34495e; margin-top: 30px; }}
            .warning {{ background: #fff3cd; border: 1px solid #ffc107; color: #856404; padding: 15px; border-radius: 5px; margin: 20px 0; }}
            table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
            th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
            th {{ background-color: #3498db; color: white; }}
            tr:nth-child(even) {{ background-color: #f2f2f2; }}
            .success {{ color: #27ae60; font-weight: bold; }}
            .highlight {{ background-color: #fff8dc; font-weight: bold; }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>🧪 PyLate ColBERT RAGAS Evaluation - DEMO</h1>
            <p><strong>Generated:</strong> {datetime.now().strftime("%Y-%m-%d %H:%M:%S")}</p>

            <div class="warning">
                <strong>⚠️  DEMO MODE:</strong> This report uses mock data to demonstrate the evaluation flow.
                For real results with actual IRIS database and LLM calls, use <code>test_pylate_ragas_comparison.py</code>.
            </div>

            <h2>📊 Pipeline Rankings</h2>

            <h3>🏆 By Success Rate</h3>
            <table>
                <tr><th>Rank</th><th>Pipeline</th><th>Success Rate</th></tr>
    """

    for i, item in enumerate(comparison["rankings"]["by_success_rate"], 1):
        highlight = 'class="highlight"' if item['pipeline'] == 'pylate_colbert' else ''
        html_content += f"""
            <tr {highlight}>
                <td>{i}</td>
                <td><strong>{item['pipeline']}</strong></td>
                <td class="success">{item['score']*100:.1f}%</td>
            </tr>
        """

    html_content += """
        </table>

        <h3>⚡ By Query Speed (Faster is Better)</h3>
        <table>
            <tr><th>Rank</th><th>Pipeline</th><th>Avg Query Time</th></tr>
    """

    for i, item in enumerate(comparison["rankings"]["by_speed"], 1):
        highlight = 'class="highlight"' if item['pipeline'] == 'pylate_colbert' else ''
        html_content += f"""
            <tr {highlight}>
                <td>{i}</td>
                <td><strong>{item['pipeline']}</strong></td>
                <td>{item['avg_time']:.3f}s</td>
            </tr>
        """

    html_content += """
        </table>

        <h3>📝 By Answer Quality (Length as Proxy)</h3>
        <table>
            <tr><th>Rank</th><th>Pipeline</th><th>Avg Answer Length</th></tr>
    """

    for i, item in enumerate(comparison["rankings"]["by_answer_quality"], 1):
        highlight = 'class="highlight"' if item['pipeline'] == 'pylate_colbert' else ''
        html_content += f"""
            <tr {highlight}>
                <td>{i}</td>
                <td><strong>{item['pipeline']}</strong></td>
                <td>{item['avg_length']:.0f} chars</td>
            </tr>
        """

    html_content += """
        </table>

        <h2>📈 Detailed Results</h2>
        <table>
            <tr>
                <th>Pipeline</th>
                <th>Success Rate</th>
                <th>Avg Answer Length</th>
                <th>Avg Contexts</th>
                <th>Avg Query Time</th>
                <th>Queries/sec</th>
            </tr>
    """

    for result in all_results:
        highlight = 'class="highlight"' if result['pipeline_type'] == 'pylate_colbert' else ''
        html_content += f"""
            <tr {highlight}>
                <td><strong>{result['pipeline_type']}</strong></td>
                <td class="success">{result.get('success_rate', 0)*100:.1f}%</td>
                <td>{result.get('avg_answer_length', 0):.0f} chars</td>
                <td>{result.get('avg_contexts_retrieved', 0):.1f}</td>
                <td>{result.get('avg_query_time', 0):.3f}s</td>
                <td>{result.get('queries_per_second', 0):.2f}</td>
            </tr>
        """

    html_content += """
        </table>

        <h2>🎯 Key Insights</h2>
        <ul>
            <li><strong>PyLate ColBERT</strong> achieves 100% success rate with slightly longer (more detailed) answers</li>
            <li>Query time is competitive at ~1.7s, only 0.5s slower than BasicRAG</li>
            <li>CRAG provides the most detailed answers but is slowest (~2.3s)</li>
            <li>BasicRAG is fastest but with shorter answers</li>
            <li>All pipelines achieve 100% success rate on the test queries</li>
        </ul>

        <p style="margin-top: 40px; padding-top: 20px; border-top: 1px solid #ddd; color: #666;">
            <em>This is a demonstration report with mock data. To run actual evaluations, ensure IRIS database is running and use the full script.</em>
        </p>
        </div>
    </body>
    </html>
    """

    with open(html_file, "w") as f:
        f.write(html_content)
    logger.info(f"  ✅ HTML report saved: {html_file}")

    return json_file, html_file


def main():
    """Main execution function."""
    logger.info("=" * 80)
    logger.info("PyLate ColBERT RAGAS Evaluation - DEMO MODE")
    logger.info("=" * 80)
    logger.info("\n⚠️  Running in DEMO mode with mock data")
    logger.info("For real evaluation, use test_pylate_ragas_comparison.py\n")

    # Pipelines to test
    pipelines_to_test = ["basic", "basic_rerank", "pylate_colbert", "crag"]

    # Generate mock results
    logger.info("🧪 Generating mock evaluation results...\n")
    all_results = []
    for pipeline_type in pipelines_to_test:
        logger.info(f"  Simulating {pipeline_type} pipeline...")
        time.sleep(0.5)  # Simulate processing time
        result = generate_mock_results(pipeline_type, num_queries=5)
        all_results.append(result)
        logger.info(f"    ✅ {result['success_rate']*100:.0f}% success, "
                   f"{result['avg_query_time']:.2f}s avg time, "
                   f"{result['avg_answer_length']:.0f} char answers")

    # Compare results
    comparison = compare_pipelines(all_results)

    # Generate reports
    output_dir = Path("outputs/reports/ragas_evaluations")
    json_file, html_file = generate_report(all_results, comparison, output_dir)

    # Print summary
    logger.info("\n" + "=" * 80)
    logger.info("🎯 DEMO EVALUATION COMPLETE")
    logger.info("=" * 80)

    logger.info(f"\n📊 Rankings by Success Rate:")
    for i, item in enumerate(comparison["rankings"]["by_success_rate"], 1):
        star = "⭐" if item['pipeline'] == "pylate_colbert" else "  "
        logger.info(f"  {star} {i}. {item['pipeline']}: {item['score']*100:.1f}%")

    logger.info(f"\n⚡ Rankings by Speed:")
    for i, item in enumerate(comparison["rankings"]["by_speed"], 1):
        star = "⭐" if item['pipeline'] == "pylate_colbert" else "  "
        logger.info(f"  {star} {i}. {item['pipeline']}: {item['avg_time']:.3f}s")

    logger.info(f"\n📄 Reports Generated:")
    logger.info(f"  JSON: {json_file}")
    logger.info(f"  HTML: {html_file}")

    logger.info(f"\n💡 To open the HTML report:")
    logger.info(f"  open {html_file}")

    logger.info("\n✨ Demo complete! For real evaluation with IRIS:")
    logger.info("  python scripts/test_pylate_ragas_comparison.py")


if __name__ == "__main__":
    main()
