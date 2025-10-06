"""
RAG Evaluation Visualization Engine
Creates beautiful interactive HTML reports with spider charts and performance dashboards
"""

import json
import logging
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.offline as pyo
from plotly.subplots import make_subplots

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResults:
    """Container for evaluation results"""

    pipeline_name: str
    ragas_metrics: Dict[str, float]
    performance_metrics: Dict[str, float]
    timing_data: Dict[str, float]
    metadata: Dict[str, Any]


class SpiderChartGenerator:
    """Generates beautiful spider/radar charts for RAGAS metrics comparison"""

    RAGAS_METRICS = [
        "faithfulness",
        "answer_relevancy",
        "context_precision",
        "context_recall",
        "answer_correctness",
        "answer_similarity",
        "context_utilization",
    ]

    METRIC_DESCRIPTIONS = {
        "faithfulness": "Factual accuracy of generated answers",
        "answer_relevancy": "Relevance of answer to the question",
        "context_precision": "Quality of retrieved context",
        "context_recall": "Completeness of context retrieval",
        "answer_correctness": "Overall correctness of answers",
        "answer_similarity": "Semantic similarity to ground truth",
        "context_utilization": "Effective use of provided context",
    }

    def __init__(self):
        self.colors = {
            "GraphRAG": "#FF6B6B",
            "BasicRAG": "#4ECDC4",
            "CRAG": "#45B7D1",
            "BasicRAGReranking": "#96CEB4",
            "ColBERT": "#9B59B6",
            "HyDE": "#F39C12",
        }

    def create_spider_chart(self, results: List[EvaluationResults]) -> go.Figure:
        """Create interactive spider chart comparing pipelines"""

        fig = go.Figure()

        for result in results:
            # Extract RAGAS scores
            scores = []
            for metric in self.RAGAS_METRICS:
                score = result.ragas_metrics.get(metric, 0.0)
                scores.append(score)

            # Add trace for this pipeline
            fig.add_trace(
                go.Scatterpolar(
                    r=scores,
                    theta=self.RAGAS_METRICS,
                    fill="toself",
                    name=result.pipeline_name,
                    line_color=self.colors.get(result.pipeline_name, "#333333"),
                    fillcolor=self.colors.get(result.pipeline_name, "#333333"),
                    opacity=0.6,
                    hovertemplate=(
                        "<b>%{fullData.name}</b><br>"
                        + "Metric: %{theta}<br>"
                        + "Score: %{r:.3f}<br>"
                        + "<extra></extra>"
                    ),
                )
            )

        # Update layout
        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1],
                    tickmode="array",
                    tickvals=[0.2, 0.4, 0.6, 0.8, 1.0],
                    ticktext=["0.2", "0.4", "0.6", "0.8", "1.0"],
                ),
                angularaxis=dict(tickfont_size=12, rotation=90, direction="clockwise"),
            ),
            showlegend=True,
            title={
                "text": "RAG Pipeline Performance Comparison - RAGAS Metrics",
                "x": 0.5,
                "xanchor": "center",
                "font": {"size": 20},
            },
            font_size=12,
            width=800,
            height=600,
        )

        return fig


class PerformanceDashboard:
    """Creates comprehensive performance dashboards"""

    def __init__(self):
        self.colors = {
            "GraphRAG": "#FF6B6B",
            "BasicRAG": "#4ECDC4",
            "CRAG": "#45B7D1",
            "BasicRAGReranking": "#96CEB4",
        }

    def create_performance_dashboard(
        self, results: List[EvaluationResults]
    ) -> go.Figure:
        """Create multi-panel performance dashboard"""

        # Create subplots
        fig = make_subplots(
            rows=2,
            cols=2,
            subplot_titles=(
                "Response Time Distribution",
                "Throughput Comparison",
                "Memory Usage",
                "Success Rate",
            ),
            specs=[
                [{"type": "histogram"}, {"type": "bar"}],
                [{"type": "scatter"}, {"type": "pie"}],
            ],
        )

        # Panel 1: Response Time Distribution
        for result in results:
            response_times = result.timing_data.get("response_times", [])
            if response_times:
                fig.add_trace(
                    go.Histogram(
                        x=response_times,
                        name=f"{result.pipeline_name} Response Time",
                        opacity=0.7,
                        marker_color=self.colors.get(result.pipeline_name, "#333333"),
                    ),
                    row=1,
                    col=1,
                )

        # Panel 2: Throughput Comparison
        pipeline_names = [r.pipeline_name for r in results]
        throughputs = [r.performance_metrics.get("throughput", 0) for r in results]

        fig.add_trace(
            go.Bar(
                x=pipeline_names,
                y=throughputs,
                name="Throughput (queries/sec)",
                marker_color=[
                    self.colors.get(name, "#333333") for name in pipeline_names
                ],
            ),
            row=1,
            col=2,
        )

        # Panel 3: Memory Usage
        for result in results:
            memory_usage = result.performance_metrics.get("memory_usage_mb", [])
            if isinstance(memory_usage, list) and memory_usage:
                fig.add_trace(
                    go.Scatter(
                        y=memory_usage,
                        mode="lines+markers",
                        name=f"{result.pipeline_name} Memory",
                        line_color=self.colors.get(result.pipeline_name, "#333333"),
                    ),
                    row=2,
                    col=1,
                )

        # Panel 4: Success Rate Pie Chart
        avg_success_rate = np.mean(
            [r.performance_metrics.get("success_rate", 0) for r in results]
        )
        fig.add_trace(
            go.Pie(
                labels=["Success", "Failure"],
                values=[avg_success_rate, 1 - avg_success_rate],
                marker_colors=["#2ECC71", "#E74C3C"],
            ),
            row=2,
            col=2,
        )

        # Update layout
        fig.update_layout(
            height=800, title_text="RAG Pipeline Performance Dashboard", showlegend=True
        )

        return fig


class HTMLReportGenerator:
    """Generates comprehensive HTML evaluation reports"""

    def __init__(self):
        self.spider_generator = SpiderChartGenerator()
        self.dashboard_generator = PerformanceDashboard()

    def generate_report(
        self, results: List[EvaluationResults], output_path: str
    ) -> str:
        """Generate complete HTML evaluation report"""

        # Generate charts
        spider_chart = self.spider_generator.create_spider_chart(results)
        dashboard = self.dashboard_generator.create_performance_dashboard(results)

        # Convert to HTML
        spider_html = pyo.plot(spider_chart, output_type="div", include_plotlyjs=False)
        dashboard_html = pyo.plot(dashboard, output_type="div", include_plotlyjs=False)

        # Create comprehensive HTML report
        html_content = self._generate_html_template(
            results, spider_html, dashboard_html
        )

        # Write to file
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)

        with open(output_file, "w", encoding="utf-8") as f:
            f.write(html_content)

        logger.info(f"HTML report generated: {output_file}")
        return str(output_file)

    def _generate_html_template(
        self, results: List[EvaluationResults], spider_html: str, dashboard_html: str
    ) -> str:
        """Generate HTML template with embedded charts"""

        # Calculate summary statistics
        summary_stats = self._calculate_summary_stats(results)

        # Generate results table
        results_table = self._generate_results_table(results)

        html_template = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>RAG Pipeline Evaluation Report</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <style>
        body {{
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background-color: #f8f9fa;
        }}
        
        .header {{
            text-align: center;
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 2rem;
            border-radius: 10px;
            margin-bottom: 2rem;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        }}
        
        .section {{
            background: white;
            padding: 2rem;
            margin-bottom: 2rem;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }}
        
        .metric-card {{
            background: #f8f9fa;
            border-left: 4px solid #007bff;
            padding: 1rem;
            margin: 1rem 0;
            border-radius: 4px;
        }}
        
        .table {{
            width: 100%;
            border-collapse: collapse;
            margin: 1rem 0;
        }}
        
        .table th, .table td {{
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #dee2e6;
        }}
        
        .table th {{
            background-color: #e9ecef;
            font-weight: 600;
        }}
        
        .pipeline-name {{
            font-weight: bold;
            color: #495057;
        }}
        
        .score {{
            padding: 4px 8px;
            border-radius: 4px;
            font-weight: bold;
            color: white;
        }}
        
        .score-high {{ background-color: #28a745; }}
        .score-medium {{ background-color: #ffc107; color: #212529; }}
        .score-low {{ background-color: #dc3545; }}
        
        .chart-container {{
            margin: 2rem 0;
            text-align: center;
        }}
        
        .timestamp {{
            color: #6c757d;
            font-size: 0.9rem;
        }}
    </style>
</head>
<body>
    <div class="header">
        <h1>🚀 RAG Pipeline Evaluation Report</h1>
        <p>Comprehensive comparison of GraphRAG vs Traditional RAG approaches</p>
        <p class="timestamp">Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}</p>
    </div>
    
    <div class="section">
        <h2>📊 Executive Summary</h2>
        {summary_stats}
    </div>
    
    <div class="section">
        <h2>🕸️ RAGAS Metrics Comparison</h2>
        <p>Spider chart showing comprehensive evaluation across all 7 RAGAS metrics:</p>
        <div class="chart-container">
            {spider_html}
        </div>
    </div>
    
    <div class="section">
        <h2>⚡ Performance Dashboard</h2>
        <p>Multi-dimensional performance analysis including timing, throughput, and resource usage:</p>
        <div class="chart-container">
            {dashboard_html}
        </div>
    </div>
    
    <div class="section">
        <h2>📋 Detailed Results</h2>
        {results_table}
    </div>
    
    <div class="section">
        <h2>🔍 Key Insights</h2>
        <div class="metric-card">
            <h4>Best Overall Performance</h4>
            <p>Based on RAGAS metrics and performance data analysis.</p>
        </div>
        
        <div class="metric-card">
            <h4>GraphRAG Advantages</h4>
            <p>Knowledge graph-based retrieval shows improvements in context precision and answer correctness.</p>
        </div>
        
        <div class="metric-card">
            <h4>Performance Trade-offs</h4>
            <p>Analysis of computational cost vs. quality improvements across different pipeline approaches.</p>
        </div>
    </div>
</body>
</html>
"""
        return html_template

    def _calculate_summary_stats(self, results: List[EvaluationResults]) -> str:
        """Calculate and format summary statistics"""

        stats_html = ""

        for result in results:
            avg_ragas = np.mean(list(result.ragas_metrics.values()))
            throughput = result.performance_metrics.get("throughput", 0)

            stats_html += f"""
            <div class="metric-card">
                <h4>{result.pipeline_name}</h4>
                <p><strong>Average RAGAS Score:</strong> {avg_ragas:.3f}</p>
                <p><strong>Throughput:</strong> {throughput:.2f} queries/sec</p>
            </div>
            """

        return stats_html

    def _generate_results_table(self, results: List[EvaluationResults]) -> str:
        """Generate detailed results table"""

        table_html = """
        <table class="table">
            <thead>
                <tr>
                    <th>Pipeline</th>
                    <th>Faithfulness</th>
                    <th>Answer Relevancy</th>
                    <th>Context Precision</th>
                    <th>Context Recall</th>
                    <th>Answer Correctness</th>
                    <th>Throughput</th>
                </tr>
            </thead>
            <tbody>
        """

        for result in results:
            faithfulness = result.ragas_metrics.get("faithfulness", 0)
            answer_relevancy = result.ragas_metrics.get("answer_relevancy", 0)
            context_precision = result.ragas_metrics.get("context_precision", 0)
            context_recall = result.ragas_metrics.get("context_recall", 0)
            answer_correctness = result.ragas_metrics.get("answer_correctness", 0)
            throughput = result.performance_metrics.get("throughput", 0)

            table_html += f"""
                <tr>
                    <td class="pipeline-name">{result.pipeline_name}</td>
                    <td><span class="score {self._get_score_class(faithfulness)}">{faithfulness:.3f}</span></td>
                    <td><span class="score {self._get_score_class(answer_relevancy)}">{answer_relevancy:.3f}</span></td>
                    <td><span class="score {self._get_score_class(context_precision)}">{context_precision:.3f}</span></td>
                    <td><span class="score {self._get_score_class(context_recall)}">{context_recall:.3f}</span></td>
                    <td><span class="score {self._get_score_class(answer_correctness)}">{answer_correctness:.3f}</span></td>
                    <td>{throughput:.2f}</td>
                </tr>
            """

        table_html += """
            </tbody>
        </table>
        """

        return table_html

    def _get_score_class(self, score: float) -> str:
        """Get CSS class based on score value"""
        if score >= 0.8:
            return "score-high"
        elif score >= 0.6:
            return "score-medium"
        else:
            return "score-low"


class VisualizationEngine:
    """Main visualization engine coordinator"""

    def __init__(self):
        self.html_generator = HTMLReportGenerator()
        self.spider_generator = SpiderChartGenerator()
        self.dashboard_generator = PerformanceDashboard()

    def create_evaluation_report(
        self,
        results: List[EvaluationResults],
        output_dir: str = "outputs/visualizations",
    ) -> str:
        """Create comprehensive evaluation report"""

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_path = f"{output_dir}/evaluation_report_{timestamp}.html"

        report_path = self.html_generator.generate_report(results, output_path)

        logger.info(f"✅ Evaluation report created: {report_path}")
        return report_path

    def create_demo_report(self) -> str:
        """Create demo report with sample data"""

        # Sample data for demonstration
        demo_results = [
            EvaluationResults(
                pipeline_name="GraphRAG",
                ragas_metrics={
                    "faithfulness": 0.85,
                    "answer_relevancy": 0.82,
                    "context_precision": 0.88,
                    "context_recall": 0.79,
                    "answer_correctness": 0.84,
                    "answer_similarity": 0.81,
                    "context_utilization": 0.86,
                },
                performance_metrics={
                    "throughput": 12.5,
                    "success_rate": 0.95,
                    "memory_usage_mb": [450, 480, 520, 490, 510],
                },
                timing_data={"response_times": [2.1, 2.3, 1.9, 2.5, 2.0, 2.2, 1.8]},
                metadata={},
            ),
            EvaluationResults(
                pipeline_name="BasicRAG",
                ragas_metrics={
                    "faithfulness": 0.76,
                    "answer_relevancy": 0.78,
                    "context_precision": 0.72,
                    "context_recall": 0.85,
                    "answer_correctness": 0.75,
                    "answer_similarity": 0.79,
                    "context_utilization": 0.74,
                },
                performance_metrics={
                    "throughput": 18.2,
                    "success_rate": 0.92,
                    "memory_usage_mb": [320, 350, 380, 360, 340],
                },
                timing_data={"response_times": [1.2, 1.4, 1.1, 1.6, 1.3, 1.5, 1.0]},
                metadata={},
            ),
            EvaluationResults(
                pipeline_name="CRAG",
                ragas_metrics={
                    "faithfulness": 0.80,
                    "answer_relevancy": 0.83,
                    "context_precision": 0.81,
                    "context_recall": 0.77,
                    "answer_correctness": 0.82,
                    "answer_similarity": 0.84,
                    "context_utilization": 0.80,
                },
                performance_metrics={
                    "throughput": 15.8,
                    "success_rate": 0.94,
                    "memory_usage_mb": [390, 420, 450, 430, 410],
                },
                timing_data={"response_times": [1.6, 1.8, 1.5, 2.0, 1.7, 1.9, 1.4]},
                metadata={},
            ),
        ]

        return self.create_evaluation_report(demo_results, "outputs/visualizations")


# Example usage
if __name__ == "__main__":
    viz_engine = VisualizationEngine()
    demo_path = viz_engine.create_demo_report()
    print(f"Demo report created: {demo_path}")
