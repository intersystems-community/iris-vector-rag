"""
Performance Metrics Page

Comprehensive analytics and visualizations for RAG pipeline performance.
Provides detailed insights into execution times, accuracy metrics, and system performance.
"""

import streamlit as st
import sys
import json
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from components.header import render_page_header, render_metric_card
from utils.session_state import get_performance_metrics, get_query_history, get_session_summary
from utils.app_config import get_config

def main():
    """Main function for the performance metrics page."""
    
    # Page configuration
    render_page_header(
        "Performance Analytics",
        "Comprehensive performance metrics and analytics for RAG pipeline evaluation",
        "📊"
    )
    
    # Check if we have data
    metrics = get_performance_metrics()
    history = get_query_history()
    
    if not metrics and not history:
        render_empty_state()
        return
    
    # Main analytics dashboard
    render_overview_metrics()
    render_detailed_analytics()

def render_empty_state():
    """Render empty state when no metrics are available."""
    st.markdown("""
    <div style="text-align: center; padding: 3rem; color: #666;">
        <h3>📊 No Metrics Available Yet</h3>
        <p>Performance metrics will appear here after you run some queries</p>
        <p>Try the Single Pipeline or Compare Pipelines pages to generate data</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Show example metrics
    with st.expander("📋 Example Metrics Preview"):
        render_demo_metrics()

def render_overview_metrics():
    """Render high-level overview metrics."""
    st.markdown("## 📈 Performance Overview")
    
    # Get session summary and metrics
    summary = get_session_summary()
    metrics = get_performance_metrics()
    
    # Top-level metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        render_metric_card(
            "Total Queries",
            str(summary.get("total_queries", 0)),
            "",
            "🔍"
        )
    
    with col2:
        avg_time = summary.get("average_execution_time", 0)
        render_metric_card(
            "Avg Response Time",
            f"{avg_time:.2f}s",
            "",
            "⚡"
        )
    
    with col3:
        pipelines_used = summary.get("unique_pipelines_used", 0)
        render_metric_card(
            "Pipelines Used", 
            str(pipelines_used),
            "",
            "🤖"
        )
    
    with col4:
        # Calculate success rate
        success_rate = calculate_success_rate(metrics)
        render_metric_card(
            "Success Rate",
            f"{success_rate:.1%}",
            "",
            "✅"
        )
    
    st.markdown("---")

def render_detailed_analytics():
    """Render detailed analytics with visualizations."""
    
    # Analytics tabs
    tab1, tab2, tab3, tab4 = st.tabs(["⚡ Performance", "🎯 Quality", "📊 Usage", "🔍 Deep Dive"])
    
    with tab1:
        render_performance_analytics()
    
    with tab2:
        render_quality_analytics()
    
    with tab3:
        render_usage_analytics()
    
    with tab4:
        render_deep_dive_analytics()

def render_performance_analytics():
    """Render performance-focused analytics."""
    st.markdown("### ⚡ Performance Analysis")
    
    metrics = get_performance_metrics()
    
    if not metrics:
        st.info("No performance data available yet")
        return
    
    # Convert to DataFrame for analysis
    df = pd.DataFrame(metrics)
    
    # Performance over time
    col1, col2 = st.columns(2)
    
    with col1:
        render_execution_time_chart(df)
    
    with col2:
        render_pipeline_performance_comparison(df)
    
    # Performance distribution
    render_performance_distribution(df)
    
    # Performance insights
    render_performance_insights(df)

def render_quality_analytics():
    """Render quality-focused analytics."""
    st.markdown("### 🎯 Quality Analysis")
    
    history = get_query_history()
    
    if not history:
        st.info("No quality data available yet")
        return
    
    # Quality metrics over time
    render_quality_trends(history)
    
    # Answer length analysis
    render_answer_length_analysis(history)
    
    # Document retrieval analysis
    render_document_analysis(history)

def render_usage_analytics():
    """Render usage pattern analytics."""
    st.markdown("### 📊 Usage Patterns")
    
    summary = get_session_summary()
    history = get_query_history()
    
    col1, col2 = st.columns(2)
    
    with col1:
        render_pipeline_usage_chart(summary)
    
    with col2:
        render_query_frequency_chart(history)
    
    # Usage insights
    render_usage_insights(summary, history)

def render_deep_dive_analytics():
    """Render deep dive analytics."""
    st.markdown("### 🔍 Deep Dive Analysis")
    
    # Advanced analytics options
    analysis_type = st.selectbox(
        "Select Analysis Type:",
        ["Pipeline Comparison", "Query Complexity Analysis", "Performance Trends", "Resource Utilization"],
        key="deep_dive_analysis"
    )
    
    if analysis_type == "Pipeline Comparison":
        render_advanced_pipeline_comparison()
    elif analysis_type == "Query Complexity Analysis":
        render_query_complexity_analysis()
    elif analysis_type == "Performance Trends":
        render_performance_trends()
    elif analysis_type == "Resource Utilization":
        render_resource_utilization()

def render_execution_time_chart(df: pd.DataFrame):
    """Render execution time chart."""
    st.markdown("**Execution Time Trends**")
    
    if df.empty:
        st.info("No data available")
        return
    
    # Create time series chart
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    fig = px.line(
        df, 
        x='timestamp', 
        y='execution_time',
        color='pipeline',
        title="Execution Time Over Time",
        labels={'execution_time': 'Time (seconds)', 'timestamp': 'Time'}
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def render_pipeline_performance_comparison(df: pd.DataFrame):
    """Render pipeline performance comparison."""
    st.markdown("**Pipeline Performance Comparison**")
    
    if df.empty:
        st.info("No data available")
        return
    
    # Group by pipeline and calculate avg execution time
    pipeline_avg = df.groupby('pipeline')['execution_time'].agg(['mean', 'std', 'count']).reset_index()
    
    fig = px.bar(
        pipeline_avg,
        x='pipeline',
        y='mean',
        error_y='std',
        title="Average Execution Time by Pipeline",
        labels={'mean': 'Avg Time (seconds)', 'pipeline': 'Pipeline'}
    )
    
    fig.update_layout(height=400)
    st.plotly_chart(fig, use_container_width=True)

def render_performance_distribution(df: pd.DataFrame):
    """Render performance distribution chart."""
    st.markdown("**Performance Distribution**")
    
    if df.empty:
        st.info("No data available")
        return
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Histogram of execution times
        fig = px.histogram(
            df,
            x='execution_time',
            nbins=20,
            title="Execution Time Distribution",
            labels={'execution_time': 'Time (seconds)', 'count': 'Frequency'}
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        # Box plot by pipeline
        fig = px.box(
            df,
            x='pipeline',
            y='execution_time',
            title="Execution Time by Pipeline",
            labels={'execution_time': 'Time (seconds)', 'pipeline': 'Pipeline'}
        )
        fig.update_layout(height=300)
        st.plotly_chart(fig, use_container_width=True)

def render_performance_insights(df: pd.DataFrame):
    """Render performance insights."""
    st.markdown("**Performance Insights**")
    
    if df.empty:
        st.info("No data available")
        return
    
    insights = []
    
    # Fastest pipeline
    fastest_pipeline = df.loc[df['execution_time'].idxmin(), 'pipeline']
    fastest_time = df['execution_time'].min()
    insights.append(f"🚀 Fastest query: {fastest_pipeline} at {fastest_time:.2f}s")
    
    # Slowest pipeline
    slowest_pipeline = df.loc[df['execution_time'].idxmax(), 'pipeline']
    slowest_time = df['execution_time'].max()
    insights.append(f"🐌 Slowest query: {slowest_pipeline} at {slowest_time:.2f}s")
    
    # Most consistent pipeline
    consistency = df.groupby('pipeline')['execution_time'].std().sort_values()
    if not consistency.empty:
        most_consistent = consistency.index[0]
        insights.append(f"📊 Most consistent: {most_consistent} (low variance)")
    
    for insight in insights:
        st.markdown(f"• {insight}")

def render_quality_trends(history: List[Dict]):
    """Render quality trends over time."""
    st.markdown("**Quality Trends**")
    
    if not history:
        st.info("No quality data available")
        return
    
    # Extract quality metrics from history
    quality_data = []
    for entry in history:
        results = entry.get('results', {})
        quality_data.append({
            'timestamp': entry.get('timestamp'),
            'pipeline': entry.get('pipeline'),
            'answer_length': results.get('answer_length', 0),
            'num_documents': results.get('num_documents', 0)
        })
    
    if quality_data:
        df = pd.DataFrame(quality_data)
        df['timestamp'] = pd.to_datetime(df['timestamp'])
        
        # Answer length trend
        fig = px.line(
            df,
            x='timestamp',
            y='answer_length',
            color='pipeline',
            title="Answer Length Trend",
            labels={'answer_length': 'Answer Length (chars)', 'timestamp': 'Time'}
        )
        st.plotly_chart(fig, use_container_width=True)

def render_answer_length_analysis(history: List[Dict]):
    """Render answer length analysis."""
    st.markdown("**Answer Length Analysis**")
    
    # Extract answer lengths
    lengths = []
    for entry in history:
        results = entry.get('results', {})
        if results.get('answer_length'):
            lengths.append({
                'pipeline': entry.get('pipeline'),
                'length': results.get('answer_length'),
                'quality_estimate': min(1.0, results.get('answer_length', 0) / 500)  # Simple quality estimate
            })
    
    if lengths:
        df = pd.DataFrame(lengths)
        
        col1, col2 = st.columns(2)
        
        with col1:
            avg_length = df.groupby('pipeline')['length'].mean().reset_index()
            fig = px.bar(avg_length, x='pipeline', y='length', title="Average Answer Length by Pipeline")
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            fig = px.scatter(df, x='length', y='quality_estimate', color='pipeline', 
                           title="Answer Length vs Quality Estimate")
            st.plotly_chart(fig, use_container_width=True)

def render_document_analysis(history: List[Dict]):
    """Render document retrieval analysis."""
    st.markdown("**Document Retrieval Analysis**")
    
    # Extract document counts
    doc_counts = []
    for entry in history:
        results = entry.get('results', {})
        if results.get('num_documents'):
            doc_counts.append({
                'pipeline': entry.get('pipeline'),
                'num_documents': results.get('num_documents'),
                'timestamp': entry.get('timestamp')
            })
    
    if doc_counts:
        df = pd.DataFrame(doc_counts)
        
        avg_docs = df.groupby('pipeline')['num_documents'].mean().reset_index()
        fig = px.bar(avg_docs, x='pipeline', y='num_documents', 
                    title="Average Documents Retrieved by Pipeline")
        st.plotly_chart(fig, use_container_width=True)

def render_pipeline_usage_chart(summary: Dict):
    """Render pipeline usage chart."""
    st.markdown("**Pipeline Usage Distribution**")
    
    usage = summary.get('pipeline_usage', {})
    
    if not usage:
        st.info("No usage data available")
        return
    
    df = pd.DataFrame(list(usage.items()), columns=['Pipeline', 'Count'])
    
    fig = px.pie(df, values='Count', names='Pipeline', title="Pipeline Usage Distribution")
    st.plotly_chart(fig, use_container_width=True)

def render_query_frequency_chart(history: List[Dict]):
    """Render query frequency over time."""
    st.markdown("**Query Frequency**")
    
    if not history:
        st.info("No query data available")
        return
    
    # Group queries by hour
    query_times = [datetime.fromisoformat(entry['timestamp'].replace('Z', '+00:00')) for entry in history if entry.get('timestamp')]
    
    if query_times:
        df = pd.DataFrame({'timestamp': query_times})
        df['hour'] = df['timestamp'].dt.hour
        
        hourly_counts = df['hour'].value_counts().sort_index()
        
        fig = px.bar(x=hourly_counts.index, y=hourly_counts.values, 
                    title="Query Frequency by Hour of Day",
                    labels={'x': 'Hour', 'y': 'Number of Queries'})
        st.plotly_chart(fig, use_container_width=True)

def render_usage_insights(summary: Dict, history: List[Dict]):
    """Render usage insights."""
    st.markdown("**Usage Insights**")
    
    insights = []
    
    # Most active pipeline
    usage = summary.get('pipeline_usage', {})
    if usage:
        most_used = max(usage.items(), key=lambda x: x[1])
        insights.append(f"🔥 Most used pipeline: {most_used[0]} ({most_used[1]} queries)")
    
    # Query patterns
    if history:
        recent_queries = len([h for h in history if 
                            datetime.fromisoformat(h['timestamp'].replace('Z', '+00:00')) > 
                            datetime.now() - timedelta(hours=1)])
        insights.append(f"📈 Queries in last hour: {recent_queries}")
    
    for insight in insights:
        st.markdown(f"• {insight}")

def render_advanced_pipeline_comparison():
    """Render advanced pipeline comparison."""
    st.markdown("**Advanced Pipeline Comparison**")
    
    metrics = get_performance_metrics()
    
    if not metrics:
        st.info("No data available for comparison")
        return
    
    df = pd.DataFrame(metrics)
    
    # Multi-dimensional comparison
    comparison_metrics = df.groupby('pipeline').agg({
        'execution_time': ['mean', 'std'],
        'success': 'mean'
    }).round(3)
    
    st.dataframe(comparison_metrics)

def render_query_complexity_analysis():
    """Render query complexity analysis."""
    st.markdown("**Query Complexity Analysis**")
    st.info("Query complexity analysis would examine query characteristics and performance correlation")

def render_performance_trends():
    """Render performance trends analysis."""
    st.markdown("**Performance Trends**")
    st.info("Performance trends would show how pipeline performance changes over time")

def render_resource_utilization():
    """Render resource utilization analysis."""
    st.markdown("**Resource Utilization**")
    st.info("Resource utilization would show memory, CPU, and other system metrics")

def render_demo_metrics():
    """Render demo metrics for empty state."""
    st.markdown("**Example Performance Dashboard:**")
    
    # Mock data for demonstration
    demo_data = {
        "Total Queries": "156",
        "Avg Response Time": "2.3s", 
        "Success Rate": "94.2%",
        "Most Used Pipeline": "BasicRAG"
    }
    
    cols = st.columns(len(demo_data))
    for i, (metric, value) in enumerate(demo_data.items()):
        with cols[i]:
            st.metric(metric, value)
    
    st.markdown("*These are example metrics. Real data will appear after running queries.*")

# Helper functions
def calculate_success_rate(metrics: List[Dict]) -> float:
    """Calculate overall success rate."""
    if not metrics:
        return 0.0
    
    successful = sum(1 for m in metrics if m.get('success', False))
    return successful / len(metrics)

if __name__ == "__main__":
    main()