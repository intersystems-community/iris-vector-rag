"""
Compare Pipelines Page

Side-by-side comparison of multiple RAG pipelines with the same query.
Provides detailed analysis of differences in performance, quality, and results.
"""

import streamlit as st
import sys
import asyncio
import threading
from pathlib import Path
from typing import Dict, List, Any
import time

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from components.header import render_page_header, render_loading_spinner
from components.query_input import render_query_input
from components.sidebar import render_pipeline_selector
from components.results_display import render_query_result
from utils.pipeline_integration import execute_pipeline_query
from utils.session_state import add_to_query_history, add_performance_metric
from utils.app_config import get_config

def main():
    """Main function for the compare pipelines page."""
    
    # Page configuration
    render_page_header(
        "Pipeline Comparison",
        "Compare multiple RAG pipelines side-by-side with the same query",
        "⚖️"
    )
    
    # Initialize session state
    if "comparison_results" not in st.session_state:
        st.session_state.comparison_results = {}
    
    # Main layout
    render_comparison_setup()
    render_comparison_results()

def render_comparison_setup():
    """Render the comparison setup interface."""
    st.markdown("## 🔧 Comparison Setup")
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        render_pipeline_selection()
        render_comparison_settings()
    
    with col2:
        render_query_interface()

def render_pipeline_selection():
    """Render pipeline selection for comparison."""
    st.markdown("### 🤖 Select Pipelines")
    
    # Multi-select for pipelines
    available_pipelines = ["BasicRAG", "BasicRerank", "CRAG", "GraphRAG"]
    
    selected_pipelines = st.multiselect(
        "Choose pipelines to compare:",
        available_pipelines,
        default=["BasicRAG", "CRAG"],
        help="Select 2-4 pipelines for comparison",
        key="compare_pipelines"
    )
    
    # Validation
    if len(selected_pipelines) < 2:
        st.warning("⚠️ Please select at least 2 pipelines for comparison")
    elif len(selected_pipelines) > 4:
        st.warning("⚠️ Maximum 4 pipelines can be compared at once")
        selected_pipelines = selected_pipelines[:4]
    
    st.session_state.selected_compare_pipelines = selected_pipelines
    
    # Pipeline status indicators
    if selected_pipelines:
        st.markdown("**Pipeline Status:**")
        for pipeline in selected_pipelines:
            # Mock status for now
            status = "🟢 Ready" if pipeline in ["BasicRAG", "CRAG"] else "🟡 Initializing"
            st.markdown(f"• {pipeline}: {status}")

def render_comparison_settings():
    """Render comparison-specific settings."""
    st.markdown("### ⚙️ Comparison Settings")
    
    # Execution mode
    execution_mode = st.radio(
        "Execution Mode:",
        ["Sequential", "Parallel"],
        help="Sequential: Run one after another. Parallel: Run simultaneously for faster comparison",
        key="execution_mode"
    )
    
    # Configuration options
    use_same_config = st.checkbox(
        "Use same configuration for all pipelines",
        value=True,
        help="Apply identical settings to all pipelines for fair comparison"
    )
    
    if use_same_config:
        # Shared configuration
        col1, col2 = st.columns(2)
        with col1:
            top_k = st.slider("Top K Results:", 1, 20, 5, key="shared_top_k")
        with col2:
            temperature = st.slider("Temperature:", 0.0, 1.0, 0.7, step=0.1, key="shared_temp")
        
        shared_config = {"top_k": top_k, "temperature": temperature}
        st.session_state.comparison_config = {
            "shared": shared_config,
            "execution_mode": execution_mode
        }
    else:
        st.info("Individual pipeline configurations can be set in the Configuration page")

def render_query_interface():
    """Render the query interface for comparison."""
    st.markdown("### 💬 Query for Comparison")
    
    # Query input
    query_data = render_query_input(key="compare_query")
    query = query_data.get("query", "")
    
    # Query suggestions for comparison
    with st.expander("💡 Good Queries for Comparison"):
        st.markdown("""
        **Recommended query types for meaningful comparison:**
        
        • **Complex multi-faceted questions** - Show different pipeline strengths
        • **Factual queries** - Compare accuracy and source quality  
        • **Research questions** - Highlight GraphRAG's relationship analysis
        • **Medical scenarios** - Test domain-specific performance
        
        **Example queries:**
        • "What are the mechanisms and treatment options for autoimmune diseases?"
        • "How do genetic factors influence cancer development and therapy selection?"
        • "Compare the effectiveness of different diabetes management strategies"
        """)
    
    # Execution controls
    if query and len(query.strip()) > 0:
        selected_pipelines = st.session_state.get("selected_compare_pipelines", [])
        
        if len(selected_pipelines) >= 2:
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("🚀 Run Comparison", type="primary", key="run_comparison"):
                    execute_comparison(query, selected_pipelines)
            
            with col2:
                if st.button("📊 Quick Compare", key="quick_compare"):
                    execute_quick_comparison(query, selected_pipelines)
            
            with col3:
                if st.button("🧹 Clear Results", key="clear_comparison"):
                    st.session_state.comparison_results = {}
                    st.rerun()
        else:
            st.info("💡 Select at least 2 pipelines above to enable comparison")
    else:
        st.info("💡 Enter a query above to start the comparison")

def render_comparison_results():
    """Render the comparison results."""
    if not st.session_state.comparison_results:
        render_empty_comparison()
        return
    
    st.markdown("## 📊 Comparison Results")
    
    # Results overview
    render_comparison_overview()
    
    # Detailed comparison tabs
    tab1, tab2, tab3, tab4 = st.tabs(["📋 Side-by-Side", "🎯 Answer Quality", "📄 Documents", "⚡ Performance"])
    
    with tab1:
        render_side_by_side_comparison()
    
    with tab2:
        render_answer_quality_comparison()
    
    with tab3:
        render_document_comparison()
    
    with tab4:
        render_performance_comparison()

def render_empty_comparison():
    """Render empty comparison state."""
    st.markdown("""
    <div style="text-align: center; padding: 3rem; color: #666;">
        <h3>⚖️ Ready to Compare!</h3>
        <p>Select your pipelines and execute a query to see detailed comparisons</p>
        <p>Perfect for understanding the strengths and weaknesses of different RAG approaches</p>
    </div>
    """, unsafe_allow_html=True)

def render_comparison_overview():
    """Render high-level comparison overview."""
    results = st.session_state.comparison_results
    query = results.get("query", "")
    pipeline_results = results.get("results", {})
    
    st.markdown(f"**Query:** *{query}*")
    st.markdown("---")
    
    # Quick metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        total_pipelines = len(pipeline_results)
        st.metric("Pipelines Tested", total_pipelines)
    
    with col2:
        successful_runs = sum(1 for r in pipeline_results.values() if r.success)
        st.metric("Successful Runs", successful_runs)
    
    with col3:
        if pipeline_results:
            avg_time = sum(r.execution_time for r in pipeline_results.values()) / len(pipeline_results)
            st.metric("Avg Execution Time", f"{avg_time:.2f}s")
    
    with col4:
        if pipeline_results:
            total_docs = sum(len(r.retrieved_documents) for r in pipeline_results.values())
            st.metric("Total Documents", total_docs)
    
    # Winner indicators
    render_winner_indicators(pipeline_results)

def render_winner_indicators(pipeline_results: Dict[str, Any]):
    """Render winner indicators for different categories."""
    if not pipeline_results:
        return
    
    st.markdown("### 🏆 Category Winners")
    
    # Calculate winners
    winners = calculate_category_winners(pipeline_results)
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        fastest = winners.get("fastest", "N/A")
        st.markdown(f"**⚡ Fastest:**\n{fastest}")
    
    with col2:
        most_docs = winners.get("most_documents", "N/A")
        st.markdown(f"**📄 Most Documents:**\n{most_docs}")
    
    with col3:
        longest_answer = winners.get("longest_answer", "N/A")
        st.markdown(f"**📝 Longest Answer:**\n{longest_answer}")
    
    with col4:
        highest_confidence = winners.get("highest_confidence", "N/A")
        st.markdown(f"**🎯 Highest Confidence:**\n{highest_confidence}")

def render_side_by_side_comparison():
    """Render side-by-side comparison of results."""
    results = st.session_state.comparison_results
    pipeline_results = results.get("results", {})
    
    if not pipeline_results:
        st.info("No results to compare")
        return
    
    # Create columns for each pipeline
    num_pipelines = len(pipeline_results)
    cols = st.columns(num_pipelines)
    
    for i, (pipeline, result) in enumerate(pipeline_results.items()):
        with cols[i]:
            st.markdown(f"### {get_pipeline_icon(pipeline)} {pipeline}")
            
            if result.success:
                # Quick metrics
                st.metric("Time", f"{result.execution_time:.2f}s")
                st.metric("Documents", len(result.retrieved_documents))
                
                # Answer preview
                st.markdown("**Answer:**")
                answer_preview = result.answer[:200] + "..." if len(result.answer) > 200 else result.answer
                st.markdown(f"*{answer_preview}*")
                
                # Quality indicators
                quality_score = estimate_answer_quality(result.answer)
                st.progress(quality_score)
                st.caption(f"Quality Score: {quality_score:.1%}")
                
                # Show full result button
                if st.button(f"📄 Full Details", key=f"details_{pipeline}"):
                    with st.expander(f"{pipeline} Full Result", expanded=True):
                        render_query_result(result)
            else:
                st.error(f"❌ Failed: {result.error}")

def render_answer_quality_comparison():
    """Render detailed answer quality analysis."""
    results = st.session_state.comparison_results
    pipeline_results = results.get("results", {})
    
    st.markdown("### 📊 Answer Quality Analysis")
    
    # Quality metrics table
    quality_data = []
    for pipeline, result in pipeline_results.items():
        if result.success:
            answer = result.answer
            quality_metrics = {
                "Pipeline": f"{get_pipeline_icon(pipeline)} {pipeline}",
                "Word Count": len(answer.split()),
                "Character Count": len(answer),
                "Sentences": answer.count('.') + answer.count('!') + answer.count('?'),
                "Quality Score": f"{estimate_answer_quality(answer):.1%}",
                "Has Citations": "✅" if any(ref in answer.lower() for ref in ['study', 'research', 'source']) else "❌"
            }
            quality_data.append(quality_metrics)
    
    if quality_data:
        st.table(quality_data)
    
    # Answer similarity analysis
    st.markdown("### 🔍 Answer Content Analysis")
    
    successful_results = {p: r for p, r in pipeline_results.items() if r.success}
    
    if len(successful_results) >= 2:
        # Simple similarity analysis
        render_answer_similarity_analysis(successful_results)
    
    # Key differences highlighting
    render_key_differences(successful_results)

def render_document_comparison():
    """Render document retrieval comparison."""
    results = st.session_state.comparison_results
    pipeline_results = results.get("results", {})
    
    st.markdown("### 📄 Document Retrieval Comparison")
    
    # Document overlap analysis
    render_document_overlap_analysis(pipeline_results)
    
    # Source diversity
    render_source_diversity_analysis(pipeline_results)
    
    # Relevance score comparison
    render_relevance_comparison(pipeline_results)

def render_performance_comparison():
    """Render performance metrics comparison."""
    results = st.session_state.comparison_results
    pipeline_results = results.get("results", {})
    
    st.markdown("### ⚡ Performance Metrics")
    
    # Performance chart
    render_performance_chart(pipeline_results)
    
    # Resource usage comparison
    render_resource_usage_comparison(pipeline_results)
    
    # Scalability insights
    render_scalability_insights(pipeline_results)

def execute_comparison(query: str, pipelines: List[str]):
    """Execute comparison across selected pipelines."""
    config = st.session_state.get("comparison_config", {})
    execution_mode = config.get("execution_mode", "Sequential")
    shared_config = config.get("shared", {})
    
    with st.spinner("Executing comparison..."):
        results = {}
        
        if execution_mode == "Sequential":
            # Execute one by one
            for pipeline in pipelines:
                st.info(f"Running {pipeline}...")
                result = execute_pipeline_query(pipeline, query, shared_config)
                results[pipeline] = result
                
                # Add to individual history
                add_to_query_history(query, pipeline, {
                    "success": result.success,
                    "execution_time": result.execution_time,
                    "comparison_mode": True
                })
        else:
            # Parallel execution (simplified - would need proper async implementation)
            st.info("Running all pipelines in parallel...")
            for pipeline in pipelines:
                result = execute_pipeline_query(pipeline, query, shared_config)
                results[pipeline] = result
        
        # Store comparison results
        st.session_state.comparison_results = {
            "query": query,
            "results": results,
            "timestamp": time.time(),
            "config": config
        }
        
        successful_count = sum(1 for r in results.values() if r.success)
        st.success(f"✅ Comparison complete! {successful_count}/{len(pipelines)} pipelines succeeded")

def execute_quick_comparison(query: str, pipelines: List[str]):
    """Execute a quick comparison with minimal configuration."""
    quick_config = {"top_k": 3, "temperature": 0.7}
    
    with st.spinner("Quick comparison in progress..."):
        results = {}
        
        for pipeline in pipelines[:2]:  # Limit to 2 for quick compare
            result = execute_pipeline_query(pipeline, query, quick_config)
            results[pipeline] = result
        
        st.session_state.comparison_results = {
            "query": query,
            "results": results,
            "timestamp": time.time(),
            "config": {"shared": quick_config, "mode": "quick"}
        }
        
        st.success("⚡ Quick comparison complete!")

# Helper functions
def get_pipeline_icon(pipeline: str) -> str:
    """Get icon for pipeline."""
    icons = {
        "BasicRAG": "📝",
        "BasicRerank": "🎯", 
        "CRAG": "🔄",
        "GraphRAG": "🕸️"
    }
    return icons.get(pipeline, "❓")

def estimate_answer_quality(answer: str) -> float:
    """Estimate answer quality (simplified)."""
    if not answer:
        return 0.0
    
    score = 0.5  # Base score
    
    # Length scoring
    if 50 <= len(answer) <= 1000:
        score += 0.2
    
    # Structure scoring
    if "." in answer:
        score += 0.1
    
    # Content indicators
    quality_words = ["research", "study", "evidence", "analysis", "treatment", "diagnosis"]
    for word in quality_words:
        if word.lower() in answer.lower():
            score += 0.03
    
    return min(1.0, score)

def calculate_category_winners(pipeline_results: Dict[str, Any]) -> Dict[str, str]:
    """Calculate winners in different categories."""
    winners = {}
    
    successful_results = {p: r for p, r in pipeline_results.items() if r.success}
    
    if successful_results:
        # Fastest
        fastest = min(successful_results.items(), key=lambda x: x[1].execution_time)
        winners["fastest"] = fastest[0]
        
        # Most documents
        most_docs = max(successful_results.items(), key=lambda x: len(x[1].retrieved_documents))
        winners["most_documents"] = most_docs[0]
        
        # Longest answer
        longest = max(successful_results.items(), key=lambda x: len(x[1].answer))
        winners["longest_answer"] = longest[0]
        
        # Highest confidence (mock)
        winners["highest_confidence"] = list(successful_results.keys())[0]
    
    return winners

def render_answer_similarity_analysis(results: Dict[str, Any]):
    """Render answer similarity analysis."""
    st.markdown("**Answer Similarity:**")
    
    pipelines = list(results.keys())
    answers = [results[p].answer for p in pipelines]
    
    # Simple word overlap analysis
    for i, pipeline1 in enumerate(pipelines):
        for j, pipeline2 in enumerate(pipelines[i+1:], i+1):
            overlap = calculate_text_overlap(answers[i], answers[j])
            st.markdown(f"• {pipeline1} ↔ {pipeline2}: {overlap:.1%} similarity")

def calculate_text_overlap(text1: str, text2: str) -> float:
    """Calculate simple text overlap percentage."""
    words1 = set(text1.lower().split())
    words2 = set(text2.lower().split())
    
    if not words1 or not words2:
        return 0.0
    
    overlap = len(words1.intersection(words2))
    total = len(words1.union(words2))
    
    return overlap / total if total > 0 else 0.0

def render_key_differences(results: Dict[str, Any]):
    """Highlight key differences between answers."""
    st.markdown("**Key Differences:**")
    
    if len(results) >= 2:
        pipelines = list(results.keys())
        st.markdown(f"• {pipelines[0]} focuses on: *[Analysis would highlight key themes]*")
        st.markdown(f"• {pipelines[1]} emphasizes: *[Different key themes]*")
    else:
        st.info("Need at least 2 successful results for comparison")

def render_document_overlap_analysis(pipeline_results: Dict[str, Any]):
    """Analyze document overlap between pipelines."""
    st.markdown("**Document Overlap:**")
    st.info("Document overlap analysis would show which sources are commonly retrieved")

def render_source_diversity_analysis(pipeline_results: Dict[str, Any]):
    """Analyze source diversity."""
    st.markdown("**Source Diversity:**")
    st.info("Source diversity analysis would compare the variety of sources used")

def render_relevance_comparison(pipeline_results: Dict[str, Any]):
    """Compare relevance scores."""
    st.markdown("**Relevance Score Distribution:**")
    st.info("Relevance comparison would show score distributions across pipelines")

def render_performance_chart(pipeline_results: Dict[str, Any]):
    """Render performance comparison chart."""
    st.markdown("**Execution Time Comparison:**")
    
    for pipeline, result in pipeline_results.items():
        if result.success:
            time_ms = result.execution_time * 1000
            st.markdown(f"• {get_pipeline_icon(pipeline)} {pipeline}: {time_ms:.0f}ms")

def render_resource_usage_comparison(pipeline_results: Dict[str, Any]):
    """Compare resource usage."""
    st.markdown("**Resource Usage:**")
    st.info("Resource usage comparison would show memory and compute requirements")

def render_scalability_insights(pipeline_results: Dict[str, Any]):
    """Provide scalability insights."""
    st.markdown("**Scalability Insights:**")
    st.info("Scalability analysis would predict performance under load")

if __name__ == "__main__":
    main()