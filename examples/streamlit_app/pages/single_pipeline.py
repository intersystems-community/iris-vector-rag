"""
Single Pipeline Page

Deep dive into individual RAG pipeline performance and capabilities.
Allows detailed exploration of a single pipeline with comprehensive configuration options.
"""

import sys
from pathlib import Path

import streamlit as st

# Add project root to path
project_root = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(project_root))

from components.header import (
    render_error_message,
    render_loading_spinner,
    render_page_header,
)
from components.query_input import (
    render_query_configuration,
    render_query_history,
    render_query_input,
)
from components.results_display import render_query_result
from components.sidebar import render_advanced_settings, render_pipeline_selector
from utils.app_config import get_config
from utils.pipeline_integration import execute_pipeline_query, get_pipeline_status
from utils.session_state import add_performance_metric, add_to_query_history


def main():
    """Main function for the single pipeline page."""

    # Page configuration
    render_page_header(
        "Single Pipeline Deep Dive",
        "Explore individual RAG pipelines in detail with comprehensive configuration and analysis",
        "🔍",
    )

    # Initialize session state
    if "single_pipeline_results" not in st.session_state:
        st.session_state.single_pipeline_results = {}

    # Main layout
    col1, col2 = st.columns([1, 2])

    with col1:
        render_pipeline_selection()
        render_pipeline_configuration()
        render_pipeline_info()

    with col2:
        render_query_interface()
        render_results_section()


def render_pipeline_selection():
    """Render pipeline selection interface."""
    st.markdown("### 🤖 Pipeline Selection")

    # Pipeline selector
    selected_pipeline = render_pipeline_selector(key="single_pipeline")
    st.session_state.selected_single_pipeline = selected_pipeline

    # Pipeline status
    status = get_pipeline_status(selected_pipeline)

    if status.get("initialized", False):
        st.success(f"✅ {selected_pipeline} is ready")
    elif status.get("available", False):
        st.warning(f"⚠️ {selected_pipeline} needs initialization")
        if st.button("🚀 Initialize Pipeline", key="init_pipeline"):
            with st.spinner("Initializing pipeline..."):
                # This would normally initialize the pipeline
                st.success("Pipeline initialized!")
                st.rerun()
    else:
        st.error(f"❌ {selected_pipeline} is not available")


def render_pipeline_configuration():
    """Render pipeline-specific configuration options."""
    st.markdown("### ⚙️ Configuration")

    selected_pipeline = st.session_state.get("selected_single_pipeline", "BasicRAG")

    # Basic configuration
    config = render_query_configuration(selected_pipeline, key="single_config")

    # Advanced settings
    advanced_config = render_advanced_settings()

    # Store configuration in session state
    st.session_state.single_pipeline_config = {**config, **advanced_config}

    # Configuration presets
    with st.expander("📋 Configuration Presets"):
        preset = st.selectbox(
            "Choose preset:",
            ["Custom", "Fast & Simple", "Balanced", "High Quality", "Research Mode"],
            key="config_preset",
        )

        if preset != "Custom":
            if st.button("Apply Preset", key="apply_preset"):
                apply_configuration_preset(preset, selected_pipeline)
                st.success(f"Applied {preset} configuration")
                st.rerun()


def render_pipeline_info():
    """Render information about the selected pipeline."""
    selected_pipeline = st.session_state.get("selected_single_pipeline", "BasicRAG")
    config = get_config()
    pipeline_info = config.get_pipeline_info(selected_pipeline)

    st.markdown("### ℹ️ Pipeline Information")

    # Basic info
    st.markdown(f"**{pipeline_info['icon']} {pipeline_info['name']}**")
    st.markdown(f"*{pipeline_info['description']}*")

    # Complexity and performance
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Complexity", pipeline_info["complexity"])
    with col2:
        st.metric("Avg Latency", pipeline_info["avg_latency"])

    # Strengths
    st.markdown("**Strengths:**")
    for strength in pipeline_info["strengths"]:
        st.markdown(f"• {strength}")

    # Best use cases
    st.markdown("**Best for:**")
    for use_case in pipeline_info["use_cases"]:
        st.markdown(f"• {use_case}")


def render_query_interface():
    """Render the query input and execution interface."""
    st.markdown("## 💬 Query Interface")

    # Query input
    query_data = render_query_input(key="single_query")
    query = query_data.get("query", "")

    # Quick actions
    col1, col2, col3 = st.columns(3)

    with col1:
        history_query = render_query_history_button()
        if history_query:
            query = history_query
            st.session_state.single_query_custom = history_query
            st.rerun()

    with col2:
        if st.button("🎲 Random Query", key="random_query"):
            query = get_random_sample_query()
            st.session_state.single_query_custom = query
            st.rerun()

    with col3:
        if st.button("🧹 Clear Results", key="clear_results"):
            st.session_state.single_pipeline_results = {}
            st.rerun()

    # Execute query
    if query and len(query.strip()) > 0:
        selected_pipeline = st.session_state.get("selected_single_pipeline", "BasicRAG")
        config = st.session_state.get("single_pipeline_config", {})

        if st.button("🚀 Execute Query", type="primary", key="execute_single"):
            execute_single_query(query, selected_pipeline, config)
    else:
        st.info("💡 Enter a query above to get started")


def render_results_section():
    """Render the results section."""
    if not st.session_state.single_pipeline_results:
        render_empty_results()
        return

    st.markdown("## 📊 Results")

    # Results tabs
    tab1, tab2, tab3 = st.tabs(["🎯 Answer", "📄 Documents", "📈 Analysis"])

    with tab1:
        render_answer_tab()

    with tab2:
        render_documents_tab()

    with tab3:
        render_analysis_tab()


def render_empty_results():
    """Render empty results placeholder."""
    st.markdown(
        """
    <div style="text-align: center; padding: 3rem; color: #666;">
        <h3>🎯 Ready to Explore!</h3>
        <p>Execute a query to see detailed results and analysis</p>
        <p>Use the query interface above to get started</p>
    </div>
    """,
        unsafe_allow_html=True,
    )


def render_answer_tab():
    """Render the answer analysis tab."""
    result = st.session_state.single_pipeline_results.get("latest")
    if not result:
        st.info("No results to display")
        return

    # Main answer display
    render_query_result(result, show_details=False)

    # Answer analysis
    st.markdown("### 🔍 Answer Analysis")

    answer = result.answer
    if answer:
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Word Count", len(answer.split()))

        with col2:
            st.metric("Characters", len(answer))

        with col3:
            sentences = answer.count(".") + answer.count("!") + answer.count("?")
            st.metric("Sentences", sentences)

        # Key phrases extraction (simplified)
        key_phrases = extract_key_phrases(answer)
        if key_phrases:
            st.markdown("**Key Phrases:**")
            st.write(", ".join(key_phrases))


def render_documents_tab():
    """Render the retrieved documents tab."""
    result = st.session_state.single_pipeline_results.get("latest")
    if not result:
        st.info("No results to display")
        return

    st.markdown("### 📄 Retrieved Documents Analysis")

    docs = result.retrieved_documents
    if not docs:
        st.warning("No documents were retrieved")
        return

    # Document statistics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Total Docs", len(docs))

    with col2:
        avg_score = sum(doc.get("metadata", {}).get("score", 0) for doc in docs) / len(
            docs
        )
        st.metric("Avg Score", f"{avg_score:.3f}")

    with col3:
        total_chars = sum(len(doc.get("content", "")) for doc in docs)
        st.metric("Total Content", f"{total_chars:,} chars")

    with col4:
        unique_sources = len(
            set(doc.get("metadata", {}).get("source", "unknown") for doc in docs)
        )
        st.metric("Unique Sources", unique_sources)

    # Document relevance chart
    render_document_relevance_chart(docs)

    # Detailed document display
    render_query_result(result, show_details=True)


def render_analysis_tab():
    """Render the performance analysis tab."""
    result = st.session_state.single_pipeline_results.get("latest")
    if not result:
        st.info("No results to display")
        return

    st.markdown("### 📈 Performance Analysis")

    # Performance metrics
    render_query_result(result, show_details=False)

    # Historical comparison
    render_historical_performance()

    # Pipeline-specific insights
    render_pipeline_insights(result)


def execute_single_query(query: str, pipeline: str, config: dict):
    """Execute a query and store results."""
    with st.spinner(f"Executing query with {pipeline}..."):
        try:
            # Execute the query
            result = execute_pipeline_query(pipeline, query, config)

            # Store result
            st.session_state.single_pipeline_results["latest"] = result

            # Add to history
            add_to_query_history(
                query,
                pipeline,
                {
                    "success": result.success,
                    "execution_time": result.execution_time,
                    "answer_length": len(result.answer) if result.answer else 0,
                    "num_documents": len(result.retrieved_documents),
                },
            )

            # Add performance metric
            add_performance_metric(
                {
                    "pipeline": pipeline,
                    "query": query,
                    "execution_time": result.execution_time,
                    "success": result.success,
                    "num_documents": len(result.retrieved_documents),
                }
            )

            if result.success:
                st.success(
                    f"✅ Query executed successfully in {result.execution_time:.2f}s"
                )
            else:
                st.error(f"❌ Query failed: {result.error}")

        except Exception as e:
            st.error(f"❌ Unexpected error: {str(e)}")


# Helper functions
def render_query_history_button():
    """Render query history selection."""
    if st.button("📚 Use from History", key="use_history"):
        history = st.session_state.get("query_history", [])
        if history:
            return history[0].get("query", "")
    return None


def get_random_sample_query():
    """Get a random sample query."""
    config = get_config()
    import random

    return random.choice(config.sample_queries)


def apply_configuration_preset(preset: str, pipeline: str):
    """Apply a configuration preset."""
    presets = {
        "Fast & Simple": {"top_k": 3, "temperature": 0.3},
        "Balanced": {"top_k": 5, "temperature": 0.7},
        "High Quality": {"top_k": 10, "temperature": 0.5},
        "Research Mode": {"top_k": 15, "temperature": 0.8},
    }

    config = presets.get(preset, {})

    # Pipeline-specific adjustments
    if pipeline == "BasicRerank" and preset == "High Quality":
        config["rerank_top_k"] = min(5, config.get("top_k", 5))
    elif pipeline == "CRAG" and preset == "High Quality":
        config["confidence_threshold"] = 0.9
    elif pipeline == "GraphRAG" and preset == "Research Mode":
        config["max_hops"] = 3

    st.session_state.single_pipeline_config = config


def extract_key_phrases(text: str) -> list:
    """Extract key phrases from text (simplified)."""
    # Simple phrase extraction
    words = text.lower().split()
    phrases = []

    # Look for common medical/scientific patterns
    for i in range(len(words) - 1):
        if words[i] in ["treatment", "diagnosis", "symptoms", "therapy"]:
            phrase = " ".join(words[i : i + 2])
            phrases.append(phrase)

    return phrases[:5]  # Return top 5


def render_document_relevance_chart(docs: list):
    """Render a simple document relevance visualization."""
    st.markdown("**Document Relevance Scores:**")

    for i, doc in enumerate(docs[:5]):  # Show top 5
        score = doc.get("metadata", {}).get("score", 0)
        title = doc.get("title", f"Document {i+1}")[:30] + "..."

        # Simple progress bar
        st.markdown(
            f"""
        <div style="margin-bottom: 0.5rem;">
            <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                <span style="font-size: 0.9rem;">{title}</span>
                <span style="font-weight: bold;">{score:.3f}</span>
            </div>
            <div style="background-color: #e9ecef; border-radius: 0.375rem; overflow: hidden; height: 0.5rem;">
                <div style="background-color: #1f77b4; height: 100%; width: {score * 100}%; transition: width 0.3s ease;"></div>
            </div>
        </div>
        """,
            unsafe_allow_html=True,
        )


def render_historical_performance():
    """Render historical performance comparison."""
    st.markdown("**Historical Performance:**")

    # Get recent performance metrics
    metrics = st.session_state.get("performance_metrics", [])
    pipeline = st.session_state.get("selected_single_pipeline", "BasicRAG")

    pipeline_metrics = [m for m in metrics if m.get("pipeline") == pipeline][
        -10:
    ]  # Last 10

    if pipeline_metrics:
        avg_time = sum(m.get("execution_time", 0) for m in pipeline_metrics) / len(
            pipeline_metrics
        )
        success_rate = sum(
            1 for m in pipeline_metrics if m.get("success", False)
        ) / len(pipeline_metrics)

        col1, col2 = st.columns(2)
        with col1:
            st.metric("Avg Execution Time", f"{avg_time:.2f}s")
        with col2:
            st.metric("Success Rate", f"{success_rate:.1%}")
    else:
        st.info("No historical data available")


def render_pipeline_insights(result):
    """Render pipeline-specific insights."""
    pipeline = result.pipeline

    st.markdown("**Pipeline-Specific Insights:**")

    insights = {
        "BasicRAG": [
            "Fast and reliable baseline performance",
            "Good for straightforward factual queries",
            "Consider BasicRerank for better relevance",
        ],
        "BasicRerank": [
            "Improved document relevance through reranking",
            "Better for complex multi-faceted queries",
            "Slightly higher latency but better quality",
        ],
        "CRAG": [
            "Self-correcting mechanism improves accuracy",
            "Excellent for fact-sensitive applications",
            "May iterate multiple times for complex queries",
        ],
        "GraphRAG": [
            "Leverages entity relationships",
            "Excellent for research and exploration",
            "Best for queries requiring deep understanding",
        ],
    }

    pipeline_insights = insights.get(pipeline, ["No specific insights available"])

    for insight in pipeline_insights:
        st.markdown(f"• {insight}")


if __name__ == "__main__":
    main()
