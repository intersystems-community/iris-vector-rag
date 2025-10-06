"""
RAG Templates Streamlit Demo Application

An interactive demonstration of the RAG Templates framework showcasing
all available RAG pipelines with side-by-side comparisons and performance metrics.
"""

import os
import sys
from pathlib import Path

import streamlit as st

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from components.header import render_header
from components.sidebar import render_sidebar
from utils.app_config import AppConfig
from utils.session_state import initialize_session_state

# Configure Streamlit page
st.set_page_config(
    page_title="RAG Templates Demo",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/your-org/rag-templates",
        "Report a bug": "https://github.com/your-org/rag-templates/issues",
        "About": """
        # RAG Templates Demo
        
        Interactive demonstration of the RAG Templates framework.
        
        **Features:**
        - 4 RAG pipeline implementations
        - Side-by-side comparisons
        - Performance analytics
        - Real biomedical queries
        
        Built with Streamlit and the RAG Templates framework.
        """,
    },
)


def main():
    """Main application entry point."""

    # Initialize configuration and session state
    config = AppConfig()
    initialize_session_state()

    # Render header
    render_header()

    # Render sidebar navigation
    current_page = render_sidebar()

    # Main content area
    if current_page == "Home":
        render_home_page()
    elif current_page == "Single Pipeline":
        st.switch_page("pages/single_pipeline.py")
    elif current_page == "Compare Pipelines":
        st.switch_page("pages/compare_pipelines.py")
    elif current_page == "Performance Metrics":
        st.switch_page("pages/performance_metrics.py")
    elif current_page == "Configuration":
        st.switch_page("pages/configuration.py")


def render_home_page():
    """Render the home page content."""

    # Hero section
    st.markdown(
        """
    <div style="text-align: center; padding: 2rem 0;">
        <h1 style="color: #1f77b4; margin-bottom: 0;">🤖 RAG Templates Demo</h1>
        <p style="font-size: 1.2rem; color: #666; margin-top: 0.5rem;">
            Interactive demonstration of cutting-edge RAG pipeline implementations
        </p>
    </div>
    """,
        unsafe_allow_html=True,
    )

    # Quick stats
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            label="🔍 Available Pipelines",
            value="4",
            help="BasicRAG, BasicRerank, CRAG, GraphRAG",
        )

    with col2:
        st.metric(
            label="📊 Sample Queries",
            value="15",
            help="Biomedical domain queries for testing",
        )

    with col3:
        st.metric(
            label="⚡ Performance Metrics",
            value="6+",
            help="Latency, accuracy, token usage, and more",
        )

    with col4:
        st.metric(
            label="🎯 Success Rate",
            value="95%+",
            help="Average pipeline initialization success",
        )

    st.markdown("---")

    # Feature overview
    st.markdown("## 🚀 What You Can Do")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown(
            """
        ### 🔍 **Single Pipeline Deep Dive**
        - Test individual RAG pipelines
        - View retrieved documents with highlighting
        - Analyze response quality and sources
        - Adjust pipeline-specific parameters
        """
        )

        if st.button(
            "🔍 Explore Single Pipeline", type="primary", use_container_width=True
        ):
            st.switch_page("pages/single_pipeline.py")

    with col2:
        st.markdown(
            """
        ### ⚖️ **Side-by-Side Comparison**
        - Compare multiple pipelines simultaneously
        - See response differences in real-time
        - Analyze relative strengths and weaknesses
        - Export comparison results
        """
        )

        if st.button("⚖️ Compare Pipelines", type="primary", use_container_width=True):
            st.switch_page("pages/compare_pipelines.py")

    st.markdown("---")

    # Pipeline overview
    st.markdown("## 🤖 Available RAG Pipelines")

    pipeline_info = {
        "BasicRAG": {
            "icon": "📝",
            "description": "Standard RAG with vector similarity search and LLM generation",
            "strengths": ["Fast", "Reliable", "Good baseline"],
            "use_cases": [
                "General queries",
                "Quick prototyping",
                "Baseline comparison",
            ],
        },
        "BasicRerank": {
            "icon": "🎯",
            "description": "Enhanced RAG with document reranking for improved relevance",
            "strengths": [
                "Better relevance",
                "Improved accuracy",
                "Context optimization",
            ],
            "use_cases": [
                "Quality-focused queries",
                "Complex questions",
                "Domain-specific tasks",
            ],
        },
        "CRAG": {
            "icon": "🔄",
            "description": "Corrective RAG with self-correction and quality assessment",
            "strengths": ["Self-correcting", "Quality validation", "Robust responses"],
            "use_cases": [
                "Critical applications",
                "Fact verification",
                "High-stakes queries",
            ],
        },
        "GraphRAG": {
            "icon": "🕸️",
            "description": "Graph-based RAG using entity relationships and knowledge graphs",
            "strengths": [
                "Relationship awareness",
                "Deep insights",
                "Complex reasoning",
            ],
            "use_cases": [
                "Research queries",
                "Multi-hop reasoning",
                "Knowledge exploration",
            ],
        },
    }

    for i, (name, info) in enumerate(pipeline_info.items()):
        if i % 2 == 0:
            col1, col2 = st.columns(2)

        current_col = col1 if i % 2 == 0 else col2

        with current_col:
            with st.container():
                st.markdown(
                    f"""
                <div style="border: 1px solid #ddd; border-radius: 10px; padding: 1rem; margin-bottom: 1rem; background: #f9f9f9;">
                    <h4 style="margin-top: 0; color: #1f77b4;">{info['icon']} {name}</h4>
                    <p>{info['description']}</p>
                    <p><strong>Strengths:</strong> {', '.join(info['strengths'])}</p>
                    <p><strong>Best for:</strong> {', '.join(info['use_cases'])}</p>
                </div>
                """,
                    unsafe_allow_html=True,
                )

    st.markdown("---")

    # Quick start section
    st.markdown("## 🏃‍♂️ Quick Start")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown(
            """
        ### 1️⃣ Choose Your Adventure
        - **Single Pipeline**: Deep dive into one pipeline
        - **Compare Mode**: Side-by-side analysis
        - **Metrics**: Performance analytics
        """
        )

    with col2:
        st.markdown(
            """
        ### 2️⃣ Select a Query
        - Use our sample biomedical queries
        - Or enter your own custom question
        - Try different complexity levels
        """
        )

    with col3:
        st.markdown(
            """
        ### 3️⃣ Analyze Results
        - View generated responses
        - Inspect retrieved documents
        - Compare performance metrics
        """
        )

    # Sample queries preview
    st.markdown("## 💡 Sample Queries Preview")

    sample_queries = [
        "What are the current treatment options for type 2 diabetes mellitus?",
        "How does COVID-19 affect respiratory function and lung capacity?",
        "What are the most effective interventions for managing hypertension?",
        "How do immunotherapy treatments work in cancer patients?",
        "What are the genetic factors associated with breast cancer risk?",
    ]

    col1, col2 = st.columns([3, 1])

    with col1:
        selected_query = st.selectbox(
            "Try one of these biomedical queries:",
            sample_queries,
            help="These queries are designed to test different aspects of RAG performance",
        )

    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
        if st.button("🚀 Test This Query", type="secondary", use_container_width=True):
            st.session_state.selected_query = selected_query
            st.switch_page("pages/single_pipeline.py")

    # Footer
    st.markdown("---")
    st.markdown(
        """
    <div style="text-align: center; color: #666; padding: 1rem;">
        <p>Built with ❤️ using <strong>Streamlit</strong> and the <strong>RAG Templates</strong> framework</p>
        <p>📧 Questions? Issues? <a href="https://github.com/your-org/rag-templates/issues">Report them here</a></p>
    </div>
    """,
        unsafe_allow_html=True,
    )


if __name__ == "__main__":
    main()
