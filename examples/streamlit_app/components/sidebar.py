"""
Sidebar Navigation Component

Renders the main navigation sidebar for the RAG Templates demo application.
"""

from typing import List

import streamlit as st
from utils.session_state import get_session_summary


def render_sidebar() -> str:
    """
    Render the main navigation sidebar.

    Returns:
        str: The selected page name
    """

    with st.sidebar:
        # App branding
        st.markdown(
            """
        <div style="text-align: center; padding: 1rem 0;">
            <h2 style="color: #1f77b4; margin: 0;">🤖 RAG Templates</h2>
            <p style="color: #666; margin: 0; font-size: 0.9rem;">Interactive Demo</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown("---")

        # Navigation menu
        st.markdown("### 📋 Navigation")

        pages = {
            "🏠 Home": "Home",
            "🔍 Single Pipeline": "Single Pipeline",
            "⚖️ Compare Pipelines": "Compare Pipelines",
            "📊 Performance Metrics": "Performance Metrics",
            "⚙️ Configuration": "Configuration",
        }

        # Use radio buttons for navigation
        selected_page_key = st.radio(
            "Choose a page:", list(pages.keys()), index=0, label_visibility="collapsed"
        )

        selected_page = pages[selected_page_key]

        st.markdown("---")

        # Pipeline status indicator
        render_pipeline_status()

        st.markdown("---")

        # Session summary
        render_session_summary()

        st.markdown("---")

        # Quick actions
        render_quick_actions()

    return selected_page


def render_pipeline_status():
    """Render pipeline status indicators."""
    st.markdown("### 🤖 Pipeline Status")

    # This would normally check actual pipeline status
    # For now, showing mock status
    pipelines = ["BasicRAG", "BasicRerank", "CRAG", "GraphRAG"]

    for pipeline in pipelines:
        # Simulate different statuses
        if pipeline in ["BasicRAG", "CRAG"]:
            status = "🟢 Ready"
            color = "green"
        elif pipeline == "BasicRerank":
            status = "🟡 Loading"
            color = "orange"
        else:
            status = "🔴 Error"
            color = "red"

        st.markdown(
            f"""
        <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 0.5rem;">
            <span style="font-size: 0.9rem;">{pipeline}</span>
            <span style="color: {color}; font-size: 0.8rem;">{status}</span>
        </div>
        """,
            unsafe_allow_html=True,
        )


def render_session_summary():
    """Render session summary statistics."""
    st.markdown("### 📈 Session Stats")

    try:
        summary = get_session_summary()

        # Display key metrics
        col1, col2 = st.columns(2)

        with col1:
            st.metric("Queries", summary.get("total_queries", 0))
            st.metric("Pipelines", summary.get("unique_pipelines_used", 0))

        with col2:
            avg_time = summary.get("average_execution_time", 0)
            st.metric("Avg Time", f"{avg_time:.1f}s" if avg_time > 0 else "0.0s")

            cache_status = "On" if summary.get("cache_enabled", True) else "Off"
            st.metric("Cache", cache_status)

        # Most used pipeline
        pipeline_usage = summary.get("pipeline_usage", {})
        if pipeline_usage:
            most_used = max(pipeline_usage.items(), key=lambda x: x[1])
            st.markdown(f"**Most Used:** {most_used[0]} ({most_used[1]}x)")

    except Exception as e:
        st.markdown("*Session stats unavailable*")


def render_quick_actions():
    """Render quick action buttons."""
    st.markdown("### ⚡ Quick Actions")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("🔄 Reset", use_container_width=True, help="Reset session state"):
            # Clear session state
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()

    with col2:
        if st.button("📥 Export", use_container_width=True, help="Export session data"):
            st.session_state.show_export_dialog = True

    # Cache controls
    if st.button(
        "🗑️ Clear Cache", use_container_width=True, help="Clear pipeline cache"
    ):
        st.session_state.cache_cleared = True
        st.success("Cache cleared!")

    # Help button
    if st.button("❓ Help", use_container_width=True, help="Show help information"):
        st.session_state.show_help = True


def render_export_dialog():
    """Render export dialog if requested."""
    if st.session_state.get("show_export_dialog", False):
        with st.expander("📥 Export Session Data", expanded=True):
            st.markdown("Export your session data for analysis or sharing.")

            export_format = st.selectbox(
                "Format:", ["JSON", "CSV", "Excel"], help="Choose export format"
            )

            col1, col2 = st.columns(2)

            with col1:
                if st.button("📄 Download"):
                    # This would trigger actual download
                    st.success(f"Downloaded session data as {export_format}")
                    st.session_state.show_export_dialog = False
                    st.rerun()

            with col2:
                if st.button("❌ Cancel"):
                    st.session_state.show_export_dialog = False
                    st.rerun()


def render_help_dialog():
    """Render help dialog if requested."""
    if st.session_state.get("show_help", False):
        with st.expander("❓ Help & Instructions", expanded=True):
            st.markdown(
                """
            ### 🚀 Getting Started
            
            1. **Home**: Overview and quick start
            2. **Single Pipeline**: Test individual pipelines
            3. **Compare Pipelines**: Side-by-side analysis
            4. **Performance Metrics**: Detailed analytics
            5. **Configuration**: Customize settings
            
            ### 💡 Tips
            
            - Use sample queries for best results
            - Check pipeline status indicators
            - Export results for further analysis
            - Reset session to start fresh
            
            ### 🔧 Troubleshooting
            
            - Red status: Check logs and configuration
            - Slow responses: Try smaller top_k values
            - Errors: Reset session or clear cache
            """
            )

            if st.button("✅ Got it!", use_container_width=True):
                st.session_state.show_help = False
                st.rerun()


# Additional sidebar components that can be used on specific pages
def render_pipeline_selector(multiselect: bool = False, key: str = "pipeline_selector"):
    """Render pipeline selector widget."""
    pipelines = ["BasicRAG", "BasicRerank", "CRAG", "GraphRAG"]

    if multiselect:
        selected = st.multiselect(
            "Select Pipeline(s):",
            pipelines,
            default=["BasicRAG", "CRAG"],
            key=key,
            help="Choose pipelines to compare",
        )
    else:
        selected = st.selectbox(
            "Select Pipeline:", pipelines, key=key, help="Choose a RAG pipeline"
        )

    return selected


def render_query_settings():
    """Render query configuration settings."""
    st.markdown("### ⚙️ Query Settings")

    col1, col2 = st.columns(2)

    with col1:
        top_k = st.slider(
            "Top K Results:",
            min_value=1,
            max_value=20,
            value=5,
            help="Number of documents to retrieve",
        )

    with col2:
        temperature = st.slider(
            "Temperature:",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.1,
            help="LLM generation temperature",
        )

    return {"top_k": top_k, "temperature": temperature}


def render_advanced_settings():
    """Render advanced pipeline settings."""
    with st.expander("🔧 Advanced Settings"):

        # Timeout setting
        timeout = st.slider(
            "Query Timeout (seconds):",
            min_value=10,
            max_value=120,
            value=60,
            help="Maximum time to wait for results",
        )

        # Cache setting
        use_cache = st.checkbox(
            "Enable Caching",
            value=True,
            help="Cache results for faster subsequent queries",
        )

        # Debug mode
        debug_mode = st.checkbox(
            "Debug Mode", value=False, help="Show detailed execution information"
        )

        return {"timeout": timeout, "use_cache": use_cache, "debug_mode": debug_mode}
