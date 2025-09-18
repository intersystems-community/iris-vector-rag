"""
Session State Management

Manages Streamlit session state for the RAG Templates demo application.
"""

import streamlit as st
from typing import Dict, Any, List, Optional
from datetime import datetime
import json

def initialize_session_state():
    """Initialize session state variables with default values."""
    
    # Navigation state
    if "current_page" not in st.session_state:
        st.session_state.current_page = "Home"
    
    # Query state
    if "selected_query" not in st.session_state:
        st.session_state.selected_query = ""
    
    if "custom_query" not in st.session_state:
        st.session_state.custom_query = ""
    
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    
    # Pipeline state
    if "selected_pipeline" not in st.session_state:
        st.session_state.selected_pipeline = "BasicRAG"
    
    if "selected_pipelines_compare" not in st.session_state:
        st.session_state.selected_pipelines_compare = ["BasicRAG", "CRAG"]
    
    if "pipeline_configs" not in st.session_state:
        st.session_state.pipeline_configs = {
            "BasicRAG": {"top_k": 5, "temperature": 0.7},
            "BasicRerank": {"top_k": 5, "temperature": 0.7, "rerank_top_k": 3},
            "CRAG": {"top_k": 5, "temperature": 0.7, "confidence_threshold": 0.8},
            "GraphRAG": {"top_k": 5, "temperature": 0.7, "max_hops": 2}
        }
    
    # Results state
    if "last_results" not in st.session_state:
        st.session_state.last_results = {}
    
    if "comparison_results" not in st.session_state:
        st.session_state.comparison_results = {}
    
    # Performance tracking
    if "performance_metrics" not in st.session_state:
        st.session_state.performance_metrics = []
    
    # UI state
    if "show_retrieved_docs" not in st.session_state:
        st.session_state.show_retrieved_docs = True
    
    if "show_performance_details" not in st.session_state:
        st.session_state.show_performance_details = False
    
    if "export_format" not in st.session_state:
        st.session_state.export_format = "JSON"
    
    # Cache control
    if "cache_enabled" not in st.session_state:
        st.session_state.cache_enabled = True
    
    if "last_cache_clear" not in st.session_state:
        st.session_state.last_cache_clear = datetime.now()

def add_to_query_history(query: str, pipeline: str, results: Dict[str, Any]):
    """Add a query and its results to the history."""
    if "query_history" not in st.session_state:
        st.session_state.query_history = []
    
    history_entry = {
        "timestamp": datetime.now().isoformat(),
        "query": query,
        "pipeline": pipeline,
        "results": results,
        "execution_time": results.get("execution_time", 0)
    }
    
    # Add to beginning of list and limit to 50 entries
    st.session_state.query_history.insert(0, history_entry)
    st.session_state.query_history = st.session_state.query_history[:50]

def get_query_history(limit: Optional[int] = None) -> List[Dict[str, Any]]:
    """Get query history, optionally limited to a certain number of entries."""
    if "query_history" not in st.session_state:
        return []
    
    history = st.session_state.query_history
    return history[:limit] if limit else history

def clear_query_history():
    """Clear the query history."""
    st.session_state.query_history = []

def update_pipeline_config(pipeline: str, config: Dict[str, Any]):
    """Update configuration for a specific pipeline."""
    if "pipeline_configs" not in st.session_state:
        st.session_state.pipeline_configs = {}
    
    st.session_state.pipeline_configs[pipeline] = config

def get_pipeline_config(pipeline: str) -> Dict[str, Any]:
    """Get configuration for a specific pipeline."""
    if "pipeline_configs" not in st.session_state:
        initialize_session_state()
    
    return st.session_state.pipeline_configs.get(pipeline, {})

def add_performance_metric(metric: Dict[str, Any]):
    """Add a performance metric to the tracking list."""
    if "performance_metrics" not in st.session_state:
        st.session_state.performance_metrics = []
    
    metric["timestamp"] = datetime.now().isoformat()
    st.session_state.performance_metrics.append(metric)
    
    # Limit to 1000 entries
    st.session_state.performance_metrics = st.session_state.performance_metrics[-1000:]

def get_performance_metrics(pipeline: Optional[str] = None) -> List[Dict[str, Any]]:
    """Get performance metrics, optionally filtered by pipeline."""
    if "performance_metrics" not in st.session_state:
        return []
    
    metrics = st.session_state.performance_metrics
    
    if pipeline:
        metrics = [m for m in metrics if m.get("pipeline") == pipeline]
    
    return metrics

def export_session_data() -> Dict[str, Any]:
    """Export current session data for download."""
    return {
        "query_history": get_query_history(),
        "pipeline_configs": st.session_state.get("pipeline_configs", {}),
        "performance_metrics": get_performance_metrics(),
        "last_results": st.session_state.get("last_results", {}),
        "comparison_results": st.session_state.get("comparison_results", {}),
        "export_timestamp": datetime.now().isoformat()
    }

def reset_session_state():
    """Reset all session state variables."""
    # Store keys to avoid modifying dict during iteration
    keys_to_remove = list(st.session_state.keys())
    
    for key in keys_to_remove:
        del st.session_state[key]
    
    # Reinitialize
    initialize_session_state()

def get_session_summary() -> Dict[str, Any]:
    """Get a summary of current session statistics."""
    history = get_query_history()
    metrics = get_performance_metrics()
    
    pipeline_usage = {}
    for entry in history:
        pipeline = entry.get("pipeline", "Unknown")
        pipeline_usage[pipeline] = pipeline_usage.get(pipeline, 0) + 1
    
    avg_execution_time = 0
    if metrics:
        total_time = sum(m.get("execution_time", 0) for m in metrics)
        avg_execution_time = total_time / len(metrics)
    
    return {
        "total_queries": len(history),
        "unique_pipelines_used": len(pipeline_usage),
        "pipeline_usage": pipeline_usage,
        "total_performance_metrics": len(metrics),
        "average_execution_time": avg_execution_time,
        "cache_enabled": st.session_state.get("cache_enabled", True),
        "last_cache_clear": st.session_state.get("last_cache_clear")
    }

def save_state_to_url() -> str:
    """Generate a shareable URL with current state (simplified)."""
    # For a real implementation, you'd encode essential state into URL parameters
    # For now, return a placeholder
    base_url = "http://localhost:8501"  # Default Streamlit URL
    
    # Get current query and pipeline
    query = st.session_state.get("selected_query", "")
    pipeline = st.session_state.get("selected_pipeline", "BasicRAG")
    
    # Simple URL encoding (in production, use proper URL encoding)
    if query:
        return f"{base_url}?query={query.replace(' ', '+')}&pipeline={pipeline}"
    
    return base_url

def load_state_from_url_params():
    """Load state from URL parameters (if available)."""
    # Streamlit doesn't have built-in URL parameter support in open source version
    # This would be implemented with query_params in Streamlit Cloud/Enterprise
    # For now, this is a placeholder for the feature
    pass