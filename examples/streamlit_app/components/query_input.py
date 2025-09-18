"""
Query Input Component

Provides query input interface with sample queries and configuration options.
"""

import streamlit as st
from typing import List, Dict, Any, Optional
from utils.app_config import get_config

def render_query_input(key: str = "main_query") -> Dict[str, Any]:
    """
    Render query input interface.
    
    Args:
        key: Unique key for the input widgets
        
    Returns:
        Dict containing query and configuration
    """
    config = get_config()
    
    st.markdown("### 💬 Enter Your Query")
    
    # Query input tabs
    tab1, tab2 = st.tabs(["✍️ Custom Query", "📝 Sample Queries"])
    
    with tab1:
        custom_query = st.text_area(
            "Enter your question:",
            value=st.session_state.get("custom_query", ""),
            height=100,
            max_chars=config.max_query_length,
            placeholder="Ask a question about biomedical research, treatments, or health conditions...",
            key=f"{key}_custom",
            help=f"Maximum {config.max_query_length} characters"
        )
        
        # Character counter
        char_count = len(custom_query)
        color = "green" if char_count < config.max_query_length * 0.8 else "orange" if char_count < config.max_query_length else "red"
        st.markdown(f"<div style='text-align: right; color: {color}; font-size: 0.8rem;'>{char_count}/{config.max_query_length} characters</div>", unsafe_allow_html=True)
    
    with tab2:
        # Categorized sample queries
        query_categories = config.get_query_categories()
        
        selected_category = st.selectbox(
            "Choose a category:",
            list(query_categories.keys()),
            key=f"{key}_category"
        )
        
        sample_queries = query_categories[selected_category]
        
        selected_sample = st.selectbox(
            "Select a sample query:",
            sample_queries,
            key=f"{key}_sample"
        )
        
        # Preview selected query
        st.markdown("**Selected Query Preview:**")
        st.info(selected_sample)
        
        if st.button("📋 Use This Query", key=f"{key}_use_sample"):
            custom_query = selected_sample
            st.session_state.custom_query = selected_sample
            st.rerun()
    
    # Determine final query
    final_query = custom_query.strip()
    
    # Query validation
    if final_query:
        if len(final_query) < 10:
            st.warning("⚠️ Query seems too short. Consider adding more details for better results.")
        elif len(final_query) > config.max_query_length:
            st.error(f"❌ Query exceeds maximum length of {config.max_query_length} characters.")
            final_query = ""
    
    return {
        "query": final_query,
        "category": selected_category if 'selected_category' in locals() else "Custom",
        "is_sample": final_query in sample_queries if 'sample_queries' in locals() else False
    }

def render_query_configuration(pipeline: str, key: str = "config") -> Dict[str, Any]:
    """
    Render query configuration options.
    
    Args:
        pipeline: Selected pipeline name
        key: Unique key for the widgets
        
    Returns:
        Dict containing configuration parameters
    """
    st.markdown("### ⚙️ Query Configuration")
    
    col1, col2 = st.columns(2)
    
    with col1:
        top_k = st.slider(
            "📊 Top K Results:",
            min_value=1,
            max_value=20,
            value=st.session_state.get(f"{key}_top_k", 5),
            key=f"{key}_top_k",
            help="Number of documents to retrieve from the knowledge base"
        )
    
    with col2:
        temperature = st.slider(
            "🌡️ Temperature:",
            min_value=0.0,
            max_value=1.0,
            value=st.session_state.get(f"{key}_temperature", 0.7),
            step=0.1,
            key=f"{key}_temperature",
            help="Controls randomness in response generation (0 = deterministic, 1 = creative)"
        )
    
    # Pipeline-specific configurations
    pipeline_config = {}
    
    if pipeline == "BasicRerank":
        with st.expander("🎯 Reranking Settings"):
            rerank_top_k = st.slider(
                "Rerank Top K:",
                min_value=1,
                max_value=min(top_k, 10),
                value=min(3, top_k),
                key=f"{key}_rerank_top_k",
                help="Number of top documents to rerank for final selection"
            )
            pipeline_config["rerank_top_k"] = rerank_top_k
    
    elif pipeline == "CRAG":
        with st.expander("🔄 CRAG Settings"):
            confidence_threshold = st.slider(
                "Confidence Threshold:",
                min_value=0.0,
                max_value=1.0,
                value=0.8,
                step=0.05,
                key=f"{key}_confidence",
                help="Minimum confidence score for self-correction"
            )
            pipeline_config["confidence_threshold"] = confidence_threshold
    
    elif pipeline == "GraphRAG":
        with st.expander("🕸️ Graph Settings"):
            max_hops = st.slider(
                "Maximum Hops:",
                min_value=1,
                max_value=5,
                value=2,
                key=f"{key}_max_hops",
                help="Maximum relationship traversal depth in the knowledge graph"
            )
            pipeline_config["max_hops"] = max_hops
    
    return {
        "top_k": top_k,
        "temperature": temperature,
        **pipeline_config
    }

def render_query_history(limit: int = 10) -> Optional[str]:
    """
    Render query history interface.
    
    Args:
        limit: Maximum number of history items to show
        
    Returns:
        Selected query from history or None
    """
    if "query_history" not in st.session_state or not st.session_state.query_history:
        st.markdown("*No query history available*")
        return None
    
    st.markdown("### 📚 Query History")
    
    history = st.session_state.query_history[:limit]
    
    # Create a list of query options with metadata
    query_options = []
    for i, entry in enumerate(history):
        timestamp = entry.get("timestamp", "Unknown")
        pipeline = entry.get("pipeline", "Unknown")
        query = entry.get("query", "")
        execution_time = entry.get("execution_time", 0)
        
        # Truncate long queries for display
        display_query = query[:60] + "..." if len(query) > 60 else query
        
        option_text = f"{display_query} | {pipeline} | {execution_time:.1f}s"
        query_options.append((option_text, query))
    
    if query_options:
        col1, col2 = st.columns([3, 1])
        
        with col1:
            selected_option = st.selectbox(
                "Previous queries:",
                [opt[0] for opt in query_options],
                key="history_selector",
                help="Select a previous query to rerun"
            )
        
        with col2:
            st.markdown("<br>", unsafe_allow_html=True)  # Add spacing
            if st.button("🔄 Reuse Query", key="reuse_history"):
                # Find the selected query
                for opt_text, query in query_options:
                    if opt_text == selected_option:
                        return query
        
        # Clear history option
        if st.button("🗑️ Clear History", key="clear_history"):
            st.session_state.query_history = []
            st.rerun()
    
    return None

def render_quick_queries() -> Optional[str]:
    """
    Render quick query buttons for common use cases.
    
    Returns:
        Selected quick query or None
    """
    st.markdown("### ⚡ Quick Queries")
    
    quick_queries = [
        "What are the symptoms of diabetes?",
        "How does COVID-19 affect the lungs?",
        "What are cancer treatment options?",
        "How do vaccines work?",
        "What causes hypertension?"
    ]
    
    cols = st.columns(len(quick_queries))
    
    for i, query in enumerate(quick_queries):
        with cols[i]:
            if st.button(f"❓ {query[:20]}...", key=f"quick_{i}", use_container_width=True):
                return query
    
    return None

def render_query_suggestions(current_query: str) -> List[str]:
    """
    Render query suggestions based on current input.
    
    Args:
        current_query: Current query text
        
    Returns:
        List of suggested queries
    """
    if not current_query or len(current_query) < 5:
        return []
    
    # Simple keyword-based suggestions (in a real app, this would use ML)
    keywords_suggestions = {
        "diabetes": [
            "What are the complications of diabetes?",
            "How is diabetes diagnosed?",
            "What lifestyle changes help manage diabetes?"
        ],
        "covid": [
            "What are the long-term effects of COVID-19?",
            "How effective are COVID-19 vaccines?",
            "What treatments are available for COVID-19?"
        ],
        "cancer": [
            "What are the different types of cancer treatment?",
            "How is cancer staged?",
            "What are cancer risk factors?"
        ],
        "heart": [
            "What causes heart disease?",
            "How can heart disease be prevented?",
            "What are the symptoms of heart attack?"
        ]
    }
    
    suggestions = []
    query_lower = current_query.lower()
    
    for keyword, queries in keywords_suggestions.items():
        if keyword in query_lower:
            suggestions.extend(queries)
    
    if suggestions:
        st.markdown("### 💡 Related Queries")
        
        for suggestion in suggestions[:3]:  # Limit to 3 suggestions
            if st.button(f"💭 {suggestion}", key=f"suggest_{hash(suggestion)}"):
                return suggestion
    
    return []

def render_query_validation(query: str) -> Dict[str, Any]:
    """
    Validate and provide feedback on the query.
    
    Args:
        query: Query to validate
        
    Returns:
        Dict containing validation results
    """
    validation = {
        "is_valid": True,
        "warnings": [],
        "suggestions": [],
        "estimated_complexity": "Medium"
    }
    
    if not query:
        validation["is_valid"] = False
        validation["warnings"].append("Query cannot be empty")
        return validation
    
    # Length validation
    if len(query) < 10:
        validation["warnings"].append("Query is very short - consider adding more details")
    elif len(query) > 500:
        validation["warnings"].append("Very long query - consider breaking into smaller questions")
    
    # Complexity estimation
    question_words = ["what", "how", "why", "when", "where", "which", "who"]
    complex_terms = ["mechanism", "pathway", "interaction", "comparison", "analysis"]
    
    if any(word in query.lower() for word in complex_terms):
        validation["estimated_complexity"] = "High"
        validation["suggestions"].append("Complex query detected - GraphRAG may provide better results")
    elif any(word in query.lower() for word in question_words):
        validation["estimated_complexity"] = "Medium"
    else:
        validation["estimated_complexity"] = "Low"
    
    # Medical term detection
    medical_terms = ["treatment", "symptoms", "diagnosis", "therapy", "medication", "disease"]
    if any(term in query.lower() for term in medical_terms):
        validation["suggestions"].append("Medical query detected - all pipelines should work well")
    
    return validation

def render_advanced_query_options():
    """Render advanced query options in an expander."""
    with st.expander("🔧 Advanced Options"):
        
        col1, col2 = st.columns(2)
        
        with col1:
            # Query mode
            query_mode = st.selectbox(
                "Query Mode:",
                ["Standard", "Research", "Clinical", "Educational"],
                help="Adjust response style based on intended use"
            )
            
            # Include references
            include_refs = st.checkbox(
                "Include References",
                value=True,
                help="Include source document references in response"
            )
        
        with col2:
            # Response length
            response_length = st.selectbox(
                "Response Length:",
                ["Concise", "Standard", "Detailed", "Comprehensive"],
                index=1,
                help="Preferred length of generated response"
            )
            
            # Language
            language = st.selectbox(
                "Language:",
                ["English", "Spanish", "French", "German"],
                help="Response language (if supported)"
            )
        
        return {
            "query_mode": query_mode,
            "include_refs": include_refs,
            "response_length": response_length,
            "language": language
        }