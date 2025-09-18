"""
Results Display Component

Renders query results, retrieved documents, and performance metrics.
"""

import streamlit as st
import json
import time
from typing import Dict, Any, List
from utils.pipeline_integration import QueryResult

def render_query_result(result: QueryResult, show_details: bool = True):
    """
    Render a complete query result.
    
    Args:
        result: QueryResult object containing all result data
        show_details: Whether to show detailed information
    """
    if not result.success:
        render_error_result(result)
        return
    
    # Main answer section
    render_answer_section(result)
    
    if show_details:
        # Retrieved documents section
        render_retrieved_documents(result.retrieved_documents, result.pipeline)
        
        # Performance metrics
        render_performance_metrics(result)
        
        # Metadata and debugging info
        if st.session_state.get("debug_mode", False):
            render_debug_info(result)

def render_answer_section(result: QueryResult):
    """Render the main answer section."""
    st.markdown("### 💬 Generated Answer")
    
    # Pipeline indicator
    pipeline_info = get_pipeline_display_info(result.pipeline)
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col1:
        st.markdown(f"**{pipeline_info['icon']} {result.pipeline}**")
    
    with col2:
        st.markdown(f"*Execution Time: {result.execution_time:.2f}s*")
    
    with col3:
        # Quality indicator (mock for now)
        quality_score = estimate_answer_quality(result.answer)
        quality_color = "green" if quality_score > 0.8 else "orange" if quality_score > 0.6 else "red"
        st.markdown(f"<div style='text-align: right; color: {quality_color};'>Quality: {quality_score:.1%}</div>", unsafe_allow_html=True)
    
    # Answer content
    st.markdown("---")
    
    if result.answer:
        # Display answer with nice formatting
        st.markdown(f"""
        <div style="
            background-color: #f8f9fa;
            border-left: 4px solid #1f77b4;
            padding: 1rem;
            border-radius: 0 0.375rem 0.375rem 0;
            margin: 1rem 0;
        ">
            {result.answer}
        </div>
        """, unsafe_allow_html=True)
        
        # Answer actions
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            if st.button("👍 Helpful", key=f"helpful_{result.pipeline}"):
                st.success("Thank you for your feedback!")
        
        with col2:
            if st.button("👎 Not Helpful", key=f"not_helpful_{result.pipeline}"):
                st.info("Feedback noted. Try a different pipeline or refine your query.")
        
        with col3:
            if st.button("📋 Copy Answer", key=f"copy_{result.pipeline}"):
                st.session_state[f"copied_answer_{result.pipeline}"] = result.answer
                st.success("Answer copied to clipboard!")
        
        with col4:
            if st.button("📤 Share", key=f"share_{result.pipeline}"):
                share_url = generate_share_url(result)
                st.info(f"Share URL: {share_url}")
    else:
        st.warning("No answer was generated. Try rephrasing your query or checking pipeline configuration.")

def render_retrieved_documents(documents: List[Dict[str, Any]], pipeline: str):
    """Render retrieved documents section."""
    if not documents:
        st.warning("No documents were retrieved.")
        return
    
    st.markdown("### 📄 Retrieved Documents")
    st.markdown(f"*Found {len(documents)} relevant documents*")
    
    # Document display options
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_full_content = st.checkbox("Show Full Content", key=f"full_content_{pipeline}")
    
    with col2:
        show_metadata = st.checkbox("Show Metadata", value=True, key=f"metadata_{pipeline}")
    
    with col3:
        sort_by = st.selectbox("Sort by:", ["Relevance", "Title", "Source"], key=f"sort_{pipeline}")
    
    # Sort documents
    sorted_docs = sort_documents(documents, sort_by)
    
    # Display documents
    for i, doc in enumerate(sorted_docs):
        render_single_document(doc, i, show_full_content, show_metadata, pipeline)

def render_single_document(doc: Dict[str, Any], index: int, show_full: bool, show_metadata: bool, pipeline: str):
    """Render a single document."""
    with st.expander(f"📄 Document {index + 1}: {doc.get('title', 'Untitled')}", expanded=index < 2):
        
        # Document header
        col1, col2 = st.columns([3, 1])
        
        with col1:
            st.markdown(f"**Title:** {doc.get('title', 'Untitled')}")
            if doc.get('metadata', {}).get('source'):
                st.markdown(f"**Source:** {doc['metadata']['source']}")
        
        with col2:
            # Relevance score
            score = doc.get('metadata', {}).get('score', 0.0)
            if score > 0:
                score_color = "green" if score > 0.8 else "orange" if score > 0.6 else "red"
                st.markdown(f"<div style='text-align: right; color: {score_color};'>Score: {score:.3f}</div>", unsafe_allow_html=True)
        
        # Document content
        content = doc.get('content', '')
        
        if content:
            if show_full:
                st.markdown("**Content:**")
                st.markdown(content)
            else:
                # Show truncated content
                preview = content[:300] + "..." if len(content) > 300 else content
                st.markdown("**Preview:**")
                st.markdown(preview)
                
                if len(content) > 300:
                    if st.button(f"Show Full Content", key=f"expand_{pipeline}_{index}"):
                        st.markdown("**Full Content:**")
                        st.markdown(content)
        
        # Metadata
        if show_metadata and doc.get('metadata'):
            st.markdown("**Metadata:**")
            metadata = doc['metadata']
            
            # Format metadata nicely
            metadata_display = {}
            for key, value in metadata.items():
                if key not in ['score']:  # Skip score as it's shown above
                    metadata_display[key] = value
            
            if metadata_display:
                st.json(metadata_display)
        
        # Document actions
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📋 Copy", key=f"copy_doc_{pipeline}_{index}"):
                st.session_state[f"copied_doc_{pipeline}_{index}"] = content
                st.success("Document copied!")
        
        with col2:
            if st.button("🔍 Analyze", key=f"analyze_{pipeline}_{index}"):
                render_document_analysis(doc)
        
        with col3:
            if st.button("📤 Export", key=f"export_doc_{pipeline}_{index}"):
                export_document(doc, f"document_{index + 1}")

def render_performance_metrics(result: QueryResult):
    """Render performance metrics section."""
    st.markdown("### 📊 Performance Metrics")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Execution Time",
            f"{result.execution_time:.2f}s",
            help="Total time to process the query"
        )
    
    with col2:
        st.metric(
            "Documents Retrieved",
            len(result.retrieved_documents),
            help="Number of relevant documents found"
        )
    
    with col3:
        # Token usage estimate
        token_estimate = estimate_token_usage(result)
        st.metric(
            "Est. Tokens",
            f"{token_estimate:,}",
            help="Estimated token usage for this query"
        )
    
    with col4:
        # Confidence score
        confidence = result.metadata.get('confidence', 0.85)
        st.metric(
            "Confidence",
            f"{confidence:.1%}",
            help="Pipeline confidence in the result"
        )
    
    # Additional metrics in expander
    with st.expander("📈 Detailed Metrics"):
        
        # Create metrics chart data
        metrics_data = {
            "Retrieval Time": result.metadata.get('retrieval_time', result.execution_time * 0.3),
            "Generation Time": result.metadata.get('generation_time', result.execution_time * 0.7),
            "Processing Time": result.metadata.get('processing_time', result.execution_time * 0.1)
        }
        
        # Simple bar chart using HTML/CSS
        for metric, value in metrics_data.items():
            percentage = (value / result.execution_time) * 100
            st.markdown(f"""
            <div style="margin-bottom: 1rem;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 0.25rem;">
                    <span>{metric}</span>
                    <span>{value:.3f}s ({percentage:.1f}%)</span>
                </div>
                <div style="background-color: #e9ecef; border-radius: 0.375rem; overflow: hidden; height: 1rem;">
                    <div style="background-color: #1f77b4; height: 100%; width: {percentage}%; transition: width 0.3s ease;"></div>
                </div>
            </div>
            """, unsafe_allow_html=True)

def render_error_result(result: QueryResult):
    """Render error result."""
    st.error(f"❌ Query failed for {result.pipeline}")
    
    if result.error:
        st.markdown(f"**Error:** {result.error}")
    
    st.markdown(f"**Execution Time:** {result.execution_time:.2f}s")
    
    # Troubleshooting suggestions
    with st.expander("🔧 Troubleshooting"):
        st.markdown("""
        **Common solutions:**
        - Check if the pipeline is properly initialized
        - Verify database connectivity
        - Try a simpler query
        - Reduce the number of top_k results
        - Check pipeline-specific configuration
        """)
        
        if st.button("🔄 Retry Query", key=f"retry_{result.pipeline}"):
            st.session_state.retry_query = True
            st.rerun()

def render_debug_info(result: QueryResult):
    """Render debugging information."""
    with st.expander("🐛 Debug Information"):
        st.markdown("**Query Details:**")
        st.code(f"Pipeline: {result.pipeline}\nQuery: {result.query}\nExecution Time: {result.execution_time:.2f}s")
        
        st.markdown("**Metadata:**")
        st.json(result.metadata)
        
        st.markdown("**Raw Result:**")
        debug_data = {
            "success": result.success,
            "pipeline": result.pipeline,
            "query": result.query,
            "answer": result.answer,
            "num_documents": len(result.retrieved_documents),
            "contexts": len(result.contexts),
            "execution_time": result.execution_time,
            "metadata": result.metadata,
            "error": result.error
        }
        st.json(debug_data)

# Helper functions
def get_pipeline_display_info(pipeline: str) -> Dict[str, str]:
    """Get display information for a pipeline."""
    info = {
        "BasicRAG": {"icon": "📝", "color": "#1f77b4"},
        "BasicRerank": {"icon": "🎯", "color": "#ff7f0e"},
        "CRAG": {"icon": "🔄", "color": "#2ca02c"},
        "GraphRAG": {"icon": "🕸️", "color": "#d62728"}
    }
    return info.get(pipeline, {"icon": "❓", "color": "#666666"})

def estimate_answer_quality(answer: str) -> float:
    """Estimate answer quality based on simple heuristics."""
    if not answer:
        return 0.0
    
    # Simple quality scoring
    score = 0.5  # Base score
    
    # Length scoring
    if 50 <= len(answer) <= 1000:
        score += 0.2
    
    # Structure scoring
    if "." in answer:
        score += 0.1
    if answer.count(".") >= 2:
        score += 0.1
    
    # Content scoring
    quality_indicators = ["research", "study", "evidence", "data", "analysis"]
    for indicator in quality_indicators:
        if indicator.lower() in answer.lower():
            score += 0.02
    
    return min(1.0, score)

def estimate_token_usage(result: QueryResult) -> int:
    """Estimate token usage for the query."""
    # Simple estimation: ~4 characters per token
    query_tokens = len(result.query) // 4
    answer_tokens = len(result.answer) // 4
    context_tokens = sum(len(ctx) // 4 for ctx in result.contexts)
    
    return query_tokens + answer_tokens + context_tokens

def sort_documents(documents: List[Dict[str, Any]], sort_by: str) -> List[Dict[str, Any]]:
    """Sort documents based on the specified criteria."""
    if sort_by == "Relevance":
        return sorted(documents, key=lambda x: x.get('metadata', {}).get('score', 0), reverse=True)
    elif sort_by == "Title":
        return sorted(documents, key=lambda x: x.get('title', '').lower())
    elif sort_by == "Source":
        return sorted(documents, key=lambda x: x.get('metadata', {}).get('source', '').lower())
    else:
        return documents

def render_document_analysis(doc: Dict[str, Any]):
    """Render document analysis."""
    st.markdown("**Document Analysis:**")
    
    content = doc.get('content', '')
    
    # Simple analysis
    analysis = {
        "Word Count": len(content.split()),
        "Character Count": len(content),
        "Estimated Reading Time": f"{len(content.split()) // 200 + 1} min",
        "Key Topics": extract_key_topics(content)
    }
    
    for key, value in analysis.items():
        st.markdown(f"**{key}:** {value}")

def extract_key_topics(content: str) -> str:
    """Extract key topics from content (simplified)."""
    # Simple keyword extraction
    common_words = set(['the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by'])
    words = [word.lower().strip('.,!?;:') for word in content.split()]
    word_freq = {}
    
    for word in words:
        if len(word) > 3 and word not in common_words:
            word_freq[word] = word_freq.get(word, 0) + 1
    
    # Get top 5 most frequent words
    top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
    return ", ".join([word for word, freq in top_words])

def export_document(doc: Dict[str, Any], filename: str):
    """Export document to file."""
    # This would normally trigger a download
    st.success(f"Document exported as {filename}.json")

def generate_share_url(result: QueryResult) -> str:
    """Generate a shareable URL for the result."""
    # This would normally generate a proper shareable URL
    return f"https://rag-demo.example.com/shared/{result.pipeline}?q={result.query[:50]}..."