"""
Application Configuration Module

Manages configuration settings for the Streamlit RAG demo application.
"""

import os
import logging
from typing import Dict, Any, List
from pathlib import Path
from dataclasses import dataclass

@dataclass
class AppConfig:
    """Configuration settings for the RAG Templates Streamlit app."""
    
    # Application settings
    app_title: str = "RAG Templates Demo"
    app_icon: str = "🤖"
    debug_mode: bool = False
    
    # Pipeline settings
    available_pipelines: List[str] = None
    default_pipeline: str = "BasicRAG"
    max_query_length: int = 1000
    default_top_k: int = 5
    max_top_k: int = 20
    
    # Performance settings
    query_timeout: int = 60  # seconds
    max_concurrent_queries: int = 4
    cache_ttl: int = 300  # 5 minutes
    
    # UI settings
    theme: str = "light"
    sidebar_state: str = "expanded"
    layout: str = "wide"
    
    # Sample queries
    sample_queries: List[str] = None
    
    def __post_init__(self):
        """Initialize derived configuration values."""
        if self.available_pipelines is None:
            self.available_pipelines = [
                "BasicRAG",
                "BasicRerank", 
                "CRAG",
                "GraphRAG"
            ]
        
        if self.sample_queries is None:
            self.sample_queries = [
                "What are the current treatment options for type 2 diabetes mellitus?",
                "How does COVID-19 affect respiratory function and lung capacity?",
                "What are the most effective interventions for managing hypertension?",
                "What are the risk factors and prevention strategies for cardiovascular disease?",
                "How do immunotherapy treatments work in cancer patients?",
                "What are the latest developments in Alzheimer's disease research?",
                "What are the symptoms and diagnostic criteria for depression?",
                "How effective are different pain management strategies for chronic conditions?",
                "What are the mechanisms of antibiotic resistance in bacterial infections?",
                "What nutritional interventions are recommended for metabolic syndrome?",
                "How do vaccines stimulate immune system responses?",
                "What are the genetic factors associated with breast cancer risk?",
                "How does inflammation contribute to autoimmune disease progression?",
                "What are the biomarkers used for early detection of kidney disease?",
                "What surgical techniques are most effective for treating heart valve disorders?"
            ]
        
        # Override with environment variables if available
        self.debug_mode = os.getenv("STREAMLIT_DEBUG", "false").lower() == "true"
        self.query_timeout = int(os.getenv("QUERY_TIMEOUT", str(self.query_timeout)))
        self.max_top_k = int(os.getenv("MAX_TOP_K", str(self.max_top_k)))
        
        # Setup logging based on debug mode
        self._setup_logging()
    
    def _setup_logging(self):
        """Setup application logging."""
        level = logging.DEBUG if self.debug_mode else logging.INFO
        logging.basicConfig(
            level=level,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    def get_pipeline_info(self, pipeline_name: str) -> Dict[str, Any]:
        """Get detailed information about a specific pipeline."""
        pipeline_descriptions = {
            "BasicRAG": {
                "name": "Basic RAG",
                "icon": "📝",
                "description": "Standard RAG with vector similarity search and LLM generation",
                "strengths": ["Fast execution", "Reliable baseline", "Simple architecture"],
                "use_cases": ["General queries", "Quick prototyping", "Baseline comparison"],
                "complexity": "Low",
                "avg_latency": "1-3 seconds"
            },
            "BasicRerank": {
                "name": "Basic RAG with Reranking",
                "icon": "🎯",
                "description": "Enhanced RAG with document reranking for improved relevance",
                "strengths": ["Better relevance", "Improved accuracy", "Context optimization"],
                "use_cases": ["Quality-focused queries", "Complex questions", "Domain-specific tasks"],
                "complexity": "Medium",
                "avg_latency": "2-5 seconds"
            },
            "CRAG": {
                "name": "Corrective RAG",
                "icon": "🔄",
                "description": "Self-correcting RAG with quality assessment and iterative refinement",
                "strengths": ["Self-correcting", "Quality validation", "Robust responses"],
                "use_cases": ["Critical applications", "Fact verification", "High-stakes queries"],
                "complexity": "High",
                "avg_latency": "3-8 seconds"
            },
            "GraphRAG": {
                "name": "Graph RAG",
                "icon": "🕸️",
                "description": "Graph-based RAG using entity relationships and knowledge graphs",
                "strengths": ["Relationship awareness", "Deep insights", "Complex reasoning"],
                "use_cases": ["Research queries", "Multi-hop reasoning", "Knowledge exploration"],
                "complexity": "High",
                "avg_latency": "4-10 seconds"
            }
        }
        
        return pipeline_descriptions.get(pipeline_name, {
            "name": pipeline_name,
            "icon": "❓",
            "description": "Pipeline information not available",
            "strengths": [],
            "use_cases": [],
            "complexity": "Unknown",
            "avg_latency": "Unknown"
        })
    
    def get_query_categories(self) -> Dict[str, List[str]]:
        """Get categorized sample queries."""
        return {
            "General Medical": [
                "What are the current treatment options for type 2 diabetes mellitus?",
                "What are the most effective interventions for managing hypertension?",
                "What are the risk factors and prevention strategies for cardiovascular disease?"
            ],
            "Infectious Diseases": [
                "How does COVID-19 affect respiratory function and lung capacity?",
                "What are the mechanisms of antibiotic resistance in bacterial infections?",
                "How do vaccines stimulate immune system responses?"
            ],
            "Cancer & Oncology": [
                "How do immunotherapy treatments work in cancer patients?",
                "What are the genetic factors associated with breast cancer risk?"
            ],
            "Neurological": [
                "What are the latest developments in Alzheimer's disease research?",
                "What are the symptoms and diagnostic criteria for depression?"
            ],
            "Pain & Chronic Conditions": [
                "How effective are different pain management strategies for chronic conditions?",
                "What nutritional interventions are recommended for metabolic syndrome?"
            ],
            "Diagnostics & Biomarkers": [
                "What are the biomarkers used for early detection of kidney disease?",
                "How does inflammation contribute to autoimmune disease progression?"
            ],
            "Surgical": [
                "What surgical techniques are most effective for treating heart valve disorders?"
            ]
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "app_title": self.app_title,
            "app_icon": self.app_icon,
            "debug_mode": self.debug_mode,
            "available_pipelines": self.available_pipelines,
            "default_pipeline": self.default_pipeline,
            "max_query_length": self.max_query_length,
            "default_top_k": self.default_top_k,
            "max_top_k": self.max_top_k,
            "query_timeout": self.query_timeout,
            "max_concurrent_queries": self.max_concurrent_queries,
            "cache_ttl": self.cache_ttl,
            "theme": self.theme,
            "sidebar_state": self.sidebar_state,
            "layout": self.layout
        }

# Global configuration instance
_config_instance = None

def get_config() -> AppConfig:
    """Get the global configuration instance."""
    global _config_instance
    if _config_instance is None:
        _config_instance = AppConfig()
    return _config_instance

def reset_config():
    """Reset the global configuration instance."""
    global _config_instance
    _config_instance = None