"""
GraphRAG module - Knowledge Graph enhanced RAG
"""

# Import the optimized V3 pipeline as the default
from .pipeline_v3 import GraphRAGPipelineV3 as GraphRAGPipeline

# Also make V2 available for backward compatibility
from .pipeline_v2 import GraphRAGPipelineV2

# Make V3 available explicitly
from .pipeline_v3 import GraphRAGPipelineV3

__all__ = ['GraphRAGPipeline', 'GraphRAGPipelineV2', 'GraphRAGPipelineV3']
