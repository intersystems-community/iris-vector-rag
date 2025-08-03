"""
Pipeline requirements definitions.

This module defines the data and embedding requirements for different RAG pipelines.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any
from dataclasses import dataclass

@dataclass
class EmbeddingRequirement:
    """Defines an embedding requirement for a pipeline."""
    name: str
    table: str
    column: str
    description: str
    required: bool = True


@dataclass
class TableRequirement:
    """Defines a table requirement for a pipeline."""
    name: str
    schema: str
    description: str
    required: bool = True
    min_rows: int = 0
    # Enhanced capabilities for DDL generation
    text_content_type: str = "LONGVARCHAR"  # LONGVARCHAR vs VARCHAR(MAX)
    supports_ifind: bool = False  # Whether table needs iFind support
    supports_vector_search: bool = True  # Whether table needs vector search


class PipelineRequirements(ABC):
    """
    Abstract base class for defining pipeline requirements.
    
    Each pipeline type should inherit from this class and define its specific
    data and embedding requirements.
    """
    
    @property
    @abstractmethod
    def pipeline_name(self) -> str:
        """Name of the pipeline."""
        pass
    
    @property
    @abstractmethod
    def required_tables(self) -> List[TableRequirement]:
        """List of required database tables."""
        pass
    
    @property
    @abstractmethod
    def required_embeddings(self) -> List[EmbeddingRequirement]:
        """List of required embeddings."""
        pass
    
    @property
    def optional_tables(self) -> List[TableRequirement]:
        """List of optional database tables."""
        return []
    
    @property
    def optional_embeddings(self) -> List[EmbeddingRequirement]:
        """List of optional embeddings."""
        return []
    
    def get_all_requirements(self) -> Dict[str, Any]:
        """Get all requirements as a dictionary."""
        return {
            "pipeline_name": self.pipeline_name,
            "required_tables": self.required_tables,
            "required_embeddings": self.required_embeddings,
            "optional_tables": self.optional_tables,
            "optional_embeddings": self.optional_embeddings
        }


class BasicRAGRequirements(PipelineRequirements):
    """Requirements for Basic RAG pipeline."""
    
    @property
    def pipeline_name(self) -> str:
        return "basic_rag"
    
    @property
    def required_tables(self) -> List[TableRequirement]:
        return [
            TableRequirement(
                name="SourceDocuments",
                schema="RAG",
                description="Main document storage table",
                min_rows=1
            )
        ]
    
    @property
    def required_embeddings(self) -> List[EmbeddingRequirement]:
        return [
            EmbeddingRequirement(
                name="document_embeddings",
                table="RAG.SourceDocuments",
                column="embedding",
                description="Document-level embeddings for vector search"
            )
        ]
    
    @property
    def optional_tables(self) -> List[TableRequirement]:
        """Optional tables for enhanced functionality."""
        return [
            TableRequirement(
                name="DocumentChunks",
                schema="RAG",
                description="Document chunks for granular retrieval (optional enhancement)",
                required=False,
                min_rows=0
            )
        ]
    
    @property
    def optional_embeddings(self) -> List[EmbeddingRequirement]:
        """Optional embeddings for enhanced functionality."""
        return [
            EmbeddingRequirement(
                name="chunk_embeddings",
                table="RAG.DocumentChunks",
                column="embedding",
                description="Chunk-level embeddings for enhanced retrieval (optional)",
                required=False
            )
        ]


class ColBERTRequirements(PipelineRequirements):
    """Requirements for ColBERT RAG pipeline."""
    
    @property
    def pipeline_name(self) -> str:
        return "colbert_rag"
    
    @property
    def required_tables(self) -> List[TableRequirement]:
        return [
            TableRequirement(
                name="SourceDocuments",
                schema="RAG",
                description="Main document storage table",
                min_rows=1
            ),
            TableRequirement(
                name="DocumentTokenEmbeddings",
                schema="RAG",
                description="Token-level embeddings for ColBERT",
                min_rows=1
            )
        ]
    
    @property
    def required_embeddings(self) -> List[EmbeddingRequirement]:
        return [
            EmbeddingRequirement(
                name="document_embeddings",
                table="RAG.SourceDocuments",
                column="embedding",
                description="Document-level embeddings for initial retrieval"
            ),
            EmbeddingRequirement(
                name="token_embeddings",
                table="RAG.DocumentTokenEmbeddings",
                column="token_embedding",
                description="Token-level embeddings for fine-grained matching"
            )
        ]
    
    @property
    def optional_tables(self) -> List[TableRequirement]:
        """Optional tables for enhanced functionality."""
        return [
            TableRequirement(
                name="DocumentChunks",
                schema="RAG",
                description="Document chunks for granular retrieval (optional enhancement)",
                required=False,
                min_rows=0
            )
        ]
    
    @property
    def optional_embeddings(self) -> List[EmbeddingRequirement]:
        """Optional embeddings for enhanced functionality."""
        return [
            EmbeddingRequirement(
                name="chunk_embeddings",
                table="RAG.DocumentChunks",
                column="embedding",
                description="Chunk-level embeddings for enhanced retrieval (optional)",
                required=False
            )
        ]


class CRAGRequirements(PipelineRequirements):
    """Requirements for CRAG (Corrective RAG) pipeline."""
    
    @property
    def pipeline_name(self) -> str:
        return "crag"
    
    @property
    def required_tables(self) -> List[TableRequirement]:
        return [
            TableRequirement(
                name="SourceDocuments",
                schema="RAG",
                description="Main document storage table",
                min_rows=1
            )
        ]
    
    @property
    def required_embeddings(self) -> List[EmbeddingRequirement]:
        return [
            EmbeddingRequirement(
                name="document_embeddings",
                table="RAG.SourceDocuments",
                column="embedding",
                description="Document-level embeddings for vector search"
            )
        ]
    
    @property
    def optional_tables(self) -> List[TableRequirement]:
        """Optional tables for enhanced functionality."""
        return [
            TableRequirement(
                name="DocumentChunks",
                schema="RAG",
                description="Document chunks for granular retrieval (optional enhancement)",
                required=False,
                min_rows=0
            )
        ]
    
    @property
    def optional_embeddings(self) -> List[EmbeddingRequirement]:
        """Optional embeddings for enhanced functionality."""
        return [
            EmbeddingRequirement(
                name="chunk_embeddings",
                table="RAG.DocumentChunks",
                column="embedding",
                description="Chunk-level embeddings for enhanced retrieval (optional)",
                required=False
            )
        ]


class HyDERequirements(PipelineRequirements):
    """Requirements for HyDE RAG pipeline."""
    
    @property
    def pipeline_name(self) -> str:
        return "hyde_rag"
    
    @property
    def required_tables(self) -> List[TableRequirement]:
        return [
            TableRequirement(
                name="SourceDocuments",
                schema="RAG",
                description="Main document storage table",
                min_rows=1
            )
        ]
    
    @property
    def required_embeddings(self) -> List[EmbeddingRequirement]:
        return [
            EmbeddingRequirement(
                name="document_embeddings",
                table="RAG.SourceDocuments",
                column="embedding",
                description="Document-level embeddings for vector search"
            )
        ]
    
    @property
    def optional_tables(self) -> List[TableRequirement]:
        """Optional tables for enhanced functionality."""
        return [
            TableRequirement(
                name="DocumentChunks",
                schema="RAG",
                description="Document chunks for granular retrieval (optional enhancement)",
                required=False,
                min_rows=0
            )
        ]
    
    @property
    def optional_embeddings(self) -> List[EmbeddingRequirement]:
        """Optional embeddings for enhanced functionality."""
        return [
            EmbeddingRequirement(
                name="chunk_embeddings",
                table="RAG.DocumentChunks",
                column="embedding",
                description="Chunk-level embeddings for enhanced retrieval (optional)",
                required=False
            )
        ]
    
    @property
    def optional_tables(self) -> List[TableRequirement]:
        """Optional tables for enhanced functionality."""
        return [
            TableRequirement(
                name="DocumentChunks",
                schema="RAG",
                description="Document chunks for granular retrieval (optional enhancement)",
                required=False,
                min_rows=0
            )
        ]
    
    @property
    def optional_embeddings(self) -> List[EmbeddingRequirement]:
        """Optional embeddings for enhanced functionality."""
        return [
            EmbeddingRequirement(
                name="chunk_embeddings",
                table="RAG.DocumentChunks",
                column="embedding",
                description="Chunk-level embeddings for enhanced retrieval (optional)",
                required=False
            )
        ]


class GraphRAGRequirements(PipelineRequirements):
    """Requirements for GraphRAG pipeline."""
    
    @property
    def pipeline_name(self) -> str:
        return "graphrag"
    
    @property
    def required_tables(self) -> List[TableRequirement]:
        return [
            TableRequirement(
                name="SourceDocuments",
                schema="RAG",
                description="Main document storage table",
                min_rows=1
            ),
            TableRequirement(
                name="DocumentEntities",
                schema="RAG",
                description="Entity storage for graph-based retrieval",
                min_rows=1
            ),
            TableRequirement(
                name="EntityRelationships",
                schema="RAG",
                description="Entity relationships for graph traversal",
                min_rows=1
            )
        ]
    
    @property
    def required_embeddings(self) -> List[EmbeddingRequirement]:
        return [
            EmbeddingRequirement(
                name="document_embeddings",
                table="RAG.SourceDocuments",
                column="embedding",
                description="Document-level embeddings for fallback search"
            ),
            EmbeddingRequirement(
                name="entity_embeddings",
                table="RAG.DocumentEntities",
                column="embedding",
                description="Entity embeddings for graph-based matching"
            )
        ]
    
    @property
    def optional_tables(self) -> List[TableRequirement]:
        """Optional tables for enhanced functionality."""
        return [
            TableRequirement(
                name="DocumentChunks",
                schema="RAG",
                description="Document chunks for granular retrieval (optional enhancement)",
                required=False,
                min_rows=0
            )
        ]
    
    @property
    def optional_embeddings(self) -> List[EmbeddingRequirement]:
        """Optional embeddings for enhanced functionality."""
        return [
            EmbeddingRequirement(
                name="chunk_embeddings",
                table="RAG.DocumentChunks",
                column="embedding",
                description="Chunk-level embeddings for enhanced retrieval (optional)",
                required=False
            )
        ]


class HybridIFindRequirements(PipelineRequirements):
    """Requirements for Hybrid IFind RAG pipeline."""
    
    @property
    def pipeline_name(self) -> str:
        return "hybrid_ifind_rag"
    
    @property
    def required_tables(self) -> List[TableRequirement]:
        return [
            TableRequirement(
                name="SourceDocuments",
                schema="RAG",
                description="Main document storage table with IFind support",
                min_rows=1,
                text_content_type="VARCHAR(MAX)",  # Must use VARCHAR for iFind
                supports_ifind=True,  # Enables iFind full-text search
                supports_vector_search=True  # Also supports vector search
            )
        ]
    
    @property
    def required_embeddings(self) -> List[EmbeddingRequirement]:
        return [
            EmbeddingRequirement(
                name="document_embeddings",
                table="RAG.SourceDocuments",
                column="embedding",
                description="Document-level embeddings for vector search"
            )
        ]
    
    @property
    def optional_tables(self) -> List[TableRequirement]:
        """Optional tables for enhanced functionality."""
        return [
            TableRequirement(
                name="DocumentChunks",
                schema="RAG",
                description="Document chunks for granular retrieval (optional enhancement)",
                required=False,
                min_rows=0
            )
        ]
    
    @property
    def optional_embeddings(self) -> List[EmbeddingRequirement]:
        """Optional embeddings for enhanced functionality."""
        return [
            EmbeddingRequirement(
                name="chunk_embeddings",
                table="RAG.DocumentChunks",
                column="embedding",
                description="Chunk-level embeddings for enhanced retrieval (optional)",
                required=False
            )
        ]


class HybridVectorTextRequirements(PipelineRequirements):
    """Requirements for Hybrid Vector-Text RAG pipeline (single table approach)."""
    
    @property
    def pipeline_name(self) -> str:
        return "hybrid_vector_text"
    
    @property
    def required_tables(self) -> List[TableRequirement]:
        return [
            TableRequirement(
                name="SourceDocuments",
                schema="RAG", 
                description="Main document storage table with vector embeddings and text search support",
                min_rows=1,
                text_content_type="VARCHAR(MAX)",  # Must use VARCHAR for iFind
                supports_ifind=True,  # Enables iFind full-text search
                supports_vector_search=True  # Also supports vector search
            )
        ]
    
    @property
    def required_embeddings(self) -> List[EmbeddingRequirement]:
        return [
            EmbeddingRequirement(
                name="document_embeddings",
                table="RAG.SourceDocuments",
                column="embedding",
                description="Document-level embeddings for vector similarity search"
            )
        ]
    
    @property
    def optional_tables(self) -> List[TableRequirement]:
        """Optional tables for enhanced functionality."""
        return []
    
    @property
    def optional_embeddings(self) -> List[EmbeddingRequirement]:
        """Optional embeddings for enhanced functionality."""
        return []


class NodeRAGRequirements(PipelineRequirements):
    """Requirements for NodeRAG pipeline."""
    
    @property
    def pipeline_name(self) -> str:
        return "noderag"
    
    @property
    def required_tables(self) -> List[TableRequirement]:
        return [
            TableRequirement(
                name="SourceDocuments",
                schema="RAG",
                description="Main document storage table for node-based retrieval",
                min_rows=1
            )
        ]
    
    @property
    def required_embeddings(self) -> List[EmbeddingRequirement]:
        return [
            EmbeddingRequirement(
                name="document_embeddings",
                table="RAG.SourceDocuments",
                column="embedding",
                description="Document-level embeddings for initial node identification"
            )
        ]
    
    @property
    def optional_tables(self) -> List[TableRequirement]:
        """Optional tables for enhanced functionality."""
        return [
            TableRequirement(
                name="KnowledgeGraphNodes",
                schema="RAG",
                description="Knowledge graph nodes for advanced graph-based retrieval (optional)",
                required=False,
                min_rows=0
            ),
            TableRequirement(
                name="KnowledgeGraphEdges",
                schema="RAG",
                description="Knowledge graph edges for graph traversal (optional)",
                required=False,
                min_rows=0
            ),
            TableRequirement(
                name="DocumentChunks",
                schema="RAG",
                description="Document chunks for granular retrieval (optional enhancement)",
                required=False,
                min_rows=0
            )
        ]
    
    @property
    def optional_embeddings(self) -> List[EmbeddingRequirement]:
        """Optional embeddings for enhanced functionality."""
        return [
            EmbeddingRequirement(
                name="node_embeddings",
                table="RAG.KnowledgeGraphNodes",
                column="embedding",
                description="Knowledge graph node embeddings for graph-based matching (optional)",
                required=False
            ),
            EmbeddingRequirement(
                name="chunk_embeddings",
                table="RAG.DocumentChunks",
                column="embedding",
                description="Chunk-level embeddings for enhanced retrieval (optional)",
                required=False
            )
        ]


class BasicRAGRerankingRequirements(PipelineRequirements):
    """Requirements for Basic RAG with Reranking pipeline."""
    
    @property
    def pipeline_name(self) -> str:
        return "basic_rerank"
    
    @property
    def required_tables(self) -> List[TableRequirement]:
        return [
            TableRequirement(
                name="SourceDocuments",
                schema="RAG",
                description="Main document storage table",
                min_rows=1
            )
        ]
    
    @property
    def required_embeddings(self) -> List[EmbeddingRequirement]:
        return [
            EmbeddingRequirement(
                name="document_embeddings",
                table="RAG.SourceDocuments",
                column="embedding",
                description="Document-level embeddings for vector search"
            )
        ]
    
    @property
    def optional_tables(self) -> List[TableRequirement]:
        """Optional tables for enhanced functionality."""
        return [
            TableRequirement(
                name="DocumentChunks",
                schema="RAG",
                description="Document chunks for granular retrieval (optional enhancement)",
                required=False,
                min_rows=0
            )
        ]
    
    @property
    def optional_embeddings(self) -> List[EmbeddingRequirement]:
        """Optional embeddings for enhanced functionality."""
        return [
            EmbeddingRequirement(
                name="chunk_embeddings",
                table="RAG.DocumentChunks",
                column="embedding",
                description="Chunk-level embeddings for enhanced retrieval (optional)",
                required=False
            )
        ]


# Registry of pipeline requirements
PIPELINE_REQUIREMENTS_REGISTRY = {
    "basic": BasicRAGRequirements,
    "basic_rerank": BasicRAGRerankingRequirements,
    "colbert": ColBERTRequirements,
    "crag": CRAGRequirements,
    "hyde": HyDERequirements,
    "graphrag": GraphRAGRequirements,
    "hybrid_ifind": HybridIFindRequirements,
    "hybrid_vector_text": HybridVectorTextRequirements,
    "noderag": NodeRAGRequirements
}


def get_pipeline_requirements(pipeline_type: str) -> PipelineRequirements:
    """
    Get requirements for a specific pipeline type.
    
    Args:
        pipeline_type: Type of pipeline (e.g., 'basic', 'colbert')
        
    Returns:
        PipelineRequirements instance
        
    Raises:
        ValueError: If pipeline type is not recognized
    """
    if pipeline_type not in PIPELINE_REQUIREMENTS_REGISTRY:
        available_types = list(PIPELINE_REQUIREMENTS_REGISTRY.keys())
        raise ValueError(f"Unknown pipeline type: {pipeline_type}. Available types: {available_types}")
    
    requirements_class = PIPELINE_REQUIREMENTS_REGISTRY[pipeline_type]
    return requirements_class()