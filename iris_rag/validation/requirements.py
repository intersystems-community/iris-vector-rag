"""
Pipeline requirements definitions.

This module defines the data and embedding requirements for different RAG pipelines.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, List


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

    @property
    @abstractmethod
    def required_tables(self) -> List[TableRequirement]:
        """List of required database tables."""

    @property
    @abstractmethod
    def required_embeddings(self) -> List[EmbeddingRequirement]:
        """List of required embeddings."""

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
            "optional_embeddings": self.optional_embeddings,
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
                min_rows=1,
            )
        ]

    @property
    def required_embeddings(self) -> List[EmbeddingRequirement]:
        return [
            EmbeddingRequirement(
                name="embedding",
                table="RAG.SourceDocuments",
                column="embedding",
                description="Document-level embeddings for vector search",
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
                min_rows=0,
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
                required=False,
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
                min_rows=1,
            )
        ]

    @property
    def required_embeddings(self) -> List[EmbeddingRequirement]:
        return [
            EmbeddingRequirement(
                name="document_embeddings",
                table="RAG.SourceDocuments",
                column="embedding",
                description="Document-level embeddings for vector search",
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
                min_rows=0,
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
                required=False,
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
                min_rows=1,
            )
        ]

    @property
    def required_embeddings(self) -> List[EmbeddingRequirement]:
        return [
            EmbeddingRequirement(
                name="document_embeddings",
                table="RAG.SourceDocuments",
                column="embedding",
                description="Document-level embeddings for vector search",
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
                min_rows=0,
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
                required=False,
            )
        ]


class GraphRAGRequirements(PipelineRequirements):
    """
    Requirements for GraphRAG pipeline.

    This defines Phase 1 requirements (SQL tables) while maintaining compatibility
    with the planned Phase 2 IRIS Globals Optimization for high-performance
    pointer chasing operations.

    Future Enhancement Roadmap:
    - Phase 2: IRIS globals optimization with direct pointer chasing
    - ObjectScript integration for graph traversal operations
    - Globals structure: ^RAG.Entity, ^RAG.Graph, ^RAG.Path
    - See: docs/architecture/GRAPHRAG_KNOWLEDGE_GRAPH_ARCHITECTURE.md
    """

    @property
    def pipeline_name(self) -> str:
        return "graphrag"

    @property
    def required_tables(self) -> List[TableRequirement]:
        return [
            TableRequirement(
                name="SourceDocuments",
                schema="RAG",
                description="Document storage with embeddings",
                min_rows=1,
                supports_vector_search=True,
            ),
            TableRequirement(
                name="Entities",
                schema="RAG",
                description="Extracted entities with embeddings",
                min_rows=1,  # Require minimum entities for graph queries
                supports_vector_search=True,
            ),
            TableRequirement(
                name="EntityRelationships",
                schema="RAG",
                description="Entity relationships for graph traversal",
                min_rows=1,  # Require minimum relationships for connectivity
                supports_vector_search=False,  # Relationships don't need vector search
            ),
            TableRequirement(
                name="Communities",
                schema="RAG",
                description="Entity communities for hierarchical summarization",
                min_rows=0,  # Optional - can work without communities initially
                required=False,  # Optional table
                supports_vector_search=False,
            ),
        ]

    @property
    def required_embeddings(self) -> List[EmbeddingRequirement]:
        return [
            EmbeddingRequirement(
                name="document_embeddings",
                table="RAG.SourceDocuments",
                column="embedding",
                description="Document-level embeddings for vector search",
            )
        ]

    @property
    def optional_embeddings(self) -> List[EmbeddingRequirement]:
        """Optional embeddings for enhanced functionality."""
        return [
            EmbeddingRequirement(
                name="entity_embeddings",
                table="RAG.Entities",
                column="entity_embeddings",
                description="Entity-level embeddings for semantic entity search (optional)",
                required=False,
            )
        ]


class HybridGraphRAGRequirements(PipelineRequirements):
    """
    Requirements for HybridGraphRAG pipeline.

    Inherits GraphRAG requirements but with enhanced capabilities when iris_graph_core
    is available. Provides graceful fallback to GraphRAG behavior when enhanced
    features are unavailable.
    """

    @property
    def pipeline_name(self) -> str:
        return "hybrid_graphrag"

    @property
    def required_tables(self) -> List[TableRequirement]:
        """Same core requirements as GraphRAG - ensures graceful fallback."""
        return [
            TableRequirement(
                name="SourceDocuments",
                schema="RAG",
                description="Document storage with embeddings",
                min_rows=1,
                supports_vector_search=True,
            ),
            TableRequirement(
                name="Entities",
                schema="RAG",
                description="Extracted entities with embeddings",
                min_rows=1,
                supports_vector_search=True,
            ),
            TableRequirement(
                name="EntityRelationships",
                schema="RAG",
                description="Entity relationships for graph traversal",
                min_rows=1,
                supports_vector_search=False,
            ),
        ]

    @property
    def required_embeddings(self) -> List[EmbeddingRequirement]:
        """Same core embeddings as GraphRAG."""
        return [
            EmbeddingRequirement(
                name="document_embeddings",
                table="RAG.SourceDocuments",
                column="embedding",
                description="Document-level embeddings for vector search",
            )
        ]

    @property
    def optional_tables(self) -> List[TableRequirement]:
        """Optional iris_graph_core enhancement tables."""
        return [
            TableRequirement(
                name="kg_NodeEmbeddings_optimized",
                schema="SQLUSER",
                description="Optimized HNSW vector indexes (iris_graph_core)",
                required=False,
                supports_vector_search=True,
            ),
            TableRequirement(
                name="rdf_labels",
                schema="SQLUSER",
                description="RDF label mappings (iris_graph_core)",
                required=False,
                supports_vector_search=False,
            ),
            TableRequirement(
                name="rdf_props",
                schema="SQLUSER",
                description="RDF property mappings (iris_graph_core)",
                required=False,
                supports_vector_search=False,
            ),
            TableRequirement(
                name="rdf_edges",
                schema="SQLUSER",
                description="RDF edge mappings (iris_graph_core)",
                required=False,
                supports_vector_search=False,
            ),
        ]

    @property
    def optional_embeddings(self) -> List[EmbeddingRequirement]:
        """Optional enhanced embeddings for hybrid capabilities."""
        return [
            EmbeddingRequirement(
                name="entity_embeddings",
                table="RAG.Entities",
                column="entity_embeddings",
                description="Entity-level embeddings for semantic entity search",
                required=False,
            ),
            EmbeddingRequirement(
                name="hnsw_optimized_embeddings",
                table="SQLUSER.kg_NodeEmbeddings_optimized",
                column="embedding",
                description="HNSW-optimized embeddings (iris_graph_core enhancement)",
                required=False,
            ),
        ]


class PyLateColBERTRequirements(PipelineRequirements):
    """Requirements for PyLate ColBERT pipeline."""

    @property
    def pipeline_name(self) -> str:
        return "pylate_colbert"

    @property
    def required_tables(self) -> List[TableRequirement]:
        return [
            TableRequirement(
                name="SourceDocuments",
                schema="RAG",
                description="Main document storage table",
                min_rows=1,
            )
        ]

    @property
    def required_embeddings(self) -> List[EmbeddingRequirement]:
        return [
            EmbeddingRequirement(
                name="document_embeddings",
                table="RAG.SourceDocuments",
                column="embedding",
                description="Document-level embeddings for vector search",
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
                min_rows=0,
            )
        ]

    @property
    def optional_embeddings(self) -> List[EmbeddingRequirement]:
        """Optional embeddings for ColBERT late interaction."""
        return [
            EmbeddingRequirement(
                name="chunk_embeddings",
                table="RAG.DocumentChunks",
                column="embedding",
                description="Chunk-level embeddings for reranking (optional)",
                required=False,
            )
        ]


# Registry of pipeline requirements
PIPELINE_REQUIREMENTS_REGISTRY = {
    "basic": BasicRAGRequirements,
    "basic_rerank": BasicRAGRerankingRequirements,
    "pylate_colbert": PyLateColBERTRequirements,
    "crag": CRAGRequirements,
    "graphrag": HybridGraphRAGRequirements,  # Map to HybridGraphRAGRequirements since factory creates HybridGraphRAGPipeline
    "hybrid_graphrag": HybridGraphRAGRequirements,
}


def get_pipeline_requirements(pipeline_type: str) -> PipelineRequirements:
    """
    Get requirements for a specific pipeline type.

    Args:
        pipeline_type: Type of pipeline (e.g., 'basic')

    Returns:
        PipelineRequirements instance

    Raises:
        ValueError: If pipeline type is not recognized
    """
    if pipeline_type not in PIPELINE_REQUIREMENTS_REGISTRY:
        available_types = list(PIPELINE_REQUIREMENTS_REGISTRY.keys())
        raise ValueError(
            f"Unknown pipeline type: {pipeline_type}. Available types: {available_types}"
        )

    requirements_class = PIPELINE_REQUIREMENTS_REGISTRY[pipeline_type]
    return requirements_class()
