"""
IRIS Global GraphRAG Pipeline

Integrates the IRIS-Global-GraphRAG project's core functionality into our RAG framework,
providing academic paper retrieval with IRIS globals-based graph storage and interactive
visualization capabilities.

This pipeline wraps the core functions from the external IRIS-Global-GraphRAG project
while leveraging our framework's enterprise features like configuration management,
validation, and monitoring.
"""

import logging
import os
import sys
import time
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from ..config.manager import ConfigurationManager
from ..core.base import RAGPipeline
from ..core.connection import ConnectionManager
from ..core.exceptions import PipelineConfigurationError, RAGException
from ..core.models import Document
from ..embeddings.manager import EmbeddingManager

logger = logging.getLogger(__name__)


class IRISGlobalGraphRAGException(RAGException):
    """Exceptions specific to IRIS Global GraphRAG pipeline."""

    pass


class IRISGlobalGraphRAGPipeline(RAGPipeline):
    """
    IRIS Global GraphRAG Pipeline with Globals-based Graph Storage.

    This pipeline integrates the IRIS-Global-GraphRAG project's functionality
    for academic paper retrieval using IRIS globals for graph storage and
    SQL tables for vector embeddings.

    Features:
    - IRIS Globals for graph storage (^GraphContent, ^GraphRelations)
    - HNSW vector search for paper embeddings
    - Interactive graph visualization
    - Side-by-side comparison capabilities
    - Academic paper entity extraction
    """

    def __init__(
        self,
        connection_manager: Optional[ConnectionManager] = None,
        config_manager: Optional[ConfigurationManager] = None,
        llm_func: Optional[Callable[[str], str]] = None,
        vector_store=None,
        embedding_manager: Optional[EmbeddingManager] = None,
        **kwargs,
    ):
        super().__init__()

        self.connection_manager = connection_manager
        self.config_manager = config_manager or ConfigurationManager()
        self.llm_func = llm_func
        self.vector_store = vector_store
        self.embedding_manager = embedding_manager

        # IRIS Global GraphRAG specific components
        self.iris_engine = None
        self.irispy = None
        self.emb_model = None
        self.global_graphrag_module = None

        # Configuration
        self.config = self._load_configuration()

        # Initialize pipeline
        self._initialize_pipeline()

    def _load_configuration(self) -> Dict[str, Any]:
        """Load IRIS Global GraphRAG specific configuration."""
        default_config = {
            "iris_global_graphrag": {
                "enabled": True,
                "project_path": None,  # Auto-discover or from config
                "graph_content_global": "GraphContent",
                "graph_relations_global": "GraphRelations",
                "embedding_model": "sentence-transformers/all-MiniLM-L6-v2",
                "vector_dimension": 384,
                "top_k": 5,
                "enable_visualization": True,
                "enable_comparison_ui": True,
            }
        }

        if self.config_manager:
            user_config = self.config_manager.get("iris_global_graphrag", {})
            default_config["iris_global_graphrag"].update(user_config)

        return default_config["iris_global_graphrag"]

    def _discover_iris_global_graphrag_path(self) -> Optional[Path]:
        """
        Discover the IRIS-Global-GraphRAG project path.

        Priority order:
        1. Configuration setting
        2. Environment variable IRIS_GLOBAL_GRAPHRAG_PATH
        3. Sibling directory search
        """
        # Priority 1: Configuration
        if self.config.get("project_path"):
            config_path = Path(self.config["project_path"]).expanduser().resolve()
            if self._validate_iris_global_graphrag_path(config_path):
                return config_path

        # Priority 2: Environment variable
        env_path = os.getenv("IRIS_GLOBAL_GRAPHRAG_PATH")
        if env_path:
            env_path = Path(env_path).expanduser().resolve()
            if self._validate_iris_global_graphrag_path(env_path):
                return env_path

        # Priority 3: Sibling directory search
        current_dir = Path(
            __file__
        ).parent.parent.parent.parent  # Go up to workspace level
        candidates = [
            current_dir / "IRIS-Global-GraphRAG",
            current_dir / "iris-global-graphrag",
            current_dir.parent / "IRIS-Global-GraphRAG",
            current_dir.parent / "iris-global-graphrag",
        ]

        for candidate in candidates:
            if self._validate_iris_global_graphrag_path(candidate):
                logger.info(f"Auto-discovered IRIS-Global-GraphRAG at: {candidate}")
                return candidate

        return None

    def _validate_iris_global_graphrag_path(self, path: Path) -> bool:
        """Validate that path contains IRIS-Global-GraphRAG project."""
        if not path.exists() or not path.is_dir():
            return False

        # Check for key files
        required_files = ["app/iris_db.py", "app/app.py", "README.md"]

        return all((path / file).exists() for file in required_files)

    def _initialize_pipeline(self):
        """Initialize the IRIS Global GraphRAG pipeline components."""
        try:
            # Discover project path
            project_path = self._discover_iris_global_graphrag_path()
            if not project_path:
                raise IRISGlobalGraphRAGException(
                    "IRIS-Global-GraphRAG project not found. "
                    "Please set project_path in configuration or IRIS_GLOBAL_GRAPHRAG_PATH environment variable."
                )

            # Add project to Python path
            app_path = str(project_path / "app")
            if app_path not in sys.path:
                sys.path.insert(0, app_path)

            # Import IRIS Global GraphRAG modules
            try:
                import iris_db as global_graphrag

                self.global_graphrag_module = global_graphrag
                logger.info("✅ Successfully imported IRIS-Global-GraphRAG modules")
            except ImportError as e:
                raise IRISGlobalGraphRAGException(
                    f"Failed to import IRIS-Global-GraphRAG modules: {e}"
                )

            # Initialize IRIS connections
            self._initialize_iris_connections()

            # Initialize embedding model
            self._initialize_embedding_model()

            # Validate schema
            self._validate_schema()

            logger.info("✅ IRIS Global GraphRAG pipeline initialized successfully")

        except Exception as e:
            logger.error(f"Failed to initialize IRIS Global GraphRAG pipeline: {e}")
            raise IRISGlobalGraphRAGException(f"Pipeline initialization failed: {e}")

    def _initialize_iris_connections(self):
        """Initialize IRIS database connections."""
        try:
            # Use our connection manager if available
            if self.connection_manager:
                connection = self.connection_manager.get_connection()
                self.iris_engine = self.global_graphrag_module.get_sqlalchemy_engine()
                self.irispy = self.global_graphrag_module.get_irispy()
            else:
                # Fall back to direct connection
                self.iris_engine = self.global_graphrag_module.get_sqlalchemy_engine()
                self.irispy = self.global_graphrag_module.get_irispy()

            logger.info("✅ IRIS connections established")

        except Exception as e:
            raise IRISGlobalGraphRAGException(
                f"Failed to establish IRIS connections: {e}"
            )

    def _initialize_embedding_model(self):
        """Initialize the embedding model using cached unified service."""
        try:
            if self.embedding_manager:
                # Use our framework's embedding manager (preferred)
                self.emb_model = self.embedding_manager
                logger.info("✅ Using framework embedding manager")
            else:
                # Use unified caching service to prevent redundant model loads
                from ..embeddings.manager import _get_cached_sentence_transformer

                model_name = self.config.get(
                    "embedding_model", "sentence-transformers/all-MiniLM-L6-v2"
                )
                device = self.config.get("device", "cpu")

                # Use cached model (eliminates 3-5 second load on repeated access)
                self.emb_model = _get_cached_sentence_transformer(model_name, device)
                logger.info(f"✅ Using cached embedding model: {model_name} on {device}")

        except Exception as e:
            raise IRISGlobalGraphRAGException(
                f"Failed to initialize embedding model: {e}"
            )

    def _validate_schema(self):
        """Validate that required database schema exists."""
        try:
            # Ensure paper_content table exists
            self.global_graphrag_module.ensure_schema(self.iris_engine)

            # Try to create HNSW index (may already exist)
            try:
                self.global_graphrag_module.create_hnsw_index(self.iris_engine)
            except Exception:
                # Index might already exist
                pass

            logger.info("✅ Database schema validated")

        except Exception as e:
            logger.warning(f"Schema validation warning: {e}")

    def load_documents(self, documents: List[Document], **kwargs) -> bool:
        """
        Load documents into IRIS Global GraphRAG storage.

        Args:
            documents: List of documents to load
            **kwargs: Additional parameters

        Returns:
            bool: True if successful
        """
        try:
            if not self.global_graphrag_module:
                raise IRISGlobalGraphRAGException("Pipeline not properly initialized")

            for doc in documents:
                # Generate embedding
                if hasattr(self.emb_model, "encode"):
                    # sentence-transformers style
                    embedding = self.emb_model.encode(
                        doc.page_content, normalize_embeddings=True
                    ).tolist()
                else:
                    # Our framework's embedding manager style
                    embedding = self.emb_model.embed_text(doc.page_content)

                # Store in SQL table (paper_content)
                self._store_document_sql(doc, embedding)

                # Store graph content in globals if available
                if hasattr(doc, "metadata") and "entities" in doc.metadata:
                    self._store_graph_content(doc)
                    self._store_graph_relations(doc)

            logger.info(
                f"✅ Loaded {len(documents)} documents into IRIS Global GraphRAG"
            )
            return True

        except Exception as e:
            logger.error(f"Failed to load documents: {e}")
            raise IRISGlobalGraphRAGException(f"Document loading failed: {e}")

    def _store_document_sql(self, doc: Document, embedding: List[float]):
        """Store document in SQL table for vector search."""
        from sqlalchemy import text

        # Extract metadata
        metadata = doc.metadata or {}
        title = metadata.get("title", doc.id)
        abstract = doc.page_content[:2000]  # Truncate for abstract field
        url = metadata.get("url", "")
        published = metadata.get("published", "")
        authors = metadata.get("authors", "")
        combined = f"{title} {abstract}"

        # Insert into paper_content table
        sql = text(
            """
            INSERT INTO paper_content
            (docid, title, abstract, url, published, authors, combined, paper_vector)
            VALUES (?, ?, ?, ?, ?, ?, ?, TO_VECTOR(?))
        """
        )

        with self.iris_engine.connect() as conn:
            with conn.begin():
                conn.execute(
                    sql,
                    [
                        doc.id,
                        title,
                        abstract,
                        url,
                        published,
                        authors,
                        combined,
                        str(embedding),
                    ],
                )

    def _store_graph_content(self, doc: Document):
        """Store graph content in IRIS globals."""
        metadata = doc.metadata or {}

        global_name = self.config.get("graph_content_global", "GraphContent")

        self.global_graphrag_module.upsert_graph_content(
            self.irispy,
            doc.id,
            metadata.get("title", doc.id),
            doc.page_content[:2000],  # Abstract
            metadata.get("url", ""),
            metadata.get("published", ""),
            metadata.get("authors", ""),
            global_name,
        )

    def _store_graph_relations(self, doc: Document):
        """Store graph relations in IRIS globals."""
        metadata = doc.metadata or {}
        entities = metadata.get("entities", [])
        relationships = metadata.get("relationships", [])

        global_name = self.config.get("graph_relations_global", "GraphRelations")

        # Store relationships
        for rel in relationships:
            if "source" in rel and "target" in rel:
                self.global_graphrag_module.upsert_graph_relations(
                    self.irispy,
                    doc.id,
                    rel["source"],
                    rel.get("source_type", "Entity"),
                    rel["target"],
                    rel.get("target_type", "Entity"),
                    rel.get("relation", "RELATES_TO"),
                    global_name,
                )

    def query(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Execute a query using IRIS Global GraphRAG.

        Args:
            query: The user query
            **kwargs: Additional parameters including:
                - top_k: Number of results to return
                - mode: 'rag' or 'graphrag'
                - enable_visualization: Whether to include graph data

        Returns:
            Dict containing answer and optional graph data
        """
        start_time = time.time()

        try:
            if not self.global_graphrag_module:
                raise IRISGlobalGraphRAGException("Pipeline not properly initialized")

            # Get parameters
            top_k = kwargs.get("top_k", self.config.get("top_k", 5))
            mode = kwargs.get("mode", "graphrag")
            enable_visualization = kwargs.get(
                "enable_visualization", self.config.get("enable_visualization", True)
            )

            if mode == "rag":
                # Simple RAG without graph
                answer = self.global_graphrag_module.ask_question_rag(
                    query, self.iris_engine, self.emb_model, top_k
                )

                response = {
                    "answer": answer,
                    "mode": "rag",
                    "query": query,
                    "processing_time": time.time() - start_time,
                }

            else:
                # GraphRAG with graph visualization
                answer = self.global_graphrag_module.ask_question_graphrag(
                    query, self.iris_engine, self.emb_model, self.irispy, top_k
                )

                response = {
                    "answer": answer,
                    "mode": "graphrag",
                    "query": query,
                    "processing_time": time.time() - start_time,
                }

                # Add visualization data if requested
                if enable_visualization:
                    try:
                        combined_results = (
                            self.global_graphrag_module.prepare_combined_results(
                                query,
                                self.iris_engine,
                                self.emb_model,
                                self.irispy,
                                top_k,
                            )
                        )

                        # Combine graphs for visualization
                        graph_data = self.global_graphrag_module.combine_graphs(
                            combined_results["graphs"]
                        )

                        response.update(
                            {
                                "graph_data": graph_data,
                                "retrieved_papers": combined_results["papers"],
                                "doc_ids": combined_results["doc_ids"],
                            }
                        )

                    except Exception as e:
                        logger.warning(f"Failed to generate visualization data: {e}")

            return response

        except Exception as e:
            logger.error(f"Query failed: {e}")
            raise IRISGlobalGraphRAGException(f"Query execution failed: {e}")

    def get_pipeline_info(self) -> Dict[str, Any]:
        """Get information about the pipeline."""
        return {
            "name": "IRIS Global GraphRAG",
            "description": "Academic paper retrieval with IRIS globals-based graph storage",
            "version": "1.0.0",
            "features": [
                "IRIS Globals graph storage",
                "HNSW vector search",
                "Interactive graph visualization",
                "Academic paper entity extraction",
                "Side-by-side comparison capabilities",
            ],
            "configuration": {
                "embedding_model": self.config.get("embedding_model"),
                "vector_dimension": self.config.get("vector_dimension"),
                "top_k": self.config.get("top_k"),
                "graph_content_global": self.config.get("graph_content_global"),
                "graph_relations_global": self.config.get("graph_relations_global"),
                "visualization_enabled": self.config.get("enable_visualization"),
                "comparison_ui_enabled": self.config.get("enable_comparison_ui"),
            },
            "status": {
                "initialized": self.global_graphrag_module is not None,
                "iris_engine_ready": self.iris_engine is not None,
                "irispy_ready": self.irispy is not None,
                "embedding_model_ready": self.emb_model is not None,
            },
        }

    def compare_modes(self, query: str, **kwargs) -> Dict[str, Any]:
        """
        Compare LLM, RAG, and GraphRAG responses side-by-side.

        Args:
            query: The user query
            **kwargs: Additional parameters

        Returns:
            Dict containing responses from all three modes
        """
        try:
            top_k = kwargs.get("top_k", self.config.get("top_k", 5))

            # LLM only (no retrieval)
            llm_response = self.global_graphrag_module.send_to_llm(
                [
                    {
                        "role": "user",
                        "content": f"Answer this question concisely: {query}",
                    }
                ]
            )
            llm_answer = llm_response.choices[0].message.content

            # RAG response
            rag_answer = self.global_graphrag_module.ask_question_rag(
                query, self.iris_engine, self.emb_model, top_k
            )

            # GraphRAG response
            graphrag_answer = self.global_graphrag_module.ask_question_graphrag(
                query, self.iris_engine, self.emb_model, self.irispy, top_k
            )

            # Get graph data for visualization
            combined_results = self.global_graphrag_module.prepare_combined_results(
                query, self.iris_engine, self.emb_model, self.irispy, top_k
            )
            graph_data = self.global_graphrag_module.combine_graphs(
                combined_results["graphs"]
            )

            return {
                "query": query,
                "llm": {"answer": llm_answer, "mode": "llm_only"},
                "rag": {"answer": rag_answer, "mode": "rag"},
                "graphrag": {
                    "answer": graphrag_answer,
                    "mode": "graphrag",
                    "graph_data": graph_data,
                    "retrieved_papers": combined_results["papers"],
                },
            }

        except Exception as e:
            logger.error(f"Comparison failed: {e}")
            raise IRISGlobalGraphRAGException(f"Mode comparison failed: {e}")
