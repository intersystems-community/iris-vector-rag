"""
ColBERT RAG Pipeline implementation for iris_rag package.

This pipeline implements ColBERT (Contextualized Late Interaction over BERT) approach:
1. Token-level embeddings for both queries and documents
2. MaxSim operation for fine-grained matching
3. Late interaction between query and document tokens
"""

import logging
from typing import List, Dict, Any, Optional, Callable
import numpy as np

from ...core.base import RAGPipeline
from ...core.models import Document
from ...config.manager import ConfigurationManager
from .retriever import ColBERTRetriever

logger = logging.getLogger(__name__)

class ColBERTRAGPipeline(RAGPipeline):
    """
    ColBERT RAG pipeline implementation using iris_rag architecture.
    
    This pipeline uses token-level embeddings and MaxSim operations for
    fine-grained query-document matching.
    """
    
    def __init__(self, iris_connector,
                 config_manager: ConfigurationManager,
                 colbert_query_encoder: Optional[Callable[[str], List[List[float]]]] = None,
                 llm_func: Optional[Callable[[str], str]] = None,
                 embedding_func: Optional[Callable] = None,
                 vector_store=None):
        """
        Initialize ColBERT RAG pipeline.
        
        Args:
            iris_connector: Database connection
            config_manager: Configuration manager
            colbert_query_encoder: Function to encode queries into token embeddings
            llm_func: Function for answer generation
            embedding_func: Function for document-level embeddings (used for candidate retrieval)
            vector_store: Optional VectorStore instance
        """
        super().__init__(config_manager, vector_store)
        
        # Store iris_connector for use in this pipeline
        self.iris_connector = iris_connector
        
        # Initialize schema manager for dimension management
        from ...storage.schema_manager import SchemaManager
        
        # Create a simple connection manager wrapper
        connection_manager = type('ConnectionManager', (), {
            'get_connection': lambda *args, **kwargs: iris_connector
        })()
        
        self.schema_manager = SchemaManager(connection_manager, config_manager)
        
        # Get dimensions from schema manager
        self.doc_embedding_dim = self.schema_manager.get_vector_dimension("SourceDocuments")
        self.token_embedding_dim = self.schema_manager.get_vector_dimension("DocumentTokenEmbeddings")
        
        logger.info(f"ColBERT: Document embeddings = {self.doc_embedding_dim}D, Token embeddings = {self.token_embedding_dim}D")
        
        # Store embedding functions with proper naming
        self.doc_embedding_func = embedding_func  # 384D for document-level retrieval
        self.colbert_query_encoder = colbert_query_encoder  # 768D for token-level scoring
        self.llm_func = llm_func
        
        # Get ColBERT interface from config if not provided
        if not self.colbert_query_encoder:
            from ...embeddings.colbert_interface import get_colbert_interface_from_config
            self.colbert_interface = get_colbert_interface_from_config(
                config_manager, iris_connector
            )
            # Wrap interface methods for backwards compatibility
            self.colbert_query_encoder = self.colbert_interface.encode_query
        else:
            # If custom encoder provided, use RAG Templates interface for other operations
            from ...embeddings.colbert_interface import RAGTemplatesColBERTInterface
            self.colbert_interface = RAGTemplatesColBERTInterface(self.token_embedding_dim)
        
        if not self.llm_func:
            from common.utils import get_llm_func
            self.llm_func = get_llm_func()
            
        if not self.doc_embedding_func:
            from common.utils import get_embedding_func
            self.doc_embedding_func = get_embedding_func()
        
        # Validate dimensions match expectations
        self._validate_embedding_dimensions()

        # Ensure DocumentTokenEmbeddings table exists before initializing retriever
        try:
            self.schema_manager.ensure_table_schema("DocumentTokenEmbeddings")
            logger.info("✅ DocumentTokenEmbeddings table schema ensured")
        except Exception as e:
            logger.error(f"Failed to ensure DocumentTokenEmbeddings table schema: {e}")
            # Continue anyway - the retriever will handle missing table gracefully
        
        # Initialize the retriever
        self.retriever = ColBERTRetriever(
            config_manager=self.config_manager,
            vector_store=self.vector_store,
            doc_embedding_func=self.doc_embedding_func,
            doc_embedding_dim=self.doc_embedding_dim,
            token_embedding_dim=self.token_embedding_dim
        )
        
        logger.info("ColBERTRAGPipeline initialized with proper dimension handling")
    
    def _validate_embedding_dimensions(self):
        """
        Validate that embedding functions produce the expected dimensions.
        """
        try:
            # Test document embedding function
            if self.doc_embedding_func:
                test_doc_embedding = self.doc_embedding_func("test")
                if len(test_doc_embedding) != self.doc_embedding_dim:
                    logger.warning(f"ColBERT: Document embedding function produces {len(test_doc_embedding)}D, expected {self.doc_embedding_dim}D")
            
            # Test ColBERT query encoder  
            if self.colbert_query_encoder:
                test_token_embeddings = self.colbert_query_encoder("test")
                if test_token_embeddings and len(test_token_embeddings[0]) != self.token_embedding_dim:
                    logger.warning(f"ColBERT: Token embedding function produces {len(test_token_embeddings[0])}D, expected {self.token_embedding_dim}D")
                    
        except Exception as e:
            logger.warning(f"ColBERT: Could not validate embedding dimensions: {e}")

    def _format_vector_for_sql(self, vector: List[float]) -> str:
        """Formats a vector list into a comma-separated string for IRIS SQL."""
        if not vector:
            return "[]"
        return "[" + ",".join(f"{x:.15g}" for x in vector) + "]"
    
    def validate_setup(self) -> bool:
        """
        Validate that ColBERT pipeline is properly set up with token embeddings.
        Auto-populates missing token embeddings if needed.
        
        Returns:
            bool: True if setup is valid, False otherwise
        """
        # Handle both connection manager and raw connection
        if hasattr(self.iris_connector, 'get_connection'):
            connection = self.iris_connector.get_connection()
        else:
            connection = self.iris_connector
        cursor = connection.cursor()
        
        try:
            # Check if DocumentTokenEmbeddings table exists
            check_table_sql = """
                SELECT COUNT(*) FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = 'DocumentTokenEmbeddings'
            """
            cursor.execute(check_table_sql)
            table_exists = cursor.fetchone()[0] > 0
            
            if not table_exists:
                logger.warning("DocumentTokenEmbeddings table does not exist, creating it...")
                try:
                    self.schema_manager.ensure_table_schema("DocumentTokenEmbeddings")
                    logger.info("DocumentTokenEmbeddings table created successfully")
                except Exception as e:
                    logger.error(f"Failed to create DocumentTokenEmbeddings table: {e}")
                    return False
            
            # Check if we have token embeddings
            check_tokens_sql = """
                SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings
            """
            cursor.execute(check_tokens_sql)
            token_count = cursor.fetchone()[0]
            
            if token_count == 0:
                logger.warning("No token embeddings found, auto-populating...")
                try:
                    # Auto-populate missing token embeddings
                    from ...services.token_embedding_service import TokenEmbeddingService
                    
                    # Create connection manager wrapper for the service
                    connection_manager = type('ConnectionManager', (), {
                        'get_connection': lambda: self.iris_connector if not hasattr(self.iris_connector, 'get_connection') else self.iris_connector.get_connection()
                    })()
                    
                    token_service = TokenEmbeddingService(self.config_manager, connection_manager)
                    stats = token_service.ensure_token_embeddings_exist()
                    
                    logger.info(f"Auto-populated token embeddings: {stats.documents_processed} docs, "
                               f"{stats.tokens_generated} tokens in {stats.processing_time:.2f}s")
                    
                    if stats.documents_processed == 0 and stats.errors > 0:
                        logger.error("Failed to auto-populate token embeddings")
                        return False
                        
                except Exception as e:
                    logger.error(f"Failed to auto-populate token embeddings: {e}")
                    return False
            
            # Re-check token count after auto-population
            cursor.execute(check_tokens_sql)
            final_token_count = cursor.fetchone()[0]
            
            if final_token_count == 0:
                logger.error("ColBERT validation failed: No token embeddings available after auto-population")
                return False
            
            logger.info(f"ColBERT validation passed: Found {final_token_count} token embeddings")
            return True
            
        except Exception as e:
            logger.error(f"ColBERT validation failed with error: {e}")
            return False
        finally:
            cursor.close()

    def execute(self, query_text: str, **kwargs) -> Dict[str, Any]:
        """
        Execute the ColBERT RAG pipeline (required abstract method).
        
        Args:
            query_text: The input query string
            **kwargs: Additional parameters including top_k
            
        Returns:
            Dictionary containing query, answer, and retrieved documents
        """
        top_k = kwargs.pop("top_k", 5)
        return self.run(query_text, top_k, **kwargs)
    
    def run(self, query: str, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Execute the ColBERT RAG pipeline.
        
        Args:
            query: The input query string
            top_k: Number of documents to retrieve
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing query, answer, and retrieved documents
        """
        logger.info(f"ColBERT: Processing query: '{query[:50]}...'")
        
        if not self.validate_setup():
            logger.warning("ColBERT setup validation failed - pipeline may not work correctly")
        
        start_time = self._get_current_time()
        
        try:
            query_token_embeddings = self.colbert_query_encoder(query)
            logger.debug(f"ColBERT: Generated {len(query_token_embeddings)} query token embeddings")
            
            if len(query_token_embeddings) == 0:
                raise ValueError("ColBERT query encoder returned empty token embeddings")
            
            retrieved_docs = self.retriever._retrieve_documents_with_colbert(
                query_text=query,
                query_token_embeddings=np.array(query_token_embeddings),
                top_k=top_k
            )
            
            answer = self._generate_answer(query, retrieved_docs)
            
            execution_time = self._get_current_time() - start_time
            
            result = {
                "query": query,
                "answer": answer,
                "retrieved_documents": retrieved_docs,
                "execution_time": execution_time,
                "technique": "ColBERT",
                "token_count": len(query_token_embeddings)
            }
            
            logger.info(f"ColBERT: Completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"ColBERT pipeline failed: {e}")
            # Fallback to basic retrieval if ColBERT fails
            logger.info("Falling back to basic vector retrieval.")
            start_time = self._get_current_time()
            retrieved_docs = self.retriever._fallback_to_basic_retrieval(query, top_k)
            if not retrieved_docs:
                raise e from None # Re-raise original exception if fallback also fails
            answer = self._generate_answer(query, retrieved_docs)
            return {
                "query": query,
                "answer": answer,
                "retrieved_documents": retrieved_docs,
                "execution_time": self._get_current_time() - start_time,
                "technique": "ColBERT (fallback)",
                "token_count": 0
            }

    def _generate_answer(self, query: str, documents: List[Document]) -> str:
        """
        Generate answer using retrieved documents and LLM.
        
        Args:
            query: Original query
            documents: Retrieved documents
            
        Returns:
            Generated answer string
        """
        if not documents:
            return "No relevant documents found to answer the query."
        
        context_parts = []
        for i, doc in enumerate(documents, 1):
            context_parts.append(f"Document {i}: {doc.page_content[:500]}...")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""Based on the following documents, please answer the question.

Question: {query}

Documents:
{context}

Answer:"""
        
        answer = self.llm_func(prompt)
        
        return answer.strip()
    
    def load_documents(self, documents_path: str, **kwargs) -> dict:
        """
        Load and process documents into the ColBERT pipeline's knowledge base.
        
        ColBERT uses token-level embeddings, so chunking is disabled to preserve
        the full document context for token-level analysis.
        
        Args:
            documents_path: Path to documents or directory of documents
            **kwargs: Additional keyword arguments including:
                - auto_chunk: Whether to enable automatic chunking (default: False for ColBERT)
                - chunking_strategy: Strategy to use (ignored for ColBERT)
                
        Returns:
            Dictionary with loading results including chunking information
        """
        logger.info(f"ColBERT: Loading documents from {documents_path}")
        
        # Get chunking configuration from pipeline overrides
        chunking_config = self.config_manager.get_config("pipeline_overrides:colbert:chunking", {})
        
        # ColBERT disables chunking by default (uses token-level embeddings)
        auto_chunk = kwargs.get('auto_chunk', chunking_config.get('enabled', False))
        chunking_strategy = kwargs.get('chunking_strategy', chunking_config.get('strategy', 'fixed_size'))
        
        logger.info(f"ColBERT: Using chunking - enabled: {auto_chunk}, strategy: {chunking_strategy}")
        
        # Load documents from path
        from ...core.models import Document
        import os
        
        documents = []
        if os.path.isfile(documents_path):
            # Single file
            with open(documents_path, 'r', encoding='utf-8') as f:
                content = f.read()
                documents.append(Document(
                    id=os.path.basename(documents_path),
                    page_content=content,
                    metadata={"source": documents_path}
                ))
        elif os.path.isdir(documents_path):
            # Directory of files
            for filename in os.listdir(documents_path):
                if filename.endswith(('.txt', '.md', '.json')):
                    filepath = os.path.join(documents_path, filename)
                    with open(filepath, 'r', encoding='utf-8') as f:
                        content = f.read()
                        documents.append(Document(
                            id=filename,
                            page_content=content,
                            metadata={"source": filepath}
                        ))
        
        if not documents:
            logger.warning(f"ColBERT: No documents found at {documents_path}")
            return {
                "documents_loaded": 0,
                "chunks_created": 0,
                "chunking_enabled": auto_chunk,
                "chunking_strategy": chunking_strategy
            }
        
        # Use vector store to add documents with chunking disabled
        result = self.vector_store.add_documents(
            documents=documents,
            auto_chunk=auto_chunk,
            chunking_strategy=chunking_strategy
        )
        
        # Auto-generate token embeddings for all loaded documents
        try:
            logger.info("ColBERT: Auto-generating token embeddings for loaded documents...")
            
            from ...services.token_embedding_service import TokenEmbeddingService
            
            # Create connection manager wrapper for the service
            connection_manager = type('ConnectionManager', (), {
                'get_connection': lambda: self.iris_connector if not hasattr(self.iris_connector, 'get_connection') else self.iris_connector.get_connection()
            })()
            
            token_service = TokenEmbeddingService(self.config_manager, connection_manager)
            stats = token_service.ensure_token_embeddings_exist()
            
            logger.info(f"ColBERT: Token embedding auto-population completed: "
                       f"{stats.documents_processed} docs, {stats.tokens_generated} tokens, "
                       f"{stats.processing_time:.2f}s, {stats.errors} errors")
            
            if stats.errors > 0:
                logger.warning(f"ColBERT: {stats.errors} errors occurred during token embedding generation")
                
        except Exception as e:
            logger.error(f"ColBERT: Failed to auto-generate token embeddings: {e}")
            # Don't fail the entire load operation, just warn
            logger.warning("ColBERT: Document loading completed but token embeddings may be incomplete")
        
        logger.info(f"ColBERT: Loaded {len(documents)} documents, created {result.get('chunks_created', 0)} chunks")
        
        return {
            "documents_loaded": len(documents),
            "chunks_created": result.get('chunks_created', 0),
            "chunking_enabled": auto_chunk,
            "chunking_strategy": chunking_strategy,
            "vector_store_result": result
        }
    
    def query(self, query_text: str, top_k: int = 5, **kwargs) -> list:
        """
        Perform the retrieval step of the ColBERT RAG pipeline.
        
        Given a query, returns the most relevant document chunks using
        ColBERT token-level matching.
        
        Args:
            query_text: The input query string
            top_k: Number of top relevant documents to retrieve
            **kwargs: Additional keyword arguments
            
        Returns:
            List of retrieved Document objects
        """
        logger.info(f"ColBERT: Querying for '{query_text[:50]}...' (top_k={top_k})")
        
        try:
            # Generate query token embeddings
            query_token_embeddings = self.colbert_query_encoder(query_text)
            
            if len(query_token_embeddings) == 0:
                logger.warning("ColBERT: Empty query token embeddings, falling back to basic retrieval")
                return self.retriever._fallback_to_basic_retrieval(query_text, top_k)
            
            # Use ColBERT retriever for token-level matching
            retrieved_docs = self.retriever._retrieve_documents_with_colbert(
                query_text=query_text,
                query_token_embeddings=np.array(query_token_embeddings),
                top_k=top_k
            )
            
            logger.info(f"ColBERT: Retrieved {len(retrieved_docs)} documents")
            return retrieved_docs
            
        except Exception as e:
            logger.error(f"ColBERT query failed: {e}")
            # Fallback to basic retrieval
            logger.info("ColBERT: Falling back to basic vector retrieval")
            return self.retriever._fallback_to_basic_retrieval(query_text, top_k)

    def _get_current_time(self) -> float:
        """Get current time for performance measurement"""
        import time
        return time.time()
    
    def setup_database(self) -> bool:
        """
        Set up database tables and indexes required for ColBERT pipeline.
        
        Returns:
            bool: True if setup successful, False otherwise
        """
        try:
            logger.info("Setting up ColBERT database tables and indexes...")
            
            # Use SchemaManager for consistent database schema management
            from iris_rag.storage.schema_manager import SchemaManager
            
            # Create connection manager wrapper for SchemaManager
            connection_manager = type('ConnectionManager', (), {
                'get_connection': lambda: self.iris_connector
            })()
            
            schema_manager = SchemaManager(connection_manager, self.config_manager)
            
            # Ensure DocumentTokenEmbeddings table schema using SchemaManager
            schema_manager.ensure_table_schema("DocumentTokenEmbeddings")
            
            logger.info("ColBERT database setup completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"ColBERT database setup failed: {e}")
            return False