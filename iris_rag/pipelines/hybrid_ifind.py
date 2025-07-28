"""
Hybrid IFind RAG Pipeline - Single Unified Implementation.

This module provides the single, unified RAG implementation that combines:
1. Vector similarity search for semantic matching
2. IRIS IFind keyword search for precise text matching
3. Reciprocal Rank Fusion (RRF) for optimal result combination

This is the only active Hybrid IFind implementation in the codebase.
"""

import logging
import time
from typing import List, Dict, Any, Optional, Callable
from ..core.base import RAGPipeline
from ..core.models import Document
from common.iris_connection_manager import get_iris_connection
from ..config.manager import ConfigurationManager
from ..storage.iris import IRISStorage
from ..embeddings.manager import EmbeddingManager

logger = logging.getLogger(__name__)


class HybridIFindRAGPipeline(RAGPipeline):
    """
    Hybrid IFind RAG Pipeline - Single Unified Implementation.
    
    This is the only active Hybrid IFind implementation in the codebase.
    It provides a unified pipeline that combines:
    1. Vector similarity search for semantic matching
    2. IRIS IFind text search for precise keyword matching
    3. Reciprocal Rank Fusion (RRF) for optimal result combination
    
    The pipeline integrates these search methods seamlessly to provide
    superior retrieval performance compared to individual approaches.
    """
    
    def __init__(self, config_manager: ConfigurationManager,
                 vector_store=None, llm_func: Optional[Callable[[str], str]] = None,
                 connection_manager=None):
        """
        Initialize the Hybrid IFind RAG Pipeline.
        
        Args:
            config_manager: Manager for configuration settings
            vector_store: Optional VectorStore instance
            llm_func: Optional LLM function for answer generation
            connection_manager: Optional IRISConnectionManager instance
        """
        super().__init__(config_manager, vector_store, connection_manager)
        self.llm_func = llm_func
        
        # Initialize components
        self.storage = IRISStorage(config_manager)
        self.embedding_manager = EmbeddingManager(config_manager)
        
        # Get pipeline configuration
        self.pipeline_config = self.config_manager.get("pipelines:hybrid_ifind", {})
        self.top_k = self.pipeline_config.get("top_k", 5)
        self.vector_weight = self.pipeline_config.get("vector_weight", 0.6)
        self.ifind_weight = self.pipeline_config.get("ifind_weight", 0.4)
        self.min_ifind_score = self.pipeline_config.get("min_ifind_score", 0.1)
        
        logger.info(f"Initialized HybridIFindRAGPipeline with vector_weight={self.vector_weight}")
    
    def execute(self, query_text: str, **kwargs) -> dict:
        """
        Execute the Hybrid IFind pipeline (required abstract method).
        
        Args:
            query_text: The input query string
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing query, answer, and retrieved documents
        """
        top_k = kwargs.get("top_k", 5)
        return self.query(query_text, top_k)
    
    # HybridIFind uses the base class load_documents implementation
    # with a custom _preprocess_documents hook to ensure IFind indexes
    # Pipeline-specific configuration is handled via pipeline_overrides:hybrid_ifind:chunking
    
    def _preprocess_documents(self, documents: List[Document]) -> List[Document]:
        """
        Preprocess documents for HybridIFind - ensures IFind indexes are built after loading.
        
        Args:
            documents: List of documents to preprocess
            
        Returns:
            List of preprocessed documents
        """
        # Call parent preprocessing first
        processed_docs = super()._preprocess_documents(documents)
        
        # Ensure IFind indexes are built after document loading
        self._ensure_ifind_indexes()
        
        return processed_docs
    
    def ingest_documents(self, documents: List[Document], auto_chunk: bool = True,
                        chunking_strategy: str = "hybrid") -> Dict[str, Any]:
        """
        Ingest documents with automatic chunking and IFind indexing support.
        
        Args:
            documents: List of documents to ingest
            auto_chunk: Whether to enable automatic chunking
            chunking_strategy: Strategy to use ('fixed_size', 'semantic', 'hybrid')
            
        Returns:
            Dictionary with ingestion results including chunking information
        """
        start_time = time.time()
        logger.info(f"Starting Hybrid IFind ingestion of {len(documents)} documents with chunking: {auto_chunk}")
        
        try:
            # Get chunking configuration from pipeline overrides
            chunking_config = self.config_manager.get_config("pipeline_overrides:hybrid_ifind:chunking", {})
            
            # Use provided parameters or fall back to config
            auto_chunk = chunking_config.get('enabled', auto_chunk)
            chunking_strategy = chunking_config.get('strategy', chunking_strategy)
            
            # Use vector store to add documents with chunking
            result = self.vector_store.add_documents(
                documents=documents,
                auto_chunk=auto_chunk,
                chunking_strategy=chunking_strategy
            )
            
            # Ensure IFind indexes are built
            self._ensure_ifind_indexes()
            
            end_time = time.time()
            result.update({
                "processing_time": end_time - start_time,
                "pipeline_type": "hybrid_ifind_rag",
                "chunking_enabled": auto_chunk,
                "chunking_strategy": chunking_strategy
            })
            
            logger.info(f"Hybrid IFind ingestion completed: {result}")
            return result
            
        except Exception as e:
            logger.error(f"Hybrid IFind ingestion failed: {e}")
            return {
                "status": "error",
                "error": str(e),
                "pipeline_type": "hybrid_ifind_rag",
                "chunking_enabled": auto_chunk,
                "chunking_strategy": chunking_strategy
            }
    
    def query(self, query_text: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Execute a query using hybrid vector + IFind search.
        
        Args:
            query_text: The query string
            top_k: Number of top documents to retrieve
            
        Returns:
            Dictionary with query results
        """
        start_time = time.time()
        logger.info(f"Processing Hybrid IFind query: {query_text}")
        
        try:
            # Use IRISVectorStore for hybrid search (replaces broken SQL)
            query_embedding = self.embedding_manager.embed_text(query_text)
            
            # Use vector store hybrid search method
            search_results = self.vector_store.hybrid_search(
                query_embedding=query_embedding,
                query_text=query_text,
                k=top_k,
                vector_weight=self.vector_weight,
                ifind_weight=self.ifind_weight
            )
            
            # Convert results to Document list for compatibility
            retrieved_documents = [doc for doc, score in search_results]
            
            # Generate answer if LLM function is available
            answer = None
            if self.llm_func and retrieved_documents:
                context = self._build_context_from_documents(retrieved_documents)
                prompt = self._build_prompt(query_text, context)
                answer = self.llm_func(prompt)
            
            # Ensure answer is never None
            if answer is None:
                if not self.llm_func:
                    answer = "No LLM function available for answer generation."
                elif not retrieved_documents:
                    answer = "No documents retrieved to generate an answer."
                else:
                    answer = "Unable to generate answer from retrieved documents."
            
            end_time = time.time()
            
            result = {
                "query": query_text,
                "answer": answer,
                "retrieved_documents": retrieved_documents,
                "vector_results_count": len(retrieved_documents),  # Using hybrid results
                "ifind_results_count": 0,  # No separate IFind results yet
                "num_documents_retrieved": len(retrieved_documents),
                "processing_time": end_time - start_time,
                "pipeline_type": "hybrid_ifind_rag"
            }
            
            logger.info(f"Hybrid IFind query completed in {end_time - start_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"Hybrid IFind query failed: {e}")
            return {
                "query": query_text,
                "answer": f"Hybrid IFind pipeline error: {e}",
                "retrieved_documents": [],
                "error": str(e),
                "pipeline_type": "hybrid_ifind_rag"
            }
    
    def _ensure_ifind_indexes(self):
        """Ensure IFind indexes are created for text search."""
        connection = get_iris_connection()
        cursor = connection.cursor()
        
        try:
            # Check if IFind index exists
            check_sql = """
            SELECT COUNT(*) FROM INFORMATION_SCHEMA.INDEXES 
            WHERE TABLE_NAME = 'SourceDocuments' 
            AND TABLE_SCHEMA = 'RAG' 
            AND INDEX_NAME LIKE '%IFIND%'
            """
            
            cursor.execute(check_sql)
            result = cursor.fetchone()
            
            if result[0] == 0:
                logger.info("Creating IFind indexes for text search")
                
                # Create IFind index on content
                create_index_sql = """
                CREATE INDEX IF NOT EXISTS idx_sourcedocs_ifind_content
                ON RAG.SourceDocuments (text_content)
                WITH (TYPE = 'IFIND')
                """
                
                cursor.execute(create_index_sql)
                connection.commit()
                logger.info("IFind index created successfully")
            
        except Exception as e:
            logger.error(f"HybridIFind: Could not create IFind index - {e}. HybridIFind requires working IFind functionality.")
            # FAIL instead of silent fallback
            raise RuntimeError(f"HybridIFind pipeline failed: Cannot create IFind indexes. Please use BasicRAG or fix IFind configuration. Error: {e}")
        finally:
            cursor.close()
    
    def _vector_search(self, query_embedding: List[float], top_k: int) -> List[Dict[str, Any]]:
        """Perform vector similarity search."""
        connection = get_iris_connection()
        cursor = connection.cursor()
        
        try:
            # Use vector_sql_utils for proper parameter handling
            from common.vector_sql_utils import format_vector_search_sql, execute_vector_search
            
            # Get column names from schema manager
            table_config = self.schema_manager.get_table_config("SourceDocumentsIFind")
            if not table_config:
                # Fallback to default SourceDocuments config
                table_config = self.schema_manager.get_table_config("SourceDocuments")
            
            id_column = table_config.get("id_column", "ID")
            embedding_column = table_config.get("embedding_column", "embedding")
            content_column = table_config.get("content_column", "TEXT_CONTENT")
            
            # Format vector with brackets for vector_sql_utils
            query_vector_str = f"[{','.join(f'{x:.10f}' for x in query_embedding)}]"
            
            sql = format_vector_search_sql(
                table_name="RAG.SourceDocumentsIFind",
                vector_column=embedding_column,
                vector_string=query_vector_str,
                embedding_dim=len(query_embedding),
                top_k=top_k,
                id_column=id_column,
                content_column=content_column
            )
            
            # Use execute_vector_search utility 
            results = execute_vector_search(cursor, sql)
            
            documents = []
            for row in results:
                documents.append({
                    "doc_id": row[0],
                    "title": row[1],
                    "content": row[2],
                    "vector_score": float(row[3]),
                    "search_type": "vector"
                })
            
            return documents
            
        finally:
            cursor.close()
    
    def _ifind_search(self, query_text: str, top_k: int) -> List[Dict[str, Any]]:
        """Perform IFind text search."""
        connection = get_iris_connection()
        cursor = connection.cursor()
        
        try:
            # Try IFind search first
            ifind_sql = f"""
            SELECT TOP {top_k}
                doc_id, title, text_content,
                1.0 as ifind_score
            FROM RAG.SourceDocumentsIFind
            WHERE %CONTAINS(text_content, ?)
            ORDER BY ifind_score DESC
            """
            
            try:
                cursor.execute(ifind_sql, [query_text])
                results = cursor.fetchall()
                
                documents = []
                for row in results:
                    ifind_score = float(row[3]) if row[3] is not None else 0.0
                    if ifind_score >= self.min_ifind_score:
                        documents.append({
                            "doc_id": row[0],
                            "title": row[1],
                            "content": row[2],
                            "ifind_score": ifind_score,
                            "search_type": "ifind"
                        })
                
                return documents
                
            except Exception as ifind_error:
                logger.error(f"HybridIFind: IFind search failed - {ifind_error}. HybridIFind requires working IFind indexes.")
                # FAIL instead of falling back to LIKE search
                raise RuntimeError(f"HybridIFind pipeline failed: IFind search not working. Please use BasicRAG or ensure IFind indexes are properly configured. Error: {ifind_error}")
            
        finally:
            cursor.close()
    
    def _fuse_results(self, vector_results: List[Dict[str, Any]], 
                     ifind_results: List[Dict[str, Any]], top_k: int) -> List[Dict[str, Any]]:
        """Fuse vector and IFind results using hybrid ranking."""
        
        # Normalize scores
        vector_results = self._normalize_scores(vector_results, "vector_score")
        ifind_results = self._normalize_scores(ifind_results, "ifind_score")
        
        # Create combined results dictionary
        combined_docs = {}
        
        # Add vector results
        for doc in vector_results:
            doc_id = doc["doc_id"]
            combined_docs[doc_id] = doc.copy()
            combined_docs[doc_id]["hybrid_score"] = self.vector_weight * doc["vector_score"]
            combined_docs[doc_id]["has_vector"] = True
            combined_docs[doc_id]["has_ifind"] = False
        
        # Add/merge IFind results
        for doc in ifind_results:
            doc_id = doc["doc_id"]
            if doc_id in combined_docs:
                # Merge scores
                combined_docs[doc_id]["hybrid_score"] += self.ifind_weight * doc["ifind_score"]
                combined_docs[doc_id]["has_ifind"] = True
                combined_docs[doc_id]["ifind_score"] = doc["ifind_score"]
            else:
                # New document from IFind
                combined_docs[doc_id] = doc.copy()
                combined_docs[doc_id]["hybrid_score"] = self.ifind_weight * doc["ifind_score"]
                combined_docs[doc_id]["has_vector"] = False
                combined_docs[doc_id]["has_ifind"] = True
                combined_docs[doc_id]["vector_score"] = 0.0
        
        # Sort by hybrid score and return top_k
        sorted_docs = sorted(combined_docs.values(), 
                           key=lambda x: x["hybrid_score"], 
                           reverse=True)
        
        return sorted_docs[:top_k]
    
    def _normalize_scores(self, results: List[Dict[str, Any]], score_field: str) -> List[Dict[str, Any]]:
        """Normalize scores to 0-1 range."""
        if not results:
            return results
        
        scores = [doc[score_field] for doc in results]
        min_score = min(scores)
        max_score = max(scores)
        
        if max_score == min_score:
            # All scores are the same
            for doc in results:
                doc[score_field] = 1.0
        else:
            # Normalize to 0-1
            for doc in results:
                doc[score_field] = (doc[score_field] - min_score) / (max_score - min_score)
        
        return results
    
    def _build_context(self, documents: List[Dict[str, Any]]) -> str:
        """Build context string from retrieved documents (legacy method)."""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            title = doc.get('title', 'Untitled')
            content = doc.get('content', '')
            score = doc.get('hybrid_score', 0.0)
            context_parts.append(f"[Document {i}: {title} (Score: {score:.3f})]\n{content}")
        
        return "\n\n".join(context_parts)
    
    def _build_context_from_documents(self, documents: List[Document]) -> str:
        """Build context string from Document objects."""
        context_parts = []
        for i, doc in enumerate(documents, 1):
            title = doc.metadata.get('title', 'Untitled')
            content = doc.page_content
            score = doc.metadata.get('hybrid_score', 0.0)
            context_parts.append(f"[Document {i}: {title} (Score: {score:.3f})]\n{content}")
        
        return "\n\n".join(context_parts)
    
    def _build_prompt(self, query: str, context: str) -> str:
        """Build prompt for LLM generation."""
        return f"""Based on the following retrieved documents (ranked by hybrid vector + text search), please answer the question.

Context:
{context}

Question: {query}

Answer:"""