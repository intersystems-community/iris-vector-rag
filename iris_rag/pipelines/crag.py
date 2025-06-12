"""
CRAG (Corrective RAG) Pipeline implementation for iris_rag package.

This pipeline implements Corrective Retrieval-Augmented Generation:
1. Initial document retrieval
2. Retrieval evaluation (confident/ambiguous/disoriented)
3. Corrective actions based on evaluation
4. Enhanced answer generation
"""

import logging
from typing import List, Dict, Any, Optional, Callable, Literal
import time

from ..core.base import RAGPipeline
from ..core.models import Document
from ..core.connection import ConnectionManager
from ..config.manager import ConfigurationManager

logger = logging.getLogger(__name__)

# Define evaluation status
RetrievalStatus = Literal["confident", "ambiguous", "disoriented"]

class CRAGPipeline(RAGPipeline):
    """
    CRAG (Corrective RAG) pipeline implementation using iris_rag architecture.
    
    This pipeline evaluates retrieval quality and applies corrective measures
    to improve answer generation.
    """
    
    def __init__(self, connection_manager=None, config_manager=None, vector_store=None,
                 iris_connector=None, embedding_func=None, llm_func=None, web_search_func=None):
        """
        Initialize CRAG pipeline with new architecture as primary.
        
        Args:
            connection_manager: Database connection manager (new architecture)
            config_manager: Configuration manager (new architecture)
            vector_store: Optional VectorStore instance
            iris_connector: Database connection (legacy parameter)
            embedding_func: Function to generate embeddings
            llm_func: Function for answer generation
            web_search_func: Function for web search (optional)
        """
        # Handle new architecture first, then backward compatibility
        if connection_manager is None and iris_connector is not None:
            # Legacy mode - create a simple connection manager wrapper
            class LegacyConnectionManager:
                def __init__(self, connection):
                    self._connection = connection
                def get_connection(self):
                    return self._connection
            connection_manager = LegacyConnectionManager(iris_connector)
        
        if config_manager is None:
            # Create a minimal config manager
            class MinimalConfigManager:
                def get(self, key, default=None):
                    return default
                def get_embedding_config(self):
                    return {'model': 'all-MiniLM-L6-v2', 'dimension': 384}
                def get_vector_index_config(self):
                    return {'type': 'HNSW', 'M': 16, 'efConstruction': 200}
            config_manager = MinimalConfigManager()
        
        # Initialize parent with vector store
        super().__init__(connection_manager, config_manager, vector_store)
        
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        self.web_search_func = web_search_func
        
        # Get functions from config if not provided
        if not self.embedding_func:
            try:
                from common.utils import get_embedding_func
                self.embedding_func = get_embedding_func()
            except ImportError:
                logger.warning("Could not import get_embedding_func from common.utils")
        
        if not self.llm_func:
            try:
                from common.utils import get_llm_func
                self.llm_func = get_llm_func()
            except ImportError:
                logger.warning("Could not import get_llm_func from common.utils")
        
        # Initialize retrieval evaluator
        self.evaluator = RetrievalEvaluator(self.llm_func, self.embedding_func)
        
        logger.info("CRAGPipeline initialized successfully")
    
    def execute(self, query_text: str, **kwargs) -> dict:
        """
        Execute the CRAG pipeline (required abstract method).
        
        Args:
            query_text: The input query string
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing query, answer, and retrieved documents
        """
        top_k_execute = kwargs.pop("top_k", 5) # Get top_k and remove from kwargs
        return self.run(query_text, top_k=top_k_execute, **kwargs) # Pass top_k explicitly
    
    def load_documents(self, documents_path: str, **kwargs) -> None:
        """
        Load documents into the knowledge base (required abstract method).
        
        Args:
            documents_path: Path to documents or directory
            **kwargs: Additional keyword arguments including:
                - documents: List of Document objects (if providing directly)
                - chunk_documents: Whether to chunk documents (default: True)
                - generate_embeddings: Whether to generate embeddings (default: True)
        """
        # Handle direct document input
        if "documents" in kwargs:
            documents = kwargs["documents"]
            if not isinstance(documents, list):
                raise ValueError("Documents must be provided as a list")
        else:
            # Load documents from path - basic implementation
            import os
            documents = []
            
            if os.path.isfile(documents_path):
                with open(documents_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                doc = Document(
                    page_content=content,
                    metadata={"source": documents_path}
                )
                documents.append(doc)
            elif os.path.isdir(documents_path):
                for filename in os.listdir(documents_path):
                    file_path = os.path.join(documents_path, filename)
                    if os.path.isfile(file_path):
                        try:
                            with open(file_path, 'r', encoding='utf-8') as f:
                                content = f.read()
                            doc = Document(
                                page_content=content,
                                metadata={"source": file_path, "filename": filename}
                            )
                            documents.append(doc)
                        except Exception as e:
                            logger.warning(f"Failed to load file {file_path}: {e}")
        
        # Store documents using vector store
        embeddings = None
        if self.embedding_func:
            embeddings = [self.embedding_func([doc.page_content])[0] for doc in documents]
        
        document_ids = self._store_documents(documents, embeddings)
        logger.info(f"CRAG: Loaded {len(documents)} documents with IDs: {document_ids}")
    
    def query(self, query_text: str, top_k: int = 5, **kwargs) -> list:
        """
        Perform the retrieval step of the CRAG pipeline (required abstract method).
        
        Args:
            query_text: The input query string
            top_k: Number of top relevant documents to retrieve
            **kwargs: Additional keyword arguments
            
        Returns:
            List of retrieved Document objects
        """
        # Perform initial retrieval
        initial_docs = self._initial_retrieval(query_text, top_k)
        
        # Evaluate retrieval quality
        retrieval_status = self.evaluator.evaluate(query_text, initial_docs)
        
        # Apply corrective actions
        corrected_docs = self._apply_corrective_actions(
            query_text, initial_docs, retrieval_status, top_k
        )
        
        return corrected_docs
    
    def run(self, query: str, top_k: int = 5, **kwargs) -> Dict[str, Any]:
        """
        Execute the CRAG pipeline.
        
        Args:
            query: The input query string
            top_k: Number of documents to retrieve
            **kwargs: Additional parameters
            
        Returns:
            Dictionary containing query, answer, and retrieved documents
        """
        logger.info(f"CRAG: Processing query: '{query[:50]}...'")
        
        start_time = time.time()
        
        try:
            # Stage 1: Initial retrieval
            initial_docs = self._initial_retrieval(query, top_k)
            
            # Stage 2: Evaluate retrieval quality
            retrieval_status = self.evaluator.evaluate(query, initial_docs)
            logger.info(f"CRAG: Retrieval status: {retrieval_status}")
            
            # Stage 3: Apply corrective actions based on evaluation
            corrected_docs = self._apply_corrective_actions(
                query, initial_docs, retrieval_status, top_k
            )
            
            # Stage 4: Generate answer
            answer = self._generate_answer(query, corrected_docs, retrieval_status)
            
            execution_time = time.time() - start_time
            
            result = {
                "query": query,
                "answer": answer,
                "retrieved_documents": corrected_docs,
                "execution_time": execution_time,
                "technique": "CRAG",
                "retrieval_status": retrieval_status,
                "initial_doc_count": len(initial_docs),
                "final_doc_count": len(corrected_docs)
            }
            
            logger.info(f"CRAG: Completed in {execution_time:.2f}s")
            return result
            
        except Exception as e:
            logger.error(f"CRAG pipeline failed: {e}")
            raise
    
    def _initial_retrieval(self, query: str, top_k: int) -> List[Document]:
        """
        Perform initial document retrieval using vector store.
        
        Args:
            query: Query string
            top_k: Number of documents to retrieve
            
        Returns:
            List of initially retrieved documents
        """
        logger.debug(f"CRAG _initial_retrieval() entry: query='{query}', top_k={top_k}")
        
        # Generate query embedding
        query_embedding = self.embedding_func([query])[0]
        logger.debug(f"CRAG _initial_retrieval() generated embedding: length={len(query_embedding)}")
        
        # Use vector store helper method
        results = self._retrieve_documents_by_vector(
            query_embedding=query_embedding,
            top_k=top_k
        )
        
        retrieved_docs = []
        for doc, similarity_score in results:
            # Update metadata with retrieval information
            doc.metadata.update({
                "similarity_score": similarity_score,
                "retrieval_method": "initial_vector_similarity"
            })
            retrieved_docs.append(doc)
        
        logger.debug(f"CRAG _initial_retrieval() exit: found {len(retrieved_docs)} documents")
        return retrieved_docs
    
    def _apply_corrective_actions(self, query: str, initial_docs: List[Document],
                                status: RetrievalStatus, top_k: int) -> List[Document]:
        """
        Apply corrective actions based on retrieval evaluation.
        
        Args:
            query: Original query
            initial_docs: Initially retrieved documents
            status: Retrieval evaluation status
            top_k: Number of documents to retrieve
            
        Returns:
            Corrected/enhanced document list
        """
        logger.debug(f"CRAG _apply_corrective_actions() entry: query='{query}', status={status}, initial_docs_count={len(initial_docs)}, top_k={top_k}")
        
        if status == "confident":
            # High confidence - use initial results
            logger.debug("CRAG _apply_corrective_actions(): High confidence - using initial retrieval")
            logger.debug(f"CRAG _apply_corrective_actions() exit (confident): returning {len(initial_docs)} documents")
            return initial_docs
        
        elif status == "ambiguous":
            # Medium confidence - enhance with additional retrieval
            logger.debug("CRAG _apply_corrective_actions(): Ambiguous results - enhancing with additional retrieval")
            enhanced_docs = self._enhance_retrieval(query, initial_docs, top_k)
            logger.debug(f"CRAG _apply_corrective_actions() exit (ambiguous): returning {len(enhanced_docs)} enhanced documents")
            return enhanced_docs
        
        else:  # disoriented
            # Low confidence - perform web search or knowledge base expansion
            logger.debug("CRAG _apply_corrective_actions(): Low confidence - performing knowledge base expansion")
            expanded_docs = self._knowledge_base_expansion(query, top_k)
            logger.debug(f"CRAG _apply_corrective_actions() exit (disoriented): returning {len(expanded_docs)} expanded documents")
            return expanded_docs
    
    def _enhance_retrieval(self, query: str, initial_docs: List[Document], top_k: int) -> List[Document]:
        """
        Enhance retrieval with additional strategies for ambiguous cases.
        
        Args:
            query: Original query
            initial_docs: Initially retrieved documents
            top_k: Number of documents to retrieve
            
        Returns:
            Enhanced document list
        """
        logger.debug(f"CRAG _enhance_retrieval() entry: query='{query}', initial_docs_count={len(initial_docs)}, top_k={top_k}")
        
        # Try chunk-based retrieval for more granular results
        enhanced_docs = list(initial_docs)  # Start with initial docs
        logger.debug(f"CRAG _enhance_retrieval() starting with {len(enhanced_docs)} initial documents")
        
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()
        
        try:
            # Generate query embedding
            query_embedding = self.embedding_func([query])[0]
            # Format embedding for IRIS SQL - must embed directly in SQL
            query_embedding_str = ','.join([f'{x:.10f}' for x in query_embedding])
            
            # Try chunk-based retrieval with parameterized query
            chunk_sql = f"""
                SELECT TOP {top_k}
                    source_doc_id,
                    chunk_text,
                    VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) as similarity_score
                FROM RAG.ChunkedDocuments
                WHERE embedding IS NOT NULL
                ORDER BY similarity_score DESC
            """
            
            # Execute with embedding as parameter
            cursor.execute(chunk_sql, [query_embedding_str])
            chunk_results = cursor.fetchall()
            logger.debug(f"CRAG _enhance_retrieval() chunk-based query returned {len(chunk_results)} chunks")
            
            # Add chunk-based results
            for row in chunk_results:
                # VectorStore guarantees string content
                page_content = str(row[1])
                
                doc = Document(
                    id=f"{row[0]}_chunk",
                    page_content=page_content,
                    metadata={
                        "similarity_score": float(row[2]),
                        "retrieval_method": "chunk_based_enhancement",
                        "source_doc_id": row[0]
                    }
                )
                enhanced_docs.append(doc)
            
            logger.debug(f"CRAG _enhance_retrieval() after adding chunks: {len(enhanced_docs)} total documents")
            
            # Remove duplicates and limit results
            seen_content = set()
            unique_docs = []
            for doc in enhanced_docs:
                content_hash = hash(doc.page_content[:100])  # Use first 100 chars as hash
                if content_hash not in seen_content:
                    seen_content.add(content_hash)
                    unique_docs.append(doc)
                    if len(unique_docs) >= top_k * 2:  # Allow more docs for ambiguous cases
                        break
            
            logger.debug(f"CRAG _enhance_retrieval() exit: found {len(unique_docs)} unique documents after deduplication")
            return unique_docs
            
        finally:
            cursor.close()
    
    def _knowledge_base_expansion(self, query: str, top_k: int) -> List[Document]:
        """
        Perform knowledge base expansion for disoriented cases.
        
        Args:
            query: Original query
            top_k: Number of documents to retrieve
            
        Returns:
            Expanded document list from knowledge base
        """
        logger.debug(f"CRAG _knowledge_base_expansion() entry: query='{query}', top_k={top_k}")
        
        # For disoriented cases, try broader search strategies
        connection = self.connection_manager.get_connection()
        cursor = connection.cursor()
        
        try:
            # ADDED: Log sample text_content from DB for comparison (moved to top)
            try:
                sample_docs_sql = "SELECT TOP 3 doc_id, %ID, text_content FROM RAG.SourceDocuments"
                sample_cursor = connection.cursor() # Create a new cursor for this
                sample_cursor.execute(sample_docs_sql)
                sample_db_docs = sample_cursor.fetchall()
                logger.debug(f"CRAG _knowledge_base_expansion() sample DB text_content (ID, RowID, Content): {sample_db_docs}")
                sample_cursor.close()
            except Exception as e_sample:
                logger.error(f"CRAG _knowledge_base_expansion(): Error fetching sample docs for logging: {e_sample}")

            # Use semantic search instead of keyword search for better results
            # This avoids IRIS stream field limitations with LIKE operations
            try:
                # Generate query embedding
                query_embedding = self.embedding_func(query)
                if isinstance(query_embedding, list) and len(query_embedding) > 0:
                    if isinstance(query_embedding[0], list):
                        query_embedding = query_embedding[0]
                # Use working pattern from archived CRAG V2 (lines 50-59)
                query_embedding_str = ','.join([f'{x:.10f}' for x in query_embedding])
                logger.debug(f"CRAG _knowledge_base_expansion() generated embedding for semantic search: length={len(query_embedding)}")
                
                # Use semantic search with parameterized query
                sql = f"""
                    SELECT TOP {top_k * 2}
                        doc_id,
                        text_content,
                        VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) as similarity_score
                    FROM RAG.SourceDocuments
                    WHERE embedding IS NOT NULL AND embedding <> ''
                    ORDER BY similarity_score DESC
                """
                
                like_params = [query_embedding_str]
                
            except Exception as e:
                logger.warning(f"CRAG _knowledge_base_expansion(): Semantic search failed, using simple retrieval: {e}")
                # Fallback: just get some documents
                sql = f"""
                    SELECT TOP {top_k * 2}
                        doc_id,
                        text_content,
                        0.5 as similarity_score
                    FROM RAG.SourceDocuments
                    WHERE text_content IS NOT NULL
                    ORDER BY doc_id
                """
                like_params = []
                
            # Execute with parameters
            cursor.execute(sql, like_params)
            results = cursor.fetchall()
            logger.debug(f"CRAG _knowledge_base_expansion() database query returned {len(results)} rows")
            
            expanded_docs = []
            for row in results:
                # VectorStore guarantees string content
                page_content = str(row[1])
                
                doc = Document(
                    id=row[0],
                    page_content=page_content,
                    metadata={
                        "similarity_score": float(row[2]),
                        "retrieval_method": "knowledge_base_expansion"
                    }
                )
                expanded_docs.append(doc)
            
            final_docs = expanded_docs[:top_k]
            logger.debug(f"CRAG _knowledge_base_expansion() exit: returning {len(final_docs)} documents (limited from {len(expanded_docs)})")
            return final_docs
            
        finally:
            cursor.close()
    
    def _generate_answer(self, query: str, documents: List[Document], status: RetrievalStatus) -> str:
        """
        Generate answer using retrieved documents and LLM.
        
        Args:
            query: Original query
            documents: Retrieved documents
            status: Retrieval evaluation status
            
        Returns:
            Generated answer string
        """
        logger.debug(f"CRAG _generate_answer() entry: query='{query}', documents_count={len(documents)}, status={status}")
        
        if not documents:
            logger.debug("CRAG _generate_answer(): No documents provided, returning default message")
            return "I couldn't find relevant information to answer your question. Please try rephrasing your query."
        
        # Prepare context based on retrieval status
        if status == "confident":
            context_intro = "Based on highly relevant documents:"
        elif status == "ambiguous":
            context_intro = "Based on available information (with some uncertainty):"
        else:
            context_intro = "Based on broader search results:"
        
        logger.debug(f"CRAG _generate_answer() using context_intro: '{context_intro}'")
        
        # Prepare context from retrieved documents
        context_parts = [context_intro]
        for i, doc in enumerate(documents[:5], 1):  # Limit to top 5 for context
            # VectorStore guarantees string content, but ensure it's a string for safety
            page_content = str(doc.page_content)
            context_parts.append(f"Document {i}: {page_content[:400]}...")
            logger.debug(f"CRAG _generate_answer() added document {i}: id={doc.id}, content_preview='{page_content[:100]}...'")
        
        context = "\n\n".join(context_parts)
        logger.debug(f"CRAG _generate_answer() prepared context: length={len(context)} characters")
        
        # Create prompt for LLM with confidence indication
        prompt = f"""Please answer the question based on the provided documents.

Question: {query}

{context}

Please provide a comprehensive answer. If the information is uncertain or incomplete, please indicate this in your response.

Answer:"""
        
        logger.debug(f"CRAG _generate_answer() calling LLM with prompt length={len(prompt)}")
        
        # Generate answer using LLM
        answer = self.llm_func(prompt)
        
        logger.debug(f"CRAG _generate_answer() exit: generated answer length={len(answer)}, preview='{answer[:100]}...'")
        return answer.strip()


class RetrievalEvaluator:
    """
    Evaluates the quality of retrieved documents for CRAG.
    """
    
    def __init__(self, llm_func: Optional[Callable] = None, embedding_func: Optional[Callable] = None):
        self.llm_func = llm_func
        self.embedding_func = embedding_func
        logger.debug("RetrievalEvaluator initialized")
    
    def evaluate(self, query_text: str, documents: List[Document]) -> RetrievalStatus:
        """
        Evaluate retrieved documents and return a status.
        
        Args:
            query_text: Original query
            documents: Retrieved documents
            
        Returns:
            Retrieval status (confident/ambiguous/disoriented)
        """
        logger.debug(f"CRAG RetrievalEvaluator.evaluate() entry: query='{query_text[:50]}...', documents_count={len(documents)}")
        
        if not documents:
            logger.debug("CRAG RetrievalEvaluator.evaluate(): No documents provided - status: disoriented")
            return "disoriented"
        
        # VectorStore guarantees string content, but ensure it's a string for safety
        safe_documents = []
        for doc in documents:
            # Ensure page_content is a string (VectorStore should guarantee this)
            safe_page_content = str(doc.page_content)
            # Create a new document with safe content if needed
            if safe_page_content != doc.page_content:
                from ..core.models import Document
                safe_doc = Document(
                    id=doc.id,
                    page_content=safe_page_content,
                    metadata=doc.metadata
                )
                safe_documents.append(safe_doc)
            else:
                safe_documents.append(doc)
        
        logger.debug(f"CRAG RetrievalEvaluator.evaluate(): processed {len(safe_documents)} safe documents")
        
        # Simple heuristic evaluation based on similarity scores
        scores = []
        for i, doc in enumerate(safe_documents):
            if hasattr(doc, 'metadata') and 'similarity_score' in doc.metadata:
                score = doc.metadata['similarity_score']
                scores.append(score)
                logger.debug(f"CRAG RetrievalEvaluator.evaluate(): document {i} (id={doc.id}) similarity_score={score}")
            else:
                logger.debug(f"CRAG RetrievalEvaluator.evaluate(): document {i} (id={doc.id}) has no similarity_score")
        
        if not scores:
            logger.debug("CRAG RetrievalEvaluator.evaluate(): No similarity scores found - status: ambiguous")
            return "ambiguous"
        
        avg_score = sum(scores) / len(scores)
        max_score = max(scores)
        logger.debug(f"CRAG RetrievalEvaluator.evaluate(): score analysis - avg={avg_score:.3f}, max={max_score:.3f}")
        
        # Confidence thresholds
        if max_score > 0.8 and avg_score > 0.6:
            status = "confident"
        elif max_score > 0.6 or avg_score > 0.4:
            status = "ambiguous"
        else:
            status = "disoriented"
        
        logger.debug(f"CRAG RetrievalEvaluator.evaluate() exit: status={status}")
        return status