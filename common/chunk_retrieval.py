"""
Chunk Retrieval Service

This module provides infrastructure for retrieving document chunks from the database
for use by RAG techniques that benefit from granular document access.
"""

import logging
from typing import List, Dict, Any, Optional
from .utils import Document # Changed to relative import

logger = logging.getLogger(__name__)

class ChunkRetrievalService:
    """Service for retrieving document chunks from the RAG.DocumentChunks table"""
    
    def __init__(self, iris_connector):
        self.iris_connector = iris_connector
        logger.info("ChunkRetrievalService initialized")
    
    def retrieve_chunks_for_query(self, 
                                query_embedding: List[float], 
                                top_k: int = 20,
                                chunk_types: Optional[List[str]] = None,
                                similarity_threshold: float = 0.7) -> List[Document]:
        """
        Retrieve relevant chunks using vector similarity search
        
        Args:
            query_embedding: Query embedding vector
            top_k: Number of chunks to retrieve
            chunk_types: List of chunk types to search (e.g., ['adaptive', 'hybrid'])
            similarity_threshold: Minimum similarity score
            
        Returns:
            List of Document objects representing chunks
        """
        logger.info(f"Retrieving top {top_k} chunks for query with threshold {similarity_threshold}")
        
        if chunk_types is None:
            chunk_types = ['adaptive']
        
        # Convert embedding to IRIS vector string format
        query_vector_str = ','.join(map(str, query_embedding))
        
        # Build chunk type filter
        chunk_type_placeholders = ','.join(['?' for _ in chunk_types])
        
        # Build SQL using correct IRIS vector syntax (embedding is already VECTOR type)
        # Check if we're using JDBC (which has issues with parameter binding for vectors)
        conn_type = type(self.iris_connector).__name__
        is_jdbc = 'JDBC' in conn_type or hasattr(self.iris_connector, '_jdbc_connection')
        
        retrieved_chunks: List[Document] = []
        cursor = None
        
        try:
            cursor = self.iris_connector.cursor()
            
            # Unified SQL for both JDBC and ODBC, using parameters
            # Corrected column to chunk_embedding and using "both TO_VECTOR()" principle
            chunk_type_placeholders = ','.join(['?' for _ in chunk_types])
            sql_query = f"""
                SELECT TOP ?
                    chunk_id,
                    chunk_text,
                    doc_id,
                    chunk_type,
                    chunk_index,
                    VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) AS score
                FROM RAG.DocumentChunks
                WHERE embedding IS NOT NULL
                  AND chunk_type IN ({chunk_type_placeholders})
                  AND VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) > ?
                ORDER BY score DESC
            """
            
            # Parameters: top_k, query_vector_str (for first VECTOR_COSINE),
            #             *chunk_types (for IN clause),
            #             query_vector_str (for second VECTOR_COSINE), similarity_threshold
            params = [top_k, query_vector_str] + chunk_types + [query_vector_str, similarity_threshold]
            
            # Log a representation of parameters for debugging, not the full vector string
            log_params = [top_k, f"vec_str_len_{len(query_vector_str)}"] + chunk_types + [f"vec_str_len_{len(query_vector_str)}", similarity_threshold]
            logger.info(f"Executing chunk retrieval. SQL: {sql_query.strip()}, Params: {log_params}")
            
            cursor.execute(sql_query, params)
            results = cursor.fetchall()
            
            for row in results:
                chunk_id = row[0]
                chunk_text = row[1] if row[1] else ""
                
                # Handle potential stream objects (for JDBC)
                if hasattr(chunk_text, 'read'):
                    chunk_text = chunk_text.read()
                if isinstance(chunk_text, bytes):
                    chunk_text = chunk_text.decode('utf-8', errors='ignore')
                
                doc_id = row[2]
                chunk_type = row[3]
                chunk_index = row[4]
                score = float(row[5]) if row[5] else 0.0
                
                # Create Document object with chunk information
                doc = Document(
                    id=chunk_id,
                    content=str(chunk_text),
                    score=score
                )
                # Add metadata as an attribute after creation
                doc.metadata = {
                    'doc_id': doc_id,
                    'chunk_type': chunk_type,
                    'chunk_index': chunk_index,
                    'source': 'chunk'
                }
                retrieved_chunks.append(doc)
            
            logger.info(f"Retrieved {len(retrieved_chunks)} chunks above threshold {similarity_threshold}")
            
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            
        finally:
            if cursor:
                cursor.close()
        
        return retrieved_chunks
    
    def get_chunks_for_document(self, 
                              doc_id: str, 
                              chunk_type: Optional[str] = None) -> List[Document]:
        """
        Get all chunks for a specific document
        
        Args:
            doc_id: Document ID
            chunk_type: Optional chunk type filter
            
        Returns:
            List of Document objects representing chunks for the document
        """
        logger.debug(f"Retrieving chunks for document {doc_id}")
        
        sql_query = """
            SELECT chunk_id, chunk_text, chunk_type, chunk_index, chunk_metadata
            FROM RAG.DocumentChunks
            WHERE doc_id = ?
        """
        
        params = [doc_id]
        
        if chunk_type:
            sql_query += " AND chunk_type = ?"
            params.append(chunk_type)
        
        sql_query += " ORDER BY chunk_index"
        
        chunks: List[Document] = []
        cursor = None
        
        try:
            cursor = self.iris_connector.cursor()
            cursor.execute(sql_query, params)
            results = cursor.fetchall()
            
            for row in results:
                chunk_id = row[0]
                chunk_text = row[1] if row[1] else ""
                chunk_type = row[2]
                chunk_index = row[3]
                chunk_metadata = row[4]
                
                chunks.append(Document(
                    id=chunk_id,
                    content=chunk_text,
                    score=1.0,  # No similarity score for direct retrieval
                    metadata={
                        'doc_id': doc_id,
                        'chunk_type': chunk_type,
                        'chunk_index': chunk_index,
                        'chunk_metadata': chunk_metadata,
                        'source': 'chunk'
                    }
                ))
            
            logger.debug(f"Retrieved {len(chunks)} chunks for document {doc_id}")
            
        except Exception as e:
            logger.error(f"Error retrieving chunks for document {doc_id}: {e}")
            
        finally:
            if cursor:
                cursor.close()
        
        return chunks
    
    def get_chunk_context(self, 
                         chunk_id: str, 
                         context_window: int = 2) -> List[Document]:
        """
        Get surrounding chunks for context
        
        Args:
            chunk_id: Target chunk ID
            context_window: Number of chunks before and after to include
            
        Returns:
            List of Document objects including the target chunk and context
        """
        logger.debug(f"Retrieving context for chunk {chunk_id} with window {context_window}")
        
        # First get the target chunk to find its document and index
        sql_get_chunk = """
            SELECT doc_id, chunk_index, chunk_type
            FROM RAG.DocumentChunks
            WHERE chunk_id = ?
        """
        
        cursor = None
        
        try:
            cursor = self.iris_connector.cursor()
            cursor.execute(sql_get_chunk, [chunk_id])
            result = cursor.fetchone()
            
            if not result:
                logger.warning(f"Chunk {chunk_id} not found")
                return []
            
            doc_id = result[0]
            chunk_index = result[1]
            chunk_type = result[2]
            
            # Get surrounding chunks
            min_index = max(0, chunk_index - context_window)
            max_index = chunk_index + context_window
            
            sql_get_context = """
                SELECT chunk_id, chunk_text, chunk_index, chunk_metadata
                FROM RAG.DocumentChunks
                WHERE doc_id = ? 
                  AND chunk_type = ?
                  AND chunk_index BETWEEN ? AND ?
                ORDER BY chunk_index
            """
            
            cursor.execute(sql_get_context, [doc_id, chunk_type, min_index, max_index])
            results = cursor.fetchall()
            
            context_chunks: List[Document] = []
            
            for row in results:
                ctx_chunk_id = row[0]
                chunk_text = row[1] if row[1] else ""
                ctx_chunk_index = row[2]
                chunk_metadata = row[3]
                
                # Mark the target chunk
                is_target = ctx_chunk_id == chunk_id
                
                context_chunks.append(Document(
                    id=ctx_chunk_id,
                    content=chunk_text,
                    score=1.0,
                    metadata={
                        'doc_id': doc_id,
                        'chunk_type': chunk_type,
                        'chunk_index': ctx_chunk_index,
                        'chunk_metadata': chunk_metadata,
                        'source': 'chunk_context',
                        'is_target': is_target
                    }
                ))
            
            logger.debug(f"Retrieved {len(context_chunks)} chunks for context of {chunk_id}")
            return context_chunks
            
        except Exception as e:
            logger.error(f"Error retrieving context for chunk {chunk_id}: {e}")
            return []
            
        finally:
            if cursor:
                cursor.close()
    
    def check_chunk_availability(self) -> Dict[str, Any]:
        """
        Check if chunks are available in the database
        
        Returns:
            Dictionary with chunk availability statistics
        """
        logger.info("Checking chunk availability in database")
        
        cursor = None
        stats = {
            'total_chunks': 0,
            'chunks_with_embeddings': 0,
            'unique_documents': 0,
            'chunk_types': {},
            'available': False
        }
        
        try:
            cursor = self.iris_connector.cursor()
            
            # Total chunks
            cursor.execute("SELECT COUNT(*) FROM RAG.DocumentChunks")
            stats['total_chunks'] = cursor.fetchone()[0]
            
            # Chunks with embeddings
            cursor.execute("SELECT COUNT(*) FROM RAG.DocumentChunks WHERE embedding IS NOT NULL")
            stats['chunks_with_embeddings'] = cursor.fetchone()[0]
            
            # Unique documents with chunks
            cursor.execute("SELECT COUNT(DISTINCT doc_id) FROM RAG.DocumentChunks")
            stats['unique_documents'] = cursor.fetchone()[0]
            
            # Chunk types distribution
            cursor.execute("""
                SELECT chunk_type, COUNT(*) 
                FROM RAG.DocumentChunks 
                GROUP BY chunk_type
            """)
            
            for row in cursor.fetchall():
                chunk_type = row[0]
                count = row[1]
                stats['chunk_types'][chunk_type] = count
            
            stats['available'] = stats['chunks_with_embeddings'] > 0
            
            logger.info(f"Chunk availability: {stats['total_chunks']} total, "
                       f"{stats['chunks_with_embeddings']} with embeddings, "
                       f"{stats['unique_documents']} unique documents")
            
        except Exception as e:
            logger.error(f"Error checking chunk availability: {e}")
            
        finally:
            if cursor:
                cursor.close()
        
        return stats