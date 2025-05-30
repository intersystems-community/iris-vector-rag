# noderag/pipeline_v2.py

import logging
from typing import List, Dict, Any, Callable, Optional
from common.utils import Document, timing_decorator

logger = logging.getLogger(__name__)

class NodeRAGPipelineV2:
    """
    Node-based RAG Pipeline V2 with HNSW support
    
    This implementation uses native IRIS VECTOR columns and HNSW indexes
    for accelerated similarity search on both documents and chunks in the _V2 tables.
    """
    
    def __init__(self, iris_connector, embedding_func: Callable, llm_func: Callable):
        self.iris_connector = iris_connector
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        self.schema = "RAG"
    
    @timing_decorator
    def retrieve_documents(self, query: str, top_k: int = 5) -> List[Document]:
        """
        Retrieve documents using HNSW-accelerated vector search
        """
        # Generate query embedding
        query_embedding = self.embedding_func([query])[0]
        query_embedding_str = ','.join([f'{x:.10f}' for x in query_embedding])
        
        retrieved_docs = []
        
        # Retrieve from documents using HNSW on VECTOR column
        sql_query = f"""
            SELECT TOP {top_k} doc_id, title, text_content,
                   VECTOR_COSINE(document_embedding_vector, TO_VECTOR('{query_embedding_str}', 'DOUBLE', 384)) AS score
            FROM {self.schema}.SourceDocuments_V2
            WHERE document_embedding_vector IS NOT NULL
              AND VECTOR_COSINE(document_embedding_vector, TO_VECTOR('{query_embedding_str}', 'DOUBLE', 384)) > 0.1
            ORDER BY score DESC
        """
        
        cursor = None
        try:
            cursor = self.iris_connector.cursor()
            logger.debug("NodeRAG V2 Document Retrieve with HNSW")
            cursor.execute(sql_query)
            results = cursor.fetchall()
            
            for row in results:
                doc_id, title, content, score = row
                doc = Document(
                    id=doc_id,
                    content=content,
                    score=score
                )
                # Store metadata separately
                doc._metadata = {
                    "title": title,
                    "similarity_score": score,
                    "source": "NodeRAG_V2_Documents_HNSW"
                }
                retrieved_docs.append(doc)
                
            print(f"NodeRAG V2: Retrieved {len(retrieved_docs)} documents")
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            print(f"Error retrieving documents: {e}")
        finally:
            if cursor:
                cursor.close()
        
        return retrieved_docs
    
    @timing_decorator
    def retrieve_chunks(self, query: str, top_k: int = 10) -> List[Document]:
        """
        Retrieve chunks using HNSW-accelerated vector search
        """
        # Generate query embedding
        query_embedding = self.embedding_func([query])[0]
        query_embedding_str = ','.join([f'{x:.10f}' for x in query_embedding])
        
        retrieved_chunks = []
        
        # Retrieve from chunks using HNSW on VECTOR column
        sql_query = f"""
            SELECT TOP {top_k} c.chunk_id, c.doc_id, c.chunk_text, c.chunk_index,
                   d.title,
                   VECTOR_COSINE(c.chunk_embedding_vector, TO_VECTOR('{query_embedding_str}', 'DOUBLE', 384)) AS score
            FROM {self.schema}.DocumentChunks_V2 c
            JOIN {self.schema}.SourceDocuments_V2 d ON c.doc_id = d.doc_id
            WHERE c.chunk_embedding_vector IS NOT NULL
              AND VECTOR_COSINE(c.chunk_embedding_vector, TO_VECTOR('{query_embedding_str}', 'DOUBLE', 384)) > 0.1
            ORDER BY score DESC
        """
        
        cursor = None
        try:
            cursor = self.iris_connector.cursor()
            logger.debug("NodeRAG V2 Chunk Retrieve with HNSW")
            cursor.execute(sql_query)
            results = cursor.fetchall()
            
            for row in results:
                chunk_id, doc_id, chunk_text, chunk_index, title, score = row
                chunk = Document(
                    id=chunk_id,
                    content=chunk_text,
                    score=score
                )
                # Store metadata separately
                chunk._metadata = {
                    "doc_id": doc_id,
                    "chunk_index": chunk_index,
                    "title": title,
                    "similarity_score": score,
                    "source": "NodeRAG_V2_Chunks_HNSW"
                }
                retrieved_chunks.append(chunk)
                
            print(f"NodeRAG V2: Retrieved {len(retrieved_chunks)} chunks")
        except Exception as e:
            logger.error(f"Error retrieving chunks: {e}")
            print(f"Error retrieving chunks: {e}")
        finally:
            if cursor:
                cursor.close()
        
        return retrieved_chunks
    
    @timing_decorator
    def merge_and_rank_results(self, documents: List[Document], chunks: List[Document], top_k: int = 5) -> List[Document]:
        """
        Merge and rank results from documents and chunks
        """
        # Create a unified list with adjusted scores
        all_results = []
        
        # Add documents with higher weight
        for doc in documents:
            doc._node_type = "document"
            doc._adjusted_score = (doc.score or 0) * 1.2  # Boost document scores
            all_results.append(doc)
        
        # Add chunks
        for chunk in chunks:
            chunk._node_type = "chunk"
            chunk._adjusted_score = chunk.score or 0
            all_results.append(chunk)
        
        # Sort by adjusted score
        all_results.sort(key=lambda x: x._adjusted_score, reverse=True)
        
        # Deduplicate by content similarity (keep diverse results)
        final_results = []
        seen_content = set()
        
        for result in all_results:
            # Create a content fingerprint (first 100 chars)
            content_fingerprint = result.content[:100] if result.content else ""
            
            if content_fingerprint not in seen_content:
                final_results.append(result)
                seen_content.add(content_fingerprint)
                
                if len(final_results) >= top_k:
                    break
        
        print(f"NodeRAG V2: Merged to {len(final_results)} unique results")
        return final_results
    
    @timing_decorator
    def generate_answer(self, query: str, nodes: List[Document]) -> str:
        """
        Generate answer using retrieved nodes (documents and chunks)
        """
        if not nodes:
            return "I couldn't find relevant information to answer your question."
        
        # Prepare context from nodes
        context_parts = []
        for i, node in enumerate(nodes[:5], 1):
            metadata = getattr(node, '_metadata', {})
            node_type = getattr(node, '_node_type', 'unknown')
            title = metadata.get('title', 'Untitled')
            score = node.score or 0
            
            content_preview = node.content[:400] if node.content else ""
            
            if node_type == "document":
                context_parts.append(
                    f"Document {i} (Score: {score:.3f}, Title: {title}):\n{content_preview}..."
                )
            else:
                chunk_index = metadata.get('chunk_index', 'unknown')
                context_parts.append(
                    f"Chunk {i} (Score: {score:.3f}, From: {title}, Part {chunk_index}):\n{content_preview}..."
                )
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""Based on the following documents and text chunks, answer the question comprehensively.
The context includes both full documents and specific relevant chunks.

Context:
{context}

Question: {query}

Please provide a detailed answer based on the available information:"""
        
        try:
            response = self.llm_func(prompt)
            return response
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"
    
    def run(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Execute the NodeRAG V2 pipeline with HNSW acceleration
        """
        print(f"\n{'='*50}")
        print(f"NodeRAG V2 Pipeline (HNSW) - Query: {query}")
        print(f"{'='*50}")
        
        # Retrieve from both documents and chunks
        documents = self.retrieve_documents(query, top_k=top_k)
        chunks = self.retrieve_chunks(query, top_k=top_k*2)  # Get more chunks
        
        # Merge and rank results
        merged_nodes = self.merge_and_rank_results(documents, chunks, top_k=top_k)
        
        # Generate answer
        answer = self.generate_answer(query, merged_nodes)
        
        # Prepare results
        result = {
            "query": query,
            "answer": answer,
            "retrieved_nodes": [
                {
                    "id": node.id,
                    "type": getattr(node, '_node_type', 'unknown'),
                    "content": node.content[:200] + "..." if len(node.content) > 200 else node.content,
                    "metadata": getattr(node, '_metadata', {})
                }
                for node in merged_nodes
            ],
            "metadata": {
                "pipeline": "NodeRAG_V2",
                "uses_hnsw": True,
                "top_k": top_k,
                "num_documents_retrieved": len(documents),
                "num_chunks_retrieved": len(chunks),
                "num_nodes_used": len(merged_nodes)
            }
        }
        
        return result