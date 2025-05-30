# crag/pipeline_v2.py

import logging
from typing import List, Dict, Any, Callable, Optional
from common.utils import Document, timing_decorator
import numpy as np

logger = logging.getLogger(__name__)

class CRAGPipelineV2:
    """
    Corrective RAG (CRAG) Pipeline V2 with HNSW support
    
    This implementation uses native IRIS VECTOR columns and HNSW indexes
    for accelerated similarity search on the _V2 tables.
    """
    
    def __init__(self, iris_connector, embedding_func: Callable, llm_func: Callable):
        self.iris_connector = iris_connector
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        self.schema = "RAG"
    
    @timing_decorator
    def retrieve_documents(self, query: str, top_k: int = 5, similarity_threshold: float = 0.3) -> List[Document]:
        """
        Retrieve documents using HNSW-accelerated vector search with corrective mechanisms
        """
        # Generate query embedding
        query_embedding = self.embedding_func([query])[0]
        query_embedding_str = ','.join([f'{x:.10f}' for x in query_embedding])
        
        retrieved_docs = []
        
        # Initial retrieval using HNSW on VECTOR column
        sql_query = f"""
            SELECT TOP {top_k * 2} doc_id, title, text_content,
                   VECTOR_COSINE(document_embedding_vector, TO_VECTOR('{query_embedding_str}', 'DOUBLE', 384)) AS score
            FROM {self.schema}.SourceDocuments_V2
            WHERE document_embedding_vector IS NOT NULL
              AND VECTOR_COSINE(document_embedding_vector, TO_VECTOR('{query_embedding_str}', 'DOUBLE', 384)) > {similarity_threshold}
            ORDER BY score DESC
        """
        
        cursor = None
        try:
            cursor = self.iris_connector.cursor()
            logger.debug(f"CRAG V2 Document Retrieve with threshold {similarity_threshold}")
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
                    "source": "CRAG_V2_HNSW"
                }
                retrieved_docs.append(doc)
                
            print(f"CRAG V2: Initial retrieval found {len(retrieved_docs)} documents above threshold {similarity_threshold}")
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            print(f"Error retrieving documents: {e}")
        finally:
            if cursor:
                cursor.close()
        
        # Corrective step: If not enough high-quality results, lower threshold
        if len(retrieved_docs) < top_k and similarity_threshold > 0.1:
            print(f"CRAG V2: Applying corrective retrieval with lower threshold")
            additional_docs = self.retrieve_documents(
                query, 
                top_k=top_k - len(retrieved_docs), 
                similarity_threshold=similarity_threshold * 0.7
            )
            
            # Add only unique documents
            existing_ids = {doc.id for doc in retrieved_docs}
            for doc in additional_docs:
                if doc.id not in existing_ids:
                    retrieved_docs.append(doc)
        
        # Sort by score and return top_k
        retrieved_docs.sort(key=lambda x: x.score or 0, reverse=True)
        return retrieved_docs[:top_k]
    
    @timing_decorator
    def generate_answer(self, query: str, documents: List[Document]) -> str:
        """
        Generate answer using retrieved documents with corrective context
        """
        if not documents:
            return "I couldn't find relevant information to answer your question."
        
        # Prepare context with corrective information
        context_parts = []
        for i, doc in enumerate(documents[:3], 1):
            metadata = getattr(doc, '_metadata', {})
            title = metadata.get('title', 'Untitled')
            score = doc.score or 0
            
            # Add confidence indicator based on score
            confidence = "High" if score > 0.7 else "Medium" if score > 0.5 else "Low"
            
            content_preview = doc.content[:500] if doc.content else ""
            context_parts.append(
                f"Document {i} (Confidence: {confidence}, Score: {score:.3f}, Title: {title}):\n{content_preview}..."
            )
        
        context = "\n\n".join(context_parts)
        
        # Include corrective instructions in the prompt
        prompt = f"""Based on the following documents with varying confidence levels, answer the question. 
Pay more attention to high-confidence documents, but consider all provided information.
If the information seems incomplete or contradictory, acknowledge this in your answer.

Context:
{context}

Question: {query}

Please provide a comprehensive answer based on the available information:"""
        
        try:
            response = self.llm_func(prompt)
            return response
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"
    
    def run(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Execute the CRAG V2 pipeline with HNSW acceleration
        """
        print(f"\n{'='*50}")
        print(f"CRAG V2 Pipeline (HNSW) - Query: {query}")
        print(f"{'='*50}")
        
        # Retrieve documents with corrective mechanisms
        documents = self.retrieve_documents(query, top_k=top_k)
        
        # Generate answer
        answer = self.generate_answer(query, documents)
        
        # Prepare results
        result = {
            "query": query,
            "answer": answer,
            "retrieved_documents": [
                {
                    "id": doc.id,
                    "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                    "metadata": getattr(doc, '_metadata', {})
                }
                for doc in documents
            ],
            "metadata": {
                "pipeline": "CRAG_V2",
                "uses_hnsw": True,
                "top_k": top_k,
                "num_documents_retrieved": len(documents)
            }
        }
        
        return result