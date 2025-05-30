# hyde/pipeline_v2.py

import logging
from typing import List, Dict, Any, Callable
from common.utils import Document, timing_decorator

logger = logging.getLogger(__name__)

class HyDEPipelineV2:
    """
    Hypothetical Document Embeddings (HyDE) Pipeline V2 with HNSW support
    
    This implementation uses native IRIS VECTOR columns and HNSW indexes
    for accelerated similarity search on the _V2 tables.
    """
    
    def __init__(self, iris_connector, embedding_func: Callable, llm_func: Callable):
        self.iris_connector = iris_connector
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        self.schema = "RAG"
    
    @timing_decorator
    def generate_hypothetical_document(self, query: str) -> str:
        """
        Generate a hypothetical document that would answer the query
        """
        prompt = f"""Given the following question, write a detailed, factual paragraph that would answer this question. 
Write as if you are writing an excerpt from a scientific paper or medical textbook.

Question: {query}

Hypothetical Answer:"""
        
        try:
            hypothetical_doc = self.llm_func(prompt)
            print(f"HyDE V2: Generated hypothetical document of length {len(hypothetical_doc)}")
            return hypothetical_doc
        except Exception as e:
            logger.error(f"Error generating hypothetical document: {e}")
            # Fallback to using the query itself
            return query
    
    @timing_decorator
    def retrieve_documents(self, query: str, hypothetical_doc: str, top_k: int = 5) -> List[Document]:
        """
        Retrieve documents using the hypothetical document embedding with HNSW acceleration
        """
        # Generate embedding for the hypothetical document
        hyde_embedding = self.embedding_func([hypothetical_doc])[0]
        hyde_embedding_str = ','.join(map(str, hyde_embedding))
        
        # Also generate embedding for the original query for hybrid scoring
        query_embedding = self.embedding_func([query])[0]
        query_embedding_str = ','.join([f'{x:.10f}' for x in query_embedding])
        
        retrieved_docs = []
        
        # Retrieve using HNSW on VECTOR column with hybrid scoring
        sql_query = f"""
            SELECT TOP {top_k} doc_id, title, text_content,
                   VECTOR_COSINE(document_embedding_vector, TO_VECTOR('{hyde_embedding_str}', 'DOUBLE', 384)) as hyde_score,
                   VECTOR_COSINE(document_embedding_vector, TO_VECTOR('{query_embedding_str}', 'DOUBLE', 384)) as query_score,
                   (0.7 * VECTOR_COSINE(document_embedding_vector, TO_VECTOR('{hyde_embedding_str}', 'DOUBLE', 384)) +
                    0.3 * VECTOR_COSINE(document_embedding_vector, TO_VECTOR('{query_embedding_str}', 'DOUBLE', 384))) as combined_score
            FROM {self.schema}.SourceDocuments_V2
            WHERE document_embedding_vector IS NOT NULL
              AND VECTOR_COSINE(document_embedding_vector, TO_VECTOR('{hyde_embedding_str}', 'DOUBLE', 384)) > 0.1
            ORDER BY combined_score DESC
        """
        
        cursor = None
        try:
            cursor = self.iris_connector.cursor()
            logger.debug("HyDE V2 Document Retrieve with HNSW")
            cursor.execute(sql_query)
            results = cursor.fetchall()
            
            for row in results:
                doc_id, title, content, hyde_score, query_score, combined_score = row
                doc = Document(
                    id=doc_id,
                    content=content,
                    score=combined_score
                )
                # Store metadata separately
                doc._metadata = {
                    "title": title,
                    "hyde_score": hyde_score,
                    "query_score": query_score,
                    "combined_score": combined_score,
                    "source": "HyDE_V2_HNSW"
                }
                retrieved_docs.append(doc)
                
            print(f"HyDE V2: Retrieved {len(retrieved_docs)} documents using hypothetical document embedding")
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            print(f"Error retrieving documents: {e}")
        finally:
            if cursor:
                cursor.close()
        
        return retrieved_docs
    
    @timing_decorator
    def generate_answer(self, query: str, documents: List[Document], hypothetical_doc: str) -> str:
        """
        Generate answer using retrieved documents and the hypothetical document context
        """
        if not documents:
            return "I couldn't find relevant information to answer your question."
        
        # Prepare context from documents
        context_parts = []
        for i, doc in enumerate(documents[:3], 1):
            metadata = getattr(doc, '_metadata', {})
            title = metadata.get('title', 'Untitled')
            hyde_score = metadata.get('hyde_score', 0)
            query_score = metadata.get('query_score', 0)
            
            content_preview = doc.content[:500] if doc.content else ""
            context_parts.append(
                f"Document {i} (HyDE Score: {hyde_score:.3f}, Query Score: {query_score:.3f}, Title: {title}):\n{content_preview}..."
            )
        
        context = "\n\n".join(context_parts)
        
        # Include the hypothetical document in the prompt for comparison
        prompt = f"""Based on the following retrieved documents and a hypothetical answer, provide a comprehensive response to the question.

Hypothetical Answer (generated):
{hypothetical_doc[:300]}...

Retrieved Documents:
{context}

Question: {query}

Please provide a factual answer based primarily on the retrieved documents, using the hypothetical answer only as a guide for structure and completeness:"""
        
        try:
            response = self.llm_func(prompt)
            return response
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"
    
    def run(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """
        Execute the HyDE V2 pipeline with HNSW acceleration
        """
        print(f"\n{'='*50}")
        print(f"HyDE V2 Pipeline (HNSW) - Query: {query}")
        print(f"{'='*50}")
        
        # Generate hypothetical document
        hypothetical_doc = self.generate_hypothetical_document(query)
        print(f"Hypothetical document preview: {hypothetical_doc[:100]}...")
        
        # Retrieve documents using hypothetical document embedding
        documents = self.retrieve_documents(query, hypothetical_doc, top_k=top_k)
        
        # Generate answer
        answer = self.generate_answer(query, documents, hypothetical_doc)
        
        # Prepare results
        result = {
            "query": query,
            "answer": answer,
            "hypothetical_document": hypothetical_doc,
            "retrieved_documents": [
                {
                    "id": doc.id,
                    "content": doc.content[:200] + "..." if len(doc.content) > 200 else doc.content,
                    "metadata": getattr(doc, '_metadata', {})
                }
                for doc in documents
            ],
            "metadata": {
                "pipeline": "HyDE_V2",
                "uses_hnsw": True,
                "top_k": top_k,
                "num_documents_retrieved": len(documents)
            }
        }
        
        return result