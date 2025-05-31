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
    def retrieve_documents(self, query: str, hypothetical_doc: str, top_k: int = 5, similarity_threshold: float = 0.0) -> List[Document]:
        """
        Retrieve documents using the hypothetical document embedding with HNSW acceleration.
        Uses the proven BasicRAG V2 pattern for reliable results.
        """
        logger.debug(f"HyDE V2: Retrieving documents for hypothetical doc: '{hypothetical_doc[:50]}...'")
        
        # Generate embedding for the hypothetical document (primary search)
        hyde_embedding = self.embedding_func([hypothetical_doc])[0]
        # Use same format as SourceDocuments (comma-separated, no brackets)
        hyde_embedding_str = ','.join([f'{x:.10f}' for x in hyde_embedding])
        
        retrieved_docs = []
        cursor = None
        try:
            cursor = self.iris_connector.cursor()
            
            # Use same SQL pattern as BasicRAG V2 (no WHERE threshold, filter in Python)
            sql = f"""
                SELECT TOP {top_k * 2}
                    doc_id,
                    title,
                    text_content,
                    VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) as similarity_score
                FROM {self.schema}.SourceDocuments
                WHERE embedding IS NOT NULL
                ORDER BY similarity_score DESC
            """
            
            cursor.execute(sql, [hyde_embedding_str])
            all_results = cursor.fetchall()
            
            logger.info(f"Retrieved {len(all_results)} raw results from database")
            
            # Filter by similarity threshold and limit to top_k (like BasicRAG V2)
            filtered_results = []
            for row in all_results:
                score = float(row[3]) if row[3] is not None else 0.0
                if score > similarity_threshold:
                    filtered_results.append(row)
                if len(filtered_results) >= top_k:
                    break
            
            logger.info(f"Filtered to {len(filtered_results)} documents above threshold {similarity_threshold}")
            
            for row in filtered_results:
                doc_id = row[0]
                title = row[1] or ""
                content = row[2] or ""
                similarity = row[3]
                
                # Ensure score is float (like BasicRAG V2)
                similarity = float(similarity) if similarity is not None else 0.0
                
                doc = Document(
                    id=doc_id,
                    content=content,
                    score=similarity
                )
                # Store metadata separately for later use
                doc._metadata = {
                    "title": title,
                    "similarity_score": similarity,
                    "source": "HyDE_V2_HNSW"
                }
                retrieved_docs.append(doc)
                
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
        
        return retrieved_docs
    
    @timing_decorator
    def generate_answer(self, query: str, documents: List[Document], hypothetical_doc: str) -> str:
        """Generate answer using LLM based on retrieved documents and hypothetical document context."""
        if not documents:
            return "I couldn't find any relevant information to answer your question."
        
        # Prepare context from documents (same pattern as BasicRAG V2)
        context_parts = []
        for i, doc in enumerate(documents[:3], 1):  # Use top 3 documents
            # Get metadata from _metadata attribute if it exists
            metadata = getattr(doc, '_metadata', {})
            title = metadata.get('title', 'Untitled')
            score = float(doc.score) if doc.score is not None else 0.0
            content_preview = doc.content[:500] if doc.content else ""
            context_parts.append(f"Document {i} (Score: {score:.3f}, Title: {title}):\n{content_preview}...")
        
        context = "\n\n".join(context_parts)
        
        # Include the hypothetical document in the prompt for HyDE-specific context
        prompt = f"""Based on the following context and hypothetical answer, please answer the question.

Hypothetical Answer (generated for guidance):
{hypothetical_doc[:300]}...

Retrieved Documents:
{context}
 
Question: {query}
 
Please provide a comprehensive answer based primarily on the retrieved documents, using the hypothetical answer as guidance for structure and completeness. If the context doesn't contain enough information to fully answer the question, please state what information is available and what is missing.
 
Answer:"""
        
        # Generate answer
        try:
            answer = self.llm_func(prompt)
            return answer
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