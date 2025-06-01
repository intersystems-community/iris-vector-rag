"""
Minimal BasicRAG Pipeline V2 - Using EXACT pattern from working techniques
Copies the exact SQL and parameter handling from NodeRAG and CRAG
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List, Dict, Any, Callable
import logging

try:
    import iris
    IRISConnection = iris.IRISConnection
except ImportError:
    IRISConnection = Any

from common.utils import Document, timing_decorator, get_embedding_func, get_llm_func
from common.iris_connector import get_iris_connection
from common.jdbc_stream_utils import read_iris_stream

logger = logging.getLogger(__name__)

class BasicRAGPipelineMinimal:
    def __init__(self, iris_connector: IRISConnection,
                 embedding_func: Callable[[List[str]], List[List[float]]],
                 llm_func: Callable[[str], str],
                 schema: str = "RAG"):
        self.iris_connector = iris_connector
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        self.schema = schema
        logger.info(f"BasicRAGPipelineMinimal initialized with schema: {schema}")

    @timing_decorator
    def retrieve_documents(self, query_text: str, top_k: int = 5, similarity_threshold: float = 0.0) -> List[Document]:
        """
        Retrieve documents using EXACT pattern from working NodeRAG/CRAG
        """
        logger.debug(f"BasicRAG Minimal: Retrieving documents for query: '{query_text[:50]}...'")
        
        # Generate query embedding - use EXACT format from NodeRAG (with brackets)
        query_embedding = self.embedding_func([query_text])[0]
        query_embedding_str = f"[{','.join([f'{x:.10f}' for x in query_embedding])}]"
        
        retrieved_docs = []
        cursor = None
        try:
            cursor = self.iris_connector.cursor()
            
            # Use EXACT SQL pattern from working techniques (no WHERE threshold, filter in Python)
            sql = f"""
                SELECT TOP {top_k * 2} doc_id, title, text_content,
                       VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) AS score
                FROM {self.schema}.SourceDocuments
                WHERE embedding IS NOT NULL
                ORDER BY score DESC
            """
            
            # Pass embedding only ONCE (like working techniques)
            cursor.execute(sql, [query_embedding_str])
            all_results = cursor.fetchall()
            
            logger.info(f"Retrieved {len(all_results)} raw results from database")
            
            # Filter by similarity threshold and limit to top_k (like working techniques)
            filtered_results = []
            for row in all_results:
                score = float(row[3]) if row[3] is not None else 0.0
                if score > similarity_threshold:
                    filtered_results.append(row)
                if len(filtered_results) >= top_k:
                    break
            
            logger.info(f"Filtered to {len(filtered_results)} documents above threshold {similarity_threshold}")
            
            for row in filtered_results:
                doc_id, title, content, score = row
                
                # Handle potential stream objects (like working techniques)
                content = read_iris_stream(content) if content else ""
                title = read_iris_stream(title) if title else ""
                
                # Ensure score is float
                score = float(score) if score is not None else 0.0
                
                doc = Document(
                    id=doc_id,
                    content=content,
                    score=score
                )
                # Store metadata separately (like working techniques)
                doc._metadata = {
                    "title": title,
                    "similarity_score": score,
                    "source": "BasicRAG_Minimal_Fixed"
                }
                retrieved_docs.append(doc)
                
        except Exception as e:
            logger.error(f"Error retrieving documents: {e}")
            print(f"Error retrieving documents: {e}")
            raise
        finally:
            if cursor:
                cursor.close()
        
        return retrieved_docs

    @timing_decorator
    def generate_answer(self, query: str, documents: List[Document]) -> str:
        """Generate answer using LLM based on retrieved documents."""
        if not documents:
            return "I couldn't find any relevant information to answer your question."
        
        # Prepare context from documents
        context_parts = []
        for i, doc in enumerate(documents[:3], 1):  # Use top 3 documents
            # Get metadata from _metadata attribute if it exists
            metadata = getattr(doc, '_metadata', {})
            title = metadata.get('title', 'Untitled')
            score = doc.score or 0
            content_preview = doc.content[:500] if doc.content else ""
            context_parts.append(f"Document {i} (Score: {score:.3f}, Title: {title}):\n{content_preview}...")
        
        context = "\n\n".join(context_parts)
        
        # Create prompt
        prompt = f"""Based on the following context, please answer the question.
 
Context:
{context}
 
Question: {query}
 
Please provide a comprehensive answer based on the information provided in the context. If the context doesn't contain enough information to fully answer the question, please state what information is available and what is missing.
 
Answer:"""
        
        # Generate answer
        try:
            answer = self.llm_func(prompt)
            return answer
        except Exception as e:
            logger.error(f"Error generating answer: {e}")
            return f"Error generating answer: {str(e)}"

    def run(self, query: str, top_k: int = 5) -> Dict[str, Any]:
        """Run the complete RAG pipeline."""
        print(f"\n{'='*50}")
        print(f"BasicRAG Minimal Pipeline - Query: {query}")
        print(f"{'='*50}")
        
        # Retrieve documents
        documents = self.retrieve_documents(query, top_k=top_k)
        
        # Generate answer
        answer = self.generate_answer(query, documents)
        
        # Prepare response
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
                "pipeline": "BasicRAG_Minimal_Fixed",
                "top_k": top_k,
                "num_retrieved": len(documents),
                "uses_hnsw": True
            }
        }
        
        return result


# Example usage
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Initialize components
    iris_conn = get_iris_connection()
    embedding_func = get_embedding_func()
    llm_func = get_llm_func()
    
    # Create pipeline
    pipeline = BasicRAGPipelineMinimal(
        iris_connector=iris_conn,
        embedding_func=embedding_func,
        llm_func=llm_func
    )
    
    # Test query
    test_query = "What are the symptoms of diabetes?"
    result = pipeline.run(test_query)
    
    print(f"Query: {result['query']}")
    print(f"Answer: {result['answer'][:200]}...")
    print(f"Retrieved {result['metadata']['num_retrieved']} documents")