"""
Basic RAG Pipeline - Simple Version
Focuses on using TO_VECTOR() with CLOB-like embeddings from RAG.SourceDocuments
due to current JDBC interoperation characteristics.
Designed for the 10 sample documents.
"""

import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from typing import List, Dict, Any, Callable
import logging
# import yaml # For config, though mostly used by common utils

try:
    import iris
    IRISConnection = iris.IRISConnection
except ImportError:
    IRISConnection = Any # type: ignore

from common.utils import Document, timing_decorator, get_embedding_func, get_llm_func
from common.iris_connector import get_iris_connection # Implicitly uses config.yaml

logger = logging.getLogger(__name__)

class BasicRAGPipelineSimple:
    def __init__(self, iris_connector: IRISConnection,
                 embedding_func: Callable[[List[str]], List[List[float]]],
                 llm_func: Callable[[str], str],
                 schema: str = "RAG"):
        self.iris_connector = iris_connector
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        self.schema = schema
        logger.info(f"BasicRAGPipelineSimple initialized with schema: {self.schema}")

    @timing_decorator
    def retrieve_documents(self, query_text: str, top_k: int = 5) -> List[Document]:
        """
        Retrieves documents from IRIS using TO_VECTOR() on the 'embedding' column.
        The 'embedding' column in RAG.SourceDocuments is treated as a string (CLOB-like) 
        of comma-separated floats (e.g., "0.1,0.2,...") for the purpose of this query
        due to JDBC behavior. It needs to be formatted as '[0.1,0.2,...]' by
        concatenating '[' and ']' before TO_VECTOR can parse it.
        """
        logger.debug(f"BasicRAGSimple: Retrieving documents for query: '{query_text[:50]}...'")
        
        query_embedding_list = self.embedding_func([query_text])[0]
        # Format for TO_VECTOR(?): needs to be a string like "[0.1,0.2,...,0.N]"
        # Using fixed-point notation for consistency with other pipelines, though str() should also work.
        query_embedding_str = f"[{','.join([f'{x:.10f}' for x in query_embedding_list])}]"
        
        retrieved_docs: List[Document] = []
        cursor = None
        try:
            cursor = self.iris_connector.cursor()
            
            # SQL query using TO_VECTOR on the 'embedding' field (treated as string)
            # and on the query embedding string.
            # VECTOR_COSINE returns similarity, higher is better.
            sql = f"""
                SELECT TOP ?
                    doc_id,
                    title,
                    text_content,
                    VECTOR_COSINE(TO_VECTOR('[' || embedding || ']'), TO_VECTOR(?)) as similarity_score
                FROM {self.schema}.SourceDocuments
                WHERE embedding IS NOT NULL AND embedding <> '' 
                ORDER BY similarity_score DESC
            """
            # This approach is per user feedback regarding current JDBC limitations,
            # requiring explicit TO_VECTOR on the string-formatted stored embedding.
            
            logger.debug(f"Executing SQL with TOP {top_k} and query embedding (first 50 chars): {query_embedding_str[:50]}...")
            cursor.execute(sql, (top_k, query_embedding_str))
            
            rows = cursor.fetchall()
            logger.info(f"Retrieved {len(rows)} raw results from database for query '{query_text[:50]}...'")
            
            for row in rows:
                doc_id, title, content, similarity = row[0], row[1], row[2], row[3]
                
                score = float(similarity) if similarity is not None else 0.0

                doc = Document(
                    id=str(doc_id), 
                    content=content or "",
                    score=score
                )
                doc._metadata = {
                    "title": title or "",
                    "similarity_score": score,
                    "source": "BasicRAG_Simple_TO_VECTOR_on_CLOB_like"
                }
                retrieved_docs.append(doc)
                
        except Exception as e:
            logger.error(f"Error retrieving documents for query '{query_text[:50]}...': {e}", exc_info=True)
            raise 
        finally:
            if cursor:
                cursor.close()
        
        logger.info(f"Returning {len(retrieved_docs)} documents after processing for query '{query_text[:50]}...'")
        return retrieved_docs

    @timing_decorator
    def generate_answer(self, query: str, documents: List[Document]) -> str:
        """Generate answer using LLM based on retrieved documents."""
        if not documents:
            logger.warning(f"No documents provided to generate_answer for query: '{query[:50]}...'")
            return "I couldn't find any relevant information to answer your question based on the documents provided."
        
        context_parts = []
        for i, doc in enumerate(documents[:3], 1): 
            metadata = getattr(doc, '_metadata', {})
            title = metadata.get('title', 'Untitled')
            score = doc.score if doc.score is not None else 0.0
            content_preview = (doc.content[:200] + "...") if doc.content and len(doc.content) > 200 else (doc.content or "")
            
            context_parts.append(f"Document {i} (ID: {doc.id}, Score: {score:.4f}, Title: {title}):\n{content_preview}")
        
        context = "\n\n".join(context_parts)
        
        prompt = f"""Based on the following context, please answer the question.
Context:
{context}

Question: {query}

Answer:"""
        
        logger.debug(f"Generating answer with LLM for query: '{query[:50]}...'")
        try:
            answer = self.llm_func(prompt)
            return answer
        except Exception as e:
            logger.error(f"Error generating answer with LLM: {e}", exc_info=True)
            return f"Error generating answer: {str(e)}"

    def run(self, query: str, top_k: int = 3) -> Dict[str, Any]:
        """Run the complete RAG pipeline."""
        logger.info(f"Running BasicRAGSimple pipeline for query: '{query[:50]}...' with top_k={top_k}")
        
        documents = self.retrieve_documents(query, top_k=top_k)
        answer = self.generate_answer(query, documents)
        
        result = {
            "query": query,
            "answer": answer,
            "retrieved_documents": [
                {
                    "id": doc.id,
                    "content": (doc.content[:100] + "..." if doc.content and len(doc.content) > 100 else (doc.content or "")),
                    "metadata": getattr(doc, '_metadata', {}) 
                }
                for doc in documents
            ],
            "metadata": { 
                "pipeline_version": "BasicRAG_Simple_TO_VECTOR_on_CLOB_like",
                "top_k_retrieval": top_k,
                "num_retrieved": len(documents)
            }
        }
        logger.info(f"Pipeline run completed for query: '{query[:50]}...'. Retrieved {len(documents)} docs.")
        return result

if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    
    logger.info("Attempting to run BasicRAGPipelineSimple example...")
    
    iris_conn_main = None # Define outside try for finally block
    try:
        iris_conn_main = get_iris_connection()
        if iris_conn_main is None:
            logger.error("Failed to get IRIS connection for example run. Exiting.")
            sys.exit(1)
            
        embedding_fn = get_embedding_func()
        
        def mock_llm_for_example(prompt_str: str) -> str:
            return f"Mock LLM response to: {prompt_str[:100]}..."

        llm_fn = mock_llm_for_example 
        
        pipeline = BasicRAGPipelineSimple(
            iris_connector=iris_conn_main,
            embedding_func=embedding_fn,
            llm_func=llm_fn,
            schema="RAG" 
        )
        
        test_query = "What are the main applications of gene editing?"
        logger.info(f"Running pipeline with test query: '{test_query}'")
        
        pipeline_result = pipeline.run(test_query, top_k=2)
        
        print("\n--- Pipeline Result ---")
        print(f"Query: {pipeline_result['query']}")
        print(f"Answer: {pipeline_result['answer']}")
        print(f"Number of Retrieved Documents: {len(pipeline_result['retrieved_documents'])}")
        for idx, doc_res in enumerate(pipeline_result['retrieved_documents']):
            print(f"  Doc {idx+1}: ID={doc_res['id']}, Score={doc_res['metadata'].get('similarity_score', 'N/A')}")
            print(f"     Content Snippet: {doc_res['content']}")
        print("-----------------------\n")

    except Exception as e:
        logger.error(f"Error during example run: {e}", exc_info=True)
    finally:
        if iris_conn_main: # Check if it was successfully assigned
            iris_conn_main.close()
            logger.info("IRIS connection closed for example run.")