import os
import sys
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

import logging
from iris_rag.pipelines.hyde import HyDERAGPipeline # Updated import
from common.iris_connector import get_iris_connection # Updated import
from common.utils import get_embedding_func, get_llm_func # Updated import

# Configure basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_hyde_document_retrieval():
    logger.info("Starting HyDE document retrieval test...")
    db_conn = None
    try:
        db_conn = get_iris_connection()
        if db_conn is None:
            logger.error("Failed to get IRIS connection for HyDE test.")
            raise ConnectionError("Failed to get IRIS connection for HyDE test.")

        embed_fn = get_embedding_func()
        llm_fn = get_llm_func(provider="stub") 

        pipeline = HyDEPipeline(
            iris_connector=db_conn,
            embedding_func=embed_fn,
            llm_func=llm_fn
        )

        test_query = "What are the effects of climate change on polar bears?"
        logger.info(f"Test query: '{test_query}'")
        
        hypothetical_doc_text = pipeline._generate_hypothetical_document(test_query)
        logger.info(f"Generated hypothetical document text: '{hypothetical_doc_text}'")
        
        hypothetical_doc_embedding = pipeline.embedding_func([hypothetical_doc_text])[0]
        logger.info(f"Hypothetical document embedding (first 5 elements): {hypothetical_doc_embedding[:5]}")

        # Fetch sample embeddings from the database
        cursor = db_conn.cursor()
        sample_sql = "SELECT TOP 3 doc_id, embedding FROM RAG.SourceDocuments WHERE embedding IS NOT NULL AND embedding NOT LIKE '0.1,0.1,0.1%'"
        logger.info(f"Executing sample SQL: {sample_sql}")
        cursor.execute(sample_sql)
        sample_embeddings = cursor.fetchall()
        logger.info(f"Fetched {len(sample_embeddings)} sample embeddings from DB:")
        for i, row in enumerate(sample_embeddings):
            logger.info(f"  Sample DB Doc {row[0]} Embedding (first 70 chars): {str(row[1])[:70]}...")
        cursor.close()
        
        # Using an extremely permissive similarity threshold for testing
        retrieved_docs = pipeline.retrieve_documents(test_query, top_k=3, similarity_threshold=0.0)

        logger.info(f"Number of documents retrieved: {len(retrieved_docs)}")

        assert len(retrieved_docs) > 0, "HyDE should retrieve at least one document."

        logger.info("Retrieved documents:")
        for i, doc in enumerate(retrieved_docs):
            logger.info(f"  Doc {i+1}: ID={doc.id}, Score={doc.score:.4f}, Content='{doc.content[:100]}...'")
        
        logger.info("HyDE document retrieval test PASSED.")

    except ConnectionError as ce:
        logger.error(f"Connection Error: {ce}")
        assert False, f"Test failed due to connection error: {ce}"
    except Exception as e:
        logger.error(f"An unexpected error occurred: {e}", exc_info=True)
        assert False, f"Test failed due to an unexpected error: {e}"
    finally:
        if db_conn:
            try:
                db_conn.close()
                logger.info("Database connection closed.")
            except Exception as e_close:
                logger.error(f"Error closing DB connection: {e_close}")

if __name__ == "__main__":
    test_hyde_document_retrieval()