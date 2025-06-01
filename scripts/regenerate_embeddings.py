#!/usr/bin/env python3
"""
Regenerate Embeddings for all documents in RAG.SourceDocuments.

This script will:
1. Optionally set all existing embeddings to NULL.
2. Use the generate_document_embeddings utility to create and store new embeddings.
   This ensures consistency in embedding model and dimension.
"""
import sys
sys.path.append('.')
import argparse
import logging
from datetime import datetime

from common.iris_connector import get_iris_connection
from common.embedding_utils import get_embedding_model, generate_document_embeddings

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'regenerate_embeddings_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def nullify_embeddings(iris_conn, schema="RAG"):
    """Sets all embeddings in RAG.SourceDocuments to NULL."""
    logger.info(f"Setting all embeddings in {schema}.SourceDocuments to NULL...")
    try:
        cursor = iris_conn.cursor()
        sql = f"UPDATE {schema}.SourceDocuments SET embedding = NULL WHERE embedding IS NOT NULL"
        cursor.execute(sql)
        iris_conn.commit()
        updated_rows = cursor.rowcount
        logger.info(f"Successfully set embeddings to NULL for {updated_rows} rows.")
        cursor.close()
        return True
    except Exception as e:
        logger.error(f"Error nullifying embeddings: {e}")
        iris_conn.rollback()
        if cursor:
            cursor.close()
        return False

def main():
    parser = argparse.ArgumentParser(description="Regenerate embeddings for RAG.SourceDocuments.")
    parser.add_argument(
        "--skip-nullify",
        action="store_true",
        help="Skip the step of setting existing embeddings to NULL. Only populates for currently NULL embeddings."
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing documents during embedding generation."
    )
    parser.add_argument(
        "--embedding-model-name",
        type=str,
        default="sentence-transformers/all-MiniLM-L6-v2",
        help="Name of the sentence-transformer model to use."
    )

    args = parser.parse_args()

    logger.info("Starting embedding regeneration process...")
    iris_conn = None
    try:
        iris_conn = get_iris_connection()
        logger.info("Successfully connected to IRIS.")

        if not args.skip_nullify:
            if not nullify_embeddings(iris_conn):
                logger.error("Failed to nullify existing embeddings. Aborting.")
                return
        else:
            logger.info("Skipping nullification of existing embeddings.")

        logger.info(f"Initializing embedding model: {args.embedding_model_name}")
        # Pass mock=False to ensure real model is used if sentence-transformers is available
        embedding_model = get_embedding_model(model_name=args.embedding_model_name, mock=False) 

        logger.info("Starting to generate and update document embeddings...")
        # Determine schema to use, defaulting to RAG
        schema_to_use = "RAG" # Default, can be made configurable if needed later
        
        stats = generate_document_embeddings(
            connection=iris_conn,
            embedding_model=embedding_model,
            batch_size=args.batch_size,
            schema=schema_to_use # Pass the schema
        )
        
        logger.info("Embedding regeneration process finished.")
        logger.info(f"Stats: {stats}")

    except Exception as e:
        logger.error(f"An error occurred during the regeneration process: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if iris_conn:
            iris_conn.close()
            logger.info("IRIS connection closed.")

if __name__ == "__main__":
    main()