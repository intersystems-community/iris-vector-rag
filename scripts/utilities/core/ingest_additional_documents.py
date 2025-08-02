#!/usr/bin/env python3
"""
Ingest Additional Documents
===========================

This script ingests a specified number of additional documents into the
SourceDocuments table without cleaning or modifying existing data.
It assumes the database schema (RAG.SourceDocuments table) already exists.

Example Usage:
python core_scripts/ingest_additional_documents.py 100
"""

import os
import sys
import time
import logging
import argparse
from pathlib import Path

# Add project root to path to allow importing project modules
# Assumes 'core_scripts' is at the project root level, alongside 'common', 'data', etc.
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.iris_connector import get_iris_connection
from common.utils import get_embedding_func
from data.pmc_processor import process_pmc_files # Used to iterate over PMC documents

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class DocumentIngestor:
    def __init__(self, num_docs_to_ingest):
        self.schema = "RAG"  # Database schema name
        self.num_docs_to_ingest = num_docs_to_ingest
        self.embedding_func = None

    def ingest_documents(self):
        """
        Ingests a specified number of additional documents into the SourceDocuments table.
        Skips documents if they already exist (based on doc_id primary key).
        """
        logger.info(f"📚 Attempting to ingest {self.num_docs_to_ingest} additional documents into {self.schema}.SourceDocuments")

        try:
            # Initialize embedding function
            self.embedding_func = get_embedding_func()
            if not self.embedding_func:
                logger.error("   ❌ Failed to initialize embedding function. Aborting.")
                return False
            logger.info("   ✅ Embedding function initialized")

            # Determine data directory (assumes 'data' is in project root)
            data_dir = Path(__file__).parent.parent / "data"
            if not data_dir.exists() or not data_dir.is_dir():
                logger.error(f"   ❌ Data directory not found at {data_dir}. Aborting.")
                return False
            logger.info(f"   📁 Using data directory: {data_dir}")

            conn = get_iris_connection()
            cursor = conn.cursor()
            logger.info("   ✅ Database connection established")

            docs_successfully_ingested = 0
            docs_attempted_from_source = 0

            # process_pmc_files yields document data from the specified directory
            # The 'limit' parameter controls how many documents it tries to process from the source
            for doc_data in process_pmc_files(str(data_dir), limit=self.num_docs_to_ingest):
                docs_attempted_from_source += 1
                
                doc_id = doc_data.get('doc_id', 'unknown_id')
                doc_title = doc_data.get('title', 'No Title')
                doc_content = doc_data.get('content', '')

                if not doc_content.strip():
                    logger.warning(f"   ⚠️  Document {doc_id} has no content. Skipping.")
                    continue

                try:
                    # Generate embedding for the document content
                    embedding = self.embedding_func([doc_content])[0]
                    # Convert embedding to string format for IRIS TO_VECTOR function, ensuring float format
                    embedding_vector_str = f"[{','.join([f'{x:.8f}' for x in embedding])}]"

                    # SQL to insert document data
                    insert_sql = f"""
                        INSERT INTO {self.schema}.SourceDocuments
                        (doc_id, title, text_content, embedding)
                        VALUES (?, ?, ?, TO_VECTOR(?))
                    """
                    
                    cursor.execute(insert_sql, [
                        doc_id,
                        doc_title,
                        doc_content,
                        embedding_vector_str
                    ])
                    
                    docs_successfully_ingested += 1
                    
                    if docs_successfully_ingested % 50 == 0 and docs_successfully_ingested > 0:
                        logger.info(f"   📄 Successfully ingested {docs_successfully_ingested} new documents (processed {docs_attempted_from_source} from source files)")

                except Exception as e:
                    # This will catch errors like primary key violation (document already exists)
                    # or other database/embedding issues.
                    if "PRIMARY KEY constraint" in str(e) or "unique constraint" in str(e).lower(): # Adapt based on actual JDBC error
                        logger.info(f"   ↪️ Document {doc_id} likely already exists. Skipping. (Error: {e})")
                    else:
                        logger.warning(f"   ⚠️  Error processing/inserting document {doc_id}: {e}. Skipping.")
                    continue
            
            conn.commit()  # Commit all successful insertions
            logger.info("   ✅ All pending changes committed to the database.")

            cursor.close()
            conn.close()
            logger.info("   ✅ Database connection closed.")

            logger.info(f"✅ INGESTION COMPLETE: Successfully ingested {docs_successfully_ingested} new documents.")
            if docs_attempted_from_source > 0:
                 logger.info(f"   Processed {docs_attempted_from_source} files from the source directory for this batch.")
            if docs_attempted_from_source > docs_successfully_ingested:
                skipped_or_failed = docs_attempted_from_source - docs_successfully_ingested
                logger.info(f"   {skipped_or_failed} documents were skipped or failed (e.g., duplicates, empty content, errors).")
            
            return True # Indicates the process ran, even if 0 new docs were added (e.g., all were duplicates)

        except Exception as e:
            logger.error(f"❌ INGESTION FAILED CATASTROPHICALLY: {e}", exc_info=True)
            # Try to close connection if it was opened
            try:
                if 'conn' in locals() and conn:
                    conn.close()
            except Exception as ex_close:
                logger.error(f"   Failed to close connection during error handling: {ex_close}")
            return False

def main():
    parser = argparse.ArgumentParser(
        description=f"Ingest additional documents into the {DocumentIngestor('').schema}.SourceDocuments table.",
        formatter_class=argparse.RawTextHelpFormatter,
        epilog="Example:\n  python core_scripts/ingest_additional_documents.py 100"
    )
    parser.add_argument(
        "num_docs",
        type=int,
        help="The number of additional documents to attempt to process from the source and ingest if new."
    )
    args = parser.parse_args()

    if args.num_docs <= 0:
        logger.error("Number of documents to ingest must be a positive integer.")
        sys.exit(1)

    logger.info(f"🚀 STARTING DOCUMENT INGESTION: Attempting to process and add up to {args.num_docs} new documents.")
    logger.info("=" * 70)
    
    start_time = time.time()
    
    ingestor = DocumentIngestor(num_docs_to_ingest=args.num_docs)
    success = ingestor.ingest_documents() # This now returns True if process ran, False on catastrophic failure
    
    total_time = time.time() - start_time
    logger.info("=" * 70)
    
    if success:
        logger.info(f"🎉 INGESTION PROCESS FINISHED.")
    else:
        logger.error(f"❌ INGESTION PROCESS FAILED CATASTROPHICALLY.")
    
    logger.info(f"⏱️  Total time: {total_time:.1f} seconds")
    logger.info("=" * 70)

    if success:
        sys.exit(0) # Exit 0 if the process completed, regardless of how many new docs were added
    else:
        sys.exit(1) # Exit 1 only if there was a catastrophic failure in the ingest_documents method

if __name__ == "__main__":
    main()