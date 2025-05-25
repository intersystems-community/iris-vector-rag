"""
PMC Document Loader Module

This module provides functions to process and load PMC documents into an IRIS database.
"""

import os
import time
import logging
import json
from typing import Dict, List, Any, Optional, Tuple
import pyodbc
import numpy as np
from tqdm import tqdm

from data.pmc_processor import process_pmc_files
from common.embedding_utils import get_embedding_model

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def process_and_load_documents(
    pmc_directory: str,
    connection: pyodbc.Connection,
    limit: int = 1000,
    batch_size: int = 50,
    embedding_model: str = "intfloat/e5-base-v2",
    use_mock: bool = False
) -> Dict[str, Any]:
    """
    Process PMC documents and load them into the IRIS database.
    
    Args:
        pmc_directory: Path to the directory containing PMC documents
        connection: IRIS database connection
        limit: Maximum number of documents to process
        batch_size: Number of documents to process in each batch
        embedding_model: Name of the embedding model to use
        use_mock: Whether to use mock embeddings
        
    Returns:
        Dict with statistics about the processing and loading
    """
    start_time = time.time()
    
    try:
        # Get embedding function
        embedding_func = get_embedding_model(embedding_model, mock=use_mock)
        
        # Process PMC documents
        logger.info(f"Processing up to {limit} documents from {pmc_directory}")
        documents = list(process_pmc_files(pmc_directory, limit=limit))
        
        if not documents:
            return {
                "success": False,
                "error": "No documents found or processed",
                "processed_count": 0,
                "loaded_doc_count": 0,
                "loaded_token_count": 0,
                "time_taken": time.time() - start_time
            }
        
        # Load documents into database
        logger.info(f"Loading {len(documents)} SourceDocuments in {len(documents) // batch_size + 1} batches.")
        
        # Create a sample document to test the SQL
        sample_doc = {
            "doc_id": "sample",
            "title": "Diabetes Treatment Review",
            "content": "Common treatments for type 2 diabetes include lifestyle changes such as diet and exercise, oral medications like metformin, and in some cases, insulin therapy. Metformin is often the first-line medication prescribed for type 2 diabetes. It helps lower glucose production in the liver and improves insulin sensitivity.",
            "authors": ["John A. Smith"],
            "keywords": ["diabetes", "metformin", "insulin"]
        }
        
        # Get embedding for sample document
        sample_embedding = embedding_func([sample_doc["content"]])[0]
        
        # Insert sample document to test SQL
        try:
            with connection.cursor() as cursor:
                # Convert embedding to string for SQL
                embedding_str = ','.join([str(x) for x in sample_embedding])
                
                # Convert authors and keywords to JSON strings
                authors_json = json.dumps(sample_doc["authors"])
                keywords_json = json.dumps(sample_doc["keywords"])
                
                # Log the SQL being executed
                sql = """
                INSERT INTO RAG.SourceDocuments 
                (doc_id, title, text_content, authors, keywords, embedding) 
                VALUES (?, ?, ?, ?, ?, TO_VECTOR(?, "double", 768))
                """
                logger.info(f"Executing SQL (with embedding): {sql}")
                
                # Execute the SQL
                cursor.execute(
                    sql,
                    (
                        sample_doc["doc_id"],
                        sample_doc["title"],
                        sample_doc["content"],
                        authors_json,
                        keywords_json,
                        embedding_str
                    )
                )
                connection.commit()
        except Exception as e:
            logger.error(f"Error loading sample document: {e}")
            return {
                "success": False,
                "error": str(e),
                "processed_count": len(documents),
                "loaded_doc_count": 0,
                "loaded_token_count": 0,
                "time_taken": time.time() - start_time
            }
        
        # Load documents in batches
        loaded_doc_count = 0
        loaded_token_count = 0
        
        for i in range(0, len(documents), batch_size):
            batch = documents[i:i+batch_size]
            try:
                # Get embeddings for batch
                texts = [doc["content"] for doc in batch]
                embeddings = embedding_func(texts)
                
                with connection.cursor() as cursor:
                    for j, doc in enumerate(batch):
                        # Convert embedding to string for SQL
                        embedding_str = ','.join([str(x) for x in embeddings[j]])
                        
                        # Convert authors and keywords to JSON strings
                        authors_json = json.dumps(doc.get("authors", []))
                        keywords_json = json.dumps(doc.get("keywords", []))
                        
                        # Execute the SQL
                        cursor.execute(
                            """
                            INSERT INTO RAG.SourceDocuments 
                            (doc_id, title, text_content, authors, keywords, embedding) 
                            VALUES (?, ?, ?, ?, ?, TO_VECTOR(?, "double", 768))
                            """,
                            (
                                doc["doc_id"],
                                doc.get("title", ""),
                                doc["content"],
                                authors_json,
                                keywords_json,
                                embedding_str
                            )
                        )
                    connection.commit()
                    loaded_doc_count += len(batch)
                    loaded_token_count += sum(len(doc["content"].split()) for doc in batch)
                    logger.info(f"Loaded batch {i//batch_size + 1}/{len(documents)//batch_size + 1} ({loaded_doc_count}/{len(documents)} documents)")
            except Exception as e:
                logger.error(f"Error loading batch {i//batch_size}: {e}")
        
        end_time = time.time()
        time_taken = end_time - start_time
        
        return {
            "success": True,
            "processed_count": len(documents),
            "loaded_doc_count": loaded_doc_count,
            "loaded_token_count": loaded_token_count,
            "time_taken": time_taken
        }
    
    except Exception as e:
        logger.error(f"Error processing and loading documents: {e}")
        return {
            "success": False,
            "error": str(e),
            "processed_count": 0,
            "loaded_doc_count": 0,
            "loaded_token_count": 0,
            "time_taken": time.time() - start_time
        }
