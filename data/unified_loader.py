"""
Unified Data Loader for IRIS RAG Templates

This module provides a single, configurable data loader that consolidates the logic
from five previous loaders:
- loader_fixed.py
- loader_vector_fixed.py
- loader_varchar_fixed.py
- loader_optimized_performance.py
- loader_conservative_optimized.py

It supports various embedding column types, performance optimizations, and robust error handling.
Uses the standardized db_vector_utils for all vector insertions to ensure consistency.
"""

import logging
import time
import json
import gc
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable, Tuple
import sys
import numpy as np

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from common.iris_connector import get_iris_connection
from common.vector_format_fix import format_vector_for_iris, validate_vector_for_iris, VectorFormatError
from common.db_vector_utils import insert_vector
from common.dimension_utils import get_vector_dimension
from data.pmc_processor import process_pmc_files

logger = logging.getLogger(__name__)

class UnifiedDocumentLoader:
    """
    A unified, configurable document loader for IRIS.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initialize the loader with a configuration dictionary.

        Args:
            config: Configuration dictionary, typically from config/pipelines.yaml.
        """
        self.config = self._validate_config(config)
        self.checkpoint_file = Path(self.config.get("checkpoint_path", "data/unified_checkpoint.json"))
        self.performance_log = []
        self.embedding_column_type = self.config.get("embedding_column_type", "VECTOR").upper()
        
        # Cache vector dimensions for performance
        self._doc_embedding_dim = None
        self._token_embedding_dim = None

    def _validate_config(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """Validate and set defaults for configuration."""
        validated_config = config.copy()
        
        # Set sensible defaults
        defaults = {
            "batch_size": 50,
            "token_batch_size": 1000,
            "embedding_column_type": "VECTOR",
            "use_checkpointing": True,
            "refresh_connection": False,
            "refresh_connection_interval": 100,
            "gc_collect_interval": 50,
            "limit": 1000
        }
        
        for key, default_value in defaults.items():
            if key not in validated_config:
                validated_config[key] = default_value
                
        # Validate critical settings
        if validated_config["batch_size"] <= 0:
            logger.warning("Invalid batch_size, using default of 50")
            validated_config["batch_size"] = 50
            
        if validated_config["embedding_column_type"].upper() not in ["VECTOR", "VARCHAR"]:
            logger.warning("Invalid embedding_column_type, using VECTOR")
            validated_config["embedding_column_type"] = "VECTOR"
            
        return validated_config

    def _get_vector_dimensions(self):
        """Cache vector dimensions for performance."""
        from iris_rag.config.manager import ConfigurationManager
        from common.iris_connection_manager import IRISConnectionManager

        config_manager = ConfigurationManager()
        connection_manager = IRISConnectionManager(config_manager=config_manager)
        if self._doc_embedding_dim is None:
            self._doc_embedding_dim = get_vector_dimension("SourceDocuments", connection_manager, config_manager)
        if self._token_embedding_dim is None:
            self._token_embedding_dim = get_vector_dimension("DocumentTokenEmbeddings", connection_manager, config_manager)
        return self._doc_embedding_dim, self._token_embedding_dim

    def _format_vector(self, vector: List[float]) -> Any:
        """Format vector based on the configured column type."""
        if vector is None:
            return None
            
        try:
            clean_vector = format_vector_for_iris(vector)
            if self.embedding_column_type == "VARCHAR":
                return ','.join(f"{x:.15g}" for x in clean_vector)
            else:  # Default to VECTOR
                if not validate_vector_for_iris(clean_vector):
                    raise VectorFormatError("Formatted vector is invalid.")
                return clean_vector
        except Exception as e:
            logger.error(f"Failed to format vector of length {len(vector) if vector else 0}: {e}")
            raise VectorFormatError(f"Failed to format vector: {e}")

    def _validate_and_fix_text(self, text: Any) -> str:
        """Validate and fix text fields."""
        if text is None:
            return ""
        if isinstance(text, (list, dict)):
            return json.dumps(text)
        try:
            return str(text).replace('\x00', '')
        except Exception as e:
            logger.warning(f"Error processing text field: {e}")
            return ""

    def load_checkpoint(self):
        """Load checkpoint if it exists."""
        if self.config.get("use_checkpointing", False) and self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {"processed_count": 0, "last_doc_id": None}

    def save_checkpoint(self, processed_count: int, last_doc_id: str):
        """Save checkpoint."""
        if not self.config.get("use_checkpointing", False):
            return
            
        checkpoint = {
            "processed_count": processed_count,
            "last_doc_id": last_doc_id,
            "timestamp": time.time(),
        }
        self.checkpoint_file.parent.mkdir(exist_ok=True)
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)

    def _refresh_connection(self, iris_connector):
        """Refresh database connection if enabled."""
        if not self.config.get("refresh_connection", False):
            return iris_connector
        try:
            cursor = iris_connector.cursor()
            cursor.execute("SELECT 1")
            cursor.close()
            return iris_connector
        except Exception:
            logger.info("🔄 Refreshing database connection")
            try:
                iris_connector.close()
            except Exception:
                pass
            return get_iris_connection()

    def process_document(
        self,
        doc: Dict[str, Any],
        embedding_func: Optional[Callable] = None,
        colbert_doc_encoder_func: Optional[Callable] = None,
    ) -> Optional[Tuple[Optional[Tuple], Optional[List[Tuple]]]]:
        """Process a single document to prepare it for insertion."""
        doc_id = doc.get("pmc_id") or doc.get("id")
        if not doc_id:
            logger.warning("Skipping document with no ID.")
            return None

        title = self._validate_and_fix_text(doc.get("title", ""))
        abstract = self._validate_and_fix_text(doc.get("abstract", ""))
        authors = self._validate_and_fix_text(doc.get("authors", []))
        keywords = self._validate_and_fix_text(doc.get("keywords", []))
        
        content_to_embed = abstract if abstract else title
        embedding = None
        if embedding_func and content_to_embed:
            try:
                embedding = embedding_func([content_to_embed])[0]
            except Exception as e:
                logger.error(f"Error generating embedding for doc {doc_id}: {e}")

        doc_params = (
            doc_id, title, abstract, abstract, authors, keywords,
            self._format_vector(embedding) if embedding else None
        )

        token_params = []
        if colbert_doc_encoder_func and content_to_embed:
            try:
                # ColBERT function returns list of (token, embedding) tuples
                colbert_result = colbert_doc_encoder_func(content_to_embed)
                
                # Handle the actual format: list of (token, embedding) tuples
                if isinstance(colbert_result, list) and colbert_result:
                    for i, (token, token_embedding) in enumerate(colbert_result):
                        token_params.append((
                            doc_id, i, token, self._format_vector(token_embedding)
                        ))
                else:
                    logger.warning(f"Unexpected ColBERT result format for doc {doc_id}: {type(colbert_result)}")
                    
            except Exception as e:
                logger.error(f"Error generating ColBERT embeddings for doc {doc_id}: {e}")

        return doc_params, token_params

    def insert_document(self, cursor, doc_params: Tuple, token_params: List[Tuple]):
        """Insert a single document and its tokens into the database using standardized vector utilities."""
        try:
            doc_id, title, abstract, text_content, authors, keywords, embedding = doc_params
            
            # Get cached vector dimensions for performance
            doc_embedding_dim, token_embedding_dim = self._get_vector_dimensions()
            
            # Insert into SourceDocuments using insert_vector utility
            if embedding is not None:
                # Ensure embedding is in the correct format
                if isinstance(embedding, str):
                    # Handle comma-separated string format
                    embedding = [float(x) for x in embedding.strip('[]').split(',')]
                elif not isinstance(embedding, list):
                    logger.warning(f"Unexpected embedding type {type(embedding)} for doc {doc_id}")
                    embedding = list(embedding) if hasattr(embedding, '__iter__') else [float(embedding)]
                
                success = insert_vector(
                    cursor=cursor,
                    table_name="RAG.SourceDocuments",
                    vector_column_name="embedding",
                    vector_data=embedding,
                    target_dimension=doc_embedding_dim,
                    key_columns={"doc_id": doc_id},
                    additional_data={
                        "title": title,
                        "abstract": abstract,
                        "text_content": text_content,
                        "authors": authors,
                        "keywords": keywords
                    }
                )
                if not success:
                    logger.error(f"Failed to insert document {doc_id} using insert_vector utility")
                    raise RuntimeError(f"Vector insertion failed for document {doc_id}")
            else:
                # Fallback for documents without embeddings
                doc_sql = """
                    INSERT INTO RAG.SourceDocuments
                    (doc_id, title, abstract, text_content, authors, keywords, embedding)
                    VALUES (?, ?, ?, ?, ?, ?, NULL)
                """
                cursor.execute(doc_sql, (doc_id, title, abstract, text_content, authors, keywords))

            # Insert token embeddings using insert_vector utility
            for token_data in token_params:
                token_doc_id, token_index, token_text, token_embedding = token_data
                
                if token_embedding is not None:
                    # Ensure token embedding is in the correct format
                    if isinstance(token_embedding, str):
                        token_embedding = [float(x) for x in token_embedding.strip('[]').split(',')]
                    elif not isinstance(token_embedding, list):
                        token_embedding = list(token_embedding) if hasattr(token_embedding, '__iter__') else [float(token_embedding)]
                    
                    success = insert_vector(
                        cursor=cursor,
                        table_name="RAG.DocumentTokenEmbeddings",
                        vector_column_name="token_embedding",
                        vector_data=token_embedding,
                        target_dimension=token_embedding_dim,
                        key_columns={"doc_id": token_doc_id, "token_index": token_index},
                        additional_data={"token_text": token_text}
                    )
                    if not success:
                        logger.warning(f"Failed to insert token embedding for doc {token_doc_id}, token {token_index}")
                        
        except Exception as e:
            logger.error(f"Error inserting document {doc_params[0] if doc_params else 'unknown'}: {e}")
            raise

    def load_documents(
        self,
        iris_connector,
        documents: List[Dict[str, Any]],
        embedding_func: Optional[Callable] = None,
        colbert_doc_encoder_func: Optional[Callable] = None,
    ) -> Dict[str, Any]:
        """
        Load documents into IRIS based on the instance configuration.
        
        Args:
            iris_connector: IRIS database connection
            documents: List of document dictionaries to load
            embedding_func: Optional function to generate document embeddings
            colbert_doc_encoder_func: Optional function to generate ColBERT token embeddings
            
        Returns:
            Dictionary with loading statistics and performance metrics
        """
        
        start_time = time.time()
        loaded_doc_count = 0
        loaded_token_count = 0
        error_count = 0

        batch_size = self.config.get("batch_size", 50)
        token_batch_size = self.config.get("token_batch_size", 1000)

        checkpoint = self.load_checkpoint()
        start_idx = checkpoint.get("processed_count", 0)

        logger.info(f"🚀 UNIFIED LOADING INITIATED")
        logger.info(f"   Total documents: {len(documents)}")
        logger.info(f"   Batch size: {batch_size}")
        logger.info(f"   Token batch size: {token_batch_size}")
        logger.info(f"   Embedding column type: {self.embedding_column_type}")
        logger.info(f"   Starting from index: {start_idx}")

        try:
            cursor = iris_connector.cursor()
            
            for batch_idx in range(start_idx // batch_size, (len(documents) + batch_size - 1) // batch_size):
                batch_start = batch_idx * batch_size
                if batch_start < start_idx:
                    continue # Skip already processed batches
                
                batch_end = min(batch_start + batch_size, len(documents))
                current_batch = documents[batch_start:batch_end]
                
                if not current_batch:
                    break

                batch_start_time = time.time()
                source_doc_params = []
                all_token_params = []

                for doc in current_batch:
                    processed_data = self.process_document(doc, embedding_func, colbert_doc_encoder_func)
                    if processed_data:
                        doc_params, token_params = processed_data
                        if doc_params:
                            source_doc_params.append(doc_params)
                        if token_params:
                            all_token_params.extend(token_params)
                    else:
                        error_count += 1

                # Insert documents and track performance
                if source_doc_params:
                    try:
                        # Process each document in the batch
                        for doc_params in source_doc_params:
                            doc_token_params = [p for p in all_token_params if p[0] == doc_params[0]]
                            self.insert_document(cursor, doc_params, doc_token_params)
                            loaded_doc_count += 1
                            loaded_token_count += len(doc_token_params)
                        
                        iris_connector.commit()
                        self.save_checkpoint(batch_end, source_doc_params[-1][0])

                    except Exception as e:
                        logger.error(f"Database error in batch {batch_idx}: {e}")
                        iris_connector.rollback()
                        error_count += len(current_batch)
                
                # Performance logging and connection management
                batch_duration = time.time() - batch_start_time
                self.performance_log.append(batch_duration)
                logger.info(f"📊 Batch {batch_idx} complete in {batch_duration:.2f}s. Total loaded: {loaded_doc_count} docs, {loaded_token_count} tokens.")

                if self.config.get("refresh_connection_interval") and batch_idx % self.config["refresh_connection_interval"] == 0:
                    iris_connector = self._refresh_connection(iris_connector)
                    cursor = iris_connector.cursor()

                if self.config.get("gc_collect_interval") and batch_idx % self.config["gc_collect_interval"] == 0:
                    gc.collect()

            cursor.close()

        except Exception as e:
            logger.error(f"Critical error in loading process: {e}", exc_info=True)
            error_count = len(documents) - loaded_doc_count

        duration = time.time() - start_time
        return {
            "total_documents": len(documents),
            "loaded_doc_count": loaded_doc_count,
            "loaded_token_count": loaded_token_count,
            "error_count": error_count,
            "duration_seconds": duration,
            "documents_per_second": loaded_doc_count / duration if duration > 0 else 0,
            "performance_log": self.performance_log
        }

def process_and_load_documents_unified(
    config: Dict[str, Any],
    pmc_directory: str,
    embedding_func: Optional[Callable] = None,
    colbert_doc_encoder_func: Optional[Callable] = None,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Process PMC files and load them using the UnifiedDocumentLoader.
    
    Args:
        config: Configuration dictionary with loader settings
        pmc_directory: Directory containing PMC XML files
        embedding_func: Optional function to generate document embeddings
        colbert_doc_encoder_func: Optional function to generate ColBERT token embeddings
        limit: Optional limit on number of documents to process
        
    Returns:
        Dictionary with processing results and statistics
    """
    start_time = time.time()
    
    iris_connector = get_iris_connection()
    if not iris_connector:
        return {"success": False, "error": "Failed to establish database connection"}

    try:
        limit = limit or config.get("limit", 1000)
        documents = list(process_pmc_files(pmc_directory, limit))
        processed_count = len(documents)
        logger.info(f"Processed {processed_count} documents from {pmc_directory}")

        loader = UnifiedDocumentLoader(config)
        load_stats = loader.load_documents(
            iris_connector,
            documents,
            embedding_func,
            colbert_doc_encoder_func
        )

        return {
            "success": True,
            "processed_count": processed_count,
            **load_stats
        }
    except Exception as e:
        logger.error(f"Error in unified processing and loading: {e}", exc_info=True)
        return {"success": False, "error": str(e)}
    finally:
        if iris_connector:
            iris_connector.close()