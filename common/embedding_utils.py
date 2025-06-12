"""
Embedding Utilities for IRIS RAG

This module contains utilities for generating document-level and token-level
embeddings for use in RAG applications.
"""

import logging
import time
import json
import numpy as np
import ast
from typing import Dict, List, Any, Optional, Tuple, Callable

# Configure logging
logger = logging.getLogger(__name__)

class MockEmbeddingModel:
    """Mock embedding model for testing without dependencies"""
    
    def __init__(self, embedding_dim: int = 384):
        self.embedding_dim = embedding_dim
        logger.info(f"Initialized MockEmbeddingModel with dimension {embedding_dim}")
        
    def encode(self, texts, batch_size=32, show_progress_bar=True):
        """Generate mock embeddings"""
        if isinstance(texts, str):
            texts = [texts]
            
        embeddings = []
        for text in texts:
            # Create a deterministic but unique embedding based on text
            np.random.seed(hash(text) % 10000)
            embedding = np.random.randn(self.embedding_dim)
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            embeddings.append(embedding)
            
        return np.array(embeddings)

    def embed_query(self, query: str) -> List[float]:
        """Generate a mock embedding for a single query."""
        # Reuse the encode method, take the first result, and convert to list
        return self.encode(query)[0].tolist()

    def embed_documents(self, documents: List[str]) -> List[List[float]]:
        """Generate mock embeddings for a list of documents."""
        return self.encode(documents).tolist()

class MockColBERTModel:
    """Mock ColBERT model for token-level embeddings"""
    
    def __init__(self, embedding_dim: int = 128):
        self.embedding_dim = embedding_dim
        logger.info(f"Initialized MockColBERTModel with dimension {embedding_dim}")
        
    def tokenize(self, text):
        """Simple tokenization for testing"""
        # Just split by space and punctuation - not accurate but OK for testing
        import re
        tokens = re.findall(r'\w+|[^\w\s]', text.lower())
        return tokens[:512]  # Limit to 512 tokens
        
    def encode(self, text):
        """Generate mock token-level embeddings"""
        tokens = self.tokenize(text)
        
        token_embeddings = []
        for i, token in enumerate(tokens):
            # Create deterministic but "unique" embedding based on token and position
            np.random.seed((hash(token) + i) % 10000)
            embedding = np.random.randn(self.embedding_dim)
            # Normalize
            embedding = embedding / np.linalg.norm(embedding)
            token_embeddings.append(embedding)
            
        return tokens, np.array(token_embeddings)

def get_embedding_model(model_name: str = "intfloat/e5-base-v2", mock: bool = False):
    """
    Get the embedding model for document embeddings.
    
    Args:
        model_name: Name of the SentenceTransformer model to use
        mock: Whether to use a mock model
        
    Returns:
        SentenceTransformer model or a mock model
    """
    if mock:
        logger.info("Using mock embedding model")
        return MockEmbeddingModel()
        
    try:
        from sentence_transformers import SentenceTransformer
        logger.info(f"Loading SentenceTransformer model: {model_name}")
        model = SentenceTransformer(model_name)
        return model
    except ImportError:
        logger.warning("SentenceTransformer package not installed. Using mock embedding model.")
        return MockEmbeddingModel()
    except Exception as e:
        logger.error(f"Failed to load SentenceTransformer model {model_name}: {e}")
        logger.warning("Falling back to mock embedding model.")
        return MockEmbeddingModel()

def get_colbert_model(model_name: str = "colbert-ir/colbertv2.0", mock: bool = False):
    """
    Get the ColBERT model for token-level embeddings.
    
    Args:
        model_name: Name of the ColBERT model to use
        mock: Whether to use a mock model
        
    Returns:
        ColBERT model or a mock model
    """
    if mock:
        logger.info("Using mock ColBERT model")
        return MockColBERTModel()
        
    # Try to import ColBERT
    try:
        # This is a placeholder - you'll need to replace with actual ColBERT import
        logger.warning("Real ColBERT not available. Using mock token embeddings.")
        return MockColBERTModel()
    except ImportError:
        logger.warning("ColBERT not available. Using mock token embeddings.")
        return MockColBERTModel()

def get_stub_embedding_func(embedding_dim: int = 384) -> Callable[[List[str]], List[List[float]]]:
    """
    Returns a stub function for generating embeddings, using MockEmbeddingModel.
    """
    mock_model = MockEmbeddingModel(embedding_dim=embedding_dim)
    def stub_embed(texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        # Ensure texts is a list of strings
        if isinstance(texts, str):
            texts = [texts]
        elif not isinstance(texts, list) or not all(isinstance(t, str) for t in texts):
            logger.error(f"Invalid input to stub_embed. Expected List[str], got {type(texts)}")
            # Return a list of empty lists or raise error, depending on desired handling
            return [[] for _ in range(len(texts))]
        
        # MockEmbeddingModel.encode returns np.array, convert to list of lists
        encoded_embeddings = mock_model.encode(texts)
        return [emb.tolist() for emb in encoded_embeddings]
    logger.info(f"Stub embedding function created (dim: {embedding_dim}).")
    return stub_embed

def get_embedding_func(provider: str = "stub", model_name: str = "intfloat/e5-base-v2", embedding_dim: int = 384) -> Callable[[List[str]], List[List[float]]]:
    """
    Get an embedding function based on the provider.
    This function returns another function that takes a list of texts and returns embeddings.
    """
    logger.info(f"Attempting to get embedding function for provider: {provider}, model: {model_name}, dim: {embedding_dim}")
    if provider == "stub":
        return get_stub_embedding_func(embedding_dim=embedding_dim)
    elif provider == "openai":
        # Placeholder for OpenAI embedding function
        # You would typically import the OpenAI client and define a function
        # that calls client.embeddings.create(...)
        logger.warning("OpenAI embedding provider function not fully implemented, using stub.")
        return get_stub_embedding_func(embedding_dim=embedding_dim) # Fallback for now
    elif provider == "huggingface":
        try:
            # Use the existing get_embedding_model to load the actual SentenceTransformer model
            # Assuming mock=False for a real HuggingFace model.
            # The get_embedding_model handles SentenceTransformer import and errors.
            actual_model = get_embedding_model(model_name=model_name, mock=False)
            
            # Check if the loaded model is a real model or a fallback mock
            if isinstance(actual_model, MockEmbeddingModel):
                logger.warning(f"Failed to load real HuggingFace model '{model_name}', falling back to stub embedding function via MockEmbeddingModel.")
                return get_stub_embedding_func(embedding_dim=embedding_dim)

            def hf_embed(texts: List[str]) -> List[List[float]]:
                if not texts:
                    return []
                if isinstance(texts, str):
                    texts = [texts]
                # SentenceTransformer models usually have an .encode() method
                encoded_embeddings = actual_model.encode(texts)
                # Ensure it's a list of lists of floats
                if isinstance(encoded_embeddings, np.ndarray):
                    return encoded_embeddings.tolist()
                elif isinstance(encoded_embeddings, list) and all(isinstance(e, np.ndarray) for e in encoded_embeddings):
                     return [e.tolist() for e in encoded_embeddings]
                elif isinstance(encoded_embeddings, list) and all(isinstance(e, list) for e in encoded_embeddings):
                    return encoded_embeddings # Already in correct format
                else:
                    logger.error(f"Unexpected embedding format from HuggingFace model {model_name}: {type(encoded_embeddings)}")
                    return [[] for _ in range(len(texts))]
            logger.info(f"HuggingFace embedding function created for model: {model_name}")
            return hf_embed
        except Exception as e:
            logger.error(f"Error setting up HuggingFace embedding function for model {model_name}: {e}", exc_info=True)
            logger.warning("Falling back to stub embedding function due to HuggingFace setup error.")
            return get_stub_embedding_func(embedding_dim=embedding_dim)
    else:
        logger.warning(f"Unknown embedding provider: {provider}. Using stub embedding function.")
        return get_stub_embedding_func(embedding_dim=embedding_dim)

def generate_document_embeddings(
    connection, 
    embedding_model,
    batch_size: int = 32,
    limit: Optional[int] = None,
    schema: str = "RAG"  # Added schema parameter
) -> Dict[str, Any]:
    """
    Generate document-level embeddings for documents in IRIS.
    
    Args:
        connection: IRIS connection
        embedding_model: Model to generate embeddings
        batch_size: Number of documents to process at once
        limit: Maximum number of documents to process (None for all)
        
    Returns:
        Dictionary with statistics
    """
    start_time = time.time()
    processed_count = 0
    error_count = 0
    
    try:
        cursor = connection.cursor()
        
        # Get all documents (or limited) to regenerate/generate embeddings
        # This ensures all embeddings are processed/reprocessed with the current logic
        if limit:
            sql = f"""
                SELECT TOP ? doc_id, text_content
                FROM {schema}.SourceDocuments
            """
            cursor.execute(sql, (limit,))
        else:
            sql = f"""
                SELECT doc_id, text_content
                FROM {schema}.SourceDocuments
            """
            cursor.execute(sql)
            
        rows = cursor.fetchall()
        logger.info(f"Found {len(rows)} documents to process for embeddings")
        
        # Process in batches
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i+batch_size]
            batch_ids = [row[0] for row in batch]
            batch_texts = [row[1] for row in batch]
            
            # Skip empty texts
            valid_indices = []
            valid_texts = []
            for j, text in enumerate(batch_texts):
                if text and len(text.strip()) > 0:
                    valid_indices.append(j)
                    valid_texts.append(text)
            
            if not valid_texts:
                logger.warning(f"Skipping batch {i//batch_size} - no valid texts")
                continue
                
            try:
                # Generate embeddings
                logger.debug(f"Generating embeddings for batch {i//batch_size + 1}")
                embeddings = embedding_model.encode(
                    valid_texts, 
                    batch_size=batch_size,
                    show_progress_bar=False
                )
                
                # Update database
                update_cursor = connection.cursor()
                for idx, embedding in zip(valid_indices, embeddings):
                    doc_id = batch_ids[idx]
                    embedding_str = json.dumps(embedding.tolist())
                    
                    # For mock database, update our stored data directly
                    if hasattr(connection, '_cursor') and hasattr(connection._cursor, 'stored_docs'):
                        # This is a mock connection
                        if doc_id in connection._cursor.stored_docs:
                            connection._cursor.stored_docs[doc_id]["embedding"] = embedding_str
                        else:
                            connection._cursor.stored_docs[doc_id] = {
                                "content": batch_texts[idx],
                                "embedding": embedding_str
                            }
                    else:
                        # For real database, use SQL
                        update_cursor.execute(
                            f"""
                            UPDATE {schema}.SourceDocuments
                            SET embedding = ?
                            WHERE doc_id = ?
                            """,
                            (embedding_str, doc_id)
                        )
                
                connection.commit()
                processed_count += len(valid_indices)
                
                # Log progress
                if (i//batch_size + 1) % 5 == 0 or i + batch_size >= len(rows):
                    elapsed = time.time() - start_time
                    docs_per_second = processed_count / elapsed if elapsed > 0 else 0
                    logger.info(
                        f"Generated embeddings for {processed_count}/{len(rows)} documents "
                        f"({docs_per_second:.2f} docs/sec)"
                    )
                    
            except Exception as e:
                logger.error(f"Error processing batch {i//batch_size}: {e}")
                connection.rollback()
                error_count += len(valid_indices)
                
        cursor.close()
        
    except Exception as e:
        logger.error(f"Error generating document embeddings: {e}")
        error_count += len(rows) - processed_count if 'rows' in locals() else 0
    
    duration = time.time() - start_time
    
    return {
        "type": "document_embeddings",
        "total_documents": len(rows) if 'rows' in locals() else 0,
        "processed_count": processed_count,
        "error_count": error_count,
        "duration_seconds": duration,
        "documents_per_second": processed_count / duration if duration > 0 else 0
    }

def generate_token_embeddings(
    connection, 
    token_encoder,
    batch_size: int = 10,
    limit: Optional[int] = None
) -> Dict[str, Any]:
    """
    Generate token-level embeddings for documents in IRIS.
    
    Args:
        connection: IRIS connection
        token_encoder: Model to generate token-level embeddings
        batch_size: Number of documents to process at once
        limit: Maximum number of documents to process (None for all)
        
    Returns:
        Dictionary with statistics
    """
    start_time = time.time()
    processed_count = 0
    tokens_count = 0
    error_count = 0
    
    try:
        cursor = connection.cursor()
        
        # Get documents without token embeddings
        # We use a LEFT JOIN to find documents that don't have entries in DocumentTokenEmbeddings
        if limit:
            sql = """
                SELECT TOP ? d.doc_id, d.content 
                FROM SourceDocuments d 
                LEFT JOIN (
                    SELECT DISTINCT doc_id FROM DocumentTokenEmbeddings
                ) t ON d.doc_id = t.doc_id 
                WHERE t.doc_id IS NULL
            """
            cursor.execute(sql, (limit,))
        else:
            sql = """
                SELECT d.doc_id, d.content 
                FROM SourceDocuments d 
                LEFT JOIN (
                    SELECT DISTINCT doc_id FROM DocumentTokenEmbeddings
                ) t ON d.doc_id = t.doc_id 
                WHERE t.doc_id IS NULL
            """
            cursor.execute(sql)
            
        rows = cursor.fetchall()
        logger.info(f"Found {len(rows)} documents without token embeddings")
        
        # Process in batches
        for i in range(0, len(rows), batch_size):
            batch = rows[i:i+min(batch_size, len(rows)-i)]
            
            # Process one document at a time within batch
            batch_token_count = 0
            for doc_id, content in batch:
                if not content or len(content.strip()) == 0:
                    logger.warning(f"Skipping empty document: {doc_id}")
                    continue
                    
                try:
                    # Generate token embeddings
                    tokens, token_embeddings = token_encoder.encode(content)
                    
                    # Insert token embeddings
                    insert_cursor = connection.cursor()
                    token_batch = []
                    
                    for idx, (token, embedding) in enumerate(zip(tokens, token_embeddings)):
                        # Convert embedding to string and create metadata
                        embedding_str = json.dumps(embedding.tolist())
                        metadata = json.dumps({
                            "token_idx": idx,
                            "token": token
                        })
                        
                        token_batch.append((
                            doc_id, 
                            idx, 
                            token, 
                            embedding_str, 
                            metadata
                        ))
                        
                    # Insert all tokens for this document
                    # For mock database, update our stored data directly
                    if hasattr(connection, '_cursor') and hasattr(connection._cursor, 'stored_token_embeddings'):
                        # This is a mock connection
                        if doc_id not in connection._cursor.stored_token_embeddings:
                            connection._cursor.stored_token_embeddings[doc_id] = []
                        
                        for token_data in token_batch:
                            doc_id, token_idx, token_text, token_embedding, metadata = token_data
                            connection._cursor.stored_token_embeddings[doc_id].append({
                                "idx": token_idx,
                                "text": token_text,
                                "embedding": token_embedding,
                                "metadata": metadata
                            })
                    else:
                        # For real database, use SQL
                        insert_cursor.executemany(
                            """
                            INSERT INTO DocumentTokenEmbeddings 
                            (doc_id, token_idx, token_text, token_embedding, metadata_json) 
                            VALUES (?, ?, ?, ?, ?)
                            """,
                            token_batch
                        )
                    
                    connection.commit()
                    processed_count += 1
                    batch_token_count += len(tokens)
                    tokens_count += len(tokens)
                    
                except Exception as e:
                    logger.error(f"Error processing document {doc_id}: {e}")
                    connection.rollback()
                    error_count += 1
            
            # Log progress
            if (i//batch_size + 1) % 2 == 0 or i + batch_size >= len(rows):
                elapsed = time.time() - start_time
                docs_per_second = processed_count / elapsed if elapsed > 0 else 0
                tokens_per_second = tokens_count / elapsed if elapsed > 0 else 0
                logger.info(
                    f"Generated token embeddings for {processed_count}/{len(rows)} documents "
                    f"({docs_per_second:.2f} docs/sec, {tokens_per_second:.2f} tokens/sec)"
                )
                logger.info(f"Processed {tokens_count} tokens in total")
                
        cursor.close()
        
    except Exception as e:
        logger.error(f"Error generating token embeddings: {e}")
        error_count += len(rows) - processed_count if 'rows' in locals() else 0
    
    duration = time.time() - start_time
    
    return {
        "type": "token_embeddings",
        "total_documents": len(rows) if 'rows' in locals() else 0,
        "processed_count": processed_count,
        "tokens_count": tokens_count,
        "error_count": error_count,
        "duration_seconds": duration,
        "documents_per_second": processed_count / duration if duration > 0 else 0,
        "tokens_per_second": tokens_count / duration if duration > 0 else 0
    }

def create_tables_if_needed(connection) -> None:
    """
    Create tables for embeddings if they don't exist
    
    Args:
        connection: IRIS connection
    """
    try:
        cursor = connection.cursor()
        
        # Check if DocumentTokenEmbeddings table exists
        try:
            cursor.execute("SELECT TOP 1 * FROM DocumentTokenEmbeddings")
            logger.info("DocumentTokenEmbeddings table already exists")
        except Exception:
            logger.info("Creating DocumentTokenEmbeddings table")
            cursor.execute("""
                CREATE TABLE DocumentTokenEmbeddings (
                    id INT IDENTITY PRIMARY KEY,
                    doc_id VARCHAR(100) NOT NULL,
                    token_idx INT NOT NULL,
                    token_text VARCHAR(100),
                    token_embedding TEXT,
                    metadata_json TEXT,
                    INDEX idx_doc_id ON (doc_id)
                )
            """)
            connection.commit()
            logger.info("Created DocumentTokenEmbeddings table")
            
        # Make sure SourceDocuments has an embedding column
        try:
            cursor.execute("SELECT embedding FROM SourceDocuments WHERE 1=0")
            logger.info("Embedding column already exists in SourceDocuments")
        except Exception:
            logger.info("Adding embedding column to SourceDocuments")
            cursor.execute("ALTER TABLE SourceDocuments ADD embedding TEXT")
            connection.commit()
            logger.info("Added embedding column to SourceDocuments")
            
        cursor.close()
            
    except Exception as e:
        logger.error(f"Error creating tables: {e}")
