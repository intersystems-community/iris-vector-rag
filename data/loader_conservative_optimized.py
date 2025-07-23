"""
Conservative Performance Optimization - Minimal Changes

This loader makes minimal, safe changes to improve performance:
1. Smaller batch sizes
2. Better connection management  
3. More frequent commits
4. Progress checkpointing
5. Memory management
"""

import logging
import time
import json
import gc
import os
from pathlib import Path
from typing import List, Dict, Any, Optional, Callable
import sys

sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from common.iris_connector import get_iris_connection
from common.vector_format_fix import format_vector_for_iris

logger = logging.getLogger(__name__)

class ConservativeOptimizedLoader:
    """Conservatively optimized loader with minimal changes."""
    
    def __init__(self):
        self.checkpoint_file = Path("data") / "conservative_checkpoint.json"
        self.performance_log = []
        
    def load_checkpoint(self):
        """Load checkpoint if it exists."""
        if self.checkpoint_file.exists():
            with open(self.checkpoint_file, 'r') as f:
                return json.load(f)
        return {"processed_count": 0, "last_doc_id": None}
    
    def save_checkpoint(self, processed_count: int, last_doc_id: str):
        """Save checkpoint."""
        checkpoint = {
            "processed_count": processed_count,
            "last_doc_id": last_doc_id,
            "timestamp": time.time(),
            "datetime": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        # Ensure data directory exists
        self.checkpoint_file.parent.mkdir(exist_ok=True)
        
        with open(self.checkpoint_file, 'w') as f:
            json.dump(checkpoint, f, indent=2)
    
    def log_performance(self, batch_time: float, docs_per_sec: float, batch_idx: int):
        """Log performance metrics."""
        entry = {
            "batch_idx": batch_idx,
            "batch_time": batch_time,
            "docs_per_sec": docs_per_sec,
            "timestamp": time.time()
        }
        self.performance_log.append(entry)
        
        # Check for performance degradation
        if len(self.performance_log) >= 5:
            recent_times = [e["batch_time"] for e in self.performance_log[-5:]]
            avg_recent = sum(recent_times) / len(recent_times)
            
            if avg_recent > 10.0:  # More than 10 seconds per batch
                logger.warning(f"⚠️  Performance degrading: {avg_recent:.1f}s per batch")
                
            if avg_recent > 30.0:  # Critical threshold
                logger.error(f"🚨 CRITICAL: Performance severely degraded: {avg_recent:.1f}s per batch")
                logger.error("🚨 Consider stopping and investigating")
    
    def refresh_connection(self, connection):
        """Refresh database connection."""
        try:
            # Test connection
            cursor = connection.cursor()
            cursor.execute("SELECT 1")
            cursor.fetchone()
            cursor.close()
            return connection
        except:
            # Connection is bad, create new one
            logger.info("🔄 Refreshing database connection")
            try:
                connection.close()
            except:
                pass
            return get_iris_connection()
    
    def load_documents_conservative(
        self,
        connection,
        documents: List[Dict[str, Any]],
        embedding_func: Optional[Callable] = None,
        colbert_doc_encoder_func: Optional[Callable] = None,
        batch_size: int = 8,  # VERY conservative batch size
        token_batch_size: int = 150  # VERY conservative token batch size
    ) -> Dict[str, Any]:
        """Load documents with conservative optimizations."""
        
        start_time = time.time()
        loaded_doc_count = 0
        loaded_token_count = 0
        error_count = 0
        
        # Load checkpoint
        checkpoint = self.load_checkpoint()
        start_idx = checkpoint.get("processed_count", 0)
        
        logger.info(f"🚀 CONSERVATIVE OPTIMIZED LOADING")
        logger.info(f"   Total documents: {len(documents)}")
        logger.info(f"   Batch size: {batch_size} (conservative)")
        logger.info(f"   Token batch size: {token_batch_size} (conservative)")
        logger.info(f"   Starting from: {start_idx}")
        
        try:
            cursor = connection.cursor()
            
            # Process in very small batches
            for batch_idx in range(start_idx // batch_size, len(documents) // batch_size + 1):
                batch_start = batch_idx * batch_size
                batch_end = min(batch_start + batch_size, len(documents))
                
                if batch_start >= len(documents):
                    break
                    
                current_batch = documents[batch_start:batch_end]
                batch_start_time = time.time()
                
                # Process documents
                source_doc_params = []
                all_token_params = []
                
                for doc in current_batch:
                    try:
                        doc_id = doc.get("doc_id") or doc.get("pmc_id")
                        if not doc_id:
                            continue
                        
                        # Generate document embedding (same as before)
                        embedding_str = None
                        if embedding_func:
                            text_to_embed = doc.get("abstract") or doc.get("title", "")
                            if text_to_embed:
                                try:
                                    embedding = embedding_func([text_to_embed])[0]
                                    embedding_clean = format_vector_for_iris(embedding)
                                    embedding_str = ','.join(f"{x:.15g}" for x in embedding_clean)
                                except Exception as e:
                                    logger.warning(f"Embedding error for {doc_id}: {e}")
                        
                        # Prepare document parameters
                        doc_params = (
                            str(doc_id),
                            doc.get("title", ""),
                            doc.get("abstract", ""),
                            doc.get("abstract", ""),
                            json.dumps(doc.get("authors", [])),
                            json.dumps(doc.get("keywords", [])),
                            embedding_str
                        )
                        source_doc_params.append(doc_params)
                        
                        # Generate token embeddings (same as before, but smaller batches)
                        if colbert_doc_encoder_func:
                            try:
                                text_for_tokens = doc.get("abstract") or doc.get("title", "")
                                if text_for_tokens:
                                    token_data = colbert_doc_encoder_func(text_for_tokens)
                                    if token_data and len(token_data) == 2:
                                        tokens, embeddings = token_data
                                        
                                        for idx, (token_text, token_vec) in enumerate(zip(tokens, embeddings)):
                                            try:
                                                token_vec_clean = format_vector_for_iris(token_vec)
                                                token_vec_str = ','.join(f"{x:.15g}" for x in token_vec_clean)
                                                
                                                all_token_params.append((
                                                    str(doc_id),
                                                    idx,
                                                    str(token_text)[:1000],
                                                    token_vec_str,
                                                    "{}"
                                                ))
                                            except Exception as e:
                                                logger.warning(f"Token error: {e}")
                            except Exception as e:
                                logger.error(f"ColBERT error for {doc_id}: {e}")
                        
                    except Exception as e:
                        logger.error(f"Error processing document: {e}")
                        error_count += 1
                
                # Insert documents
                if source_doc_params:
                    try:
                        sql_docs = """
                        INSERT INTO RAG.SourceDocuments
                        (doc_id, title, text_content, abstract, authors, keywords, embedding)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                        """
                        cursor.executemany(sql_docs, source_doc_params)
                        loaded_doc_count += len(source_doc_params)
                        
                        # Insert token embeddings in smaller sub-batches
                        if all_token_params:
                            sql_tokens = """
                            INSERT INTO RAG.DocumentTokenEmbeddings
                            (doc_id, token_sequence_index, token_text, token_embedding, metadata_json)
                            VALUES (?, ?, ?, ?, ?)
                            """
                            
                            # Process tokens in very small sub-batches
                            for i in range(0, len(all_token_params), token_batch_size):
                                token_sub_batch = all_token_params[i:i+token_batch_size]
                                cursor.executemany(sql_tokens, token_sub_batch)
                                loaded_token_count += len(token_sub_batch)
                        
                        # FREQUENT COMMITS for stability
                        connection.commit()
                        
                        # Save checkpoint frequently
                        if source_doc_params:
                            last_doc_id = source_doc_params[-1][0]
                            self.save_checkpoint(batch_end, last_doc_id)
                        
                    except Exception as e:
                        logger.error(f"Database error in batch {batch_idx}: {e}")
                        connection.rollback()
                        error_count += len(current_batch)
                
                # Performance monitoring and connection management
                batch_duration = time.time() - batch_start_time
                elapsed = time.time() - start_time
                rate = loaded_doc_count / elapsed if elapsed > 0 else 0
                
                self.log_performance(batch_duration, rate, batch_idx)
                
                logger.info(f"📊 Batch {batch_idx}: {loaded_doc_count} docs, {loaded_token_count} tokens ({rate:.2f} docs/sec, {batch_duration:.1f}s)")
                
                # Refresh connection every 50 batches
                if batch_idx % 50 == 0 and batch_idx > 0:
                    connection = self.refresh_connection(connection)
                    cursor = connection.cursor()
                    logger.info("🔄 Connection refreshed")
                
                # Force garbage collection every 20 batches
                if batch_idx % 20 == 0:
                    gc.collect()
            
            cursor.close()
            
        except Exception as e:
            logger.error(f"Error in conservative loading: {e}")
        
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

# Export the main function for use
def load_documents_conservative_optimized(connection, documents, embedding_func=None, colbert_doc_encoder_func=None, batch_size=8, token_batch_size=150):
    """Conservative optimized document loading function."""
    loader = ConservativeOptimizedLoader()
    return loader.load_documents_conservative(
        connection, documents, embedding_func, colbert_doc_encoder_func, batch_size, token_batch_size
    )
