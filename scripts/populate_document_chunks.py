#!/usr/bin/env python3
"""
Populate ChunkedDocuments table for CRAG and NodeRAG pipelines.

This script creates document chunks with different strategies:
- Sliding window chunks for better context overlap
- Semantic chunks based on paragraph boundaries
- Fixed-size chunks for consistency
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
import hashlib
import re
from typing import List, Dict, Any, Tuple
from common.iris_connection_manager import get_iris_connection
from common.utils import get_embedding_func
from common.db_vector_utils import insert_vector

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DocumentChunker:
    """Create document chunks using various strategies."""
    
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 128):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.embedding_func = get_embedding_func()
        
    def create_sliding_window_chunks(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        """Create overlapping chunks using sliding window."""
        chunks = []
        words = text.split()
        
        if not words:
            return chunks
            
        # Calculate chunk boundaries
        step = self.chunk_size - self.chunk_overlap
        
        for i in range(0, len(words), step):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            # Skip very short chunks
            if len(chunk_text) < 50:
                continue
                
            # Generate chunk ID
            chunk_id = hashlib.md5(f"{doc_id}_sliding_{i}_{chunk_text[:50]}".encode()).hexdigest()[:16]
            
            chunks.append({
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "chunk_text": chunk_text,
                "chunk_index": len(chunks),
                "chunk_type": "sliding_window",
                "metadata": {
                    "chunk_type": "sliding_window",
                    "start_word": i,
                    "end_word": min(i + self.chunk_size, len(words)),
                    "overlap_size": self.chunk_overlap
                }
            })
            
            # Stop if we've reached the end
            if i + self.chunk_size >= len(words):
                break
                
        return chunks
    
    def create_semantic_chunks(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        """Create chunks based on semantic boundaries (paragraphs, sections)."""
        chunks = []
        
        # Split by paragraphs (double newline or common section markers)
        paragraphs = re.split(r'\n\n+|(?=\n(?:Abstract|Introduction|Methods|Results|Discussion|Conclusion)\s*\n)', text)
        
        current_chunk = []
        current_size = 0
        
        for para in paragraphs:
            para = para.strip()
            if not para:
                continue
                
            para_words = para.split()
            para_size = len(para_words)
            
            # If adding this paragraph exceeds chunk size, save current chunk
            if current_size + para_size > self.chunk_size and current_chunk:
                chunk_text = '\n\n'.join(current_chunk)
                chunk_id = hashlib.md5(f"{doc_id}_semantic_{len(chunks)}_{chunk_text[:50]}".encode()).hexdigest()[:16]
                
                chunks.append({
                    "chunk_id": chunk_id,
                    "document_id": doc_id,
                    "chunk_text": chunk_text,
                    "chunk_index": len(chunks),
                    "chunk_type": "semantic",
                    "metadata": {
                        "chunk_type": "semantic",
                        "paragraph_count": len(current_chunk),
                        "word_count": current_size
                    }
                })
                
                # Start new chunk with overlap (include last paragraph)
                if self.chunk_overlap > 0 and para_size < self.chunk_overlap:
                    current_chunk = [para]
                    current_size = para_size
                else:
                    current_chunk = []
                    current_size = 0
            
            # Add paragraph to current chunk
            current_chunk.append(para)
            current_size += para_size
        
        # Save final chunk
        if current_chunk:
            chunk_text = '\n\n'.join(current_chunk)
            chunk_id = hashlib.md5(f"{doc_id}_semantic_{len(chunks)}_{chunk_text[:50]}".encode()).hexdigest()[:16]
            
            chunks.append({
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "chunk_text": chunk_text,
                "chunk_index": len(chunks),
                "chunk_type": "semantic",
                "metadata": {
                    "paragraph_count": len(current_chunk),
                    "word_count": current_size
                }
            })
            
        return chunks
    
    def create_fixed_chunks(self, text: str, doc_id: str) -> List[Dict[str, Any]]:
        """Create fixed-size chunks without overlap."""
        chunks = []
        words = text.split()
        
        for i in range(0, len(words), self.chunk_size):
            chunk_words = words[i:i + self.chunk_size]
            chunk_text = ' '.join(chunk_words)
            
            # Skip very short chunks
            if len(chunk_text) < 50:
                continue
                
            chunk_id = hashlib.md5(f"{doc_id}_fixed_{i}_{chunk_text[:50]}".encode()).hexdigest()[:16]
            
            chunks.append({
                "chunk_id": chunk_id,
                "document_id": doc_id,
                "chunk_text": chunk_text,
                "chunk_index": len(chunks),
                "chunk_type": "fixed",
                "metadata": {
                    "chunk_type": "fixed",
                    "start_word": i,
                    "end_word": min(i + self.chunk_size, len(words))
                }
            })
            
        return chunks

def populate_chunks(limit: int = 100, chunk_strategy: str = "all"):
    """Populate ChunkedDocuments table with document chunks."""
    
    connection = get_iris_connection()
    cursor = connection.cursor()
    chunker = DocumentChunker()
    
    try:
        # Get documents that don't have chunks yet
        cursor.execute("""
            SELECT d.doc_id, d.title, d.text_content 
            FROM RAG.SourceDocuments d
            WHERE d.doc_id NOT IN (
                SELECT DISTINCT doc_id FROM RAG.ChunkedDocuments
            )
            AND d.text_content IS NOT NULL
            LIMIT ?
        """, [limit])
        
        documents = cursor.fetchall()
        logger.info(f"Found {len(documents)} documents without chunks")
        
        total_chunks = 0
        chunk_strategies = {
            "sliding": chunker.create_sliding_window_chunks,
            "semantic": chunker.create_semantic_chunks,
            "fixed": chunker.create_fixed_chunks
        }
        
        # Determine which strategies to use
        if chunk_strategy == "all":
            strategies_to_use = chunk_strategies.keys()
        elif chunk_strategy in chunk_strategies:
            strategies_to_use = [chunk_strategy]
        else:
            logger.error(f"Unknown chunk strategy: {chunk_strategy}")
            return
        
        for i, (doc_id, title, content) in enumerate(documents):
            if i % 10 == 0:
                logger.info(f"Processing document {i+1}/{len(documents)}...")
            
            # Combine title and content
            full_text = f"{title or ''}\n\n{content or ''}"
            
            # Create chunks using selected strategies
            for strategy_name in strategies_to_use:
                strategy_func = chunk_strategies[strategy_name]
                chunks = strategy_func(full_text, doc_id)
                
                # Store chunks
                for chunk in chunks:
                    try:
                        # Generate embedding for chunk
                        embedding = chunker.embedding_func(chunk["chunk_text"])
                        
                        # Convert metadata to JSON string
                        import json
                        metadata_str = json.dumps(chunk.get("metadata", {}))
                        
                        # Insert chunk with embedding
                        success = insert_vector(
                            cursor=cursor,
                            table_name="RAG.ChunkedDocuments",
                            vector_column_name="embedding",
                            vector_data=embedding,
                            target_dimension=384,
                            key_columns={"chunk_id": chunk["chunk_id"]},
                            additional_data={
                                "doc_id": chunk["document_id"],
                                "chunk_text": chunk["chunk_text"][:10000],  # Limit text length
                                "chunk_index": chunk["chunk_index"],
                                "metadata": metadata_str
                            }
                        )
                        
                        if success:
                            total_chunks += 1
                        
                    except Exception as e:
                        logger.warning(f"Failed to insert chunk: {e}")
            
            # Commit periodically
            if (i + 1) % 10 == 0:
                connection.commit()
                logger.info(f"Committed {total_chunks} chunks so far...")
        
        # Final commit
        connection.commit()
        
        logger.info(f"\n✅ Successfully populated {total_chunks} chunks")
        
        # Show statistics - simplified since we can't use LENGTH on stream fields
        cursor.execute("""
            SELECT COUNT(*) as chunk_count
            FROM RAG.ChunkedDocuments
        """)
        
        row = cursor.fetchone()
        if row:
            logger.info("\nChunk statistics:")
            logger.info(f"  Total chunks: {row[0]}")
        
        # Show documents with chunks
        cursor.execute("""
            SELECT COUNT(DISTINCT doc_id) as doc_count,
                   COUNT(*) as total_chunks,
                   AVG(chunks_per_doc) as avg_chunks_per_doc
            FROM (
                SELECT doc_id, COUNT(*) as chunks_per_doc
                FROM RAG.ChunkedDocuments
                GROUP BY doc_id
            ) doc_chunks
        """)
        
        row = cursor.fetchone()
        if row:
            logger.info(f"\nChunking statistics:")
            logger.info(f"  Documents with chunks: {row[0]}")
            logger.info(f"  Total chunks: {row[1]}")
            logger.info(f"  Average chunks per document: {row[2]:.1f}")
            
    except Exception as e:
        logger.error(f"Error populating chunks: {e}")
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=50, help="Number of documents to process")
    parser.add_argument("--strategy", choices=["sliding", "semantic", "fixed", "all"], 
                       default="all", help="Chunking strategy to use")
    args = parser.parse_args()
    
    logger.info(f"Populating ChunkedDocuments table using {args.strategy} strategy...")
    populate_chunks(args.limit, args.strategy)