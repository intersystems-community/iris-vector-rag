#!/usr/bin/env python3
"""
Populate existing RAG.DocumentChunks table using schema manager.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from typing import List, Dict
from common.database_schema_manager import get_schema_manager
from common.iris_connector import get_iris_connection
from common.utils import get_embedding_func

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ChunkPopulator:
    def __init__(self):
        self.schema = get_schema_manager()
        self.connection = get_iris_connection()
        self.embedding_func = get_embedding_func()
    
    def chunk_text(self, text: str, chunk_size: int = 500) -> List[str]:
        """Simple text chunking by character count."""
        chunks = []
        words = text.split()
        current_chunk = []
        current_length = 0
        
        for word in words:
            if current_length + len(word) + 1 > chunk_size and current_chunk:
                chunks.append(' '.join(current_chunk))
                current_chunk = [word]
                current_length = len(word)
            else:
                current_chunk.append(word)
                current_length += len(word) + 1
        
        if current_chunk:
            chunks.append(' '.join(current_chunk))
        
        return chunks[:10]  # Max 10 chunks per document
    
    def populate_chunks(self, limit: int = 100):
        """Populate document chunks table."""
        logger.info(f"Populating chunks for up to {limit} documents...")
        
        cursor = self.connection.cursor()
        
        # Get documents
        docs_table = self.schema.get_table_name('source_documents', fully_qualified=True)
        cursor.execute(f"SELECT doc_id, title, text_content FROM {docs_table} LIMIT {limit}")
        documents = cursor.fetchall()
        
        logger.info(f"Processing {len(documents)} documents...")
        
        chunks_table = self.schema.get_table_name('document_chunks', fully_qualified=True)
        
        # Clear existing chunks
        cursor.execute(f"DELETE FROM {chunks_table}")
        logger.info(f"Cleared existing chunks")
        
        total_chunks = 0
        for i, (doc_id, title, content) in enumerate(documents):
            if i % 50 == 0:
                logger.info(f"Processing document {i+1}/{len(documents)}")
            
            # Create chunks
            full_text = f"{title} {content}"
            chunks = self.chunk_text(full_text)
            
            # Insert chunks
            for chunk_idx, chunk_text in enumerate(chunks):
                try:
                    chunk_id = f"{doc_id}_chunk_{chunk_idx}"
                    
                    # Compute embedding
                    embedding = self.embedding_func(chunk_text)
                    embedding_str = ','.join(map(str, embedding))
                    
                    cursor.execute(f"""
                        INSERT INTO {chunks_table} 
                        (chunk_id, doc_id, chunk_text, chunk_index, chunk_embedding) 
                        VALUES (?, ?, ?, ?, ?)
                    """, (chunk_id, doc_id, chunk_text, chunk_idx, embedding_str))
                    total_chunks += 1
                except Exception as e:
                    logger.warning(f"Failed to insert chunk {chunk_id}: {e}")
            
            if i % 100 == 0:
                self.connection.commit()
        
        self.connection.commit()
        logger.info(f"✅ Populated {total_chunks} chunks for {len(documents)} documents")
        cursor.close()

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--limit', type=int, default=970, help='Number of documents to process')
    args = parser.parse_args()
    
    populator = ChunkPopulator()
    populator.populate_chunks(args.limit)