#!/usr/bin/env python3
"""
Direct V2 Chunking Service - Simplified Version
Chunks documents and writes directly to DocumentChunks_V2 with VECTOR columns
Uses a two-step approach to avoid SQL parsing issues
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from common.iris_connector import get_iris_connection
from common.utils import get_embedding_func
import uuid
import time
from typing import List, Tuple

class DirectV2ChunkingService:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50, batch_size: int = 100):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.batch_size = batch_size
        self.conn = get_iris_connection()
        self.embedding_func = get_embedding_func()
        
    def chunk_text(self, text: str) -> List[str]:
        """Split text into overlapping chunks"""
        if not text or len(text.strip()) == 0:
            return []
            
        words = text.split()
        if len(words) <= self.chunk_size:
            return [text]
            
        chunks = []
        start = 0
        
        while start < len(words):
            end = min(start + self.chunk_size, len(words))
            chunk = ' '.join(words[start:end])
            chunks.append(chunk)
            
            if end >= len(words):
                break
                
            start = end - self.chunk_overlap
            
        return chunks
    
    def process_and_insert_document(self, doc_id: str, text_content: str) -> int:
        """Process a single document and insert chunks directly"""
        chunks = self.chunk_text(text_content)
        cursor = self.conn.cursor()
        chunks_inserted = 0
        
        for i, chunk_text in enumerate(chunks):
            if len(chunk_text.strip()) < 10:  # Skip very short chunks
                continue
                
            try:
                # Generate embedding
                embedding = self.embedding_func([chunk_text])[0]
                embedding_str = ','.join([f'{x:.10f}' for x in embedding])
                chunk_id = str(uuid.uuid4())
                
                # Step 1: Insert without vector column
                insert_sql = '''
                INSERT INTO RAG.DocumentChunks_V2 (
                    chunk_id, doc_id, chunk_text, chunk_index, 
                    embedding, chunk_type
                ) VALUES (?, ?, ?, ?, ?, ?)
                '''
                
                cursor.execute(insert_sql, (
                    chunk_id, doc_id, chunk_text, i, 
                    embedding_str, 'text'
                ))
                
                # Step 2: Update with vector column using direct SQL
                update_sql = f'''
                UPDATE RAG.DocumentChunks_V2 
                SET chunk_embedding_vector = TO_VECTOR('{embedding_str}', 'FLOAT', 384)
                WHERE chunk_id = '{chunk_id}'
                '''
                
                cursor.execute(update_sql)
                chunks_inserted += 1
                
            except Exception as e:
                print(f"⚠️  Error processing chunk {i} of doc {doc_id}: {e}")
                self.conn.rollback()
                continue
                
        self.conn.commit()
        return chunks_inserted
    
    def run_chunking(self, limit: int = None):
        """Run the chunking process"""
        cursor = self.conn.cursor()
        
        try:
            # Get documents to process
            if limit:
                cursor.execute(f'SELECT TOP {limit} doc_id, text_content FROM RAG.SourceDocuments WHERE text_content IS NOT NULL ORDER BY doc_id')
            else:
                cursor.execute('SELECT doc_id, text_content FROM RAG.SourceDocuments WHERE text_content IS NOT NULL ORDER BY doc_id')
            
            documents = cursor.fetchall()
            total_docs = len(documents)
            print(f"🚀 Starting chunking for {total_docs:,} documents")
            
            processed_docs = 0
            total_chunks = 0
            start_time = time.time()
            
            for doc_id, text_content in documents:
                try:
                    # Process and insert document chunks
                    chunks_inserted = self.process_and_insert_document(doc_id, text_content)
                    total_chunks += chunks_inserted
                    processed_docs += 1
                    
                    # Progress reporting
                    if processed_docs % 10 == 0:
                        elapsed = time.time() - start_time
                        rate = processed_docs / elapsed
                        eta = (total_docs - processed_docs) / rate if rate > 0 else 0
                        print(f"📊 Progress: {processed_docs:,}/{total_docs:,} docs ({processed_docs/total_docs*100:.1f}%) | "
                              f"Rate: {rate:.1f} docs/s | ETA: {eta/60:.1f}m | Chunks: {total_chunks:,}")
                
                except Exception as e:
                    print(f"⚠️  Error processing document {doc_id}: {e}")
                    continue
            
            elapsed = time.time() - start_time
            print(f"\n🎉 CHUNKING COMPLETE!")
            print(f"📊 Processed: {processed_docs:,} documents")
            print(f"📊 Generated: {total_chunks:,} chunks")
            print(f"📊 Time: {elapsed/60:.1f} minutes")
            print(f"📊 Rate: {processed_docs/elapsed:.1f} docs/s")
            
            # Verify results
            cursor.execute('SELECT COUNT(*) FROM RAG.DocumentChunks_V2')
            final_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM RAG.DocumentChunks_V2 WHERE chunk_embedding_vector IS NOT NULL')
            vector_count = cursor.fetchone()[0]
            
            print(f"✅ Final verification:")
            print(f"   - Total chunks in DB: {final_count:,}")
            print(f"   - Chunks with VECTOR data: {vector_count:,}")
            
        except Exception as e:
            print(f"❌ Chunking failed: {e}")
            raise
        finally:
            cursor.close()

def main():
    print("🚀 Direct V2 Chunking Service (Simplified)")
    print("=" * 50)
    
    # Initialize service
    service = DirectV2ChunkingService(
        chunk_size=512,
        chunk_overlap=50,
        batch_size=50  # Not used in this version
    )
    
    # Run chunking (limit to 1000 docs for initial test)
    service.run_chunking(limit=1000)

if __name__ == "__main__":
    main()