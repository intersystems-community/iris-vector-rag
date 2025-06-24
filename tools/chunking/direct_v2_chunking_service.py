#!/usr/bin/env python3
"""
Direct V2 Chunking Service
Chunks documents and writes directly to DocumentChunks_V2 with VECTOR columns
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
    
    def process_document(self, doc_id: str, text_content: str) -> List[Tuple]:
        """Process a single document into chunks with embeddings"""
        chunks = self.chunk_text(text_content)
        chunk_data = []
        
        for i, chunk_text in enumerate(chunks):
            if len(chunk_text.strip()) < 10:  # Skip very short chunks
                continue
                
            # Generate embedding
            try:
                embedding = self.embedding_func([chunk_text])[0]
                embedding_str = ','.join([f'{x:.10f}' for x in embedding])
                
                chunk_data.append((
                    str(uuid.uuid4()),  # chunk_id
                    doc_id,             # doc_id
                    chunk_text,         # chunk_text
                    i,                  # chunk_index
                    embedding_str,      # embedding (VARCHAR)
                    'text',             # chunk_type
                    embedding_str       # for VECTOR conversion
                ))
            except Exception as e:
                print(f"⚠️  Error generating embedding for chunk {i} of doc {doc_id}: {e}")
                continue
                
        return chunk_data
    
    def insert_chunks_batch(self, chunk_batch: List[Tuple]):
        """Insert a batch of chunks into DocumentChunks_V2"""
        cursor = self.conn.cursor()
        
        try:
            # Insert chunks one by one to handle TO_VECTOR properly
            for chunk_data in chunk_batch:
                chunk_id, doc_id, chunk_text, chunk_index, embedding_str, chunk_type, vector_str = chunk_data
                
                # Build SQL with direct string interpolation for TO_VECTOR
                sql = f'''
                INSERT INTO RAG.DocumentChunks_V2 (
                    chunk_id, doc_id, chunk_text, chunk_index,
                    embedding, chunk_type, chunk_embedding_vector
                ) VALUES (?, ?, ?, ?, ?, ?, TO_VECTOR('{vector_str}', 'FLOAT', 384))
                '''
                
                # Execute with the non-vector parameters
                cursor.execute(sql, (chunk_id, doc_id, chunk_text, chunk_index, embedding_str, chunk_type))
            
            self.conn.commit()
            print(f"✅ Inserted batch of {len(chunk_batch)} chunks")
            
        except Exception as e:
            print(f"❌ Error inserting batch: {e}")
            self.conn.rollback()
            raise
        finally:
            cursor.close()
    
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
            
            chunk_batch = []
            processed_docs = 0
            total_chunks = 0
            start_time = time.time()
            
            for doc_id, text_content in documents:
                try:
                    # Process document
                    doc_chunks = self.process_document(doc_id, text_content)
                    chunk_batch.extend(doc_chunks)
                    total_chunks += len(doc_chunks)
                    processed_docs += 1
                    
                    # Insert batch when it reaches batch_size
                    if len(chunk_batch) >= self.batch_size:
                        self.insert_chunks_batch(chunk_batch)
                        chunk_batch = []
                    
                    # Progress reporting
                    if processed_docs % 100 == 0:
                        elapsed = time.time() - start_time
                        rate = processed_docs / elapsed
                        eta = (total_docs - processed_docs) / rate if rate > 0 else 0
                        print(f"📊 Progress: {processed_docs:,}/{total_docs:,} docs ({processed_docs/total_docs*100:.1f}%) | "
                              f"Rate: {rate:.1f} docs/s | ETA: {eta/60:.1f}m | Chunks: {total_chunks:,}")
                
                except Exception as e:
                    print(f"⚠️  Error processing document {doc_id}: {e}")
                    continue
            
            # Insert remaining chunks
            if chunk_batch:
                self.insert_chunks_batch(chunk_batch)
            
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
    print("🚀 Direct V2 Chunking Service")
    print("=" * 50)
    
    # Initialize service
    service = DirectV2ChunkingService(
        chunk_size=512,
        chunk_overlap=50,
        batch_size=50  # Smaller batches for stability
    )
    
    # Run chunking (limit to 1000 docs for initial test)
    service.run_chunking(limit=1000)

if __name__ == "__main__":
    main()