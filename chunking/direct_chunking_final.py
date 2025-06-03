#!/usr/bin/env python3
"""
Final Direct Chunking Service
Chunks documents and writes directly to both original and _V2 tables
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from common.iris_connector import get_iris_connection
from common.utils import get_embedding_func
import uuid
import time
from typing import List, Tuple

class FinalChunkingService:
    def __init__(self, chunk_size: int = 512, chunk_overlap: int = 50):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
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
    
    def run_chunking(self, limit: int = None):
        """Run the chunking process"""
        cursor = self.conn.cursor()
        
        try:
            # Clear existing chunks
            print("🧹 Clearing existing chunks...")
            cursor.execute('DELETE FROM RAG.DocumentChunks')
            cursor.execute('DELETE FROM RAG.DocumentChunks_V2')
            self.conn.commit()
            
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
                    # Chunk the text
                    chunks = self.chunk_text(text_content)
                    
                    for i, chunk_text in enumerate(chunks):
                        if len(chunk_text.strip()) < 10:  # Skip very short chunks
                            continue
                            
                        chunk_id = str(uuid.uuid4())
                        
                        # Generate embedding
                        try:
                            embedding = self.embedding_func([chunk_text])[0]
                            embedding_str = ','.join([f'{x:.10f}' for x in embedding])
                            
                            # Insert into original table
                            cursor.execute('''
                                INSERT INTO RAG.DocumentChunks 
                                (chunk_id, doc_id, chunk_text, chunk_index, embedding, chunk_type)
                                VALUES (?, ?, ?, ?, ?, ?)
                            ''', (chunk_id, doc_id, chunk_text, i, embedding_str, 'text'))
                            
                            # Insert into _V2 table (without vector for now)
                            cursor.execute('''
                                INSERT INTO RAG.DocumentChunks_V2 
                                (chunk_id, doc_id, chunk_text, chunk_index, embedding, chunk_type)
                                VALUES (?, ?, ?, ?, ?, ?)
                            ''', (chunk_id, doc_id, chunk_text, i, embedding_str, 'text'))
                            
                            total_chunks += 1
                            
                        except Exception as e:
                            print(f"⚠️  Error generating embedding for chunk {i} of doc {doc_id}: {e}")
                            continue
                    
                    processed_docs += 1
                    
                    # Commit every 10 documents
                    if processed_docs % 10 == 0:
                        self.conn.commit()
                        
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
            
            # Final commit
            self.conn.commit()
            
            elapsed = time.time() - start_time
            print(f"\n🎉 CHUNKING COMPLETE!")
            print(f"📊 Processed: {processed_docs:,} documents")
            print(f"📊 Generated: {total_chunks:,} chunks")
            print(f"📊 Time: {elapsed/60:.1f} minutes")
            print(f"📊 Rate: {processed_docs/elapsed:.1f} docs/s")
            
            # Verify results
            cursor.execute('SELECT COUNT(*) FROM RAG.DocumentChunks')
            chunks_count = cursor.fetchone()[0]
            
            cursor.execute('SELECT COUNT(*) FROM RAG.DocumentChunks_V2')
            v2_count = cursor.fetchone()[0]
            
            print(f"\n✅ Final verification:")
            print(f"   - Chunks in original table: {chunks_count:,}")
            print(f"   - Chunks in _V2 table: {v2_count:,}")
            
            # Now update the vector columns in _V2 using a simple approach
            print(f"\n🔄 Updating VECTOR columns in _V2 table...")
            self.update_vector_columns(cursor)
            
        except Exception as e:
            print(f"❌ Chunking failed: {e}")
            raise
        finally:
            cursor.close()
    
    def update_vector_columns(self, cursor):
        """Update vector columns using a batch approach"""
        try:
            # Get count of records needing update
            cursor.execute('SELECT COUNT(*) FROM RAG.DocumentChunks_V2 WHERE chunk_embedding_vector IS NULL')
            to_update = cursor.fetchone()[0]
            print(f"📊 Records to update: {to_update:,}")
            
            # Update in batches using a simple SQL approach
            batch_size = 100
            updated = 0
            
            while updated < to_update:
                # Use a simple UPDATE with subquery
                cursor.execute(f'''
                    UPDATE RAG.DocumentChunks_V2 
                    SET chunk_embedding_vector = TO_VECTOR(embedding, 'FLOAT', 384)
                    WHERE chunk_id IN (
                        SELECT TOP {batch_size} chunk_id 
                        FROM RAG.DocumentChunks_V2 
                        WHERE chunk_embedding_vector IS NULL
                    )
                ''')
                
                rows_affected = cursor.rowcount
                if rows_affected == 0:
                    break
                    
                updated += rows_affected
                self.conn.commit()
                
                print(f"📊 Updated {updated:,}/{to_update:,} records ({updated/to_update*100:.1f}%)")
            
            # Final verification
            cursor.execute('SELECT COUNT(*) FROM RAG.DocumentChunks_V2 WHERE chunk_embedding_vector IS NOT NULL')
            final_count = cursor.fetchone()[0]
            print(f"\n✅ Vector update complete: {final_count:,} records with VECTOR data")
            
        except Exception as e:
            print(f"⚠️  Vector update failed: {e}")
            print("💡 You may need to update vectors manually using IRIS SQL or ObjectScript")

def main():
    print("🚀 Final Direct Chunking Service")
    print("=" * 50)
    
    # Initialize service
    service = FinalChunkingService(
        chunk_size=512,
        chunk_overlap=50
    )
    
    # Run chunking (limit to 1000 docs for initial test)
    service.run_chunking(limit=1000)

if __name__ == "__main__":
    main()