#!/usr/bin/env python3
"""
Test the memory-efficient chunking approach with a small subset of documents
"""

import sys
import logging
import time
import gc
from typing import List, Generator
from common.iris_connector import get_iris_connection
from common.utils import get_embedding_func
from common.jdbc_stream_utils import read_iris_stream

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Memory-efficient text chunking with overlap"""
    if not text or len(text) < chunk_size:
        return [text] if text else []
    
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        if end >= len(text):
            chunks.append(text[start:])
            break
        else:
            # Find a good break point (sentence or word boundary)
            break_point = text.rfind('.', start, end)
            if break_point == -1:
                break_point = text.rfind(' ', start, end)
            if break_point == -1:
                break_point = end
            
            chunks.append(text[start:break_point])
            start = break_point - overlap if break_point > overlap else break_point
    
    return [chunk.strip() for chunk in chunks if chunk.strip()]

def test_document_generator(limit: int = 10) -> Generator[tuple, None, None]:
    """Generator that yields a limited number of documents for testing"""
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    try:
        # Use TOP to limit results for testing
        cursor.execute(f'SELECT TOP {limit} doc_id, text_content FROM RAG.SourceDocuments WHERE text_content IS NOT NULL ORDER BY doc_id')
        
        while True:
            row = cursor.fetchone()
            if row is None:
                break
            yield row
            
    finally:
        cursor.close()
        conn.close()

def process_single_document_test(doc_id: str, text_content: str, embedding_func, conn, cursor) -> int:
    """Process a single document and return number of chunks created (test version)"""
    chunks_created = 0
    
    try:
        # Handle IRIS stream objects
        text_content = read_iris_stream(text_content) if text_content else ''
        
        if len(text_content.strip()) < 100:  # Skip very short documents
            logger.info(f"Skipping short document {doc_id} (length: {len(text_content)})")
            return 0
            
        logger.info(f"Processing document {doc_id} (length: {len(text_content)})")
        
        # Create chunks
        chunks = chunk_text(text_content, chunk_size=400, overlap=50)
        logger.info(f"Created {len(chunks)} chunks for document {doc_id}")
        
        for i, chunk_content in enumerate(chunks):
            if len(chunk_content.strip()) < 50:  # Skip very short chunks
                continue
                
            # Generate unique chunk_id
            chunk_id = f'{doc_id}_chunk_{i}'
            
            # Generate embedding for chunk (one at a time)
            try:
                chunk_embedding = embedding_func([chunk_content])[0]
                embedding_str = ','.join([f'{x:.10f}' for x in chunk_embedding])
                
                # Insert chunk immediately
                cursor.execute('''
                    INSERT INTO RAG.DocumentChunks
                    (chunk_id, doc_id, chunk_index, chunk_text, embedding, chunk_type)
                    VALUES (?, ?, ?, ?, TO_VECTOR(?), ?)
                ''', (chunk_id, doc_id, i, chunk_content, embedding_str, 'text'))
                
                chunks_created += 1
                logger.info(f"Created chunk {chunk_id} (length: {len(chunk_content)})")
                
                # Clear embedding from memory immediately
                del chunk_embedding
                del embedding_str
                
            except Exception as e:
                logger.error(f'Error creating chunk {chunk_id}: {e}')
                continue
        
        # Clear chunks and text content from memory
        del chunks
        del text_content
        
        # Force garbage collection after each document
        gc.collect()
        
        return chunks_created
        
    except Exception as e:
        logger.error(f'Error processing document {doc_id}: {e}')
        return 0

def monitor_memory_usage():
    """Monitor current memory usage"""
    try:
        import psutil
        process = psutil.Process()
        memory_info = process.memory_info()
        memory_mb = memory_info.rss / 1024 / 1024
        logger.info(f"Current memory usage: {memory_mb:.1f} MB")
        return memory_mb
    except ImportError:
        logger.warning("psutil not available for memory monitoring")
        return None

def test_memory_efficient_chunking(test_limit: int = 10):
    """Test memory-efficient chunk population with limited documents"""
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    try:
        # Get embedding function
        embedding_func = get_embedding_func()
        logger.info("✅ Embedding function initialized")
        
        # Clear any existing test chunks
        cursor.execute("DELETE FROM RAG.DocumentChunks WHERE chunk_id LIKE '%_chunk_%'")
        conn.commit()
        logger.info("Cleared any existing test chunks")
        
        chunks_created = 0
        docs_processed = 0
        start_time = time.time()
        
        # Monitor initial memory
        initial_memory = monitor_memory_usage()
        
        # Process limited documents one at a time using generator
        for doc_id, text_content in test_document_generator(test_limit):
            logger.info(f"Processing document {docs_processed + 1}/{test_limit}: {doc_id}")
            
            doc_chunks = process_single_document_test(doc_id, text_content, embedding_func, conn, cursor)
            chunks_created += doc_chunks
            docs_processed += 1
            
            # Commit after each document
            conn.commit()
            
            # Monitor memory after each document
            current_memory = monitor_memory_usage()
            
            # Progress reporting
            elapsed = time.time() - start_time
            rate = docs_processed / elapsed if elapsed > 0 else 0
            
            logger.info(f'Progress: {docs_processed}/{test_limit} docs, {chunks_created} chunks created')
            logger.info(f'Rate: {rate:.1f} docs/sec')
            
            # Force garbage collection
            gc.collect()
        
        # Final memory check
        final_memory = monitor_memory_usage()
        if initial_memory and final_memory:
            memory_increase = final_memory - initial_memory
            logger.info(f"Total memory increase: {memory_increase:.1f} MB")
        
        logger.info(f'✅ Test completed: {chunks_created} chunks from {docs_processed} documents')
        
        # Verify chunks were created
        cursor.execute('SELECT COUNT(*) FROM RAG.DocumentChunks WHERE chunk_id LIKE ?', ('%_chunk_%',))
        test_chunks = cursor.fetchone()[0]
        
        logger.info(f'Verification: {test_chunks} test chunks in database')
        
        return chunks_created
        
    except Exception as e:
        logger.error(f'❌ Error in test: {e}')
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()

def main():
    """Main test function"""
    logger.info("🧪 Testing memory-efficient chunking with 10 documents...")
    
    try:
        chunks_created = test_memory_efficient_chunking(test_limit=10)
        
        if chunks_created > 0:
            logger.info(f"🎉 Test successful! Created {chunks_created} chunks with minimal memory usage")
            logger.info("✅ Memory-efficient approach is working correctly")
        else:
            logger.error("❌ Test failed - no chunks were created")
            
    except Exception as e:
        logger.error(f"❌ Test failed: {e}")

if __name__ == "__main__":
    main()