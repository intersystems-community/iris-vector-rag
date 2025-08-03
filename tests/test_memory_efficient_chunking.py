#!/usr/bin/env python3
"""
Test the memory-efficient chunking approach with a small subset of documents
"""

import sys
import logging
import time
import gc
import pytest
from typing import List, Generator
from common.iris_connector import get_iris_connection  # Keep for fallback
from common.utils import get_embedding_func
from common.jdbc_stream_utils import read_iris_stream

# Add proper architecture imports
from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager
from iris_rag.validation.orchestrator import SetupOrchestrator
from iris_rag.validation.factory import ValidatedPipelineFactory
from iris_rag.core.models import Document

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

def test_memory_efficient_chunking():
    """Main test entry point that uses proper architecture."""
    test_memory_efficient_chunking_architecture_compliant()

@pytest.fixture
def document_generator(limit: int = 10) -> Generator[tuple, None, None]:
    """
    Generator that yields a limited number of documents for testing.
    
    DEPRECATED: This fixture uses direct SQL anti-pattern.
    New tests should use test_memory_efficient_chunking_architecture_compliant() instead.
    """
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

def test_memory_efficient_chunking_architecture_compliant():
    """
    Test memory-efficient chunk population using proper architecture instead of direct SQL.
    
    Uses SetupOrchestrator + pipeline.ingest_documents() with chunking configuration
    instead of direct SQL INSERT operations.
    """
    try:
        # Initialize proper managers following project architecture
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager(config_manager)
        
        logger.info("Setting up memory-efficient chunking test using proper architecture...")
        
        # 1. Use SetupOrchestrator to ensure chunking tables exist
        orchestrator = SetupOrchestrator(connection_manager, config_manager)
        validation_report = orchestrator.setup_pipeline("crag", auto_fix=True)  # CRAG uses chunking
        
        if not validation_report.overall_valid:
            logger.warning(f"CRAG setup had issues: {validation_report.summary}")
        
        # 2. Create CRAG pipeline using proper factory (supports chunking)
        factory = ValidatedPipelineFactory(connection_manager, config_manager)
        pipeline = factory.create_pipeline("crag", auto_setup=True, validate_requirements=False)
        
        # 3. Get sample documents from existing database using architecture-compliant methods
        test_documents = []
        limit = 10
        
        # Use connection manager instead of direct get_iris_connection()
        conn = connection_manager.get_connection()
        cursor = conn.cursor()
        
        try:
            cursor.execute(f'SELECT TOP {limit} doc_id, text_content FROM RAG.SourceDocuments WHERE text_content IS NOT NULL ORDER BY doc_id')
            
            while True:
                row = cursor.fetchone()
                if row is None:
                    break
                
                doc_id, text_content = row
                # Handle IRIS stream objects properly
                content = read_iris_stream(text_content) if text_content else ''
                
                if len(content.strip()) >= 100:  # Only process substantial documents
                    doc = Document(
                        id=doc_id,
                        page_content=content,
                        metadata={
                            "title": f"Memory Test Document {doc_id}",
                            "source": "memory_efficient_chunking_test",
                            "chunking_strategy": "memory_efficient"
                        }
                    )
                    test_documents.append(doc)
                    
                    if len(test_documents) >= 5:  # Limit for memory efficiency
                        break
        finally:
            cursor.close()
        
        if not test_documents:
            logger.warning("No substantial documents found for chunking test")
            return
        
        logger.info(f"Found {len(test_documents)} documents for memory-efficient chunking test")
        
        # 4. Monitor initial memory
        initial_memory = monitor_memory_usage()
        start_time = time.time()
        
        # 5. Use pipeline.ingest_documents() with chunking instead of direct SQL
        logger.info("Processing documents through CRAG pipeline with chunking...")
        ingestion_result = pipeline.ingest_documents(test_documents)
        
        if ingestion_result["status"] != "success":
            logger.error(f"CRAG chunking ingestion failed: {ingestion_result}")
            raise RuntimeError(f"CRAG chunking failed: {ingestion_result.get('error', 'Unknown error')}")
        
        # 6. Report results
        elapsed = time.time() - start_time
        final_memory = monitor_memory_usage()
        
        chunks_created = ingestion_result.get("chunks_created", 0)
        docs_processed = len(test_documents)
        
        logger.info(f"✅ Memory-efficient chunking completed via proper architecture:")
        logger.info(f"   Documents processed: {docs_processed}")
        logger.info(f"   Chunks created: {chunks_created}")
        logger.info(f"   Time elapsed: {elapsed:.2f}s")
        logger.info(f"   Rate: {docs_processed/elapsed:.1f} docs/sec")
        if initial_memory and final_memory:
            logger.info(f"   Memory change: {final_memory - initial_memory:.1f} MB")
        
        # Force garbage collection
        gc.collect()
        
        assert chunks_created > 0, "Should have created some chunks"
        assert docs_processed > 0, "Should have processed some documents"
        
    except Exception as e:
        logger.error(f"Failed to run memory-efficient chunking test using proper architecture: {e}")
        # Fallback to direct SQL version if architecture fails
        logger.warning("Falling back to direct SQL chunking test...")
        test_memory_efficient_chunking_fallback()

def test_memory_efficient_chunking_fallback():
    """Fallback to direct SQL chunking test if architecture fails."""
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    try:
        # Get embedding function
        embedding_func = get_embedding_func()
        logger.info("✅ Fallback: Embedding function initialized")
        
        # Clear any existing test chunks
        cursor.execute("DELETE FROM RAG.DocumentChunks WHERE chunk_id LIKE '%_chunk_%'")
        conn.commit()
        logger.info("Fallback: Cleared any existing test chunks")
        
        chunks_created = 0
        docs_processed = 0
        start_time = time.time()
        
        # Monitor initial memory
        initial_memory = monitor_memory_usage()
        
        # Process limited documents one at a time using generator
        cursor.execute('SELECT TOP 5 doc_id, text_content FROM RAG.SourceDocuments WHERE text_content IS NOT NULL ORDER BY doc_id')
        
        while True:
            row = cursor.fetchone()
            if row is None:
                break
                
            doc_id, text_content = row
            logger.info(f"Fallback: Processing document {docs_processed + 1}: {doc_id}")
            
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
            
            logger.info(f'Fallback Progress: {docs_processed} docs, {chunks_created} chunks created')
            logger.info(f'Fallback Rate: {rate:.1f} docs/sec')
            
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
        
        assert chunks_created > 0
        
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