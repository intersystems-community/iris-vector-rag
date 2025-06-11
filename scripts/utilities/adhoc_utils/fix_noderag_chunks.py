#!/usr/bin/env python3
"""
Fix NodeRAG by populating DocumentChunks table
"""

import sys
import logging
from typing import List
import os # Added for path manipulation

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.iris_connector import get_iris_connection # Updated import
from common.utils import get_embedding_func # Updated import
from common.jdbc_stream_utils import read_iris_stream # Updated import

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def chunk_text(text: str, chunk_size: int = 400, overlap: int = 50) -> List[str]:
    """Simple text chunking with overlap"""
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

def populate_document_chunks():
    """Populate DocumentChunks table with proper chunking"""
    
    conn = get_iris_connection()
    cursor = conn.cursor()
    
    try:
        # Get embedding function
        embedding_func = get_embedding_func()
        logger.info("✅ Embedding function initialized")
        
        # Check current chunks
        cursor.execute('SELECT COUNT(*) FROM RAG.DocumentChunks')
        existing_chunks = cursor.fetchone()[0]
        logger.info(f"Existing chunks: {existing_chunks}")
        
        if existing_chunks > 0:
            user_input = input(f"Found {existing_chunks} existing chunks. Clear and recreate? (y/N): ")
            if user_input.lower() == 'y':
                cursor.execute('DELETE FROM RAG.DocumentChunks')
                conn.commit()
                logger.info("Cleared existing chunks")
        
        # Get documents to chunk
        cursor.execute('SELECT TOP 100 doc_id, text_content FROM RAG.SourceDocuments WHERE text_content IS NOT NULL')
        documents = cursor.fetchall()
        
        logger.info(f'Found {len(documents)} documents to chunk')
        
        chunks_created = 0
        for doc_id, text_content in documents:
            if not text_content:
                continue
                
            # Handle IRIS stream objects
            text_content = read_iris_stream(text_content) if text_content else ''
            
            if len(text_content.strip()) < 100:  # Skip very short documents
                continue
                
            # Create chunks
            chunks = chunk_text(text_content, chunk_size=400, overlap=50)
            
            for i, chunk_content in enumerate(chunks):
                if len(chunk_content.strip()) < 50:  # Skip very short chunks
                    continue
                    
                # Generate unique chunk_id
                chunk_id = f'{doc_id}_chunk_{i}'
                
                # Generate embedding for chunk
                try:
                    chunk_embedding = embedding_func([chunk_content])[0]
                    embedding_str = ','.join([f'{x:.10f}' for x in chunk_embedding])
                    
                    # Insert chunk with all required fields
                    cursor.execute('''
                        INSERT INTO RAG.DocumentChunks
                        (chunk_id, doc_id, chunk_index, chunk_text, embedding, chunk_type)
                        VALUES (?, ?, ?, ?, TO_VECTOR(?), ?)
                    ''', (chunk_id, doc_id, i, chunk_content, embedding_str, 'text'))
                    
                    chunks_created += 1
                    
                except Exception as e:
                    logger.error(f'Error creating chunk {chunk_id}: {e}')
                    continue
            
            if chunks_created % 50 == 0 and chunks_created > 0:
                logger.info(f'Created {chunks_created} chunks...')
        
        conn.commit()
        logger.info(f'✅ Successfully created {chunks_created} chunks')
        
        # Verify chunks were created
        cursor.execute('SELECT COUNT(*) FROM RAG.DocumentChunks')
        total_chunks = cursor.fetchone()[0]
        
        cursor.execute('SELECT COUNT(*) FROM RAG.DocumentChunks WHERE embedding IS NOT NULL')
        chunks_with_embeddings = cursor.fetchone()[0]
        
        logger.info(f'Total chunks: {total_chunks}')
        logger.info(f'Chunks with embeddings: {chunks_with_embeddings}')
        
        return total_chunks
        
    except Exception as e:
        logger.error(f'❌ Error populating chunks: {e}')
        conn.rollback()
        raise
    finally:
        cursor.close()
        conn.close()

def test_noderag():
    """Test NodeRAG after fixing chunks"""
    from src.deprecated.noderag.pipeline_v2 import NodeRAGPipelineV2 # Updated import
    from common.utils import get_llm_func # Updated import
    
    try:
        # Initialize components
        iris_connector = get_iris_connection()
        embedding_func = get_embedding_func()
        llm_func = get_llm_func()
        
        # Create NodeRAG pipeline
        noderag = NodeRAGPipelineV2(iris_connector, embedding_func, llm_func)
        
        # Test with a simple query
        test_query = 'What is diabetes?'
        logger.info(f'Testing NodeRAG with query: {test_query}')
        
        result = noderag.run(test_query, top_k=3)
        
        logger.info('✅ NodeRAG test successful!')
        logger.info(f'Answer length: {len(result["answer"])}')
        logger.info(f'Nodes retrieved: {result["metadata"]["num_nodes_used"]}')
        logger.info(f'Documents: {result["metadata"]["num_documents_retrieved"]}')
        logger.info(f'Chunks: {result["metadata"]["num_chunks_retrieved"]}')
        
        return True
        
    except Exception as e:
        logger.error(f'❌ NodeRAG test failed: {e}')
        return False

def main():
    """Main function"""
    logger.info("🔧 Starting NodeRAG fix...")
    
    try:
        # Populate chunks
        chunks_created = populate_document_chunks()
        
        if chunks_created > 0:
            logger.info("🧪 Testing NodeRAG...")
            success = test_noderag()
            
            if success:
                logger.info("🎉 NodeRAG is now fully functional!")
            else:
                logger.error("❌ NodeRAG test failed after chunk creation")
        else:
            logger.error("❌ No chunks were created")
            
    except Exception as e:
        logger.error(f"❌ NodeRAG fix failed: {e}")

if __name__ == "__main__":
    main()