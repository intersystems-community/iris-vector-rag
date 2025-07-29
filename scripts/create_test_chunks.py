#!/usr/bin/env python3
"""
Create test chunks for CRAG pipeline testing.
Since documents are very short, create synthetic chunks.
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from common.iris_connector import get_iris_connection
from common.utils import get_embedding_func

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def create_test_chunks():
    """Create test document chunks for CRAG."""
    connection = get_iris_connection()
    cursor = connection.cursor()
    
    try:
        # Get embedding function
        embedding_func = get_embedding_func()
        
        # Clear existing chunks
        logger.info("Clearing existing document chunks...")
        cursor.execute("DELETE FROM RAG.DocumentChunks")
        
        # Get some document IDs
        cursor.execute("SELECT TOP 10 doc_id FROM RAG.SourceDocuments")
        doc_ids = [row[0] for row in cursor.fetchall()]
        
        # Create test chunks with medical content
        test_chunks = [
            "Heart disease is caused by several factors including high cholesterol, high blood pressure, and smoking.",
            "Diabetes symptoms include increased thirst, frequent urination, and unexplained weight loss.",
            "Cancer treatments include chemotherapy, radiation therapy, and surgical procedures.",
            "Vaccines work by stimulating the immune system to recognize and fight specific pathogens.",
            "Insulin regulates blood sugar levels by facilitating glucose uptake into cells.",
            "Cardiovascular disease prevention involves regular exercise, healthy diet, and stress management.",
            "Type 2 diabetes management includes medication, dietary changes, and blood glucose monitoring.",
            "Oncological treatments are personalized based on cancer type, stage, and patient factors.",
            "Immunization programs have significantly reduced infectious disease mortality worldwide.",
            "Metabolic disorders often require long-term management and lifestyle modifications."
        ]
        
        chunk_count = 0
        for i, doc_id in enumerate(doc_ids):
            for j, chunk_text in enumerate(test_chunks):
                # Generate embedding for chunk
                try:
                    chunk_embedding = embedding_func([chunk_text])[0]
                    embedding_str = ','.join(f'{x:.10f}' for x in chunk_embedding)
                except Exception as e:
                    logger.warning(f"Failed to generate embedding for chunk: {e}")
                    continue
                
                # Insert chunk
                chunk_id = f"{doc_id}_chunk_{j}"
                try:
                    cursor.execute("""
                        INSERT INTO RAG.DocumentChunks 
                        (chunk_id, doc_id, chunk_text, chunk_embedding, chunk_type, chunk_index, metadata)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (
                        chunk_id,
                        doc_id, 
                        chunk_text,
                        embedding_str,
                        'content',
                        j,
                        '{}'
                    ))
                    chunk_count += 1
                except Exception as e:
                    logger.error(f"Failed to insert chunk {chunk_id}: {e}")
        
        connection.commit()
        logger.info(f"✅ Successfully created {chunk_count} test document chunks!")
        
        # Verify
        cursor.execute("SELECT COUNT(*) FROM RAG.DocumentChunks")
        total_chunks = cursor.fetchone()[0]
        logger.info(f"📊 Total chunks in database: {total_chunks}")
        
    except Exception as e:
        logger.error(f"❌ Error creating test chunks: {e}")
        connection.rollback()
        raise
    finally:
        cursor.close()
        connection.close()

if __name__ == "__main__":
    create_test_chunks()