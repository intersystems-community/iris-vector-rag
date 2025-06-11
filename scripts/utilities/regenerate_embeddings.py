#!/usr/bin/env python3
"""
Script to regenerate embeddings for documents using text_content instead of abstract.

This script:
1. Sets all existing embeddings to NULL
2. Uses the SetupOrchestrator to regenerate embeddings with the improved logic
"""

import sys
import os
import logging

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from common.iris_connection_manager import get_iris_connection
from iris_rag.validation.orchestrator import SetupOrchestrator
from iris_rag.config.manager import ConfigurationManager
from iris_rag.core.connection import ConnectionManager

def clear_existing_embeddings():
    """Clear all existing embeddings to force regeneration."""
    connection = get_iris_connection()
    cursor = connection.cursor()
    
    try:
        # Set all embeddings to NULL
        cursor.execute("UPDATE RAG.SourceDocuments SET embedding = NULL")
        affected_rows = cursor.rowcount
        connection.commit()
        
        print(f"Cleared embeddings for {affected_rows} documents")
        return affected_rows
        
    except Exception as e:
        print(f"Error clearing embeddings: {e}")
        connection.rollback()
        return 0
    finally:
        cursor.close()
        connection.close()

def regenerate_embeddings():
    """Use SetupOrchestrator to regenerate embeddings."""
    try:
        # Initialize configuration and connection managers
        config_manager = ConfigurationManager()
        connection_manager = ConnectionManager()
        orchestrator = SetupOrchestrator(connection_manager, config_manager)
        
        # Generate missing embeddings (which should now be all of them)
        orchestrator._generate_missing_document_embeddings()
        
        print("Embedding regeneration completed")
        return True
        
    except Exception as e:
        print(f"Error regenerating embeddings: {e}")
        return False

def verify_embeddings():
    """Verify that embeddings were successfully regenerated."""
    connection = get_iris_connection()
    cursor = connection.cursor()
    
    try:
        # Count documents with NULL embeddings
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NULL")
        null_count = cursor.fetchone()[0]
        
        # Count total documents
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
        total_count = cursor.fetchone()[0]
        
        print(f"Verification: {total_count - null_count}/{total_count} documents have embeddings")
        print(f"Documents still missing embeddings: {null_count}")
        
        return null_count == 0
        
    except Exception as e:
        print(f"Error verifying embeddings: {e}")
        return False
    finally:
        cursor.close()
        connection.close()

if __name__ == "__main__":
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    
    print("=== Regenerating Document Embeddings ===")
    print("This will clear existing embeddings and regenerate them using text_content")
    
    # Step 1: Clear existing embeddings
    print("\n1. Clearing existing embeddings...")
    cleared_count = clear_existing_embeddings()
    if cleared_count == 0:
        print("Failed to clear embeddings. Exiting.")
        sys.exit(1)
    
    # Step 2: Regenerate embeddings
    print("\n2. Regenerating embeddings...")
    if not regenerate_embeddings():
        print("Failed to regenerate embeddings. Exiting.")
        sys.exit(1)
    
    # Step 3: Verify results
    print("\n3. Verifying results...")
    if verify_embeddings():
        print("\n✅ Embedding regeneration successful!")
    else:
        print("\n❌ Some embeddings are still missing.")
        sys.exit(1)