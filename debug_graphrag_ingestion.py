#!/usr/bin/env python3
"""
Debug script to investigate GraphRAG data ingestion and table population.

This script will:
1. Check if the required GraphRAG tables exist
2. Verify table schemas
3. Check if tables are being populated during ingestion
4. Compare current implementation with archived versions
5. Test the ingestion process step by step
"""

import sys
import os
import logging
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from common.iris_connection_manager import get_iris_connection
from iris_rag.core.connection import ConnectionManager
from iris_rag.config.manager import ConfigurationManager
from iris_rag.pipelines.graphrag import GraphRAGPipeline
from iris_rag.core.models import Document

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GraphRAGInvestigator:
    """Investigates GraphRAG data ingestion issues."""
    
    def __init__(self):
        self.connection = get_iris_connection()
        if not self.connection:
            raise RuntimeError("Could not establish IRIS connection")
        
        # Initialize new-style pipeline components
        self.connection_manager = ConnectionManager()
        self.config_manager = ConfigurationManager()
        
    def check_table_existence(self):
        """Check if GraphRAG tables exist and their schemas."""
        logger.info("=== CHECKING TABLE EXISTENCE ===")
        
        cursor = self.connection.cursor()
        try:
            # Check for required GraphRAG tables
            required_tables = [
                'DocumentEntities',
                'EntityRelationships', 
                'Entities',  # From archived implementation
                'Relationships'  # From archived implementation
            ]
            
            for table in required_tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM RAG.{table}")
                    count = cursor.fetchone()[0]
                    logger.info(f"✓ RAG.{table} exists with {count} rows")
                    
                    # Get table schema
                    cursor.execute(f"""
                        SELECT COLUMN_NAME, DATA_TYPE, IS_NULLABLE 
                        FROM INFORMATION_SCHEMA.COLUMNS 
                        WHERE TABLE_SCHEMA = 'RAG' AND TABLE_NAME = '{table}'
                        ORDER BY ORDINAL_POSITION
                    """)
                    columns = cursor.fetchall()
                    logger.info(f"  Schema for RAG.{table}:")
                    for col_name, data_type, nullable in columns:
                        logger.info(f"    {col_name}: {data_type} ({'NULL' if nullable == 'YES' else 'NOT NULL'})")
                        
                except Exception as e:
                    logger.error(f"✗ RAG.{table} does not exist or is not accessible: {e}")
                    
        finally:
            cursor.close()
    
    def test_current_ingestion(self):
        """Test the current GraphRAG ingestion process."""
        logger.info("=== TESTING CURRENT GRAPHRAG INGESTION ===")
        
        try:
            # Create pipeline instance
            pipeline = GraphRAGPipeline(
                connection_manager=self.connection_manager,
                config_manager=self.config_manager
            )
            
            # Create test documents
            test_docs = [
                Document(
                    id="test_doc_1",
                    page_content="BRCA1 is a protein that helps repair DNA damage. It interacts with BRCA2 and p53 proteins in cancer pathways.",
                    metadata={"source": "test", "title": "BRCA1 Research"}
                ),
                Document(
                    id="test_doc_2", 
                    page_content="Diabetes mellitus affects insulin production. Metformin is commonly prescribed for type 2 diabetes treatment.",
                    metadata={"source": "test", "title": "Diabetes Treatment"}
                )
            ]
            
            logger.info(f"Testing ingestion with {len(test_docs)} test documents")
            
            # Test ingestion
            result = pipeline.ingest_documents(test_docs)
            logger.info(f"Ingestion result: {result}")
            
            # Check if entities and relationships were created
            self.verify_ingestion_results()
            
        except Exception as e:
            logger.error(f"Error testing current ingestion: {e}", exc_info=True)
    
    def verify_ingestion_results(self):
        """Verify that ingestion actually populated the tables."""
        logger.info("=== VERIFYING INGESTION RESULTS ===")
        
        cursor = self.connection.cursor()
        try:
            # Check DocumentEntities table
            try:
                cursor.execute("SELECT COUNT(*) FROM RAG.DocumentEntities")
                entity_count = cursor.fetchone()[0]
                logger.info(f"DocumentEntities table has {entity_count} entries")
                
                if entity_count > 0:
                    cursor.execute("SELECT TOP 5 entity_id, document_id, entity_text, entity_type FROM RAG.DocumentEntities")
                    entities = cursor.fetchall()
                    logger.info("Sample entities:")
                    for entity in entities:
                        logger.info(f"  {entity}")
                        
            except Exception as e:
                logger.error(f"Error checking DocumentEntities: {e}")
            
            # Check EntityRelationships table
            try:
                cursor.execute("SELECT COUNT(*) FROM RAG.EntityRelationships")
                rel_count = cursor.fetchone()[0]
                logger.info(f"EntityRelationships table has {rel_count} entries")
                
                if rel_count > 0:
                    cursor.execute("SELECT TOP 5 relationship_id, source_entity, target_entity, relationship_type FROM RAG.EntityRelationships")
                    relationships = cursor.fetchall()
                    logger.info("Sample relationships:")
                    for rel in relationships:
                        logger.info(f"  {rel}")
                        
            except Exception as e:
                logger.error(f"Error checking EntityRelationships: {e}")
                
        finally:
            cursor.close()
    
    def compare_with_archived_implementation(self):
        """Compare current implementation with archived versions."""
        logger.info("=== COMPARING WITH ARCHIVED IMPLEMENTATIONS ===")
        
        # Check if archived implementations use different table structures
        cursor = self.connection.cursor()
        try:
            # Check for tables used by archived implementations
            archived_tables = ['Entities', 'Relationships']
            
            for table in archived_tables:
                try:
                    cursor.execute(f"SELECT COUNT(*) FROM RAG.{table}")
                    count = cursor.fetchone()[0]
                    logger.info(f"✓ Archived table RAG.{table} exists with {count} rows")
                    
                    if count > 0:
                        cursor.execute(f"SELECT TOP 3 * FROM RAG.{table}")
                        sample_data = cursor.fetchall()
                        logger.info(f"  Sample data from RAG.{table}: {sample_data}")
                        
                except Exception as e:
                    logger.error(f"✗ Archived table RAG.{table} does not exist: {e}")
                    
        finally:
            cursor.close()
    
    def analyze_ingestion_methods(self):
        """Analyze the different entity extraction and storage methods."""
        logger.info("=== ANALYZING INGESTION METHODS ===")
        
        try:
            pipeline = GraphRAGPipeline(
                connection_manager=self.connection_manager,
                config_manager=self.config_manager
            )
            
            # Test entity extraction
            test_doc = Document(
                id="analysis_doc",
                page_content="The BRCA1 gene is associated with breast cancer. It works with BRCA2 and p53 proteins.",
                metadata={"source": "analysis"}
            )
            
            logger.info("Testing entity extraction...")
            entities = pipeline._extract_entities(test_doc)
            logger.info(f"Extracted {len(entities)} entities: {[e['entity_text'] for e in entities]}")
            
            logger.info("Testing relationship extraction...")
            relationships = pipeline._extract_relationships(test_doc, entities)
            logger.info(f"Extracted {len(relationships)} relationships")
            for rel in relationships:
                logger.info(f"  {rel['source_entity']} -> {rel['target_entity']} ({rel['relationship_type']})")
            
            # Test storage methods
            logger.info("Testing entity storage...")
            pipeline._store_entities(test_doc.id, entities)
            
            logger.info("Testing relationship storage...")
            pipeline._store_relationships(test_doc.id, relationships)
            
            logger.info("Storage test completed successfully")
            
        except Exception as e:
            logger.error(f"Error analyzing ingestion methods: {e}", exc_info=True)
    
    def create_missing_tables_if_needed(self):
        """Create missing tables if they don't exist."""
        logger.info("=== CREATING MISSING TABLES IF NEEDED ===")
        
        cursor = self.connection.cursor()
        try:
            # Check and create DocumentEntities table
            try:
                cursor.execute("SELECT COUNT(*) FROM RAG.DocumentEntities")
                logger.info("✓ DocumentEntities table exists")
            except:
                logger.info("Creating DocumentEntities table...")
                cursor.execute("""
                    CREATE TABLE RAG.DocumentEntities (
                        entity_id VARCHAR(255) PRIMARY KEY,
                        document_id VARCHAR(255) NOT NULL,
                        entity_text VARCHAR(1000) NOT NULL,
                        entity_type VARCHAR(100),
                        position INTEGER,
                        embedding VECTOR(DOUBLE, 1536),
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                logger.info("✓ Created DocumentEntities table")
            
            # Check and create EntityRelationships table
            try:
                cursor.execute("SELECT COUNT(*) FROM RAG.EntityRelationships")
                logger.info("✓ EntityRelationships table exists")
            except:
                logger.info("Creating EntityRelationships table...")
                cursor.execute("""
                    CREATE TABLE RAG.EntityRelationships (
                        relationship_id VARCHAR(255) PRIMARY KEY,
                        document_id VARCHAR(255) NOT NULL,
                        source_entity VARCHAR(255) NOT NULL,
                        target_entity VARCHAR(255) NOT NULL,
                        relationship_type VARCHAR(100),
                        strength DOUBLE DEFAULT 1.0,
                        created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                    )
                """)
                logger.info("✓ Created EntityRelationships table")
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Error creating tables: {e}")
            self.connection.rollback()
        finally:
            cursor.close()
    
    def run_full_investigation(self):
        """Run the complete investigation."""
        logger.info("🔍 STARTING GRAPHRAG INGESTION INVESTIGATION")
        
        try:
            # Step 1: Check table existence
            self.check_table_existence()
            
            # Step 2: Create missing tables if needed
            self.create_missing_tables_if_needed()
            
            # Step 3: Analyze ingestion methods
            self.analyze_ingestion_methods()
            
            # Step 4: Test current ingestion
            self.test_current_ingestion()
            
            # Step 5: Compare with archived implementation
            self.compare_with_archived_implementation()
            
            logger.info("🎉 INVESTIGATION COMPLETED")
            
        except Exception as e:
            logger.error(f"Investigation failed: {e}", exc_info=True)
        finally:
            if self.connection:
                self.connection.close()

def main():
    """Main function to run the investigation."""
    try:
        investigator = GraphRAGInvestigator()
        investigator.run_full_investigation()
    except Exception as e:
        logger.error(f"Failed to run investigation: {e}", exc_info=True)
        return False
    return True

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)