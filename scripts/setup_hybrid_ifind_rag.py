#!/usr/bin/env python3
"""
Setup script for Hybrid iFind+Graph+Vector RAG Pipeline

This script sets up the database schema, ObjectScript classes, and initial data
required for the hybrid RAG pipeline that combines iFind keyword search,
graph-based retrieval, and vector similarity search.
"""

import sys
import os
import logging
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.iris_connector import get_iris_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class HybridiFindRAGSetup:
    """Setup manager for Hybrid iFind RAG pipeline."""
    
    def __init__(self, iris_connector):
        """
        Initialize setup manager.
        
        Args:
            iris_connector: IRIS database connection
        """
        self.iris_connector = iris_connector
        # Note: ObjectScript bridge functionality will be implemented separately
        
    def create_database_schema(self) -> bool:
        """
        Create database schema for hybrid iFind RAG.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Creating database schema for Hybrid iFind RAG...")
        
        try:
            # Read schema SQL file
            schema_file = project_root / "hybrid_ifind_rag" / "schema.sql"
            
            if not schema_file.exists():
                logger.error(f"Schema file not found: {schema_file}")
                return False
            
            with open(schema_file, 'r') as f:
                schema_sql = f.read()
            
            # Split into individual statements (simple approach)
            statements = []
            current_statement = ""
            in_comment_block = False
            
            for line in schema_sql.split('\n'):
                line = line.strip()
                
                # Skip empty lines and single-line comments
                if not line or line.startswith('--'):
                    continue
                
                # Handle multi-line comments
                if '/*' in line and '*/' in line:
                    # Single line comment block
                    continue
                elif '/*' in line:
                    in_comment_block = True
                    continue
                elif '*/' in line:
                    in_comment_block = False
                    continue
                elif in_comment_block:
                    continue
                
                current_statement += line + " "
                
                # Check for statement terminator
                if line.endswith(';'):
                    statements.append(current_statement.strip())
                    current_statement = ""
            
            # Execute each statement
            success_count = 0
            for i, statement in enumerate(statements):
                if not statement or statement == ';':
                    continue
                
                try:
                    logger.debug(f"Executing statement {i+1}: {statement[:100]}...")
                    cursor = self.iris_connector.execute_query(statement)
                    success_count += 1
                    logger.debug(f"Statement {i+1} executed successfully")
                    
                except Exception as e:
                    # Some statements might fail if objects already exist
                    if "already exists" in str(e).lower() or "duplicate" in str(e).lower():
                        logger.warning(f"Statement {i+1} skipped (object exists): {e}")
                        success_count += 1
                    else:
                        logger.error(f"Error executing statement {i+1}: {e}")
                        logger.error(f"Statement: {statement}")
            
            logger.info(f"Schema creation completed: {success_count}/{len(statements)} statements executed")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error creating database schema: {e}")
            return False
    
    def deploy_objectscript_classes(self) -> bool:
        """
        Deploy ObjectScript classes for iFind integration.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Deploying ObjectScript classes...")
        
        try:
            # List of ObjectScript class files to deploy
            class_files = [
                "objectscript/RAGDemo.KeywordFinder.cls",
                "objectscript/RAGDemo.KeywordProcessor.cls"
            ]
            
            # For now, just verify the files exist
            # TODO: Implement actual ObjectScript compilation when bridge is available
            success_count = 0
            for class_file in class_files:
                class_path = project_root / class_file
                
                if class_path.exists():
                    logger.info(f"ObjectScript class file found: {class_file}")
                    success_count += 1
                else:
                    logger.warning(f"Class file not found: {class_path}")
            
            logger.info(f"ObjectScript class verification completed: {success_count}/{len(class_files)} classes found")
            logger.info("Note: Actual ObjectScript compilation requires manual deployment to IRIS")
            return success_count > 0
            
        except Exception as e:
            logger.error(f"Error verifying ObjectScript classes: {e}")
            return False
    
    def initialize_configuration(self) -> bool:
        """
        Initialize hybrid search configuration.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Initializing hybrid search configuration...")
        
        try:
            # Check if default configuration exists
            check_query = "SELECT COUNT(*) FROM hybrid_search_config WHERE id = 1"
            cursor = self.iris_connector.execute_query(check_query)
            count = cursor.fetchone()[0]
            
            if count > 0:
                logger.info("Default configuration already exists")
                return True
            
            # Insert default configuration
            insert_query = """
            INSERT INTO hybrid_search_config 
            (id, config_name, ifind_weight, graph_weight, vector_weight, rrf_k, max_results_per_method, final_results_limit)
            VALUES (1, 'default', 0.33, 0.33, 0.34, 60, 20, 10)
            """
            
            cursor = self.iris_connector.execute_query(insert_query)
            logger.info("Default configuration initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing configuration: {e}")
            return False
    
    def create_sample_keyword_index(self) -> bool:
        """
        Create sample keyword index for testing.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Creating sample keyword index...")
        
        try:
            # Check if we have documents to index
            check_docs_query = "SELECT COUNT(*) FROM documents WHERE content IS NOT NULL"
            cursor = self.iris_connector.execute_query(check_docs_query)
            doc_count = cursor.fetchone()[0]
            
            if doc_count == 0:
                logger.warning("No documents found to index")
                return True
            
            # Get a sample of documents to index
            sample_query = """
            SELECT TOP 100 id, content 
            FROM documents 
            WHERE content IS NOT NULL 
            ORDER BY id
            """
            
            cursor = self.iris_connector.execute_query(sample_query)
            documents = cursor.fetchall()
            
            logger.info(f"Indexing keywords for {len(documents)} sample documents...")
            
            # Simple keyword extraction and indexing
            indexed_count = 0
            for doc_id, content in documents:
                try:
                    # Extract keywords (simple approach)
                    keywords = self._extract_keywords(content)
                    
                    # Index keywords for this document
                    for keyword, frequency in keywords.items():
                        insert_query = """
                        INSERT INTO keyword_index (document_id, keyword, frequency)
                        VALUES (?, ?, ?)
                        """
                        
                        try:
                            self.iris_connector.execute_query(insert_query, [doc_id, keyword, frequency])
                        except Exception as e:
                            # Skip if keyword already exists for this document
                            if "duplicate" not in str(e).lower():
                                logger.debug(f"Error indexing keyword '{keyword}' for doc {doc_id}: {e}")
                    
                    indexed_count += 1
                    
                    if indexed_count % 10 == 0:
                        logger.info(f"Indexed {indexed_count}/{len(documents)} documents...")
                        
                except Exception as e:
                    logger.warning(f"Error indexing document {doc_id}: {e}")
            
            logger.info(f"Keyword indexing completed for {indexed_count} documents")
            
            # Create sample bitmap chunks
            self._create_sample_bitmap_chunks()
            
            return True
            
        except Exception as e:
            logger.error(f"Error creating sample keyword index: {e}")
            return False
    
    def _extract_keywords(self, content: str) -> dict:
        """
        Simple keyword extraction for testing.
        
        Args:
            content: Document content
            
        Returns:
            Dictionary of keywords and their frequencies
        """
        import re
        from collections import Counter
        
        # Simple tokenization
        words = re.findall(r'\b[a-zA-Z]{3,}\b', content.lower())
        
        # Filter stop words
        stop_words = {
            'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to',
            'for', 'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be',
            'been', 'have', 'has', 'had', 'do', 'does', 'did', 'will',
            'would', 'could', 'should', 'may', 'might', 'can', 'this',
            'that', 'these', 'those', 'they', 'them', 'their', 'there'
        }
        
        # Count word frequencies
        word_counts = Counter(word for word in words if word not in stop_words)
        
        # Return top 20 keywords
        return dict(word_counts.most_common(20))
    
    def _create_sample_bitmap_chunks(self) -> bool:
        """
        Create sample bitmap chunks for testing.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Creating sample bitmap chunks...")
        
        try:
            # Get unique keywords
            keywords_query = "SELECT DISTINCT keyword FROM keyword_index ORDER BY keyword"
            cursor = self.iris_connector.execute_query(keywords_query)
            keywords = [row[0] for row in cursor.fetchall()]
            
            chunk_count = 0
            for keyword in keywords[:50]:  # Limit to first 50 keywords for testing
                try:
                    # Get documents for this keyword
                    docs_query = """
                    SELECT document_id, frequency 
                    FROM keyword_index 
                    WHERE keyword = ? 
                    ORDER BY document_id
                    """
                    
                    cursor = self.iris_connector.execute_query(docs_query, [keyword])
                    docs = cursor.fetchall()
                    
                    if not docs:
                        continue
                    
                    # Create bitmap data (simple format: doc_id:frequency,...)
                    bitmap_data = ",".join(f"{doc_id}:{freq}" for doc_id, freq in docs)
                    
                    # Insert bitmap chunk
                    insert_query = """
                    INSERT INTO keyword_bitmap_chunks (keyword, chunk_number, bitmap_data, document_count)
                    VALUES (?, 1, ?, ?)
                    """
                    
                    self.iris_connector.execute_query(insert_query, [keyword, bitmap_data, len(docs)])
                    chunk_count += 1
                    
                except Exception as e:
                    logger.debug(f"Error creating bitmap chunk for keyword '{keyword}': {e}")
            
            logger.info(f"Created {chunk_count} bitmap chunks")
            return True
            
        except Exception as e:
            logger.error(f"Error creating bitmap chunks: {e}")
            return False
    
    def verify_setup(self) -> bool:
        """
        Verify that the setup was successful.
        
        Returns:
            True if verification passes, False otherwise
        """
        logger.info("Verifying hybrid iFind RAG setup...")
        
        try:
            # Check tables exist
            tables_to_check = [
                'keyword_index',
                'keyword_bitmap_chunks', 
                'hybrid_search_config'
            ]
            
            for table in tables_to_check:
                try:
                    query = f"SELECT COUNT(*) FROM {table}"
                    cursor = self.iris_connector.execute_query(query)
                    count = cursor.fetchone()[0]
                    logger.info(f"Table {table}: {count} rows")
                except Exception as e:
                    logger.error(f"Table {table} not accessible: {e}")
                    return False
            
            # Check configuration
            config_query = "SELECT config_name, ifind_weight, graph_weight, vector_weight FROM hybrid_search_config WHERE id = 1"
            cursor = self.iris_connector.execute_query(config_query)
            config = cursor.fetchone()
            
            if config:
                logger.info(f"Configuration '{config[0]}': iFind={config[1]}, Graph={config[2]}, Vector={config[3]}")
            else:
                logger.error("Default configuration not found")
                return False
            
            # Check ObjectScript classes (if possible)
            try:
                # This would require ObjectScript execution capability
                logger.info("ObjectScript classes deployment verification skipped (requires ObjectScript execution)")
            except Exception as e:
                logger.warning(f"Could not verify ObjectScript classes: {e}")
            
            logger.info("Setup verification completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error during setup verification: {e}")
            return False
    
    def run_complete_setup(self) -> bool:
        """
        Run the complete setup process.
        
        Returns:
            True if successful, False otherwise
        """
        logger.info("Starting Hybrid iFind RAG setup process...")
        start_time = time.time()
        
        steps = [
            ("Creating database schema", self.create_database_schema),
            ("Deploying ObjectScript classes", self.deploy_objectscript_classes),
            ("Initializing configuration", self.initialize_configuration),
            ("Creating sample keyword index", self.create_sample_keyword_index),
            ("Verifying setup", self.verify_setup)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"Step: {step_name}")
            
            try:
                if not step_func():
                    logger.error(f"Setup failed at step: {step_name}")
                    return False
                    
                logger.info(f"Step completed: {step_name}")
                
            except Exception as e:
                logger.error(f"Error in step '{step_name}': {e}")
                return False
        
        total_time = time.time() - start_time
        logger.info(f"Hybrid iFind RAG setup completed successfully in {total_time:.2f} seconds")
        return True


def main():
    """Main setup function."""
    logger.info("Hybrid iFind+Graph+Vector RAG Setup")
    logger.info("=" * 50)
    
    try:
        # Create IRIS connection
        logger.info("Connecting to IRIS database...")
        iris_connector = get_iris_connection(use_mock=True)  # Use mock for testing
        
        # Create setup manager
        setup_manager = HybridiFindRAGSetup(iris_connector)
        
        # Run setup
        success = setup_manager.run_complete_setup()
        
        if success:
            logger.info("Setup completed successfully!")
            logger.info("\nNext steps:")
            logger.info("1. Test the hybrid pipeline with: python -m pytest tests/test_hybrid_ifind_rag.py")
            logger.info("2. Run integration tests with real data")
            logger.info("3. Configure weights in hybrid_search_config table as needed")
            return 0
        else:
            logger.error("Setup failed!")
            return 1
            
    except Exception as e:
        logger.error(f"Setup error: {e}")
        return 1
    
    finally:
        # Close connection if it exists
        try:
            if 'iris_connector' in locals():
                iris_connector.close()
        except:
            pass


if __name__ == "__main__":
    sys.exit(main())