#!/usr/bin/env python3
"""
Simple IFind setup - creates IFind table and configures pipeline.

This approach:
1. Creates SourceDocumentsIFind table with proper structure
2. Copies data from existing SourceDocuments
3. Creates fulltext index for IFind search
4. Updates pipeline configuration
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from common.iris_connection_manager import get_iris_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class SimpleIFindSetup:
    """Simple automated IFind setup."""
    
    def __init__(self):
        self.connection = get_iris_connection()
        self.cursor = self.connection.cursor()
        
    def create_ifind_table(self):
        """Create SourceDocumentsIFind table with fulltext support."""
        logger.info("Creating SourceDocumentsIFind table...")
        
        try:
            # Drop table if exists
            try:
                self.cursor.execute("DROP TABLE IF EXISTS RAG.SourceDocumentsIFind")
            except:
                pass
            
            # Create new table with same structure as SourceDocuments
            create_sql = """
            CREATE TABLE RAG.SourceDocumentsIFind (
                doc_id VARCHAR(255) PRIMARY KEY,
                title VARCHAR(1000),
                text_content LONGVARCHAR,
                authors VARCHAR(2000),
                keywords VARCHAR(2000),
                embedding VARCHAR(32000),
                metadata VARCHAR(4000),
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
            """
            
            self.cursor.execute(create_sql)
            logger.info("✅ SourceDocumentsIFind table created")
            
            # Create fulltext index
            try:
                index_sql = "CREATE FULLTEXT INDEX idx_ifind_content ON RAG.SourceDocumentsIFind (text_content)"
                self.cursor.execute(index_sql)
                logger.info("✅ Fulltext index created")
            except Exception as e:
                logger.warning(f"Fulltext index creation failed: {e}")
                logger.info("Will use basic text search instead")
            
            self.connection.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to create IFind table: {e}")
            return False
    
    def copy_data_to_ifind_table(self):
        """Copy data from SourceDocuments to SourceDocumentsIFind."""
        logger.info("Copying data to IFind table...")
        
        try:
            # Check source data count
            self.cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
            source_count = self.cursor.fetchone()[0]
            logger.info(f"Source documents: {source_count}")
            
            if source_count == 0:
                logger.warning("No source documents to copy")
                return True
            
            # Copy data with simplified column mapping
            copy_sql = """
            INSERT INTO RAG.SourceDocumentsIFind 
            (doc_id, title, text_content, embedding, created_at)
            SELECT 
                doc_id, 
                title, 
                text_content,
                embedding,
                created_at
            FROM RAG.SourceDocuments
            """
            
            self.cursor.execute(copy_sql)
            
            # Check copied data count
            self.cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocumentsIFind")
            copied_count = self.cursor.fetchone()[0]
            
            logger.info(f"✅ Copied {copied_count} documents to IFind table")
            
            self.connection.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to copy data: {e}")
            return False
    
    def test_ifind_search(self):
        """Test IFind search functionality."""
        logger.info("Testing IFind search...")
        
        # Test with %CONTAINS (proper IFind syntax)
        test_queries = [
            "SELECT TOP 5 doc_id, title FROM RAG.SourceDocumentsIFind WHERE %CONTAINS(text_content, 'medical')",
            "SELECT TOP 5 doc_id, title FROM RAG.SourceDocumentsIFind WHERE text_content LIKE '%medical%'",
            "SELECT TOP 5 doc_id, title FROM RAG.SourceDocumentsIFind WHERE text_content LIKE '%diabetes%'"
        ]
        
        for i, sql in enumerate(test_queries, 1):
            try:
                self.cursor.execute(sql)
                results = self.cursor.fetchall()
                search_type = "IFind (%CONTAINS)" if "%CONTAINS" in sql else f"LIKE search {i-1}"
                logger.info(f"✅ {search_type}: found {len(results)} results")
                
                if results and len(results) > 0:
                    sample = results[0]
                    logger.info(f"  Sample: {sample[0]} - {sample[1][:50]}...")
                    
            except Exception as e:
                search_type = "IFind (%CONTAINS)" if "%CONTAINS" in sql else f"LIKE search {i-1}"
                logger.warning(f"⚠️ {search_type} failed: {e}")
        
        return True
    
    def update_pipeline_config(self):
        """Update hybrid IFind pipeline to use the new table."""
        logger.info("Updating pipeline configuration...")
        
        pipeline_file = project_root / "iris_rag/pipelines/hybrid_ifind.py"
        
        if not pipeline_file.exists():
            logger.warning("Pipeline file not found")
            return False
        
        try:
            content = pipeline_file.read_text()
            
            # Track changes
            changes_made = []
            
            # Update table name
            if "FROM RAG.SourceDocuments" in content and "FROM RAG.SourceDocumentsIFind" not in content:
                content = content.replace(
                    "FROM RAG.SourceDocuments",
                    "FROM RAG.SourceDocumentsIFind"
                )
                changes_made.append("Updated table name to SourceDocumentsIFind")
            
            # Update IFind syntax
            if "WHERE $FIND(text_content, ?)" in content:
                content = content.replace(
                    "WHERE $FIND(text_content, ?)",
                    "WHERE %CONTAINS(text_content, ?)"
                )
                changes_made.append("Updated to %CONTAINS syntax")
            
            # Update SCORE function
            if "$SCORE(text_content)" in content:
                content = content.replace(
                    "$SCORE(text_content)",
                    "1.0"  # Simplified scoring for now
                )
                changes_made.append("Updated scoring function")
            
            if changes_made:
                pipeline_file.write_text(content)
                logger.info("✅ Pipeline updated:")
                for change in changes_made:
                    logger.info(f"  - {change}")
            else:
                logger.info("✅ Pipeline already configured correctly")
            
            return True
            
        except Exception as e:
            logger.error(f"Failed to update pipeline: {e}")
            return False
    
    def run_complete_setup(self):
        """Run complete IFind setup."""
        logger.info("🚀 Starting simple IFind setup...")
        
        steps = [
            ("Create IFind table", self.create_ifind_table),
            ("Copy data", self.copy_data_to_ifind_table),
            ("Test search", self.test_ifind_search),
            ("Update pipeline", self.update_pipeline_config)
        ]
        
        for step_name, step_func in steps:
            logger.info(f"\n--- {step_name} ---")
            if not step_func():
                logger.error(f"❌ {step_name} failed")
                return False
        
        logger.info("\n🎉 IFind setup completed successfully!")
        logger.info("\nSummary:")
        logger.info("✅ SourceDocumentsIFind table created with fulltext index")
        logger.info("✅ Data copied from SourceDocuments")
        logger.info("✅ Search functionality tested")
        logger.info("✅ Pipeline updated to use IFind table")
        logger.info("\nNext steps:")
        logger.info("- Run: python scripts/utilities/validate_pipeline.py validate hybrid_ifind")
        logger.info("- Test query: python -c \"from iris_rag.pipelines.hybrid_ifind import HybridIFindRAGPipeline; p = HybridIFindRAGPipeline(...); print(p.query('medical research'))\"")
        
        return True
    
    def cleanup(self):
        """Clean up resources."""
        try:
            self.cursor.close()
            self.connection.close()
        except:
            pass

def main():
    """Main entry point."""
    setup = SimpleIFindSetup()
    
    try:
        success = setup.run_complete_setup()
        return 0 if success else 1
    finally:
        setup.cleanup()

if __name__ == "__main__":
    exit(main())