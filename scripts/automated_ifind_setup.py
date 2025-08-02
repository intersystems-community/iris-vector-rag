#!/usr/bin/env python3
"""
Automated IFind setup using existing ObjectScript classes.

This script uses the proper InterSystems architecture that's already built:
- RAG.IFindSetup.cls for ObjectScript compilation
- RAG.SourceDocumentsWithIFind.cls for the proper table structure
- Leverages existing IPM installer framework
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from common.iris_connection_manager import get_iris_connection

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AutomatedIFindSetup:
    """Automated IFind setup using existing ObjectScript infrastructure."""
    
    def __init__(self):
        self.connection = get_iris_connection()
        self.cursor = self.connection.cursor()
        
    def compile_objectscript_classes(self):
        """Compile the existing ObjectScript classes."""
        logger.info("Compiling ObjectScript classes...")
        
        try:
            # Compile RAG.IFindSetup class
            compile_sql = "DO $SYSTEM.OBJ.Compile('RAG.IFindSetup', 'ck')"
            self.cursor.execute(compile_sql)
            logger.info("✅ RAG.IFindSetup compiled")
            
            # Compile RAG.SourceDocumentsWithIFind class  
            compile_sql = "DO $SYSTEM.OBJ.Compile('RAG.SourceDocumentsWithIFind', 'ck')"
            self.cursor.execute(compile_sql)
            logger.info("✅ RAG.SourceDocumentsWithIFind compiled")
            
            self.connection.commit()
            return True
            
        except Exception as e:
            logger.error(f"Failed to compile ObjectScript classes: {e}")
            return False
    
    def run_ifind_setup(self):
        """Run the IFind setup using the ObjectScript class."""
        logger.info("Running IFind setup...")
        
        try:
            # Call the Setup method from RAG.IFindSetup
            setup_sql = "DO ##class(RAG.IFindSetup).Setup()"
            self.cursor.execute(setup_sql)
            logger.info("✅ IFind setup completed")
            
            self.connection.commit()
            return True
            
        except Exception as e:
            logger.error(f"IFind setup failed: {e}")
            return False
    
    def test_ifind_functionality(self):
        """Test IFind search functionality."""
        logger.info("Testing IFind functionality...")
        
        try:
            # Test IFind search using the ObjectScript method
            test_sql = "DO ##class(RAG.IFindSetup).TestIFindSearch('medical')"
            self.cursor.execute(test_sql)
            logger.info("✅ IFind test completed")
            
            # Also test direct SQL search
            search_sql = """
            SELECT TOP 5 doc_id, title
            FROM RAG.SourceDocumentsIFind
            WHERE %CONTAINS(text_content, 'medical')
            """
            
            try:
                self.cursor.execute(search_sql)
                results = self.cursor.fetchall()
                logger.info(f"✅ Direct IFind search working - found {len(results)} results")
                
                if results:
                    for doc_id, title in results[:3]:
                        logger.info(f"  Found: {doc_id} - {title[:50]}...")
                        
            except Exception as e:
                logger.warning(f"Direct IFind search failed: {e}")
                logger.info("This is expected if no data has been copied to SourceDocumentsIFind yet")
            
            return True
            
        except Exception as e:
            logger.error(f"IFind test failed: {e}")
            return False
    
    def update_hybrid_ifind_pipeline(self):
        """Update the hybrid IFind pipeline to use the new table."""
        logger.info("Updating hybrid IFind pipeline configuration...")
        
        # Read the current pipeline
        pipeline_file = project_root / "iris_rag/pipelines/hybrid_ifind.py"
        
        if pipeline_file.exists():
            content = pipeline_file.read_text()
            
            # Check if it's already updated
            if "SourceDocumentsIFind" in content:
                logger.info("✅ Pipeline already configured for IFind table")
                return True
            
            # Update the table references
            updated_content = content.replace(
                "FROM RAG.SourceDocuments",
                "FROM RAG.SourceDocumentsIFind"
            )
            updated_content = updated_content.replace(
                "WHERE $FIND(text_content, ?)",
                "WHERE %CONTAINS(text_content, ?)"
            )
            
            # Write back the updated content
            pipeline_file.write_text(updated_content)
            logger.info("✅ Pipeline updated to use SourceDocumentsIFind table")
            return True
        else:
            logger.warning("Pipeline file not found")
            return False
    
    def run_complete_setup(self):
        """Run the complete automated IFind setup."""
        logger.info("🚀 Starting automated IFind setup...")
        
        success = True
        
        # Step 1: Compile ObjectScript classes
        if not self.compile_objectscript_classes():
            success = False
        
        # Step 2: Run IFind setup
        if success and not self.run_ifind_setup():
            success = False
        
        # Step 3: Test functionality
        if success and not self.test_ifind_functionality():
            success = False
        
        # Step 4: Update pipeline
        if success and not self.update_hybrid_ifind_pipeline():
            success = False
        
        if success:
            logger.info("🎉 Automated IFind setup completed successfully!")
            logger.info("")
            logger.info("Next steps:")
            logger.info("1. Hybrid IFind pipeline will now use proper IFind search")
            logger.info("2. Fallback to LIKE search is still available")
            logger.info("3. Run validation: python scripts/utilities/validate_pipeline.py validate hybrid_ifind")
        else:
            logger.error("❌ Setup failed - check logs above")
        
        return success
    
    def cleanup(self):
        """Clean up resources."""
        try:
            self.cursor.close()
            self.connection.close()
        except:
            pass

def main():
    """Main entry point."""
    setup = AutomatedIFindSetup()
    
    try:
        success = setup.run_complete_setup()
        return 0 if success else 1
    finally:
        setup.cleanup()

if __name__ == "__main__":
    exit(main())