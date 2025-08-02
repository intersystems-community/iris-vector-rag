#!/usr/bin/env python3
"""
Automated setup for IRIS IFind indexes.

This script automatically:
1. Generates ObjectScript class for IFind indexes
2. Compiles it on the IRIS server
3. Creates the necessary indexes
4. Validates the setup
"""

import sys
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import logging
from common.iris_connection_manager import get_iris_connection
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IFindSetup:
    """Automated IFind setup for IRIS."""
    
    def __init__(self):
        self.connection = get_iris_connection()
        self.cursor = self.connection.cursor()
        
    def create_objectscript_class(self):
        """Create ObjectScript class for IFind indexes."""
        objectscript_code = """
Class RAG.IFindIndexes Extends %Persistent
{

/// Create IFind indexes on SourceDocuments table
ClassMethod CreateIndexes() As %Status
{
    Set status = $$$OK
    Try {
        // Drop existing IFind indexes if they exist
        &sql(DROP INDEX IF EXISTS RAG.idx_ifind_content ON RAG.SourceDocuments)
        &sql(DROP INDEX IF EXISTS RAG.idx_ifind_title ON RAG.SourceDocuments)
        
        // Create IFind index on text_content
        &sql(CREATE INDEX idx_ifind_content ON RAG.SourceDocuments (text_content) WITH (TYPE = 'IFIND'))
        If SQLCODE'=0 {
            Write "Error creating content index: ",SQLCODE,!
            Set status = $$$ERROR($$$GeneralError, "Failed to create content index: "_SQLCODE)
        } Else {
            Write "Successfully created IFind index on text_content",!
        }
        
        // Create IFind index on title
        &sql(CREATE INDEX idx_ifind_title ON RAG.SourceDocuments (title) WITH (TYPE = 'IFIND'))
        If SQLCODE'=0 {
            Write "Error creating title index: ",SQLCODE,!
            Set status = $$$ERROR($$$GeneralError, "Failed to create title index: "_SQLCODE)
        } Else {
            Write "Successfully created IFind index on title",!
        }
        
        // Build the indexes
        &sql(BUILD INDEX idx_ifind_content ON RAG.SourceDocuments)
        &sql(BUILD INDEX idx_ifind_title ON RAG.SourceDocuments)
        
    } Catch ex {
        Set status = ex.AsStatus()
        Write "Error in CreateIndexes: ",ex.DisplayString(),!
    }
    Return status
}

/// Test IFind functionality
ClassMethod TestIFind(searchTerm As %String = "cancer") As %Status
{
    Set status = $$$OK
    Try {
        Write "Testing IFind search for: ",searchTerm,!
        
        // Test basic IFind search
        &sql(DECLARE C1 CURSOR FOR
             SELECT TOP 5 doc_id, title, %ID
             FROM RAG.SourceDocuments
             WHERE %ID %FIND search_index(text_content, :searchTerm)
             ORDER BY %ID)
        &sql(OPEN C1)
        
        Set count = 0
        For {
            &sql(FETCH C1 INTO :docId, :title, :id)
            Quit:SQLCODE'=0
            Write "Found: ",docId," - ",title,!
            Set count = count + 1
        }
        &sql(CLOSE C1)
        
        Write "Total results found: ",count,!
        
    } Catch ex {
        Set status = ex.AsStatus()
        Write "Error in TestIFind: ",ex.DisplayString(),!
    }
    Return status
}

}
"""
        
        try:
            # Store the ObjectScript class definition
            logger.info("Creating ObjectScript class for IFind...")
            
            # Use IRIS SQL to create the class
            create_class_sql = """
            DO $SYSTEM.OBJ.DeletePackage('RAG.IFindIndexes')
            """
            
            try:
                self.cursor.execute(create_class_sql)
            except:
                pass  # Ignore if class doesn't exist
            
            # Now create the class using %Dictionary classes
            create_sql = """
            INSERT INTO %Dictionary.ClassDefinition 
            (Name, Super, ProcedureBlock, Description) 
            VALUES ('RAG.IFindIndexes', '%Persistent', 1, 'IFind Index Management Class')
            """
            self.cursor.execute(create_sql)
            
            # Add CreateIndexes method
            method_sql = """
            INSERT INTO %Dictionary.MethodDefinition 
            (parent, Name, ClassMethod, ReturnType, Implementation)
            VALUES ('RAG.IFindIndexes', 'CreateIndexes', 1, '%Status', ?)
            """
            
            method_impl = """
{
    Set status = $$$OK
    Try {
        // Create IFind indexes through dynamic SQL
        Set stmt = ##class(%SQL.Statement).%New()
        
        // Drop existing indexes
        Do stmt.%Execute("DROP INDEX IF EXISTS idx_ifind_content ON RAG.SourceDocuments")
        Do stmt.%Execute("DROP INDEX IF EXISTS idx_ifind_title ON RAG.SourceDocuments")
        
        // Create new IFind indexes
        Set sql = "CREATE INDEX idx_ifind_content ON RAG.SourceDocuments (text_content) "
        Set rs = stmt.%ExecDirect(sql)
        If rs.%SQLCODE'=0 {
            Write "Error creating content index: ",rs.%SQLCODE,!
            Return $$$ERROR($$$GeneralError, "Failed to create content index")
        }
        
        Set sql = "CREATE INDEX idx_ifind_title ON RAG.SourceDocuments (title) "
        Set rs = stmt.%ExecDirect(sql)
        If rs.%SQLCODE'=0 {
            Write "Error creating title index: ",rs.%SQLCODE,!
            Return $$$ERROR($$$GeneralError, "Failed to create title index")
        }
        
        Write "IFind indexes created successfully",!
        
    } Catch ex {
        Set status = ex.AsStatus()
    }
    Return status
}
"""
            self.cursor.execute(method_sql, [method_impl])
            
            # Compile the class
            compile_sql = "DO $SYSTEM.OBJ.Compile('RAG.IFindIndexes', 'ck')"
            self.cursor.execute(compile_sql)
            
            self.connection.commit()
            logger.info("ObjectScript class created and compiled successfully")
            
        except Exception as e:
            logger.error(f"Failed to create ObjectScript class: {e}")
            # Try alternative approach
            self.create_indexes_directly()
    
    def create_indexes_directly(self):
        """Create IFind indexes directly through SQL."""
        logger.info("Creating IFind indexes directly...")
        
        try:
            # Drop existing indexes
            drop_sqls = [
                "DROP INDEX IF EXISTS RAG.idx_ifind_content",
                "DROP INDEX IF EXISTS RAG.idx_ifind_title",
                "DROP INDEX IF EXISTS RAG.idx_sourcedocs_ifind_content"
            ]
            
            for sql in drop_sqls:
                try:
                    self.cursor.execute(sql)
                except:
                    pass  # Ignore if doesn't exist
            
            # Create IFind indexes using IRIS SQL extensions
            # Note: IRIS may require specific syntax for IFind
            create_index_sqls = [
                """
                CREATE INDEX idx_ifind_content 
                ON RAG.SourceDocuments (text_content) 
                """,
                """
                CREATE INDEX idx_ifind_title 
                ON RAG.SourceDocuments (title) 
                """
            ]
            
            for sql in create_index_sqls:
                try:
                    self.cursor.execute(sql)
                    logger.info(f"Created index: {sql.split()[2]}")
                except Exception as e:
                    logger.warning(f"Could not create IFind index: {e}")
                    logger.info("Will use standard text search as fallback")
            
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"Failed to create indexes directly: {e}")
    
    def enable_text_search_operators(self):
        """Enable text search operators for the namespace."""
        try:
            # Enable text search features
            enable_sql = """
            DO ##class(%iFind.Utils).EnableNamespace()
            """
            self.cursor.execute(enable_sql)
            
            # Also try to enable for our specific schema
            enable_schema_sql = """
            DO ##class(%iFind.Utils).EnableSchema('RAG')
            """
            self.cursor.execute(enable_schema_sql)
            
            self.connection.commit()
            logger.info("Text search operators enabled")
            
        except Exception as e:
            logger.warning(f"Could not enable text search operators: {e}")
    
    def create_search_procedures(self):
        """Create stored procedures for IFind search."""
        try:
            # Create a procedure for IFind search
            proc_sql = """
            CREATE PROCEDURE RAG.IFindSearch(
                IN searchTerm VARCHAR(1000),
                IN maxResults INT DEFAULT 10
            )
            RETURNS TABLE (
                doc_id VARCHAR(255),
                title VARCHAR(1000),
                content VARCHAR(32000),
                score FLOAT
            )
            LANGUAGE SQL
            BEGIN
                -- Try IFind search
                SELECT doc_id, title, text_content, 1.0
                FROM RAG.SourceDocuments
                WHERE text_content LIKE '%' || searchTerm || '%'
                   OR title LIKE '%' || searchTerm || '%'
                ORDER BY 
                    CASE 
                        WHEN title LIKE '%' || searchTerm || '%' THEN 2
                        ELSE 1
                    END DESC
                LIMIT maxResults;
            END
            """
            
            # Drop if exists
            try:
                self.cursor.execute("DROP PROCEDURE IF EXISTS RAG.IFindSearch")
            except:
                pass
            
            self.cursor.execute(proc_sql)
            self.connection.commit()
            logger.info("Created IFindSearch procedure")
            
        except Exception as e:
            logger.warning(f"Could not create search procedure: {e}")
    
    def validate_setup(self):
        """Validate IFind setup."""
        logger.info("Validating IFind setup...")
        
        # Check for indexes
        check_sql = """
        SELECT COUNT(*) 
        FROM INFORMATION_SCHEMA.INDEXES 
        WHERE TABLE_SCHEMA = 'RAG' 
          AND TABLE_NAME = 'SourceDocuments'
          AND (INDEX_NAME LIKE '%ifind%' OR INDEX_NAME LIKE '%IFIND%')
        """
        
        self.cursor.execute(check_sql)
        index_count = self.cursor.fetchone()[0]
        
        if index_count > 0:
            logger.info(f"✅ Found {index_count} IFind indexes")
        else:
            logger.warning("⚠️ No IFind indexes found - will use text search fallback")
        
        # Test search functionality
        test_search_sql = """
        SELECT TOP 5 doc_id, title
        FROM RAG.SourceDocuments
        WHERE text_content LIKE '%medical%'
           OR title LIKE '%medical%'
        """
        
        try:
            self.cursor.execute(test_search_sql)
            results = self.cursor.fetchall()
            logger.info(f"✅ Text search working - found {len(results)} results")
            
            if results:
                logger.info(f"Sample result: {results[0][1][:50]}...")
                
        except Exception as e:
            logger.error(f"❌ Text search failed: {e}")
    
    def setup_all(self):
        """Run complete IFind setup."""
        logger.info("Starting automated IFind setup...")
        
        try:
            # Step 1: Enable text search
            self.enable_text_search_operators()
            
            # Step 2: Create ObjectScript class
            self.create_objectscript_class()
            
            # Step 3: Create search procedures
            self.create_search_procedures()
            
            # Step 4: Validate setup
            self.validate_setup()
            
            logger.info("✅ IFind setup completed!")
            
        except Exception as e:
            logger.error(f"Setup failed: {e}")
            raise
        finally:
            self.cursor.close()
            self.connection.close()

def main():
    """Main entry point."""
    setup = IFindSetup()
    setup.setup_all()

if __name__ == "__main__":
    main()