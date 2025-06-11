#!/usr/bin/env python3
"""
Comprehensive debugging script for schema discrepancy and missing chunk embeddings.

This script investigates:
1. Schema discrepancy: Why INFORMATION_SCHEMA reports VECTOR columns as varchar
2. Missing chunk embeddings: Why RAG.DocumentChunks has no chunk_embedding values
"""

import os
import sys
import logging
from typing import Dict, Any, List, Optional

# Add project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.iris_connector import get_iris_connection, IRISConnectionError

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

class SchemaAndEmbeddingDebugger:
    """Comprehensive debugger for schema and embedding issues."""
    
    def __init__(self):
        self.connection = None
        self.results = {}
        
    def connect(self):
        """Establish database connection."""
        try:
            self.connection = get_iris_connection()
            logger.info("✅ Successfully connected to IRIS database")
            return True
        except IRISConnectionError as e:
            logger.error(f"❌ Failed to connect to IRIS: {e}")
            return False
    
    def debug_schema_discrepancy(self):
        """Part 1: Investigate schema discrepancy for VECTOR columns."""
        logger.info("\n" + "="*60)
        logger.info("PART 1: SCHEMA DISCREPANCY INVESTIGATION")
        logger.info("="*60)
        
        cursor = self.connection.cursor()
        
        # 1.1: Check INFORMATION_SCHEMA.COLUMNS for both tables
        logger.info("\n--- 1.1: INFORMATION_SCHEMA.COLUMNS Analysis ---")
        
        tables_to_check = [
            ("DocumentChunks", "chunk_embedding"),
            ("DocumentTokenEmbeddings", "token_embedding")
        ]
        
        for table_name, column_name in tables_to_check:
            logger.info(f"\nChecking {table_name}.{column_name}:")
            
            cursor.execute("""
                SELECT 
                    COLUMN_NAME,
                    DATA_TYPE,
                    CHARACTER_MAXIMUM_LENGTH,
                    NUMERIC_PRECISION,
                    NUMERIC_SCALE,
                    IS_NULLABLE,
                    COLUMN_DEFAULT
                FROM INFORMATION_SCHEMA.COLUMNS
                WHERE TABLE_SCHEMA = 'RAG' 
                AND TABLE_NAME = ? 
                AND COLUMN_NAME = ?
            """, (table_name, column_name))
            
            result = cursor.fetchone()
            if result:
                col_name, data_type, char_max_len, num_prec, num_scale, is_null, col_default = result
                logger.info(f"  Column: {col_name}")
                logger.info(f"  Data Type: {data_type}")
                logger.info(f"  Character Max Length: {char_max_len}")
                logger.info(f"  Numeric Precision: {num_prec}")
                logger.info(f"  Numeric Scale: {num_scale}")
                logger.info(f"  Is Nullable: {is_null}")
                logger.info(f"  Default: {col_default}")
                
                self.results[f"{table_name}_{column_name}_schema"] = {
                    "data_type": data_type,
                    "char_max_len": char_max_len,
                    "num_prec": num_prec,
                    "num_scale": num_scale
                }
            else:
                logger.warning(f"  ❌ Column not found in INFORMATION_SCHEMA")
                self.results[f"{table_name}_{column_name}_schema"] = None
        
        # 1.2: Try IRIS-specific system tables for more accurate type information
        logger.info("\n--- 1.2: IRIS-Specific System Tables ---")
        
        try:
            # Check %Dictionary.CompiledClass for table definitions
            cursor.execute("""
                SELECT Name, Type, Collection
                FROM %Dictionary.CompiledProperty
                WHERE parent->Name = 'RAG.DocumentChunks'
                AND Name = 'chunk_embedding'
            """)
            
            iris_result = cursor.fetchone()
            if iris_result:
                name, prop_type, collection = iris_result
                logger.info(f"IRIS Dictionary - DocumentChunks.chunk_embedding:")
                logger.info(f"  Name: {name}")
                logger.info(f"  Type: {prop_type}")
                logger.info(f"  Collection: {collection}")
                self.results["iris_dictionary_chunks"] = {
                    "name": name,
                    "type": prop_type,
                    "collection": collection
                }
            else:
                logger.info("No IRIS Dictionary information found for DocumentChunks.chunk_embedding")
                
        except Exception as e:
            logger.warning(f"Could not query IRIS Dictionary tables: {e}")
            
        # 1.3: Test actual vector operations to verify true column type
        logger.info("\n--- 1.3: Vector Operation Tests ---")
        
        for table_name, column_name in tables_to_check:
            logger.info(f"\nTesting vector operations on {table_name}.{column_name}:")
            
            try:
                # Test if we can use VECTOR functions on the column
                if table_name == "DocumentChunks":
                    test_sql = f"""
                        SELECT TOP 1 
                            chunk_id,
                            CASE 
                                WHEN {column_name} IS NOT NULL THEN 'HAS_VALUE'
                                ELSE 'NULL'
                            END as value_status
                        FROM RAG.{table_name}
                    """
                else:
                    test_sql = f"""
                        SELECT TOP 1 
                            doc_id,
                            CASE 
                                WHEN {column_name} IS NOT NULL THEN 'HAS_VALUE'
                                ELSE 'NULL'
                            END as value_status
                        FROM RAG.{table_name}
                    """
                
                cursor.execute(test_sql)
                result = cursor.fetchone()
                if result:
                    logger.info(f"  ✅ Basic query successful: {result}")
                else:
                    logger.info(f"  ⚠️ No rows found in {table_name}")
                    
                # Try vector function if we have data
                if result and result[1] == 'HAS_VALUE':
                    try:
                        vector_test_sql = f"""
                            SELECT TOP 1 
                                VECTOR_DIMENSION({column_name}) as dimension
                            FROM RAG.{table_name}
                            WHERE {column_name} IS NOT NULL
                        """
                        cursor.execute(vector_test_sql)
                        dim_result = cursor.fetchone()
                        if dim_result:
                            logger.info(f"  ✅ VECTOR_DIMENSION() works: {dim_result[0]} dimensions")
                            self.results[f"{table_name}_vector_dimension"] = dim_result[0]
                        else:
                            logger.info(f"  ⚠️ VECTOR_DIMENSION() returned no result")
                    except Exception as ve:
                        logger.warning(f"  ❌ VECTOR_DIMENSION() failed: {ve}")
                        self.results[f"{table_name}_vector_test"] = f"FAILED: {ve}"
                        
            except Exception as e:
                logger.error(f"  ❌ Basic query failed: {e}")
                
        cursor.close()
    
    def debug_missing_chunk_embeddings(self):
        """Part 2: Investigate missing chunk embeddings."""
        logger.info("\n" + "="*60)
        logger.info("PART 2: MISSING CHUNK EMBEDDINGS INVESTIGATION")
        logger.info("="*60)
        
        cursor = self.connection.cursor()
        
        # 2.1: Check DocumentChunks table state
        logger.info("\n--- 2.1: DocumentChunks Table Analysis ---")
        
        try:
            # Total rows
            cursor.execute("SELECT COUNT(*) FROM RAG.DocumentChunks")
            total_chunks = cursor.fetchone()[0]
            logger.info(f"Total chunks in DocumentChunks: {total_chunks:,}")
            
            # Rows with non-NULL chunk_embedding
            cursor.execute("SELECT COUNT(*) FROM RAG.DocumentChunks WHERE chunk_embedding IS NOT NULL")
            chunks_with_embedding = cursor.fetchone()[0]
            logger.info(f"Chunks with non-NULL chunk_embedding: {chunks_with_embedding:,}")
            
            # Rows with non-empty chunk_embedding (in case it's empty string)
            cursor.execute("SELECT COUNT(*) FROM RAG.DocumentChunks WHERE chunk_embedding IS NOT NULL AND chunk_embedding != ''")
            chunks_with_nonempty_embedding = cursor.fetchone()[0]
            logger.info(f"Chunks with non-empty chunk_embedding: {chunks_with_nonempty_embedding:,}")
            
            self.results["chunk_stats"] = {
                "total_chunks": total_chunks,
                "chunks_with_embedding": chunks_with_embedding,
                "chunks_with_nonempty_embedding": chunks_with_nonempty_embedding
            }
            
            # Sample chunk data
            if total_chunks > 0:
                logger.info("\nSample chunk data:")
                cursor.execute("""
                    SELECT TOP 3 
                        chunk_id, 
                        doc_id,
                        SUBSTRING(chunk_text, 1, 100) as text_preview,
                        CASE 
                            WHEN chunk_embedding IS NULL THEN 'NULL'
                            WHEN chunk_embedding = '' THEN 'EMPTY_STRING'
                            ELSE SUBSTRING(CAST(chunk_embedding AS VARCHAR), 1, 50) + '...'
                        END as embedding_preview
                    FROM RAG.DocumentChunks
                """)
                
                samples = cursor.fetchall()
                for i, (chunk_id, doc_id, text_preview, embedding_preview) in enumerate(samples):
                    logger.info(f"  Sample {i+1}:")
                    logger.info(f"    Chunk ID: {chunk_id}")
                    logger.info(f"    Doc ID: {doc_id}")
                    logger.info(f"    Text: {text_preview}")
                    logger.info(f"    Embedding: {embedding_preview}")
                    
        except Exception as e:
            logger.error(f"❌ Error analyzing DocumentChunks: {e}")
            
        # 2.2: Check data ingestion pipeline
        logger.info("\n--- 2.2: Data Ingestion Pipeline Analysis ---")
        
        # Check if SourceDocuments have embeddings (for comparison)
        try:
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
            total_docs = cursor.fetchone()[0]
            logger.info(f"Total documents in SourceDocuments: {total_docs:,}")
            
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE embedding IS NOT NULL")
            docs_with_embedding = cursor.fetchone()[0]
            logger.info(f"Documents with embeddings: {docs_with_embedding:,}")
            
            self.results["source_doc_stats"] = {
                "total_docs": total_docs,
                "docs_with_embedding": docs_with_embedding
            }
            
        except Exception as e:
            logger.error(f"❌ Error analyzing SourceDocuments: {e}")
            
        # 2.3: Check for chunking service usage
        logger.info("\n--- 2.3: Chunking Service Investigation ---")
        
        # Look for evidence of chunking process
        try:
            # Check if chunks have metadata indicating chunking strategy
            cursor.execute("""
                SELECT 
                    chunk_type,
                    COUNT(*) as count
                FROM RAG.DocumentChunks
                GROUP BY chunk_type
            """)
            
            chunk_types = cursor.fetchall()
            if chunk_types:
                logger.info("Chunk types found:")
                for chunk_type, count in chunk_types:
                    logger.info(f"  {chunk_type}: {count:,} chunks")
            else:
                logger.info("No chunk type information found")
                
            # Check chunk sizes
            cursor.execute("""
                SELECT 
                    MIN(LENGTH(chunk_text)) as min_length,
                    MAX(LENGTH(chunk_text)) as max_length,
                    AVG(LENGTH(chunk_text)) as avg_length
                FROM RAG.DocumentChunks
                WHERE chunk_text IS NOT NULL
            """)
            
            size_stats = cursor.fetchone()
            if size_stats:
                min_len, max_len, avg_len = size_stats
                logger.info(f"Chunk text lengths - Min: {min_len}, Max: {max_len}, Avg: {avg_len:.1f}")
                
        except Exception as e:
            logger.error(f"❌ Error analyzing chunk metadata: {e}")
            
        cursor.close()
    
    def test_vector_insertion(self):
        """Test if we can insert vectors into the problematic columns."""
        logger.info("\n" + "="*60)
        logger.info("PART 3: VECTOR INSERTION TESTS")
        logger.info("="*60)
        
        cursor = self.connection.cursor()
        
        # Test DocumentChunks insertion
        logger.info("\n--- 3.1: DocumentChunks Vector Insertion Test ---")
        
        try:
            # Create test vector (384 dimensions for chunk_embedding)
            test_vector = [0.1] * 384
            vector_str = ','.join(map(str, test_vector))
            
            # Try inserting
            test_chunk_id = "debug_test_chunk_001"
            
            cursor.execute("""
                INSERT INTO RAG.DocumentChunks 
                (chunk_id, doc_id, chunk_text, chunk_embedding, chunk_index)
                VALUES (?, ?, ?, TO_VECTOR(?), ?)
            """, (test_chunk_id, "debug_test_doc", "Test chunk for debugging", vector_str, 0))
            
            self.connection.commit()
            logger.info("✅ Successfully inserted test vector into DocumentChunks")
            
            # Verify insertion
            cursor.execute("""
                SELECT chunk_id, 
                       CASE WHEN chunk_embedding IS NOT NULL THEN 'HAS_EMBEDDING' ELSE 'NO_EMBEDDING' END,
                       VECTOR_DIMENSION(chunk_embedding) as dimension
                FROM RAG.DocumentChunks 
                WHERE chunk_id = ?
            """, (test_chunk_id,))
            
            result = cursor.fetchone()
            if result:
                chunk_id, has_embedding, dimension = result
                logger.info(f"✅ Verification successful: {chunk_id}, {has_embedding}, {dimension} dimensions")
                self.results["vector_insertion_test"] = "SUCCESS"
            else:
                logger.warning("⚠️ Could not verify inserted vector")
                
            # Clean up
            cursor.execute("DELETE FROM RAG.DocumentChunks WHERE chunk_id = ?", (test_chunk_id,))
            self.connection.commit()
            
        except Exception as e:
            logger.error(f"❌ Vector insertion test failed: {e}")
            self.results["vector_insertion_test"] = f"FAILED: {e}"
            
        cursor.close()
    
    def generate_diagnostic_report(self):
        """Generate comprehensive diagnostic report."""
        logger.info("\n" + "="*60)
        logger.info("DIAGNOSTIC REPORT")
        logger.info("="*60)
        
        logger.info("\n--- SCHEMA DISCREPANCY FINDINGS ---")
        
        # Analyze schema findings
        chunks_schema = self.results.get("DocumentChunks_chunk_embedding_schema")
        tokens_schema = self.results.get("DocumentTokenEmbeddings_token_embedding_schema")
        
        if chunks_schema:
            logger.info(f"DocumentChunks.chunk_embedding:")
            logger.info(f"  INFORMATION_SCHEMA reports: {chunks_schema['data_type']}")
            logger.info(f"  Character length: {chunks_schema['char_max_len']}")
            
            if chunks_schema['data_type'].upper() == 'VARCHAR':
                logger.info("  🔍 FINDING: Column is reported as VARCHAR, not VECTOR")
                logger.info("  💡 HYPOTHESIS: IRIS may store VECTOR as VARCHAR internally")
        
        if tokens_schema:
            logger.info(f"DocumentTokenEmbeddings.token_embedding:")
            logger.info(f"  INFORMATION_SCHEMA reports: {tokens_schema['data_type']}")
            logger.info(f"  Character length: {tokens_schema['char_max_len']}")
        
        # Check vector dimension tests
        chunks_dim = self.results.get("DocumentChunks_vector_dimension")
        tokens_dim = self.results.get("DocumentTokenEmbeddings_vector_dimension")
        
        if chunks_dim:
            logger.info(f"  ✅ VECTOR_DIMENSION() works on DocumentChunks: {chunks_dim} dimensions")
        if tokens_dim:
            logger.info(f"  ✅ VECTOR_DIMENSION() works on DocumentTokenEmbeddings: {tokens_dim} dimensions")
            
        logger.info("\n--- MISSING CHUNK EMBEDDINGS FINDINGS ---")
        
        chunk_stats = self.results.get("chunk_stats", {})
        source_stats = self.results.get("source_doc_stats", {})
        
        logger.info(f"DocumentChunks analysis:")
        logger.info(f"  Total chunks: {chunk_stats.get('total_chunks', 'Unknown'):,}")
        logger.info(f"  Chunks with embeddings: {chunk_stats.get('chunks_with_embedding', 'Unknown'):,}")
        logger.info(f"  Chunks with non-empty embeddings: {chunk_stats.get('chunks_with_nonempty_embedding', 'Unknown'):,}")
        
        logger.info(f"SourceDocuments comparison:")
        logger.info(f"  Total documents: {source_stats.get('total_docs', 'Unknown'):,}")
        logger.info(f"  Documents with embeddings: {source_stats.get('docs_with_embedding', 'Unknown'):,}")
        
        # Determine root causes
        logger.info("\n--- ROOT CAUSE ANALYSIS ---")
        
        if chunks_schema and chunks_schema['data_type'].upper() == 'VARCHAR':
            if chunks_dim:
                logger.info("🔍 SCHEMA DISCREPANCY ROOT CAUSE:")
                logger.info("  - INFORMATION_SCHEMA reports VECTOR columns as VARCHAR")
                logger.info("  - This appears to be normal IRIS behavior")
                logger.info("  - VECTOR functions work correctly, indicating true VECTOR storage")
                logger.info("  - CHARACTER_MAXIMUM_LENGTH indicates serialized vector size")
            else:
                logger.info("🔍 SCHEMA DISCREPANCY ROOT CAUSE:")
                logger.info("  - Columns may actually be VARCHAR, not VECTOR")
                logger.info("  - Need to check DDL execution and table creation")
        
        if chunk_stats.get('chunks_with_embedding', 0) == 0:
            logger.info("🔍 MISSING CHUNK EMBEDDINGS ROOT CAUSE:")
            logger.info("  - No chunk embeddings are being generated or stored")
            logger.info("  - Check data ingestion pipeline in data/loader.py")
            logger.info("  - Verify embedding function is being called for chunks")
            logger.info("  - Check if chunking service is being used")
            
        vector_test = self.results.get("vector_insertion_test")
        if vector_test == "SUCCESS":
            logger.info("✅ VECTOR INSERTION TEST: Successful")
            logger.info("  - Schema supports vector insertion")
            logger.info("  - Problem is likely in data ingestion pipeline")
        elif vector_test:
            logger.info(f"❌ VECTOR INSERTION TEST: {vector_test}")
            logger.info("  - Schema may have issues")
            
    def run_full_diagnosis(self):
        """Run complete diagnostic process."""
        logger.info("Starting comprehensive schema and embedding diagnosis...")
        
        if not self.connect():
            return False
            
        try:
            self.debug_schema_discrepancy()
            self.debug_missing_chunk_embeddings()
            self.test_vector_insertion()
            self.generate_diagnostic_report()
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Diagnosis failed: {e}")
            return False
            
        finally:
            if self.connection:
                self.connection.close()
                logger.info("Database connection closed")

def main():
    """Main function."""
    debugger = SchemaAndEmbeddingDebugger()
    success = debugger.run_full_diagnosis()
    
    if success:
        logger.info("\n🎉 Diagnosis completed successfully!")
        logger.info("Check the output above for detailed findings and recommendations.")
    else:
        logger.error("\n❌ Diagnosis failed!")
        return 1
        
    return 0

if __name__ == "__main__":
    sys.exit(main())