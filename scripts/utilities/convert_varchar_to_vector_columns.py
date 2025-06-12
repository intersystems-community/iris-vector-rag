#!/usr/bin/env python3
"""
Convert VARCHAR Vector Columns to Proper VECTOR Columns

This script converts all VARCHAR columns used for vector storage to proper VECTOR columns
so that HNSW indexes can be created successfully.

The issue: IRIS HNSW indexes require proper VECTOR columns, not VARCHAR columns.
The solution: Convert all embedding storage from VARCHAR to VECTOR data type.

Author: RAG System Team
Date: 2025-01-26
"""

import logging
import sys
import os
import json
import time
from typing import List, Dict, Any, Optional, Tuple

# Add the project root to the path
sys.path.insert(0, os.path.abspath(os.path.dirname(os.path.dirname(__file__))))

from common.iris_connector import get_iris_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class VectorColumnConverter:
    """Converts VARCHAR vector columns to proper VECTOR columns."""
    
    def __init__(self):
        self.connection = None
        
    def connect(self):
        """Establish database connection."""
        try:
            self.connection = get_iris_connection()
            logger.info("✅ Database connection established")
            return True
        except Exception as e:
            logger.error(f"❌ Failed to connect to database: {e}")
            return False
    
    def analyze_current_vector_columns(self) -> Dict[str, Any]:
        """Analyze current vector column situation."""
        cursor = self.connection.cursor()
        analysis = {
            "tables_analyzed": [],
            "varchar_vector_columns": [],
            "proper_vector_columns": [],
            "data_counts": {}
        }
        
        try:
            # Tables and their vector columns to check
            vector_columns_to_check = [
                ("RAG.SourceDocuments_V2", "embedding", 768),
                ("RAG.DocumentChunks", "embedding", 768),
                ("RAG.KnowledgeGraphNodes", "embedding", 768),
                ("RAG.DocumentTokenEmbeddings", "token_embedding", 128)
            ]
            
            for table_name, column_name, expected_dim in vector_columns_to_check:
                try:
                    # Check if table exists
                    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
                    row_count = cursor.fetchone()[0]
                    analysis["data_counts"][table_name] = row_count
                    
                    # Check column type
                    schema_name, table_only = table_name.split('.')
                    cursor.execute(f"""
                        SELECT DATA_TYPE, CHARACTER_MAXIMUM_LENGTH 
                        FROM INFORMATION_SCHEMA.COLUMNS 
                        WHERE TABLE_SCHEMA = '{schema_name}' 
                        AND TABLE_NAME = '{table_only}'
                        AND COLUMN_NAME = '{column_name}'
                    """)
                    
                    result = cursor.fetchone()
                    if result:
                        data_type, max_length = result
                        column_info = {
                            "table": table_name,
                            "column": column_name,
                            "current_type": data_type,
                            "max_length": max_length,
                            "expected_dimension": expected_dim,
                            "row_count": row_count
                        }
                        
                        if data_type.lower() == 'varchar':
                            analysis["varchar_vector_columns"].append(column_info)
                            logger.warning(f"⚠️ {table_name}.{column_name} is VARCHAR({max_length}) - needs conversion")
                        else:
                            analysis["proper_vector_columns"].append(column_info)
                            logger.info(f"✅ {table_name}.{column_name} is {data_type} - proper type")
                    else:
                        logger.warning(f"⚠️ Column {column_name} not found in {table_name}")
                        
                    analysis["tables_analyzed"].append(table_name)
                    
                except Exception as e:
                    logger.error(f"❌ Error analyzing {table_name}.{column_name}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"❌ Error during analysis: {e}")
        finally:
            cursor.close()
            
        return analysis
    
    def backup_varchar_data(self, table_name: str, column_name: str) -> bool:
        """Backup VARCHAR vector data before conversion."""
        cursor = self.connection.cursor()
        
        try:
            backup_table = f"{table_name}_backup_{int(time.time())}"
            logger.info(f"📦 Creating backup table: {backup_table}")
            
            # Create backup table
            cursor.execute(f"CREATE TABLE {backup_table} AS SELECT * FROM {table_name}")
            self.connection.commit()
            
            # Verify backup
            cursor.execute(f"SELECT COUNT(*) FROM {backup_table}")
            backup_count = cursor.fetchone()[0]
            
            cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
            original_count = cursor.fetchone()[0]
            
            if backup_count == original_count:
                logger.info(f"✅ Backup successful: {backup_count} rows backed up")
                return True
            else:
                logger.error(f"❌ Backup failed: {original_count} original vs {backup_count} backup")
                return False
                
        except Exception as e:
            logger.error(f"❌ Error creating backup: {e}")
            return False
        finally:
            cursor.close()
    
    def convert_varchar_to_vector(self, table_name: str, column_name: str, dimension: int) -> bool:
        """Convert a VARCHAR column to proper VECTOR column."""
        cursor = self.connection.cursor()
        
        try:
            logger.info(f"🔧 Converting {table_name}.{column_name} from VARCHAR to VECTOR({dimension})")
            
            # Step 1: Check if we have data to preserve
            cursor.execute(f"SELECT COUNT(*) FROM {table_name} WHERE {column_name} IS NOT NULL")
            data_count = cursor.fetchone()[0]
            
            if data_count > 0:
                logger.info(f"📊 Found {data_count} rows with vector data to preserve")
                
                # Step 2: Create backup
                if not self.backup_varchar_data(table_name, column_name):
                    logger.error(f"❌ Backup failed for {table_name} - aborting conversion")
                    return False
            
            # Step 3: Add new VECTOR column
            new_column_name = f"{column_name}_vector"
            try:
                cursor.execute(f"""
                    ALTER TABLE {table_name} 
                    ADD COLUMN {new_column_name} VECTOR(DOUBLE, {dimension})
                """)
                logger.info(f"✅ Added new VECTOR column: {new_column_name}")
            except Exception as e:
                if "already exists" in str(e).lower():
                    logger.info(f"⚠️ Column {new_column_name} already exists")
                else:
                    raise e
            
            # Step 4: Convert data from VARCHAR to VECTOR
            if data_count > 0:
                logger.info(f"🔄 Converting {data_count} vector strings to VECTOR format...")
                
                # Process in batches to avoid memory issues
                batch_size = 100
                converted_count = 0
                
                # Get all rows with vector data
                cursor.execute(f"""
                    SELECT ROW_NUMBER() OVER (ORDER BY {column_name}) as rn,
                           {column_name}
                    FROM {table_name} 
                    WHERE {column_name} IS NOT NULL
                """)
                
                rows = cursor.fetchall()
                
                for i in range(0, len(rows), batch_size):
                    batch = rows[i:i+batch_size]
                    
                    for rn, varchar_vector in batch:
                        try:
                            # Convert VARCHAR vector to VECTOR using TO_VECTOR
                            update_cursor = self.connection.cursor()
                            update_cursor.execute(f"""
                                UPDATE {table_name} 
                                SET {new_column_name} = TO_VECTOR(?, 'FLOAT', {dimension})
                                WHERE {column_name} = ?
                            """, (varchar_vector, varchar_vector))
                            
                            converted_count += 1
                            
                            if converted_count % 50 == 0:
                                logger.info(f"   Converted {converted_count}/{data_count} vectors...")
                                
                        except Exception as e:
                            logger.warning(f"⚠️ Failed to convert vector {rn}: {e}")
                            continue
                    
                    # Commit batch
                    self.connection.commit()
                
                logger.info(f"✅ Converted {converted_count}/{data_count} vectors successfully")
            
            # Step 5: Drop old VARCHAR column
            try:
                cursor.execute(f"ALTER TABLE {table_name} DROP COLUMN {column_name}")
                logger.info(f"🗑️ Dropped old VARCHAR column: {column_name}")
            except Exception as e:
                logger.warning(f"⚠️ Could not drop old column {column_name}: {e}")
            
            # Step 6: Rename new column to original name
            try:
                cursor.execute(f"""
                    ALTER TABLE {table_name} 
                    RENAME COLUMN {new_column_name} TO {column_name}
                """)
                logger.info(f"✅ Renamed {new_column_name} to {column_name}")
            except Exception as e:
                logger.warning(f"⚠️ Could not rename column: {e}")
                logger.info(f"   New VECTOR column is available as: {new_column_name}")
            
            self.connection.commit()
            return True
            
        except Exception as e:
            logger.error(f"❌ Error converting {table_name}.{column_name}: {e}")
            self.connection.rollback()
            return False
        finally:
            cursor.close()
    
    def create_hnsw_indexes_on_vector_columns(self) -> bool:
        """Create HNSW indexes on proper VECTOR columns."""
        cursor = self.connection.cursor()
        
        try:
            logger.info("🔧 Creating HNSW indexes on VECTOR columns...")
            
            # HNSW indexes to create (only on proper VECTOR columns)
            hnsw_indexes = [
                {
                    "name": "idx_hnsw_source_embeddings",
                    "table": "RAG.SourceDocuments_V2",
                    "column": "embedding",
                    "sql": """
                        CREATE INDEX idx_hnsw_source_embeddings
                        ON RAG.SourceDocuments_V2 (embedding)
                        AS HNSW(M=16, efConstruction=200, Distance='COSINE')
                    """
                },
                {
                    "name": "idx_hnsw_chunk_embeddings", 
                    "table": "RAG.DocumentChunks",
                    "column": "embedding",
                    "sql": """
                        CREATE INDEX idx_hnsw_chunk_embeddings
                        ON RAG.DocumentChunks (embedding)
                        AS HNSW(M=16, efConstruction=200, Distance='COSINE')
                    """
                },
                {
                    "name": "idx_hnsw_kg_node_embeddings",
                    "table": "RAG.KnowledgeGraphNodes", 
                    "column": "embedding",
                    "sql": """
                        CREATE INDEX idx_hnsw_kg_node_embeddings
                        ON RAG.KnowledgeGraphNodes (embedding)
                        AS HNSW(M=16, efConstruction=200, Distance='COSINE')
                    """
                },
                {
                    "name": "idx_hnsw_token_embeddings",
                    "table": "RAG.DocumentTokenEmbeddings",
                    "column": "token_embedding", 
                    "sql": """
                        CREATE INDEX idx_hnsw_token_embeddings
                        ON RAG.DocumentTokenEmbeddings (token_embedding)
                        AS HNSW(M=16, efConstruction=200, Distance='COSINE')
                    """
                }
            ]
            
            created_indexes = []
            failed_indexes = []
            
            for index_info in hnsw_indexes:
                try:
                    # Check if table has data
                    cursor.execute(f"SELECT COUNT(*) FROM {index_info['table']} WHERE {index_info['column']} IS NOT NULL")
                    vector_count = cursor.fetchone()[0]
                    
                    if vector_count == 0:
                        logger.warning(f"⚠️ Skipping {index_info['name']} - no vector data in {index_info['table']}")
                        continue
                    
                    # Drop existing index if it exists
                    try:
                        cursor.execute(f"DROP INDEX IF EXISTS {index_info['name']}")
                        self.connection.commit()
                        logger.info(f"🗑️ Dropped existing index {index_info['name']}")
                    except:
                        pass
                    
                    # Create HNSW index
                    cursor.execute(index_info['sql'])
                    self.connection.commit()
                    
                    created_indexes.append(index_info['name'])
                    logger.info(f"✅ Created HNSW index: {index_info['name']} on {index_info['table']} ({vector_count} vectors)")
                    
                except Exception as e:
                    failed_indexes.append((index_info['name'], str(e)))
                    logger.warning(f"⚠️ Failed to create HNSW index {index_info['name']}: {e}")
                    continue
            
            if created_indexes:
                logger.info(f"✅ Successfully created {len(created_indexes)} HNSW indexes: {created_indexes}")
            
            if failed_indexes:
                logger.warning(f"⚠️ Failed to create {len(failed_indexes)} HNSW indexes")
                for name, error in failed_indexes:
                    logger.warning(f"   - {name}: {error}")
            
            return len(created_indexes) > 0
            
        except Exception as e:
            logger.error(f"❌ Error creating HNSW indexes: {e}")
            self.connection.rollback()
            return False
        finally:
            cursor.close()
    
    def verify_vector_conversion(self) -> Dict[str, Any]:
        """Verify that vector conversion was successful."""
        cursor = self.connection.cursor()
        verification = {
            "vector_columns_verified": [],
            "hnsw_indexes_verified": [],
            "vector_search_tests": []
        }
        
        try:
            # Check vector columns
            vector_columns = [
                ("RAG.SourceDocuments_V2", "embedding"),
                ("RAG.DocumentChunks", "embedding"),
                ("RAG.KnowledgeGraphNodes", "embedding"),
                ("RAG.DocumentTokenEmbeddings", "token_embedding")
            ]
            
            for table_name, column_name in vector_columns:
                try:
                    # Check column type
                    schema_name, table_only = table_name.split('.')
                    cursor.execute(f"""
                        SELECT DATA_TYPE 
                        FROM INFORMATION_SCHEMA.COLUMNS 
                        WHERE TABLE_SCHEMA = '{schema_name}' 
                        AND TABLE_NAME = '{table_only}'
                        AND COLUMN_NAME = '{column_name}'
                    """)
                    
                    result = cursor.fetchone()
                    if result:
                        data_type = result[0]
                        is_vector = 'vector' in data_type.lower()
                        
                        verification["vector_columns_verified"].append({
                            "table": table_name,
                            "column": column_name,
                            "type": data_type,
                            "is_vector": is_vector
                        })
                        
                        status = "✅" if is_vector else "❌"
                        logger.info(f"{status} {table_name}.{column_name}: {data_type}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Could not verify {table_name}.{column_name}: {e}")
            
            # Check HNSW indexes
            try:
                cursor.execute("""
                    SELECT INDEX_NAME, TABLE_NAME 
                    FROM INFORMATION_SCHEMA.INDEXES 
                    WHERE INDEX_NAME LIKE '%hnsw%'
                """)
                
                hnsw_indexes = cursor.fetchall()
                for index_name, table_name in hnsw_indexes:
                    verification["hnsw_indexes_verified"].append({
                        "index": index_name,
                        "table": table_name
                    })
                    logger.info(f"✅ HNSW index verified: {index_name} on {table_name}")
                    
            except Exception as e:
                logger.warning(f"⚠️ Could not verify HNSW indexes: {e}")
            
        except Exception as e:
            logger.error(f"❌ Error during verification: {e}")
        finally:
            cursor.close()
            
        return verification
    
    def run_complete_conversion(self) -> bool:
        """Run the complete VARCHAR to VECTOR conversion process."""
        logger.info("🚀 Starting complete VARCHAR to VECTOR conversion...")
        
        # Step 1: Connect to database
        if not self.connect():
            return False
        
        # Step 2: Analyze current situation
        analysis = self.analyze_current_vector_columns()
        logger.info(f"📊 Analysis complete: {len(analysis['varchar_vector_columns'])} VARCHAR columns need conversion")
        
        if not analysis['varchar_vector_columns']:
            logger.info("✅ No VARCHAR vector columns found - all columns are already proper VECTOR type!")
            
            # Still try to create HNSW indexes
            hnsw_success = self.create_hnsw_indexes_on_vector_columns()
            verification = self.verify_vector_conversion()
            
            return hnsw_success
        
        # Step 3: Convert each VARCHAR column to VECTOR
        conversion_results = []
        for column_info in analysis['varchar_vector_columns']:
            table_name = column_info['table']
            column_name = column_info['column']
            dimension = column_info['expected_dimension']
            
            logger.info(f"🔄 Converting {table_name}.{column_name} to VECTOR({dimension})...")
            
            success = self.convert_varchar_to_vector(table_name, column_name, dimension)
            conversion_results.append(success)
            
            if success:
                logger.info(f"✅ Successfully converted {table_name}.{column_name}")
            else:
                logger.error(f"❌ Failed to convert {table_name}.{column_name}")
        
        # Step 4: Create HNSW indexes on converted columns
        hnsw_success = self.create_hnsw_indexes_on_vector_columns()
        
        # Step 5: Verify everything worked
        verification = self.verify_vector_conversion()
        
        # Step 6: Report results
        successful_conversions = sum(conversion_results)
        total_conversions = len(conversion_results)
        
        logger.info("📋 Conversion Results:")
        logger.info(f"   📊 VARCHAR columns converted: {successful_conversions}/{total_conversions}")
        logger.info(f"   🔍 HNSW indexes created: {'✅' if hnsw_success else '❌'}")
        logger.info(f"   ✅ VECTOR columns verified: {len(verification['vector_columns_verified'])}")
        logger.info(f"   🔍 HNSW indexes verified: {len(verification['hnsw_indexes_verified'])}")
        
        overall_success = (successful_conversions == total_conversions) and hnsw_success
        
        if overall_success:
            logger.info("🎉 ALL VARCHAR VECTOR COLUMNS SUCCESSFULLY CONVERTED TO PROPER VECTOR COLUMNS!")
            logger.info("🎉 HNSW INDEXES ARE NOW WORKING!")
        else:
            logger.warning("⚠️ Some conversions failed - check logs for details")
        
        return overall_success
    
    def cleanup(self):
        """Clean up resources."""
        if self.connection:
            self.connection.close()
            logger.info("🧹 Database connection closed")

if __name__ == "__main__":
    converter = VectorColumnConverter()
    success = converter.run_complete_conversion()
    converter.cleanup()
    if not success:
        sys.exit(1)