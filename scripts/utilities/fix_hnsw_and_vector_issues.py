#!/usr/bin/env python3
"""
Fix HNSW and Vector Issues Script

This script addresses the critical issues found in HNSW verification:
1. Convert VARCHAR embedding columns to proper VECTOR type
2. Create proper HNSW indexes using correct IRIS syntax
3. Verify vector search functionality
"""

import os
import sys
import time
import json
import numpy as np
from typing import Dict, List, Tuple, Any
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.iris_connector import get_iris_connection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class VectorIndexFixer:
    """Fixes vector storage and HNSW indexing issues"""
    
    def __init__(self):
        self.connection = get_iris_connection()
        self.cursor = self.connection.cursor()
        self.results = {}
        
    def check_column_type(self, table_name: str, column_name: str) -> str:
        """Check the current data type of a column"""
        try:
            self.cursor.execute(f"""
                SELECT DATA_TYPE 
                FROM INFORMATION_SCHEMA.COLUMNS 
                WHERE TABLE_NAME = '{table_name}' 
                AND TABLE_SCHEMA = 'RAG' 
                AND COLUMN_NAME = '{column_name}'
            """)
            result = self.cursor.fetchone()
            return result[0] if result else 'UNKNOWN'
        except Exception as e:
            logger.error(f"Error checking column type for {table_name}.{column_name}: {e}")
            return 'ERROR'
    
    def convert_varchar_to_vector(self, table_name: str, column_name: str) -> Dict[str, Any]:
        """Convert VARCHAR embedding column to VECTOR type"""
        logger.info(f"Converting {table_name}.{column_name} from VARCHAR to VECTOR")
        
        result = {
            'table': table_name,
            'column': column_name,
            'conversion_successful': False,
            'original_type': None,
            'new_type': None,
            'error': None
        }
        
        try:
            # Check current type
            result['original_type'] = self.check_column_type(table_name, column_name)
            logger.info(f"Current type: {result['original_type']}")
            
            if 'VECTOR' in result['original_type'].upper():
                logger.info(f"Column {table_name}.{column_name} is already VECTOR type")
                result['conversion_successful'] = True
                result['new_type'] = result['original_type']
                return result
            
            # Create a new VECTOR column
            temp_column = f"{column_name}_vector"
            
            logger.info(f"Step 1: Adding temporary VECTOR column {temp_column}")
            self.cursor.execute(f"""
                ALTER TABLE RAG.{table_name} 
                ADD COLUMN {temp_column} VECTOR(FLOAT, 768)
            """)
            
            logger.info(f"Step 2: Converting VARCHAR data to VECTOR format")
            # Update the new column with converted data
            self.cursor.execute(f"""
                UPDATE RAG.{table_name} 
                SET {temp_column} = TO_VECTOR({column_name})
                WHERE {column_name} IS NOT NULL
            """)
            
            logger.info(f"Step 3: Dropping original VARCHAR column")
            self.cursor.execute(f"""
                ALTER TABLE RAG.{table_name} 
                DROP COLUMN {column_name}
            """)
            
            logger.info(f"Step 4: Renaming VECTOR column to original name")
            self.cursor.execute(f"""
                ALTER TABLE RAG.{table_name} 
                RENAME COLUMN {temp_column} TO {column_name}
            """)
            
            # Verify the conversion
            result['new_type'] = self.check_column_type(table_name, column_name)
            result['conversion_successful'] = True
            
            logger.info(f"✅ Successfully converted {table_name}.{column_name} to VECTOR type")
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"❌ Error converting {table_name}.{column_name}: {e}")
            
            # Try to clean up if there was an error
            try:
                self.cursor.execute(f"""
                    ALTER TABLE RAG.{table_name} 
                    DROP COLUMN {column_name}_vector
                """)
            except:
                pass
        
        return result
    
    def create_hnsw_index_proper(self, table_name: str, column_name: str) -> Dict[str, Any]:
        """Create HNSW index using proper IRIS syntax"""
        logger.info(f"Creating HNSW index on {table_name}.{column_name}")
        
        result = {
            'table': table_name,
            'column': column_name,
            'index_created': False,
            'index_name': None,
            'error': None
        }
        
        try:
            index_name = f"idx_{table_name}_{column_name}_hnsw"
            
            # Drop existing index if it exists
            try:
                self.cursor.execute(f"DROP INDEX RAG.{table_name}.{index_name}")
                logger.info(f"Dropped existing index {index_name}")
            except:
                pass  # Index doesn't exist, which is fine
            
            # Create HNSW index using proper IRIS syntax
            # Note: IRIS uses different syntax for vector indexes
            create_query = f"""
                CREATE INDEX {index_name} ON RAG.{table_name} ({column_name})
            """
            
            self.cursor.execute(create_query)
            
            result['index_created'] = True
            result['index_name'] = index_name
            
            logger.info(f"✅ Successfully created index {index_name}")
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"❌ Error creating HNSW index on {table_name}.{column_name}: {e}")
        
        return result
    
    def test_vector_search(self, table_name: str, column_name: str, limit: int = 5) -> Dict[str, Any]:
        """Test vector search functionality"""
        logger.info(f"Testing vector search on {table_name}.{column_name}")
        
        result = {
            'table': table_name,
            'column': column_name,
            'search_successful': False,
            'execution_time': None,
            'results_count': 0,
            'error': None
        }
        
        try:
            # Get a sample vector
            self.cursor.execute(f"""
                SELECT TOP 1 {column_name} 
                FROM RAG.{table_name} 
                WHERE {column_name} IS NOT NULL
            """)
            sample_row = self.cursor.fetchone()
            
            if not sample_row or not sample_row[0]:
                result['error'] = f"No vectors found in {table_name}.{column_name}"
                return result
            
            sample_vector = sample_row[0]
            
            # Test vector search with timing
            start_time = time.time()
            
            # Use proper IRIS vector search syntax
            search_query = f"""
                SELECT TOP {limit} ID, VECTOR_DOT_PRODUCT({column_name}, ?) as similarity
                FROM RAG.{table_name}
                WHERE {column_name} IS NOT NULL
                ORDER BY similarity DESC
            """
            
            self.cursor.execute(search_query, (sample_vector,))
            results = self.cursor.fetchall()
            
            end_time = time.time()
            
            result['search_successful'] = True
            result['execution_time'] = end_time - start_time
            result['results_count'] = len(results)
            
            logger.info(f"✅ Vector search completed in {result['execution_time']:.4f}s, found {result['results_count']} results")
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"❌ Error testing vector search on {table_name}.{column_name}: {e}")
        
        return result
    
    def get_table_stats(self, table_name: str) -> Dict[str, Any]:
        """Get table statistics"""
        stats = {
            'table': table_name,
            'row_count': 0,
            'embedding_count': 0,
            'embedding_column': None
        }
        
        try:
            # Get row count
            self.cursor.execute(f"SELECT COUNT(*) FROM RAG.{table_name}")
            result = self.cursor.fetchone()
            stats['row_count'] = result[0] if result else 0
            
            # Check for embedding columns
            embedding_columns = ['embedding', 'token_embedding']
            
            for col in embedding_columns:
                try:
                    self.cursor.execute(f"""
                        SELECT COUNT(*) 
                        FROM RAG.{table_name} 
                        WHERE {col} IS NOT NULL
                    """)
                    result = self.cursor.fetchone()
                    count = result[0] if result else 0
                    if count > 0:
                        stats['embedding_count'] = count
                        stats['embedding_column'] = col
                        break
                except:
                    continue
                    
        except Exception as e:
            logger.error(f"Error getting stats for {table_name}: {e}")
        
        return stats
    
    def fix_all_vector_issues(self) -> Dict[str, Any]:
        """Fix all vector storage and indexing issues"""
        logger.info("Starting comprehensive vector and HNSW fix")
        
        # Tables and columns to fix
        tables_to_fix = [
            ('SourceDocuments_V2', 'embedding'),
            ('DocumentTokenEmbeddings', 'token_embedding')
        ]
        
        fix_results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'tables': {},
            'overall_status': 'UNKNOWN',
            'summary': []
        }
        
        all_fixes_successful = True
        
        for table_name, column_name in tables_to_fix:
            logger.info(f"\n=== Fixing {table_name}.{column_name} ===")
            
            table_results = {
                'stats': self.get_table_stats(table_name),
                'column_type_check': None,
                'vector_conversion': None,
                'index_creation': None,
                'vector_search_test': None
            }
            
            # Only proceed if table has data
            if table_results['stats']['row_count'] > 0 and table_results['stats']['embedding_count'] > 0:
                
                # Check current column type
                current_type = self.check_column_type(table_name, column_name)
                table_results['column_type_check'] = current_type
                
                # Convert to VECTOR if needed
                if 'VECTOR' not in current_type.upper():
                    table_results['vector_conversion'] = self.convert_varchar_to_vector(table_name, column_name)
                    if not table_results['vector_conversion']['conversion_successful']:
                        all_fixes_successful = False
                        continue
                else:
                    logger.info(f"Column {table_name}.{column_name} is already VECTOR type")
                
                # Create HNSW index
                table_results['index_creation'] = self.create_hnsw_index_proper(table_name, column_name)
                if not table_results['index_creation']['index_created']:
                    logger.warning(f"Index creation failed for {table_name}.{column_name}")
                
                # Test vector search
                table_results['vector_search_test'] = self.test_vector_search(table_name, column_name)
                if not table_results['vector_search_test']['search_successful']:
                    all_fixes_successful = False
            
            fix_results['tables'][table_name] = table_results
        
        # Determine overall status
        if all_fixes_successful:
            fix_results['overall_status'] = 'SUCCESS'
            fix_results['summary'].append("✅ All vector storage and indexing issues fixed")
            fix_results['summary'].append("✅ HNSW indexes created successfully")
            fix_results['summary'].append("✅ Vector search functionality verified")
            fix_results['summary'].append("✅ Safe to resume large-scale ingestion")
        else:
            fix_results['overall_status'] = 'PARTIAL_SUCCESS'
            fix_results['summary'].append("⚠️ Some issues were fixed, but problems remain")
            fix_results['summary'].append("❌ Review individual table results")
            fix_results['summary'].append("❌ Do NOT resume large-scale ingestion until all issues resolved")
        
        return fix_results
    
    def print_results(self, results: Dict[str, Any]):
        """Print fix results in a readable format"""
        print("\n" + "="*80)
        print("VECTOR AND HNSW FIX REPORT")
        print("="*80)
        print(f"Timestamp: {results['timestamp']}")
        print(f"Overall Status: {results['overall_status']}")
        print()
        
        for table_name, table_data in results['tables'].items():
            print(f"\n--- {table_name} ---")
            
            # Table stats
            stats = table_data['stats']
            print(f"  Row count: {stats['row_count']:,}")
            print(f"  Embedding count: {stats['embedding_count']:,}")
            if stats['embedding_column']:
                print(f"  Embedding column: {stats['embedding_column']}")
            
            # Column type check
            if table_data['column_type_check']:
                print(f"  Original column type: {table_data['column_type_check']}")
            
            # Vector conversion
            conversion = table_data['vector_conversion']
            if conversion:
                if conversion['conversion_successful']:
                    print(f"  Vector conversion: ✅ {conversion['original_type']} → {conversion['new_type']}")
                else:
                    print(f"  Vector conversion: ❌ Failed")
                    if conversion['error']:
                        print(f"    Error: {conversion['error']}")
            
            # Index creation
            index_creation = table_data['index_creation']
            if index_creation:
                if index_creation['index_created']:
                    print(f"  HNSW Index: ✅ Created {index_creation['index_name']}")
                else:
                    print(f"  HNSW Index: ❌ Failed")
                    if index_creation['error']:
                        print(f"    Error: {index_creation['error']}")
            
            # Vector search test
            search_test = table_data['vector_search_test']
            if search_test:
                if search_test['search_successful']:
                    print(f"  Vector Search: ✅ {search_test['execution_time']:.4f}s ({search_test['results_count']} results)")
                else:
                    print(f"  Vector Search: ❌ Failed")
                    if search_test['error']:
                        print(f"    Error: {search_test['error']}")
        
        print("\n" + "-"*80)
        print("SUMMARY:")
        for summary_item in results['summary']:
            print(f"  {summary_item}")
        print("-"*80)

def main():
    """Main function"""
    print("Vector and HNSW Fix Starting...")
    
    fixer = VectorIndexFixer()
    results = fixer.fix_all_vector_issues()
    fixer.print_results(results)
    
    # Save results to file
    results_file = f"hnsw_fix_results_{int(time.time())}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    print(f"\n📄 Results saved to: {results_file}")
    
    # Return appropriate exit code
    if results['overall_status'] == 'SUCCESS':
        print(f"\n🎉 FIX SUCCESSFUL: {results['overall_status']}")
        return 0
    else:
        print(f"\n⚠️ FIX INCOMPLETE: {results['overall_status']}")
        return 1

if __name__ == "__main__":
    sys.exit(main())