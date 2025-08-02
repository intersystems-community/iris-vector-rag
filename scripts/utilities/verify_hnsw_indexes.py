#!/usr/bin/env python3
"""
HNSW Index Verification Script

This script verifies that HNSW indexes exist and are working properly
on all main RAG tables before resuming large-scale ingestion.
"""

import os
import sys
import time
from typing import Dict, Any
import logging

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from common.iris_connector import get_iris_connection

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class HNSWVerifier:
    """Verifies HNSW indexes and vector search performance"""
    
    def __init__(self):
        self.connection = get_iris_connection()
        self.cursor = self.connection.cursor()
        self.results = {}
        
    def check_table_exists(self, table_name: str) -> bool:
        """Check if a table exists"""
        try:
            self.cursor.execute(f"""
                SELECT COUNT(*)
                FROM INFORMATION_SCHEMA.TABLES
                WHERE TABLE_NAME = '{table_name}' AND TABLE_SCHEMA = 'RAG'
            """)
            result = self.cursor.fetchone()
            count = result[0] if result else 0
            return count > 0
        except Exception as e:
            logger.error(f"Error checking table {table_name}: {e}")
            return False
    
    def check_hnsw_index(self, table_name: str, column_name: str) -> Dict[str, Any]:
        """Check if HNSW index exists on specified table/column"""
        logger.info(f"Checking HNSW index on {table_name}.{column_name}")
        
        result = {
            'table': table_name,
            'column': column_name,
            'index_exists': False,
            'index_name': None,
            'index_type': None,
            'error': None
        }
        
        try:
            # Check for HNSW index using IRIS system tables
            index_query = f"""
                SELECT
                    i.Name as IndexName,
                    i.Type as IndexType,
                    i.Properties as Properties
                FROM %Dictionary.IndexDefinition i
                WHERE i.parent = 'RAG.{table_name}'
                AND i.Data LIKE '%{column_name}%'
            """
            
            self.cursor.execute(index_query)
            indexes = self.cursor.fetchall()
            
            for idx in indexes:
                index_name = idx[0] if idx[0] else 'Unknown'
                index_type = idx[1] if idx[1] else 'Unknown'
                properties = idx[2] if idx[2] else ''
                
                logger.info(f"Found index: {index_name}, Type: {index_type}, Properties: {properties}")
                
                # Check if it's an HNSW index
                if 'HNSW' in properties.upper() or 'VECTOR' in index_type.upper():
                    result['index_exists'] = True
                    result['index_name'] = index_name
                    result['index_type'] = index_type
                    break
            
            # Alternative check using SQL_USER_INDEXES
            if not result['index_exists']:
                alt_query = f"""
                    SELECT INDEX_NAME, INDEX_TYPE
                    FROM INFORMATION_SCHEMA.INDEXES
                    WHERE TABLE_NAME = '{table_name}'
                    AND TABLE_SCHEMA = 'RAG'
                    AND COLUMN_NAME = '{column_name}'
                """
                
                self.cursor.execute(alt_query)
                alt_indexes = self.cursor.fetchall()
                
                for idx in alt_indexes:
                    if 'VECTOR' in str(idx[1]).upper() or 'HNSW' in str(idx[0]).upper():
                        result['index_exists'] = True
                        result['index_name'] = idx[0]
                        result['index_type'] = idx[1]
                        break
                            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error checking HNSW index on {table_name}.{column_name}: {e}")
        
        return result
    
    def test_vector_search_performance(self, table_name: str, column_name: str, limit: int = 5) -> Dict[str, Any]:
        """Test vector search performance to verify HNSW is working"""
        logger.info(f"Testing vector search performance on {table_name}.{column_name}")
        
        result = {
            'table': table_name,
            'column': column_name,
            'search_successful': False,
            'execution_time': None,
            'results_count': 0,
            'error': None
        }
        
        try:
            # First, get a sample embedding from the table
            sample_query = f"""
                SELECT TOP 1 {column_name}
                FROM RAG.{table_name}
                WHERE {column_name} IS NOT NULL
            """
            
            self.cursor.execute(sample_query)
            sample_row = self.cursor.fetchone()
            
            if not sample_row or not sample_row[0]:
                result['error'] = f"No embeddings found in {table_name}.{column_name}"
                return result
            
            # Use the sample embedding for vector search
            sample_embedding = sample_row[0]
            
            # Test vector search with timing
            start_time = time.time()
            
            search_query = f"""
                SELECT TOP {limit} ID, VECTOR_DOT_PRODUCT({column_name}, ?) as similarity
                FROM RAG.{table_name}
                WHERE {column_name} IS NOT NULL
                ORDER BY similarity DESC
            """
            
            self.cursor.execute(search_query, (sample_embedding,))
            results = self.cursor.fetchall()
            
            end_time = time.time()
            
            result['search_successful'] = True
            result['execution_time'] = end_time - start_time
            result['results_count'] = len(results)
            
            logger.info(f"Vector search completed in {result['execution_time']:.4f}s, found {result['results_count']} results")
                
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error testing vector search on {table_name}.{column_name}: {e}")
        
        return result
    
    def get_table_stats(self, table_name: str) -> Dict[str, Any]:
        """Get basic statistics about a table"""
        logger.info(f"Getting statistics for {table_name}")
        
        stats = {
            'table': table_name,
            'exists': False,
            'row_count': 0,
            'embedding_count': 0,
            'error': None
        }
        
        try:
            if not self.check_table_exists(table_name):
                stats['error'] = f"Table {table_name} does not exist"
                return stats
            
            stats['exists'] = True
            
            # Get row count
            count_query = f"SELECT COUNT(*) FROM RAG.{table_name}"
            self.cursor.execute(count_query)
            count_result = self.cursor.fetchone()
            stats['row_count'] = count_result[0] if count_result else 0
            
            # Try to get embedding count for different possible embedding columns
            embedding_columns = ['embedding', 'token_embedding', 'embeddings']
            
            for col in embedding_columns:
                try:
                    embed_query = f"""
                        SELECT COUNT(*)
                        FROM RAG.{table_name}
                        WHERE {col} IS NOT NULL
                    """
                    self.cursor.execute(embed_query)
                    embed_result = self.cursor.fetchone()
                    embed_count = embed_result[0] if embed_result else 0
                    if embed_count > 0:
                        stats['embedding_count'] = embed_count
                        stats['embedding_column'] = col
                        break
                except:
                    continue
                        
        except Exception as e:
            stats['error'] = str(e)
            logger.error(f"Error getting stats for {table_name}: {e}")
        
        return stats
    
    def create_hnsw_index(self, table_name: str, column_name: str) -> Dict[str, Any]:
        """Create HNSW index if missing"""
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
            
            # Create HNSW index using IRIS syntax
            create_query = f"""
                CREATE INDEX {index_name} ON RAG.{table_name} ({column_name})
                USING HNSW
            """
            
            self.cursor.execute(create_query)
            
            result['index_created'] = True
            result['index_name'] = index_name
            
            logger.info(f"Successfully created HNSW index {index_name}")
                
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"Error creating HNSW index on {table_name}.{column_name}: {e}")
        
        return result
    
    def run_comprehensive_verification(self) -> Dict[str, Any]:
        """Run comprehensive HNSW verification"""
        logger.info("Starting comprehensive HNSW verification")
        
        # Tables and their embedding columns to check
        tables_to_check = [
            ('SourceDocuments_V2', 'embedding'),
            ('DocumentTokenEmbeddings', 'token_embedding'),
            ('DocumentChunks', 'embedding'),
            ('KnowledgeGraphNodes', 'embedding')
        ]
        
        verification_results = {
            'timestamp': time.strftime('%Y-%m-%d %H:%M:%S'),
            'tables': {},
            'overall_status': 'UNKNOWN',
            'recommendations': []
        }
        
        all_indexes_ok = True
        all_searches_ok = True
        
        for table_name, column_name in tables_to_check:
            logger.info(f"\n=== Checking {table_name}.{column_name} ===")
            
            table_results = {
                'stats': self.get_table_stats(table_name),
                'index_check': None,
                'performance_test': None,
                'index_creation': None
            }
            
            # Only proceed if table exists and has data
            if table_results['stats']['exists'] and table_results['stats']['row_count'] > 0:
                
                # Check HNSW index
                table_results['index_check'] = self.check_hnsw_index(table_name, column_name)
                
                # If no HNSW index, try to create one
                if not table_results['index_check']['index_exists']:
                    all_indexes_ok = False
                    table_results['index_creation'] = self.create_hnsw_index(table_name, column_name)
                    
                    # Re-check after creation
                    if table_results['index_creation']['index_created']:
                        table_results['index_check'] = self.check_hnsw_index(table_name, column_name)
                
                # Test vector search performance
                if table_results['stats']['embedding_count'] > 0:
                    table_results['performance_test'] = self.test_vector_search_performance(table_name, column_name)
                    
                    if not table_results['performance_test']['search_successful']:
                        all_searches_ok = False
            
            verification_results['tables'][table_name] = table_results
        
        # Determine overall status
        if all_indexes_ok and all_searches_ok:
            verification_results['overall_status'] = 'READY'
            verification_results['recommendations'].append("✅ All HNSW indexes are present and working correctly")
            verification_results['recommendations'].append("✅ Vector search performance is good")
            verification_results['recommendations'].append("✅ Safe to resume large-scale ingestion")
        elif all_searches_ok:
            verification_results['overall_status'] = 'READY_WITH_FIXES'
            verification_results['recommendations'].append("✅ HNSW indexes have been created/fixed")
            verification_results['recommendations'].append("✅ Vector search performance is good")
            verification_results['recommendations'].append("✅ Safe to resume large-scale ingestion")
        else:
            verification_results['overall_status'] = 'NOT_READY'
            verification_results['recommendations'].append("❌ HNSW indexes or vector search have issues")
            verification_results['recommendations'].append("❌ Do NOT resume large-scale ingestion until fixed")
        
        return verification_results
    
    def print_results(self, results: Dict[str, Any]):
        """Print verification results in a readable format"""
        print("\n" + "="*80)
        print("HNSW INDEX VERIFICATION REPORT")
        print("="*80)
        print(f"Timestamp: {results['timestamp']}")
        print(f"Overall Status: {results['overall_status']}")
        print()
        
        for table_name, table_data in results['tables'].items():
            print(f"\n--- {table_name} ---")
            
            # Table stats
            stats = table_data['stats']
            if stats['exists']:
                print(f"  Table exists: ✅")
                print(f"  Row count: {stats['row_count']:,}")
                print(f"  Embedding count: {stats.get('embedding_count', 0):,}")
                if 'embedding_column' in stats:
                    print(f"  Embedding column: {stats['embedding_column']}")
            else:
                print(f"  Table exists: ❌")
                if stats['error']:
                    print(f"  Error: {stats['error']}")
                continue
            
            # Index check
            index_check = table_data['index_check']
            if index_check:
                if index_check['index_exists']:
                    print(f"  HNSW Index: ✅ {index_check['index_name']} ({index_check['index_type']})")
                else:
                    print(f"  HNSW Index: ❌ Not found")
                    if index_check['error']:
                        print(f"  Index Error: {index_check['error']}")
            
            # Index creation
            index_creation = table_data['index_creation']
            if index_creation:
                if index_creation['index_created']:
                    print(f"  Index Creation: ✅ Created {index_creation['index_name']}")
                else:
                    print(f"  Index Creation: ❌ Failed")
                    if index_creation['error']:
                        print(f"  Creation Error: {index_creation['error']}")
            
            # Performance test
            perf_test = table_data['performance_test']
            if perf_test:
                if perf_test['search_successful']:
                    print(f"  Vector Search: ✅ {perf_test['execution_time']:.4f}s ({perf_test['results_count']} results)")
                else:
                    print(f"  Vector Search: ❌ Failed")
                    if perf_test['error']:
                        print(f"  Search Error: {perf_test['error']}")
        
        print("\n" + "-"*80)
        print("RECOMMENDATIONS:")
        for rec in results['recommendations']:
            print(f"  {rec}")
        print("-"*80)

def main():
    """Main function"""
    print("HNSW Index Verification Starting...")
    
    verifier = HNSWVerifier()
    results = verifier.run_comprehensive_verification()
    verifier.print_results(results)
    
    # Return appropriate exit code
    if results['overall_status'] in ['READY', 'READY_WITH_FIXES']:
        print(f"\n🎉 VERIFICATION SUCCESSFUL: {results['overall_status']}")
        return 0
    else:
        print(f"\n❌ VERIFICATION FAILED: {results['overall_status']}")
        return 1

if __name__ == "__main__":
    sys.exit(main())