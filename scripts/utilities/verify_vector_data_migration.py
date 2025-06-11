#!/usr/bin/env python3
"""
Vector Data Migration Verification Script

This script verifies that the vector data migration from VECTOR(FLOAT) to VECTOR(FLOAT)
was completed successfully. It checks:
- Database schema correctness
- Data integrity
- Vector operations functionality
- End-to-end RAG pipeline compatibility
"""

import os
import sys
import json
import logging
import argparse
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any

# Add project root to path for imports
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from common.iris_connector import get_iris_connection
    IRIS_CONNECTOR_AVAILABLE = True
except ImportError:
    IRIS_CONNECTOR_AVAILABLE = False
    print("Warning: IRIS connector not available. Database operations will be limited.")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: NumPy not available. Some verification tests will be limited.")

class MigrationVerifier:
    """Comprehensive verification of vector data migration"""
    
    def __init__(self, verbose: bool = False):
        self.verbose = verbose
        self.connection = None
        self.verification_results = {
            'start_time': datetime.now().isoformat(),
            'schema_checks': {},
            'data_integrity_checks': {},
            'functionality_tests': {},
            'performance_tests': {},
            'errors': [],
            'warnings': []
        }
        
        # Setup logging
        log_level = logging.DEBUG if verbose else logging.INFO
        logging.basicConfig(
            level=log_level,
            format='%(levelname)s: %(message)s'
        )
        self.logger = logging.getLogger(__name__)
        
        # Define expected tables and their vector columns
        self.vector_tables = {
            'RAG.SourceDocuments': {
                'embedding': {'dimension': 384, 'type': 'VECTOR(FLOAT)'}
            },
            'RAG.DocumentChunks': {
                'chunk_embedding': {'dimension': 384, 'type': 'VECTOR(FLOAT)'}
            },
            'RAG.Entities': {
                'embedding': {'dimension': 384, 'type': 'VECTOR(FLOAT)'}
            },
            'RAG.KnowledgeGraphNodes': {
                'embedding': {'dimension': 384, 'type': 'VECTOR(FLOAT)'}
            },
            'RAG.DocumentTokenEmbeddings': {
                'token_embedding': {'dimension': 128, 'type': 'VECTOR(FLOAT)'}
            }
        }
    
    def connect_to_database(self) -> bool:
        """Establish database connection"""
        if not IRIS_CONNECTOR_AVAILABLE:
            self.logger.error("IRIS connector not available")
            return False
        
        try:
            self.connection = get_iris_connection()
            self.logger.info("Successfully connected to IRIS database")
            return True
        except Exception as e:
            self.logger.error(f"Failed to connect to database: {e}")
            return False
    
    def check_table_exists(self, table_name: str) -> bool:
        """Check if a table exists in the database"""
        try:
            cursor = self.connection.cursor()
            schema, table = table_name.split('.')
            sql = """
            SELECT COUNT(*) as table_count
            FROM INFORMATION_SCHEMA.TABLES 
            WHERE TABLE_NAME = ? AND TABLE_SCHEMA = ?
            """
            cursor.execute(sql, (table, schema))
            result = cursor.fetchone()
            exists = result[0] > 0
            
            self.logger.debug(f"Table {table_name} exists: {exists}")
            return exists
            
        except Exception as e:
            self.logger.warning(f"Could not check if table {table_name} exists: {e}")
            return False
    
    def verify_column_type(self, table_name: str, column_name: str, expected_type: str) -> bool:
        """Verify that a column has the expected data type"""
        try:
            cursor = self.connection.cursor()
            schema, table = table_name.split('.')
            sql = """
            SELECT DATA_TYPE, CHARACTER_MAXIMUM_LENGTH, NUMERIC_PRECISION
            FROM INFORMATION_SCHEMA.COLUMNS 
            WHERE TABLE_NAME = ? AND COLUMN_NAME = ? AND TABLE_SCHEMA = ?
            """
            cursor.execute(sql, (table, column_name, schema))
            result = cursor.fetchone()
            
            if result:
                data_type = result[0]
                max_length = result[1]
                precision = result[2]
                
                # Check if it's a VECTOR type
                if 'VECTOR' in expected_type.upper():
                    # For VECTOR types, check if the data type indicates vector storage
                    is_vector_type = (
                        'VECTOR' in data_type.upper() or 
                        'LONGVARBINARY' in data_type.upper() or  # IRIS might store vectors as binary
                        'VARBINARY' in data_type.upper()
                    )
                    
                    if is_vector_type:
                        self.logger.info(f"✓ {table_name}.{column_name} has vector-compatible type: {data_type}")
                        return True
                    else:
                        self.logger.warning(f"✗ {table_name}.{column_name} type {data_type} may not be vector-compatible")
                        return False
                else:
                    # For non-vector types, do exact match
                    type_matches = expected_type.upper() in data_type.upper()
                    if type_matches:
                        self.logger.info(f"✓ {table_name}.{column_name} has correct type: {data_type}")
                        return True
                    else:
                        self.logger.warning(f"✗ {table_name}.{column_name} type mismatch: {data_type} vs {expected_type}")
                        return False
            else:
                self.logger.error(f"✗ Column {table_name}.{column_name} not found")
                return False
                
        except Exception as e:
            self.logger.error(f"Error checking column type for {table_name}.{column_name}: {e}")
            return False
    
    def verify_vector_data_integrity(self, table_name: str, column_name: str, expected_dimension: int) -> Dict[str, Any]:
        """Verify vector data integrity and dimensions"""
        try:
            cursor = self.connection.cursor()
            
            # Get basic statistics
            sql_count = f"SELECT COUNT(*) FROM {table_name} WHERE {column_name} IS NOT NULL"
            cursor.execute(sql_count)
            vector_count = cursor.fetchone()[0]
            
            sql_total = f"SELECT COUNT(*) FROM {table_name}"
            cursor.execute(sql_total)
            total_count = cursor.fetchone()[0]
            
            # Try to get a sample vector for dimension verification
            sql_sample = f"SELECT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL LIMIT 1"
            cursor.execute(sql_sample)
            sample_result = cursor.fetchone()
            
            result = {
                'total_rows': total_count,
                'vector_rows': vector_count,
                'null_rows': total_count - vector_count,
                'sample_available': sample_result is not None,
                'dimension_verified': False,
                'actual_dimension': None
            }
            
            if sample_result and sample_result[0]:
                # Try to verify dimension (this is database-specific)
                try:
                    vector_data = sample_result[0]
                    # If it's a string representation, try to parse it
                    if isinstance(vector_data, str):
                        if '[' in vector_data and ']' in vector_data:
                            elements = vector_data.strip('[]').split(',')
                            actual_dimension = len(elements)
                            result['actual_dimension'] = actual_dimension
                            result['dimension_verified'] = actual_dimension == expected_dimension
                        else:
                            # Try comma-separated format
                            elements = vector_data.split(',')
                            if len(elements) > 1:
                                actual_dimension = len(elements)
                                result['actual_dimension'] = actual_dimension
                                result['dimension_verified'] = actual_dimension == expected_dimension
                    
                    self.logger.info(f"Vector data sample for {table_name}.{column_name}: {str(vector_data)[:100]}...")
                    
                except Exception as e:
                    self.logger.debug(f"Could not parse vector dimension: {e}")
            
            self.logger.info(f"Data integrity for {table_name}.{column_name}: {vector_count}/{total_count} rows have vectors")
            return result
            
        except Exception as e:
            self.logger.error(f"Error verifying data integrity for {table_name}.{column_name}: {e}")
            return {
                'error': str(e),
                'total_rows': 0,
                'vector_rows': 0,
                'null_rows': 0,
                'sample_available': False,
                'dimension_verified': False
            }
    
    def test_vector_operations(self, table_name: str, column_name: str) -> bool:
        """Test basic vector operations to ensure functionality"""
        try:
            cursor = self.connection.cursor()
            
            # Test 1: Basic vector selection
            sql_select = f"SELECT {column_name} FROM {table_name} WHERE {column_name} IS NOT NULL LIMIT 1"
            cursor.execute(sql_select)
            result = cursor.fetchone()
            
            if not result:
                self.logger.warning(f"No vector data found in {table_name}.{column_name}")
                return False
            
            self.logger.info(f"✓ Basic vector selection works for {table_name}.{column_name}")
            
            # Test 2: Vector similarity (if we have at least 2 vectors)
            sql_count = f"SELECT COUNT(*) FROM {table_name} WHERE {column_name} IS NOT NULL"
            cursor.execute(sql_count)
            vector_count = cursor.fetchone()[0]
            
            if vector_count >= 2:
                try:
                    # Test VECTOR_COSINE function
                    sql_similarity = f"""
                    SELECT VECTOR_COSINE(a.{column_name}, b.{column_name}) as similarity
                    FROM {table_name} a, {table_name} b
                    WHERE a.{column_name} IS NOT NULL 
                    AND b.{column_name} IS NOT NULL
                    AND a.ROWID != b.ROWID
                    LIMIT 1
                    """
                    cursor.execute(sql_similarity)
                    similarity_result = cursor.fetchone()
                    
                    if similarity_result:
                        similarity = similarity_result[0]
                        self.logger.info(f"✓ Vector similarity calculation works: {similarity}")
                        return True
                    else:
                        self.logger.warning(f"Vector similarity calculation returned no results")
                        return False
                        
                except Exception as e:
                    self.logger.warning(f"Vector similarity test failed: {e}")
                    return False
            else:
                self.logger.info(f"✓ Basic operations work (insufficient data for similarity test)")
                return True
            
        except Exception as e:
            self.logger.error(f"Vector operations test failed for {table_name}.{column_name}: {e}")
            return False
    
    def test_to_vector_function(self) -> bool:
        """Test TO_VECTOR function with FLOAT parameter"""
        try:
            cursor = self.connection.cursor()
            
            # Test TO_VECTOR with FLOAT
            test_vector = "0.1,0.2,0.3"
            sql_test = "SELECT TO_VECTOR(?, 'FLOAT', 3) as test_vector"
            cursor.execute(sql_test, (test_vector,))
            result = cursor.fetchone()
            
            if result:
                self.logger.info("✓ TO_VECTOR function works with 'FLOAT' parameter")
                return True
            else:
                self.logger.error("✗ TO_VECTOR function failed")
                return False
                
        except Exception as e:
            self.logger.error(f"TO_VECTOR function test failed: {e}")
            return False
    
    def run_comprehensive_verification(self) -> bool:
        """Run all verification tests"""
        self.logger.info("Starting comprehensive vector migration verification")
        
        if not self.connect_to_database():
            return False
        
        overall_success = True
        
        try:
            # Test 1: Schema verification
            self.logger.info("=== Schema Verification ===")
            schema_success = True
            
            for table_name, columns in self.vector_tables.items():
                if not self.check_table_exists(table_name):
                    self.logger.warning(f"Table {table_name} does not exist, skipping")
                    continue
                
                for column_name, specs in columns.items():
                    expected_type = specs['type']
                    type_correct = self.verify_column_type(table_name, column_name, expected_type)
                    
                    self.verification_results['schema_checks'][f"{table_name}.{column_name}"] = {
                        'type_correct': type_correct,
                        'expected_type': expected_type
                    }
                    
                    if not type_correct:
                        schema_success = False
            
            # Test 2: Data integrity verification
            self.logger.info("=== Data Integrity Verification ===")
            data_success = True
            
            for table_name, columns in self.vector_tables.items():
                if not self.check_table_exists(table_name):
                    continue
                
                for column_name, specs in columns.items():
                    expected_dimension = specs['dimension']
                    integrity_result = self.verify_vector_data_integrity(table_name, column_name, expected_dimension)
                    
                    self.verification_results['data_integrity_checks'][f"{table_name}.{column_name}"] = integrity_result
                    
                    if 'error' in integrity_result:
                        data_success = False
            
            # Test 3: Functionality tests
            self.logger.info("=== Functionality Tests ===")
            func_success = True
            
            # Test TO_VECTOR function
            to_vector_works = self.test_to_vector_function()
            self.verification_results['functionality_tests']['to_vector_float'] = to_vector_works
            if not to_vector_works:
                func_success = False
            
            # Test vector operations on each table
            for table_name, columns in self.vector_tables.items():
                if not self.check_table_exists(table_name):
                    continue
                
                for column_name, specs in columns.items():
                    ops_work = self.test_vector_operations(table_name, column_name)
                    self.verification_results['functionality_tests'][f"{table_name}.{column_name}"] = ops_work
                    if not ops_work:
                        func_success = False
            
            # Overall assessment
            overall_success = schema_success and data_success and func_success
            
            # Generate summary
            self.logger.info("=== Verification Summary ===")
            self.logger.info(f"Schema verification: {'✓ PASSED' if schema_success else '✗ FAILED'}")
            self.logger.info(f"Data integrity: {'✓ PASSED' if data_success else '✗ FAILED'}")
            self.logger.info(f"Functionality tests: {'✓ PASSED' if func_success else '✗ FAILED'}")
            self.logger.info(f"Overall result: {'✓ MIGRATION SUCCESSFUL' if overall_success else '✗ MIGRATION ISSUES DETECTED'}")
            
        except Exception as e:
            self.logger.critical(f"Verification failed with critical error: {e}")
            overall_success = False
        
        finally:
            if self.connection:
                self.connection.close()
        
        # Save verification report
        self.verification_results['end_time'] = datetime.now().isoformat()
        self.verification_results['overall_success'] = overall_success
        
        report_file = f"verification_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
        with open(report_file, 'w') as f:
            json.dump(self.verification_results, f, indent=2)
        
        self.logger.info(f"Verification report saved: {report_file}")
        
        return overall_success
    
    def run_quick_check(self) -> bool:
        """Run a quick verification check"""
        self.logger.info("Running quick migration verification check")
        
        if not self.connect_to_database():
            return False
        
        try:
            # Quick test: Check if TO_VECTOR with FLOAT works
            cursor = self.connection.cursor()
            sql_test = "SELECT TO_VECTOR('0.1,0.2,0.3', 'FLOAT', 3) as test_vector"
            cursor.execute(sql_test)
            result = cursor.fetchone()
            
            if result:
                self.logger.info("✓ Quick check PASSED: TO_VECTOR with FLOAT works")
                return True
            else:
                self.logger.error("✗ Quick check FAILED: TO_VECTOR with FLOAT failed")
                return False
                
        except Exception as e:
            self.logger.error(f"Quick check failed: {e}")
            return False
        
        finally:
            if self.connection:
                self.connection.close()

def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(description="Verify vector data migration from DOUBLE to FLOAT")
    parser.add_argument('--verbose', '-v', action='store_true', help='Verbose output')
    parser.add_argument('--quick', '-q', action='store_true', help='Run quick check only')
    
    args = parser.parse_args()
    
    verifier = MigrationVerifier(verbose=args.verbose)
    
    if args.quick:
        success = verifier.run_quick_check()
    else:
        success = verifier.run_comprehensive_verification()
    
    if success:
        print("\n🎉 Vector migration verification PASSED!")
    else:
        print("\n❌ Vector migration verification FAILED!")
        print("Check the verification report for details.")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()