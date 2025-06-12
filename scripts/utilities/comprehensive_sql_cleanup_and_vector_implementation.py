#!/usr/bin/env python3
"""
Comprehensive SQL Cleanup and Vector Implementation Script

This script examines all SQL files, cleans them up, and attempts to implement
actual working HNSW vector indexing in IRIS.
"""

import os
import sys
import json
import time
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from common.iris_connector import get_iris_connection

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('sql_cleanup_and_vector_implementation.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class SQLAnalyzer:
    """Analyzes SQL files and identifies issues"""
    
    def __init__(self):
        self.sql_files = []
        self.issues = {}
        self.recommendations = {}
    
    def find_sql_files(self, root_dir: Path) -> List[Path]:
        """Find all SQL files in the repository"""
        sql_files = []
        for sql_file in root_dir.rglob("*.sql"):
            if '.venv' not in str(sql_file) and '__pycache__' not in str(sql_file):
                sql_files.append(sql_file)
        return sql_files
    
    def analyze_sql_file(self, file_path: Path) -> Dict[str, Any]:
        """Analyze a single SQL file for issues"""
        issues = []
        recommendations = []
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Check for common issues
            if 'VECTOR(FLOAT,' in content:
                issues.append("Uses VECTOR data type which may not work in Community Edition")
                recommendations.append("Consider using VARCHAR with TO_VECTOR() conversion")
            
            if 'HNSW(' in content:
                issues.append("Contains HNSW index creation which may fail")
                recommendations.append("Test HNSW creation conditionally")
            
            if 'LIMIT ' in content and 'TOP ' not in content:
                issues.append("Uses LIMIT syntax instead of IRIS-compatible TOP")
                recommendations.append("Replace LIMIT with TOP for IRIS compatibility")
            
            if 'CREATE OR REPLACE FUNCTION' in content:
                issues.append("Uses generic SQL function syntax")
                recommendations.append("May need ObjectScript implementation for IRIS")
            
            return {
                'file_path': str(file_path),
                'size': len(content),
                'lines': len(content.split('\n')),
                'issues': issues,
                'recommendations': recommendations,
                'content_preview': content[:500] + '...' if len(content) > 500 else content
            }
            
        except Exception as e:
            return {
                'file_path': str(file_path),
                'error': str(e),
                'issues': [f"Failed to read file: {e}"],
                'recommendations': ["Check file permissions and encoding"]
            }
    
    def analyze_all_files(self, root_dir: Path) -> Dict[str, Any]:
        """Analyze all SQL files in the repository"""
        self.sql_files = self.find_sql_files(root_dir)
        analysis_results = {}
        
        logger.info(f"Found {len(self.sql_files)} SQL files to analyze")
        
        for sql_file in self.sql_files:
            logger.info(f"Analyzing {sql_file}")
            analysis_results[str(sql_file)] = self.analyze_sql_file(sql_file)
        
        return analysis_results

class VectorCapabilityTester:
    """Tests actual IRIS vector capabilities"""
    
    def __init__(self):
        self.connection = None
        self.test_results = {}
    
    def connect_to_iris(self) -> bool:
        """Connect to IRIS database"""
        try:
            self.connection = get_iris_connection()
            logger.info("Successfully connected to IRIS")
            return True
        except Exception as e:
            logger.error(f"Failed to connect to IRIS: {e}")
            return False
    
    def test_vector_data_type(self) -> Dict[str, Any]:
        """Test if VECTOR data type is supported"""
        test_name = "vector_data_type"
        logger.info(f"Testing {test_name}")
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("DROP TABLE IF EXISTS test_vector_table")
                cursor.execute("""
                    CREATE TABLE test_vector_table (
                        id INTEGER PRIMARY KEY,
                        embedding VECTOR(FLOAT, 768)
                    )
                """)
                
                cursor.execute("""
                    SELECT COLUMN_NAME, DATA_TYPE, CHARACTER_MAXIMUM_LENGTH
                    FROM INFORMATION_SCHEMA.COLUMNS 
                    WHERE TABLE_NAME = 'test_vector_table' 
                    AND COLUMN_NAME = 'embedding'
                """)
                
                result = cursor.fetchone()
                if result:
                    actual_type = result[1]
                    return {
                        'test': test_name,
                        'success': 'VECTOR' in actual_type.upper(),
                        'actual_type': actual_type,
                        'message': f"VECTOR data type {'supported' if 'VECTOR' in actual_type.upper() else 'falls back to ' + actual_type}"
                    }
                else:
                    return {'test': test_name, 'success': False, 'message': "Could not retrieve column information"}
                    
        except Exception as e:
            return {'test': test_name, 'success': False, 'error': str(e), 'message': f"VECTOR data type test failed: {e}"}
        finally:
            try:
                with self.connection.cursor() as cursor:
                    cursor.execute("DROP TABLE IF EXISTS test_vector_table")
            except:
                pass
    
    def test_to_vector_function(self) -> Dict[str, Any]:
        """Test TO_VECTOR function"""
        test_name = "to_vector_function"
        logger.info(f"Testing {test_name}")
        
        try:
            with self.connection.cursor() as cursor:
                test_embedding = "0.1,0.2,0.3,0.4,0.5"
                cursor.execute(f"SELECT TO_VECTOR('{test_embedding}', 'FLOAT', 5) AS vector_result")
                result = cursor.fetchone()
                
                if result:
                    return {
                        'test': test_name,
                        'success': True,
                        'result': str(result[0]),
                        'message': "TO_VECTOR function works"
                    }
                else:
                    return {'test': test_name, 'success': False, 'message': "TO_VECTOR returned no result"}
                    
        except Exception as e:
            return {'test': test_name, 'success': False, 'error': str(e), 'message': f"TO_VECTOR function test failed: {e}"}
    
    def test_vector_cosine_function(self) -> Dict[str, Any]:
        """Test VECTOR_COSINE function"""
        test_name = "vector_cosine_function"
        logger.info(f"Testing {test_name}")
        
        try:
            with self.connection.cursor() as cursor:
                embedding1 = "0.1,0.2,0.3,0.4,0.5"
                embedding2 = "0.2,0.3,0.4,0.5,0.6"
                
                cursor.execute(f"""
                    SELECT VECTOR_COSINE(
                        TO_VECTOR('{embedding1}', 'FLOAT', 5),
                        TO_VECTOR('{embedding2}', 'FLOAT', 5)
                    ) AS cosine_similarity
                """)
                
                result = cursor.fetchone()
                if result:
                    similarity = float(result[0])
                    return {
                        'test': test_name,
                        'success': True,
                        'similarity': similarity,
                        'message': f"VECTOR_COSINE works, similarity: {similarity:.4f}"
                    }
                else:
                    return {'test': test_name, 'success': False, 'message': "VECTOR_COSINE returned no result"}
                    
        except Exception as e:
            return {'test': test_name, 'success': False, 'error': str(e), 'message': f"VECTOR_COSINE function test failed: {e}"}
    
    def test_hnsw_index_creation(self) -> Dict[str, Any]:
        """Test HNSW index creation"""
        test_name = "hnsw_index_creation"
        logger.info(f"Testing {test_name}")
        
        try:
            with self.connection.cursor() as cursor:
                cursor.execute("DROP TABLE IF EXISTS test_hnsw_table")
                cursor.execute("""
                    CREATE TABLE test_hnsw_table (
                        id INTEGER PRIMARY KEY,
                        embedding_str VARCHAR(30000),
                        embedding_vector VECTOR(FLOAT, 768) COMPUTECODE {
                            if ({embedding_str} '= "") {
                                set {embedding_vector} = $$$TO_VECTOR({embedding_str}, 'FLOAT', 768)
                            } else {
                                set {embedding_vector} = ""
                            }
                        } CALCULATED
                    )
                """)
                
                try:
                    cursor.execute("""
                        CREATE INDEX idx_test_hnsw_embedding
                        ON test_hnsw_table (embedding_vector)
                        AS HNSW(M=16, efConstruction=200, Distance='COSINE')
                    """)
                    return {'test': test_name, 'success': True, 'message': "HNSW index created successfully"}
                    
                except Exception as hnsw_error:
                    return {
                        'test': test_name,
                        'success': False,
                        'error': str(hnsw_error),
                        'message': f"HNSW index creation failed: {hnsw_error}"
                    }
                        
        except Exception as e:
            return {'test': test_name, 'success': False, 'error': str(e), 'message': f"HNSW test setup failed: {e}"}
        finally:
            try:
                with self.connection.cursor() as cursor:
                    cursor.execute("DROP TABLE IF EXISTS test_hnsw_table")
            except:
                pass
    
    def run_all_tests(self) -> Dict[str, Any]:
        """Run all vector capability tests"""
        if not self.connect_to_iris():
            return {'error': 'Could not connect to IRIS'}
        
        tests = [
            self.test_vector_data_type,
            self.test_to_vector_function,
            self.test_vector_cosine_function,
            self.test_hnsw_index_creation
        ]
        
        results = {}
        for test_func in tests:
            try:
                result = test_func()
                results[result['test']] = result
                logger.info(f"Test {result['test']}: {'PASS' if result['success'] else 'FAIL'}")
            except Exception as e:
                test_name = test_func.__name__
                results[test_name] = {
                    'test': test_name,
                    'success': False,
                    'error': str(e),
                    'message': f"Test execution failed: {e}"
                }
                logger.error(f"Test {test_name} failed with exception: {e}")
        
        return results

class WorkingVectorImplementation:
    """Implements working vector operations based on test results"""
    
    def __init__(self, test_results: Dict[str, Any]):
        self.test_results = test_results
        self.connection = None
    
    def connect_to_iris(self) -> bool:
        """Connect to IRIS database"""
        try:
            self.connection = get_iris_connection()
            return True
        except Exception as e:
            logger.error(f"Failed to connect to IRIS: {e}")
            return False
    
    def create_optimized_schema(self) -> Dict[str, Any]:
        """Create optimized schema based on what actually works"""
        if not self.connect_to_iris():
            return {'success': False, 'error': 'Could not connect to IRIS'}
        
        schema_sql = self.generate_working_schema()
        
        try:
            with self.connection.cursor() as cursor:
                for statement in schema_sql.split(';'):
                    statement = statement.strip()
                    if statement:
                        cursor.execute(statement)
            
            return {
                'success': True,
                'message': 'Optimized schema created successfully',
                'schema': schema_sql
            }
            
        except Exception as e:
            return {
                'success': False,
                'error': str(e),
                'message': f'Schema creation failed: {e}'
            }
    
    def generate_working_schema(self) -> str:
        """Generate schema SQL based on test results"""
        vector_works = self.test_results.get('vector_data_type', {}).get('success', False)
        hnsw_works = self.test_results.get('hnsw_index_creation', {}).get('success', False)
        
        schema_parts = []
        
        # Drop existing schema
        schema_parts.append("""
-- Drop existing schema for clean slate
DROP TABLE IF EXISTS RAG_HNSW.DocumentTokenEmbeddings CASCADE;
DROP TABLE IF EXISTS RAG_HNSW.KnowledgeGraphEdges CASCADE; 
DROP TABLE IF EXISTS RAG_HNSW.KnowledgeGraphNodes CASCADE;
DROP TABLE IF EXISTS RAG_HNSW.SourceDocuments CASCADE;
DROP SCHEMA IF EXISTS RAG_HNSW CASCADE;

-- Create optimized schema
CREATE SCHEMA RAG_HNSW;
""")
        
        # Create SourceDocuments table
        if vector_works:
            schema_parts.append("""
-- SourceDocuments with VECTOR support
CREATE TABLE RAG_HNSW.SourceDocuments (
    doc_id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(500),
    text_content LONGVARCHAR,
    abstract LONGVARCHAR,
    authors LONGVARCHAR,
    keywords LONGVARCHAR,
    embedding_str VARCHAR(60000) NULL,
    embedding_vector VECTOR(FLOAT, 768) COMPUTECODE {
        if ({embedding_str} '= "") {
            set {embedding_vector} = $$$TO_VECTOR({embedding_str}, 'FLOAT', 768)
        } else {
            set {embedding_vector} = ""
        }
    } CALCULATED
);
""")
        else:
            schema_parts.append("""
-- SourceDocuments with VARCHAR embeddings (fallback)
CREATE TABLE RAG_HNSW.SourceDocuments (
    doc_id VARCHAR(255) PRIMARY KEY,
    title VARCHAR(500),
    text_content LONGVARCHAR,
    abstract LONGVARCHAR,
    authors LONGVARCHAR,
    keywords LONGVARCHAR,
    embedding VARCHAR(60000) NULL
);
""")
        
        # Add HNSW indexes if supported
        if hnsw_works and vector_works:
            schema_parts.append("""
-- HNSW indexes (supported)
CREATE INDEX idx_hnsw_source_embeddings
ON RAG_HNSW.SourceDocuments (embedding_vector)
AS HNSW(M=16, efConstruction=200, Distance='COSINE');
""")
        
        return '\n'.join(schema_parts)

def main():
    """Main execution function"""
    logger.info("Starting comprehensive SQL cleanup and vector implementation")
    
    # Initialize components
    sql_analyzer = SQLAnalyzer()
    vector_tester = VectorCapabilityTester()
    
    # Step 1: Analyze all SQL files
    logger.info("Step 1: Analyzing SQL files")
    project_root = Path(__file__).parent.parent
    sql_analysis = sql_analyzer.analyze_all_files(project_root)
    
    # Step 2: Test vector capabilities
    logger.info("Step 2: Testing IRIS vector capabilities")
    vector_test_results = vector_tester.run_all_tests()
    
    # Step 3: Implement working vector solution
    logger.info("Step 3: Implementing working vector solution")
    vector_impl = WorkingVectorImplementation(vector_test_results)
    schema_result = vector_impl.create_optimized_schema()
    
    # Compile final report
    final_report = {
        'timestamp': datetime.now().isoformat(),
        'sql_analysis': {
            'files_analyzed': len(sql_analysis),
            'files': sql_analysis
        },
        'vector_capabilities': vector_test_results,
        'schema_implementation': schema_result,
        'summary': generate_summary(sql_analysis, vector_test_results, schema_result)
    }
    
    # Save report
    report_file = f"sql_cleanup_vector_implementation_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(report_file, 'w') as f:
        json.dump(final_report, f, indent=2, default=str)
    
    logger.info(f"Report saved to {report_file}")
    
    # Print summary
    print("\n" + "="*80)
    print("SQL CLEANUP AND VECTOR IMPLEMENTATION SUMMARY")
    print("="*80)
    print(final_report['summary'])
    print("="*80)
    
    return final_report

def generate_summary(sql_analysis: Dict, vector_tests: Dict, schema_result: Dict) -> str:
    """Generate a summary of all results"""
    
    # SQL Analysis Summary
    total_files = len(sql_analysis)
    files_with_issues = sum(1 for analysis in sql_analysis.values() if analysis.get('issues', []))
    
    # Vector Test Summary
    tests_passed = sum(1 for test in vector_tests.values() if test.get('success', False))
    total_tests = len(vector_tests)
    
    summary = f"""
📊 SQL FILE ANALYSIS:
- Total SQL files analyzed: {total_files}
- Files with issues: {files_with_issues}
- Common issues: VECTOR type limitations, HNSW index challenges, syntax compatibility

🧪 VECTOR CAPABILITY TESTS:
- Tests passed: {tests_passed}/{total_tests}
- VECTOR data type: {'✅ SUPPORTED' if vector_tests.get('vector_data_type', {}).get('success') else '❌ NOT SUPPORTED'}
- TO_VECTOR function: {'✅ WORKS' if vector_tests.get('to_vector_function', {}).get('success') else '❌ FAILS'}
- VECTOR_COSINE function: {'✅ WORKS' if vector_tests.get('vector_cosine_function', {}).get('success') else '❌ FAILS'}
- HNSW indexing: {'✅ SUPPORTED' if vector_tests.get('hnsw_index_creation', {}).get('success') else '❌ NOT SUPPORTED'}

🏗️ SCHEMA IMPLEMENTATION:
- Schema creation: {'✅ SUCCESS' if schema_result.get('success') else '❌ FAILED'}
- Approach: {'Native VECTOR with HNSW' if vector_tests.get('hnsw_index_creation', {}).get('success') else 'VARCHAR with TO_VECTOR fallback'}

🎯 RECOMMENDATIONS:
1. {'Use native VECTOR types with HNSW indexing' if vector_tests.get('hnsw_index_creation', {}).get('success') else 'Use VARCHAR storage with TO_VECTOR() in queries'}
2. {'Optimize with computed columns' if vector_tests.get('vector_data_type', {}).get('success') else 'Implement application-level vector operations'}
3. Clean up SQL files to remove non-functional code
4. Implement proper error handling for vector operations
5. Consider IRIS Enterprise Edition for full vector support
"""
    
    return summary

if __name__ == "__main__":
    main()