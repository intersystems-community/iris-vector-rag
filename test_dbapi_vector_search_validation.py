#!/usr/bin/env python3
"""
DBAPI Vector Search Validation Test

This test validates that vector search SQL patterns work correctly through DBAPI
with real data, ensuring that VECTOR_COSINE, TO_VECTOR, and other vector operations
function properly through the DBAPI interface.

This addresses the critical question: Do vector search SQL patterns work the same
way through DBAPI as they do through JDBC?
"""

import os
import sys
import json
import time
import logging
import traceback
from datetime import datetime
from typing import Dict, List, Any, Optional, Tuple

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.'))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(f'logs/dbapi_vector_search_validation_{int(time.time())}.log')
    ]
)
logger = logging.getLogger(__name__)

class DBAPIVectorSearchValidator:
    """Validates vector search functionality through DBAPI with real data"""
    
    def __init__(self):
        self.start_time = time.time()
        self.test_results = {}
        self.dbapi_connection = None
        self.jdbc_connection = None
        
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
        
        logger.info("=" * 80)
        logger.info("DBAPI VECTOR SEARCH VALIDATION TEST")
        logger.info("=" * 80)
        logger.info(f"Test started at: {datetime.now().isoformat()}")
    
    def get_dbapi_connection(self):
        """Get DBAPI connection"""
        try:
            import intersystems_iris.dbapi._DBAPI as irisdbapi
            
            host = os.environ.get("IRIS_HOST", "localhost")
            port = int(os.environ.get("IRIS_PORT", 1972))
            namespace = os.environ.get("IRIS_NAMESPACE", "USER")
            user = os.environ.get("IRIS_USER", "_SYSTEM")
            password = os.environ.get("IRIS_PASSWORD", "SYS")
            
            logger.info(f"Connecting to IRIS via DBAPI at {host}:{port}/{namespace} as {user}")
            conn = irisdbapi.connect(host, port, namespace, user, password)
            logger.info("✓ DBAPI connection established")
            return conn
            
        except Exception as e:
            logger.error(f"Failed to establish DBAPI connection: {e}")
            return None
    
    def get_jdbc_connection(self):
        """Get JDBC connection for comparison"""
        try:
            from common.iris_connector import get_iris_connection
            conn = get_iris_connection()
            logger.info("✓ JDBC connection established")
            return conn
        except Exception as e:
            logger.warning(f"Could not establish JDBC connection for comparison: {e}")
            return None
    
    def setup_test_data(self, conn):
        """Set up test data with real vector embeddings"""
        logger.info("Setting up test data with vector embeddings...")
        
        try:
            cursor = conn.cursor()
            
            # Create test table
            cursor.execute("DROP TABLE IF EXISTS TestVectorData")
            cursor.execute("""
                CREATE TABLE TestVectorData (
                    id INT PRIMARY KEY,
                    text_content VARCHAR(1000),
                    embedding VECTOR(FLOAT, 384)
                )
            """)
            
            # Insert test data with real embeddings
            test_data = [
                (1, "Cancer treatment research shows promising results", [0.1] * 384),
                (2, "Machine learning algorithms improve diagnosis", [0.2] * 384),
                (3, "Clinical trials demonstrate drug effectiveness", [0.3] * 384),
                (4, "Patient outcomes improve with new therapy", [0.4] * 384),
                (5, "Medical imaging advances detection accuracy", [0.5] * 384)
            ]
            
            for doc_id, text, embedding in test_data:
                embedding_str = f"[{','.join(map(str, embedding))}]"
                cursor.execute(
                    "INSERT INTO TestVectorData (id, text_content, embedding) VALUES (?, ?, TO_VECTOR(?))",
                    (doc_id, text, embedding_str)
                )
            
            # Create HNSW index
            try:
                cursor.execute("CREATE INDEX idx_test_vector ON TestVectorData (embedding) USING HNSW")
                logger.info("✓ HNSW index created successfully")
            except Exception as e:
                logger.warning(f"HNSW index creation failed (non-critical): {e}")
            
            cursor.close()
            logger.info("✓ Test data setup complete")
            return True
            
        except Exception as e:
            logger.error(f"Failed to setup test data: {e}")
            logger.error(traceback.format_exc())
            return False
    
    def test_vector_search_patterns(self, conn, connection_type: str) -> Dict[str, Any]:
        """Test various vector search SQL patterns"""
        logger.info(f"Testing vector search patterns via {connection_type}...")
        
        results = {
            'connection_type': connection_type,
            'tests': {},
            'overall_success': True
        }
        
        # Test query embedding
        query_embedding = [0.15] * 384
        query_vector_str = f"[{','.join(map(str, query_embedding))}]"
        
        test_patterns = [
            {
                'name': 'basic_vector_cosine',
                'sql': """
                    SELECT TOP 3 id, text_content,
                           VECTOR_COSINE(embedding, TO_VECTOR(?)) AS score
                    FROM TestVectorData
                    ORDER BY score DESC
                """,
                'params': [query_vector_str]
            },
            {
                'name': 'vector_cosine_with_threshold',
                'sql': """
                    SELECT id, text_content,
                           VECTOR_COSINE(embedding, TO_VECTOR(?)) AS score
                    FROM TestVectorData
                    WHERE VECTOR_COSINE(embedding, TO_VECTOR(?)) > 0.5
                    ORDER BY score DESC
                """,
                'params': [query_vector_str, query_vector_str]
            },
            {
                'name': 'to_vector_embedding_retrieval',
                'sql': """
                    SELECT TOP 2 id, text_content,
                           VECTOR_COSINE(TO_VECTOR(embedding), TO_VECTOR(?)) AS score
                    FROM TestVectorData
                    ORDER BY score DESC
                """,
                'params': [query_vector_str]
            },
            {
                'name': 'vector_with_text_filter',
                'sql': """
                    SELECT id, text_content,
                           VECTOR_COSINE(embedding, TO_VECTOR(?)) AS score
                    FROM TestVectorData
                    WHERE text_content LIKE '%treatment%'
                    ORDER BY score DESC
                """,
                'params': [query_vector_str]
            }
        ]
        
        for test in test_patterns:
            test_result = {
                'success': False,
                'error': None,
                'execution_time': 0,
                'result_count': 0,
                'sample_results': []
            }
            
            start_time = time.time()
            
            try:
                cursor = conn.cursor()
                cursor.execute(test['sql'], test['params'])
                rows = cursor.fetchall()
                
                test_result['success'] = True
                test_result['result_count'] = len(rows)
                test_result['sample_results'] = [
                    {'id': row[0], 'text': row[1], 'score': float(row[2])}
                    for row in rows[:2]  # Sample first 2 results
                ]
                
                logger.info(f"✓ {test['name']} ({connection_type}): {len(rows)} results")
                
                cursor.close()
                
            except Exception as e:
                test_result['error'] = str(e)
                results['overall_success'] = False
                logger.error(f"✗ {test['name']} ({connection_type}): {e}")
            
            test_result['execution_time'] = time.time() - start_time
            results['tests'][test['name']] = test_result
        
        return results
    
    def compare_results(self, dbapi_results: Dict, jdbc_results: Dict) -> Dict[str, Any]:
        """Compare DBAPI vs JDBC results"""
        logger.info("Comparing DBAPI vs JDBC results...")
        
        comparison = {
            'tests_compared': 0,
            'identical_results': 0,
            'different_results': 0,
            'dbapi_only_success': 0,
            'jdbc_only_success': 0,
            'both_failed': 0,
            'differences': []
        }
        
        if not jdbc_results:
            logger.warning("No JDBC results available for comparison")
            return comparison
        
        for test_name in dbapi_results['tests']:
            if test_name not in jdbc_results['tests']:
                continue
                
            comparison['tests_compared'] += 1
            dbapi_test = dbapi_results['tests'][test_name]
            jdbc_test = jdbc_results['tests'][test_name]
            
            # Compare success status
            if dbapi_test['success'] and jdbc_test['success']:
                # Compare result counts
                if dbapi_test['result_count'] == jdbc_test['result_count']:
                    comparison['identical_results'] += 1
                    logger.info(f"✓ {test_name}: Identical results ({dbapi_test['result_count']} rows)")
                else:
                    comparison['different_results'] += 1
                    comparison['differences'].append({
                        'test': test_name,
                        'issue': 'different_result_count',
                        'dbapi_count': dbapi_test['result_count'],
                        'jdbc_count': jdbc_test['result_count']
                    })
                    logger.warning(f"⚠ {test_name}: Different result counts - DBAPI: {dbapi_test['result_count']}, JDBC: {jdbc_test['result_count']}")
                    
            elif dbapi_test['success'] and not jdbc_test['success']:
                comparison['dbapi_only_success'] += 1
                comparison['differences'].append({
                    'test': test_name,
                    'issue': 'dbapi_success_jdbc_fail',
                    'jdbc_error': jdbc_test['error']
                })
                logger.warning(f"⚠ {test_name}: DBAPI succeeded, JDBC failed: {jdbc_test['error']}")
                
            elif not dbapi_test['success'] and jdbc_test['success']:
                comparison['jdbc_only_success'] += 1
                comparison['differences'].append({
                    'test': test_name,
                    'issue': 'jdbc_success_dbapi_fail',
                    'dbapi_error': dbapi_test['error']
                })
                logger.warning(f"⚠ {test_name}: JDBC succeeded, DBAPI failed: {dbapi_test['error']}")
                
            else:
                comparison['both_failed'] += 1
                logger.error(f"✗ {test_name}: Both DBAPI and JDBC failed")
        
        return comparison
    
    def cleanup_test_data(self, conn):
        """Clean up test data"""
        try:
            cursor = conn.cursor()
            cursor.execute("DROP TABLE IF EXISTS TestVectorData")
            cursor.close()
            logger.info("✓ Test data cleaned up")
        except Exception as e:
            logger.warning(f"Cleanup failed (non-critical): {e}")
    
    def generate_report(self, dbapi_results: Dict, jdbc_results: Dict, comparison: Dict) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        
        total_time = time.time() - self.start_time
        
        report = {
            'test_summary': {
                'test_timestamp': datetime.now().isoformat(),
                'total_execution_time': total_time,
                'dbapi_overall_success': dbapi_results.get('overall_success', False),
                'jdbc_available': jdbc_results is not None,
                'jdbc_overall_success': jdbc_results.get('overall_success', False) if jdbc_results else False
            },
            'dbapi_results': dbapi_results,
            'jdbc_results': jdbc_results,
            'comparison': comparison,
            'conclusions': [],
            'recommendations': []
        }
        
        # Generate conclusions
        if dbapi_results.get('overall_success', False):
            report['conclusions'].append("✅ DBAPI vector search functionality is working")
            
            if jdbc_results and comparison['tests_compared'] > 0:
                if comparison['identical_results'] == comparison['tests_compared']:
                    report['conclusions'].append("✅ DBAPI and JDBC produce identical results")
                    report['recommendations'].append("🎉 DBAPI is fully compatible with existing vector search patterns")
                elif comparison['different_results'] > 0:
                    report['conclusions'].append("⚠️ DBAPI and JDBC produce different results for some queries")
                    report['recommendations'].append("🔍 Investigate differences in vector search result handling")
                    
                if comparison['dbapi_only_success'] > 0:
                    report['conclusions'].append("✅ Some queries work better through DBAPI")
                    
                if comparison['jdbc_only_success'] > 0:
                    report['conclusions'].append("⚠️ Some queries work better through JDBC")
                    report['recommendations'].append("🔧 Review DBAPI-specific vector query patterns")
            else:
                report['recommendations'].append("✅ DBAPI vector search is functional - ready for production use")
        else:
            report['conclusions'].append("❌ DBAPI vector search has issues")
            report['recommendations'].append("🔧 Fix DBAPI vector search issues before production deployment")
        
        return report
    
    def save_results(self, report: Dict[str, Any]):
        """Save test results"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON report
        json_file = f"test_results/dbapi_vector_search_validation_{timestamp}.json"
        os.makedirs('test_results', exist_ok=True)
        
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✓ Results saved to {json_file}")
        
        # Save markdown summary
        md_file = f"test_results/dbapi_vector_search_validation_{timestamp}.md"
        self.generate_markdown_report(report, md_file)
        
        return json_file, md_file
    
    def generate_markdown_report(self, report: Dict[str, Any], filename: str):
        """Generate markdown report"""
        
        with open(filename, 'w') as f:
            f.write("# DBAPI Vector Search Validation Report\n\n")
            f.write(f"**Test Date:** {report['test_summary']['test_timestamp']}\n")
            f.write(f"**Total Execution Time:** {report['test_summary']['total_execution_time']:.2f} seconds\n\n")
            
            # Summary
            f.write("## Test Summary\n\n")
            f.write(f"- **DBAPI Success:** {'✅ YES' if report['test_summary']['dbapi_overall_success'] else '❌ NO'}\n")
            f.write(f"- **JDBC Available:** {'✅ YES' if report['test_summary']['jdbc_available'] else '❌ NO'}\n")
            if report['test_summary']['jdbc_available']:
                f.write(f"- **JDBC Success:** {'✅ YES' if report['test_summary']['jdbc_overall_success'] else '❌ NO'}\n")
            f.write("\n")
            
            # DBAPI Results
            f.write("## DBAPI Vector Search Results\n\n")
            if report['dbapi_results']:
                for test_name, test_result in report['dbapi_results']['tests'].items():
                    status = "✅ SUCCESS" if test_result['success'] else "❌ FAILED"
                    f.write(f"### {status} {test_name}\n")
                    f.write(f"- **Execution Time:** {test_result['execution_time']:.4f} seconds\n")
                    f.write(f"- **Result Count:** {test_result['result_count']}\n")
                    if test_result['error']:
                        f.write(f"- **Error:** {test_result['error']}\n")
                    f.write("\n")
            
            # Comparison
            if report['comparison']['tests_compared'] > 0:
                f.write("## DBAPI vs JDBC Comparison\n\n")
                comp = report['comparison']
                f.write(f"- **Tests Compared:** {comp['tests_compared']}\n")
                f.write(f"- **Identical Results:** {comp['identical_results']}\n")
                f.write(f"- **Different Results:** {comp['different_results']}\n")
                f.write(f"- **DBAPI Only Success:** {comp['dbapi_only_success']}\n")
                f.write(f"- **JDBC Only Success:** {comp['jdbc_only_success']}\n")
                f.write("\n")
            
            # Conclusions
            f.write("## Conclusions\n\n")
            for conclusion in report['conclusions']:
                f.write(f"- {conclusion}\n")
            f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            for rec in report['recommendations']:
                f.write(f"- {rec}\n")
        
        logger.info(f"✓ Markdown report saved to {filename}")
    
    def run_validation(self) -> bool:
        """Run the complete vector search validation"""
        
        try:
            # Step 1: Establish connections
            logger.info("Step 1: Establishing database connections...")
            self.dbapi_connection = self.get_dbapi_connection()
            
            if not self.dbapi_connection:
                logger.error("❌ Cannot proceed without DBAPI connection")
                return False
            
            self.jdbc_connection = self.get_jdbc_connection()
            
            # Step 2: Setup test data
            logger.info("Step 2: Setting up test data...")
            if not self.setup_test_data(self.dbapi_connection):
                logger.error("❌ Failed to setup test data")
                return False
            
            # Step 3: Test DBAPI vector search
            logger.info("Step 3: Testing DBAPI vector search patterns...")
            dbapi_results = self.test_vector_search_patterns(self.dbapi_connection, "DBAPI")
            
            # Step 4: Test JDBC vector search (if available)
            jdbc_results = None
            if self.jdbc_connection:
                logger.info("Step 4: Testing JDBC vector search patterns...")
                jdbc_results = self.test_vector_search_patterns(self.jdbc_connection, "JDBC")
            else:
                logger.warning("Step 4: Skipping JDBC tests (connection not available)")
            
            # Step 5: Compare results
            logger.info("Step 5: Comparing results...")
            comparison = self.compare_results(dbapi_results, jdbc_results)
            
            # Step 6: Generate report
            logger.info("Step 6: Generating comprehensive report...")
            report = self.generate_report(dbapi_results, jdbc_results, comparison)
            
            # Step 7: Save results
            logger.info("Step 7: Saving results...")
            json_file, md_file = self.save_results(report)
            
            # Step 8: Cleanup
            logger.info("Step 8: Cleaning up...")
            self.cleanup_test_data(self.dbapi_connection)
            
            # Final summary
            logger.info("\n" + "=" * 80)
            logger.info("VECTOR SEARCH VALIDATION COMPLETE")
            logger.info("=" * 80)
            
            dbapi_success = report['test_summary']['dbapi_overall_success']
            logger.info(f"DBAPI Vector Search: {'✅ SUCCESS' if dbapi_success else '❌ FAILED'}")
            
            if jdbc_results:
                jdbc_success = report['test_summary']['jdbc_overall_success']
                logger.info(f"JDBC Vector Search: {'✅ SUCCESS' if jdbc_success else '❌ FAILED'}")
                
                if comparison['tests_compared'] > 0:
                    identical = comparison['identical_results']
                    total = comparison['tests_compared']
                    logger.info(f"Result Compatibility: {identical}/{total} tests identical")
            
            logger.info(f"Results saved to: {json_file}")
            logger.info(f"Report saved to: {md_file}")
            
            return dbapi_success
            
        except Exception as e:
            logger.error(f"Validation failed with error: {e}")
            logger.error(traceback.format_exc())
            return False
        
        finally:
            # Cleanup connections
            if self.dbapi_connection:
                try:
                    self.dbapi_connection.close()
                    logger.info("✓ DBAPI connection closed")
                except:
                    pass
            
            if self.jdbc_connection:
                try:
                    self.jdbc_connection.close()
                    logger.info("✓ JDBC connection closed")
                except:
                    pass

def main():
    """Main entry point"""
    
    # Set environment variables if not already set
    if not os.environ.get('IRIS_HOST'):
        os.environ['IRIS_HOST'] = 'localhost'
    if not os.environ.get('IRIS_PORT'):
        os.environ['IRIS_PORT'] = '1972'
    if not os.environ.get('IRIS_NAMESPACE'):
        os.environ['IRIS_NAMESPACE'] = 'USER'
    if not os.environ.get('IRIS_USER'):
        os.environ['IRIS_USER'] = '_SYSTEM'
    if not os.environ.get('IRIS_PASSWORD'):
        os.environ['IRIS_PASSWORD'] = 'SYS'
    
    validator = DBAPIVectorSearchValidator()
    success = validator.run_validation()
    
    if success:
        logger.info("\n🎉 DBAPI vector search validation completed successfully!")
        logger.info("Vector search patterns work correctly through DBAPI.")
    else:
        logger.warning("\n⚠️ DBAPI vector search validation found issues.")
        logger.warning("Review the results to understand DBAPI vector search limitations.")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()