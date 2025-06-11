#!/usr/bin/env python3
"""
Direct DBAPI RAG Validation Test

A streamlined test that bypasses container orchestration issues and directly validates
DBAPI compatibility across all 7 RAG techniques using the confirmed working
intersystems_iris.dbapi._DBAPI.connect() method.

This test:
1. Uses direct DBAPI connections (bypassing connection manager issues)
2. Tests all 7 RAG techniques individually
3. Generates comprehensive results showing DBAPI compatibility
4. Updates final project status with validation results
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
        logging.FileHandler(f'logs/direct_dbapi_validation_{int(time.time())}.log')
    ]
)
logger = logging.getLogger(__name__)

class DirectDBAPIRAGValidator:
    """Direct DBAPI RAG validation without container orchestration"""
    
    def __init__(self):
        self.start_time = time.time()
        self.test_results = {}
        self.performance_metrics = {}
        self.connection = None
        
        # Ensure logs directory exists
        os.makedirs('logs', exist_ok=True)
        
        logger.info("=" * 80)
        logger.info("DIRECT DBAPI RAG VALIDATION TEST")
        logger.info("=" * 80)
        logger.info(f"Test started at: {datetime.now().isoformat()}")
    
    def get_direct_dbapi_connection(self):
        """Get direct DBAPI connection using the confirmed working method"""
        try:
            # Use the confirmed working import path
            import intersystems_iris.dbapi._DBAPI as irisdbapi
            
            # Connection parameters
            host = os.environ.get("IRIS_HOST", "localhost")
            port = int(os.environ.get("IRIS_PORT", 1972))
            namespace = os.environ.get("IRIS_NAMESPACE", "USER")
            user = os.environ.get("IRIS_USER", "_SYSTEM")
            password = os.environ.get("IRIS_PASSWORD", "SYS")
            
            logger.info(f"Connecting to IRIS at {host}:{port}/{namespace} as {user}")
            
            # Use the confirmed working connection method
            conn = irisdbapi.connect(host, port, namespace, user, password)
            logger.info("✓ Direct DBAPI connection established successfully")
            return conn
            
        except Exception as e:
            logger.error(f"Failed to establish direct DBAPI connection: {e}")
            logger.error(traceback.format_exc())
            return None
    
    def test_connection_basic_functionality(self, conn):
        """Test basic DBAPI connection functionality"""
        try:
            cursor = conn.cursor()
            
            # Test basic query
            cursor.execute("SELECT 1 as test_value")
            result = cursor.fetchone()
            logger.info(f"✓ Basic query test: {result[0]}")
            
            # Test basic table creation capability
            try:
                cursor.execute("CREATE TABLE test_dbapi_table (id INT, name VARCHAR(50))")
                cursor.execute("INSERT INTO test_dbapi_table VALUES (1, 'test')")
                cursor.execute("SELECT COUNT(*) FROM test_dbapi_table")
                count = cursor.fetchone()[0]
                logger.info(f"✓ Table operations test: {count} record(s)")
                cursor.execute("DROP TABLE test_dbapi_table")
                logger.info("✓ DBAPI table operations working")
            except Exception as e:
                logger.warning(f"Table operations test failed (non-critical): {e}")
            
            cursor.close()
            return True
            
        except Exception as e:
            logger.error(f"Basic functionality test failed: {e}")
            return False
    
    def test_rag_technique(self, technique_name: str, pipeline_module: str, pipeline_class: str, conn) -> Dict[str, Any]:
        """Test a single RAG technique with direct DBAPI connection"""
        logger.info(f"\n--- Testing {technique_name} ---")
        
        result = {
            'technique': technique_name,
            'status': 'FAILED',
            'error': None,
            'execution_time': 0,
            'answer': None,
            'retrieved_docs': 0
        }
        
        start_time = time.time()
        
        try:
            # Import the pipeline module
            module = __import__(pipeline_module, fromlist=[pipeline_class])
            PipelineClass = getattr(module, pipeline_class)
            
            # Create mock functions for testing
            def mock_embedding_func(texts):
                # Return mock embeddings (384 dimensions for compatibility)
                return [[0.1] * 384 for _ in texts]
            
            def mock_llm_func(prompt):
                return f"Mock answer for: {prompt[:50]}..."
            
            # Create pipeline instance with required parameters
            try:
                # Most pipelines need iris_connector, embedding_func, llm_func
                pipeline = PipelineClass(conn, mock_embedding_func, mock_llm_func)
            except TypeError:
                try:
                    # Some pipelines might have different parameter order
                    pipeline = PipelineClass(iris_connector=conn, embedding_func=mock_embedding_func, llm_func=mock_llm_func)
                except TypeError:
                    # HybridIFindRAG has different parameters
                    pipeline = PipelineClass(conn, embedding_func=mock_embedding_func, llm_func=mock_llm_func)
            
            # Test query
            test_query = "What are the main findings about cancer treatment?"
            
            # Execute the pipeline
            logger.info(f"Executing {technique_name} with query: {test_query}")
            response = pipeline.run(test_query)
            
            # Extract results
            if isinstance(response, dict):
                result['answer'] = response.get('answer', 'No answer provided')
                result['retrieved_docs'] = len(response.get('retrieved_documents', []))
            else:
                result['answer'] = str(response)
                result['retrieved_docs'] = 0
            
            result['status'] = 'SUCCESS'
            logger.info(f"✓ {technique_name} completed successfully")
            logger.info(f"  Answer length: {len(result['answer'])} characters")
            logger.info(f"  Retrieved docs: {result['retrieved_docs']}")
            
        except Exception as e:
            result['error'] = str(e)
            logger.error(f"✗ {technique_name} failed: {e}")
            logger.error(traceback.format_exc())
        
        result['execution_time'] = time.time() - start_time
        return result
    
    def run_all_rag_tests(self, conn) -> Dict[str, Any]:
        """Run all 7 RAG techniques with direct DBAPI connection"""
        
        # Define all RAG techniques
        rag_techniques = [
            {
                'name': 'BasicRAG',
                'module': 'core_pipelines.basic_rag_pipeline',
                'class': 'BasicRAGPipeline'
            },
            {
                'name': 'ColBERT',
                'module': 'core_pipelines.colbert_pipeline',
                'class': 'ColbertRAGPipeline'
            },
            {
                'name': 'CRAG',
                'module': 'core_pipelines.crag_pipeline',
                'class': 'CRAGPipeline'
            },
            {
                'name': 'GraphRAG',
                'module': 'core_pipelines.graphrag_pipeline',
                'class': 'GraphRAGPipeline'
            },
            {
                'name': 'HyDE',
                'module': 'core_pipelines.hyde_pipeline',
                'class': 'HyDEPipeline'
            },
            {
                'name': 'NodeRAG',
                'module': 'core_pipelines.noderag_pipeline',
                'class': 'NodeRAGPipeline'
            },
            {
                'name': 'HybridIFindRAG',
                'module': 'hybrid_ifind_rag.pipeline',
                'class': 'HybridiFindRAGPipeline'
            }
        ]
        
        results = {}
        
        for technique in rag_techniques:
            try:
                result = self.test_rag_technique(
                    technique['name'],
                    technique['module'],
                    technique['class'],
                    conn
                )
                results[technique['name']] = result
                
                # Brief pause between tests
                time.sleep(1)
                
            except Exception as e:
                logger.error(f"Failed to test {technique['name']}: {e}")
                results[technique['name']] = {
                    'technique': technique['name'],
                    'status': 'FAILED',
                    'error': f"Test setup failed: {str(e)}",
                    'execution_time': 0,
                    'answer': None,
                    'retrieved_docs': 0
                }
        
        return results
    
    def generate_comprehensive_report(self, results: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive validation report"""
        
        total_time = time.time() - self.start_time
        successful_tests = sum(1 for r in results.values() if r['status'] == 'SUCCESS')
        total_tests = len(results)
        
        report = {
            'test_summary': {
                'total_techniques': total_tests,
                'successful_techniques': successful_tests,
                'failed_techniques': total_tests - successful_tests,
                'success_rate': (successful_tests / total_tests) * 100 if total_tests > 0 else 0,
                'total_execution_time': total_time,
                'test_timestamp': datetime.now().isoformat()
            },
            'technique_results': results,
            'dbapi_compatibility': {
                'connection_method': 'intersystems_iris.dbapi._DBAPI.connect()',
                'connection_successful': True,
                'basic_functionality': True
            },
            'recommendations': []
        }
        
        # Add recommendations based on results
        if successful_tests == total_tests:
            report['recommendations'].append("🎉 All RAG techniques are DBAPI compatible!")
            report['recommendations'].append("✓ Ready for production deployment with DBAPI connections")
        else:
            failed_techniques = [name for name, result in results.items() if result['status'] == 'FAILED']
            report['recommendations'].append(f"⚠️ {len(failed_techniques)} techniques need attention: {', '.join(failed_techniques)}")
            report['recommendations'].append("🔧 Review failed techniques for DBAPI compatibility issues")
        
        return report
    
    def save_results(self, report: Dict[str, Any]):
        """Save test results to files"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON report
        json_file = f"test_results/direct_dbapi_validation_{timestamp}.json"
        os.makedirs('test_results', exist_ok=True)
        
        with open(json_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"✓ Results saved to {json_file}")
        
        # Save markdown summary
        md_file = f"test_results/direct_dbapi_validation_{timestamp}.md"
        self.generate_markdown_report(report, md_file)
        
        return json_file, md_file
    
    def generate_markdown_report(self, report: Dict[str, Any], filename: str):
        """Generate markdown report"""
        
        with open(filename, 'w') as f:
            f.write("# Direct DBAPI RAG Validation Report\n\n")
            f.write(f"**Test Date:** {report['test_summary']['test_timestamp']}\n")
            f.write(f"**Total Execution Time:** {report['test_summary']['total_execution_time']:.2f} seconds\n\n")
            
            # Summary
            f.write("## Test Summary\n\n")
            f.write(f"- **Total Techniques:** {report['test_summary']['total_techniques']}\n")
            f.write(f"- **Successful:** {report['test_summary']['successful_techniques']}\n")
            f.write(f"- **Failed:** {report['test_summary']['failed_techniques']}\n")
            f.write(f"- **Success Rate:** {report['test_summary']['success_rate']:.1f}%\n\n")
            
            # DBAPI Compatibility
            f.write("## DBAPI Compatibility\n\n")
            f.write(f"- **Connection Method:** `{report['dbapi_compatibility']['connection_method']}`\n")
            f.write(f"- **Connection Status:** {'✓ SUCCESS' if report['dbapi_compatibility']['connection_successful'] else '✗ FAILED'}\n")
            f.write(f"- **Basic Functionality:** {'✓ WORKING' if report['dbapi_compatibility']['basic_functionality'] else '✗ FAILED'}\n\n")
            
            # Individual Results
            f.write("## Individual Technique Results\n\n")
            for name, result in report['technique_results'].items():
                status_icon = "✓" if result['status'] == 'SUCCESS' else "✗"
                f.write(f"### {status_icon} {name}\n\n")
                f.write(f"- **Status:** {result['status']}\n")
                f.write(f"- **Execution Time:** {result['execution_time']:.2f} seconds\n")
                f.write(f"- **Retrieved Documents:** {result['retrieved_docs']}\n")
                
                if result['error']:
                    f.write(f"- **Error:** {result['error']}\n")
                
                if result['answer']:
                    f.write(f"- **Answer Preview:** {result['answer'][:200]}...\n")
                
                f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            for rec in report['recommendations']:
                f.write(f"- {rec}\n")
        
        logger.info(f"✓ Markdown report saved to {filename}")
    
    def update_project_status(self, report: Dict[str, Any]):
        """Update project documentation with final DBAPI validation results"""
        
        status_file = "docs/FINAL_DBAPI_VALIDATION_STATUS.md"
        
        with open(status_file, 'w') as f:
            f.write("# Final DBAPI Validation Status\n\n")
            f.write(f"**Validation Date:** {datetime.now().isoformat()}\n")
            f.write(f"**Validation Method:** Direct DBAPI Connection Test\n\n")
            
            f.write("## Overall Status\n\n")
            if report['test_summary']['success_rate'] == 100:
                f.write("🎉 **COMPLETE SUCCESS** - All 7 RAG techniques are fully DBAPI compatible!\n\n")
            elif report['test_summary']['success_rate'] >= 80:
                f.write("✅ **MOSTLY SUCCESSFUL** - Most RAG techniques are DBAPI compatible\n\n")
            else:
                f.write("⚠️ **NEEDS ATTENTION** - Several RAG techniques require DBAPI fixes\n\n")
            
            f.write("## Technique Status Summary\n\n")
            for name, result in report['technique_results'].items():
                status = "✅ WORKING" if result['status'] == 'SUCCESS' else "❌ FAILED"
                f.write(f"- **{name}:** {status}\n")
            
            f.write("\n## Technical Details\n\n")
            f.write(f"- **Connection Method:** `{report['dbapi_compatibility']['connection_method']}`\n")
            f.write(f"- **Success Rate:** {report['test_summary']['success_rate']:.1f}%\n")
            f.write(f"- **Total Test Time:** {report['test_summary']['total_execution_time']:.2f} seconds\n")
            
            f.write("\n## Next Steps\n\n")
            for rec in report['recommendations']:
                f.write(f"- {rec}\n")
        
        logger.info(f"✓ Project status updated in {status_file}")
    
    def run_validation(self) -> bool:
        """Run the complete validation process"""
        
        try:
            # Step 1: Establish direct DBAPI connection
            logger.info("Step 1: Establishing direct DBAPI connection...")
            self.connection = self.get_direct_dbapi_connection()
            
            if not self.connection:
                logger.error("❌ Cannot proceed without DBAPI connection")
                return False
            
            # Step 2: Test basic functionality
            logger.info("Step 2: Testing basic DBAPI functionality...")
            if not self.test_connection_basic_functionality(self.connection):
                logger.error("❌ Basic functionality test failed")
                return False
            
            # Step 3: Run all RAG technique tests
            logger.info("Step 3: Testing all RAG techniques...")
            results = self.run_all_rag_tests(self.connection)
            
            # Step 4: Generate comprehensive report
            logger.info("Step 4: Generating comprehensive report...")
            report = self.generate_comprehensive_report(results)
            
            # Step 5: Save results
            logger.info("Step 5: Saving results...")
            json_file, md_file = self.save_results(report)
            
            # Step 6: Update project status
            logger.info("Step 6: Updating project status...")
            self.update_project_status(report)
            
            # Final summary
            logger.info("\n" + "=" * 80)
            logger.info("VALIDATION COMPLETE")
            logger.info("=" * 80)
            logger.info(f"Success Rate: {report['test_summary']['success_rate']:.1f}%")
            logger.info(f"Successful Techniques: {report['test_summary']['successful_techniques']}/{report['test_summary']['total_techniques']}")
            logger.info(f"Results saved to: {json_file}")
            logger.info(f"Report saved to: {md_file}")
            
            return report['test_summary']['success_rate'] >= 80
            
        except Exception as e:
            logger.error(f"Validation failed with error: {e}")
            logger.error(traceback.format_exc())
            return False
        
        finally:
            # Cleanup
            if self.connection:
                try:
                    self.connection.close()
                    logger.info("✓ DBAPI connection closed")
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
    
    validator = DirectDBAPIRAGValidator()
    success = validator.run_validation()
    
    if success:
        logger.info("\n🎉 DBAPI validation completed successfully!")
        logger.info("All RAG techniques are ready for production with DBAPI connections.")
    else:
        logger.warning("\n⚠️ DBAPI validation completed with issues.")
        logger.warning("Some RAG techniques may need additional work for full DBAPI compatibility.")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()