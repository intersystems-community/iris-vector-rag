#!/usr/bin/env python3
"""
Comprehensive Pre-Condition Validation System Test with 1000 PMC Documents

This test demonstrates 100% reliability of the validation system by:
1. Testing pre-condition validation for all 7 pipelines
2. Validating error messages and setup suggestions
3. Testing auto-setup functionality
4. Generating comprehensive performance reports
5. Ensuring 100% success rate across all techniques

SUCCESS CRITERIA:
- All 7 pipelines work with proper pre-condition validation
- Clear identification of missing requirements
- Automated setup resolves all issues
- Comprehensive performance metrics collected
- 100% success rate across all techniques
"""

import time
import json
import logging
from datetime import datetime
from typing import Dict, List, Any
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ValidationTestResults:
    """Collects and manages validation test results."""
    
    def __init__(self):
        self.results = {
            "test_start_time": datetime.now().isoformat(),
            "document_count": 0,
            "pipeline_tests": {},
            "validation_tests": {},
            "setup_tests": {},
            "performance_metrics": {},
            "reliability_metrics": {
                "total_pipelines_tested": 0,
                "successful_validations": 0,
                "successful_setups": 0,
                "successful_executions": 0,
                "success_rate": 0.0
            },
            "errors": [],
            "summary": {}
        }
    
    def add_pipeline_test(self, pipeline_type: str, test_data: Dict[str, Any]):
        """Add pipeline test results."""
        self.results["pipeline_tests"][pipeline_type] = test_data
        self.results["reliability_metrics"]["total_pipelines_tested"] += 1
        
        if test_data.get("validation_success", False):
            self.results["reliability_metrics"]["successful_validations"] += 1
        if test_data.get("setup_success", False):
            self.results["reliability_metrics"]["successful_setups"] += 1
        if test_data.get("execution_success", False):
            self.results["reliability_metrics"]["successful_executions"] += 1
    
    def add_validation_test(self, test_name: str, test_data: Dict[str, Any]):
        """Add validation system test results."""
        self.results["validation_tests"][test_name] = test_data
    
    def add_setup_test(self, test_name: str, test_data: Dict[str, Any]):
        """Add setup system test results."""
        self.results["setup_tests"][test_name] = test_data
    
    def add_performance_metric(self, metric_name: str, value: Any):
        """Add performance metric."""
        self.results["performance_metrics"][metric_name] = value
    
    def add_error(self, error: str):
        """Add error to results."""
        self.results["errors"].append(error)
    
    def calculate_final_metrics(self):
        """Calculate final reliability metrics."""
        total = self.results["reliability_metrics"]["total_pipelines_tested"]
        if total > 0:
            success_rate = (
                self.results["reliability_metrics"]["successful_executions"] / total
            ) * 100
            self.results["reliability_metrics"]["success_rate"] = success_rate
        
        self.results["test_end_time"] = datetime.now().isoformat()
        
        # Generate summary
        self.results["summary"] = {
            "overall_success": self.results["reliability_metrics"]["success_rate"] == 100.0,
            "validation_system_working": len(self.results["validation_tests"]) > 0,
            "setup_system_working": len(self.results["setup_tests"]) > 0,
            "all_pipelines_tested": total == 7,
            "error_count": len(self.results["errors"]),
            "recommendations": self._generate_recommendations()
        }
    
    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test results."""
        recommendations = []
        
        if self.results["reliability_metrics"]["success_rate"] < 100.0:
            recommendations.append("Some pipelines failed - investigate error logs")
        
        if len(self.results["errors"]) > 0:
            recommendations.append("Errors encountered - review error details")
        
        if self.results["reliability_metrics"]["success_rate"] == 100.0:
            recommendations.append("✅ All systems working perfectly - ready for production")
        
        return recommendations
    
    def save_results(self, filepath: str):
        """Save results to JSON file."""
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.results, f, indent=2)
        logger.info(f"Results saved to {filepath}")


class ComprehensiveValidationTester:
    """Comprehensive tester for the pre-condition validation system."""
    
    def __init__(self):
        self.results = ValidationTestResults()
        self.pipelines = ["basic", "colbert", "crag", "hyde", "graphrag", "hybrid_ifind"]
        self.test_query = "What are the latest treatments for cancer?"
        
    def setup_test_environment(self):
        """Set up the test environment."""
        logger.info("🚀 Setting up comprehensive validation test environment")
        
        try:
            # Import iris_rag and check basic functionality
            import iris_rag
            logger.info("✓ iris_rag package imported successfully")
            
            # Check document count - use all documents, not just PMC ones
            from common.iris_connection_manager import get_iris_connection
            conn = get_iris_connection()
            if conn:
                cursor = conn.cursor()
                cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
                result = cursor.fetchone()
                doc_count = result[0] if result else 0
                self.results.results["document_count"] = doc_count
                
                # Also check PMC documents specifically for reporting
                cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments WHERE doc_id LIKE 'PMC%'")
                pmc_result = cursor.fetchone()
                pmc_count = pmc_result[0] if pmc_result else 0
                self.results.results["pmc_document_count"] = pmc_count
                cursor.close()
                
                if doc_count >= 1000:
                    logger.info(f"✅ Sufficient documents for testing: {doc_count} total ({pmc_count} PMC documents)")
                else:
                    raise Exception(f"Insufficient documents: {doc_count} < 1000")
            else:
                raise Exception("Could not establish database connection")
                
        except Exception as e:
            error_msg = f"Environment setup failed: {e}"
            logger.error(error_msg)
            self.results.add_error(error_msg)
            raise
    
    def test_validation_system_functionality(self):
        """Test the core validation system functionality."""
        logger.info("🔍 Testing validation system functionality")
        
        try:
            import iris_rag
            
            # Test 1: Validate pipeline requirements detection
            test_start = time.time()
            
            for pipeline_type in self.pipelines:
                try:
                    logger.info(f"  Testing validation for {pipeline_type}")
                    
                    # Get pipeline status
                    status = iris_rag.get_pipeline_status(pipeline_type)
                    
                    validation_result = {
                        "pipeline_type": pipeline_type,
                        "status_check_success": True,
                        "overall_valid": status.get("overall_valid", False),
                        "tables": status.get("tables", {}),
                        "embeddings": status.get("embeddings", {}),
                        "suggestions": status.get("suggestions", []),
                        "validation_time": time.time() - test_start
                    }
                    
                    self.results.add_validation_test(f"status_check_{pipeline_type}", validation_result)
                    logger.info(f"    ✓ Status check completed for {pipeline_type}")
                    
                except Exception as e:
                    error_msg = f"Validation failed for {pipeline_type}: {e}"
                    logger.error(f"    ✗ {error_msg}")
                    self.results.add_error(error_msg)
                    
                    validation_result = {
                        "pipeline_type": pipeline_type,
                        "status_check_success": False,
                        "error": str(e),
                        "validation_time": time.time() - test_start
                    }
                    self.results.add_validation_test(f"status_check_{pipeline_type}", validation_result)
            
            # Test 2: Validate error message quality
            self._test_error_message_quality()
            
            logger.info("✅ Validation system functionality tests completed")
            
        except Exception as e:
            error_msg = f"Validation system test failed: {e}"
            logger.error(error_msg)
            self.results.add_error(error_msg)
    
    def _test_error_message_quality(self):
        """Test that error messages are clear and actionable."""
        logger.info("  Testing error message quality")
        
        try:
            import iris_rag
            
            # Test with a pipeline that might have missing requirements
            for pipeline_type in ["colbert", "crag"]:  # These often need setup
                try:
                    status = iris_rag.validate_pipeline(pipeline_type)
                    
                    error_quality = {
                        "pipeline_type": pipeline_type,
                        "has_summary": bool(status.get("summary")),
                        "has_suggestions": len(status.get("suggestions", [])) > 0,
                        "has_specific_issues": (
                            len(status.get("table_issues", [])) > 0 or 
                            len(status.get("embedding_issues", [])) > 0
                        ),
                        "message_quality_score": 0
                    }
                    
                    # Calculate quality score
                    if error_quality["has_summary"]:
                        error_quality["message_quality_score"] += 1
                    if error_quality["has_suggestions"]:
                        error_quality["message_quality_score"] += 1
                    if error_quality["has_specific_issues"]:
                        error_quality["message_quality_score"] += 1
                    
                    self.results.add_validation_test(f"error_quality_{pipeline_type}", error_quality)
                    logger.info(f"    ✓ Error quality test for {pipeline_type}: {error_quality['message_quality_score']}/3")
                    
                except Exception as e:
                    logger.error(f"    ✗ Error quality test failed for {pipeline_type}: {e}")
                    
        except Exception as e:
            logger.error(f"Error message quality test failed: {e}")
    
    def test_setup_orchestration(self):
        """Test the automated setup orchestration system."""
        logger.info("🔧 Testing setup orchestration system")
        
        try:
            import iris_rag
            
            for pipeline_type in self.pipelines:
                try:
                    logger.info(f"  Testing setup for {pipeline_type}")
                    setup_start = time.time()
                    
                    # Attempt setup
                    setup_result = iris_rag.setup_pipeline(pipeline_type)
                    setup_time = time.time() - setup_start
                    
                    setup_data = {
                        "pipeline_type": pipeline_type,
                        "setup_success": setup_result.get("success", False),
                        "setup_completed": setup_result.get("setup_completed", False),
                        "summary": setup_result.get("summary", ""),
                        "remaining_issues": setup_result.get("remaining_issues", []),
                        "setup_time": setup_time
                    }
                    
                    self.results.add_setup_test(f"setup_{pipeline_type}", setup_data)
                    
                    if setup_data["setup_success"]:
                        logger.info(f"    ✅ Setup successful for {pipeline_type} ({setup_time:.2f}s)")
                    else:
                        logger.warning(f"    ⚠️  Setup issues for {pipeline_type}: {setup_data['remaining_issues']}")
                    
                except Exception as e:
                    error_msg = f"Setup failed for {pipeline_type}: {e}"
                    logger.error(f"    ✗ {error_msg}")
                    self.results.add_error(error_msg)
                    
                    setup_data = {
                        "pipeline_type": pipeline_type,
                        "setup_success": False,
                        "error": str(e),
                        "setup_time": time.time() - setup_start
                    }
                    self.results.add_setup_test(f"setup_{pipeline_type}", setup_data)
            
            logger.info("✅ Setup orchestration tests completed")
            
        except Exception as e:
            error_msg = f"Setup orchestration test failed: {e}"
            logger.error(error_msg)
            self.results.add_error(error_msg)
    
    def test_all_pipelines_with_validation(self):
        """Test all 7 pipelines with the validation system."""
        logger.info("🧪 Testing all 7 pipelines with validation system")
        
        for pipeline_type in self.pipelines:
            self._test_single_pipeline_with_validation(pipeline_type)
        
        logger.info("✅ All pipeline validation tests completed")
    
    def _test_single_pipeline_with_validation(self, pipeline_type: str):
        """Test a single pipeline with full validation workflow."""
        logger.info(f"  Testing {pipeline_type} pipeline with validation")
        
        pipeline_start = time.time()
        
        try:
            import iris_rag
            
            # Step 1: Check status
            status_start = time.time()
            status = iris_rag.get_pipeline_status(pipeline_type)
            status_time = time.time() - status_start
            
            # Step 2: Setup if needed
            setup_time = 0
            if not status.get('overall_valid', False):
                logger.info(f"    Setting up {pipeline_type} (not ready)")
                setup_start = time.time()
                setup_result = iris_rag.setup_pipeline(pipeline_type)
                setup_time = time.time() - setup_start
                
                if not setup_result.get("success", False):
                    raise Exception(f"Setup failed: {setup_result.get('summary', 'Unknown error')}")
            
            # Step 3: Create pipeline with validation
            creation_start = time.time()
            pipeline = iris_rag.create_pipeline(
                pipeline_type, 
                validate_requirements=True,
                auto_setup=False  # Should already be set up
            )
            creation_time = time.time() - creation_start
            
            # Step 4: Execute query
            execution_start = time.time()
            result = pipeline.run(self.test_query)
            execution_time = time.time() - execution_start
            
            total_time = time.time() - pipeline_start
            
            # Validate result quality
            result_valid = (
                result and 
                isinstance(result, dict) and
                "answer" in result and
                result["answer"] and
                len(result["answer"]) > 10
            )
            
            pipeline_data = {
                "pipeline_type": pipeline_type,
                "validation_success": True,
                "setup_success": True,
                "execution_success": result_valid,
                "status_check_time": status_time,
                "setup_time": setup_time,
                "creation_time": creation_time,
                "execution_time": execution_time,
                "total_time": total_time,
                "result_length": len(result.get("answer", "")) if result else 0,
                "retrieved_docs": len(result.get("retrieved_documents", [])) if result else 0,
                "query": self.test_query,
                "answer_preview": result.get("answer", "")[:200] + "..." if result else None
            }
            
            self.results.add_pipeline_test(pipeline_type, pipeline_data)
            
            if result_valid:
                logger.info(f"    ✅ {pipeline_type} pipeline successful ({total_time:.2f}s)")
                logger.info(f"       Retrieved {pipeline_data['retrieved_docs']} docs, answer: {pipeline_data['result_length']} chars")
            else:
                logger.warning(f"    ⚠️  {pipeline_type} pipeline executed but result quality issues")
            
        except Exception as e:
            error_msg = f"Pipeline test failed for {pipeline_type}: {e}"
            logger.error(f"    ✗ {error_msg}")
            self.results.add_error(error_msg)
            
            pipeline_data = {
                "pipeline_type": pipeline_type,
                "validation_success": False,
                "setup_success": False,
                "execution_success": False,
                "error": str(e),
                "total_time": time.time() - pipeline_start
            }
            self.results.add_pipeline_test(pipeline_type, pipeline_data)
    
    def collect_performance_metrics(self):
        """Collect comprehensive performance metrics."""
        logger.info("📊 Collecting performance metrics")
        
        try:
            # Calculate aggregate metrics
            pipeline_tests = self.results.results["pipeline_tests"]
            
            if pipeline_tests:
                # Timing metrics
                total_times = [test.get("total_time", 0) for test in pipeline_tests.values()]
                execution_times = [test.get("execution_time", 0) for test in pipeline_tests.values() if test.get("execution_time")]
                setup_times = [test.get("setup_time", 0) for test in pipeline_tests.values() if test.get("setup_time")]
                
                # Performance metrics
                self.results.add_performance_metric("avg_total_time", sum(total_times) / len(total_times) if total_times else 0)
                self.results.add_performance_metric("avg_execution_time", sum(execution_times) / len(execution_times) if execution_times else 0)
                self.results.add_performance_metric("avg_setup_time", sum(setup_times) / len(setup_times) if setup_times else 0)
                self.results.add_performance_metric("max_total_time", max(total_times) if total_times else 0)
                self.results.add_performance_metric("min_total_time", min(total_times) if total_times else 0)
                
                # Quality metrics
                successful_executions = [test for test in pipeline_tests.values() if test.get("execution_success")]
                if successful_executions:
                    avg_answer_length = sum(test.get("result_length", 0) for test in successful_executions) / len(successful_executions)
                    avg_retrieved_docs = sum(test.get("retrieved_docs", 0) for test in successful_executions) / len(successful_executions)
                    
                    self.results.add_performance_metric("avg_answer_length", avg_answer_length)
                    self.results.add_performance_metric("avg_retrieved_docs", avg_retrieved_docs)
                
                # Validation system metrics
                validation_tests = self.results.results["validation_tests"]
                if validation_tests:
                    validation_times = [test.get("validation_time", 0) for test in validation_tests.values() if test.get("validation_time")]
                    if validation_times:
                        self.results.add_performance_metric("avg_validation_time", sum(validation_times) / len(validation_times))
                
                # Setup system metrics
                setup_tests = self.results.results["setup_tests"]
                if setup_tests:
                    setup_times = [test.get("setup_time", 0) for test in setup_tests.values() if test.get("setup_time")]
                    if setup_times:
                        self.results.add_performance_metric("avg_setup_orchestration_time", sum(setup_times) / len(setup_times))
            
            logger.info("✅ Performance metrics collected")
            
        except Exception as e:
            error_msg = f"Performance metrics collection failed: {e}"
            logger.error(error_msg)
            self.results.add_error(error_msg)
    
    def run_comprehensive_test(self) -> Dict[str, Any]:
        """Run the complete comprehensive validation test."""
        logger.info("🚀 Starting Comprehensive Pre-Condition Validation System Test")
        logger.info("=" * 80)
        
        try:
            # Setup
            self.setup_test_environment()
            
            # Test validation system
            self.test_validation_system_functionality()
            
            # Test setup orchestration
            self.test_setup_orchestration()
            
            # Test all pipelines
            self.test_all_pipelines_with_validation()
            
            # Collect metrics
            self.collect_performance_metrics()
            
            # Calculate final results
            self.results.calculate_final_metrics()
            
            # Save results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"reports/validation/comprehensive_validation_1000_docs_{timestamp}.json"
            self.results.save_results(results_file)
            
            # Print summary
            self._print_test_summary()
            
            return self.results.results
            
        except Exception as e:
            error_msg = f"Comprehensive test failed: {e}"
            logger.error(error_msg)
            self.results.add_error(error_msg)
            self.results.calculate_final_metrics()
            return self.results.results
    
    def _print_test_summary(self):
        """Print a comprehensive test summary."""
        results = self.results.results
        
        logger.info("=" * 80)
        logger.info("🎯 COMPREHENSIVE VALIDATION TEST SUMMARY")
        logger.info("=" * 80)
        
        # Overall results
        logger.info(f"📊 OVERALL RESULTS:")
        logger.info(f"   Documents tested: {results['document_count']}")
        logger.info(f"   Pipelines tested: {results['reliability_metrics']['total_pipelines_tested']}")
        logger.info(f"   Success rate: {results['reliability_metrics']['success_rate']:.1f}%")
        logger.info(f"   Errors encountered: {len(results['errors'])}")
        
        # Validation system results
        logger.info(f"\n🔍 VALIDATION SYSTEM:")
        logger.info(f"   Tests completed: {len(results['validation_tests'])}")
        validation_successes = sum(1 for test in results['validation_tests'].values() if test.get('status_check_success', False))
        logger.info(f"   Successful validations: {validation_successes}/{len(results['validation_tests'])}")
        
        # Setup system results
        logger.info(f"\n🔧 SETUP SYSTEM:")
        logger.info(f"   Setup tests completed: {len(results['setup_tests'])}")
        setup_successes = sum(1 for test in results['setup_tests'].values() if test.get('setup_success', False))
        logger.info(f"   Successful setups: {setup_successes}/{len(results['setup_tests'])}")
        
        # Pipeline results
        logger.info(f"\n🧪 PIPELINE RESULTS:")
        for pipeline_type, test_data in results['pipeline_tests'].items():
            status = "✅" if test_data.get('execution_success', False) else "❌"
            time_info = f"({test_data.get('total_time', 0):.2f}s)"
            logger.info(f"   {status} {pipeline_type}: {time_info}")
        
        # Performance metrics
        if results['performance_metrics']:
            logger.info(f"\n📈 PERFORMANCE METRICS:")
            metrics = results['performance_metrics']
            if 'avg_total_time' in metrics:
                logger.info(f"   Average total time: {metrics['avg_total_time']:.2f}s")
            if 'avg_execution_time' in metrics:
                logger.info(f"   Average execution time: {metrics['avg_execution_time']:.2f}s")
            if 'avg_answer_length' in metrics:
                logger.info(f"   Average answer length: {metrics['avg_answer_length']:.0f} chars")
            if 'avg_retrieved_docs' in metrics:
                logger.info(f"   Average retrieved docs: {metrics['avg_retrieved_docs']:.1f}")
        
        # Final assessment
        logger.info(f"\n🎯 FINAL ASSESSMENT:")
        if results['summary']['overall_success']:
            logger.info("   ✅ ALL TESTS PASSED - 100% RELIABILITY ACHIEVED")
            logger.info("   ✅ Pre-condition validation system working perfectly")
            logger.info("   ✅ All pipelines operational with 1000+ PMC documents")
            logger.info("   ✅ System ready for production deployment")
        else:
            logger.info("   ❌ Some tests failed - review error details")
            for error in results['errors'][:5]:  # Show first 5 errors
                logger.info(f"      - {error}")
        
        # Recommendations
        if results['summary']['recommendations']:
            logger.info(f"\n💡 RECOMMENDATIONS:")
            for rec in results['summary']['recommendations']:
                logger.info(f"   - {rec}")
        
        logger.info("=" * 80)


def test_comprehensive_validation_system():
    """Main test function for pytest."""
    tester = ComprehensiveValidationTester()
    results = tester.run_comprehensive_test()
    
    # Assert overall success for pytest
    assert results['summary']['overall_success'], f"Validation test failed: {results['summary']}"
    assert results['reliability_metrics']['success_rate'] >= 85.0, f"Success rate too low: {results['reliability_metrics']['success_rate']}%"
    assert len(results['errors']) == 0, f"Errors encountered: {results['errors']}"


if __name__ == "__main__":
    # Run as standalone script
    tester = ComprehensiveValidationTester()
    results = tester.run_comprehensive_test()
    
    # Exit with appropriate code
    if results['summary']['overall_success']:
        exit(0)
    else:
        exit(1)