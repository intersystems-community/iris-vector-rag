#!/usr/bin/env python3
"""
Chunking Test Execution Runner

This module provides infrastructure for executing and logging comprehensive
chunking architecture tests with proper result collection and analysis.

Features:
1. Automated test execution with proper logging
2. Test result collection and aggregation
3. Performance metrics tracking
4. Error analysis and reporting
5. Test report generation
"""

import pytest
import logging
import time
import json
import sys
import os
import subprocess
from typing import Dict, List, Any, Optional
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class TestResult:
    """Test result data structure."""
    test_name: str
    status: str  # "passed", "failed", "skipped", "error"
    duration: float
    error_message: Optional[str] = None
    output: Optional[str] = None
    metrics: Optional[Dict[str, Any]] = None

@dataclass
class TestSuite:
    """Test suite configuration."""
    name: str
    test_files: List[str]
    description: str
    timeout: int = 3600  # 1 hour default
    required_fixtures: List[str] = None

class ChunkingTestRunner:
    """Comprehensive test runner for chunking architecture tests."""
    
    def __init__(self, output_dir: str = "test_output"):
        """Initialize test runner."""
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        
        # Test execution tracking
        self.test_results: Dict[str, TestResult] = {}
        self.suite_results: Dict[str, Dict[str, Any]] = {}
        self.execution_start_time = None
        self.execution_end_time = None
        
        # Configure logging
        self._setup_logging()
        
        # Define test suites
        self.test_suites = self._define_test_suites()

    def _setup_logging(self):
        """Set up comprehensive logging for test execution."""
        # Create log file with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = self.output_dir / f"chunking_test_execution_{timestamp}.log"
        
        # Configure logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(log_file),
                logging.StreamHandler(sys.stdout)
            ]
        )
        
        logger.info(f"Test execution logging initialized: {log_file}")

    def _define_test_suites(self) -> Dict[str, TestSuite]:
        """Define comprehensive test suites for chunking architecture."""
        return {
            "comprehensive_e2e": TestSuite(
                name="Comprehensive End-to-End Tests",
                test_files=["tests/test_comprehensive_chunking_e2e.py"],
                description="Complete end-to-end testing of all 8 RAG pipelines with chunking",
                timeout=7200,  # 2 hours
                required_fixtures=["enterprise_document_loader_1000docs", "enterprise_embedding_manager"]
            ),
            
            "strategy_validation": TestSuite(
                name="Chunking Strategy Validation",
                test_files=["tests/test_chunking_strategy_validation.py"],
                description="Validation of fixed_size, semantic, and hybrid chunking strategies",
                timeout=3600,  # 1 hour
                required_fixtures=["enterprise_embedding_manager"]
            ),
            
            "error_handling": TestSuite(
                name="Error Handling and Edge Cases",
                test_files=["tests/test_chunking_error_handling.py"],
                description="Comprehensive error handling and edge case testing",
                timeout=1800,  # 30 minutes
                required_fixtures=["enterprise_iris_connection"]
            ),
            
            "integration": TestSuite(
                name="Integration Testing",
                test_files=["tests/test_chunking_integration.py"],
                description="Integration testing for IRISVectorStore and DocumentChunkingService",
                timeout=3600,  # 1 hour
                required_fixtures=["enterprise_schema_manager", "enterprise_document_loader_1000docs"]
            ),
            
            "configuration": TestSuite(
                name="Configuration Testing",
                test_files=["tests/test_pipeline_chunking_inheritance.py"],
                description="Pipeline configuration inheritance and override testing",
                timeout=1800,  # 30 minutes
                required_fixtures=["scale_test_config"]
            )
        }

    def run_all_suites(self, parallel: bool = False) -> Dict[str, Any]:
        """
        Run all test suites with comprehensive logging and result collection.
        
        Args:
            parallel: Whether to run suites in parallel (not implemented yet)
            
        Returns:
            Comprehensive test execution results
        """
        logger.info("Starting comprehensive chunking architecture test execution")
        self.execution_start_time = time.time()
        
        overall_results = {
            "execution_summary": {},
            "suite_results": {},
            "performance_metrics": {},
            "error_analysis": {},
            "recommendations": []
        }
        
        # Run each test suite
        for suite_name, suite_config in self.test_suites.items():
            logger.info(f"Executing test suite: {suite_name}")
            
            suite_result = self._run_test_suite(suite_config)
            overall_results["suite_results"][suite_name] = suite_result
            
            # Log suite completion
            status = "PASSED" if suite_result["success"] else "FAILED"
            logger.info(f"Test suite {suite_name} completed: {status}")
        
        self.execution_end_time = time.time()
        
        # Generate execution summary
        overall_results["execution_summary"] = self._generate_execution_summary()
        overall_results["performance_metrics"] = self._collect_performance_metrics()
        overall_results["error_analysis"] = self._analyze_errors()
        overall_results["recommendations"] = self._generate_recommendations()
        
        # Save comprehensive results
        self._save_execution_results(overall_results)
        
        return overall_results

    def _run_test_suite(self, suite: TestSuite) -> Dict[str, Any]:
        """
        Run a specific test suite with proper logging and error handling.
        
        Args:
            suite: Test suite configuration
            
        Returns:
            Suite execution results
        """
        suite_start_time = time.time()
        
        suite_result = {
            "name": suite.name,
            "description": suite.description,
            "start_time": suite_start_time,
            "test_files": suite.test_files,
            "test_results": {},
            "success": False,
            "duration": 0,
            "error_message": None
        }
        
        try:
            # Run each test file in the suite
            for test_file in suite.test_files:
                logger.info(f"Running test file: {test_file}")
                
                test_result = self._run_test_file(test_file, suite.timeout)
                suite_result["test_results"][test_file] = test_result
                
                # Log test file completion
                status = "PASSED" if test_result["success"] else "FAILED"
                logger.info(f"Test file {test_file} completed: {status}")
            
            # Determine overall suite success
            suite_result["success"] = all(
                result["success"] for result in suite_result["test_results"].values()
            )
            
        except Exception as e:
            logger.error(f"Test suite {suite.name} failed with error: {e}")
            suite_result["error_message"] = str(e)
            suite_result["success"] = False
        
        suite_result["duration"] = time.time() - suite_start_time
        return suite_result

    def _run_test_file(self, test_file: str, timeout: int) -> Dict[str, Any]:
        """
        Run a specific test file using pytest with proper logging.
        
        Args:
            test_file: Path to test file
            timeout: Timeout in seconds
            
        Returns:
            Test file execution results
        """
        test_start_time = time.time()
        
        # Create log file for this test
        test_name = Path(test_file).stem
        log_file = self.output_dir / f"{test_name}_{int(test_start_time)}.log"
        
        # Construct pytest command
        cmd = [
            "uv", "run", "pytest", 
            test_file,
            "-v",
            "--tb=short",
            "--capture=no",
            f"--timeout={timeout}"
        ]
        
        test_result = {
            "test_file": test_file,
            "start_time": test_start_time,
            "command": " ".join(cmd),
            "log_file": str(log_file),
            "success": False,
            "duration": 0,
            "output": "",
            "error_output": "",
            "exit_code": None
        }
        
        try:
            # Run pytest with output capture
            with open(log_file, 'w') as log_f:
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    universal_newlines=True,
                    cwd=os.getcwd()
                )
                
                # Capture output in real-time
                output_lines = []
                while True:
                    line = process.stdout.readline()
                    if not line and process.poll() is not None:
                        break
                    if line:
                        output_lines.append(line.strip())
                        log_f.write(line)
                        log_f.flush()
                        
                        # Also log to main logger
                        if "PASSED" in line or "FAILED" in line or "ERROR" in line:
                            logger.info(f"{test_name}: {line.strip()}")
                
                test_result["exit_code"] = process.poll()
                test_result["output"] = "\n".join(output_lines)
                test_result["success"] = test_result["exit_code"] == 0
                
        except subprocess.TimeoutExpired:
            logger.error(f"Test file {test_file} timed out after {timeout} seconds")
            test_result["error_output"] = f"Test timed out after {timeout} seconds"
            test_result["success"] = False
            
        except Exception as e:
            logger.error(f"Error running test file {test_file}: {e}")
            test_result["error_output"] = str(e)
            test_result["success"] = False
        
        test_result["duration"] = time.time() - test_start_time
        return test_result

    def _generate_execution_summary(self) -> Dict[str, Any]:
        """Generate comprehensive execution summary."""
        total_duration = self.execution_end_time - self.execution_start_time
        
        # Count test results
        total_suites = len(self.test_suites)
        passed_suites = sum(1 for result in self.suite_results.values() 
                           if result.get("success", False))
        
        return {
            "total_execution_time": total_duration,
            "start_time": datetime.fromtimestamp(self.execution_start_time).isoformat(),
            "end_time": datetime.fromtimestamp(self.execution_end_time).isoformat(),
            "total_suites": total_suites,
            "passed_suites": passed_suites,
            "failed_suites": total_suites - passed_suites,
            "success_rate": passed_suites / total_suites if total_suites > 0 else 0,
            "overall_success": passed_suites == total_suites
        }

    def _collect_performance_metrics(self) -> Dict[str, Any]:
        """Collect and analyze performance metrics from test execution."""
        metrics = {
            "suite_durations": {},
            "longest_running_suite": None,
            "shortest_running_suite": None,
            "average_suite_duration": 0,
            "total_test_files": 0
        }
        
        durations = []
        for suite_name, result in self.suite_results.items():
            duration = result.get("duration", 0)
            metrics["suite_durations"][suite_name] = duration
            durations.append((suite_name, duration))
            
            # Count test files
            metrics["total_test_files"] += len(result.get("test_files", []))
        
        if durations:
            durations.sort(key=lambda x: x[1])
            metrics["shortest_running_suite"] = durations[0]
            metrics["longest_running_suite"] = durations[-1]
            metrics["average_suite_duration"] = sum(d[1] for d in durations) / len(durations)
        
        return metrics

    def _analyze_errors(self) -> Dict[str, Any]:
        """Analyze errors and failures from test execution."""
        error_analysis = {
            "total_errors": 0,
            "error_categories": {},
            "failed_suites": [],
            "common_error_patterns": []
        }
        
        for suite_name, result in self.suite_results.items():
            if not result.get("success", False):
                error_analysis["failed_suites"].append({
                    "suite": suite_name,
                    "error": result.get("error_message", "Unknown error")
                })
                error_analysis["total_errors"] += 1
                
                # Categorize errors
                error_msg = result.get("error_message", "").lower()
                if "timeout" in error_msg:
                    error_analysis["error_categories"]["timeout"] = \
                        error_analysis["error_categories"].get("timeout", 0) + 1
                elif "connection" in error_msg:
                    error_analysis["error_categories"]["connection"] = \
                        error_analysis["error_categories"].get("connection", 0) + 1
                elif "memory" in error_msg:
                    error_analysis["error_categories"]["memory"] = \
                        error_analysis["error_categories"].get("memory", 0) + 1
                else:
                    error_analysis["error_categories"]["other"] = \
                        error_analysis["error_categories"].get("other", 0) + 1
        
        return error_analysis

    def _generate_recommendations(self) -> List[str]:
        """Generate recommendations based on test execution results."""
        recommendations = []
        
        # Analyze execution summary
        summary = self._generate_execution_summary()
        
        if summary["success_rate"] < 0.8:
            recommendations.append(
                "Test success rate is below 80%. Review failed tests and address underlying issues."
            )
        
        if summary["total_execution_time"] > 7200:  # 2 hours
            recommendations.append(
                "Test execution time exceeds 2 hours. Consider optimizing tests or running in parallel."
            )
        
        # Analyze performance metrics
        metrics = self._collect_performance_metrics()
        
        if metrics["longest_running_suite"] and metrics["longest_running_suite"][1] > 3600:
            recommendations.append(
                f"Suite '{metrics['longest_running_suite'][0]}' takes over 1 hour. "
                "Consider breaking it into smaller suites."
            )
        
        # Analyze errors
        errors = self._analyze_errors()
        
        if errors["total_errors"] > 0:
            if "timeout" in errors["error_categories"]:
                recommendations.append(
                    "Timeout errors detected. Consider increasing test timeouts or optimizing test performance."
                )
            
            if "connection" in errors["error_categories"]:
                recommendations.append(
                    "Connection errors detected. Verify database connectivity and configuration."
                )
            
            if "memory" in errors["error_categories"]:
                recommendations.append(
                    "Memory errors detected. Consider reducing test data size or increasing available memory."
                )
        
        if not recommendations:
            recommendations.append("All tests executed successfully. No immediate action required.")
        
        return recommendations

    def _save_execution_results(self, results: Dict[str, Any]):
        """Save comprehensive execution results to files."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save JSON results
        json_file = self.output_dir / f"chunking_test_results_{timestamp}.json"
        with open(json_file, 'w') as f:
            json.dump(results, f, indent=2, default=str)
        
        # Generate and save markdown report
        markdown_file = self.output_dir / f"chunking_test_report_{timestamp}.md"
        self._generate_markdown_report(results, markdown_file)
        
        logger.info(f"Test execution results saved:")
        logger.info(f"  JSON: {json_file}")
        logger.info(f"  Report: {markdown_file}")

    def _generate_markdown_report(self, results: Dict[str, Any], output_file: Path):
        """Generate comprehensive markdown test report."""
        with open(output_file, 'w') as f:
            f.write("# Chunking Architecture Test Execution Report\n\n")
            
            # Execution Summary
            summary = results["execution_summary"]
            f.write("## Execution Summary\n\n")
            f.write(f"- **Start Time**: {summary['start_time']}\n")
            f.write(f"- **End Time**: {summary['end_time']}\n")
            f.write(f"- **Total Duration**: {summary['total_execution_time']:.2f} seconds\n")
            f.write(f"- **Success Rate**: {summary['success_rate']:.1%}\n")
            f.write(f"- **Overall Status**: {'✅ PASSED' if summary['overall_success'] else '❌ FAILED'}\n\n")
            
            # Suite Results
            f.write("## Test Suite Results\n\n")
            for suite_name, suite_result in results["suite_results"].items():
                status = "✅ PASSED" if suite_result["success"] else "❌ FAILED"
                f.write(f"### {suite_result['name']} {status}\n\n")
                f.write(f"- **Description**: {suite_result['description']}\n")
                f.write(f"- **Duration**: {suite_result['duration']:.2f} seconds\n")
                f.write(f"- **Test Files**: {len(suite_result['test_files'])}\n")
                
                if not suite_result["success"] and suite_result.get("error_message"):
                    f.write(f"- **Error**: {suite_result['error_message']}\n")
                
                f.write("\n")
            
            # Performance Metrics
            metrics = results["performance_metrics"]
            f.write("## Performance Metrics\n\n")
            f.write(f"- **Total Test Files**: {metrics['total_test_files']}\n")
            f.write(f"- **Average Suite Duration**: {metrics['average_suite_duration']:.2f} seconds\n")
            
            if metrics["longest_running_suite"]:
                f.write(f"- **Longest Running Suite**: {metrics['longest_running_suite'][0]} "
                       f"({metrics['longest_running_suite'][1]:.2f}s)\n")
            
            if metrics["shortest_running_suite"]:
                f.write(f"- **Shortest Running Suite**: {metrics['shortest_running_suite'][0]} "
                       f"({metrics['shortest_running_suite'][1]:.2f}s)\n")
            
            f.write("\n")
            
            # Error Analysis
            errors = results["error_analysis"]
            if errors["total_errors"] > 0:
                f.write("## Error Analysis\n\n")
                f.write(f"- **Total Errors**: {errors['total_errors']}\n")
                
                if errors["error_categories"]:
                    f.write("- **Error Categories**:\n")
                    for category, count in errors["error_categories"].items():
                        f.write(f"  - {category.title()}: {count}\n")
                
                if errors["failed_suites"]:
                    f.write("- **Failed Suites**:\n")
                    for failure in errors["failed_suites"]:
                        f.write(f"  - {failure['suite']}: {failure['error']}\n")
                
                f.write("\n")
            
            # Recommendations
            f.write("## Recommendations\n\n")
            for i, recommendation in enumerate(results["recommendations"], 1):
                f.write(f"{i}. {recommendation}\n")
            
            f.write("\n")
            f.write("---\n")
            f.write(f"*Report generated on {datetime.now().isoformat()}*\n")

def main():
    """Main entry point for test execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Run comprehensive chunking architecture tests")
    parser.add_argument("--output-dir", default="test_output", 
                       help="Output directory for test results")
    parser.add_argument("--suite", choices=list(ChunkingTestRunner({}).test_suites.keys()),
                       help="Run specific test suite only")
    parser.add_argument("--parallel", action="store_true",
                       help="Run test suites in parallel (not implemented)")
    
    args = parser.parse_args()
    
    # Initialize test runner
    runner = ChunkingTestRunner(output_dir=args.output_dir)
    
    if args.suite:
        # Run specific suite
        suite_config = runner.test_suites[args.suite]
        result = runner._run_test_suite(suite_config)
        print(f"Suite {args.suite} completed: {'PASSED' if result['success'] else 'FAILED'}")
    else:
        # Run all suites
        results = runner.run_all_suites(parallel=args.parallel)
        summary = results["execution_summary"]
        print(f"All tests completed: {'PASSED' if summary['overall_success'] else 'FAILED'}")
        print(f"Success rate: {summary['success_rate']:.1%}")

if __name__ == "__main__":
    main()