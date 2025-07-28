#!/usr/bin/env python3
"""
Comprehensive Integration Tests for ALL RAG Pipelines with 1000+ Documents

This test suite provides the harsh reality check of which pipelines actually work
end-to-end with real PMC data. No premature celebration - just facts.

**Objective**: Establish actual production readiness status for all RAG pipelines.

**Test Coverage**:
- BasicRAG, HyDERAG, SQLRAG, ColBERT, GraphRAG, NodeRAG, CRAG, HybridIFind
- Full end-to-end testing: ingestion → embedding → retrieval → generation
- Performance metrics: load times, query response times, success rates
- Real PMC data with 1000+ documents (no synthetic data)
- Comprehensive assertions on actual results

**Success Criteria**:
- Clear identification of working vs broken pipelines
- Performance benchmarks for working pipelines
- Specific error analysis for failing pipelines
- Actionable recommendations for next steps
"""

import pytest
import logging
import time
import json
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict

# Import all available RAG pipelines
from iris_rag.pipelines.basic import BasicRAGPipeline
from iris_rag.pipelines.hyde import HyDERAGPipeline
from iris_rag.pipelines.sql_rag import SQLRAGPipeline
from iris_rag.pipelines.colbert.pipeline import ColBERTRAGPipeline
from iris_rag.pipelines.graphrag import GraphRAGPipeline
from iris_rag.pipelines.noderag import NodeRAGPipeline
from iris_rag.pipelines.crag import CRAGPipeline
from iris_rag.pipelines.hybrid_ifind import HybridIFindRAGPipeline

# Import test infrastructure
from tests.conftest_1000docs import (
    enterprise_iris_connection,
    scale_test_config,
    enterprise_schema_manager,
)

logger = logging.getLogger(__name__)

@dataclass
class PipelineTestResult:
    """Structured result for pipeline testing."""
    pipeline_name: str
    success: bool
    execution_time: float
    error_message: Optional[str] = None
    error_type: Optional[str] = None
    query: Optional[str] = None
    answer: Optional[str] = None
    retrieved_docs_count: int = 0
    performance_metrics: Dict[str, Any] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        if self.performance_metrics is None:
            result['performance_metrics'] = {}
        return result

@dataclass
class ComprehensiveTestReport:
    """Complete test report with all pipeline results."""
    timestamp: str
    total_pipelines_tested: int
    successful_pipelines: int
    failed_pipelines: int
    pipeline_results: List[PipelineTestResult]
    document_count: int
    test_queries: List[str]
    summary: Dict[str, Any]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp,
            'total_pipelines_tested': self.total_pipelines_tested,
            'successful_pipelines': self.successful_pipelines,
            'failed_pipelines': self.failed_pipelines,
            'pipeline_results': [result.to_dict() for result in self.pipeline_results],
            'document_count': self.document_count,
            'test_queries': self.test_queries,
            'summary': self.summary
        }

class ComprehensivePipelineValidator:
    """
    Comprehensive validator for all RAG pipelines.
    
    Tests each pipeline with real data and provides detailed analysis
    of what works and what doesn't.
    """
    
    def __init__(self, iris_connection, config_manager, embedding_func, llm_func, colbert_query_encoder):
        self.iris_connection = iris_connection
        self.config_manager = config_manager
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        self.colbert_query_encoder = colbert_query_encoder
        self.results: List[PipelineTestResult] = []
        
        # Test queries for validation
        self.test_queries = [
            "What are the effects of metformin on type 2 diabetes?",
            "How does SGLT2 inhibition affect kidney function?",
            "What is the mechanism of action of GLP-1 receptor agonists?"
        ]
        
        # Pipeline configurations
        self.pipeline_configs = {
            'BasicRAG': {
                'class': BasicRAGPipeline,
                'init_args': {
                    'config_manager': self.config_manager,
                    'llm_func': self.llm_func
                }
            },
            'HyDERAG': {
                'class': HyDERAGPipeline,
                'init_args': {
                    'config_manager': self.config_manager,
                    'llm_func': self.llm_func
                }
            },
            'SQLRAG': {
                'class': SQLRAGPipeline,
                'init_args': {
                    'config_manager': self.config_manager,
                    'llm_func': self.llm_func
                }
            },
            'ColBERT': {
                'class': ColBERTRAGPipeline,
                'init_args': {
                    'iris_connector': self.iris_connection,
                    'config_manager': self.config_manager,
                    'colbert_query_encoder': self.colbert_query_encoder,
                    'llm_func': self.llm_func,
                    'embedding_func': self.embedding_func
                }
            },
            'GraphRAG': {
                'class': GraphRAGPipeline,
                'init_args': {
                    'config_manager': self.config_manager,
                    'llm_func': self.llm_func
                }
            },
            'NodeRAG': {
                'class': NodeRAGPipeline,
                'init_args': {
                    'config_manager': self.config_manager,
                    'llm_func': self.llm_func
                }
            },
            'CRAG': {
                'class': CRAGPipeline,
                'init_args': {
                    'config_manager': self.config_manager,
                    'llm_func': self.llm_func
                }
            },
            'HybridIFind': {
                'class': HybridIFindRAGPipeline,
                'init_args': {
                    'config_manager': self.config_manager,
                    'llm_func': self.llm_func
                }
            }
        }
    
    def validate_document_count(self) -> int:
        """Validate that we have at least 1000 documents for testing."""
        cursor = self.iris_connection.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
            doc_count = cursor.fetchone()[0]
            logger.info(f"Found {doc_count} documents in database")
            
            if doc_count < 1000:
                pytest.skip(f"Need at least 1000 documents for comprehensive testing, found {doc_count}")
            
            return doc_count
        finally:
            cursor.close()
    
    def test_pipeline(self, pipeline_name: str, query: str) -> PipelineTestResult:
        """Test a single pipeline with comprehensive error handling."""
        logger.info(f"Testing {pipeline_name} with query: '{query[:50]}...'")
        
        start_time = time.time()
        
        try:
            # Initialize pipeline
            config = self.pipeline_configs[pipeline_name]
            pipeline = config['class'](**config['init_args'])
            
            # Execute pipeline
            result = pipeline.run(query, top_k=5)
            
            execution_time = time.time() - start_time
            
            # Validate result structure
            self._validate_result_structure(result, pipeline_name)
            
            # Extract metrics
            retrieved_docs_count = len(result.get('retrieved_documents', []))
            answer = result.get('answer', '')
            
            # Performance metrics
            performance_metrics = {
                'execution_time': execution_time,
                'retrieved_docs_count': retrieved_docs_count,
                'answer_length': len(answer) if answer else 0,
                'has_answer': bool(answer and answer.strip()),
                'has_retrieved_docs': retrieved_docs_count > 0
            }
            
            logger.info(f"✅ {pipeline_name} succeeded in {execution_time:.2f}s")
            
            return PipelineTestResult(
                pipeline_name=pipeline_name,
                success=True,
                execution_time=execution_time,
                query=query,
                answer=answer,
                retrieved_docs_count=retrieved_docs_count,
                performance_metrics=performance_metrics
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            error_message = str(e)
            error_type = type(e).__name__
            
            logger.error(f"❌ {pipeline_name} failed after {execution_time:.2f}s: {error_message}")
            logger.error(f"Full traceback: {traceback.format_exc()}")
            
            return PipelineTestResult(
                pipeline_name=pipeline_name,
                success=False,
                execution_time=execution_time,
                error_message=error_message,
                error_type=error_type,
                query=query
            )
    
    def _validate_result_structure(self, result: Dict[str, Any], pipeline_name: str):
        """Validate that pipeline result has expected structure."""
        required_fields = ['query', 'answer', 'retrieved_documents']
        
        for field in required_fields:
            if field not in result:
                raise ValueError(f"{pipeline_name} result missing required field: {field}")
        
        # Additional validations
        if not isinstance(result['retrieved_documents'], list):
            raise ValueError(f"{pipeline_name} retrieved_documents must be a list")
        
        if not isinstance(result['answer'], str):
            raise ValueError(f"{pipeline_name} answer must be a string")
    
    def run_comprehensive_validation(self) -> ComprehensiveTestReport:
        """Run comprehensive validation of all pipelines."""
        logger.info("🚀 Starting comprehensive pipeline validation with 1000+ documents")
        
        # Validate document count
        doc_count = self.validate_document_count()
        
        # Test each pipeline with each query
        all_results = []
        
        for pipeline_name in self.pipeline_configs.keys():
            for query in self.test_queries:
                result = self.test_pipeline(pipeline_name, query)
                all_results.append(result)
        
        # Generate summary statistics
        successful_tests = [r for r in all_results if r.success]
        failed_tests = [r for r in all_results if not r.success]
        
        # Group by pipeline
        pipeline_summary = {}
        for pipeline_name in self.pipeline_configs.keys():
            pipeline_results = [r for r in all_results if r.pipeline_name == pipeline_name]
            successful_pipeline_results = [r for r in pipeline_results if r.success]
            
            pipeline_summary[pipeline_name] = {
                'total_tests': len(pipeline_results),
                'successful_tests': len(successful_pipeline_results),
                'success_rate': len(successful_pipeline_results) / len(pipeline_results) if pipeline_results else 0,
                'avg_execution_time': sum(r.execution_time for r in successful_pipeline_results) / len(successful_pipeline_results) if successful_pipeline_results else 0,
                'status': 'WORKING' if len(successful_pipeline_results) == len(pipeline_results) else 'BROKEN' if len(successful_pipeline_results) == 0 else 'PARTIAL'
            }
        
        # Create comprehensive report
        report = ComprehensiveTestReport(
            timestamp=datetime.now().isoformat(),
            total_pipelines_tested=len(self.pipeline_configs),
            successful_pipelines=len([p for p in pipeline_summary.values() if p['status'] == 'WORKING']),
            failed_pipelines=len([p for p in pipeline_summary.values() if p['status'] == 'BROKEN']),
            pipeline_results=all_results,
            document_count=doc_count,
            test_queries=self.test_queries,
            summary={
                'pipeline_summary': pipeline_summary,
                'total_tests_run': len(all_results),
                'total_successful_tests': len(successful_tests),
                'total_failed_tests': len(failed_tests),
                'overall_success_rate': len(successful_tests) / len(all_results) if all_results else 0
            }
        )
        
        return report

# Test fixtures from conftest_1000docs.py
@pytest.fixture
def comprehensive_validator(enterprise_iris_connection, scale_test_config, 
                           embedding_model_fixture, llm_client_fixture, colbert_query_encoder):
    """Create comprehensive pipeline validator with all dependencies."""
    return ComprehensivePipelineValidator(
        iris_connection=enterprise_iris_connection,
        config_manager=scale_test_config['config_manager'],
        embedding_func=embedding_model_fixture,
        llm_func=llm_client_fixture,
        colbert_query_encoder=colbert_query_encoder
    )

# Main test cases
@pytest.mark.requires_real_data
@pytest.mark.requires_1000_docs
@pytest.mark.force_real
class TestComprehensivePipelineValidation:
    """Comprehensive validation test suite for all RAG pipelines."""
    
    def test_all_pipelines_comprehensive_validation(self, comprehensive_validator):
        """
        THE MAIN TEST: Comprehensive validation of all RAG pipelines with 1000+ documents.
        
        This test provides the harsh reality of which pipelines actually work.
        No skipping, no mocking, no premature celebration.
        """
        logger.info("🎯 Running comprehensive validation of ALL RAG pipelines")
        
        # Run the comprehensive validation
        report = comprehensive_validator.run_comprehensive_validation()
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = Path("test_output") / f"comprehensive_pipeline_validation_{timestamp}.json"
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        logger.info(f"📊 Detailed results saved to: {results_file}")
        
        # Print summary to console
        self._print_validation_summary(report)
        
        # Assertions for test validation
        assert report.total_pipelines_tested > 0, "No pipelines were tested"
        assert report.document_count >= 1000, f"Need at least 1000 documents, found {report.document_count}"
        
        # The harsh reality check - we want to know what's broken
        working_pipelines = [p for p, data in report.summary['pipeline_summary'].items() if data['status'] == 'WORKING']
        broken_pipelines = [p for p, data in report.summary['pipeline_summary'].items() if data['status'] == 'BROKEN']
        partial_pipelines = [p for p, data in report.summary['pipeline_summary'].items() if data['status'] == 'PARTIAL']
        
        logger.info(f"🟢 WORKING PIPELINES ({len(working_pipelines)}): {working_pipelines}")
        logger.info(f"🔴 BROKEN PIPELINES ({len(broken_pipelines)}): {broken_pipelines}")
        logger.info(f"🟡 PARTIAL PIPELINES ({len(partial_pipelines)}): {partial_pipelines}")
        
        # Generate actionable recommendations
        recommendations = self._generate_recommendations(report)
        logger.info("📋 ACTIONABLE RECOMMENDATIONS:")
        for rec in recommendations:
            logger.info(f"  • {rec}")
        
        # The test passes if we successfully tested all pipelines and got results
        # We don't fail the test for broken pipelines - we document them
        assert len(report.pipeline_results) > 0, "No pipeline results generated"
        
        # But we do want to ensure at least some pipelines work
        if len(working_pipelines) == 0:
            pytest.fail("CRITICAL: No pipelines are working! All pipelines failed validation.")
    
    def _print_validation_summary(self, report: ComprehensiveTestReport):
        """Print a comprehensive summary of validation results."""
        logger.info("=" * 80)
        logger.info("🎯 COMPREHENSIVE PIPELINE VALIDATION RESULTS")
        logger.info("=" * 80)
        logger.info(f"📅 Timestamp: {report.timestamp}")
        logger.info(f"📊 Documents tested: {report.document_count:,}")
        logger.info(f"🧪 Total pipelines tested: {report.total_pipelines_tested}")
        logger.info(f"✅ Successful pipelines: {report.successful_pipelines}")
        logger.info(f"❌ Failed pipelines: {report.failed_pipelines}")
        logger.info(f"📈 Overall success rate: {report.summary['overall_success_rate']:.1%}")
        logger.info("")
        
        logger.info("📋 PIPELINE-BY-PIPELINE BREAKDOWN:")
        for pipeline_name, data in report.summary['pipeline_summary'].items():
            status_emoji = "🟢" if data['status'] == 'WORKING' else "🔴" if data['status'] == 'BROKEN' else "🟡"
            logger.info(f"  {status_emoji} {pipeline_name}: {data['success_rate']:.1%} success rate, "
                       f"avg {data['avg_execution_time']:.2f}s")
        
        logger.info("")
        logger.info("🔍 DETAILED FAILURE ANALYSIS:")
        failed_results = [r for r in report.pipeline_results if not r.success]
        if failed_results:
            for result in failed_results:
                logger.info(f"  ❌ {result.pipeline_name}: {result.error_type} - {result.error_message}")
        else:
            logger.info("  🎉 No failures detected!")
        
        logger.info("=" * 80)
    
    def _generate_recommendations(self, report: ComprehensiveTestReport) -> List[str]:
        """Generate actionable recommendations based on test results."""
        recommendations = []
        
        # Analyze broken pipelines
        broken_pipelines = [p for p, data in report.summary['pipeline_summary'].items() if data['status'] == 'BROKEN']
        if broken_pipelines:
            recommendations.append(f"Fix broken pipelines: {', '.join(broken_pipelines)}")
        
        # Analyze partial pipelines
        partial_pipelines = [p for p, data in report.summary['pipeline_summary'].items() if data['status'] == 'PARTIAL']
        if partial_pipelines:
            recommendations.append(f"Investigate intermittent failures in: {', '.join(partial_pipelines)}")
        
        # Performance recommendations
        working_pipelines = {p: data for p, data in report.summary['pipeline_summary'].items() if data['status'] == 'WORKING'}
        if working_pipelines:
            fastest = min(working_pipelines.items(), key=lambda x: x[1]['avg_execution_time'])
            slowest = max(working_pipelines.items(), key=lambda x: x[1]['avg_execution_time'])
            
            if slowest[1]['avg_execution_time'] > fastest[1]['avg_execution_time'] * 3:
                recommendations.append(f"Optimize performance of {slowest[0]} (avg {slowest[1]['avg_execution_time']:.2f}s vs {fastest[0]} at {fastest[1]['avg_execution_time']:.2f}s)")
        
        # Error pattern analysis
        error_types = {}
        for result in report.pipeline_results:
            if not result.success and result.error_type:
                error_types[result.error_type] = error_types.get(result.error_type, 0) + 1
        
        if error_types:
            most_common_error = max(error_types.items(), key=lambda x: x[1])
            recommendations.append(f"Address common error pattern: {most_common_error[0]} (occurs {most_common_error[1]} times)")
        
        # Success rate recommendations
        if report.summary['overall_success_rate'] < 0.8:
            recommendations.append("Overall success rate below 80% - prioritize stability improvements")
        
        return recommendations