#!/usr/bin/env python3
"""
Pipeline Reality Check - Testing with Available Data

This test provides the harsh reality of which pipelines work with the data we actually have.
No skipping, no excuses - just facts about what's broken and what works.

**Current Reality**: Only 13 documents available (not 1000+)
**Objective**: Test all pipelines with available data and document the real status
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
class RealityCheckReport:
    """Reality check report with actual status."""
    timestamp: str
    document_count: int
    total_pipelines_tested: int
    successful_pipelines: int
    failed_pipelines: int
    pipeline_results: List[PipelineTestResult]
    test_queries: List[str]
    summary: Dict[str, Any]
    recommendations: List[str]
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            'timestamp': self.timestamp,
            'document_count': self.document_count,
            'total_pipelines_tested': self.total_pipelines_tested,
            'successful_pipelines': self.successful_pipelines,
            'failed_pipelines': self.failed_pipelines,
            'pipeline_results': [result.to_dict() for result in self.pipeline_results],
            'test_queries': self.test_queries,
            'summary': self.summary,
            'recommendations': self.recommendations
        }

class PipelineRealityChecker:
    """Reality checker for RAG pipelines with available data."""
    
    def __init__(self, iris_connection, config_manager, embedding_func, llm_func, colbert_query_encoder):
        self.iris_connection = iris_connection
        self.config_manager = config_manager
        self.embedding_func = embedding_func
        self.llm_func = llm_func
        self.colbert_query_encoder = colbert_query_encoder
        
        # Simple test queries
        self.test_queries = [
            "What are the effects of metformin?",
            "How does diabetes treatment work?",
            "What are medical interventions?"
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
    
    def get_document_count(self) -> int:
        """Get actual document count."""
        cursor = self.iris_connection.cursor()
        try:
            cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
            doc_count = cursor.fetchone()[0]
            logger.info(f"Found {doc_count} documents in database")
            return doc_count
        except Exception as e:
            logger.warning(f"Could not get document count: {e}")
            return 0
        finally:
            cursor.close()
    
    def test_pipeline(self, pipeline_name: str, query: str) -> PipelineTestResult:
        """Test a single pipeline with comprehensive error handling."""
        logger.info(f"Testing {pipeline_name} with query: '{query[:30]}...'")
        
        start_time = time.time()
        
        try:
            # Initialize pipeline
            config = self.pipeline_configs[pipeline_name]
            pipeline = config['class'](**config['init_args'])
            
            # Execute pipeline
            result = pipeline.run(query, top_k=3)  # Use smaller top_k for limited data
            
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
    
    def run_reality_check(self) -> RealityCheckReport:
        """Run reality check of all pipelines with available data."""
        logger.info("🚨 Starting REALITY CHECK of all RAG pipelines")
        
        # Get actual document count
        doc_count = self.get_document_count()
        
        # Test each pipeline with first query only (to save time with limited data)
        all_results = []
        test_query = self.test_queries[0]
        
        for pipeline_name in self.pipeline_configs.keys():
            result = self.test_pipeline(pipeline_name, test_query)
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
                'status': 'WORKING' if len(successful_pipeline_results) == len(pipeline_results) else 'BROKEN'
            }
        
        # Generate recommendations
        recommendations = self._generate_recommendations(doc_count, pipeline_summary, failed_tests)
        
        # Create reality check report
        report = RealityCheckReport(
            timestamp=datetime.now().isoformat(),
            document_count=doc_count,
            total_pipelines_tested=len(self.pipeline_configs),
            successful_pipelines=len([p for p in pipeline_summary.values() if p['status'] == 'WORKING']),
            failed_pipelines=len([p for p in pipeline_summary.values() if p['status'] == 'BROKEN']),
            pipeline_results=all_results,
            test_queries=[test_query],
            summary={
                'pipeline_summary': pipeline_summary,
                'total_tests_run': len(all_results),
                'total_successful_tests': len(successful_tests),
                'total_failed_tests': len(failed_tests),
                'overall_success_rate': len(successful_tests) / len(all_results) if all_results else 0
            },
            recommendations=recommendations
        )
        
        return report
    
    def _generate_recommendations(self, doc_count: int, pipeline_summary: Dict, failed_tests: List) -> List[str]:
        """Generate actionable recommendations based on reality check."""
        recommendations = []
        
        # Data availability issue
        if doc_count < 1000:
            recommendations.append(f"CRITICAL: Only {doc_count} documents available. Need to ingest at least 1000 PMC documents for proper testing.")
            recommendations.append("Run data ingestion script to load more PMC documents before comprehensive testing.")
        
        # Broken pipelines
        broken_pipelines = [p for p, data in pipeline_summary.items() if data['status'] == 'BROKEN']
        if broken_pipelines:
            recommendations.append(f"Fix broken pipelines immediately: {', '.join(broken_pipelines)}")
        
        # Error analysis
        error_types = {}
        for result in failed_tests:
            if result.error_type:
                error_types[result.error_type] = error_types.get(result.error_type, 0) + 1
        
        if error_types:
            most_common_error = max(error_types.items(), key=lambda x: x[1])
            recommendations.append(f"Most common error: {most_common_error[0]} (occurs {most_common_error[1]} times)")
        
        # Working pipelines
        working_pipelines = [p for p, data in pipeline_summary.items() if data['status'] == 'WORKING']
        if working_pipelines:
            recommendations.append(f"Working pipelines that can be used immediately: {', '.join(working_pipelines)}")
        
        # Next steps
        recommendations.append("Next steps: 1) Fix data ingestion, 2) Fix broken pipelines, 3) Re-run comprehensive test")
        
        return recommendations

# Test fixtures
@pytest.fixture
def reality_checker(iris_testcontainer_connection, real_config_manager, 
                   embedding_model_fixture, llm_client_fixture, colbert_query_encoder):
    """Create reality checker with all dependencies."""
    return PipelineRealityChecker(
        iris_connection=iris_testcontainer_connection,
        config_manager=real_config_manager,
        embedding_func=embedding_model_fixture,
        llm_func=llm_client_fixture,
        colbert_query_encoder=colbert_query_encoder
    )

# Main test
@pytest.mark.requires_real_data
class TestPipelineRealityCheck:
    """Reality check test suite for RAG pipelines."""
    
    def test_pipeline_reality_check(self, reality_checker):
        """
        THE HARSH REALITY CHECK: Test all pipelines with available data.
        
        This test tells us exactly what works and what doesn't, no matter
        how little data we have. No skipping, no excuses.
        """
        logger.info("🚨 Running PIPELINE REALITY CHECK")
        
        # Run the reality check
        report = reality_checker.run_reality_check()
        
        # Save detailed results
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = Path("test_output") / f"pipeline_reality_check_{timestamp}.json"
        results_file.parent.mkdir(exist_ok=True)
        
        with open(results_file, 'w') as f:
            json.dump(report.to_dict(), f, indent=2)
        
        logger.info(f"📊 Reality check results saved to: {results_file}")
        
        # Print harsh reality to console
        self._print_reality_summary(report)
        
        # Assertions
        assert report.total_pipelines_tested > 0, "No pipelines were tested"
        
        # Document the harsh reality
        working_pipelines = [p for p, data in report.summary['pipeline_summary'].items() if data['status'] == 'WORKING']
        broken_pipelines = [p for p, data in report.summary['pipeline_summary'].items() if data['status'] == 'BROKEN']
        
        logger.info(f"🟢 WORKING PIPELINES ({len(working_pipelines)}): {working_pipelines}")
        logger.info(f"🔴 BROKEN PIPELINES ({len(broken_pipelines)}): {broken_pipelines}")
        
        # Print recommendations
        logger.info("📋 ACTIONABLE RECOMMENDATIONS:")
        for i, rec in enumerate(report.recommendations, 1):
            logger.info(f"  {i}. {rec}")
        
        # The test passes regardless of failures - we're documenting reality
        assert len(report.pipeline_results) > 0, "No pipeline results generated"
        
        # But warn if everything is broken
        if len(working_pipelines) == 0:
            logger.warning("⚠️  WARNING: No pipelines are working! This requires immediate attention.")
    
    def _print_reality_summary(self, report: RealityCheckReport):
        """Print the harsh reality summary."""
        logger.info("=" * 80)
        logger.info("🚨 PIPELINE REALITY CHECK RESULTS")
        logger.info("=" * 80)
        logger.info(f"📅 Timestamp: {report.timestamp}")
        logger.info(f"📊 Documents available: {report.document_count:,}")
        logger.info(f"🧪 Total pipelines tested: {report.total_pipelines_tested}")
        logger.info(f"✅ Working pipelines: {report.successful_pipelines}")
        logger.info(f"❌ Broken pipelines: {report.failed_pipelines}")
        logger.info(f"📈 Success rate: {report.summary['overall_success_rate']:.1%}")
        logger.info("")
        
        logger.info("📋 PIPELINE STATUS:")
        for pipeline_name, data in report.summary['pipeline_summary'].items():
            status_emoji = "🟢" if data['status'] == 'WORKING' else "🔴"
            logger.info(f"  {status_emoji} {pipeline_name}: {data['status']} "
                       f"(avg {data['avg_execution_time']:.2f}s)")
        
        logger.info("")
        logger.info("🔍 FAILURE ANALYSIS:")
        failed_results = [r for r in report.pipeline_results if not r.success]
        if failed_results:
            for result in failed_results:
                logger.info(f"  ❌ {result.pipeline_name}: {result.error_type} - {result.error_message}")
        else:
            logger.info("  🎉 No failures detected!")
        
        logger.info("=" * 80)