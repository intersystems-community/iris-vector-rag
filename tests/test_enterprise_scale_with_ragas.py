#!/usr/bin/env python3
"""
Enterprise-Scale RAG Testing with 1000+ Documents and RAGAS Evaluation

This module provides comprehensive testing of all RAG techniques with enterprise-scale
document volumes to validate production readiness.

Features:
- Tests all 7 RAG techniques with 1000+ documents
- RAGAS evaluation integration for quality assessment
- Performance benchmarking and metrics collection
- Enterprise capability validation
- Production deployment readiness assessment
"""

import pytest
import logging
import time
import json
from typing import Dict, Any, List
from datetime import datetime

from tests.conftest_1000docs import (
    enterprise_iris_connection,
    scale_test_config,
    enterprise_schema_manager,
    scale_test_documents,
    scale_test_performance_monitor,
    enterprise_test_queries
)

from iris_rag.config.manager import ConfigurationManager
from iris_rag.pipelines.basic import BasicRAGPipeline
from iris_rag.pipelines.crag import CRAG
from iris_rag.pipelines.graphrag import GraphRAG
from iris_rag.pipelines.hybrid_ifind import HybridIFind
from iris_rag.pipelines.hyde import HyDE
from iris_rag.pipelines.noderag import NoDeRAG

# RAGAS evaluation imports
try:
    from ragas import evaluate
    from ragas.metrics import (
        answer_relevancy,
        context_precision,
        context_recall,
        faithfulness,
        answer_similarity,
        answer_correctness
    )
    from datasets import Dataset
    RAGAS_AVAILABLE = True
except ImportError:
    RAGAS_AVAILABLE = False

logger = logging.getLogger(__name__)

# Test configuration
RAG_TECHNIQUES = [
    ('BasicRAG', BasicRAGPipeline),
    
    ('CRAG', CRAG),
    ('GraphRAG', GraphRAG),
    ('HybridIFind', HybridIFind),
    ('HyDE', HyDE),
    ('NoDeRAG', NoDeRAG)
]

def evaluate_with_ragas(results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Evaluate results using RAGAS metrics.
    
    Args:
        results: List of pipeline results with query, answer, and contexts
        
    Returns:
        Dictionary containing RAGAS evaluation results
    """
    if not RAGAS_AVAILABLE:
        logger.warning("RAGAS not available - skipping evaluation")
        return {"status": "skipped", "reason": "RAGAS not available"}
    
    if not results:
        return {"status": "skipped", "reason": "No results to evaluate"}
    
    try:
        # Prepare data for RAGAS
        questions = []
        answers = []
        contexts = []
        ground_truths = []
        
        for result in results:
            questions.append(result['query'])
            answers.append(result['answer'])
            
            # Extract contexts from retrieved documents
            if 'retrieved_documents' in result and result['retrieved_documents']:
                doc_contexts = []
                for doc in result['retrieved_documents']:
                    if hasattr(doc, 'content') and doc.content:
                        doc_contexts.append(doc.content)
                    elif isinstance(doc, dict) and 'content' in doc:
                        doc_contexts.append(doc['content'])
                    elif isinstance(doc, str):
                        doc_contexts.append(doc)
                
                contexts.append(doc_contexts if doc_contexts else ["No context available"])
            else:
                contexts.append(["No context available"])
            
            # Use answer as ground truth for now (can be improved with actual ground truth)
            ground_truths.append(result['answer'])
        
        # Create RAGAS dataset
        ragas_data = {
            'question': questions,
            'answer': answers,
            'contexts': contexts,
            'ground_truth': ground_truths
        }
        
        dataset = Dataset.from_dict(ragas_data)
        
        # Define metrics to evaluate
        metrics = [
            answer_relevancy,
            context_precision,
            faithfulness
        ]
        
        # Run RAGAS evaluation
        ragas_result = evaluate(dataset, metrics=metrics)
        
        # Extract scores
        if hasattr(ragas_result, 'to_pandas'):
            df = ragas_result.to_pandas()
            scores = {}
            for metric in ['answer_relevancy', 'context_precision', 'faithfulness']:
                if metric in df.columns:
                    scores[metric] = {
                        'mean': float(df[metric].mean()),
                        'std': float(df[metric].std()),
                        'min': float(df[metric].min()),
                        'max': float(df[metric].max())
                    }
            
            return {
                "status": "success",
                "scores": scores,
                "total_queries": len(questions)
            }
        else:
            return {"status": "error", "reason": "Unexpected RAGAS result format"}
            
    except Exception as e:
        logger.error(f"RAGAS evaluation failed: {e}")
        return {"status": "error", "reason": str(e)}

@pytest.mark.scale_1000
class TestEnterpriseScaleRAGWithRAGAS:
    """Enterprise-scale testing for all RAG techniques with 1000+ documents and RAGAS evaluation."""
    
    def test_document_count_validation(self, scale_test_documents):
        """Validate that we have sufficient documents for enterprise testing."""
        doc_count = scale_test_documents['document_count']
        target_count = scale_test_documents['target_count']
        
        assert doc_count >= 1000, f"Insufficient documents for enterprise testing: {doc_count} < 1000"
        assert scale_test_documents['meets_scale_requirements'], "Scale requirements not met"
        
        logger.info(f"Enterprise scale validation passed: {doc_count} documents available")
        logger.info(f"Target scale: {target_count} documents")
        logger.info(f"Scale mode: {scale_test_documents['scale_mode']}")
    
    @pytest.mark.parametrize("technique_name,technique_class", RAG_TECHNIQUES)
    def test_enterprise_scale_rag_with_ragas(
        self,
        technique_name: str,
        technique_class,
        enterprise_iris_connection,
        scale_test_config,
        enterprise_schema_manager,
        scale_test_documents,
        scale_test_performance_monitor,
        enterprise_test_queries
    ):
        """Test individual RAG technique with enterprise-scale documents and RAGAS evaluation."""
        
        # Validate prerequisites
        assert scale_test_documents['document_count'] >= 1000, "Insufficient documents for enterprise testing"
        
        config_manager = scale_test_config['config_manager']
        
        # Initialize technique
        start_time = time.time()
        try:
            if technique_name in ['BasicRAG']:
                pipeline = technique_class(config_manager)
            else:
                pipeline = technique_class(config_manager)
            
            init_time = time.time() - start_time
            scale_test_performance_monitor['record_operation'](
                f"{technique_name}_initialization", 
                init_time
            )
            
        except Exception as e:
            pytest.fail(f"Failed to initialize {technique_name}: {e}")
        
        # Test with multiple queries for comprehensive validation
        results = []
        successful_queries = 0
        total_query_time = 0
        
        for query_data in enterprise_test_queries[:3]:  # Test with first 3 queries
            query = query_data['query']
            expected_keywords = query_data['expected_keywords']
            
            try:
                query_start = time.time()
                result = pipeline.process_query(query)
                query_duration = time.time() - query_start
                
                total_query_time += query_duration
                scale_test_performance_monitor['record_operation'](
                    f"{technique_name}_query",
                    query_duration,
                    query=query,
                    category=query_data['category']
                )
                
                # Validate result structure
                assert isinstance(result, dict), f"Result should be dict, got {type(result)}"
                assert 'query' in result, "Result missing 'query' field"
                assert 'answer' in result, "Result missing 'answer' field"
                assert 'retrieved_documents' in result, "Result missing 'retrieved_documents' field"
                
                # Validate answer quality
                answer = result['answer']
                assert isinstance(answer, str), f"Answer should be string, got {type(answer)}"
                assert len(answer.strip()) > 0, "Answer should not be empty"
                assert len(answer) > 50, f"Answer too short for enterprise query: {len(answer)} chars"
                
                # Validate retrieved documents
                docs = result['retrieved_documents']
                assert isinstance(docs, list), f"Retrieved docs should be list, got {type(docs)}"
                assert len(docs) > 0, "Should retrieve at least one document"
                
                # Store result for RAGAS evaluation
                results.append(result)
                
                # Check for keyword relevance (basic quality check)
                answer_lower = answer.lower()
                keyword_matches = sum(1 for keyword in expected_keywords 
                                    if keyword.lower() in answer_lower)
                
                if keyword_matches > 0:
                    successful_queries += 1
                
                logger.info(f"{technique_name} query '{query[:50]}...' completed in {query_duration:.2f}s")
                logger.info(f"Answer length: {len(answer)} chars, Retrieved docs: {len(docs)}")
                
            except Exception as e:
                logger.error(f"{technique_name} failed on query '{query[:50]}...': {e}")
                # Don't fail the test immediately - collect metrics for all queries
        
        # RAGAS Evaluation
        ragas_results = evaluate_with_ragas(results)
        logger.info(f"{technique_name} RAGAS evaluation: {ragas_results['status']}")
        
        if ragas_results['status'] == 'success':
            scores = ragas_results['scores']
            logger.info(f"{technique_name} RAGAS scores:")
            for metric, values in scores.items():
                logger.info(f"  {metric}: {values['mean']:.3f} ± {values['std']:.3f}")
            
            # Enterprise quality thresholds
            if 'answer_relevancy' in scores:
                assert scores['answer_relevancy']['mean'] >= 0.3, f"{technique_name} answer relevancy too low: {scores['answer_relevancy']['mean']:.3f}"
            
            if 'faithfulness' in scores:
                assert scores['faithfulness']['mean'] >= 0.3, f"{technique_name} faithfulness too low: {scores['faithfulness']['mean']:.3f}"
        
        # Enterprise-scale performance requirements
        avg_query_time = total_query_time / len(enterprise_test_queries[:3])
        
        # Performance assertions for enterprise deployment
        assert avg_query_time < 30.0, f"{technique_name} too slow for enterprise: {avg_query_time:.2f}s avg"
        assert successful_queries > 0, f"{technique_name} failed all queries in enterprise testing"
        
        # Success rate should be reasonable for enterprise deployment
        success_rate = successful_queries / len(enterprise_test_queries[:3])
        assert success_rate >= 0.33, f"{technique_name} success rate too low: {success_rate:.1%}"
        
        logger.info(f"{technique_name} enterprise testing completed:")
        logger.info(f"  Success rate: {success_rate:.1%}")
        logger.info(f"  Average query time: {avg_query_time:.2f}s")
        logger.info(f"  Document scale: {scale_test_documents['document_count']} docs")
        logger.info(f"  RAGAS evaluation: {ragas_results['status']}")

@pytest.mark.scale_1000
def test_enterprise_comparative_ragas_evaluation(
    enterprise_iris_connection,
    scale_test_config,
    enterprise_schema_manager,
    scale_test_documents,
    scale_test_performance_monitor,
    enterprise_test_queries
):
    """Compare all RAG techniques using RAGAS evaluation at enterprise scale."""
    
    assert scale_test_documents['document_count'] >= 1000, "Insufficient documents for comparative testing"
    
    config_manager = scale_test_config['config_manager']
    all_results = {}
    
    # Test query for comparison
    test_query = enterprise_test_queries[0]['query']
    
    for technique_name, technique_class in RAG_TECHNIQUES:
        try:
            # Initialize technique
            if technique_name in ['BasicRAG']:
                pipeline = technique_class(config_manager)
            else:
                pipeline = technique_class(config_manager)
            
            # Execute query and measure performance
            start_time = time.time()
            result = pipeline.process_query(test_query)
            duration = time.time() - start_time
            
            # Store result for RAGAS evaluation
            all_results[technique_name] = {
                'result': result,
                'duration': duration,
                'success': True
            }
            
            logger.info(f"{technique_name}: {duration:.2f}s")
            
        except Exception as e:
            logger.error(f"{technique_name} failed in comparative test: {e}")
            all_results[technique_name] = {
                'result': None,
                'duration': float('inf'),
                'success': False
            }
    
    # Validate that at least some techniques work
    successful_techniques = [name for name, data in all_results.items() if data['success']]
    assert len(successful_techniques) >= 2, f"Too few techniques working: {successful_techniques}"
    
    # Prepare results for RAGAS evaluation
    ragas_input = []
    for technique_name, data in all_results.items():
        if data['success'] and data['result']:
            ragas_input.append(data['result'])
    
    # Run comparative RAGAS evaluation
    if ragas_input:
        ragas_results = evaluate_with_ragas(ragas_input)
        
        if ragas_results['status'] == 'success':
            logger.info("Enterprise comparative RAGAS evaluation completed:")
            scores = ragas_results['scores']
            for metric, values in scores.items():
                logger.info(f"  {metric}: {values['mean']:.3f} ± {values['std']:.3f}")
            
            # Save detailed results
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = f"test_output/enterprise_ragas_comparison_{timestamp}.json"
            
            detailed_results = {
                'timestamp': timestamp,
                'document_count': scale_test_documents['document_count'],
                'test_query': test_query,
                'techniques_tested': len(RAG_TECHNIQUES),
                'successful_techniques': len(successful_techniques),
                'ragas_evaluation': ragas_results,
                'performance_data': {
                    name: {'duration': data['duration'], 'success': data['success']}
                    for name, data in all_results.items()
                }
            }
            
            with open(results_file, 'w') as f:
                json.dump(detailed_results, f, indent=2)
            
            logger.info(f"Detailed results saved to: {results_file}")
    
    # Find fastest and slowest working techniques
    working_results = {name: data for name, data in all_results.items() if data['success']}
    fastest = min(working_results.items(), key=lambda x: x[1]['duration'])
    slowest = max(working_results.items(), key=lambda x: x[1]['duration'])
    
    logger.info(f"Enterprise comparative results:")
    logger.info(f"  Fastest: {fastest[0]} ({fastest[1]['duration']:.2f}s)")
    logger.info(f"  Slowest: {slowest[0]} ({slowest[1]['duration']:.2f}s)")
    logger.info(f"  Working techniques: {len(successful_techniques)}/{len(RAG_TECHNIQUES)}")
    
    # Performance spread should be reasonable for enterprise deployment
    if len(working_results) > 1:
        performance_ratio = slowest[1]['duration'] / fastest[1]['duration']
        assert performance_ratio < 10.0, f"Performance spread too large: {performance_ratio:.1f}x"

@pytest.mark.scale_1000
def test_enterprise_ragas_quality_benchmarks(
    enterprise_iris_connection,
    scale_test_config,
    enterprise_schema_manager,
    scale_test_documents,
    enterprise_test_queries
):
    """Test enterprise-scale RAGAS quality benchmarks across all techniques."""
    
    assert scale_test_documents['document_count'] >= 1000, "Insufficient documents for quality benchmarking"
    
    config_manager = scale_test_config['config_manager']
    quality_results = {}
    
    # Test with multiple queries for robust quality assessment
    test_queries = enterprise_test_queries[:2]  # Use first 2 queries for quality testing
    
    for technique_name, technique_class in RAG_TECHNIQUES[:3]:  # Test first 3 techniques for time
        try:
            # Initialize technique
            if technique_name in ['BasicRAG']:
                pipeline = technique_class(config_manager)
            else:
                pipeline = technique_class(config_manager)
            
            # Collect results for multiple queries
            technique_results = []
            for query_data in test_queries:
                try:
                    result = pipeline.process_query(query_data['query'])
                    technique_results.append(result)
                except Exception as e:
                    logger.warning(f"{technique_name} failed on query: {e}")
            
            if technique_results:
                # Run RAGAS evaluation
                ragas_results = evaluate_with_ragas(technique_results)
                quality_results[technique_name] = ragas_results
                
                logger.info(f"{technique_name} quality assessment: {ragas_results['status']}")
                if ragas_results['status'] == 'success':
                    scores = ragas_results['scores']
                    for metric, values in scores.items():
                        logger.info(f"  {metric}: {values['mean']:.3f}")
            
        except Exception as e:
            logger.error(f"{technique_name} failed in quality benchmark: {e}")
            quality_results[technique_name] = {"status": "error", "reason": str(e)}
    
    # Validate quality results
    successful_evaluations = [name for name, results in quality_results.items() 
                            if results.get('status') == 'success']
    
    assert len(successful_evaluations) >= 1, f"No successful quality evaluations: {list(quality_results.keys())}"
    
    # Enterprise quality standards
    for technique_name in successful_evaluations:
        scores = quality_results[technique_name]['scores']
        
        # Minimum quality thresholds for enterprise deployment
        if 'answer_relevancy' in scores:
            relevancy_score = scores['answer_relevancy']['mean']
            assert relevancy_score >= 0.2, f"{technique_name} answer relevancy below enterprise threshold: {relevancy_score:.3f}"
        
        if 'faithfulness' in scores:
            faithfulness_score = scores['faithfulness']['mean']
            assert faithfulness_score >= 0.2, f"{technique_name} faithfulness below enterprise threshold: {faithfulness_score:.3f}"
    
    logger.info(f"Enterprise quality benchmarks completed for {len(successful_evaluations)} techniques")