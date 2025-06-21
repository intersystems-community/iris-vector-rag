#!/usr/bin/env python3
"""
Test Script for Comprehensive Scaling Evaluation Framework
Validates that all components are working correctly
"""

import sys
import os
import json
import time
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Any

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from eval.scaling_evaluation_framework import ScalingEvaluationFramework
from scripts.automated_dataset_scaling import AutomatedDatasetScaling
from eval.comprehensive_scaling_orchestrator import ComprehensiveScalingOrchestrator
from common.iris_connector_jdbc import get_iris_connection
from dotenv import load_dotenv

load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def test_database_connection():
    """Test database connection and basic queries"""
    logger.info("🔍 Testing database connection...")
    
    try:
        connection = get_iris_connection()
        cursor = connection.cursor()
        
        # Test basic queries
        cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
        doc_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM RAG.DocumentChunks")
        chunk_count = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM RAG.DocumentTokenEmbeddings")
        embedding_count = cursor.fetchone()[0]
        
        cursor.close()
        connection.close()
        
        logger.info(f"✅ Database connection successful")
        logger.info(f"   Documents: {doc_count:,}")
        logger.info(f"   Chunks: {chunk_count:,}")
        logger.info(f"   Embeddings: {embedding_count:,}")
        
        return {
            'success': True,
            'document_count': doc_count,
            'chunk_count': chunk_count,
            'embedding_count': embedding_count
        }
        
    except Exception as e:
        logger.error(f"❌ Database connection failed: {e}")
        return {'success': False, 'error': str(e)}

def test_framework_initialization():
    """Test framework component initialization"""
    logger.info("🔍 Testing framework initialization...")
    
    results = {
        'scaling_framework': False,
        'dataset_scaler': False,
        'orchestrator': False
    }
    
    try:
        # Test ScalingEvaluationFramework
        evaluator = ScalingEvaluationFramework()
        logger.info(f"✅ ScalingEvaluationFramework: {len(evaluator.technique_names)} techniques")
        results['scaling_framework'] = True
        
        # Test AutomatedDatasetScaling
        scaler = AutomatedDatasetScaling()
        logger.info(f"✅ AutomatedDatasetScaling: {len(scaler.target_sizes)} target sizes")
        results['dataset_scaler'] = True
        
        # Test ComprehensiveScalingOrchestrator
        orchestrator = ComprehensiveScalingOrchestrator()
        plan = orchestrator.evaluation_plan
        logger.info(f"✅ ComprehensiveScalingOrchestrator: {len(plan['techniques'])} techniques, {len(plan['dataset_sizes'])} sizes")
        results['orchestrator'] = True
        
    except Exception as e:
        logger.error(f"❌ Framework initialization failed: {e}")
        results['error'] = str(e)
    
    return results

def test_single_technique():
    """Test a single technique to verify pipeline works"""
    logger.info("🔍 Testing single technique (BasicRAG)...")
    
    try:
        evaluator = ScalingEvaluationFramework()
        
        # Initialize BasicRAG pipeline
        pipeline = evaluator.initialize_pipeline('BasicRAG')
        
        if not pipeline:
            logger.error("❌ Failed to initialize BasicRAG pipeline")
            return {'success': False, 'error': 'Pipeline initialization failed'}
        
        # Test with a simple query
        test_query = {
            "query": "What is the role of olfactory perception in honeybee behavior?",
            "ground_truth": "Olfactory perception plays a crucial role in honeybee behavior.",
            "keywords": ["olfactory", "honeybee"],
            "category": "neuroscience"
        }
        
        logger.info("   Running test query...")
        result = evaluator.run_single_query_with_metrics(pipeline, 'BasicRAG', test_query)
        
        if result['success']:
            logger.info(f"✅ BasicRAG test successful")
            logger.info(f"   Response time: {result['response_time']:.2f}s")
            logger.info(f"   Documents retrieved: {result['documents_retrieved']}")
            logger.info(f"   Answer length: {result['answer_length']} chars")
        else:
            logger.error(f"❌ BasicRAG test failed: {result.get('error', 'Unknown error')}")
            return {'success': False, 'error': result.get('error', 'Unknown error')}
        
        return {
            'success': True,
            'response_time': result['response_time'],
            'documents_retrieved': result['documents_retrieved'],
            'answer_length': result['answer_length']
        }
        
    except Exception as e:
        logger.error(f"❌ Single technique test failed: {e}")
        return {'success': False, 'error': str(e)}

def test_ragas_availability():
    """Test RAGAS library availability"""
    logger.info("🔍 Testing RAGAS availability...")
    
    try:
        from ragas import evaluate
        from ragas.metrics import answer_relevancy, context_precision, faithfulness
        logger.info("✅ RAGAS library available")
        
        # Check if OpenAI API key is available
        if os.getenv("OPENAI_API_KEY"):
            logger.info("✅ OpenAI API key configured - real evaluation possible")
            return {'success': True, 'real_llm': True}
        else:
            logger.warning("⚠️ OpenAI API key not found - will use stub LLM")
            return {'success': True, 'real_llm': False}
            
    except ImportError as e:
        logger.warning(f"⚠️ RAGAS not available: {e}")
        logger.info("💡 Install with: pip install ragas datasets")
        return {'success': False, 'error': str(e)}

def test_database_stats():
    """Test database statistics collection"""
    logger.info("🔍 Testing database statistics collection...")
    
    try:
        scaler = AutomatedDatasetScaling()
        stats = scaler.get_database_size_metrics()
        
        if stats:
            logger.info("✅ Database statistics collection successful")
            logger.info(f"   Documents: {stats['document_count']:,}")
            logger.info(f"   Chunks: {stats['chunk_count']:,}")
            logger.info(f"   Content size: {stats['content_size_mb']:.1f} MB")
            return {'success': True, 'stats': stats}
        else:
            logger.error("❌ Failed to collect database statistics")
            return {'success': False, 'error': 'No statistics returned'}
            
    except Exception as e:
        logger.error(f"❌ Database statistics test failed: {e}")
        return {'success': False, 'error': str(e)}

def run_comprehensive_test():
    """Run comprehensive framework test"""
    logger.info("🚀 Starting comprehensive framework test...")
    logger.info(f"📅 Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    test_results = {
        'timestamp': datetime.now().isoformat(),
        'tests': {}
    }
    
    # Test 1: Database Connection
    logger.info("\n" + "="*50)
    logger.info("TEST 1: Database Connection")
    logger.info("="*50)
    test_results['tests']['database_connection'] = test_database_connection()
    
    # Test 2: Framework Initialization
    logger.info("\n" + "="*50)
    logger.info("TEST 2: Framework Initialization")
    logger.info("="*50)
    test_results['tests']['framework_initialization'] = test_framework_initialization()
    
    # Test 3: RAGAS Availability
    logger.info("\n" + "="*50)
    logger.info("TEST 3: RAGAS Availability")
    logger.info("="*50)
    test_results['tests']['ragas_availability'] = test_ragas_availability()
    
    # Test 4: Database Statistics
    logger.info("\n" + "="*50)
    logger.info("TEST 4: Database Statistics")
    logger.info("="*50)
    test_results['tests']['database_stats'] = test_database_stats()
    
    # Test 5: Single Technique (only if database is ready)
    if test_results['tests']['database_connection']['success']:
        logger.info("\n" + "="*50)
        logger.info("TEST 5: Single Technique")
        logger.info("="*50)
        test_results['tests']['single_technique'] = test_single_technique()
    else:
        logger.warning("⚠️ Skipping single technique test - database not ready")
        test_results['tests']['single_technique'] = {'success': False, 'skipped': True}
    
    # Summary
    logger.info("\n" + "="*50)
    logger.info("TEST SUMMARY")
    logger.info("="*50)
    
    total_tests = len(test_results['tests'])
    successful_tests = sum(1 for test in test_results['tests'].values() if test.get('success', False))
    
    logger.info(f"📊 Tests completed: {successful_tests}/{total_tests}")
    
    for test_name, result in test_results['tests'].items():
        status = "✅ PASS" if result.get('success', False) else "❌ FAIL"
        if result.get('skipped'):
            status = "⏭️ SKIP"
        logger.info(f"   {test_name}: {status}")
    
    # Overall assessment
    if successful_tests == total_tests:
        logger.info("\n🎉 ALL TESTS PASSED - Framework is ready for use!")
        test_results['overall_status'] = 'ready'
    elif successful_tests >= total_tests - 1:
        logger.info("\n⚠️ MOSTLY READY - Minor issues detected")
        test_results['overall_status'] = 'mostly_ready'
    else:
        logger.info("\n❌ FRAMEWORK NOT READY - Multiple issues detected")
        test_results['overall_status'] = 'not_ready'
    
    # Save test results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"framework_test_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump(test_results, f, indent=2, default=str)
    
    logger.info(f"💾 Test results saved to {results_file}")
    
    # Recommendations
    logger.info("\n📋 RECOMMENDATIONS:")
    
    if not test_results['tests']['database_connection']['success']:
        logger.info("   1. Fix database connection issues")
        logger.info("   2. Ensure IRIS database is running and accessible")
        logger.info("   3. Verify database schema and data are loaded")
    
    if not test_results['tests']['ragas_availability']['success']:
        logger.info("   1. Install RAGAS: pip install ragas datasets")
        logger.info("   2. Set OpenAI API key for real evaluation")
    
    if test_results['tests']['single_technique'].get('success'):
        logger.info("   1. Framework is ready for evaluation")
        logger.info("   2. Run: python run_comprehensive_scaling_evaluation.py --mode current_size")
        logger.info("   3. For full evaluation: python run_comprehensive_scaling_evaluation.py --mode comprehensive")
    
    return test_results

def main():
    """Main execution function"""
    logger.info("🧪 Comprehensive Scaling Evaluation Framework Test")
    
    try:
        results = run_comprehensive_test()
        
        if results['overall_status'] == 'ready':
            return 0
        elif results['overall_status'] == 'mostly_ready':
            return 1
        else:
            return 2
            
    except Exception as e:
        logger.error(f"❌ Test execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 3

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)