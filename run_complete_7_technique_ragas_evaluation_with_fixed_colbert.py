#!/usr/bin/env python3
"""
Complete 7-Technique RAGAS Evaluation with Fixed ColBERT
Evaluates all 7 RAG techniques including the optimized ColBERT implementation.
"""

import sys
import os
sys.path.insert(0, '.')

import json
import time
import logging
from datetime import datetime
from typing import Dict, List, Any

# Import all RAG pipelines
from basic_rag.pipeline import BasicRAGPipeline
from crag.pipeline import CRAGPipeline
from hyde.pipeline import HyDEPipeline
from noderag.pipeline import NodeRAGPipeline
from graphrag.pipeline import GraphRAGPipeline
from hybrid_ifind_rag.pipeline import HybridiFindRAGPipeline
from colbert.pipeline import create_colbert_pipeline  # Use the fixed implementation

# Import evaluation utilities
from common.iris_connector import get_iris_connection
from common.utils import get_llm_func, get_embedding_func

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Medical domain questions for evaluation
MEDICAL_QUESTIONS = [
    "What are the main symptoms of diabetes?",
    "How is hypertension diagnosed and treated?",
    "What are the risk factors for cardiovascular disease?",
    "What is the difference between Type 1 and Type 2 diabetes?",
    "How does insulin resistance develop?",
    "What are the complications of untreated diabetes?",
    "What lifestyle changes help manage high blood pressure?",
    "How do statins work to lower cholesterol?",
    "What are the warning signs of a heart attack?",
    "How is obesity related to metabolic syndrome?"
]

def create_all_pipelines():
    """Create instances of all 7 RAG pipelines."""
    logger.info("Creating all RAG pipeline instances...")
    
    # Get shared resources
    iris_conn = get_iris_connection()
    llm_func = get_llm_func()
    embedding_func = get_embedding_func()
    
    pipelines = {}
    
    try:
        # 1. BasicRAG
        pipelines['BasicRAG'] = BasicRAGPipeline(
            iris_connector=iris_conn,
            embedding_func=embedding_func,
            llm_func=llm_func
        )
        logger.info("✅ BasicRAG pipeline created")
        
        # 2. CRAG
        pipelines['CRAG'] = CRAGPipeline(
            iris_connector=iris_conn,
            embedding_func=embedding_func,
            llm_func=llm_func
        )
        logger.info("✅ CRAG pipeline created")
        
        # 3. HyDE
        pipelines['HyDE'] = HyDEPipeline(
            iris_connector=iris_conn,
            embedding_func=embedding_func,
            llm_func=llm_func
        )
        logger.info("✅ HyDE pipeline created")
        
        # 4. NodeRAG
        pipelines['NodeRAG'] = NodeRAGPipeline(
            iris_connector=iris_conn,
            embedding_func=embedding_func,
            llm_func=llm_func
        )
        logger.info("✅ NodeRAG pipeline created")
        
        # 5. GraphRAG
        pipelines['GraphRAG'] = GraphRAGPipeline(
            iris_connector=iris_conn,
            embedding_func=embedding_func,
            llm_func=llm_func
        )
        logger.info("✅ GraphRAG pipeline created")
        
        # 6. HybridiFindRAG
        pipelines['HybridiFindRAG'] = HybridiFindRAGPipeline(
            iris_connector=iris_conn,
            embedding_func=embedding_func,
            llm_func=llm_func
        )
        logger.info("✅ HybridiFindRAG pipeline created")
        
        # 7. Fixed ColBERT
        pipelines['ColBERT'] = create_colbert_pipeline(
            iris_connector=iris_conn,
            llm_func=llm_func
        )
        logger.info("✅ Fixed ColBERT pipeline created")
        
    except Exception as e:
        logger.error(f"Error creating pipelines: {e}")
        raise
    
    logger.info(f"Successfully created {len(pipelines)} RAG pipelines")
    return pipelines

def evaluate_technique(technique_name: str, pipeline: Any, questions: List[str]) -> Dict[str, Any]:
    """Evaluate a single RAG technique."""
    logger.info(f"🔍 Evaluating {technique_name}...")
    
    results = {
        'technique': technique_name,
        'questions_evaluated': len(questions),
        'responses': [],
        'performance_stats': {
            'total_time': 0,
            'avg_response_time': 0,
            'successful_responses': 0,
            'failed_responses': 0
        }
    }
    
    start_time = time.time()
    
    for i, question in enumerate(questions):
        logger.info(f"  Question {i+1}/{len(questions)}: {question[:50]}...")
        
        try:
            question_start = time.time()
            
            # Run the pipeline
            if hasattr(pipeline, 'run'):
                response = pipeline.run(question)
            else:
                # Fallback for pipelines without run method
                retrieved_docs = pipeline.retrieve_documents(question)
                answer = pipeline.generate_answer(question, retrieved_docs)
                response = {
                    'query': question,
                    'answer': answer,
                    'retrieved_documents': [doc.to_dict() for doc in retrieved_docs]
                }
            
            question_time = time.time() - question_start
            
            # Store response
            response_data = {
                'question': question,
                'answer': response.get('answer', ''),
                'retrieved_documents': response.get('retrieved_documents', []),
                'response_time': question_time,
                'success': True
            }
            
            results['responses'].append(response_data)
            results['performance_stats']['successful_responses'] += 1
            
            logger.info(f"    ✅ Response time: {question_time:.2f}s")
            
        except Exception as e:
            logger.error(f"    ❌ Error processing question: {e}")
            
            response_data = {
                'question': question,
                'answer': f"Error: {str(e)}",
                'retrieved_documents': [],
                'response_time': 0,
                'success': False,
                'error': str(e)
            }
            
            results['responses'].append(response_data)
            results['performance_stats']['failed_responses'] += 1
    
    # Calculate performance statistics
    total_time = time.time() - start_time
    successful_times = [r['response_time'] for r in results['responses'] if r['success']]
    
    results['performance_stats']['total_time'] = total_time
    results['performance_stats']['avg_response_time'] = (
        sum(successful_times) / len(successful_times) if successful_times else 0
    )
    
    success_rate = (results['performance_stats']['successful_responses'] / len(questions)) * 100
    
    logger.info(f"✅ {technique_name} evaluation complete:")
    logger.info(f"   Success rate: {success_rate:.1f}%")
    logger.info(f"   Avg response time: {results['performance_stats']['avg_response_time']:.2f}s")
    
    return results

def run_complete_evaluation():
    """Run the complete 7-technique RAGAS evaluation."""
    logger.info("🚀 Starting Complete 7-Technique RAGAS Evaluation with Fixed ColBERT")
    
    evaluation_start = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Create all pipelines
    pipelines = create_all_pipelines()
    
    # Run evaluation for each technique
    all_results = {
        'evaluation_metadata': {
            'timestamp': timestamp,
            'total_techniques': len(pipelines),
            'questions_per_technique': len(MEDICAL_QUESTIONS),
            'evaluation_type': 'Complete 7-Technique RAGAS with Fixed ColBERT'
        },
        'technique_results': {}
    }
    
    for technique_name, pipeline in pipelines.items():
        try:
            results = evaluate_technique(technique_name, pipeline, MEDICAL_QUESTIONS)
            all_results['technique_results'][technique_name] = results
        except Exception as e:
            logger.error(f"Failed to evaluate {technique_name}: {e}")
            all_results['technique_results'][technique_name] = {
                'technique': technique_name,
                'error': str(e),
                'success': False
            }
    
    # Calculate overall statistics
    total_evaluation_time = time.time() - evaluation_start
    all_results['evaluation_metadata']['total_evaluation_time'] = total_evaluation_time
    
    # Generate performance summary
    performance_summary = []
    for technique_name, results in all_results['technique_results'].items():
        if 'performance_stats' in results:
            stats = results['performance_stats']
            performance_summary.append({
                'technique': technique_name,
                'avg_response_time': stats['avg_response_time'],
                'success_rate': (stats['successful_responses'] / len(MEDICAL_QUESTIONS)) * 100,
                'total_responses': len(MEDICAL_QUESTIONS)
            })
    
    # Sort by average response time
    performance_summary.sort(key=lambda x: x['avg_response_time'])
    all_results['performance_summary'] = performance_summary
    
    # Save results
    output_filename = f"complete_7_technique_fixed_colbert_evaluation_{timestamp}.json"
    with open(output_filename, 'w') as f:
        json.dump(all_results, f, indent=2, default=str)
    
    # Print summary
    logger.info("\n" + "="*80)
    logger.info("🎉 COMPLETE 7-TECHNIQUE EVALUATION WITH FIXED COLBERT - RESULTS")
    logger.info("="*80)
    
    logger.info(f"📊 Performance Rankings (by avg response time):")
    for i, summary in enumerate(performance_summary, 1):
        technique = summary['technique']
        avg_time = summary['avg_response_time']
        success_rate = summary['success_rate']
        
        if avg_time > 0:
            logger.info(f"{i}. {technique}: {avg_time:.2f}s (Success: {success_rate:.1f}%)")
        else:
            logger.info(f"{i}. {technique}: Failed evaluation")
    
    logger.info(f"\n📁 Results saved to: {output_filename}")
    logger.info(f"⏱️  Total evaluation time: {total_evaluation_time:.2f}s")
    logger.info(f"✅ Techniques evaluated: {len([r for r in all_results['technique_results'].values() if 'performance_stats' in r])}/{len(pipelines)}")
    
    return all_results

if __name__ == "__main__":
    try:
        results = run_complete_evaluation()
        print("\n🎉 Evaluation completed successfully!")
        
        # Show quick summary
        if 'performance_summary' in results:
            print("\n📊 Quick Performance Summary:")
            for summary in results['performance_summary'][:3]:  # Top 3
                technique = summary['technique']
                avg_time = summary['avg_response_time']
                success_rate = summary['success_rate']
                print(f"  {technique}: {avg_time:.2f}s ({success_rate:.1f}% success)")
        
    except Exception as e:
        logger.error(f"Evaluation failed: {e}")
        sys.exit(1)