#!/usr/bin/env python3
"""
RAGAS Smoke Test for All 7 RAG Techniques
Tests a single query across all techniques with RAGAS evaluation
"""

import sys
import time
import json
import os
from datetime import datetime
from dotenv import load_dotenv
sys.path.append('.')

# Load environment variables
load_dotenv()

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    faithfulness
)
from datasets import Dataset

from common.iris_connector import get_iris_connection
from common.embedding_utils import get_embedding_model

# Import all V2 pipelines
from basic_rag.pipeline_v2 import BasicRAGPipelineV2
from noderag.pipeline_v2 import NodeRAGPipelineV2
from hyde.pipeline_v2 import HyDEPipelineV2
from crag.pipeline_v2 import CRAGPipelineV2
from colbert.pipeline_v2 import ColBERTPipelineV2
from hybrid_ifind_rag.pipeline_v2 import HybridiFindRAGPipelineV2
from graphrag.pipeline_v2 import GraphRAGPipelineV2

# Test query
TEST_QUERY = "What is diabetes and how is it treated?"
GROUND_TRUTH = "Diabetes is a chronic metabolic disorder characterized by high blood sugar levels. It is treated through a combination of lifestyle modifications (diet and exercise), blood glucose monitoring, and medications including insulin therapy for Type 1 diabetes and various oral medications or insulin for Type 2 diabetes."

def test_pipeline(pipeline_class, pipeline_name, iris, embedding_func, llm_func):
    """Test a single pipeline and return results"""
    print(f"\n{'='*60}")
    print(f"Testing {pipeline_name}")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        
        # Initialize pipeline
        pipeline = pipeline_class(iris, embedding_func, llm_func)
        
        # Run pipeline
        result = pipeline.run(TEST_QUERY, top_k=5)
        
        end_time = time.time()
        execution_time = end_time - start_time
        
        # Extract results
        answer = result.get('answer', 'No answer generated')
        retrieved_docs = result.get('retrieved_documents', [])
        
        # Format contexts for RAGAS
        contexts = []
        for doc in retrieved_docs[:5]:  # Use top 5 documents
            if hasattr(doc, 'content'):
                contexts.append(doc.content[:500])  # Limit context length
            elif isinstance(doc, dict) and 'content' in doc:
                contexts.append(doc['content'][:500])
        
        print(f"✅ {pipeline_name} executed successfully!")
        print(f"⏱️  Execution time: {execution_time:.2f} seconds")
        print(f"📄 Documents retrieved: {len(retrieved_docs)}")
        print(f"💬 Answer preview: {answer[:200]}...")
        
        # Pipeline-specific info
        if pipeline_name == "GraphRAG":
            entities = result.get('entities', [])
            relationships = result.get('relationships', [])
            print(f"🔗 Entities found: {len(entities)}")
            print(f"🔗 Relationships found: {len(relationships)}")
        
        return {
            'success': True,
            'answer': answer,
            'contexts': contexts,
            'execution_time': execution_time,
            'num_docs': len(retrieved_docs)
        }
        
    except Exception as e:
        print(f"❌ {pipeline_name} failed: {str(e)}")
        return {
            'success': False,
            'answer': '',
            'contexts': [],
            'execution_time': 0,
            'num_docs': 0,
            'error': str(e)
        }

def evaluate_with_ragas(results):
    """Evaluate all results with RAGAS"""
    print("\n" + "="*60)
    print("🔍 RAGAS Evaluation")
    print("="*60)
    
    # Check if we have OpenAI API key for real evaluation
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("⚠️  No OpenAI API key found. Skipping RAGAS evaluation.")
        print("   Set OPENAI_API_KEY environment variable for quality metrics.")
        return None
    else:
        print(f"✅ OpenAI API key found (length: {len(api_key)})")
    
    # Prepare data for RAGAS
    eval_data = {
        'question': [],
        'answer': [],
        'contexts': [],
        'ground_truth': []
    }
    
    techniques_to_eval = []
    
    for technique, result in results.items():
        if result['success'] and result['contexts']:
            eval_data['question'].append(TEST_QUERY)
            eval_data['answer'].append(result['answer'])
            eval_data['contexts'].append(result['contexts'])
            eval_data['ground_truth'].append(GROUND_TRUTH)
            techniques_to_eval.append(technique)
    
    if not eval_data['question']:
        print("⚠️  No successful results to evaluate with RAGAS")
        return None
    
    try:
        # Create dataset
        dataset = Dataset.from_dict(eval_data)
        
        # Run RAGAS evaluation
        print(f"\n📊 Evaluating {len(techniques_to_eval)} techniques with RAGAS...")
        ragas_results = evaluate(
            dataset,
            metrics=[answer_relevancy, context_precision, faithfulness]
        )
        
        # Display results
        print("\n📈 RAGAS Scores by Technique:")
        print("-" * 50)
        
        # RAGAS returns a dataset with scores
        if hasattr(ragas_results, 'to_pandas'):
            df = ragas_results.to_pandas()
            for i, technique in enumerate(techniques_to_eval):
                print(f"\n{technique}:")
                for metric in ['answer_relevancy', 'context_precision', 'faithfulness']:
                    if metric in df.columns:
                        score = df[metric].iloc[i]
                        print(f"  {metric}: {score:.3f}")
        else:
            # Handle dictionary format
            for i, technique in enumerate(techniques_to_eval):
                print(f"\n{technique}:")
                for metric in ['answer_relevancy', 'context_precision', 'faithfulness']:
                    if metric in ragas_results:
                        score = ragas_results[metric]
                        print(f"  {metric}: {score:.3f}")
        
        return ragas_results
        
    except Exception as e:
        print(f"❌ RAGAS evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None

def main():
    """Run smoke test on all RAG techniques"""
    
    print("🚀 RAGAS Smoke Test for All 7 RAG Techniques")
    print("=" * 60)
    print(f"📝 Test Query: {TEST_QUERY}")
    print("=" * 60)
    
    # Initialize components
    iris = get_iris_connection()
    embedding_model = get_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
    
    def embedding_func(texts):
        return embedding_model.encode(texts)
    
    # Use real OpenAI LLM for RAGAS evaluation
    from common.utils import get_llm_func
    llm_func = get_llm_func(provider="openai", model_name="gpt-3.5-turbo")
    
    # Test all pipelines
    pipelines = [
        (BasicRAGPipelineV2, "BasicRAG"),
        (NodeRAGPipelineV2, "NodeRAG"),
        (GraphRAGPipelineV2, "GraphRAG"),
        (ColBERTPipelineV2, "ColBERT"),
        (HyDEPipelineV2, "HyDE"),
        (CRAGPipelineV2, "CRAG"),
        (HybridiFindRAGPipelineV2, "HybridiFindRAG"),
    ]
    
    all_results = {}
    
    for pipeline_class, pipeline_name in pipelines:
        result = test_pipeline(
            pipeline_class, 
            pipeline_name, 
            iris, 
            embedding_func, 
            llm_func
        )
        all_results[pipeline_name] = result
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 SUMMARY")
    print("=" * 60)
    
    successful = sum(1 for r in all_results.values() if r['success'])
    print(f"\n✅ Success Rate: {successful}/{len(pipelines)} ({successful/len(pipelines)*100:.0f}%)")
    
    print("\n📈 Performance Summary:")
    for name, result in all_results.items():
        if result['success']:
            print(f"  {name:20} - {result['execution_time']:.2f}s - {result['num_docs']} docs")
        else:
            print(f"  {name:20} - FAILED: {result.get('error', 'Unknown error')}")
    
    # RAGAS evaluation
    ragas_results = evaluate_with_ragas(all_results)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"ragas_smoke_test_{timestamp}.json"
    
    # Prepare serializable results
    serializable_results = {}
    for technique, data in all_results.items():
        serializable_results[technique] = {
            'success': data['success'],
            'execution_time': data['execution_time'],
            'num_docs': data['num_docs'],
            'answer_preview': data['answer'][:200] if data['answer'] else '',
            'error': data.get('error', '')
        }
    
    with open(results_file, 'w') as f:
        json.dump({
            'timestamp': timestamp,
            'query': TEST_QUERY,
            'results': serializable_results,
            'ragas_available': ragas_results is not None
        }, f, indent=2)
    
    print(f"\n💾 Results saved to: {results_file}")
    
    iris.close()
    
    if successful == len(pipelines):
        print("\n🎉 All RAG techniques passed the smoke test!")
    else:
        print(f"\n⚠️  {len(pipelines) - successful} techniques failed the smoke test")

if __name__ == "__main__":
    main()