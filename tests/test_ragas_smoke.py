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
from typing import List, Dict # Added import for type hinting
# sys.path.append('.') # Keep if script is in project root, otherwise adjust for project root
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '.')) # Assuming script is in project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

# Load environment variables
load_dotenv()

from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    faithfulness
)
from datasets import Dataset

from common.iris_connector import get_iris_connection # Updated import
from common.embedding_utils import get_embedding_model # Updated import

# Import all V2 pipelines
from iris_rag.pipelines.basic import BasicRAGPipeline as BasicRAGPipelineV2
from iris_rag.pipelines.noderag import NodeRAGPipeline as NodeRAGPipelineV2
from iris_rag.pipelines.hyde import HyDERAGPipeline as HyDEPipelineV2
from iris_rag.pipelines.crag import CRAGPipeline as CRAGPipelineV2
from iris_rag.pipelines.colbert import ColBERTRAGPipeline as ColBERTPipelineV2
from iris_rag.pipelines.hybrid_ifind import HybridIFindRAGPipeline as HybridiFindRAGPipelineV2
from iris_rag.pipelines.graphrag import GraphRAGPipeline as GraphRAGPipelineV2

# Test query (will be loaded from file)
# TEST_QUERY = "What is diabetes and how is it treated?"
# GROUND_TRUTH = "Diabetes is a chronic metabolic disorder characterized by high blood sugar levels. It is treated through a combination of lifestyle modifications (diet and exercise), blood glucose monitoring, and medications including insulin therapy for Type 1 diabetes and various oral medications or insulin for Type 2 diabetes."

def load_queries(file_path: str = "eval/sample_queries.json") -> List[Dict]:
    """Load queries from a JSON file."""
    try:
        with open(file_path, 'r') as f:
            queries_data = json.load(f)
        print(f"Loaded {len(queries_data)} queries from {file_path}")
        return queries_data
    except Exception as e:
        print(f"Error loading queries from {file_path}: {e}")
        return [{
            "query": "What is diabetes and how is it treated?", # Fallback query
            "ground_truth_answer": "Diabetes is a chronic metabolic disorder characterized by high blood sugar levels. It is treated through a combination of lifestyle modifications (diet and exercise), blood glucose monitoring, and medications including insulin therapy for Type 1 diabetes and various oral medications or insulin for Type 2 diabetes.",
            "ground_truth_contexts": ["Context related to diabetes treatment."] # Fallback context
        }]


def test_pipeline_for_query(pipeline_class, pipeline_name, iris, embedding_func, llm_func, query_text: str):
    """Test a single pipeline for a given query and return results"""
    print(f"\n--- Testing {pipeline_name} for query: '{query_text[:50]}...' ---")
    print(f"{'='*60}")
    
    try:
        start_time = time.time()
        
        # Initialize pipeline
        pipeline = pipeline_class(iris, embedding_func, llm_func)
        
        # Run pipeline
        result = pipeline.run(query_text, top_k=5)
        
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
        return None, None # Return a tuple
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
    
    # The 'results' parameter is now a list of dictionaries, one for each query run
    # Each dictionary looks like:
    # { 'query': query_text, 'answer': ..., 'contexts': ..., 'ground_truth': ...,
    #   'pipeline_name': ..., 'execution_time': ..., 'num_docs': ... }

    for item in results: # Iterate through the list of results
        if item.get('success', False) and item.get('contexts'):
            eval_data['question'].append(item['query'])
            eval_data['answer'].append(item['answer'])
            eval_data['contexts'].append(item['contexts'])
            eval_data['ground_truth'].append(item['ground_truth'])
            # techniques_to_eval is not needed here as we process a flat list for the dataset
    
    if not eval_data['question']:
        print("⚠️  No successful results to evaluate with RAGAS (eval_data['question'] is empty).")
        return None, None # Return a tuple
    
    try:
        # Create dataset
        dataset = Dataset.from_dict(eval_data)
        
        # Run RAGAS evaluation
        print(f"\n📊 Evaluating {len(eval_data['question'])} query results with RAGAS...")
        metrics_to_run = [context_precision, answer_relevancy, faithfulness]
        
        ragas_results = evaluate(
            dataset,
            metrics=metrics_to_run
        )
        
        # Display results
        print("\n📈 RAGAS Scores:")
        print("-" * 50)
        
        df = ragas_results.to_pandas()
        
        # Store individual query scores along with the query
        query_scores = []
        for i in range(len(df)):
            query_text = eval_data['question'][i] # Get the original query text
            scores = {"query": query_text}
            for metric_obj in metrics_to_run:
                metric_name = metric_obj.name
                if metric_name in df.columns:
                    scores[metric_name] = df[metric_name].iloc[i]
            query_scores.append(scores)
            
            print(f"\nQuery: {query_text[:70]}...")
            for metric_name, score_value in scores.items():
                if metric_name != "query":
                    print(f"  {metric_name}: {score_value:.3f}")

        # Calculate and print average scores
        print("\n" + "-" * 50)
        print("📊 Average RAGAS Scores:")
        for metric_obj in metrics_to_run:
            metric_name = metric_obj.name
            if metric_name in df.columns:
                average_score = df[metric_name].mean()
                print(f"  Average {metric_name}: {average_score:.3f}")
        
        return ragas_results, query_scores # Return both the full dataset and per-query scores
        
    except Exception as e:
        print(f"❌ RAGAS evaluation failed: {e}")
        import traceback
        traceback.print_exc()
        return None, None # Return a tuple

def main():
    """Run smoke test on all RAG techniques"""
    
    print("🚀 RAGAS Test for BasicRAG with Multiple Queries")
    print("=" * 60)
    
    queries_data = load_queries()
    if not queries_data:
        return

    # Initialize components
    iris = get_iris_connection()
    embedding_model = get_embedding_model('sentence-transformers/all-MiniLM-L6-v2')
    
    def embedding_func(texts):
        return embedding_model.encode(texts)
    
    # Use real OpenAI LLM for RAGAS evaluation
    from common.utils import get_llm_func # Updated import
    llm_func = get_llm_func(provider="openai", model_name="gpt-3.5-turbo")
    
    # Test only BasicRAG pipeline
    pipelines = [
        (BasicRAGPipelineV2, "BasicRAG"),
    ]
    
    all_pipeline_runs = [] # To store results for RAGAS evaluation
    
    pipeline_class, pipeline_name = pipelines[0] # We only have BasicRAG

    total_execution_time = 0
    successful_queries = 0

    for i, query_item in enumerate(queries_data):
        query_text = query_item["query"]
        ground_truth_answer = query_item["ground_truth_answer"]
        # ground_truth_contexts = query_item["ground_truth_contexts"] # For context_recall if used

        print(f"\nRunning Query {i+1}/{len(queries_data)}: {query_text[:70]}...")
        
        result = test_pipeline_for_query( # Changed function name
            pipeline_class,
            pipeline_name,
            iris,
            embedding_func,
            llm_func,
            query_text # Pass current query
        )
        
        if result['success']:
            successful_queries +=1
            total_execution_time += result['execution_time']
            all_pipeline_runs.append({
                'query': query_text,
                'answer': result['answer'],
                'contexts': result['contexts'],
                'ground_truth': ground_truth_answer,
                'pipeline_name': pipeline_name,
                'execution_time': result['execution_time'],
                'num_docs': result['num_docs'],
                'success': True  # Explicitly add success status
            })
        else:
            all_pipeline_runs.append({
                'query': query_text,
                'answer': '',
                'contexts': [],
                'ground_truth': ground_truth_answer,
                'pipeline_name': pipeline_name,
                'error': result.get('error', 'Unknown error'),
                'execution_time': 0,
                'num_docs': 0,
                'success': False # Explicitly add success status
            })

    # Summary
    print("\n" + "=" * 60)
    print(f"📊 SUMMARY for {pipeline_name}")
    print("=" * 60)
    
    print(f"\n✅ Successful Queries: {successful_queries}/{len(queries_data)} ({successful_queries/len(queries_data)*100:.0f}%)")
    if successful_queries > 0:
        print(f"⏱️  Average Execution Time per Query: {total_execution_time/successful_queries:.2f}s")
    
    # RAGAS evaluation
    ragas_eval_output = evaluate_with_ragas(all_pipeline_runs)
    
    ragas_full_results = None
    per_query_ragas_scores = None
    if ragas_eval_output: # Check if RAGAS evaluation ran successfully
        ragas_full_results, per_query_ragas_scores = ragas_eval_output
    else:
        print("RAGAS evaluation was skipped or failed. No RAGAS scores to save.")

    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    results_file = f"ragas_basic_rag_multi_query_test_{timestamp}.json"
    
    # Combine pipeline run info with RAGAS scores for saving
    final_results_to_save = []
    for i, run_info in enumerate(all_pipeline_runs):
        item_to_save = run_info.copy()
        if per_query_ragas_scores and i < len(per_query_ragas_scores):
            item_to_save['ragas_scores'] = per_query_ragas_scores[i]
        final_results_to_save.append(item_to_save)

    data_to_save = {
        'timestamp': timestamp,
        'pipeline_tested': pipeline_name,
        'num_queries_tested': len(queries_data),
        'successful_queries': successful_queries,
        'results_per_query': final_results_to_save,
        'average_ragas_scores': {}
    }

    if ragas_full_results and hasattr(ragas_full_results, 'to_pandas'):
        df_scores = ragas_full_results.to_pandas()
        # Explicitly list metric names expected from Ragas evaluation
        metric_column_names = ['context_precision', 'answer_relevancy', 'faithfulness']
        
        # Filter to only include metric columns that actually exist in the DataFrame
        valid_metric_cols = [name for name in metric_column_names if name in df_scores.columns]
        
        if valid_metric_cols:
            data_to_save['average_ragas_scores'] = df_scores[valid_metric_cols].mean().to_dict()
        else:
            data_to_save['average_ragas_scores'] = {"error": "No expected metric columns found in RAGAS results for averaging."}
    elif ragas_eval_output is None : # If RAGAS was skipped or failed initially
         data_to_save['average_ragas_scores'] = {"status": "RAGAS evaluation skipped or failed."}

    with open(results_file, 'w') as f:
        json.dump(data_to_save, f, indent=2)
    
    print(f"\n💾 Detailed results saved to: {results_file}")
    
    iris.close()
    
    if successful_queries == len(queries_data):
        print(f"\n🎉 All queries passed for {pipeline_name}!")
    else:
        print(f"\n⚠️  {len(queries_data) - successful_queries} queries had issues for {pipeline_name}")

if __name__ == "__main__":
    main()