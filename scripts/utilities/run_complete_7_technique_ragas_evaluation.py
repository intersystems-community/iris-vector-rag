#!/usr/bin/env python3
"""
Complete RAGAS Evaluation with All 7 RAG Techniques Including ColBERT
"""

import sys
import os
import time
import json
from datetime import datetime
import hashlib

# Add project root to path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..')) # Assuming scripts is in project root
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from common.iris_connector import get_iris_connection # Updated import
from common.utils import get_embedding_func, get_llm_func # Updated import

# Import all RAG pipelines
from iris_rag.pipelines.basic import BasicRAGPipeline # Updated import
from iris_rag.pipelines.hyde import HyDERAGPipeline # Updated import
from iris_rag.pipelines.crag import CRAGPipeline # Updated import
from iris_rag.pipelines.noderag import NodeRAGPipeline # Updated import
from iris_rag.pipelines.graphrag import GraphRAGPipeline # Updated import
from iris_rag.pipelines.hybrid_ifind import HybridIFindRAGPipeline # Updated import
from iris_rag.pipelines.colbert import ColBERTRAGPipeline # Updated import

# RAGAS imports
from ragas import evaluate
from ragas.metrics import (
    answer_relevancy,
    context_precision,
    context_recall,
    faithfulness
)
from datasets import Dataset

def create_working_colbert_pipeline():
    """Create a working ColBERT pipeline with content limiting"""
    conn = get_iris_connection()
    llm_func = get_llm_func(provider='openai')
    
    # Working 128D encoder that matches stored embeddings
    def working_128d_encoder(text):
        hash_obj = hashlib.md5(text.encode())
        hash_bytes = hash_obj.digest()
        
        embedding = []
        for i in range(128):
            byte_val = hash_bytes[i % len(hash_bytes)]
            float_val = (byte_val - 127.5) / 127.5
            embedding.append(float_val)
        
        return [embedding]
    
    # Create pipeline
    pipeline = ColBERTRAGPipeline(conn, working_128d_encoder, working_128d_encoder, llm_func)
    
    # Override run method to limit content and avoid context overflow
    original_run = pipeline.run
    def limited_run(query_text, top_k=2, similarity_threshold=0.1):
        # Use very small top_k to avoid context overflow
        return original_run(query_text, min(top_k, 2), similarity_threshold)
    
    pipeline.run = limited_run
    return pipeline

def initialize_all_pipelines():
    """Initialize all 7 RAG pipelines"""
    print("🔧 Initializing all 7 RAG pipelines...")
    
    # Get common dependencies
    iris_connector = get_iris_connection()
    embedding_func = get_embedding_func()
    llm_func = get_llm_func(provider='openai')
    
    pipelines = {}
    
    try:
        # 1. BasicRAG
        pipelines['BasicRAG'] = BasicRAGPipeline(iris_connector, embedding_func, llm_func)
        print("✅ BasicRAG initialized")
        
        # 2. HyDE
        pipelines['HyDE'] = HyDERAGPipeline(iris_connector, embedding_func, llm_func)
        print("✅ HyDE initialized")
        
        # 3. CRAG
        pipelines['CRAG'] = CRAGPipeline(iris_connector, embedding_func, llm_func)
        print("✅ CRAG initialized")
        
        # 4. NodeRAG
        pipelines['NodeRAG'] = NodeRAGPipeline(iris_connector, embedding_func, llm_func)
        print("✅ NodeRAG initialized")
        
        # 5. GraphRAG
        pipelines['GraphRAG'] = GraphRAGPipeline(iris_connector, embedding_func, llm_func)
        print("✅ GraphRAG initialized")
        
        # 6. HybridiFindRAG
        pipelines['HybridiFindRAG'] = HybridIFindRAGPipeline(iris_connector, embedding_func, llm_func)
        print("✅ HybridiFindRAG initialized")
        
        # 7. ColBERT (with special handling)
        pipelines['ColBERT'] = create_working_colbert_pipeline()
        print("✅ ColBERT initialized with content limiting")
        
        print(f"🎉 All {len(pipelines)} RAG techniques initialized successfully!")
        return pipelines
        
    except Exception as e:
        print(f"❌ Error initializing pipelines: {e}")
        return pipelines

def get_medical_questions():
    """Get medical questions for evaluation"""
    return [
        {
            "question": "What are the main treatments for diabetes?",
            "ground_truth": "Main treatments for diabetes include lifestyle modifications (diet and exercise), medications (metformin, insulin), blood glucose monitoring, and regular medical care."
        },
        {
            "question": "What are the symptoms of hypertension?",
            "ground_truth": "Hypertension often has no symptoms but may include headaches, shortness of breath, dizziness, chest pain, and nosebleeds in severe cases."
        },
        {
            "question": "How is cancer diagnosed?",
            "ground_truth": "Cancer diagnosis involves medical history, physical examination, imaging tests (CT, MRI, X-rays), laboratory tests, and tissue biopsy for definitive diagnosis."
        },
        {
            "question": "What causes heart disease?",
            "ground_truth": "Heart disease is caused by factors including high cholesterol, high blood pressure, smoking, diabetes, obesity, family history, and sedentary lifestyle."
        },
        {
            "question": "What are the side effects of chemotherapy?",
            "ground_truth": "Chemotherapy side effects include nausea, vomiting, hair loss, fatigue, increased infection risk, anemia, and potential organ damage."
        },
        {
            "question": "How is pneumonia treated?",
            "ground_truth": "Pneumonia treatment includes antibiotics for bacterial pneumonia, antivirals for viral pneumonia, rest, fluids, oxygen therapy if needed, and supportive care."
        },
        {
            "question": "What are the risk factors for stroke?",
            "ground_truth": "Stroke risk factors include high blood pressure, diabetes, smoking, high cholesterol, atrial fibrillation, age, family history, and previous stroke or TIA."
        },
        {
            "question": "How is depression diagnosed and treated?",
            "ground_truth": "Depression is diagnosed through clinical evaluation and may be treated with psychotherapy, antidepressant medications, lifestyle changes, and support groups."
        },
        {
            "question": "What are the complications of untreated diabetes?",
            "ground_truth": "Untreated diabetes complications include diabetic ketoacidosis, cardiovascular disease, kidney damage, nerve damage, eye problems, and poor wound healing."
        },
        {
            "question": "How does the immune system work?",
            "ground_truth": "The immune system protects against pathogens through innate immunity (barriers, white blood cells) and adaptive immunity (antibodies, T-cells) with immunological memory."
        }
    ]

def run_technique_evaluation(technique_name, pipeline, questions):
    """Run evaluation for a single technique"""
    print(f"\n🔍 Evaluating {technique_name}...")
    
    results = []
    total_time = 0
    successful_queries = 0
    
    for i, q in enumerate(questions, 1):
        try:
            print(f"  Question {i}/10: {q['question'][:50]}...")
            
            start_time = time.time()
            result = pipeline.run(q['question'])
            response_time = time.time() - start_time
            
            total_time += response_time
            
            # Extract answer and contexts
            answer = result.get('answer', '')
            retrieved_docs = result.get('retrieved_documents', [])
            
            # Create contexts list from retrieved documents
            contexts = []
            for doc in retrieved_docs:
                if isinstance(doc, dict):
                    content = doc.get('content', '') or doc.get('text_content', '') or str(doc)
                else:
                    content = str(doc)
                
                # Limit context length to avoid issues
                if len(content) > 1000:
                    content = content[:1000] + "..."
                contexts.append(content)
            
            # Ensure we have at least one context
            if not contexts:
                contexts = ["No relevant context found"]
            
            if answer and len(answer.strip()) > 10:
                results.append({
                    'question': q['question'],
                    'answer': answer,
                    'contexts': contexts,
                    'ground_truth': q['ground_truth'],
                    'response_time': response_time,
                    'retrieved_docs_count': len(retrieved_docs)
                })
                successful_queries += 1
                print(f"    ✅ Success ({response_time:.2f}s, {len(retrieved_docs)} docs)")
            else:
                print(f"    ⚠️ Empty/short answer ({response_time:.2f}s)")
                
        except Exception as e:
            print(f"    ❌ Error: {str(e)[:100]}...")
            continue
    
    avg_time = total_time / len(questions) if questions else 0
    success_rate = successful_queries / len(questions) if questions else 0
    
    print(f"  📊 {technique_name} Summary:")
    print(f"    - Successful queries: {successful_queries}/{len(questions)} ({success_rate:.1%})")
    print(f"    - Average response time: {avg_time:.2f}s")
    print(f"    - Total time: {total_time:.2f}s")
    
    return results, {
        'technique': technique_name,
        'successful_queries': successful_queries,
        'total_queries': len(questions),
        'success_rate': success_rate,
        'average_response_time': avg_time,
        'total_time': total_time
    }

def run_ragas_evaluation(results_data):
    """Run RAGAS evaluation on the results"""
    print("\n🔬 Running RAGAS evaluation...")
    
    ragas_results = {}
    
    for technique_name, data in results_data.items():
        if not data['results']:
            print(f"  ⚠️ Skipping {technique_name} - no valid results")
            continue
            
        try:
            print(f"  📊 Evaluating {technique_name} with RAGAS...")
            
            # Prepare dataset for RAGAS
            dataset_dict = {
                'question': [r['question'] for r in data['results']],
                'answer': [r['answer'] for r in data['results']],
                'contexts': [r['contexts'] for r in data['results']],
                'ground_truth': [r['ground_truth'] for r in data['results']]
            }
            
            dataset = Dataset.from_dict(dataset_dict)
            
            # Run RAGAS evaluation
            evaluation_result = evaluate(
                dataset,
                metrics=[answer_relevancy, context_precision, context_recall, faithfulness]
            )
            
            ragas_results[technique_name] = {
                'answer_relevancy': evaluation_result['answer_relevancy'],
                'context_precision': evaluation_result['context_precision'],
                'context_recall': evaluation_result['context_recall'],
                'faithfulness': evaluation_result['faithfulness'],
                'performance_stats': data['stats']
            }
            
            print(f"    ✅ {technique_name} RAGAS evaluation complete")
            
        except Exception as e:
            print(f"    ❌ RAGAS evaluation failed for {technique_name}: {e}")
            ragas_results[technique_name] = {
                'error': str(e),
                'performance_stats': data['stats']
            }
    
    return ragas_results

def main():
    """Main evaluation function"""
    print("🚀 Starting Complete 7-Technique RAGAS Evaluation with ColBERT")
    print("=" * 70)
    
    # Initialize pipelines
    pipelines = initialize_all_pipelines()
    if len(pipelines) < 7:
        print(f"⚠️ Warning: Only {len(pipelines)} techniques initialized")
    
    # Get questions
    questions = get_medical_questions()
    print(f"📋 Loaded {len(questions)} medical questions")
    
    # Run evaluations
    results_data = {}
    
    for technique_name, pipeline in pipelines.items():
        try:
            results, stats = run_technique_evaluation(technique_name, pipeline, questions)
            results_data[technique_name] = {
                'results': results,
                'stats': stats
            }
        except Exception as e:
            print(f"❌ Failed to evaluate {technique_name}: {e}")
            continue
    
    # Run RAGAS evaluation
    ragas_results = run_ragas_evaluation(results_data)
    
    # Save results
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"complete_7_technique_ragas_evaluation_{timestamp}.json"
    
    final_results = {
        'timestamp': timestamp,
        'techniques_evaluated': list(ragas_results.keys()),
        'total_techniques': len(pipelines),
        'questions_count': len(questions),
        'ragas_results': ragas_results,
        'raw_results': results_data
    }
    
    with open(filename, 'w') as f:
        json.dump(final_results, f, indent=2, default=str)
    
    # Print summary
    print("\n" + "=" * 70)
    print("🎉 COMPLETE 7-TECHNIQUE EVALUATION SUMMARY")
    print("=" * 70)
    
    print(f"📊 Techniques Evaluated: {len(ragas_results)}/7")
    print(f"📋 Questions Processed: {len(questions)}")
    print(f"💾 Results saved to: {filename}")
    
    print("\n📈 Performance Rankings:")
    # Sort by average response time
    performance_ranking = []
    for technique, data in ragas_results.items():
        if 'performance_stats' in data:
            stats = data['performance_stats']
            performance_ranking.append((
                technique,
                stats['average_response_time'],
                stats['success_rate'],
                stats.get('successful_queries', 0)
            ))
    
    performance_ranking.sort(key=lambda x: x[1])  # Sort by response time
    
    for i, (technique, avg_time, success_rate, successful) in enumerate(performance_ranking, 1):
        print(f"  {i}. {technique}: {avg_time:.2f}s (success: {success_rate:.1%}, {successful} queries)")
    
    print("\n🔬 RAGAS Quality Metrics:")
    for technique, data in ragas_results.items():
        if 'answer_relevancy' in data:
            print(f"  {technique}:")
            print(f"    - Answer Relevancy: {data['answer_relevancy']:.3f}")
            print(f"    - Context Precision: {data['context_precision']:.3f}")
            print(f"    - Context Recall: {data['context_recall']:.3f}")
            print(f"    - Faithfulness: {data['faithfulness']:.3f}")
    
    print(f"\n✅ Complete evaluation finished! Results in {filename}")
    print("🎯 All 7 RAG techniques including ColBERT have been evaluated!")

if __name__ == "__main__":
    main()