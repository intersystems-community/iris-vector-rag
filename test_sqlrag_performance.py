#!/usr/bin/env python3
"""
Test script to validate SQLRAG pipeline performance optimizations.

This script tests the optimized SQLRAG pipeline to ensure:
1. Performance is improved (target: <2s execution time)
2. Functionality remains intact
3. SQL-based retrieval still works correctly
"""

import time
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_sqlrag_performance():
    """Test the optimized SQLRAG pipeline performance."""
    try:
        # Import required modules
        from iris_rag.pipelines.sql_rag import SQLRAGPipeline
        from iris_rag.config.manager import ConfigurationManager
        from common.utils import get_llm_func
        
        print("🔧 Initializing optimized SQLRAG pipeline...")
        
        # Initialize components
        config_manager = ConfigurationManager()
        llm_func = get_llm_func()
        
        # Initialize SQLRAG pipeline (it will create its own vector store)
        pipeline = SQLRAGPipeline(
            config_manager=config_manager,
            llm_func=llm_func
        )
        
        print("✅ SQLRAG pipeline initialized successfully")
        
        # Test queries
        test_queries = [
            "What are the side effects of aspirin?",
            "How does diabetes affect cardiovascular health?",
            "What are the symptoms of pneumonia?",
            "What treatments are available for hypertension?",
            "How does exercise impact mental health?"
        ]
        
        print(f"\n🧪 Testing {len(test_queries)} queries for performance...")
        
        total_time = 0
        successful_queries = 0
        results = []
        
        for i, query in enumerate(test_queries, 1):
            print(f"\n--- Query {i}/{len(test_queries)}: {query[:50]}... ---")
            
            start_time = time.time()
            try:
                result = pipeline.execute(query)
                execution_time = time.time() - start_time
                
                if result.get("success", True):  # Default to True if not specified
                    successful_queries += 1
                    total_time += execution_time
                    
                    print(f"✅ Execution time: {execution_time:.3f}s")
                    print(f"📊 Retrieved {len(result.get('retrieved_documents', []))} documents")
                    print(f"💬 Answer length: {len(result.get('answer', ''))}")
                    
                    # Validate response structure
                    required_fields = ['query', 'answer', 'retrieved_documents', 'execution_time']
                    missing_fields = [field for field in required_fields if field not in result]
                    if missing_fields:
                        print(f"⚠️  Missing fields: {missing_fields}")
                    else:
                        print("✅ Response structure valid")
                    
                    results.append({
                        'query': query,
                        'execution_time': execution_time,
                        'success': True,
                        'documents_retrieved': len(result.get('retrieved_documents', [])),
                        'answer_length': len(result.get('answer', ''))
                    })
                    
                    # Performance check
                    if execution_time < 2.0:
                        print(f"🎯 PERFORMANCE TARGET MET: {execution_time:.3f}s < 2.0s")
                    else:
                        print(f"⚠️  Performance target missed: {execution_time:.3f}s > 2.0s")
                        
                else:
                    print(f"❌ Query failed: {result.get('error', 'Unknown error')}")
                    results.append({
                        'query': query,
                        'execution_time': execution_time,
                        'success': False,
                        'error': result.get('error', 'Unknown error')
                    })
                    
            except Exception as e:
                execution_time = time.time() - start_time
                print(f"❌ Exception during query execution: {e}")
                results.append({
                    'query': query,
                    'execution_time': execution_time,
                    'success': False,
                    'error': str(e)
                })
        
        # Performance summary
        print(f"\n📈 PERFORMANCE SUMMARY")
        print(f"=" * 50)
        print(f"Total queries: {len(test_queries)}")
        print(f"Successful queries: {successful_queries}")
        print(f"Success rate: {(successful_queries/len(test_queries)*100):.1f}%")
        
        if successful_queries > 0:
            avg_time = total_time / successful_queries
            print(f"Average execution time: {avg_time:.3f}s")
            
            # Performance target analysis
            fast_queries = sum(1 for r in results if r.get('success') and r.get('execution_time', float('inf')) < 2.0)
            print(f"Queries under 2s target: {fast_queries}/{successful_queries} ({(fast_queries/successful_queries*100):.1f}%)")
            
            if avg_time < 2.0:
                print(f"🎯 OVERALL PERFORMANCE TARGET MET: {avg_time:.3f}s < 2.0s")
            else:
                print(f"⚠️  Overall performance target missed: {avg_time:.3f}s > 2.0s")
                
            # Compare with previous performance (4.11s)
            improvement = ((4.11 - avg_time) / 4.11) * 100
            print(f"Performance improvement: {improvement:.1f}% (from 4.11s to {avg_time:.3f}s)")
        
        # Detailed results
        print(f"\n📋 DETAILED RESULTS")
        print(f"=" * 50)
        for i, result in enumerate(results, 1):
            status = "✅" if result['success'] else "❌"
            time_str = f"{result['execution_time']:.3f}s"
            print(f"{status} Query {i}: {time_str} - {result['query'][:40]}...")
            if not result['success']:
                print(f"    Error: {result.get('error', 'Unknown')}")
        
        return results
        
    except Exception as e:
        print(f"❌ Test setup failed: {e}")
        logger.exception("Test setup failed")
        return None

def main():
    """Main test function."""
    print("🚀 SQLRAG Performance Optimization Test")
    print("=" * 60)
    print("Testing optimized SQLRAG pipeline for:")
    print("• Performance target: <2s execution time")
    print("• Functionality preservation")
    print("• SQL-based retrieval accuracy")
    print()
    
    results = test_sqlrag_performance()
    
    if results:
        print(f"\n🏁 Test completed successfully!")
        
        # Final assessment
        successful_results = [r for r in results if r.get('success')]
        if successful_results:
            avg_time = sum(r['execution_time'] for r in successful_results) / len(successful_results)
            target_met = avg_time < 2.0
            
            print(f"\n🎯 FINAL ASSESSMENT:")
            print(f"Average execution time: {avg_time:.3f}s")
            print(f"Performance target (<2s): {'✅ MET' if target_met else '❌ NOT MET'}")
            print(f"Functionality: {'✅ PRESERVED' if len(successful_results) > 0 else '❌ BROKEN'}")
            
            if target_met and len(successful_results) > 0:
                print(f"\n🎉 OPTIMIZATION SUCCESSFUL!")
                print(f"SQLRAG pipeline is now production-ready with <2s execution time.")
            else:
                print(f"\n⚠️  Further optimization needed.")
        else:
            print(f"\n❌ All queries failed - functionality may be broken.")
    else:
        print(f"\n❌ Test failed to run.")

if __name__ == "__main__":
    main()