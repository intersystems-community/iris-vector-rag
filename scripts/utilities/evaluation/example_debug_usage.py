#!/usr/bin/env python3
"""
Example usage of the RAGAS Context Debug Test Harness

This script demonstrates how to use the debug harness programmatically
to verify context handling in RAG pipelines.
"""

import sys
import json
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from scripts.utilities.evaluation.debug_basicrag_ragas_context import RAGASContextDebugHarness


def example_basic_usage():
    """Demonstrate basic harness usage."""
    print("=== Basic Usage Example ===")
    
    # Create harness instance
    harness = RAGASContextDebugHarness()
    
    # Run debug session for BasicRAG with 2 queries
    results = harness.run_debug_session("BasicRAG", num_queries=2)
    
    # Print key results
    print(f"Pipeline: {results['pipeline_name']}")
    print(f"Successful executions: {results['successful_executions']}")
    print(f"Results with contexts: {results['results_with_contexts']}")
    
    if results['ragas_scores']:
        print("RAGAS Scores:")
        for metric, score in results['ragas_scores'].items():
            print(f"  {metric}: {score:.4f}")
    
    return results


def example_detailed_analysis():
    """Demonstrate detailed context analysis."""
    print("\n=== Detailed Analysis Example ===")
    
    harness = RAGASContextDebugHarness()
    
    # Initialize RAGAS framework
    harness.initialize_ragas_framework()
    
    # Load test queries
    queries = harness.load_test_queries(1)  # Just one query for detailed analysis
    
    # Get pipeline
    try:
        pipeline = harness.get_pipeline("BasicRAG")
        
        # Execute with detailed debugging
        results = harness.execute_pipeline_with_debug(pipeline, queries)
        
        # Analyze the first result in detail
        if results:
            result = results[0]
            print(f"Query: {result['query']}")
            print(f"Answer length: {len(result['answer'])} characters")
            print(f"Number of contexts: {len(result['contexts'])}")
            print(f"Execution time: {result['execution_time']:.2f} seconds")
            
            # Show debug info
            debug_info = result['debug_info']
            print(f"Raw result keys: {debug_info['raw_result_keys']}")
            print(f"Total context length: {debug_info['contexts_total_length']} characters")
            
            # Show first context sample
            if result['contexts']:
                print(f"First context sample: {result['contexts'][0][:150]}...")
        
    except Exception as e:
        print(f"Error in detailed analysis: {e}")


def example_multiple_pipelines():
    """Demonstrate testing multiple pipelines."""
    print("\n=== Multiple Pipelines Example ===")
    
    harness = RAGASContextDebugHarness()
    
    # List of pipelines to test
    pipelines_to_test = ["BasicRAG", "HyDE", "GraphRAG"]  # Add more as available
    
    results_summary = {}
    
    for pipeline_name in pipelines_to_test:
        try:
            print(f"Testing {pipeline_name}...")
            results = harness.run_debug_session(pipeline_name, num_queries=1)
            
            # Store summary
            results_summary[pipeline_name] = {
                'successful': results['successful_executions'],
                'with_contexts': results['results_with_contexts'],
                'ragas_scores': results['ragas_scores']
            }
            
        except Exception as e:
            print(f"Failed to test {pipeline_name}: {e}")
            results_summary[pipeline_name] = {'error': str(e)}
    
    # Print comparison
    print("\nPipeline Comparison:")
    for pipeline, summary in results_summary.items():
        if 'error' in summary:
            print(f"  {pipeline}: ERROR - {summary['error']}")
        else:
            context_precision = summary['ragas_scores'].get('context_precision', 'N/A')
            print(f"  {pipeline}: {summary['with_contexts']} contexts, "
                  f"precision: {context_precision}")


def example_save_results():
    """Demonstrate saving results to file."""
    print("\n=== Save Results Example ===")
    
    harness = RAGASContextDebugHarness()
    
    # Run debug session
    results = harness.run_debug_session("BasicRAG", num_queries=2)
    
    # Save to JSON file
    output_file = "debug_results_example.json"
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2, default=str)
    
    print(f"Results saved to: {output_file}")
    
    # Load and verify
    with open(output_file, 'r') as f:
        loaded_results = json.load(f)
    
    print(f"Loaded results for pipeline: {loaded_results['pipeline_name']}")
    print(f"Timestamp: {loaded_results['timestamp']}")


def example_custom_queries():
    """Demonstrate using custom queries."""
    print("\n=== Custom Queries Example ===")
    
    harness = RAGASContextDebugHarness()
    
    # Define custom queries
    custom_queries = [
        {
            "query": "What is machine learning?",
            "expected_answer": "Machine learning is a subset of AI that enables computers to learn without explicit programming.",
            "category": "technology"
        },
        {
            "query": "How do vaccines work?",
            "expected_answer": "Vaccines work by training the immune system to recognize and fight specific pathogens.",
            "category": "medical"
        }
    ]
    
    # Get pipeline
    try:
        pipeline = harness.get_pipeline("BasicRAG")
        
        # Execute with custom queries
        results = harness.execute_pipeline_with_debug(pipeline, custom_queries)
        
        # Calculate RAGAS metrics
        ragas_scores = harness.calculate_ragas_metrics(results)
        
        print("Custom Query Results:")
        for i, result in enumerate(results):
            print(f"  Query {i+1}: {len(result['contexts'])} contexts")
        
        if ragas_scores:
            print("RAGAS Scores:")
            for metric, score in ragas_scores.items():
                print(f"  {metric}: {score:.4f}")
    
    except Exception as e:
        print(f"Error with custom queries: {e}")


def main():
    """Run all examples."""
    print("RAGAS Context Debug Test Harness - Usage Examples")
    print("=" * 60)
    
    try:
        # Run examples
        example_basic_usage()
        example_detailed_analysis()
        example_multiple_pipelines()
        example_save_results()
        example_custom_queries()
        
        print("\n" + "=" * 60)
        print("All examples completed successfully!")
        
    except Exception as e:
        print(f"Error running examples: {e}")
        print("Make sure you have:")
        print("1. IRIS database running and accessible")
        print("2. OpenAI API key configured")
        print("3. Required dependencies installed")
        print("4. At least one pipeline (BasicRAG) available")


if __name__ == "__main__":
    main()