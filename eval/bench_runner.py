# eval/bench_runner.py
# RAG benchmarking runner

import os
import json
import time
import logging
from typing import List, Dict, Any, Callable, Optional
from datetime import datetime

from eval.metrics import (
    calculate_context_recall, 
    calculate_precision_at_k,
    calculate_latency_percentiles,
    calculate_throughput
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("rag_bench_runner")

def run_technique_benchmark(
    technique_name: str,
    pipeline_func: Callable,
    queries: List[Dict[str, Any]],
    ground_truth: Optional[List[Dict[str, Any]]] = None,
    iris_connector = None,  # Type will be IrisConnector from common/iris_connector.py
    embedding_func = None,
    llm_func = None,
    **technique_params
) -> Dict[str, Any]:
    """
    Run benchmark for a single RAG technique.
    
    Args:
        technique_name: Name of the RAG technique
        pipeline_func: Function that implements the RAG pipeline
        queries: List of query dictionaries to run through the pipeline
        ground_truth: Optional list of ground truth contexts for evaluation
        iris_connector: IRIS database connector
        embedding_func: Function to generate embeddings
        llm_func: Function to generate answers
        technique_params: Additional parameters specific to the technique
        
    Returns:
        Dictionary with benchmark results
    """
    logger.info(f"Running benchmark for {technique_name} with {len(queries)} queries")
    
    if ground_truth is None:
        ground_truth = []
    
    # Standardize ground truth format
    query_to_ground_truth = {}
    for gt in ground_truth:
        query = gt.get("query")
        if query:
            query_to_ground_truth[query] = gt
    
    # Prepare result storage
    results = {
        "pipeline": technique_name,
        "queries_run": len(queries),
        "start_time": datetime.now().isoformat(),
        "parameters": technique_params,
        "query_results": [],
        "metrics": {}
    }
    
    # Track all latencies
    latencies = []
    
    # Time the entire benchmark
    benchmark_start = time.time()
    
    # Run each query through the pipeline
    for i, query_obj in enumerate(queries):
        query = query_obj.get("query")
        if not query:
            logger.warning(f"Skipping query at index {i}: No query text found")
            continue
        
        logger.info(f"Processing query {i+1}/{len(queries)}: {query[:50]}...")
        
        # Time this specific query
        query_start = time.time()
        
        try:
            # Call the pipeline function with the query and any technique-specific parameters
            pipeline_result = pipeline_func(
                query=query,
                iris_connector=iris_connector,
                embedding_func=embedding_func,
                llm_func=llm_func,
                **technique_params
            )
            
            # Calculate latency
            latency_ms = (time.time() - query_start) * 1000
            latencies.append(latency_ms)
            
            # Store result with latency
            result_obj = {
                "query": query,
                "latency_ms": latency_ms
            }
            
            # Add other pipeline result fields
            if isinstance(pipeline_result, dict):
                for k, v in pipeline_result.items():
                    if k != "query":  # Avoid duplicate
                        if k == "retrieved_documents" and isinstance(v, list):
                            # Convert Document objects to dicts
                            result_obj[k] = [doc.to_dict() if hasattr(doc, 'to_dict') else doc for doc in v]
                        else:
                            result_obj[k] = v
            
            # Add ground truth if available
            if query in query_to_ground_truth:
                result_obj["ground_truth"] = query_to_ground_truth[query]
            
            results["query_results"].append(result_obj)
            
        except Exception as e:
            logger.error(f"Error processing query {i+1}: {str(e)}")
            results["query_results"].append({
                "query": query,
                "error": str(e)
            })
    
    # Calculate benchmark duration
    benchmark_duration = time.time() - benchmark_start
    results["duration_seconds"] = benchmark_duration
    
    # Calculate aggregate metrics
    if latencies:
        # Performance metrics
        results["metrics"].update(calculate_latency_percentiles(latencies))
        results["metrics"]["throughput_qps"] = calculate_throughput(
            len([r for r in results["query_results"] if "error" not in r]),
            benchmark_duration
        )
        
        # Try to calculate retrieval quality metrics if we have ground truth
        try:
            # Check if we have ground truth contexts to compare against
            has_ground_truth = any("ground_truth" in r for r in results["query_results"])
            has_contexts = any("ground_truth_contexts" in r.get("ground_truth", {}) 
                              for r in results["query_results"] if "ground_truth" in r)
            
            if has_ground_truth and has_contexts:
                # Prepare data for metrics calculation
                gt_queries = []
                for r in results["query_results"]:
                    if "ground_truth" in r and "ground_truth_contexts" in r["ground_truth"]:
                        gt_obj = {
                            "query": r["query"],
                            "ground_truth_contexts": r["ground_truth"]["ground_truth_contexts"]
                        }
                        gt_queries.append(gt_obj)
                
                # Calculate context recall
                if gt_queries:
                    recall = calculate_context_recall(results["query_results"], gt_queries)
                    results["metrics"]["context_recall"] = recall
                    
                    # Calculate precision@k
                    precision = calculate_precision_at_k(
                        results["query_results"], 
                        gt_queries,
                        k=5  # Top 5 documents
                    )
                    results["metrics"]["precision_at_5"] = precision
        except Exception as e:
            logger.error(f"Error calculating retrieval metrics: {str(e)}")
            results["metrics"]["retrieval_metrics_error"] = str(e)
    
    # Add end time
    results["end_time"] = datetime.now().isoformat()
    
    logger.info(f"Benchmark for {technique_name} completed in {benchmark_duration:.2f} seconds")
    
    return results

def run_all_techniques_benchmark(
    queries: List[Dict[str, Any]],
    ground_truth: Optional[List[Dict[str, Any]]] = None,
    techniques: Dict[str, Dict[str, Any]] = None,
    output_path: Optional[str] = None
) -> Dict[str, Dict[str, Any]]:
    """
    Run benchmarks for multiple RAG techniques and save results.
    
    Args:
        queries: List of query dictionaries to run through all pipelines
        ground_truth: Optional list of ground truth contexts for evaluation
        techniques: Dictionary mapping technique names to their configuration
                   Each technique config needs 'pipeline_func' and optional params
        output_path: Path to save the benchmark results as JSON
        
    Returns:
        Dictionary mapping technique names to their benchmark results
    """
    if not techniques:
        raise ValueError("No techniques provided for benchmarking")
    
    logger.info(f"Starting benchmarks for {len(techniques)} techniques with {len(queries)} queries")
    
    results = {}
    
    # Run benchmark for each technique
    for tech_name, tech_config in techniques.items():
        logger.info(f"Starting benchmark for {tech_name}")
        
        # Extract pipeline function and parameters
        pipeline_func = tech_config.get("pipeline_func")
        if not pipeline_func:
            logger.error(f"No pipeline function provided for {tech_name}")
            continue
        
        # Extract other parameters
        params = {k: v for k, v in tech_config.items() if k != "pipeline_func"}
        
        # Run the benchmark
        try:
            tech_result = run_technique_benchmark(
                technique_name=tech_name,
                pipeline_func=pipeline_func,
                queries=queries,
                ground_truth=ground_truth,
                **params
            )
            results[tech_name] = tech_result
        except Exception as e:
            logger.error(f"Error in benchmark for {tech_name}: {str(e)}")
            results[tech_name] = {
                "pipeline": tech_name,
                "error": str(e)
            }
    
    # Calculate summary metrics
    summary = {
        "total_techniques": len(techniques),
        "successful_techniques": sum(1 for r in results.values() if "error" not in r),
        "total_queries": len(queries),
        "run_date": datetime.now().isoformat()
    }
    
    # Add result collection
    all_results = {
        "summary": summary,
        "results": results
    }
    
    # Save results if output path provided
    if output_path:
        # Create output directory if it doesn't exist
        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            
        try:
            with open(output_path, 'w') as f:
                json.dump(all_results, f, indent=2)
            logger.info(f"Benchmark results saved to {output_path}")
        except Exception as e:
            logger.error(f"Error saving results to {output_path}: {str(e)}")
    
    return results

def load_benchmark_results(input_path: str) -> Dict[str, Dict[str, Any]]:
    """
    Load benchmark results from a JSON file.
    
    Args:
        input_path: Path to the benchmark results JSON file
        
    Returns:
        Dictionary with benchmark results
    """
    try:
        with open(input_path, 'r') as f:
            data = json.load(f)
        
        # Handle both raw results format and the nested format
        if "results" in data:
            return data["results"]
        return data
    except Exception as e:
        logger.error(f"Error loading benchmark results from {input_path}: {str(e)}")
        return {}
