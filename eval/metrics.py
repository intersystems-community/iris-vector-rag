# eval/metrics.py
# Metrics calculations for RAG benchmarking

from typing import List, Dict, Any, Union, Optional
import numpy as np
import re
import difflib
from collections import Counter

# Uncomment these when actually implementing
# import ragas
# from ragas.metrics import context_recall, answer_faithfulness
# from ragchecker import RagChecker, answer_consistency

def calculate_context_recall(results: List[Dict[str, Any]], queries: List[Dict[str, Any]]) -> float:
    """
    Calculate RAGAS context recall metric.
    
    This measures how well the retrieved documents cover the ground truth contexts.
    
    Args:
        results: List of RAG results with retrieved documents
        queries: List of queries with ground truth contexts
        
    Returns:
        Average context recall score (0.0 to 1.0)
    """
    if not results or not queries:
        return 0.0
    
    # Create lookups for easier matching
    query_to_result = {result["query"]: result for result in results}
    query_to_ground_truth = {query["query"]: query.get("ground_truth_contexts", []) for query in queries}
    
    # Calculate recall for each query
    recalls = []
    
    for query_text, ground_truth_contexts in query_to_ground_truth.items():
        if query_text not in query_to_result or not ground_truth_contexts:
            continue
            
        result = query_to_result[query_text]
        retrieved_docs = result.get("retrieved_documents", [])
        
        if not retrieved_docs:
            recalls.append(0.0)
            continue
        
        # Combine all retrieved content into a single string for comparison
        retrieved_content = " ".join([doc.get("content", "") for doc in retrieved_docs])
        
        # Count how many ground truth contexts are covered by retrieved docs
        covered_contexts = 0
        
        for gt_context in ground_truth_contexts:
            # Check for exact matches
            if gt_context.strip() in retrieved_content:
                covered_contexts += 1
                continue
                
            # Check for semantic coverage using word-level matching (simplified version)
            # In a real implementation, this would use more sophisticated semantic matching
            gt_words = set(gt_context.lower().split())
            retrieved_words = set(retrieved_content.lower().split())
            
            # If more than 70% of the ground truth words are in the retrieved content,
            # consider it partially covered
            if gt_words and len(gt_words.intersection(retrieved_words)) / len(gt_words) >= 0.7:
                covered_contexts += 0.7  # Partial credit
        
        # Calculate recall for this query
        recall = covered_contexts / len(ground_truth_contexts)
        recalls.append(recall)
    
    # Calculate average recall
    if not recalls:
        return 0.0
        
    return sum(recalls) / len(recalls)

def calculate_precision_at_k(results: List[Dict[str, Any]], queries: List[Dict[str, Any]], k: int = 5) -> float:
    """
    Calculate precision@k metric.
    
    This measures the proportion of relevant documents among the top k retrieved.
    
    Args:
        results: List of RAG results with retrieved documents
        queries: List of queries with ground truth contexts
        k: Number of top documents to consider
        
    Returns:
        Average precision@k score (0.0 to 1.0)
    """
    if not results or not queries:
        return 0.0
    
    # Create a lookup to easily match query-result pairs
    query_to_result = {result["query"]: result for result in results}
    query_to_ground_truth = {query["query"]: query.get("ground_truth_contexts", []) for query in queries}
    
    # Track precision for each query
    precisions = []
    
    for query_text, ground_truth_contexts in query_to_ground_truth.items():
        if query_text not in query_to_result or not ground_truth_contexts:
            continue
            
        result = query_to_result[query_text]
        retrieved_docs = result.get("retrieved_documents", [])
        
        # Limit to top k documents
        retrieved_docs = retrieved_docs[:k]
        
        if not retrieved_docs:
            precisions.append(0.0)
            continue
        
        # Count how many retrieved documents are in ground truth
        relevant_count = 0
        
        for doc in retrieved_docs:
            doc_content = doc.get("content", "")
            # Check if this document content matches any ground truth context
            if any(doc_content.strip() == gt_context.strip() for gt_context in ground_truth_contexts):
                relevant_count += 1
            # Also check for partial matches (contained within)
            elif any(doc_content.strip() in gt_context.strip() or gt_context.strip() in doc_content.strip() 
                    for gt_context in ground_truth_contexts):
                relevant_count += 0.5  # Give partial credit for partial matches
        
        # Calculate precision for this query
        precision = relevant_count / len(retrieved_docs)
        precisions.append(precision)
    
    # Calculate average precision
    if not precisions:
        return 0.0
        
    return sum(precisions) / len(precisions)

def calculate_answer_faithfulness(results: List[Dict[str, Any]], queries: List[Dict[str, Any]]) -> float:
    """
    Calculate answer faithfulness metric.
    
    This measures how faithful the generated answer is to the retrieved documents.
    
    Args:
        results: List of RAG results with answers and retrieved documents
        queries: List of queries with ground truth answers
        
    Returns:
        Average answer faithfulness score (0.0 to 1.0)
    """
    if not results:
        return 0.0
    
    # Create a lookup to match query-result pairs
    query_to_result = {result["query"]: result for result in results}
    
    # Calculate faithfulness for each query
    faithfulness_scores = []
    
    for result in results:
        query = result["query"]
        answer = result.get("answer", "")
        retrieved_docs = result.get("retrieved_documents", [])
        
        if not answer or not retrieved_docs:
            faithfulness_scores.append(0.0)
            continue
        
        # Combine all retrieved content
        context = " ".join([doc.get("content", "") for doc in retrieved_docs])
        
        # Simple word overlap metric - what percentage of non-stopwords in the answer
        # are found in the context
        answer_words = set(_tokenize(answer.lower()))
        context_words = set(_tokenize(context.lower()))
        
        if not answer_words:
            faithfulness_scores.append(0.0)
            continue
        
        # Calculate overlap
        overlap = len(answer_words.intersection(context_words)) / len(answer_words)
        faithfulness_scores.append(overlap)
    
    # Calculate average faithfulness
    if not faithfulness_scores:
        return 0.0
        
    return sum(faithfulness_scores) / len(faithfulness_scores)

def calculate_answer_relevance(results: List[Dict[str, Any]], queries: List[Dict[str, Any]]) -> float:
    """
    Calculate answer relevance metric.
    
    This measures how relevant the generated answer is to the original query.
    
    Args:
        results: List of RAG results with answers
        queries: List of queries
        
    Returns:
        Average answer relevance score (0.0 to 1.0)
    """
    if not results:
        return 0.0
    
    # Calculate relevance for each result
    relevance_scores = []
    
    for result in results:
        query = result["query"]
        answer = result.get("answer", "")
        
        if not answer:
            relevance_scores.append(0.0)
            continue
        
        # Simple relevance metric based on query term presence in answer
        query_words = set(_tokenize(query.lower()))
        answer_words = set(_tokenize(answer.lower()))
        
        if not query_words:
            relevance_scores.append(0.0)
            continue
        
        # Calculate what percentage of query words appear in the answer
        query_term_presence = len(query_words.intersection(answer_words)) / len(query_words)
        
        # Apply a more lenient scoring since the answer might use synonyms
        # 0.5 points for query term presence, 0.5 points for having any answer
        relevance_score = 0.5 + (0.5 * query_term_presence)
        relevance_scores.append(relevance_score)
    
    # Calculate average relevance
    if not relevance_scores:
        return 0.0
        
    return sum(relevance_scores) / len(relevance_scores)

def calculate_latency_percentiles(latencies: List[float]) -> Dict[str, float]:
    """
    Calculate P50, P95, P99 latency percentiles.
    
    Args:
        latencies: List of latency measurements in milliseconds
        
    Returns:
        Dictionary with keys 'p50', 'p95', 'p99' and their values
    """
    # Validate input
    if not latencies:
        raise ValueError("Latency list is empty")
    
    # Sort latencies if not already sorted
    sorted_latencies = sorted(latencies)
    
    # Calculate percentiles using numpy
    p50 = np.percentile(sorted_latencies, 50)
    p95 = np.percentile(sorted_latencies, 95)
    p99 = np.percentile(sorted_latencies, 99)
    
    # Return as dictionary
    return {
        "p50": float(p50),  # Convert numpy types to native Python for serialization
        "p95": float(p95),
        "p99": float(p99)
    }

def calculate_throughput(num_queries: int, total_time_sec: float) -> float:
    """
    Calculate queries per second (QPS).
    
    Args:
        num_queries: Number of queries processed
        total_time_sec: Total time taken in seconds
        
    Returns:
        Queries per second (QPS)
    """
    # Validate inputs
    if num_queries < 0:
        raise ValueError("Number of queries must be non-negative")
    if total_time_sec <= 0:
        raise ValueError("Total time must be positive")
    
    # Calculate QPS
    return num_queries / total_time_sec

def normalize_metrics(metrics: Dict[str, float], 
                     invert_latency: bool = True,
                     scale_to_unit: bool = False) -> Dict[str, float]:
    """
    Normalize metrics for visualization, optionally inverting latency metrics.
    
    For radar charts and other visualizations, we want all metrics to follow
    "higher is better" pattern, so we invert latency (lower is better) to
    latency_score (higher is better).
    
    Args:
        metrics: Dictionary of metric names to values
        invert_latency: Whether to invert latency metrics (p50, p95, p99)
        scale_to_unit: Whether to scale all metrics to 0-1 range
        
    Returns:
        Dictionary with normalized metrics
    """
    if not metrics:
        return {}
    
    # Create a copy to avoid modifying the original
    normalized = metrics.copy()
    
    # Identify latency metrics by name
    latency_metrics = [k for k in normalized.keys() 
                      if any(k.startswith(prefix) or k.endswith(suffix)
                            for prefix in ['latency', 'p50', 'p95', 'p99']
                            for suffix in ['_latency', '_ms'])]
    
    # Invert latency metrics (lower is better) to latency_score (higher is better)
    if invert_latency:
        for metric in latency_metrics:
            if normalized[metric] > 0:  # Avoid division by zero
                # Convert to score where higher is better
                # For latency, we use 1/x transformation
                normalized[f"{metric}_score"] = 1.0 / normalized[metric]
                # Delete the original metric so we don't have both
                del normalized[metric]
    
    # Scale all metrics to 0-1 range if requested
    if scale_to_unit:
        all_values = [v for v in normalized.values() if v > 0]
        if all_values:  # Check if we have any positive values
            max_value = max(all_values)
            min_value = min(all_values)
            
            # Scale only if we have a reasonable range to avoid division by zero
            if max_value > min_value:
                for k in list(normalized.keys()):
                    if normalized[k] > 0:
                        normalized[k] = (normalized[k] - min_value) / (max_value - min_value)
    
    return normalized

# Helper functions
def _tokenize(text: str) -> List[str]:
    """Tokenize text into words, removing punctuation and stopwords."""
    # Strip punctuation
    text = re.sub(r'[^\w\s]', '', text)
    
    # Split into words
    words = text.split()
    
    # Filter out stopwords (simple English stopwords)
    stopwords = {'a', 'an', 'the', 'and', 'or', 'but', 'if', 'then', 'else', 'when',
                'at', 'by', 'for', 'with', 'about', 'against', 'between', 'into',
                'through', 'during', 'before', 'after', 'above', 'below', 'to', 'from',
                'up', 'down', 'in', 'out', 'on', 'off', 'over', 'under', 'again',
                'further', 'then', 'once', 'here', 'there', 'when', 'where', 'why',
                'how', 'all', 'any', 'both', 'each', 'few', 'more', 'most', 'other',
                'some', 'such', 'no', 'nor', 'not', 'only', 'own', 'same', 'so',
                'than', 'too', 'very', 's', 't', 'can', 'will', 'just', 'don',
                'should', 'now', 'is', 'am', 'are', 'was', 'were', 'be', 'being',
                'been', 'have', 'has', 'had', 'having', 'do', 'does', 'did', 'doing',
                'would', 'could', 'should', 'shall', 'might', 'must'}
    
    return [word for word in words if word.lower() not in stopwords]

# ---- Additional metrics for standard benchmarks ----

def calculate_rouge_n(hypothesis: str, reference: str, n: int = 2) -> float:
    """
    Calculate ROUGE-N score between a hypothesis and reference text.
    
    Args:
        hypothesis: The generated text to evaluate
        reference: The reference or ground truth text
        n: The n-gram size (1 for unigrams, 2 for bigrams, etc.)
        
    Returns:
        ROUGE-N F1 score (0.0 to 1.0)
    """
    if not hypothesis or not reference:
        return 0.0
    
    # Tokenize and create n-grams
    hyp_tokens = _tokenize(hypothesis.lower())
    ref_tokens = _tokenize(reference.lower())
    
    # Generate n-grams
    def get_ngrams(tokens, n):
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    hyp_ngrams = get_ngrams(hyp_tokens, n)
    ref_ngrams = get_ngrams(ref_tokens, n)
    
    if not hyp_ngrams or not ref_ngrams:
        return 0.0
    
    # Count n-grams
    hyp_counter = Counter(hyp_ngrams)
    ref_counter = Counter(ref_ngrams)
    
    # Find overlapping n-grams
    matches = sum((hyp_counter & ref_counter).values())
    
    # Calculate precision and recall
    precision = matches / max(1, len(hyp_ngrams))
    recall = matches / max(1, len(ref_ngrams))
    
    # Calculate F1 score
    if precision + recall > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    return f1

def calculate_answer_f1(predicted: str, ground_truth: str) -> float:
    """
    Calculate token-level F1 score between predicted and ground truth answers.
    Used in MultiHopQA and other QA benchmarks.
    
    Args:
        predicted: The predicted answer text
        ground_truth: The ground truth answer text
        
    Returns:
        F1 score (0.0 to 1.0)
    """
    if not predicted or not ground_truth:
        return 0.0
    
    # Tokenize
    pred_tokens = _tokenize(predicted.lower())
    true_tokens = _tokenize(ground_truth.lower())
    
    if not pred_tokens or not true_tokens:
        return 0.0
    
    # Get token sets
    pred_set = set(pred_tokens)
    true_set = set(true_tokens)
    
    # Calculate intersection
    intersection = pred_set.intersection(true_set)
    
    # Calculate precision and recall
    precision = len(intersection) / len(pred_set) if pred_set else 0.0
    recall = len(intersection) / len(true_set) if true_set else 0.0
    
    # Calculate F1 score
    if precision + recall > 0:
        f1 = (2 * precision * recall) / (precision + recall)
    else:
        f1 = 0.0
    
    return f1

def calculate_mrr(results: List[Dict[str, Any]], queries: List[Dict[str, Any]]) -> float:
    """
    Calculate Mean Reciprocal Rank (MRR) for factoid questions.
    Used in BioASQ and other factoid QA benchmarks.
    
    MRR is the average of the reciprocal ranks of the first relevant item for each query.
    
    Args:
        results: List of RAG results with answers
        queries: List of queries with ground truth answers
        
    Returns:
        MRR score (0.0 to 1.0)
    """
    if not results or not queries:
        return 0.0
    
    # Create a lookup for ground truth answers
    query_to_ground_truth = {query["query"]: query.get("ground_truth_answer", "") for query in queries}
    
    # Calculate reciprocal rank for each query
    reciprocal_ranks = []
    
    for result in results:
        query = result["query"]
        
        if query not in query_to_ground_truth or not query_to_ground_truth[query]:
            continue
            
        # Get ground truth
        ground_truth = query_to_ground_truth[query]
        
        # For factoid questions, the result might include a ranked list of answers
        # or a single answer with supporting documents
        if "ranked_answers" in result and result["ranked_answers"]:
            # If we have ranked answers, find the first correct one
            rank = 1
            found = False
            
            for answer in result["ranked_answers"]:
                # Check if this answer is correct (exact or partial match)
                answer_text = answer.get("text", "")
                similarity = _calculate_answer_similarity(answer_text, ground_truth)
                
                if similarity >= 0.8:  # If 80% similar, consider it correct
                    found = True
                    break
                    
                rank += 1
            
            if found:
                reciprocal_ranks.append(1.0 / rank)
            else:
                reciprocal_ranks.append(0.0)
                
        elif "answer" in result and result["answer"]:
            # If we have a single answer, check if it's correct
            answer = result["answer"]
            similarity = _calculate_answer_similarity(answer, ground_truth)
            
            if similarity >= 0.8:  # If 80% similar, consider it correct
                reciprocal_ranks.append(1.0)  # Rank 1
            else:
                reciprocal_ranks.append(0.0)
    
    # Calculate MRR
    if not reciprocal_ranks:
        return 0.0
        
    return sum(reciprocal_ranks) / len(reciprocal_ranks)

def _calculate_answer_similarity(answer1: str, answer2: str) -> float:
    """
    Calculate similarity between two answer strings.
    
    Args:
        answer1: First answer string
        answer2: Second answer string
        
    Returns:
        Similarity score (0.0 to 1.0)
    """
    # Normalize and tokenize
    answer1 = re.sub(r'[^\w\s]', '', answer1.lower())
    answer2 = re.sub(r'[^\w\s]', '', answer2.lower())
    
    # If either is empty, return 0
    if not answer1 or not answer2:
        return 0.0
    
    # Use difflib for sequence comparison
    similarity = difflib.SequenceMatcher(None, answer1, answer2).ratio()
    
    return similarity
def calculate_hnsw_performance_metrics(
    hnsw_latencies: List[float],
    sequential_latencies: List[float],
    hnsw_similarities: List[List[float]],
    sequential_similarities: List[List[float]]
) -> Dict[str, float]:
    """
    Calculate HNSW-specific performance metrics.
    
    Args:
        hnsw_latencies: Query latencies with HNSW indexes (ms)
        sequential_latencies: Query latencies with sequential scan (ms)  
        hnsw_similarities: Similarity scores from HNSW queries
        sequential_similarities: Similarity scores from sequential queries
        
    Returns:
        Dictionary with HNSW performance metrics
    """
    metrics = {}
    
    if not hnsw_latencies or not sequential_latencies:
        return metrics
    
    # Performance improvement metrics
    avg_hnsw_latency = np.mean(hnsw_latencies)
    avg_sequential_latency = np.mean(sequential_latencies)
    
    if avg_sequential_latency > 0:
        speedup_ratio = avg_sequential_latency / avg_hnsw_latency
        performance_improvement = (avg_sequential_latency - avg_hnsw_latency) / avg_sequential_latency * 100
        
        metrics["hnsw_speedup_ratio"] = float(speedup_ratio)
        metrics["hnsw_performance_improvement_pct"] = float(performance_improvement)
    
    # HNSW-specific latency percentiles
    if hnsw_latencies:
        try:
            hnsw_percentiles = calculate_latency_percentiles(hnsw_latencies)
            for key, value in hnsw_percentiles.items():
                metrics[f"hnsw_{key}"] = value
        except ValueError:
            pass
    
    # Quality preservation metrics (how well HNSW approximation preserves quality)
    if hnsw_similarities and sequential_similarities:
        try:
            # Calculate quality preservation across all queries
            quality_preservation_scores = []
            
            for hnsw_sims, seq_sims in zip(hnsw_similarities, sequential_similarities):
                if hnsw_sims and seq_sims:
                    # Compare top similarities
                    hnsw_max = max(hnsw_sims)
                    seq_max = max(seq_sims)
                    
                    if seq_max > 0:
                        quality_ratio = hnsw_max / seq_max
                        quality_preservation_scores.append(quality_ratio)
            
            if quality_preservation_scores:
                metrics["hnsw_quality_preservation"] = float(np.mean(quality_preservation_scores))
                metrics["hnsw_quality_preservation_std"] = float(np.std(quality_preservation_scores))
        except Exception:
            pass
    
    return metrics

def calculate_hnsw_scalability_metrics(
    document_counts: List[int], 
    query_latencies: List[float]
) -> Dict[str, float]:
    """
    Calculate HNSW scalability metrics.
    
    Args:
        document_counts: List of document counts tested
        query_latencies: Corresponding query latencies (ms)
        
    Returns:
        Dictionary with scalability metrics
    """
    metrics = {}
    
    if len(document_counts) < 2 or len(query_latencies) < 2:
        return metrics
    
    if len(document_counts) != len(query_latencies):
        return metrics
    
    # Calculate scaling coefficient (how latency grows with document count)
    try:
        # Log-log regression to find scaling exponent
        log_docs = np.log(document_counts)
        log_latencies = np.log(query_latencies)
        
        # Simple linear regression on log-log scale: log(latency) = a * log(docs) + b
        # The coefficient 'a' tells us the scaling behavior
        coeffs = np.polyfit(log_docs, log_latencies, 1)
        scaling_exponent = coeffs[0]
        
        metrics["hnsw_scaling_exponent"] = float(scaling_exponent)
        
        # Ideal HNSW should have sub-linear scaling (exponent < 1.0)
        if scaling_exponent < 1.0:
            metrics["hnsw_sublinear_scaling"] = 1.0  # Boolean metric: 1 if true, 0 if false
        else:
            metrics["hnsw_sublinear_scaling"] = 0.0
        
        # Calculate efficiency compared to linear scaling
        linear_scaling_expected = document_counts[-1] / document_counts[0]
        actual_scaling = query_latencies[-1] / query_latencies[0]
        
        if linear_scaling_expected > 0:
            scaling_efficiency = linear_scaling_expected / actual_scaling
            metrics["hnsw_scaling_efficiency"] = float(scaling_efficiency)
    
    except Exception:
        pass
    
    return metrics

def calculate_hnsw_index_effectiveness_metrics(
    query_latencies: List[float],
    index_parameters: Dict[str, Any] = None
) -> Dict[str, float]:
    """
    Calculate metrics for HNSW index effectiveness.
    
    Args:
        query_latencies: Query latencies with current HNSW parameters (ms)
        index_parameters: HNSW parameters used (M, efConstruction, etc.)
        
    Returns:
        Dictionary with index effectiveness metrics
    """
    metrics = {}
    
    if not query_latencies:
        return metrics
    
    # Basic performance metrics
    metrics["hnsw_avg_latency"] = float(np.mean(query_latencies))
    metrics["hnsw_latency_variance"] = float(np.var(query_latencies))
    metrics["hnsw_latency_cv"] = float(np.std(query_latencies) / np.mean(query_latencies))
    
    # Consistency metrics (lower variance is better)
    latency_std = np.std(query_latencies)
    latency_mean = np.mean(query_latencies)
    
    if latency_mean > 0:
        # Coefficient of variation (normalized variance)
        consistency_score = 1.0 / (1.0 + latency_std / latency_mean)
        metrics["hnsw_consistency_score"] = float(consistency_score)
    
    # Index parameter effectiveness (if provided)
    if index_parameters:
        # Record parameters for analysis
        if "M" in index_parameters:
            metrics["hnsw_parameter_M"] = float(index_parameters["M"])
        if "efConstruction" in index_parameters:
            metrics["hnsw_parameter_efConstruction"] = float(index_parameters["efConstruction"])
        
        # Calculate efficiency score based on latency and parameters
        # Lower M and efConstruction with good performance = higher efficiency
        if "M" in index_parameters and "efConstruction" in index_parameters:
            parameter_complexity = index_parameters["M"] * index_parameters["efConstruction"]
            if parameter_complexity > 0 and latency_mean > 0:
                # Efficiency = 1 / (latency * parameter_complexity)
                efficiency = 1.0 / (latency_mean * parameter_complexity / 1000)  # Normalize
                metrics["hnsw_parameter_efficiency"] = float(efficiency)
    
    return metrics

def calculate_benchmark_metrics(results: List[Dict[str, Any]], 
                              queries: List[Dict[str, Any]], 
                              benchmark_type: str = "multihop") -> Dict[str, float]:
    """
    Calculate metrics specific to a benchmark dataset type.
    
    Args:
        results: List of RAG results
        queries: List of queries with ground truth
        benchmark_type: Type of benchmark ('multihop', 'bioasq', etc.)
        
    Returns:
        Dictionary of benchmark-specific metrics
    """
    metrics = {}
    
    if benchmark_type == "multihop":
        # For MultiHopQA, calculate answer F1 and supporting facts F1
        answer_f1_scores = []
        supporting_facts_f1_scores = []
        
        # Create lookup for results and ground truth
        query_to_result = {r["query"]: r for r in results}
        query_to_truth = {q["query"]: q for q in queries}
        
        for query in queries:
            query_text = query.get("query", "")
            if query_text not in query_to_result:
                continue
                
            result = query_to_result[query_text]
            
            # Calculate answer F1
            pred_answer = result.get("answer", "")
            true_answer = query.get("ground_truth_answer", "")
            if pred_answer and true_answer:
                answer_f1 = calculate_answer_f1(pred_answer, true_answer)
                answer_f1_scores.append(answer_f1)
            
            # Calculate supporting facts F1
            retrieved_docs = result.get("retrieved_documents", [])
            true_contexts = query.get("ground_truth_contexts", [])
            
            if retrieved_docs and true_contexts:
                # For simplicity, treat retrieved docs as supporting facts
                retrieved_content = [doc.get("content", "") for doc in retrieved_docs]
                precision = calculate_precision_at_k(results=[result], queries=[query], k=len(retrieved_docs))
                recall = calculate_context_recall(results=[result], queries=[query])
                
                # Calculate F1 from precision and recall
                if precision + recall > 0:
                    supporting_f1 = (2 * precision * recall) / (precision + recall)
                else:
                    supporting_f1 = 0.0
                    
                supporting_facts_f1_scores.append(supporting_f1)
        
        # Calculate average scores
        if answer_f1_scores:
            metrics["answer_f1"] = sum(answer_f1_scores) / len(answer_f1_scores)
        else:
            metrics["answer_f1"] = 0.0
            
        if supporting_facts_f1_scores:
            metrics["supporting_facts_f1"] = sum(supporting_facts_f1_scores) / len(supporting_facts_f1_scores)
        else:
            metrics["supporting_facts_f1"] = 0.0
            
        # Calculate joint F1 (following MultiHopQA)
        metrics["joint_f1"] = metrics["answer_f1"] * metrics["supporting_facts_f1"]
    
    elif benchmark_type == "bioasq":
        # For BioASQ, calculate yes/no accuracy, factoid MRR, list F1, and summary ROUGE
        yes_no_correct = 0
        yes_no_total = 0
        list_f1_scores = []
        summary_rouge_scores = []
        
        # Calculate metrics for each query
        for i, query in enumerate(queries):
            if i >= len(results):
                continue
                
            result = results[i]
            query_type = query.get("type", "").lower()
            
            # Yes/No questions
            if query_type == "yesno":
                true_answer = query.get("ground_truth_answer", "").lower()
                pred_answer = result.get("answer", "").lower()
                
                # Simple exact match for yes/no
                if (("yes" in true_answer and "yes" in pred_answer) or 
                    ("no" in true_answer and "no" in pred_answer)):
                    yes_no_correct += 1
                yes_no_total += 1
            
            # Factoid questions handled by calculate_mrr
            
            # List questions
            elif query_type == "list":
                true_items = query.get("ground_truth_list", [])
                
                # Extract predicted list items from answer
                # This is simplistic - in practice you'd need better extraction
                pred_answer = result.get("answer", "")
                pred_items = [item.strip() for item in re.split(r'[,;•\n]', pred_answer) if item.strip()]
                
                if true_items and pred_items:
                    # Calculate list F1
                    true_set = set(true_items)
                    pred_set = set(pred_items)
                    
                    intersection = true_set.intersection(pred_set)
                    precision = len(intersection) / len(pred_set) if pred_set else 0.0
                    recall = len(intersection) / len(true_set) if true_set else 0.0
                    
                    if precision + recall > 0:
                        list_f1 = (2 * precision * recall) / (precision + recall)
                    else:
                        list_f1 = 0.0
                        
                    list_f1_scores.append(list_f1)
            
            # Summary questions
            elif query_type == "summary":
                true_summary = query.get("ground_truth_summary", "")
                pred_summary = result.get("answer", "")
                
                if true_summary and pred_summary:
                    # Calculate ROUGE-2
                    rouge2 = calculate_rouge_n(pred_summary, true_summary, n=2)
                    summary_rouge_scores.append(rouge2)
        
        # Add metrics to results
        if yes_no_total > 0:
            metrics["yesno_accuracy"] = yes_no_correct / yes_no_total
        else:
            metrics["yesno_accuracy"] = 0.0
            
        # Add MRR for factoid questions
        metrics["factoid_mrr"] = calculate_mrr(results, queries)
        
        # Add list F1
        if list_f1_scores:
            metrics["list_f1"] = sum(list_f1_scores) / len(list_f1_scores)
        else:
            metrics["list_f1"] = 0.0
            
        # Add summary ROUGE
        if summary_rouge_scores:
            metrics["summary_rouge2"] = sum(summary_rouge_scores) / len(summary_rouge_scores)
        else:
            metrics["summary_rouge2"] = 0.0
    
    # Add more benchmark types as needed
    
    return metrics
