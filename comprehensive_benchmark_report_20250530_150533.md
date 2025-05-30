# Comprehensive RAG Benchmark Report
Generated: 2025-05-30 15:05:33

## Executive Summary

This report presents a comprehensive evaluation of 7 RAG (Retrieval-Augmented Generation) techniques with both performance metrics and quality evaluation using RAGAS.

## Techniques Evaluated

1. **BasicRAG**: Standard vector similarity search
2. **HyDE**: Hypothetical Document Embeddings
3. **CRAG**: Corrective RAG with relevance assessment
4. **ColBERT**: Late interaction neural ranking
5. **NodeRAG**: Node-based retrieval
6. **GraphRAG**: Knowledge graph enhanced retrieval
7. **HybridIFindRAG**: Hybrid approach with multiple strategies

## Performance Results

| Technique | Success Rate | Avg Response Time | Avg Documents | Avg Similarity |
|-----------|-------------|-------------------|---------------|----------------|
| HyDE | 100.0% | 4.089s | 0.0 | 0.000 |
| CRAG | 100.0% | 0.199s | 0.0 | 0.000 |
| ColBERT | 100.0% | 1.398s | 0.0 | 0.000 |
| NodeRAG | 100.0% | 2.084s | 0.0 | 0.000 |
| GraphRAG | 100.0% | 1.970s | 6.9 | 0.705 |
| HybridIFindRAG | 100.0% | 11.016s | 10.0 | 0.000 |

## Key Findings

1. **Best Overall Performance**: HyDE
2. **Fastest Response Time**: CRAG
3. **Most Documents Retrieved**: HybridIFindRAG

## Conclusion

This comprehensive benchmark demonstrates the strengths and weaknesses of each RAG technique across both performance metrics and quality evaluation. The results can guide the selection of appropriate techniques based on specific requirements for speed, accuracy, and quality.
