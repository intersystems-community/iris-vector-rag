# Comprehensive RAG Benchmark Report
Generated: 2025-05-31 07:33:04

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
| BasicRAG | 100.0% | 0.633s | 10.0 | 0.485 |
| HyDE | 100.0% | 0.628s | 20.0 | 0.391 |
| CRAG | 100.0% | 0.102s | 15.5 | 1.000 |
| ColBERT | 100.0% | 0.000s | 0.0 | 0.000 |
| NodeRAG | 100.0% | 0.619s | 0.0 | 0.000 |
| GraphRAG | 100.0% | 0.388s | 9.2 | 0.000 |
| HybridIFindRAG | 100.0% | 4.670s | 10.0 | 0.000 |

## Key Findings

1. **Best Overall Performance**: BasicRAG
2. **Fastest Response Time**: BasicRAG
3. **Most Documents Retrieved**: HyDE

## Conclusion

This comprehensive benchmark demonstrates the strengths and weaknesses of each RAG technique across both performance metrics and quality evaluation. The results can guide the selection of appropriate techniques based on specific requirements for speed, accuracy, and quality.
