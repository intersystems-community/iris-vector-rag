# Comprehensive RAGAS Performance Evaluation Report
**Generated:** 2025-06-05 08:32:36
**Configuration:** DBAPI Default with Container Optimization

## Executive Summary

- **Total Pipelines Evaluated:** 4
- **Average Success Rate:** 75.0%
- **Average Response Time:** 0.38 seconds
- **Total Queries per Pipeline:** 10
- **Iterations per Query:** 2

## Pipeline Performance Summary

| Pipeline | Success Rate | Avg Response Time | Avg Documents | RAGAS Score* |
|----------|--------------|-------------------|---------------|--------------|
| BasicRAG | 100.0% | 0.03s | 0.0 | N/A |
| HyDE | 100.0% | 1.49s | 0.0 | N/A |
| CRAG | 0.0% | 0.00s | 0.0 | N/A |
| NodeRAG | 100.0% | 0.00s | 0.0 | N/A |

*RAGAS Score is the average of available RAGAS metrics

## Performance Analysis

- **Most Reliable:** BasicRAG (100.0% success rate)
- **Fastest:** CRAG (0.00s average)

## Configuration Details

- **Connection Type:** DBAPI
- **Database Schema:** RAG
- **Embedding Model:** sentence-transformers/all-MiniLM-L6-v2
- **LLM Provider:** local
- **Top K Documents:** 5
- **Similarity Threshold:** 0.1

## Infrastructure Optimization

This evaluation leveraged the optimized container reuse infrastructure:
- ✅ Container reuse for faster iteration cycles
- ✅ DBAPI connections as default for optimal performance
- ✅ Healthcheck integration for reliable testing
- ✅ Parallel execution support for comprehensive evaluation
