# Comprehensive RAGAS Performance Evaluation Report
**Generated:** 2025-06-08 21:19:02
**Configuration:** DBAPI Default with Container Optimization

## Executive Summary

- **Total Pipelines Evaluated:** 7
- **Average Success Rate:** 85.7%
- **Average Response Time:** 2.16 seconds
- **Total Queries per Pipeline:** 3
- **Iterations per Query:** 3

## Pipeline Performance Summary

| Pipeline | Success Rate | Avg Response Time | Avg Documents | RAGAS Score* |
|----------|--------------|-------------------|---------------|--------------|
| basic | 100.0% | 3.85s | 10.0 | N/A |
| hyde | 100.0% | 5.24s | 10.0 | N/A |
| crag | 100.0% | 3.07s | 10.0 | N/A |
| colbert | 0.0% | 0.00s | 0.0 | N/A |
| noderag | 100.0% | 0.88s | 10.0 | N/A |
| graphrag | 100.0% | 0.86s | 10.0 | N/A |
| hybrid_ifind | 100.0% | 1.20s | 10.0 | N/A |

*RAGAS Score is the average of available RAGAS metrics

## Performance Analysis

- **Most Reliable:** basic (100.0% success rate)
- **Fastest:** colbert (0.00s average)

## Configuration Details

- **Connection Type:** DBAPI
- **Database Schema:** RAG
- **Embedding Model:** sentence-transformers/all-MiniLM-L6-v2
- **LLM Provider:** openai
- **Top K Documents:** 10
- **Similarity Threshold:** 0.1

## Infrastructure Optimization

This evaluation leveraged the optimized container reuse infrastructure:
- ✅ Container reuse for faster iteration cycles
- ✅ DBAPI connections as default for optimal performance
- ✅ Healthcheck integration for reliable testing
- ✅ Parallel execution support for comprehensive evaluation
