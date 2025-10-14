# Comprehensive RAGAS Performance Evaluation Report
**Generated:** 2025-06-08 18:55:09
**Configuration:** DBAPI Default with Container Optimization

## Executive Summary

- **Total Pipelines Evaluated:** 7
- **Average Success Rate:** 100.0%
- **Average Response Time:** 1.97 seconds
- **Total Queries per Pipeline:** 10
- **Iterations per Query:** 3

## Pipeline Performance Summary

| Pipeline | Success Rate | Avg Response Time | Avg Documents | RAGAS Score* |
|----------|--------------|-------------------|---------------|--------------|
| basic | 100.0% | 2.78s | 10.0 | N/A |
| hyde | 100.0% | 4.73s | 10.0 | N/A |
| crag | 100.0% | 2.74s | 10.0 | N/A |
| colbert | 100.0% | 0.70s | 10.0 | N/A |
| noderag | 100.0% | 0.77s | 10.0 | N/A |
| graphrag | 100.0% | 0.84s | 10.0 | N/A |
| hybrid_ifind | 100.0% | 1.24s | 10.0 | N/A |

*RAGAS Score is the average of available RAGAS metrics

## Performance Analysis

- **Most Reliable:** basic (100.0% success rate)
- **Fastest:** colbert (0.70s average)

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
