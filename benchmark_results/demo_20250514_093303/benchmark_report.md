# RAG Techniques Benchmark Report

Generated on: 2025-05-14 09:33:03

## Benchmark Summary

The following RAG techniques were benchmarked:

- **basic_rag**
- **hyde**
- **colbert**

## Best Performing Techniques

- **Best for Retrieval Quality**: colbert
- **Best for Answer Quality**: colbert
- **Best for Performance**: basic_rag

## Key Metrics

### Retrieval Quality

| Technique | Context Recall | Precision At 5 | Precision At 10 |
| --- | --- | --- | --- |
| basic_rag | 0.720 | N/A | N/A |
| hyde | 0.780 | N/A | N/A |
| colbert | 0.850 | N/A | N/A |

### Answer Quality

| Technique | Answer Faithfulness | Answer Relevance |
| --- | --- | --- |
| basic_rag | 0.800 | 0.750 |
| hyde | 0.820 | 0.800 |
| colbert | 0.870 | 0.830 |

### Performance

| Technique | Latency P50 | Latency P95 | Throughput Qps |
| --- | --- | --- | --- |
| basic_rag | 105.00 ms | 150.00 ms | 18.50 q/s |
| hyde | 120.00 ms | 180.00 ms | 15.20 q/s |
| colbert | 150.00 ms | 220.00 ms | 12.80 q/s |

## Comparison to Published Benchmarks

Our techniques were compared against published benchmarks for multihop datasets:

| Technique | Answer F1 | Supporting Facts F1 | Joint F1 |
| --- | --- | --- | --- |
| basic_rag | 0.000 | 0.000 | 0.000 |
| hyde | 0.000 | 0.000 | 0.000 |
| colbert | 0.000 | 0.000 | 0.000 |
| Ref: GraphRAG | 0.796 | 0.849 | 0.703 |
| Ref: ColBERT | 0.687 | 0.728 | 0.578 |
| Ref: Basic Dense Retrieval | 0.631 | 0.667 | 0.492 |

## Visualizations

### Overall Comparison

![Radar Chart Comparison](radar_comparison.png)

### Context Recall Comparison

![Bar Chart - Context Recall](bar_context_recall.png)

### Answer Faithfulness Comparison

![Bar Chart - Answer Faithfulness](bar_answer_faithfulness.png)

### Answer Relevance Comparison

![Bar Chart - Answer Relevance](bar_answer_relevance.png)

### Throughput Qps Comparison

![Bar Chart - Throughput Qps](bar_throughput_qps.png)

### Latency P50 Comparison

![Bar Chart - Latency P50](bar_latency_p50.png)

### Latency P95 Comparison

![Bar Chart - Latency P95](bar_latency_p95.png)

### Answer F1 vs Published Benchmarks

![Comparison - Answer F1](comparison_answer_f1.png)

### Supporting Facts F1 vs Published Benchmarks

![Comparison - Supporting Facts F1](comparison_supporting_facts_f1.png)

### Joint F1 vs Published Benchmarks

![Comparison - Joint F1](comparison_joint_f1.png)

## Conclusion

**colbert** emerged as the overall best technique in our benchmarks, leading in 2 out of 3 categories. For specific use cases, consider the following recommendations:

- **For retrieval-critical applications**: Use colbert
- **For answer quality focus**: Use colbert
- **For performance-critical systems**: Use basic_rag

Performance may vary with different datasets, configurations, and specific application requirements. These results should be used as guidelines for initial technique selection, with additional testing recommended for your specific use case.
