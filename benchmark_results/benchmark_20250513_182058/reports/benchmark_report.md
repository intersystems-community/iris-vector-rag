# RAG Techniques Benchmark Report

Generated on: 2025-05-13 18:21:00

## Benchmark Summary

The following RAG techniques were benchmarked:

- **basic_rag**
- **hyde**
- **colbert**
- **crag**
- **noderag**
- **graphrag**

## Best Performing Techniques

- **Best for Performance**: hyde

## Key Metrics

### Retrieval Quality

| Technique | Context Recall | Precision At 5 | Precision At 10 |
| --- | --- | --- | --- |
| basic_rag | N/A | N/A | N/A |
| hyde | N/A | N/A | N/A |
| colbert | N/A | N/A | N/A |
| crag | N/A | N/A | N/A |
| noderag | N/A | N/A | N/A |
| graphrag | N/A | N/A | N/A |

### Answer Quality

| Technique | Answer Faithfulness | Answer Relevance |
| --- | --- | --- |
| basic_rag | N/A | N/A |
| hyde | N/A | N/A |
| colbert | N/A | N/A |
| crag | N/A | N/A |
| noderag | N/A | N/A |
| graphrag | N/A | N/A |

### Performance

| Technique | Latency P50 | Latency P95 | Throughput Qps |
| --- | --- | --- | --- |
| basic_rag | N/A | N/A | 654.36 q/s |
| hyde | N/A | N/A | 972.61 q/s |
| colbert | N/A | N/A | N/A |
| crag | N/A | N/A | 493.30 q/s |
| noderag | N/A | N/A | N/A |
| graphrag | N/A | N/A | N/A |

## Comparison to Published Benchmarks

Our techniques were compared against published benchmarks for multihop datasets:

| Technique | Answer F1 | Supporting Facts F1 | Joint F1 |
| --- | --- | --- | --- |
| basic_rag | 0.000 | 0.000 | 0.000 |
| hyde | 0.000 | 0.000 | 0.000 |
| colbert | 0.000 | 0.000 | 0.000 |
| crag | 0.000 | 0.000 | 0.000 |
| noderag | 0.000 | 0.000 | 0.000 |
| graphrag | 0.000 | 0.000 | 0.000 |
| Ref: GraphRAG | 0.796 | 0.849 | 0.703 |
| Ref: ColBERT | 0.687 | 0.728 | 0.578 |
| Ref: Basic Dense Retrieval | 0.631 | 0.667 | 0.492 |

## Visualizations

### Overall Comparison

![Radar Chart Comparison](radar_comparison.png)

### Throughput Qps Comparison

![Bar Chart - Throughput Qps](bar_throughput_qps.png)

### Answer F1 vs Published Benchmarks

![Comparison - Answer F1](comparison_answer_f1.png)

### Supporting Facts F1 vs Published Benchmarks

![Comparison - Supporting Facts F1](comparison_supporting_facts_f1.png)

### Joint F1 vs Published Benchmarks

![Comparison - Joint F1](comparison_joint_f1.png)

## Conclusion

**hyde** emerged as the overall best technique in our benchmarks, leading in 1 out of 3 categories. For specific use cases, consider the following recommendations:

- **For retrieval-critical applications**: No clear winner
- **For answer quality focus**: No clear winner
- **For performance-critical systems**: Use hyde

Performance may vary with different datasets, configurations, and specific application requirements. These results should be used as guidelines for initial technique selection, with additional testing recommended for your specific use case.
