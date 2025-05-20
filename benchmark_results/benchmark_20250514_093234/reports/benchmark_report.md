# RAG Techniques Benchmark Report

Generated on: 2025-05-14 09:33:02

## Benchmark Summary

The following RAG techniques were benchmarked:

- **basic_rag**
- **hyde**
- **colbert**
- **crag**
- **noderag**
- **graphrag**

## Best Performing Techniques

- **Best for Performance**: graphrag

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
| basic_rag | N/A | N/A | 1.02 q/s |
| hyde | N/A | N/A | 0.40 q/s |
| colbert | N/A | N/A | 0.88 q/s |
| crag | N/A | N/A | 1.07 q/s |
| noderag | N/A | N/A | 1.37 q/s |
| graphrag | N/A | N/A | 1.63 q/s |

## Visualizations

### Overall Comparison

![Radar Chart Comparison](radar_comparison.png)

### Throughput Qps Comparison

![Bar Chart - Throughput Qps](bar_throughput_qps.png)

## Conclusion

**graphrag** emerged as the overall best technique in our benchmarks, leading in 1 out of 3 categories. For specific use cases, consider the following recommendations:

- **For retrieval-critical applications**: No clear winner
- **For answer quality focus**: No clear winner
- **For performance-critical systems**: Use graphrag

Performance may vary with different datasets, configurations, and specific application requirements. These results should be used as guidelines for initial technique selection, with additional testing recommended for your specific use case.
