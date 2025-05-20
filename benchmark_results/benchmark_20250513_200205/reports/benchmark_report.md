# RAG Techniques Benchmark Report

Generated on: 2025-05-13 20:02:21

## Benchmark Summary

The following RAG techniques were benchmarked:

- **basic_rag**
- **crag**

## Best Performing Techniques

- **Best for Performance**: crag

## Key Metrics

### Retrieval Quality

| Technique | Context Recall | Precision At 5 | Precision At 10 |
| --- | --- | --- | --- |
| basic_rag | N/A | N/A | N/A |
| crag | N/A | N/A | N/A |

### Answer Quality

| Technique | Answer Faithfulness | Answer Relevance |
| --- | --- | --- |
| basic_rag | N/A | N/A |
| crag | N/A | N/A |

### Performance

| Technique | Latency P50 | Latency P95 | Throughput Qps |
| --- | --- | --- | --- |
| basic_rag | N/A | N/A | 0.18 q/s |
| crag | N/A | N/A | 1.09 q/s |

## Visualizations

### Overall Comparison

![Radar Chart Comparison](radar_comparison.png)

### Throughput Qps Comparison

![Bar Chart - Throughput Qps](bar_throughput_qps.png)

## Conclusion

**crag** emerged as the overall best technique in our benchmarks, leading in 1 out of 3 categories. For specific use cases, consider the following recommendations:

- **For retrieval-critical applications**: No clear winner
- **For answer quality focus**: No clear winner
- **For performance-critical systems**: Use crag

Performance may vary with different datasets, configurations, and specific application requirements. These results should be used as guidelines for initial technique selection, with additional testing recommended for your specific use case.
