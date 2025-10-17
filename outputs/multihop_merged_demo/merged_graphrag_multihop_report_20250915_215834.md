# Merged GraphRAG Multi-Hop Query Test Report

**Generated:** 20250915_215834
**Environment:** mock_data
**Total Queries:** 8

## Executive Summary

- **Current Implementation Success Rate:** 100.00%
- **Merged Implementation Success Rate:** 100.00%
- **Average Execution Time Change:** -12.0%
- **Average Confidence Change:** +0.10

## Performance Comparison

| Metric | Current | Merged | Change |
|--------|---------|--------|---------|
| Success Rate | 100.00% | 100.00% | +0.00% |
| Avg Execution Time (ms) | 487.3 | 545.6 | -12.0% |
| Avg Confidence | 0.79 | 0.88 | +0.10 |
| Avg Documents Retrieved | 4.1 | 3.0 | -1.1 |

## Recommendations

- ✅ Merged implementation maintains or improves success rate
- ⚖️ Performance impact acceptable
- 🎯 Improved confidence scores in merged implementation
- 🚀 RECOMMENDED: Deploy merged implementation

## Detailed Query Results

### Query: What are the molecular mechanisms linking diabetes to cardiovascular complications?...

**Type:** 2-hop-molecular-pathway
**Implementation:** current
**Success:** True
**Execution Time:** 508.6ms
**Documents Retrieved:** 5

### Query: How do SGLT-2 inhibitors provide renoprotective effects in diabetic nephropathy?...

**Type:** 2-hop-drug-mechanism
**Implementation:** current
**Success:** True
**Execution Time:** 290.1ms
**Documents Retrieved:** 5

### Query: What is the relationship between insulin resistance and adipokine dysfunction?...

**Type:** 2-hop-pathophysiology
**Implementation:** current
**Success:** True
**Execution Time:** 258.4ms
**Documents Retrieved:** 2

### Query: How does COVID-19 infection lead to new-onset diabetes through ACE2 receptor mechanisms?...

**Type:** 3-hop-viral-pathogenesis
**Implementation:** current
**Success:** True
**Execution Time:** 645.1ms
**Documents Retrieved:** 2

### Query: What are the shared molecular pathways between metformin action and cardiovascular protection?...

**Type:** 3-hop-drug-cardioprotection
**Implementation:** current
**Success:** True
**Execution Time:** 677.7ms
**Documents Retrieved:** 4

### Query: How do advanced glycation end products contribute to both nephropathy and cardiovascular disease?...

**Type:** 3-hop-AGE-complications
**Implementation:** current
**Success:** True
**Execution Time:** 682.6ms
**Documents Retrieved:** 5

### Query: What are the interconnected pathways linking obesity, insulin resistance, diabetes, and cardiovascul...

**Type:** complex-metabolic-network
**Implementation:** current
**Success:** True
**Execution Time:** 386.5ms
**Documents Retrieved:** 5

### Query: How do different diabetes medications interact with COVID-19 treatment and cardiovascular risk manag...

**Type:** complex-drug-interaction-network
**Implementation:** current
**Success:** True
**Execution Time:** 449.4ms
**Documents Retrieved:** 5

