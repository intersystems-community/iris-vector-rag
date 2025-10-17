# GraphRAG Implementation Comparison Report

**Generated:** 2025-09-15T21:57:50.898091
**Test Suite:** GraphRAG Comprehensive Comparison

## Executive Summary

Test Suite Execution Summary:
        - Total Tests: 9
        - Current Implementation Success Rate: 100.00%
        - Merged Implementation Success Rate: 100.00%
        - Performance Change: +9.3%
        - Regressions Found: 2
        - Improvements Found: 8

## Performance Comparison

- **Average Execution Time:** Current: 336.4ms, Merged: 305.1ms
- **Performance Change:** +9.3%
- **Average Documents Retrieved:** Current: 1.9, Merged: 2.1
- **Average DB Executions:** Current: 5.9, Merged: 5.3

## Quality Comparison

- **Success Rates:** Current: 100.00%, Merged: 100.00%
- **Success Rate Change:** +0.00%
- **Average Answer Length:** Current: 118, Merged: 117

## Regressions Found

- basic_002: Significant performance regression (50%+ slower)
- multihop_001: Significant performance regression (50%+ slower)

## Improvements Found

- basic_001: Performance improvement (20%+ faster)
- multihop_001: Better document retrieval (2 vs 1)
- complex_001: Better document retrieval (4 vs 1)
- edge_001: Performance improvement (20%+ faster)
- edge_001: Better document retrieval (2 vs 1)
- edge_002: Performance improvement (20%+ faster)
- stress_001: Performance improvement (20%+ faster)
- stress_001: Better document retrieval (3 vs 1)

## Detailed Test Results

| Test ID | Current Success | Current Time (ms) | Merged Success | Merged Time (ms) | Performance Change |
|---------|----------------|-------------------|---------------|------------------|-------------------|
| basic_001 | True | 464.9768329691142 | True | 136.01929100695997 | +70.7% |
| basic_002 | True | 309.11725002806634 | True | 496.96062493603677 | -60.8% |
| multihop_001 | True | 177.67837503924966 | True | 435.12991699390113 | -144.9% |
| multihop_002 | True | 414.52225006651133 | True | 470.40029207710177 | -13.5% |
| complex_001 | True | 162.4971249839291 | True | 177.49570799060166 | -9.2% |
| complex_002 | True | 395.7027499563992 | True | 322.7248750627041 | +18.4% |
| edge_001 | True | 356.8153749220073 | True | 278.3822090132162 | +22.0% |
| edge_002 | True | 476.91233397927135 | True | 261.56337501015514 | +45.2% |
| stress_001 | True | 269.7667919564992 | True | 167.20120899844915 | +38.0% |
