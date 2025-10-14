# GraphRAG Implementation Comparison Report

**Generated:** 2025-09-15T22:03:25.515883
**Test Suite:** GraphRAG Comprehensive Comparison

## Executive Summary

Test Suite Execution Summary:
        - Total Tests: 9
        - Current Implementation Success Rate: 77.78%
        - Merged Implementation Success Rate: 11.11%
        - Performance Change: +97.1%
        - Regressions Found: 6
        - Improvements Found: 1

## Performance Comparison

- **Average Execution Time:** Current: 194.2ms, Merged: 5.7ms
- **Performance Change:** +97.1%
- **Average Documents Retrieved:** Current: 4.1, Merged: 5.0
- **Average DB Executions:** Current: 4.0, Merged: 5.0

## Quality Comparison

- **Success Rates:** Current: 77.78%, Merged: 11.11%
- **Success Rate Change:** -66.67%
- **Average Answer Length:** Current: 51, Merged: 51

## Regressions Found

- basic_001: Merged implementation failed where current succeeded
- basic_002: Merged implementation failed where current succeeded
- complex_001: Merged implementation failed where current succeeded
- complex_002: Merged implementation failed where current succeeded
- edge_002: Merged implementation failed where current succeeded
- stress_001: Merged implementation failed where current succeeded

## Improvements Found

- edge_001: Performance improvement (20%+ faster)

## Detailed Test Results

| Test ID | Current Success | Current Time (ms) | Merged Success | Merged Time (ms) | Performance Change |
|---------|----------------|-------------------|---------------|------------------|-------------------|
| basic_001 | True | 254.27262496668845 | False | FAIL | N/A |
| basic_002 | True | 256.7834999645129 | False | FAIL | N/A |
| multihop_001 | False | FAIL | False | FAIL | N/A |
| multihop_002 | False | FAIL | False | FAIL | N/A |
| complex_001 | True | 309.16270904708654 | False | FAIL | N/A |
| complex_002 | True | 83.63454206846654 | False | FAIL | N/A |
| edge_001 | True | 201.5230000251904 | True | 5.663083051331341 | +97.2% |
| edge_002 | True | 89.50391691178083 | False | FAIL | N/A |
| stress_001 | True | 164.69558305107057 | False | FAIL | N/A |
