# Deprecated Files Manifest

This document lists files that have been moved to the `deprecated/` directory as they are considered outdated, superseded by newer implementations, or are no longer part of the active development focus.

**Date of Last Update**: 2025-06-11

| Original Path | New Path in `deprecated/` | Reason for Deprecation | Date Moved | Notes |
|---------------|---------------------------|------------------------|------------|-------|
| src/deprecated/hybrid_ifind_rag/pipeline_v2.py | deprecated/src_archive/hybrid_ifind_rag/pipeline_v2.py | Superseded by current pipeline implementations in iris_rag/pipelines/. | 2025-06-11 | Originally in src/deprecated subfolder. |
| hybrid_ifind_rag/pipeline_v2.py | deprecated/hybrid_ifind_rag/pipeline_v2.py | Superseded by current pipeline implementations in iris_rag/pipelines/. | 2025-06-11 | |
| archived_pipelines/graphrag/pipeline_v2.py | deprecated/archived_pipelines/graphrag/pipeline_v2.py | Superseded by current pipeline implementations in iris_rag/pipelines/. | 2025-06-11 | |
| archived_pipelines/colbert/pipeline_v2.py | deprecated/archived_pipelines/colbert/pipeline_v2.py | Superseded by current pipeline implementations in iris_rag/pipelines/. | 2025-06-11 | |
| archived_pipelines/noderag/pipeline_v2.py | deprecated/archived_pipelines/noderag/pipeline_v2.py | Superseded by current pipeline implementations in iris_rag/pipelines/. | 2025-06-11 | |
| archived_pipelines/crag/pipeline_v2.py | deprecated/archived_pipelines/crag/pipeline_v2.py | Superseded by current pipeline implementations in iris_rag/pipelines/. | 2025-06-11 | |
| archived_pipelines/hyde/pipeline_v2.py | deprecated/archived_pipelines/hyde/pipeline_v2.py | Superseded by current pipeline implementations in iris_rag/pipelines/. | 2025-06-11 | |
| data/loader.py | deprecated/data/loader.py | Superseded by optimized data loaders (e.g., loader_optimized_performance.py) and current evaluation framework. | 2025-06-11 | |
| tests/test_loader.py | deprecated/tests/test_loader.py | Superseded by optimized data loaders (e.g., loader_optimized_performance.py) and current evaluation framework. | 2025-06-11 | |
| tests/test_data_loader.py | deprecated/tests/test_data_loader.py | Superseded by optimized data loaders (e.g., loader_optimized_performance.py) and current evaluation framework. | 2025-06-11 | |
| eval/loader.py | deprecated/eval/loader.py | Superseded by optimized data loaders (e.g., loader_optimized_performance.py) and current evaluation framework. | 2025-06-11 | |
| scripts/migration/fix_scientific_notation.py | deprecated/scripts/migration/fix_scientific_notation.py | One-time migration scripts for superseded V2 pipelines. | 2025-06-11 | |
| scripts/migration/fix_v2_pipelines_sql.py | deprecated/scripts/migration/fix_v2_pipelines_sql.py | One-time migration scripts for superseded V2 pipelines. | 2025-06-11 | |
| scripts/migration/fix_v2_pipelines_params.py | deprecated/scripts/migration/fix_v2_pipelines_params.py | One-time migration scripts for superseded V2 pipelines. | 2025-06-11 | |
| scripts/migration/fix_all_v2_pipelines.py | deprecated/scripts/migration/fix_all_v2_pipelines.py | One-time migration scripts for superseded V2 pipelines. | 2025-06-11 | |