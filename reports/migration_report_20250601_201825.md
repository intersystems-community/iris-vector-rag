# VECTOR(FLOAT) to VECTOR(FLOAT) Migration Report

## Migration Summary

- **Start Time**: 2025-06-01T20:17:31.411640
- **End Time**: 2025-06-01T20:18:25.033756
- **Duration**: 53.62 seconds
- **Database Tables Changed**: 0
- **SQL Files Changed**: 2
- **Python Files Changed**: 55
- **ObjectScript Files Changed**: 0
- **Backups Created**: 57
- **Errors**: 0
- **Warnings**: 0

## SQL File Changes

- **/Users/tdyar/ws/rag-templates/scripts/migration/test_vector_query.sql**: 1 changes
- **/Users/tdyar/ws/rag-templates/src/experimental/hybrid_ifind_rag/schema.sql**: 2 changes

## PYTHON File Changes

- **/Users/tdyar/ws/rag-templates/test_correct_vector_syntax.py**: 9 changes
- **/Users/tdyar/ws/rag-templates/test_simple_vector_functions.py**: 5 changes
- **/Users/tdyar/ws/rag-templates/test_direct_crag_sql.py**: 4 changes
- **/Users/tdyar/ws/rag-templates/basic_rag/pipeline_jdbc.py**: 2 changes
- **/Users/tdyar/ws/rag-templates/basic_rag/pipeline_refactored.py**: 8 changes
- **/Users/tdyar/ws/rag-templates/basic_rag/pipeline_direct_sql.py**: 2 changes
- **/Users/tdyar/ws/rag-templates/basic_rag/pipeline_stored_proc.py**: 4 changes
- **/Users/tdyar/ws/rag-templates/basic_rag/pipeline_chunked_vector.py**: 2 changes
- **/Users/tdyar/ws/rag-templates/basic_rag/pipeline_file_vector.py**: 2 changes
- **/Users/tdyar/ws/rag-templates/archive/debug_basicrag_fallback.py**: 4 changes
- **/Users/tdyar/ws/rag-templates/archive/debug_crag_sql.py**: 4 changes
- **/Users/tdyar/ws/rag-templates/jdbc_exploration/iris_jdbc_connector.py**: 3 changes
- **/Users/tdyar/ws/rag-templates/jdbc_exploration/quick_jdbc_test_fixed.py**: 3 changes
- **/Users/tdyar/ws/rag-templates/tests/test_hnsw_integration.py**: 3 changes
- **/Users/tdyar/ws/rag-templates/tests/test_hnsw_query_patterns.py**: 6 changes
- **/Users/tdyar/ws/rag-templates/tests/test_hnsw_performance.py**: 6 changes
- **/Users/tdyar/ws/rag-templates/tests/test_hnsw_indexes.py**: 2 changes
- **/Users/tdyar/ws/rag-templates/tests/test_hnsw_benchmark_integration.py**: 3 changes
- **/Users/tdyar/ws/rag-templates/docs/iris_vector_error_example.py**: 2 changes
- **/Users/tdyar/ws/rag-templates/chunking/direct_v2_chunking_service.py**: 1 changes
- **/Users/tdyar/ws/rag-templates/chunking/direct_chunking_final.py**: 1 changes
- **/Users/tdyar/ws/rag-templates/chunking/update_v2_vectors.py**: 1 changes
- **/Users/tdyar/ws/rag-templates/chunking/direct_v2_chunking_service_simple.py**: 1 changes
- **/Users/tdyar/ws/rag-templates/scripts/migrate_vector_double_to_float.py**: 2 changes
- **/Users/tdyar/ws/rag-templates/scripts/comprehensive_sql_cleanup_and_vector_implementation.py**: 5 changes
- **/Users/tdyar/ws/rag-templates/scripts/investigate_vector_indexing_reality.py**: 2 changes
- **/Users/tdyar/ws/rag-templates/scripts/fair_v2_performance_comparison.py**: 2 changes
- **/Users/tdyar/ws/rag-templates/bug_reproductions/bug4_stored_procedures.py**: 3 changes
- **/Users/tdyar/ws/rag-templates/bug_reproductions/bug2_hnsw_varchar.py**: 8 changes
- **/Users/tdyar/ws/rag-templates/bug_reproductions/bug3_vector_driver_support.py**: 3 changes
- **/Users/tdyar/ws/rag-templates/archive/performance_investigation/test_fresh_start.py**: 1 changes
- **/Users/tdyar/ws/rag-templates/archive/migration_backup_20250530_135241/test_refactored_debug.py**: 4 changes
- **/Users/tdyar/ws/rag-templates/archive/migration_backup_20250530_135241/migrate_document_chunks_v2_only.py**: 3 changes
- **/Users/tdyar/ws/rag-templates/archived_pipelines/basic_rag/pipeline_jdbc.py**: 2 changes
- **/Users/tdyar/ws/rag-templates/archived_pipelines/basic_rag/pipeline_refactored.py**: 8 changes
- **/Users/tdyar/ws/rag-templates/archived_pipelines/basic_rag/pipeline_direct_sql.py**: 2 changes
- **/Users/tdyar/ws/rag-templates/archived_pipelines/basic_rag/pipeline_stored_proc.py**: 4 changes
- **/Users/tdyar/ws/rag-templates/archived_pipelines/basic_rag/pipeline_chunked_vector.py**: 2 changes
- **/Users/tdyar/ws/rag-templates/archived_pipelines/basic_rag/pipeline_vector.py**: 2 changes
- **/Users/tdyar/ws/rag-templates/archived_pipelines/basic_rag/pipeline_file_vector.py**: 2 changes
- **/Users/tdyar/ws/rag-templates/archived_pipelines/crag/pipeline_v2.py**: 2 changes
- **/Users/tdyar/ws/rag-templates/scripts/testing/test_iris_vector_bug_dbapi.py**: 2 changes
- **/Users/tdyar/ws/rag-templates/scripts/testing/test_iris_vector_colon_bug.py**: 2 changes
- **/Users/tdyar/ws/rag-templates/scripts/testing/test_hnsw_syntax_systematic.py**: 8 changes
- **/Users/tdyar/ws/rag-templates/scripts/testing/test_iris_vector_bugs_minimal_demo.py**: 5 changes
- **/Users/tdyar/ws/rag-templates/scripts/migration/fix_v2_pipelines_sql.py**: 1 changes
- **/Users/tdyar/ws/rag-templates/scripts/migration/fix_all_v2_pipelines.py**: 2 changes
- **/Users/tdyar/ws/rag-templates/scripts/migration/fix_v2_pipelines_params.py**: 6 changes
- **/Users/tdyar/ws/rag-templates/src/deprecated/basic_rag/pipeline_jdbc.py**: 2 changes
- **/Users/tdyar/ws/rag-templates/src/deprecated/basic_rag/pipeline_refactored.py**: 8 changes
- **/Users/tdyar/ws/rag-templates/src/deprecated/basic_rag/pipeline_direct_sql.py**: 2 changes
- **/Users/tdyar/ws/rag-templates/src/deprecated/basic_rag/pipeline_stored_proc.py**: 4 changes
- **/Users/tdyar/ws/rag-templates/src/deprecated/basic_rag/pipeline_chunked_vector.py**: 2 changes
- **/Users/tdyar/ws/rag-templates/src/deprecated/basic_rag/pipeline_temp_table.py**: 2 changes
- **/Users/tdyar/ws/rag-templates/src/deprecated/basic_rag/pipeline_file_vector.py**: 2 changes

