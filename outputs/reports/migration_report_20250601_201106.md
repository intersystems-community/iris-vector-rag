# VECTOR(DOUBLE) to VECTOR(FLOAT) Migration Report

## Migration Summary

- **Start Time**: 2025-06-01T20:10:58.518957
- **End Time**: 2025-06-01T20:11:06.817231
- **Duration**: 8.30 seconds
- **Database Tables Changed**: 0
- **SQL Files Changed**: 9
- **Python Files Changed**: 43
- **ObjectScript Files Changed**: 0
- **Backups Created**: 52
- **Errors**: 0
- **Warnings**: 0

## SQL File Changes

- **/Users/tdyar/ws/rag-templates/chunking/chunking_schema.sql**: 1 changes
- **/Users/tdyar/ws/rag-templates/chunking/schema_clean.sql**: 1 changes
- **/Users/tdyar/ws/rag-templates/common/db_init_simple.sql**: 1 changes
- **/Users/tdyar/ws/rag-templates/common/db_init_complete.sql**: 5 changes
- **/Users/tdyar/ws/rag-templates/archive/test_files/test_schema.sql**: 1 changes
- **/Users/tdyar/ws/rag-templates/scripts/migration/test_iris_vector_bug_pure_sql.sql**: 1 changes
- **/Users/tdyar/ws/rag-templates/scripts/migration/iris_vector_bug_minimal.sql**: 1 changes
- **/Users/tdyar/ws/rag-templates/scripts/migration/iris_vector_bug_test.sql**: 2 changes
- **/Users/tdyar/ws/rag-templates/scripts/migration/test_iris_vector_bugs_minimal.sql**: 1 changes

## PYTHON File Changes

- **/Users/tdyar/ws/rag-templates/basic_rag/pipeline_temp_table.py**: 1 changes
- **/Users/tdyar/ws/rag-templates/archive/drop_and_repopulate_entities.py**: 2 changes
- **/Users/tdyar/ws/rag-templates/archive/repopulate_graphrag_with_vectors.py**: 1 changes
- **/Users/tdyar/ws/rag-templates/archive/migrate_graphrag_to_vector.py**: 2 changes
- **/Users/tdyar/ws/rag-templates/archive/create_graphrag_vector_tables.py**: 4 changes
- **/Users/tdyar/ws/rag-templates/scripts/vector_search_community_vs_licensed_comparison.py**: 3 changes
- **/Users/tdyar/ws/rag-templates/scripts/migrate_vector_double_to_float.py**: 12 changes
- **/Users/tdyar/ws/rag-templates/scripts/migrate_sourcedocuments_native_vector.py**: 2 changes
- **/Users/tdyar/ws/rag-templates/scripts/force_native_vector_schema.py**: 8 changes
- **/Users/tdyar/ws/rag-templates/scripts/comprehensive_hnsw_vs_nonhnsw_5000_validation.py**: 4 changes
- **/Users/tdyar/ws/rag-templates/scripts/fix_document_chunks_table.py**: 1 changes
- **/Users/tdyar/ws/rag-templates/scripts/test_correct_vector_syntax.py**: 2 changes
- **/Users/tdyar/ws/rag-templates/scripts/deploy_and_test_iris_2025_vector_search.py**: 4 changes
- **/Users/tdyar/ws/rag-templates/scripts/schema_migration_vector_and_chunking.py**: 4 changes
- **/Users/tdyar/ws/rag-templates/scripts/corrected_vector_migration.py**: 12 changes
- **/Users/tdyar/ws/rag-templates/scripts/fix_critical_schema_and_hnsw_issues.py**: 2 changes
- **/Users/tdyar/ws/rag-templates/scripts/test_iris_2025_vector_search.py**: 4 changes
- **/Users/tdyar/ws/rag-templates/scripts/fix_hnsw_infrastructure_complete.py**: 3 changes
- **/Users/tdyar/ws/rag-templates/scripts/corrected_iris_connection_test.py**: 2 changes
- **/Users/tdyar/ws/rag-templates/scripts/test_vector_schema_step1.py**: 6 changes
- **/Users/tdyar/ws/rag-templates/scripts/fix_vector_columns_urgent.py**: 24 changes
- **/Users/tdyar/ws/rag-templates/scripts/test_vector_syntax.py**: 1 changes
- **/Users/tdyar/ws/rag-templates/scripts/test_schema_locally.py**: 1 changes
- **/Users/tdyar/ws/rag-templates/scripts/fresh_1000_doc_setup_and_validation.py**: 4 changes
- **/Users/tdyar/ws/rag-templates/scripts/simple_schema_test.py**: 1 changes
- **/Users/tdyar/ws/rag-templates/scripts/complete_rag_system_fix.py**: 4 changes
- **/Users/tdyar/ws/rag-templates/scripts/fix_hnsw_and_vector_issues.py**: 1 changes
- **/Users/tdyar/ws/rag-templates/archive/test_files/test_hnsw_direct_pytest.py**: 1 changes
- **/Users/tdyar/ws/rag-templates/archive/test_files/test_hnsw_with_existing_connector.py**: 1 changes
- **/Users/tdyar/ws/rag-templates/archive/test_files/test_hnsw_direct_fixed.py**: 1 changes
- **/Users/tdyar/ws/rag-templates/archive/test_files/test_hnsw_direct.py**: 1 changes
- **/Users/tdyar/ws/rag-templates/archive/migration_backup_20250530_135241/force_sourcedocuments_migration.py**: 1 changes
- **/Users/tdyar/ws/rag-templates/archived_pipelines/basic_rag/pipeline_temp_table.py**: 1 changes
- **/Users/tdyar/ws/rag-templates/scripts/ingestion/create_knowledge_graph_schema.py**: 2 changes
- **/Users/tdyar/ws/rag-templates/scripts/testing/test_iris_vector_workaround.py**: 1 changes
- **/Users/tdyar/ws/rag-templates/scripts/testing/test_parameter_binding_approach.py**: 1 changes
- **/Users/tdyar/ws/rag-templates/scripts/testing/test_option3_corrected_vector_syntax.py**: 2 changes
- **/Users/tdyar/ws/rag-templates/scripts/testing/test_working_vector_solution.py**: 2 changes
- **/Users/tdyar/ws/rag-templates/scripts/testing/test_option3_hnsw_vector_declaration.py**: 1 changes
- **/Users/tdyar/ws/rag-templates/scripts/testing/test_vector_column_type_diagnosis.py**: 2 changes
- **/Users/tdyar/ws/rag-templates/scripts/migration/create_ragtest_schema.py**: 2 changes
- **/Users/tdyar/ws/rag-templates/scripts/validation/fast_hnsw_validation.py**: 1 changes
- **/Users/tdyar/ws/rag-templates/scripts/validation/fast_hnsw_validation_fixed.py**: 1 changes

