# Example Test Report
**Generated**: 2025-09-30 09:55:31
**Total Examples**: 3
**Passed**: 1
**Failed**: 2
**Success Rate**: 33.3%
**Average Execution Time**: 4.17s
**Average Memory Usage**: 39.4MB

## Detailed Results

### basic/try_basic_rag_pipeline.py
**Status**: ❌ FAIL
**Execution Time**: 4.36s
**Memory Usage**: 39.3MB

**Error**: Script failed with exit code 1: Traceback (most recent call last):
  File "/Users/intersystems-community/ws/rag-templates/scripts/basic/try_basic_rag_pipeline.py", line 85, in <module>
    main()
  File "/Users/intersystems-community/ws/rag-templates/scripts/basic/try_basic_rag_pipeline.py", line 64, in main
    basic_rag_pipeline = iris_rag.create_pipeline(
                         ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/intersystems-community/ws/rag-templates/iris_rag/__init__.py", line 69, in create_pipeline
    return factory.create_pipeline(
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/intersystems-community/ws/rag-templates/iris_rag/validation/factory.py", line 91, in create_pipeline
    validation_report = self._validate_and_setup(pipeline_type, auto_setup)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/intersystems-community/ws/rag-templates/iris_rag/validation/factory.py", line 117, in _validate_and_setup
    validation_report = self.orchestrator.setup_pipeline(
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/intersystems-community/ws/rag-templates/iris_rag/validation/orchestrator.py", line 111, in setup_pipeline
    self._fulfill_requirements(requirements)
  File "/Users/intersystems-community/ws/rag-templates/iris_rag/validation/orchestrator.py", line 163, in _fulfill_requirements
    self._fulfill_embedding_requirement(embedding_req)
  File "/Users/intersystems-community/ws/rag-templates/iris_rag/validation/orchestrator.py", line 187, in _fulfill_embedding_requirement
    self._ensure_document_embeddings()
  File "/Users/intersystems-community/ws/rag-templates/iris_rag/validation/orchestrator.py", line 269, in _ensure_document_embeddings
    cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
iris.dbapi.ProgrammingError: <SQL ERROR>; Details: [SQLCODE: <-30>:<Table or view not found>]
[Location: <Prepare>]
[%msg: < Table 'RAG.SOURCEDOCUMENTS' not found>]

---

### basic/try_hybrid_graphrag_pipeline.py
**Status**: ✅ PASS
**Execution Time**: 3.56s
**Memory Usage**: 39.4MB

**Validation Score**: 0.10
**Validation Issues**: 3

---

### reranking/try_basic_rerank.py
**Status**: ❌ FAIL
**Execution Time**: 4.59s
**Memory Usage**: 39.4MB

**Error**: Script failed with exit code 1: Traceback (most recent call last):
  File "/Users/intersystems-community/ws/rag-templates/scripts/reranking/try_basic_rerank.py", line 88, in <module>
    main()
  File "/Users/intersystems-community/ws/rag-templates/scripts/reranking/try_basic_rerank.py", line 67, in main
    reranking_rag_pipeline = iris_rag.create_pipeline(
                             ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/intersystems-community/ws/rag-templates/iris_rag/__init__.py", line 69, in create_pipeline
    return factory.create_pipeline(
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/intersystems-community/ws/rag-templates/iris_rag/validation/factory.py", line 91, in create_pipeline
    validation_report = self._validate_and_setup(pipeline_type, auto_setup)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/intersystems-community/ws/rag-templates/iris_rag/validation/factory.py", line 117, in _validate_and_setup
    validation_report = self.orchestrator.setup_pipeline(
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/intersystems-community/ws/rag-templates/iris_rag/validation/orchestrator.py", line 111, in setup_pipeline
    self._fulfill_requirements(requirements)
  File "/Users/intersystems-community/ws/rag-templates/iris_rag/validation/orchestrator.py", line 163, in _fulfill_requirements
    self._fulfill_embedding_requirement(embedding_req)
  File "/Users/intersystems-community/ws/rag-templates/iris_rag/validation/orchestrator.py", line 187, in _fulfill_embedding_requirement
    self._ensure_document_embeddings()
  File "/Users/intersystems-community/ws/rag-templates/iris_rag/validation/orchestrator.py", line 269, in _ensure_document_embeddings
    cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
iris.dbapi.ProgrammingError: <SQL ERROR>; Details: [SQLCODE: <-30>:<Table or view not found>]
[Location: <Prepare>]
[%msg: < Table 'RAG.SOURCEDOCUMENTS' not found>]

---

## Failure Analysis

### Import/Module Errors
**Count**: 2
**Scripts**: basic/try_basic_rag_pipeline.py, reranking/try_basic_rerank.py
