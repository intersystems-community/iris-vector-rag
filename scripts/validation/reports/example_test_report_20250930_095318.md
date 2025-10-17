# Example Test Report
**Generated**: 2025-09-30 09:53:18
**Total Examples**: 1
**Passed**: 0
**Failed**: 1
**Success Rate**: 0.0%
**Average Execution Time**: 5.41s
**Average Memory Usage**: 38.8MB

## Detailed Results

### basic/try_basic_rag_pipeline.py
**Status**: ❌ FAIL
**Execution Time**: 5.41s
**Memory Usage**: 38.8MB

**Error**: Script failed with exit code 1: Traceback (most recent call last):
  File "/Users/tdyar/ws/rag-templates/scripts/basic/try_basic_rag_pipeline.py", line 85, in <module>
    main()
  File "/Users/tdyar/ws/rag-templates/scripts/basic/try_basic_rag_pipeline.py", line 64, in main
    basic_rag_pipeline = iris_rag.create_pipeline(
                         ^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/tdyar/ws/rag-templates/iris_rag/__init__.py", line 69, in create_pipeline
    return factory.create_pipeline(
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/tdyar/ws/rag-templates/iris_rag/validation/factory.py", line 91, in create_pipeline
    validation_report = self._validate_and_setup(pipeline_type, auto_setup)
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/tdyar/ws/rag-templates/iris_rag/validation/factory.py", line 117, in _validate_and_setup
    validation_report = self.orchestrator.setup_pipeline(
                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/Users/tdyar/ws/rag-templates/iris_rag/validation/orchestrator.py", line 111, in setup_pipeline
    self._fulfill_requirements(requirements)
  File "/Users/tdyar/ws/rag-templates/iris_rag/validation/orchestrator.py", line 163, in _fulfill_requirements
    self._fulfill_embedding_requirement(embedding_req)
  File "/Users/tdyar/ws/rag-templates/iris_rag/validation/orchestrator.py", line 187, in _fulfill_embedding_requirement
    self._ensure_document_embeddings()
  File "/Users/tdyar/ws/rag-templates/iris_rag/validation/orchestrator.py", line 269, in _ensure_document_embeddings
    cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
iris.dbapi.ProgrammingError: <SQL ERROR>; Details: [SQLCODE: <-30>:<Table or view not found>]
[Location: <Prepare>]
[%msg: < Table 'RAG.SOURCEDOCUMENTS' not found>]

---

## Failure Analysis

### Import/Module Errors
**Count**: 1
**Scripts**: basic/try_basic_rag_pipeline.py
