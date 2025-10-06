# Example Test Report
**Generated**: 2025-09-30 13:51:15
**Total Examples**: 6
**Passed**: 1
**Failed**: 5
**Success Rate**: 16.7%
**Average Execution Time**: 2.81s
**Average Memory Usage**: 39.0MB

## Detailed Results

### basic/try_basic_rag_pipeline.py
**Status**: ❌ FAIL
**Execution Time**: 4.93s
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

### basic/try_hybrid_graphrag_pipeline.py
**Status**: ✅ PASS
**Execution Time**: 3.51s
**Memory Usage**: 39.0MB

**Validation Score**: 0.80
**Validation Issues**: 1

---

### crag/try_crag_pipeline.py
**Status**: ❌ FAIL
**Execution Time**: 4.10s
**Memory Usage**: 39.0MB

**Error**: Script failed with exit code 1: Traceback (most recent call last):
  File "/Users/tdyar/ws/rag-templates/scripts/crag/try_crag_pipeline.py", line 85, in <module>
    main()
  File "/Users/tdyar/ws/rag-templates/scripts/crag/try_crag_pipeline.py", line 64, in main
    crag_pipeline = iris_rag.create_pipeline(
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
  File "/Users/tdyar/ws/rag-templates/iris_rag/validation/orchestrator.py", line 113, in setup_pipeline
    self._setup_crag_pipeline(requirements)
  File "/Users/tdyar/ws/rag-templates/iris_rag/validation/orchestrator.py", line 256, in _setup_crag_pipeline
    self._ensure_document_embeddings()
  File "/Users/tdyar/ws/rag-templates/iris_rag/validation/orchestrator.py", line 269, in _ensure_document_embeddings
    cursor.execute("SELECT COUNT(*) FROM RAG.SourceDocuments")
iris.dbapi.ProgrammingError: <SQL ERROR>; Details: [SQLCODE: <-30>:<Table or view not found>]
[Location: <Prepare>]
[%msg: < Table 'RAG.SOURCEDOCUMENTS' not found>]

---

### demo_graph_visualization.py
**Status**: ❌ FAIL
**Execution Time**: 0.13s
**Memory Usage**: 39.0MB

**Error**: Script failed with exit code 1: Traceback (most recent call last):
  File "/Users/tdyar/ws/rag-templates/scripts/demo_graph_visualization.py", line 19, in <module>
    from iris_rag.pipelines.graphrag_merged import GraphRAGPipeline
  File "/Users/tdyar/ws/rag-templates/iris_rag/pipelines/graphrag_merged.py", line 24, in <module>
    from ..visualization.graph_visualizer import GraphVisualizer, GraphVisualizationException
  File "/Users/tdyar/ws/rag-templates/iris_rag/visualization/__init__.py", line 8, in <module>
    from .graph_visualizer import GraphVisualizer
  File "/Users/tdyar/ws/rag-templates/iris_rag/visualization/graph_visualizer.py", line 225
    raise ImportError(\"Plotly not installed\")
                       ^
SyntaxError: unexpected character after line continuation character

---

### demo_ontology_support.py
**Status**: ❌ FAIL
**Execution Time**: 0.08s
**Memory Usage**: 39.1MB

**Error**: Script failed with exit code 1: Traceback (most recent call last):
  File "/Users/tdyar/ws/rag-templates/scripts/demo_ontology_support.py", line 32, in <module>
    from iris_rag.pipelines.graphrag_merged import GraphRAGPipeline
  File "/Users/tdyar/ws/rag-templates/iris_rag/pipelines/graphrag_merged.py", line 24, in <module>
    from ..visualization.graph_visualizer import GraphVisualizer, GraphVisualizationException
  File "/Users/tdyar/ws/rag-templates/iris_rag/visualization/__init__.py", line 8, in <module>
    from .graph_visualizer import GraphVisualizer
  File "/Users/tdyar/ws/rag-templates/iris_rag/visualization/graph_visualizer.py", line 225
    raise ImportError(\"Plotly not installed\")
                       ^
SyntaxError: unexpected character after line continuation character

---

### reranking/try_basic_rerank.py
**Status**: ❌ FAIL
**Execution Time**: 4.13s
**Memory Usage**: 39.1MB

**Error**: Script failed with exit code 1: Traceback (most recent call last):
  File "/Users/tdyar/ws/rag-templates/scripts/reranking/try_basic_rerank.py", line 88, in <module>
    main()
  File "/Users/tdyar/ws/rag-templates/scripts/reranking/try_basic_rerank.py", line 67, in main
    reranking_rag_pipeline = iris_rag.create_pipeline(
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
**Count**: 5
**Scripts**: basic/try_basic_rag_pipeline.py, crag/try_crag_pipeline.py, demo_graph_visualization.py, demo_ontology_support.py, reranking/try_basic_rerank.py
