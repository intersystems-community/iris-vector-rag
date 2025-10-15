# Direct DBAPI RAG Validation Report

**Test Date:** 2025-06-05T06:34:19.962175
**Total Execution Time:** 7.87 seconds

## Test Summary

- **Total Techniques:** 7
- **Successful:** 0
- **Failed:** 7
- **Success Rate:** 0.0%

## DBAPI Compatibility

- **Connection Method:** `intersystems_iris.dbapi._DBAPI.connect()`
- **Connection Status:** ✓ SUCCESS
- **Basic Functionality:** ✓ WORKING

## Individual Technique Results

### ✗ BasicRAG

- **Status:** FAILED
- **Execution Time:** 0.29 seconds
- **Retrieved Documents:** 0
- **Error:** BasicRAGPipeline.__init__() missing 3 required positional arguments: 'iris_connector', 'embedding_func', and 'llm_func'

### ✗ ColBERT

- **Status:** FAILED
- **Execution Time:** 0.00 seconds
- **Retrieved Documents:** 0
- **Error:** module 'core_pipelines.colbert_pipeline' has no attribute 'ColBERTPipeline'

### ✗ CRAG

- **Status:** FAILED
- **Execution Time:** 0.00 seconds
- **Retrieved Documents:** 0
- **Error:** CRAGPipeline.__init__() missing 3 required positional arguments: 'iris_connector', 'embedding_func', and 'llm_func'

### ✗ GraphRAG

- **Status:** FAILED
- **Execution Time:** 0.00 seconds
- **Retrieved Documents:** 0
- **Error:** GraphRAGPipeline.__init__() missing 3 required positional arguments: 'iris_connector', 'embedding_func', and 'llm_func'

### ✗ HyDE

- **Status:** FAILED
- **Execution Time:** 0.00 seconds
- **Retrieved Documents:** 0
- **Error:** HyDEPipeline.__init__() missing 3 required positional arguments: 'iris_connector', 'embedding_func', and 'llm_func'

### ✗ NodeRAG

- **Status:** FAILED
- **Execution Time:** 0.00 seconds
- **Retrieved Documents:** 0
- **Error:** NodeRAGPipeline.__init__() missing 3 required positional arguments: 'iris_connector', 'embedding_func', and 'llm_func'

### ✗ HybridIFindRAG

- **Status:** FAILED
- **Execution Time:** 0.00 seconds
- **Retrieved Documents:** 0
- **Error:** module 'hybrid_ifind_rag.pipeline' has no attribute 'HybridIFindRAGPipeline'

## Recommendations

- ⚠️ 7 techniques need attention: BasicRAG, ColBERT, CRAG, GraphRAG, HyDE, NodeRAG, HybridIFindRAG
- 🔧 Review failed techniques for DBAPI compatibility issues
