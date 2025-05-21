# Real Data Test Results

*Generated on: 2025-05-21 14:02:53*

## Test Environment

- **Database**: InterSystems IRIS
- **Document Count**: 1000+ real PMC documents
- **LLM**: OpenAI API (gpt-3.5-turbo)

## End-to-End Test Results

❌ **End-to-end tests failed.**

Output directory: `/Users/tdyar/ws/rag-templates/test_results/real_data_20250521_140150`

## Benchmark Results

❌ **Benchmarks failed.**

Output directory: `/Users/tdyar/ws/rag-templates/benchmark_results/real_data_20250521_140252`

## Comparative Analysis

A detailed comparative analysis of the different RAG techniques is available in the benchmark results directory.

## Issues and Recommendations

### Issues Encountered

1. **TO_VECTOR Function Limitation**: The IRIS SQL TO_VECTOR function does not accept parameter markers, which required implementing a string interpolation workaround.
2. **Vector Search Performance**: Vector search operations in IRIS SQL can be slow with large document sets.

### Recommendations

1. **Use String Interpolation**: When working with vector operations in IRIS SQL, use string interpolation with proper validation instead of parameter markers.
2. **Optimize Vector Search**: Consider implementing indexes or other optimizations to improve vector search performance.
3. **Batch Processing**: Process documents in smaller batches to avoid memory issues and improve performance.

## Conclusion

The attempt to run end-to-end tests and benchmarks with real PMC data and a real LLM encountered significant technical challenges. The primary issue, as detailed in [`docs/IRIS_SQL_VECTOR_LIMITATIONS.md`](docs/IRIS_SQL_VECTOR_LIMITATIONS.md:1), is the inability to load documents with vector embeddings into the database due to limitations with the `TO_VECTOR()` SQL function and ODBC driver behavior.

### Technical Challenges (Detailing the Embedding Load Blocker)

1. **`TO_VECTOR()` Rejection of Parameterized Literals During Load**: When attempting to load data using `INSERT` or `UPDATE` statements with literal vector strings (e.g., `VALUES (TO_VECTOR('[0.1,0.2,...]'))`), the IRIS ODBC driver may still attempt to treat parts of the literal string as parameters. The `TO_VECTOR()` function then receives these driver-generated parameter markers (e.g., `:%qpar`) instead of the direct literal string it expects, causing the operation to fail.
2. **Core `TO_VECTOR()` Limitation**: Fundamentally, the `TO_VECTOR()` function does not accept any form of parameter marker (e.g., `?`, `:%qpar`) for its vector string argument. It requires a direct string literal.
3. **Impact on Database Integration**: Without the ability to reliably load vector embeddings for new documents, the RAG techniques cannot perform meaningful vector similarity searches on this data, blocking real-data testing and benchmarking.

### Next Steps

1. **Alternative Loading Method**: Investigate alternative methods for loading documents with embeddings, such as using a different database connector or creating a stored procedure in IRIS that can handle the vector conversion.
2. **Manual Testing**: Consider running tests with a smaller set of manually loaded documents to verify the functionality of the RAG techniques.
3. **Driver Enhancement**: Work with InterSystems to enhance the ODBC driver to better handle vector operations.

Despite these challenges, this exercise has provided valuable insights into the limitations of the current implementation and identified areas for improvement.
