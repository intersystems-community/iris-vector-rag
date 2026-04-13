/// Contract: RAG.SDK ClassMethod signatures
/// This is the API contract, not implementation.

// === RAG.SDK.Schema ===
// ClassMethod Initialize(dim As %Integer = 384) As %String
//   Reads sql/schema.sql, executes DDL, returns JSON status
// ClassMethod Drop() As %String
//   Drops all RAG.* tables, returns JSON status
// ClassMethod Status() As %String
//   Returns {"tables":[...], "embeddingDimension":N, "documentCount":N}

// === RAG.SDK.Pipeline ===
// ClassMethod AddDocument(id As %String, text As %String, metadata As %String = "{}", embedding As %String = "") As %String
//   INSERT INTO RAG.SourceDocuments ... TO_VECTOR(embedding, DOUBLE, N)
// ClassMethod AddDocumentBatch(jsonArray As %String) As %String
//   Batch insert from JSON array [{"id":"","text":"","metadata":"","embedding":""},...]
// ClassMethod AddDocumentWithEmbed(id As %String, text As %String, metadata As %String = "{}", model As %String = "all-MiniLM-L6-v2") As %String [Language = python]
//   Generates embedding via sentence-transformers, then inserts

// === RAG.SDK.Search ===
// ClassMethod VectorSearch(queryVec As %String, topK As %Integer = 10) As %String
//   SELECT ... VECTOR_COSINE(embedding, TO_VECTOR(?, DOUBLE, N)) ... ORDER BY score DESC
// ClassMethod TextSearch(text As %String, topK As %Integer = 10) As %String
//   SELECT ... WHERE text_content %CONTAINS(text) or LIKE
// ClassMethod HybridSearch(queryVec As %String, text As %String, topK As %Integer = 10, strategy As %String = "RRF") As %String
//   Runs VectorSearch + TextSearch, fuses with RRF (default) or "linear"

// === RAG.SDK.Bridge ===
// ClassMethod AttachTable(table As %String, idCol As %String, textCol As %String, embCol As %String, label As %String) As %String
//   Calls Graph.KG map_sql_table, validates columns, returns JSON summary

// === RAG.SDK.Evaluate ===
// ClassMethod RunRAGAS(questionsJson As %String, groundTruthsJson As %String, topK As %Integer = 5, llmConfig As %String = "") As %String [Language = python]
//   Runs RAGAS evaluate(), returns {"faithfulness":N, "answer_relevancy":N, ...}
