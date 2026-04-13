# Quickstart: ObjectScript SDK for IVR

## Prerequisites

- IRIS 2025.1+ (any license tier for core operations)
- For embedding generation: `Do $system.Python.Install("sentence-transformers")`
- For RAGAS evaluation: `Do $system.Python.Install("ragas")` and `Do $system.Python.Install("datasets")`
- For Bridge: IVG >= 1.27.0 deployed (`Graph.KG.*` classes loaded)

## 1. Create Schema

```objectscript
Set status = ##class(RAG.SDK.Schema).Initialize(384)
Write status
// {"tables":["RAG.SourceDocuments","RAG.DocumentChunks","RAG.Entities"],"embeddingDimension":384,"documentCount":0}
```

## 2. Add Documents (with pre-computed vectors)

```objectscript
// Single document
Do ##class(RAG.SDK.Pipeline).AddDocument("doc1", "Patient presents with chest pain and shortness of breath.", "{""source"":""ER_notes""}", "0.12,0.34,0.56,...")

// Batch insert (100 documents)
Set batch = [{"id":"doc1","text":"...","metadata":"{}","embedding":"0.12,..."},...]
Do ##class(RAG.SDK.Pipeline).AddDocumentBatch(batch.%ToJSON())
```

## 3. Add Document (auto-generate embedding)

```objectscript
// Requires sentence-transformers installed in IRIS Python env
Do ##class(RAG.SDK.Pipeline).AddDocumentWithEmbed("doc2", "Metformin is a first-line treatment for type 2 diabetes.", "{}")
```

## 4. Search

```objectscript
// Vector similarity search
Set results = ##class(RAG.SDK.Search).VectorSearch(queryVecStr, 10)
Write results
// [{"id":"doc1","text":"Patient presents...","score":0.92},...]

// Text search
Set results = ##class(RAG.SDK.Search).TextSearch("chest pain", 10)

// Hybrid (RRF fusion)
Set results = ##class(RAG.SDK.Search).HybridSearch(queryVecStr, "diabetes treatment", 10, "RRF")
```

## 5. Attach Existing Table

```objectscript
// Zero-copy bridge — use existing IRIS table as RAG corpus
Set result = ##class(RAG.SDK.Bridge).AttachTable("MyApp.ClinicalNotes", "note_id", "note_text", "embedding", "ClinNote")
Write result
// {"table":"MyApp.ClinicalNotes","label":"ClinNote","dimension":384,"rowCount":10000}
```

## 6. Evaluate with RAGAS

```objectscript
// Requires ragas + datasets installed
Set questions = ["What are symptoms of diabetes?","How is metformin metabolized?"]
Set groundTruths = ["Increased thirst, frequent urination.","Eliminated by kidneys."]
Set scores = ##class(RAG.SDK.Evaluate).RunRAGAS(questions.%ToJSON(), groundTruths.%ToJSON(), 5)
Write scores
// {"faithfulness":0.92,"answer_relevancy":0.87,"context_precision":0.81,"context_recall":0.78}
```

## Interoperability with Python IVR

Documents inserted from ObjectScript are immediately searchable from Python:

```python
from iris_vector_rag import create_pipeline
pipeline = create_pipeline("basic", connection_manager=cm, llm_func=llm)
results = pipeline.query("chest pain symptoms", top_k=5)
# Returns doc1 inserted from ObjectScript above
```

And vice versa — Python-ingested documents are searchable from ObjectScript.
