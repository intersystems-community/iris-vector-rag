/// The SDK's embedding step, kept out of RAG.SDK.Pipeline because it is embedded Python.
///
/// A class carrying a Python method cannot be compiled on every instance: on
/// iris-vector-rag-iris the compile fails with ERROR #7802 and takes the compiler's worker
/// jobs down (signal 11 in messages.log). The write path callers depend on therefore lives in
/// RAG.SDK.Pipeline, which is pure ObjectScript and compiles anywhere, and this class is
/// loaded only where embedded Python works. RAG.SDK.Pipeline.AddDocumentWithEmbed dispatches
/// here when it is present and says so plainly when it is not.
Class RAG.SDK.Embed
{

ClassMethod AddDocumentWithEmbed(id As %String, text As %String, metadata As %String = "{}", model As %String = "all-MiniLM-L6-v2", table As %String = "RAG.SourceDocuments") As %String [ Language = python ]
{
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(model)
        vec = m.encode([text])[0]
        embedding = ",".join(str(float(v)) for v in vec)
        return iris.cls("RAG.SDK.Pipeline").AddDocument(id, text, metadata, embedding, table)
    except ImportError:
        return '{"status":"error","message":"sentence-transformers not installed. Run: import subprocess; subprocess.run([\'pip\',\'install\',\'sentence-transformers\'])"}'
}

}
