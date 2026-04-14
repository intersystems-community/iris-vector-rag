Class RAG.SDK.Pipeline
{

ClassMethod AddDocument(id As %String, text As %String, metadata As %String = "{}", embedding As %String = "") As %String
{
    Set st = ##class(%SQL.Statement).%New()
    If embedding '= "" {
        Set dim = $Length(embedding, ",")
        Set sc = st.%Prepare("INSERT OR UPDATE INTO RAG.SourceDocuments (doc_id, text_content, metadata, embedding) VALUES (?,?,?,TO_VECTOR(?,DOUBLE,"_dim_"))")
        If $$$ISERR(sc) {
            Set sc = st.%Prepare("INSERT INTO RAG.SourceDocuments (doc_id, text_content, metadata, embedding) VALUES (?,?,?,TO_VECTOR(?,DOUBLE,"_dim_"))")
            If $$$ISERR(sc) Return {"status":"error","message":($SYSTEM.Status.GetErrorText(sc))}.%ToJSON()
        }
        Set rs = st.%Execute(id, text, metadata, embedding)
    } Else {
        Set sc = st.%Prepare("INSERT OR UPDATE INTO RAG.SourceDocuments (doc_id, text_content, metadata) VALUES (?,?,?)")
        If $$$ISERR(sc) {
            Set sc = st.%Prepare("INSERT INTO RAG.SourceDocuments (doc_id, text_content, metadata) VALUES (?,?,?)")
            If $$$ISERR(sc) Return {"status":"error","message":($SYSTEM.Status.GetErrorText(sc))}.%ToJSON()
        }
        Set rs = st.%Execute(id, text, metadata)
    }
    If rs.%SQLCODE = 0 Return {"status":"ok","id":(id)}.%ToJSON()
    If rs.%SQLCODE = -119 {
        Set sc2 = st.%Prepare("UPDATE RAG.SourceDocuments SET text_content=?,metadata=?,embedding=CASE WHEN ?='' THEN embedding ELSE TO_VECTOR(?,DOUBLE,384) END WHERE doc_id=?")
        Set rs2 = st2.%Execute(text, metadata, embedding, embedding, id)
        Return {"status":"ok","id":(id)}.%ToJSON()
    }
    Return {"status":"ok","id":(id)}.%ToJSON()
}

ClassMethod AddDocumentBatch(jsonArray As %String) As %String
{
    Set arr = [].%FromJSON(jsonArray)
    Set inserted = 0
    Set errors = 0
    Set n = arr.%Size()
    For i = 0:1:(n-1) {
        Set doc = arr.%Get(i)
        Set result = ..AddDocument(doc.id, doc.text, $Select(doc.metadata'="":doc.metadata, 1:"{}"), $Select(doc.embedding'="":doc.embedding, 1:""))
        Set obj = {}.%FromJSON(result)
        If obj.status = "ok" { Set inserted = inserted + 1 }
        Else { Set errors = errors + 1 }
    }
    Return {"inserted":(inserted),"errors":(errors)}.%ToJSON()
}

ClassMethod AddDocumentWithEmbed(id As %String, text As %String, metadata As %String = "{}", model As %String = "all-MiniLM-L6-v2") As %String [ Language = python ]
{
    try:
        from sentence_transformers import SentenceTransformer
        m = SentenceTransformer(model)
        vec = m.encode([text])[0]
        embedding = ",".join(str(float(v)) for v in vec)
        return iris.cls("RAG.SDK.Pipeline").AddDocument(id, text, metadata, embedding)
    except ImportError:
        return '{"status":"error","message":"sentence-transformers not installed. Run: import subprocess; subprocess.run([\'pip\',\'install\',\'sentence-transformers\'])"}'
}

}
