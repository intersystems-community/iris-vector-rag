Class RAG.SDK.Test.PipelineTest Extends %UnitTest.TestCase
{

Method TestAddDocumentWithVector()
{
    Do ##class(RAG.SDK.Schema).Initialize(384)
    Set vec = ..MakeVec(384)
    Set json = ##class(RAG.SDK.Pipeline).AddDocument("test-001", "Patient chest pain", "{}", vec)
    Set obj = {}.%FromJSON(json)
    Do $$$AssertEquals(obj.status, "ok", "AddDocument returns ok")
    Set st = ##class(%SQL.Statement).%New()
    Do st.%Prepare("SELECT COUNT(*) AS n FROM RAG.SourceDocuments WHERE doc_id='test-001'")
    Set rs = st.%Execute()
    Do rs.%Next()
    Do $$$AssertEquals(rs.%Get("n"), 1, "row inserted")
    Do st.%Prepare("SELECT CASE WHEN embedding IS NULL THEN 0 ELSE 1 END AS hasemb FROM RAG.SourceDocuments WHERE doc_id='test-001'")
    Set rs = st.%Execute()
    Do rs.%Next()
    Do $$$AssertEquals(rs.%Get("hasemb"), 1, "embedding not null")
}

Method TestAddDocumentNullEmbedding()
{
    Do ##class(RAG.SDK.Schema).Initialize(384)
    Set json = ##class(RAG.SDK.Pipeline).AddDocument("test-002", "No vector", "{}", "")
    Set obj = {}.%FromJSON(json)
    Do $$$AssertEquals(obj.status, "ok", "null embedding insert ok")
    Set st = ##class(%SQL.Statement).%New()
    Do st.%Prepare("SELECT COUNT(*) AS n FROM RAG.SourceDocuments WHERE doc_id='test-002'")
    Set rs = st.%Execute()
    Do rs.%Next()
    Do $$$AssertEquals(rs.%Get("n"), 1, "row with null embedding exists")
}

Method TestAddDocumentBatch()
{
    Do ##class(RAG.SDK.Schema).Initialize(384)
    Set arr = []
    For i = 1:1:10 {
        Set vec = ..MakeVec(384)
        Set doc = {"id":("batch-"_$ZHex(i)),"text":("Doc "_i),"metadata":"{}","embedding":(vec)}
        Do arr.%Push(doc)
    }
    Set json = ##class(RAG.SDK.Pipeline).AddDocumentBatch(arr.%ToJSON())
    Set obj = {}.%FromJSON(json)
    Do $$$AssertEquals(obj.inserted, 10, "10 docs inserted")
    Set st = ##class(%SQL.Statement).%New()
    Do st.%Prepare("SELECT COUNT(*) AS n FROM RAG.SourceDocuments WHERE doc_id LIKE 'batch-%'")
    Set rs = st.%Execute()
    Do rs.%Next()
    Do $$$AssertEquals(rs.%Get("n"), 10, "10 rows in SQL")
}

Method TestAddDocumentDuplicateId()
{
    Do ##class(RAG.SDK.Schema).Initialize(384)
    Set vec = ..MakeVec(384)
    Do ##class(RAG.SDK.Pipeline).AddDocument("dup-001", "First", "{}", vec)
    Set json = ##class(RAG.SDK.Pipeline).AddDocument("dup-001", "Second", "{}", vec)
    Set obj = {}.%FromJSON(json)
    Do $$$AssertEquals(obj.status, "ok", "duplicate upsert ok")
    Set st = ##class(%SQL.Statement).%New()
    Do st.%Prepare("SELECT COUNT(*) AS n FROM RAG.SourceDocuments WHERE doc_id='dup-001'")
    Set rs = st.%Execute()
    Do rs.%Next()
    Do $$$AssertEquals(rs.%Get("n"), 1, "exactly 1 row for dup-001")
}

Method TestVectorRoundtrip()
{
    Do ##class(RAG.SDK.Schema).Initialize(384)
    Set ones = "1.0"
    For i = 2:1:384 { Set ones = ones_",0.0" }
    Do ##class(RAG.SDK.Pipeline).AddDocument("test-rt", "Roundtrip doc", "{}", ones)
    Set st = ##class(%SQL.Statement).%New()
    Do st.%Prepare("SELECT VECTOR_COSINE(embedding, TO_VECTOR(?, DOUBLE, 384)) AS score FROM RAG.SourceDocuments WHERE doc_id='test-rt'")
    Set rs = st.%Execute(ones)
    Do rs.%Next()
    Set score = +rs.%Get("score")
    Do $$$AssertTrue(score > 0.99, "self-cosine > 0.99, got: "_score)
}

ClassMethod MakeVec(dim As %Integer) As %String [ Private ]
{
    Set vec = "1.0"
    For i = 2:1:dim { Set vec = vec _ ",0.0" }
    Return vec
}

Method TearDown()
{
    Do ##class(RAG.SDK.Schema).Initialize(384)
}

}
