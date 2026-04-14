Class RAG.SDK.Test.SchemaTest Extends %UnitTest.TestCase
{

Method TestInitialize()
{
    Do ##class(RAG.SDK.Schema).Drop()
    Set json = ##class(RAG.SDK.Schema).Initialize(384)
    Do $$$AssertEquals(##class(%SYSTEM.SQL).TableExists("RAG.SourceDocuments"), 1, "SourceDocuments exists")
    Do $$$AssertEquals(##class(%SYSTEM.SQL).TableExists("RAG.DocumentChunks"), 1, "DocumentChunks exists")
    Do $$$AssertEquals(##class(%SYSTEM.SQL).TableExists("RAG.Entities"), 1, "Entities exists")
}

Method TestEmbeddingColumnExists()
{
    Do ##class(RAG.SDK.Schema).Initialize(384)
    Set st = ##class(%SQL.Statement).%New()
    Do st.%Prepare("SELECT COUNT(*) AS cnt FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA='RAG' AND TABLE_NAME='SourceDocuments' AND COLUMN_NAME='embedding'")
    Set rs = st.%Execute()
    Do rs.%Next()
    Do $$$AssertEquals(rs.%Get("cnt"), 1, "embedding column exists")
}

Method TestInitializeIdempotent()
{
    Do ##class(RAG.SDK.Schema).Initialize(384)
    Set json = ##class(RAG.SDK.Schema).Initialize(384)
    Set obj = {}.%FromJSON(json)
    Do $$$AssertEquals(obj.status, "ok", "second Initialize succeeds")
}

Method TestStatus()
{
    Do ##class(RAG.SDK.Schema).Initialize(384)
    Set json = ##class(RAG.SDK.Schema).Status()
    Set obj = {}.%FromJSON(json)
    Do $$$AssertEquals(obj.embeddingDimension, 384, "dimension reported correctly")
    Do $$$AssertTrue(obj.tables.%IsA("%DynamicArray"), "tables is array")
}

Method TestDrop()
{
    Do ##class(RAG.SDK.Schema).Initialize(384)
    Do ##class(RAG.SDK.Schema).Drop()
    Do $$$AssertEquals(##class(%SYSTEM.SQL).TableExists("RAG.SourceDocuments"), 0, "SourceDocuments dropped")
    Do $$$AssertEquals(##class(%SYSTEM.SQL).TableExists("RAG.DocumentChunks"), 0, "DocumentChunks dropped")
}

Method TestSchemaParity()
{
    Do ##class(RAG.SDK.Schema).Initialize(384)
    Set st = ##class(%SQL.Statement).%New()
    Do st.%Prepare("SELECT COLUMN_NAME FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA='RAG' AND TABLE_NAME='SourceDocuments' ORDER BY ORDINAL_POSITION")
    Set rs = st.%Execute()
    Set cols = ""
    While rs.%Next() { Set cols = cols _ rs.%Get("COLUMN_NAME") _ "," }
    Do $$$AssertTrue($Find(cols, "doc_id") > 0, "doc_id column present")
    Do $$$AssertTrue($Find(cols, "text_content") > 0, "text_content column present")
    Do $$$AssertTrue($Find(cols, "embedding") > 0, "embedding column present")
    Do $$$AssertTrue($Find(cols, "metadata") > 0, "metadata column present")
}

Method TearDown()
{
    Do ##class(RAG.SDK.Schema).Initialize(384)
}

}
