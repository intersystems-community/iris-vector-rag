Class RAG.SDK.Schema
{

/// The element type every vector column created here is declared with. IRIS refuses to
/// compare vectors of different element types (SQLCODE -259), so one type has to be
/// canonical: FLOAT, the type this repo's Python paths create. Declared once so a second
/// creation path cannot disagree with the first.
Parameter VECTORTYPE = "FLOAT";

/// Create the document, chunk and entity tables if they are not already there.
///
/// `schema` exists so an installation, or a test, can create the tables somewhere other
/// than RAG without a second copy of the DDL to keep in step with this one.
ClassMethod Initialize(dim As %Integer = 384, schema As %String = "RAG") As %String
{
    Set sql = ..GetDefaultDDL(dim, schema)
    Set stmts = $ListFromString(sql, "GO")
    Set ptr = 0
    While $ListNext(stmts, ptr, stmt) {
        Set stmt = $ZStrip(stmt, "<>W")
        If stmt = "" Continue
        Set st = ##class(%SQL.Statement).%New()
        Set sc = st.%Prepare(stmt)
        If $$$ISERR(sc) Continue
        Do st.%Execute()
    }
    Return {"status":"ok","embeddingDimension":(dim)}.%ToJSON()
}

ClassMethod GetDefaultDDL(dim As %Integer, schema As %String = "RAG") As %String [ Private ]
{
    Set vec = "VECTOR("_..#VECTORTYPE_", "_dim_")"
    Set d1 = "CREATE TABLE IF NOT EXISTS "_schema_".SourceDocuments (doc_id VARCHAR(64) NOT NULL, title VARCHAR(500), text_content LONGVARCHAR, metadata LONGVARCHAR, embedding "_vec_", PRIMARY KEY (doc_id))"
    Set d2 = "CREATE TABLE IF NOT EXISTS "_schema_".DocumentChunks (chunk_id VARCHAR(64) NOT NULL, source_doc_id VARCHAR(64), chunk_text LONGVARCHAR, chunk_embedding "_vec_", PRIMARY KEY (chunk_id))"
    Set d3 = "CREATE TABLE IF NOT EXISTS "_schema_".Entities (entity_id VARCHAR(64) NOT NULL, entity_name VARCHAR(500), entity_type VARCHAR(100), source_doc_id VARCHAR(64), PRIMARY KEY (entity_id))"
    Return d1_"GO"_d2_"GO"_d3
}

ClassMethod Install() As %String
{
    Set version = "0.11.4"
    Set cmd = "pip install iris-vector-rag=="_version
    Set rc = $ZF(-1, cmd)
    If rc = 0 {
        Return {"status":"ok","message":("Installed iris-vector-rag "_version)}.%ToJSON()
    }
    Return {"status":"error","message":("pip exited with code "_rc)}.%ToJSON()
}

/// Drop the three tables Initialize creates, in the schema it created them in.
ClassMethod Drop(schema As %String = "RAG") As %String
{
    For tbl = schema_".Entities", schema_".DocumentChunks", schema_".SourceDocuments" {
        Set st = ##class(%SQL.Statement).%New()
        Do st.%Prepare("DROP TABLE "_tbl)
        Do st.%Execute()
    }
    Return {"status":"ok"}.%ToJSON()
}

ClassMethod Status() As %String
{
    Set tables = []
    Set st = ##class(%SQL.Statement).%New()
    Do st.%Prepare("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA='RAG' AND TABLE_NAME IN ('SourceDocuments','DocumentChunks','Entities') ORDER BY TABLE_NAME")
    Set rs = st.%Execute()
    While rs.%Next() { Do tables.%Push("RAG."_rs.%Get("TABLE_NAME")) }
    Set cnt = 0
    If ##class(%SYSTEM.SQL).TableExists("RAG.SourceDocuments") {
        Set st2 = ##class(%SQL.Statement).%New()
        Do st2.%Prepare("SELECT COUNT(*) AS n FROM RAG.SourceDocuments")
        Set rs2 = st2.%Execute()
        If rs2.%Next() Set cnt = rs2.%Get("n")
    }
    Set dim = 384
    If ##class(%SYSTEM.SQL).TableExists("RAG.SourceDocuments") {
        Set st3 = ##class(%SQL.Statement).%New()
        Do st3.%Prepare("SELECT DATA_TYPE_SIZE FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA='RAG' AND TABLE_NAME='SourceDocuments' AND COLUMN_NAME='embedding'")
        Set rs3 = st3.%Execute()
        If rs3.%Next() Set dim = +rs3.%Get("DATA_TYPE_SIZE")
    }
    Return {"tables":(tables),"embeddingDimension":(dim),"documentCount":(cnt)}.%ToJSON()
}

}
