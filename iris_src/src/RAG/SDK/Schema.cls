Class RAG.SDK.Schema
{

ClassMethod Initialize(dim As %Integer = 384) As %String
{
    Set sql = ..GetDefaultDDL(dim)
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

ClassMethod GetDefaultDDL(dim As %Integer) As %String [ Private ]
{
    Set d1 = "CREATE TABLE IF NOT EXISTS RAG.SourceDocuments (doc_id VARCHAR(64) NOT NULL, title VARCHAR(500), text_content LONGVARCHAR, metadata LONGVARCHAR, embedding VECTOR(DOUBLE, "_dim_"), PRIMARY KEY (doc_id))"
    Set d2 = "CREATE TABLE IF NOT EXISTS RAG.DocumentChunks (chunk_id VARCHAR(64) NOT NULL, source_doc_id VARCHAR(64), chunk_text LONGVARCHAR, chunk_embedding VECTOR(DOUBLE, "_dim_"), PRIMARY KEY (chunk_id))"
    Set d3 = "CREATE TABLE IF NOT EXISTS RAG.Entities (entity_id VARCHAR(64) NOT NULL, entity_name VARCHAR(500), entity_type VARCHAR(100), source_doc_id VARCHAR(64), PRIMARY KEY (entity_id))"
    Return d1_"GO"_d2_"GO"_d3
}

ClassMethod Drop() As %String
{
    For tbl = "RAG.Entities", "RAG.DocumentChunks", "RAG.SourceDocuments" {
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
