Class RAG.SDK.Bridge
{

ClassMethod AttachTable(table As %String, idCol As %String, textCol As %String, embCol As %String, label As %String) As %String
{
    Set schema = $Piece(table, ".", 1)
    Set tbl = $Piece(table, ".", 2)
    If tbl = "" { Set tbl = schema  Set schema = "SQLUser" }
    Set st = ##class(%SQL.Statement).%New()
    Do st.%Prepare("SELECT COUNT(*) AS n FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA=? AND TABLE_NAME=? AND COLUMN_NAME IN (?,?,?)")
    Set rs = st.%Execute(schema, tbl, idCol, textCol, embCol)
    Do rs.%Next()
    If rs.%Get("n") < 3 Return {"error":("Table '"_table_"' or one of columns "_idCol_", "_textCol_", "_embCol_" not found")}.%ToJSON()
    Set st2 = ##class(%SQL.Statement).%New()
    Do st2.%Prepare("SELECT TOP 1 "_embCol_" FROM "_table_" WHERE "_embCol_" IS NOT NULL")
    Set rs2 = st2.%Execute()
    Set dim = 0
    If rs2.%SQLCODE = 0, rs2.%Next() {
        Set v = rs2.%GetData(1)
        If v '= "" Set dim = $Length(v, ",")
    }
    Set st3 = ##class(%SQL.Statement).%New()
    Do st3.%Prepare("SELECT COUNT(*) AS n FROM "_table)
    Set rs3 = st3.%Execute()
    Do rs3.%Next()
    Set rowCount = +rs3.%Get("n")
    If ##class(%SYSTEM.SQL).TableExists("Graph_KG.table_mappings") {
        Set st4 = ##class(%SQL.Statement).%New()
        Do st4.%Prepare("DELETE FROM Graph_KG.table_mappings WHERE label=?")
        Do st4.%Execute(label)
        Do st4.%Prepare("INSERT INTO Graph_KG.table_mappings (table_name, id_col, text_col, vector_col, label) VALUES (?,?,?,?,?)")
        Do st4.%Execute(table, idCol, textCol, embCol, label)
    }
    Return {"table":(table),"label":(label),"id_col":(idCol),"embedding_col":(embCol),"dimension":(dim),"rowCount":(rowCount)}.%ToJSON()
}

ClassMethod SetDefaultTable(table As %String, idCol As %String, textCol As %String, embCol As %String) As %String
{
    Set ^RAG.SDK.Config("default_table") = table
    Set ^RAG.SDK.Config("default_id_col") = idCol
    Set ^RAG.SDK.Config("default_text_col") = textCol
    Set ^RAG.SDK.Config("default_emb_col") = embCol
    Return {"table":(table),"status":"ok"}.%ToJSON()
}

ClassMethod GetDefaultTable() As %String
{
    Set table = $Get(^RAG.SDK.Config("default_table"), "")
    If table = "" Return {"table":"RAG.SourceDocuments","id_col":"doc_id","text_col":"text_content","embedding_col":"embedding"}.%ToJSON()
    Return {"table":(table),"id_col":($Get(^RAG.SDK.Config("default_id_col"),"doc_id")),"text_col":($Get(^RAG.SDK.Config("default_text_col"),"text_content")),"embedding_col":($Get(^RAG.SDK.Config("default_emb_col"),"embedding"))}.%ToJSON()
}

ClassMethod BuildBM25Index(indexName As %String = "rag_bm25", textCol As %String = "text_content") As %String
{
    Set cfg = ##class(RAG.SDK.Search).GetTableConfig()
    Set table = cfg.%Get("table")
    Set idCol = cfg.%Get("id_col")
    Kill ^BM25Idx(indexName)
    Set st = ##class(%SQL.Statement).%New()
    Do st.%Prepare("SELECT "_idCol_", "_textCol_" FROM "_table)
    Set rs = st.%Execute()
    Set cnt = 0
    While rs.%Next() {
        Set docId = rs.%GetData(1)
        Set txt = rs.%GetData(2)
        Set strm = ##class(%Stream.GlobalCharacter).%Open(txt)  If $IsObject(strm) Set txt = strm.Read(8000)
        Do ##class(Graph.KG.BM25Index).Insert(indexName, docId, txt)
        Set cnt = cnt + 1
    }
    Set ^RAG.SDK.Config("bm25_index_name") = indexName
    Return {"index":(indexName),"documents":(cnt),"status":"ok"}.%ToJSON()
}

ClassMethod SetVectorIndex(indexName As %String, nprobe As %Integer = 8) As %String
{
    Set ^RAG.SDK.Config("ivf_index_name") = indexName
    Set ^RAG.SDK.Config("ivf_nprobe") = nprobe
    Return {"index":(indexName),"nprobe":(nprobe),"status":"ok"}.%ToJSON()
}

ClassMethod ClearIndexes() As %String
{
    Kill ^RAG.SDK.Config("bm25_index_name")
    Kill ^RAG.SDK.Config("ivf_index_name")
    Kill ^RAG.SDK.Config("ivf_nprobe")
    Return {"status":"ok"}.%ToJSON()
}

}
