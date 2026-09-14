Class RAG.SDK.Bridge
{

ClassMethod AttachTable(table As %String, idCol As %String, textCol As %String, embCol As %String, label As %String, metaCol As %String = "") As %String
{
    // The last dot separates schema from table, so a three-part name resolves correctly.
    Set parts = $Length(table, ".")
    Set tbl = $Piece(table, ".", parts)
    Set schema = $Select(parts > 1:$Piece(table, ".", 1, parts - 1), 1:"SQLUser")
    Set st = ##class(%SQL.Statement).%New()
    Do st.%Prepare("SELECT COUNT(*) AS n FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA=? AND TABLE_NAME=? AND COLUMN_NAME IN (?,?,?)")
    Set rs = st.%Execute(schema, tbl, idCol, textCol, embCol)
    Do rs.%Next()
    If rs.%Get("n") < 3 Return {"error":("Table '"_table_"' or one of columns "_idCol_", "_textCol_", "_embCol_" not found")}.%ToJSON()
    // A metadata column is optional, but a named one that does not exist has to be refused
    // here: every later search would select it and fail.
    If metaCol '= "" {
        Set stm = ##class(%SQL.Statement).%New()
        Do stm.%Prepare("SELECT COUNT(*) AS n FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA=? AND TABLE_NAME=? AND COLUMN_NAME=?")
        Set rsm = stm.%Execute(schema, tbl, metaCol)
        Do rsm.%Next()
        If rsm.%Get("n") < 1 Return {"error":("Table '"_table_"' has no column "_metaCol)}.%ToJSON()
    }
    // The dimension comes from the class dictionary, not from a value read back out of the
    // column: %GetData on a VECTOR column returns the packed vector rather than the
    // comma-separated text that went in, so counting commas reports 1 for every width.
    Set dim = ##class(RAG.SDK.Search).GetVectorDim(table, embCol)
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
    Set vecType = ##class(RAG.SDK.Search).GetVectorType(table, embCol)
    Return {"table":(table),"label":(label),"id_col":(idCol),"embedding_col":(embCol),"metadata_col":(metaCol),"vector_type":(vecType),"dimension":(dim),"rowCount":(rowCount)}.%ToJSON()
}

ClassMethod SetDefaultTable(table As %String, idCol As %String, textCol As %String, embCol As %String, metaCol As %String = "") As %String
{
    Set ^RAG.SDK.Config("default_table") = table
    Set ^RAG.SDK.Config("default_id_col") = idCol
    Set ^RAG.SDK.Config("default_text_col") = textCol
    Set ^RAG.SDK.Config("default_emb_col") = embCol
    // Omitting the metadata column clears any previous one. Keeping a stale setting would
    // make every search against the new table select a column it does not have.
    If metaCol = "" {
        Kill ^RAG.SDK.Config("default_meta_col")
    } Else {
        Set ^RAG.SDK.Config("default_meta_col") = metaCol
    }
    Return {"table":(table),"metadata_col":(metaCol),"status":"ok"}.%ToJSON()
}

ClassMethod GetDefaultTable() As %String
{
    Set table = $Get(^RAG.SDK.Config("default_table"), "")
    If table = "" Return {"table":"RAG.SourceDocuments","id_col":"doc_id","text_col":"text_content","embedding_col":"embedding","metadata_col":"metadata"}.%ToJSON()
    Return {"table":(table),"id_col":($Get(^RAG.SDK.Config("default_id_col"),"doc_id")),"text_col":($Get(^RAG.SDK.Config("default_text_col"),"text_content")),"embedding_col":($Get(^RAG.SDK.Config("default_emb_col"),"embedding")),"metadata_col":($Get(^RAG.SDK.Config("default_meta_col"),""))}.%ToJSON()
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
        // Same reason as RAG.SDK.Search.ReadText: opening a plain varchar value as a stream
        // OID throws <LIST>, which used to break index builds over a consumer's table.
        Set txt = ##class(RAG.SDK.Search).ReadText(rs.%GetData(2))
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
