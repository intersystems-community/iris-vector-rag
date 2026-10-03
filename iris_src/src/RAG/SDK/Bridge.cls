Class RAG.SDK.Bridge
{

/// The error shape the methods that used to report ok regardless now return.
ClassMethod ErrorJSON(pMessage As %String) As %String [ Private ]
{
    Set tSC = $$$OK
    Quit {"status":"error","message":(pMessage)}.%ToJSON()
}

/// Ok when the table and every column name is safe to concatenate into SQL. pMetaCol may be "".
ClassMethod CheckNames(pTable As %String, pIdCol As %String, pTextCol As %String, pEmbCol As %String, pMetaCol As %String = "") As %Status
{
    Set tSC = $$$OK
    Set tSC = ##class(RAG.SDK.Identifier).CheckTable(pTable)
    If $$$ISOK(tSC) Set tSC = ##class(RAG.SDK.Identifier).CheckName(pIdCol, "id column")
    If $$$ISOK(tSC) Set tSC = ##class(RAG.SDK.Identifier).CheckName(pTextCol, "text column")
    If $$$ISOK(tSC) Set tSC = ##class(RAG.SDK.Identifier).CheckName(pEmbCol, "embedding column")
    If $$$ISOK(tSC) && (pMetaCol '= "") Set tSC = ##class(RAG.SDK.Identifier).CheckName(pMetaCol, "metadata column")
    Quit tSC
}

ClassMethod AttachTable(pTable As %String, pIdCol As %String, pTextCol As %String, pEmbCol As %String, pLabel As %String, pMetaCol As %String = "") As %String
{
    Set tSC = $$$OK
    Set tSC = ..CheckNames(pTable, pIdCol, pTextCol, pEmbCol, pMetaCol)
    If $$$ISERR(tSC) Return {"error":($SYSTEM.Status.GetErrorText(tSC))}.%ToJSON()
    // The last dot separates schema from table, so a three-part name resolves correctly.
    Set tParts = $Length(pTable, ".")
    Set tTbl = $Piece(pTable, ".", tParts)
    Set tSchema = $Select(tParts > 1:$Piece(pTable, ".", 1, tParts - 1), 1:"SQLUser")
    Set tStmt = ##class(%SQL.Statement).%New()
    Set tSC = tStmt.%Prepare("SELECT COUNT(*) AS n FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA=? AND TABLE_NAME=? AND COLUMN_NAME IN (?,?,?)")
    If $$$ISERR(tSC) Return {"error":($SYSTEM.Status.GetErrorText(tSC))}.%ToJSON()
    Set tRS = tStmt.%Execute(tSchema, tTbl, pIdCol, pTextCol, pEmbCol)
    Do tRS.%Next()
    If tRS.%Get("n") < 3 Return {"error":("Table '"_pTable_"' or one of columns "_pIdCol_", "_pTextCol_", "_pEmbCol_" not found")}.%ToJSON()
    // A metadata column is optional, but a named one that does not exist has to be refused
    // here: every later search would select it and fail.
    If pMetaCol '= "" {
        Set tStmtM = ##class(%SQL.Statement).%New()
        Set tSC = tStmtM.%Prepare("SELECT COUNT(*) AS n FROM INFORMATION_SCHEMA.COLUMNS WHERE TABLE_SCHEMA=? AND TABLE_NAME=? AND COLUMN_NAME=?")
        If $$$ISERR(tSC) Return {"error":($SYSTEM.Status.GetErrorText(tSC))}.%ToJSON()
        Set tRSM = tStmtM.%Execute(tSchema, tTbl, pMetaCol)
        Do tRSM.%Next()
        If tRSM.%Get("n") < 1 Return {"error":("Table '"_pTable_"' has no column "_pMetaCol)}.%ToJSON()
    }
    // The dimension comes from the class dictionary, not from a value read back out of the
    // column: %GetData on a VECTOR column returns the packed vector rather than the
    // comma-separated text that went in, so counting commas reports 1 for every width.
    Set tDim = ##class(RAG.SDK.Search).GetVectorDim(pTable, pEmbCol)
    Set tStmt3 = ##class(%SQL.Statement).%New()
    Set tSC = tStmt3.%Prepare("SELECT COUNT(*) AS n FROM "_pTable)
    If $$$ISERR(tSC) Return {"error":($SYSTEM.Status.GetErrorText(tSC))}.%ToJSON()
    Set tRS3 = tStmt3.%Execute()
    Do tRS3.%Next()
    Set tRowCount = +tRS3.%Get("n")
    If ##class(%SYSTEM.SQL).TableExists("Graph_KG.table_mappings") {
        Set tStmt4 = ##class(%SQL.Statement).%New()
        Set tSC = tStmt4.%Prepare("DELETE FROM Graph_KG.table_mappings WHERE label=?")
        If $$$ISERR(tSC) Return {"error":($SYSTEM.Status.GetErrorText(tSC))}.%ToJSON()
        Set tRS4 = tStmt4.%Execute(pLabel)
        If tRS4.%SQLCODE < 0 Return {"error":(tRS4.%Message)}.%ToJSON()
        Set tSC = tStmt4.%Prepare("INSERT INTO Graph_KG.table_mappings (table_name, id_col, text_col, vector_col, label) VALUES (?,?,?,?,?)")
        If $$$ISERR(tSC) Return {"error":($SYSTEM.Status.GetErrorText(tSC))}.%ToJSON()
        Set tRS4 = tStmt4.%Execute(pTable, pIdCol, pTextCol, pEmbCol, pLabel)
        If tRS4.%SQLCODE < 0 Return {"error":(tRS4.%Message)}.%ToJSON()
    }
    Set tVecType = ##class(RAG.SDK.Search).GetVectorType(pTable, pEmbCol)
    Quit {"table":(pTable),"label":(pLabel),"id_col":(pIdCol),"embedding_col":(pEmbCol),"metadata_col":(pMetaCol),"vector_type":(tVecType),"dimension":(tDim),"rowCount":(tRowCount)}.%ToJSON()
}

ClassMethod SetDefaultTable(pTable As %String, pIdCol As %String, pTextCol As %String, pEmbCol As %String, pMetaCol As %String = "") As %String
{
    Set tSC = $$$OK
    Set tSC = ..CheckNames(pTable, pIdCol, pTextCol, pEmbCol, pMetaCol)
    // Nothing is written unless every name is safe: these globals feed SQL built by
    // concatenation in RAG.SDK.Search and here.
    If $$$ISERR(tSC) Return ..ErrorJSON($SYSTEM.Status.GetErrorText(tSC))
    Set ^RAG.SDK.Config("default_table") = pTable
    Set ^RAG.SDK.Config("default_id_col") = pIdCol
    Set ^RAG.SDK.Config("default_text_col") = pTextCol
    Set ^RAG.SDK.Config("default_emb_col") = pEmbCol
    // Omitting the metadata column clears any previous one. Keeping a stale setting would
    // make every search against the new table select a column it does not have.
    If pMetaCol = "" {
        Kill ^RAG.SDK.Config("default_meta_col")
    } Else {
        Set ^RAG.SDK.Config("default_meta_col") = pMetaCol
    }
    Quit {"table":(pTable),"metadata_col":(pMetaCol),"status":"ok"}.%ToJSON()
}

ClassMethod GetDefaultTable() As %String
{
    Set tSC = $$$OK
    Set tTable = $Get(^RAG.SDK.Config("default_table"), "")
    If tTable = "" Return {"table":"RAG.SourceDocuments","id_col":"doc_id","text_col":"text_content","embedding_col":"embedding","metadata_col":"metadata"}.%ToJSON()
    Quit {"table":(tTable),"id_col":($Get(^RAG.SDK.Config("default_id_col"),"doc_id")),"text_col":($Get(^RAG.SDK.Config("default_text_col"),"text_content")),"embedding_col":($Get(^RAG.SDK.Config("default_emb_col"),"embedding")),"metadata_col":($Get(^RAG.SDK.Config("default_meta_col"),""))}.%ToJSON()
}

/// Build the BM25 index named pIndexName over the default table. pTextCol is the column to
/// index; left out it is the text column the default table was configured with. Each
/// document is indexed whole: a stream column is read to its end, and the result says how
/// many documents were cut because they exceeded what a string can hold.
ClassMethod BuildBM25Index(pIndexName As %String = "rag_bm25", pTextCol As %String = "") As %String
{
    Set tSC = $$$OK
    Set tCfg = ##class(RAG.SDK.Search).GetTableConfig()
    Set tSC = ##class(RAG.SDK.Search).CheckConfig(tCfg)
    If $$$ISERR(tSC) Return ..ErrorJSON($SYSTEM.Status.GetErrorText(tSC))
    Set tTable = tCfg.%Get("table")
    Set tIdCol = tCfg.%Get("id_col")
    Set tTextCol = $Select(pTextCol '= "":pTextCol, 1:tCfg.%Get("text_col"))
    Set tSC = ##class(RAG.SDK.Identifier).CheckName(tTextCol, "text column")
    If $$$ISERR(tSC) Return ..ErrorJSON($SYSTEM.Status.GetErrorText(tSC))
    Set tStmt = ##class(%SQL.Statement).%New()
    Set tSC = tStmt.%Prepare("SELECT "_tIdCol_", "_tTextCol_" FROM "_tTable)
    If $$$ISERR(tSC) Return ..ErrorJSON($SYSTEM.Status.GetErrorText(tSC))
    Set tRS = tStmt.%Execute()
    If tRS.%SQLCODE < 0 Return ..ErrorJSON(tRS.%Message)
    // The old index is replaced only once the new rows are known to be readable.
    Kill ^BM25Idx(pIndexName)
    Set tCount = 0
    Set tTruncated = 0
    While tRS.%Next() {
        Set tDocId = tRS.%GetData(1)
        // Same reason as RAG.SDK.Search.ReadText: opening a plain varchar value as a stream
        // OID throws <LIST>, which used to break index builds over a consumer's table.
        Set tTxt = ##class(RAG.SDK.Search).ReadText(tRS.%GetData(2), .tCut)
        If tCut Set tTruncated = tTruncated + 1
        Do ##class(Graph.KG.BM25Index).Insert(pIndexName, tDocId, tTxt)
        Set tCount = tCount + 1
    }
    Set ^RAG.SDK.Config("bm25_index_name") = pIndexName
    Set tResult = {"index":(pIndexName),"documents":(tCount),"status":"ok"}
    If tTruncated Do tResult.%Set("truncated", tTruncated, "number")
    Quit tResult.%ToJSON()
}

ClassMethod SetVectorIndex(pIndexName As %String, pNprobe As %Integer = 8) As %String
{
    Set tSC = $$$OK
    Set ^RAG.SDK.Config("ivf_index_name") = pIndexName
    Set ^RAG.SDK.Config("ivf_nprobe") = pNprobe
    Quit {"index":(pIndexName),"nprobe":(pNprobe),"status":"ok"}.%ToJSON()
}

ClassMethod ClearIndexes() As %String
{
    Set tSC = $$$OK
    Kill ^RAG.SDK.Config("bm25_index_name")
    Kill ^RAG.SDK.Config("ivf_index_name")
    Kill ^RAG.SDK.Config("ivf_nprobe")
    Quit {"status":"ok"}.%ToJSON()
}

}
