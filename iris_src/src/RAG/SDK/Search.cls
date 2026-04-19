Class RAG.SDK.Search
{

ClassMethod GetTableConfig() As %DynamicObject
{
    Set table = $Get(^RAG.SDK.Config("default_table"), "")
    If table = "" Return {"table":"RAG.SourceDocuments","id_col":"doc_id","text_col":"text_content","embedding_col":"embedding"}
    Set idCol = $Get(^RAG.SDK.Config("default_id_col"), "doc_id")
    Set textCol = $Get(^RAG.SDK.Config("default_text_col"), "text_content")
    Set embCol = $Get(^RAG.SDK.Config("default_emb_col"), "embedding")
    Return {"table":(table),"id_col":(idCol),"text_col":(textCol),"embedding_col":(embCol)}
}

ClassMethod VectorSearch(queryVec As %String, topK As %Integer = 10) As %String
{
    Set ivfIndex = $Get(^RAG.SDK.Config("ivf_index_name"), "")
    If ivfIndex '= "" {
        Set nprobe = +$Get(^RAG.SDK.Config("ivf_nprobe"), 8)
        Set raw = ##class(Graph.KG.IVFIndex).Search(ivfIndex, queryVec, topK, nprobe)
        If $Extract(raw, 1) = "[" Return raw
    }
    Set cfg = ..GetTableConfig()
    Set dim = $Length(queryVec, ",")
    If dim < 2 Return [].%ToJSON()
    Set sql = "SELECT TOP "_topK_" "_cfg.%Get("id_col")_", "_cfg.%Get("text_col")_", VECTOR_COSINE("_cfg.%Get("embedding_col")_", TO_VECTOR(?,DOUBLE,"_dim_")) AS score FROM "_cfg.%Get("table")_" WHERE "_cfg.%Get("embedding_col")_" IS NOT NULL ORDER BY score DESC"
    Set st = ##class(%SQL.Statement).%New()
    Set sc = st.%Prepare(sql)
    If $$$ISERR(sc) Return {"error":($SYSTEM.Status.GetErrorText(sc))}.%ToJSON()
    Set rs = st.%Execute(queryVec)
    If rs.%SQLCODE < 0 Return {"error":(rs.%Message)}.%ToJSON()
    Set results = []
    While rs.%Next() {
        Set txt2 = rs.%GetData(2)
        Set strm2 = ##class(%Stream.GlobalCharacter).%Open(txt2)  If $IsObject(strm2) Set txt2 = strm2.Read(2000)
        Do results.%Push({"id":(rs.%GetData(1)),"text":(txt2),"score":(+rs.%GetData(3))})
    }
    Return results.%ToJSON()
}

ClassMethod TextSearch(text As %String, topK As %Integer = 10) As %String
{
    Set bm25Index = $Get(^RAG.SDK.Config("bm25_index_name"), "")
    If bm25Index '= "" {
        Set raw = ##class(Graph.KG.BM25Index).Search(bm25Index, text, topK)
        If $Extract(raw, 1) = "[" Return raw
    }
    Set cfg = ..GetTableConfig()
    Set sql = "SELECT TOP "_topK_" "_cfg.%Get("id_col")_", "_cfg.%Get("text_col")_" FROM "_cfg.%Get("table")_" WHERE SUBSTRING("_cfg.%Get("text_col")_",1,4000) LIKE ?"
    Set st = ##class(%SQL.Statement).%New()
    Set sc = st.%Prepare(sql)
    If $$$ISERR(sc) Return [].%ToJSON()
    Set rs = st.%Execute("%"_text_"%")
    If rs.%SQLCODE < 0 Return [].%ToJSON()
    Set results = []
    While rs.%Next() {
        Set txt = rs.%GetData(2)
        Set strm = ##class(%Stream.GlobalCharacter).%Open(txt)  If $IsObject(strm) Set txt = strm.Read(2000)
        Do results.%Push({"id":(rs.%GetData(1)),"text":(txt),"score":1})
    }
    Return results.%ToJSON()
}

ClassMethod HybridSearch(queryVec As %String, text As %String, topK As %Integer = 10, strategy As %String = "RRF") As %String
{
    Set vecJson = ..VectorSearch(queryVec, topK * 2)
    Set txtJson = ..TextSearch(text, topK * 2)
    Set vecArr = [].%FromJSON(vecJson)
    Set txtArr = [].%FromJSON(txtJson)
    If strategy = "linear" Return ..FuseLinear(vecArr, txtArr, topK)
    Return ..FuseRRF(vecArr, txtArr, topK)
}

ClassMethod FuseRRF(vecArr As %DynamicArray, txtArr As %DynamicArray, topK As %Integer, k As %Integer = 60) As %String [ Private ]
{
    Kill ^||RRFScores
    Set n = vecArr.%Size()
    For i = 0:1:(n-1) {
        Set id = vecArr.%Get(i).id
        Set ^||RRFScores(id) = $Get(^||RRFScores(id), 0) + (1 / (k + i + 1))
    }
    Set n = txtArr.%Size()
    For i = 0:1:(n-1) {
        Set id = txtArr.%Get(i).id
        Set ^||RRFScores(id) = $Get(^||RRFScores(id), 0) + (1 / (k + i + 1))
    }
    Set results = []
    Set id = ""
    For {
        Set id = $Order(^||RRFScores(id))
        Quit:id=""
        Do results.%Push({"id":(id),"score":(^||RRFScores(id))})
    }
    Kill ^||RRFScores
    Return ..SortByScore(results, topK).%ToJSON()
}

ClassMethod FuseLinear(vecArr As %DynamicArray, txtArr As %DynamicArray, topK As %Integer, alpha As %Double = 0.7) As %String [ Private ]
{
    Kill ^||LinScores
    Set n = vecArr.%Size()
    For i = 0:1:(n-1) {
        Set doc = vecArr.%Get(i)
        Set id = doc.id
        Set ^||LinScores(id) = $Get(^||LinScores(id), 0) + (alpha * +doc.score)
    }
    Set n = txtArr.%Size()
    For i = 0:1:(n-1) {
        Set id = txtArr.%Get(i).id
        Set ^||LinScores(id) = $Get(^||LinScores(id), 0) + ((1 - alpha) * 1)
    }
    Set results = []
    Set id = ""
    For {
        Set id = $Order(^||LinScores(id))
        Quit:id=""
        Do results.%Push({"id":(id),"score":(^||LinScores(id))})
    }
    Kill ^||LinScores
    Return ..SortByScore(results, topK).%ToJSON()
}

ClassMethod SortByScore(arr As %DynamicArray, topK As %Integer) As %DynamicArray [ Private ]
{
    Set n = arr.%Size()
    For i = 0:1:(n-2) {
        For j = 0:1:(n-i-2) {
            Set a = arr.%Get(j)
            Set b = arr.%Get(j+1)
            If +a.score < +b.score {
                Do arr.%Set(j, b)
                Do arr.%Set(j+1, a)
            }
        }
    }
    If n <= topK Return arr
    Set result = []
    For i = 0:1:(topK-1) { Do result.%Push(arr.%Get(i)) }
    Return result
}

}
