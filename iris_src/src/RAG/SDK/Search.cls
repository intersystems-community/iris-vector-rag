Class RAG.SDK.Search
{

ClassMethod GetTableConfig() As %DynamicObject
{
    Set table = $Get(^RAG.SDK.Config("default_table"), "")
    // The built-in table is known to have a metadata column. A configured table has one only
    // if the caller said so: guessing "metadata" would make every search against a table
    // without that column fail.
    If table = "" Return {"table":"RAG.SourceDocuments","id_col":"doc_id","text_col":"text_content","embedding_col":"embedding","metadata_col":"metadata"}
    Set idCol = $Get(^RAG.SDK.Config("default_id_col"), "doc_id")
    Set textCol = $Get(^RAG.SDK.Config("default_text_col"), "text_content")
    Set embCol = $Get(^RAG.SDK.Config("default_emb_col"), "embedding")
    Set metaCol = $Get(^RAG.SDK.Config("default_meta_col"), "")
    Return {"table":(table),"id_col":(idCol),"text_col":(textCol),"embedding_col":(embCol),"metadata_col":(metaCol)}
}

/// The compiled property behind a table's column, or "" when it cannot be resolved.
///
/// INFORMATION_SCHEMA.COLUMNS is no help for a VECTOR column -- it reports every one of them
/// as varchar -- so both the element type and the declared dimension come from the class
/// dictionary instead.
ClassMethod VectorProperty(table As %String, column As %String) As %Dictionary.CompiledProperty [ Private ]
{
    Set parts = $Length(table, ".")
    Set name = $Piece(table, ".", parts)
    Set schema = $Select(parts > 1:$Piece(table, ".", 1, parts - 1), 1:"SQLUser")
    Set st = ##class(%SQL.Statement).%New()
    Set sc = st.%Prepare("SELECT CLASSNAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?")
    If $$$ISERR(sc) Return ""
    Set rs = st.%Execute(schema, name)
    If 'rs.%Next() Return ""
    Set class = rs.%GetData(1)
    If class = "" Return ""
    // A DDL-created property is named after the column with the punctuation removed
    // (created_at becomes createdat), so the property is found by its SQL field name.
    Set st2 = ##class(%SQL.Statement).%New()
    Set sc = st2.%Prepare("SELECT Name FROM %Dictionary.CompiledProperty WHERE parent = ? AND (SqlFieldName = ? OR Name = ?)")
    If $$$ISERR(sc) Return ""
    Set rs2 = st2.%Execute(class, column, column)
    If 'rs2.%Next() Return ""
    Set prop = ##class(%Dictionary.CompiledProperty).%OpenId(class_"||"_rs2.%GetData(1))
    If '$IsObject(prop) Return ""
    Return prop
}

/// The element type of a VECTOR column, "FLOAT" or "DOUBLE", or "" when it cannot be
/// resolved. IRIS refuses to compare vectors of different element types (SQLCODE -259), so
/// TO_VECTOR has to name the type the column actually has: this repo's Python paths create
/// FLOAT columns, RAG.SDK.Schema creates DOUBLE ones, and the SDK has to search both.
ClassMethod GetVectorType(table As %String, column As %String) As %String
{
    Set prop = ..VectorProperty(table, column)
    If '$IsObject(prop) Return ""
    Return $ZConvert(prop.Parameters.GetAt("DATATYPE"), "U")
}

/// The declared dimension of a VECTOR column, or 0 when it cannot be resolved.
///
/// Counting commas in a value read back from the column does not work: `%GetData` on a
/// VECTOR column returns the packed vector, not the comma-separated text that went in, so
/// the count is 1 for every column of every width.
ClassMethod GetVectorDim(table As %String, column As %String) As %Integer
{
    Set prop = ..VectorProperty(table, column)
    If '$IsObject(prop) Return 0
    Return +prop.Parameters.GetAt("LEN")
}

/// The text of a row, whether the text column is a stream or a plain varchar. A stream
/// column comes back from %GetData as an OID; opening a varchar value as a stream throws
/// <LIST>, which used to take the whole search down for any table the SDK did not create.
ClassMethod ReadText(value As %String) As %String
{
    If value = "" Return ""
    If $Length(value, "@") < 2 Return value
    Try {
        Set strm = ##class(%Stream.GlobalCharacter).%Open(value)
        If $IsObject(strm) Return strm.Read(2000)
    } Catch {
        Return value
    }
    Return value
}

/// "" when the caller's fragment is acceptable, otherwise the reason it is not.
///
/// The fragment is a predicate, not a statement: a statement terminator or a comment
/// introducer would let one turn into two, or comment out the rest of the clause the SDK
/// built around it. This is a shape check and not a SQL parser — a predicate that is merely
/// wrong still reaches %Prepare and comes back as a named error.
ClassMethod CheckWhereClause(whereClause As %String) As %String [ Private ]
{
    If whereClause = "" Return ""
    If (whereClause [ ";") || (whereClause [ "--") || (whereClause [ "/*") {
        Return "whereClause must be a single predicate: ';', '--' and '/*' are not allowed"
    }
    Return ""
}

/// The caller's predicate as a clause fragment, ANDed and parenthesised so that an
/// `a OR b` predicate cannot widen the SDK's own conditions.
ClassMethod WhereFragment(whereClause As %String) As %String [ Private ]
{
    If whereClause = "" Return ""
    Return " AND ("_whereClause_")"
}

/// A row's metadata as an object or array when it parses as JSON, and as text when it does
/// not. Metadata is the caller's data: unparseable metadata is a row to hand back, not a
/// search to fail.
ClassMethod ParseMetadata(value As %String) As %RawString [ Private ]
{
    Set text = ..ReadText(value)
    If text = "" Return ""
    Set first = $Extract(text, 1)
    If (first '= "{") && (first '= "[") Return text
    Try {
        If first = "{" Return {}.%FromJSON(text)
        Return [].%FromJSON(text)
    } Catch {
        Return text
    }
}

/// The metadata column of a config, or "" when the table has none. Written as its own
/// method so a config produced before 085 -- with no metadata_col key -- reads as "".
ClassMethod MetaColOf(cfg As %DynamicObject) As %String [ Private ]
{
    If '$IsObject(cfg) Return ""
    Return cfg.%Get("metadata_col")
}

ClassMethod VectorSearch(queryVec As %String, topK As %Integer = 10, whereClause As %String = "") As %String
{
    Set reason = ..CheckWhereClause(whereClause)
    If reason '= "" Return {"error":(reason)}.%ToJSON()
    Set ivfIndex = $Get(^RAG.SDK.Config("ivf_index_name"), "")
    // The IVF index covers a whole table and its Search() takes no predicate, so a scoped
    // search cannot use it. Answering a scoped question with globally nearest rows is the
    // failure this argument exists to remove.
    If (ivfIndex '= "") && (whereClause = "") {
        Set nprobe = +$Get(^RAG.SDK.Config("ivf_nprobe"), 8)
        Set raw = ##class(Graph.KG.IVFIndex).Search(ivfIndex, queryVec, topK, nprobe)
        If $Extract(raw, 1) = "[" Return raw
    }
    Set cfg = ..GetTableConfig()
    Set dim = $Length(queryVec, ",")
    If dim < 2 Return [].%ToJSON()
    Set vecType = ..GetVectorType(cfg.%Get("table"), cfg.%Get("embedding_col"))
    If vecType = "" Return {"error":("cannot determine vector element type for "_cfg.%Get("table")_"."_cfg.%Get("embedding_col"))}.%ToJSON()
    Set metaCol = ..MetaColOf(cfg)
    Set sql = "SELECT TOP "_topK_" "_cfg.%Get("id_col")_", "_cfg.%Get("text_col")_", VECTOR_COSINE("_cfg.%Get("embedding_col")_", TO_VECTOR(?,"_vecType_","_dim_")) AS score"
        _$Select(metaCol'="":", "_metaCol, 1:"")
        _" FROM "_cfg.%Get("table")_" WHERE "_cfg.%Get("embedding_col")_" IS NOT NULL"_..WhereFragment(whereClause)_" ORDER BY score DESC"
    Set st = ##class(%SQL.Statement).%New()
    Set sc = st.%Prepare(sql)
    If $$$ISERR(sc) Return {"error":($SYSTEM.Status.GetErrorText(sc))}.%ToJSON()
    Set rs = st.%Execute(queryVec)
    If rs.%SQLCODE < 0 Return {"error":(rs.%Message)}.%ToJSON()
    Set results = []
    While rs.%Next() {
        Set hit = {"id":(rs.%GetData(1)),"text":(..ReadText(rs.%GetData(2))),"score":(+rs.%GetData(3))}
        If metaCol '= "" Do hit.%Set("metadata", ..ParseMetadata(rs.%GetData(4)))
        Do results.%Push(hit)
    }
    Return results.%ToJSON()
}

ClassMethod TextSearch(text As %String, topK As %Integer = 10, whereClause As %String = "") As %String
{
    Set reason = ..CheckWhereClause(whereClause)
    If reason '= "" Return {"error":(reason)}.%ToJSON()
    Set bm25Index = $Get(^RAG.SDK.Config("bm25_index_name"), "")
    // Same reason as the IVF path in VectorSearch: the index cannot honour a predicate.
    If (bm25Index '= "") && (whereClause = "") {
        Set raw = ##class(Graph.KG.BM25Index).Search(bm25Index, text, topK)
        If $Extract(raw, 1) = "[" Return raw
    }
    Set cfg = ..GetTableConfig()
    Set metaCol = ..MetaColOf(cfg)
    Set sql = "SELECT TOP "_topK_" "_cfg.%Get("id_col")_", "_cfg.%Get("text_col")
        _$Select(metaCol'="":", "_metaCol, 1:"")
        _" FROM "_cfg.%Get("table")_" WHERE SUBSTRING("_cfg.%Get("text_col")_",1,4000) LIKE ?"_..WhereFragment(whereClause)
    Set st = ##class(%SQL.Statement).%New()
    Set sc = st.%Prepare(sql)
    // An unscoped failure still returns [] as it did before 085. A scoped one names the
    // failure instead: a caller that asked for a scope has to be able to tell a predicate
    // it got wrong from a scope that genuinely holds no rows.
    If $$$ISERR(sc) {
        If whereClause '= "" Return {"error":($SYSTEM.Status.GetErrorText(sc))}.%ToJSON()
        Return [].%ToJSON()
    }
    Set rs = st.%Execute("%"_text_"%")
    If rs.%SQLCODE < 0 {
        If whereClause '= "" Return {"error":(rs.%Message)}.%ToJSON()
        Return [].%ToJSON()
    }
    Set results = []
    While rs.%Next() {
        Set hit = {"id":(rs.%GetData(1)),"text":(..ReadText(rs.%GetData(2))),"score":1}
        If metaCol '= "" Do hit.%Set("metadata", ..ParseMetadata(rs.%GetData(3)))
        Do results.%Push(hit)
    }
    Return results.%ToJSON()
}

ClassMethod HybridSearch(queryVec As %String, text As %String, topK As %Integer = 10, strategy As %String = "RRF", whereClause As %String = "") As %String
{
    Set vecJson = ..VectorSearch(queryVec, topK * 2, whereClause)
    // A leg that failed returns an object, not an array. Parsing it as an array would throw,
    // so the failure is handed back to the caller as it stands.
    If $Extract(vecJson, 1) '= "[" Return vecJson
    Set txtJson = ..TextSearch(text, topK * 2, whereClause)
    If $Extract(txtJson, 1) '= "[" Return txtJson
    Set vecArr = [].%FromJSON(vecJson)
    Set txtArr = [].%FromJSON(txtJson)
    If strategy = "linear" Return ..FuseLinear(vecArr, txtArr, topK)
    Return ..FuseRRF(vecArr, txtArr, topK)
}

/// The fused row for one id: its fused score, plus the text and metadata the leg that found
/// it already read. Both legs select the text column, so a caller should never have to query
/// for it again per hit.
ClassMethod FusedHit(id As %String, score As %Numeric, doc As %DynamicObject) As %DynamicObject [ Private ]
{
    Set hit = {"id":(id),"score":(score)}
    If '$IsObject(doc) Return hit
    Do hit.%Set("text", doc.%Get("text"))
    // Absent rather than empty when the table has no metadata column, matching the legs.
    If doc.%IsDefined("metadata") Do hit.%Set("metadata", doc.%Get("metadata"))
    Return hit
}

/// The fused array for a set of scores and the leg rows they came from. Locals rather than
/// process-private globals: the fusion data never outlives the fuse.
ClassMethod FusedResults(ByRef scores, ByRef docs, topK As %Integer) As %String [ Private ]
{
    Set results = []
    Set id = ""
    For {
        Set id = $Order(scores(id))
        Quit:id=""
        Do results.%Push(..FusedHit(id, scores(id), $Get(docs(id))))
    }
    Return ..SortByScore(results, topK).%ToJSON()
}

ClassMethod FuseRRF(vecArr As %DynamicArray, txtArr As %DynamicArray, topK As %Integer, k As %Integer = 60) As %String [ Private ]
{
    Set n = vecArr.%Size()
    For i = 0:1:(n-1) {
        Set doc = vecArr.%Get(i)
        Set id = doc.id
        Set scores(id) = $Get(scores(id), 0) + (1 / (k + i + 1))
        If '$Data(docs(id)) Set docs(id) = doc
    }
    Set n = txtArr.%Size()
    For i = 0:1:(n-1) {
        Set doc = txtArr.%Get(i)
        Set id = doc.id
        Set scores(id) = $Get(scores(id), 0) + (1 / (k + i + 1))
        If '$Data(docs(id)) Set docs(id) = doc
    }
    Return ..FusedResults(.scores, .docs, topK)
}

ClassMethod FuseLinear(vecArr As %DynamicArray, txtArr As %DynamicArray, topK As %Integer, alpha As %Double = 0.7) As %String [ Private ]
{
    Set n = vecArr.%Size()
    For i = 0:1:(n-1) {
        Set doc = vecArr.%Get(i)
        Set id = doc.id
        Set scores(id) = $Get(scores(id), 0) + (alpha * +doc.score)
        If '$Data(docs(id)) Set docs(id) = doc
    }
    Set n = txtArr.%Size()
    For i = 0:1:(n-1) {
        Set doc = txtArr.%Get(i)
        Set id = doc.id
        Set scores(id) = $Get(scores(id), 0) + ((1 - alpha) * 1)
        If '$Data(docs(id)) Set docs(id) = doc
    }
    Return ..FusedResults(.scores, .docs, topK)
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
