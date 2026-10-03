Class RAG.SDK.Search
{

/// The most characters ReadText hands back for one value. An ObjectScript string holds at most
/// 3,641,144; stopping short of that leaves room to say that a value was cut, rather than
/// failing with <MAXSTRING> on an outsized document.
Parameter MAXTEXT = 3000000;

ClassMethod GetTableConfig() As %DynamicObject
{
    Set tSC = $$$OK
    Set tTable = $Get(^RAG.SDK.Config("default_table"), "")
    // The built-in table is known to have a metadata column. A configured table has one only
    // if the caller said so: guessing "metadata" would make every search against a table
    // without that column fail.
    If tTable = "" Return {"table":"RAG.SourceDocuments","id_col":"doc_id","text_col":"text_content","embedding_col":"embedding","metadata_col":"metadata"}
    Set tIdCol = $Get(^RAG.SDK.Config("default_id_col"), "doc_id")
    Set tTextCol = $Get(^RAG.SDK.Config("default_text_col"), "text_content")
    Set tEmbCol = $Get(^RAG.SDK.Config("default_emb_col"), "embedding")
    Set tMetaCol = $Get(^RAG.SDK.Config("default_meta_col"), "")
    Quit {"table":(tTable),"id_col":(tIdCol),"text_col":(tTextCol),"embedding_col":(tEmbCol),"metadata_col":(tMetaCol)}
}

/// Ok when every name in a config is safe to concatenate into SQL, otherwise an error naming
/// the one that is not. The config is read from a global that SetDefaultTable validates on
/// the way in, but anything can write a global, so every search checks what it reads.
ClassMethod CheckConfig(pCfg As %DynamicObject) As %Status
{
    Set tSC = $$$OK
    Set tSC = ##class(RAG.SDK.Identifier).CheckTable(pCfg.%Get("table"))
    If $$$ISOK(tSC) Set tSC = ##class(RAG.SDK.Identifier).CheckName(pCfg.%Get("id_col"), "id column")
    If $$$ISOK(tSC) Set tSC = ##class(RAG.SDK.Identifier).CheckName(pCfg.%Get("text_col"), "text column")
    If $$$ISOK(tSC) Set tSC = ##class(RAG.SDK.Identifier).CheckName(pCfg.%Get("embedding_col"), "embedding column")
    If $$$ISOK(tSC) && (pCfg.%Get("metadata_col") '= "") {
        Set tSC = ##class(RAG.SDK.Identifier).CheckName(pCfg.%Get("metadata_col"), "metadata column")
    }
    Quit tSC
}

/// The compiled property behind a table's column, or "" when it cannot be resolved.
///
/// INFORMATION_SCHEMA.COLUMNS is no help for a VECTOR column -- it reports every one of them
/// as varchar -- so both the element type and the declared dimension come from the class
/// dictionary instead.
ClassMethod VectorProperty(pTable As %String, pColumn As %String) As %Dictionary.CompiledProperty [ Private ]
{
    Set tSC = $$$OK
    Set tParts = $Length(pTable, ".")
    Set tName = $Piece(pTable, ".", tParts)
    Set tSchema = $Select(tParts > 1:$Piece(pTable, ".", 1, tParts - 1), 1:"SQLUser")
    Set tStmt = ##class(%SQL.Statement).%New()
    Set tSC = tStmt.%Prepare("SELECT CLASSNAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA = ? AND TABLE_NAME = ?")
    If $$$ISERR(tSC) Return ""
    Set tRS = tStmt.%Execute(tSchema, tName)
    If 'tRS.%Next() Return ""
    Set tClass = tRS.%GetData(1)
    If tClass = "" Return ""
    // A DDL-created property is named after the column with the punctuation removed
    // (created_at becomes createdat), so the property is found by its SQL field name.
    Set tStmt2 = ##class(%SQL.Statement).%New()
    Set tSC = tStmt2.%Prepare("SELECT Name FROM %Dictionary.CompiledProperty WHERE parent = ? AND (SqlFieldName = ? OR Name = ?)")
    If $$$ISERR(tSC) Return ""
    Set tRS2 = tStmt2.%Execute(tClass, pColumn, pColumn)
    If 'tRS2.%Next() Return ""
    Set tProp = ##class(%Dictionary.CompiledProperty).%OpenId(tClass_"||"_tRS2.%GetData(1))
    If '$IsObject(tProp) Return ""
    Quit tProp
}

/// The element type of a VECTOR column, "FLOAT" or "DOUBLE", or "" when it cannot be
/// resolved. IRIS refuses to compare vectors of different element types (SQLCODE -259), so
/// TO_VECTOR has to name the type the column actually has. Every creation path now declares
/// the canonical FLOAT -- this repo's Python paths and RAG.SDK.Schema alike -- but the
/// resolution stays: a table created by an earlier release is still DOUBLE, and the SDK has
/// to search and write both.
ClassMethod GetVectorType(pTable As %String, pColumn As %String) As %String
{
    Set tSC = $$$OK
    Set tProp = ..VectorProperty(pTable, pColumn)
    If '$IsObject(tProp) Return ""
    Quit $ZConvert(tProp.Parameters.GetAt("DATATYPE"), "U")
}

/// The declared dimension of a VECTOR column, or 0 when it cannot be resolved.
///
/// Counting commas in a value read back from the column does not work: `%GetData` on a
/// VECTOR column returns the packed vector, not the comma-separated text that went in, so
/// the count is 1 for every column of every width.
ClassMethod GetVectorDim(pTable As %String, pColumn As %String) As %Integer
{
    Set tSC = $$$OK
    Set tProp = ..VectorProperty(pTable, pColumn)
    If '$IsObject(tProp) Return 0
    Quit +tProp.Parameters.GetAt("LEN")
}

/// The text of a row, whether the text column is a stream or a plain varchar. A stream
/// column comes back from %GetData as an OID; opening a varchar value as a stream throws
/// <LIST>, which used to take the whole search down for any table the SDK did not create.
///
/// A stream is read to its end, in chunks. pTruncated is 1 when the stream held more than
/// pMaxChars characters and the text returned is the first pMaxChars of it, so a caller that
/// indexes or displays the text can tell a whole document from a cut one.
ClassMethod ReadText(pValue As %String, Output pTruncated As %Boolean, pMaxChars As %Integer = {..#MAXTEXT}) As %String
{
    Set tSC = $$$OK
    Set pTruncated = 0
    If pValue = "" Return ""
    Try {
        Set tStrm = ..OpenStream(pValue)
        If '$IsObject(tStrm) Return pValue
        Set tText = ""
        While 'tStrm.AtEnd {
            Set tRoom = pMaxChars - $Length(tText)
            Set tChunk = tStrm.Read(32000)
            If $Length(tChunk) > tRoom {
                Set tText = tText_$Extract(tChunk, 1, tRoom)
                Set pTruncated = 1
                Quit
            }
            Set tText = tText_tChunk
        }
        Return tText
    } Catch tEx {
        // Not a stream OID after all -- a varchar that happens to hold an "@".
        Return pValue
    }
    Quit pValue
}

/// The stream a column value refers to, or "" when the value is plain text. A stream column
/// reaches %SQL.Statement as an OID: a $LISTBUILD of (id, class, ...) in current builds, and
/// "id@class" text in older ones.
ClassMethod OpenStream(pValue As %String) As %Stream.Object [ Private ]
{
    Set tSC = $$$OK
    If $ListValid(pValue) && ($ListLength(pValue) >= 2) {
        Set tClass = $ListGet(pValue, 2)
        If $Extract(tClass, 1, 8) = "%Stream." Return $ClassMethod(tClass, "%Open", pValue)
        Return ""
    }
    If $Length(pValue, "@") < 2 Return ""
    Quit ##class(%Stream.GlobalCharacter).%Open(pValue)
}

/// "" when the caller's fragment is acceptable, otherwise the reason it is not.
///
/// The fragment is a predicate, not a statement: a statement terminator or a comment
/// introducer would let one turn into two, or comment out the rest of the clause the SDK
/// built around it. This is a shape check and not a SQL parser — a predicate that is merely
/// wrong still reaches %Prepare and comes back as a named error.
ClassMethod CheckWhereClause(pWhereClause As %String) As %String [ Private ]
{
    Set tSC = $$$OK
    If pWhereClause = "" Return ""
    If (pWhereClause [ ";") || (pWhereClause [ "--") || (pWhereClause [ "/*") {
        Return "whereClause must be a single predicate: ';', '--' and '/*' are not allowed"
    }
    Quit ""
}

/// The caller's predicate as a clause fragment, ANDed and parenthesised so that an
/// `a OR b` predicate cannot widen the SDK's own conditions.
ClassMethod WhereFragment(pWhereClause As %String) As %String [ Private ]
{
    Set tSC = $$$OK
    If pWhereClause = "" Return ""
    Quit " AND ("_pWhereClause_")"
}

/// A row's metadata as an object or array when it parses as JSON, and as text when it does
/// not. Metadata is the caller's data: unparseable metadata is a row to hand back, not a
/// search to fail.
ClassMethod ParseMetadata(pValue As %String) As %RawString [ Private ]
{
    Set tSC = $$$OK
    Set tText = ..ReadText(pValue)
    If tText = "" Return ""
    Set tFirst = $Extract(tText, 1)
    If (tFirst '= "{") && (tFirst '= "[") Return tText
    Try {
        If tFirst = "{" Return {}.%FromJSON(tText)
        Return [].%FromJSON(tText)
    } Catch tEx {
        Return tText
    }
    Quit tText
}

/// The metadata column of a config, or "" when the table has none. Written as its own
/// method so a config produced before 085 -- with no metadata_col key -- reads as "".
ClassMethod MetaColOf(pCfg As %DynamicObject) As %String [ Private ]
{
    Set tSC = $$$OK
    If '$IsObject(pCfg) Return ""
    Quit pCfg.%Get("metadata_col")
}

/// A search hit for one row: id, text and, when the text was cut, "truncated":1.
ClassMethod Hit(pId As %String, pTextValue As %String, pScore As %Numeric) As %DynamicObject [ Private ]
{
    Set tSC = $$$OK
    Set tText = ..ReadText(pTextValue, .tTruncated)
    Set tHit = {"id":(pId),"text":(tText),"score":(pScore)}
    If tTruncated Do tHit.%Set("truncated", 1, "number")
    Quit tHit
}

ClassMethod VectorSearch(pQueryVec As %String, pTopK As %Integer = 10, pWhereClause As %String = "") As %String
{
    Set tSC = $$$OK
    Set tReason = ..CheckWhereClause(pWhereClause)
    If tReason '= "" Return {"error":(tReason)}.%ToJSON()
    Set tTopK = ##class(RAG.SDK.Identifier).TopK(pTopK)
    If tTopK = 0 Return {"error":("topK must be a positive integer, got '"_pTopK_"'")}.%ToJSON()
    Set tIvfIndex = $Get(^RAG.SDK.Config("ivf_index_name"), "")
    // The IVF index covers a whole table and its Search() takes no predicate, so a scoped
    // search cannot use it. Answering a scoped question with globally nearest rows is the
    // failure this argument exists to remove.
    If (tIvfIndex '= "") && (pWhereClause = "") {
        Set tNprobe = +$Get(^RAG.SDK.Config("ivf_nprobe"), 8)
        Set tRaw = ##class(Graph.KG.IVFIndex).Search(tIvfIndex, pQueryVec, tTopK, tNprobe)
        If $Extract(tRaw, 1) = "[" Return tRaw
    }
    Set tCfg = ..GetTableConfig()
    Set tSC = ..CheckConfig(tCfg)
    If $$$ISERR(tSC) Return {"error":($SYSTEM.Status.GetErrorText(tSC))}.%ToJSON()
    Set tDim = $Length(pQueryVec, ",")
    If tDim < 2 Return [].%ToJSON()
    Set tVecType = ..GetVectorType(tCfg.%Get("table"), tCfg.%Get("embedding_col"))
    If tVecType = "" Return {"error":("cannot determine vector element type for "_tCfg.%Get("table")_"."_tCfg.%Get("embedding_col"))}.%ToJSON()
    Set tMetaCol = ..MetaColOf(tCfg)
    // No TOP: the ORDER BY stays and the loop below stops at topK rows. TOP and ORDER BY are
    // not combined in this SDK's SQL.
    Set tSql = "SELECT "_tCfg.%Get("id_col")_", "_tCfg.%Get("text_col")_", VECTOR_COSINE("_tCfg.%Get("embedding_col")_", TO_VECTOR(?,"_tVecType_","_tDim_")) AS score"
        _$Select(tMetaCol'="":", "_tMetaCol, 1:"")
        _" FROM "_tCfg.%Get("table")_" WHERE "_tCfg.%Get("embedding_col")_" IS NOT NULL"_..WhereFragment(pWhereClause)_" ORDER BY score DESC"
    Set tStmt = ##class(%SQL.Statement).%New()
    Set tSC = tStmt.%Prepare(tSql)
    If $$$ISERR(tSC) Return {"error":($SYSTEM.Status.GetErrorText(tSC))}.%ToJSON()
    Set tRS = tStmt.%Execute(pQueryVec)
    If tRS.%SQLCODE < 0 Return {"error":(tRS.%Message)}.%ToJSON()
    Set tResults = []
    While tRS.%Next() {
        Quit:tResults.%Size()>=tTopK
        Set tHit = ..Hit(tRS.%GetData(1), tRS.%GetData(2), +tRS.%GetData(3))
        If tMetaCol '= "" Do tHit.%Set("metadata", ..ParseMetadata(tRS.%GetData(4)))
        Do tResults.%Push(tHit)
    }
    Quit tResults.%ToJSON()
}

ClassMethod TextSearch(pText As %String, pTopK As %Integer = 10, pWhereClause As %String = "") As %String
{
    Set tSC = $$$OK
    Set tReason = ..CheckWhereClause(pWhereClause)
    If tReason '= "" Return {"error":(tReason)}.%ToJSON()
    Set tTopK = ##class(RAG.SDK.Identifier).TopK(pTopK)
    If tTopK = 0 Return {"error":("topK must be a positive integer, got '"_pTopK_"'")}.%ToJSON()
    Set tBm25Index = $Get(^RAG.SDK.Config("bm25_index_name"), "")
    // Same reason as the IVF path in VectorSearch: the index cannot honour a predicate.
    If (tBm25Index '= "") && (pWhereClause = "") {
        Set tRaw = ##class(Graph.KG.BM25Index).Search(tBm25Index, pText, tTopK)
        If $Extract(tRaw, 1) = "[" Return tRaw
    }
    Set tCfg = ..GetTableConfig()
    Set tSC = ..CheckConfig(tCfg)
    If $$$ISERR(tSC) Return {"error":($SYSTEM.Status.GetErrorText(tSC))}.%ToJSON()
    Set tMetaCol = ..MetaColOf(tCfg)
    Set tSql = "SELECT TOP "_tTopK_" "_tCfg.%Get("id_col")_", "_tCfg.%Get("text_col")
        _$Select(tMetaCol'="":", "_tMetaCol, 1:"")
        _" FROM "_tCfg.%Get("table")_" WHERE SUBSTRING("_tCfg.%Get("text_col")_",1,4000) LIKE ?"_..WhereFragment(pWhereClause)
    Set tStmt = ##class(%SQL.Statement).%New()
    Set tSC = tStmt.%Prepare(tSql)
    // A search that could not run says so, scoped or not. It used to hand back [] when no
    // scope was given, which a caller could not tell from a search that found nothing.
    If $$$ISERR(tSC) Return {"error":($SYSTEM.Status.GetErrorText(tSC))}.%ToJSON()
    Set tRS = tStmt.%Execute("%"_pText_"%")
    If tRS.%SQLCODE < 0 Return {"error":(tRS.%Message)}.%ToJSON()
    Set tResults = []
    While tRS.%Next() {
        Set tHit = ..Hit(tRS.%GetData(1), tRS.%GetData(2), 1)
        If tMetaCol '= "" Do tHit.%Set("metadata", ..ParseMetadata(tRS.%GetData(3)))
        Do tResults.%Push(tHit)
    }
    Quit tResults.%ToJSON()
}

ClassMethod HybridSearch(pQueryVec As %String, pText As %String, pTopK As %Integer = 10, pStrategy As %String = "RRF", pWhereClause As %String = "") As %String
{
    Set tSC = $$$OK
    Set tTopK = ##class(RAG.SDK.Identifier).TopK(pTopK)
    If tTopK = 0 Return {"error":("topK must be a positive integer, got '"_pTopK_"'")}.%ToJSON()
    Set tVecJson = ..VectorSearch(pQueryVec, tTopK * 2, pWhereClause)
    // A leg that failed returns an object, not an array. Parsing it as an array would throw,
    // so the failure is handed back to the caller as it stands.
    If $Extract(tVecJson, 1) '= "[" Return tVecJson
    Set tTxtJson = ..TextSearch(pText, tTopK * 2, pWhereClause)
    If $Extract(tTxtJson, 1) '= "[" Return tTxtJson
    Set tVecArr = [].%FromJSON(tVecJson)
    Set tTxtArr = [].%FromJSON(tTxtJson)
    If pStrategy = "linear" Return ..FuseLinear(tVecArr, tTxtArr, tTopK)
    Quit ..FuseRRF(tVecArr, tTxtArr, tTopK)
}

/// The fused row for one id: its fused score, plus the text and metadata the leg that found
/// it already read. Both legs select the text column, so a caller should never have to query
/// for it again per hit.
ClassMethod FusedHit(pId As %String, pScore As %Numeric, pDoc As %DynamicObject) As %DynamicObject [ Private ]
{
    Set tSC = $$$OK
    Set tHit = {"id":(pId),"score":(pScore)}
    If '$IsObject(pDoc) Return tHit
    Do tHit.%Set("text", pDoc.%Get("text"))
    If pDoc.%IsDefined("truncated") Do tHit.%Set("truncated", 1, "number")
    // Absent rather than empty when the table has no metadata column, matching the legs.
    If pDoc.%IsDefined("metadata") Do tHit.%Set("metadata", pDoc.%Get("metadata"))
    Quit tHit
}

/// The fused array for a set of scores and the leg rows they came from. Locals rather than
/// process-private globals: the fusion data never outlives the fuse.
ClassMethod FusedResults(ByRef pScores, ByRef pDocs, pTopK As %Integer) As %String [ Private ]
{
    Set tSC = $$$OK
    Set tResults = []
    Set tId = ""
    For {
        Set tId = $Order(pScores(tId))
        Quit:tId=""
        Do tResults.%Push(..FusedHit(tId, pScores(tId), $Get(pDocs(tId))))
    }
    Quit ..SortByScore(tResults, pTopK).%ToJSON()
}

ClassMethod FuseRRF(pVecArr As %DynamicArray, pTxtArr As %DynamicArray, pTopK As %Integer, pK As %Integer = 60) As %String [ Private ]
{
    Set tSC = $$$OK
    Set tN = pVecArr.%Size()
    For tI = 0:1:(tN-1) {
        Set tDoc = pVecArr.%Get(tI)
        Set tId = tDoc.id
        Set tScores(tId) = $Get(tScores(tId), 0) + (1 / (pK + tI + 1))
        If '$Data(tDocs(tId)) Set tDocs(tId) = tDoc
    }
    Set tN = pTxtArr.%Size()
    For tI = 0:1:(tN-1) {
        Set tDoc = pTxtArr.%Get(tI)
        Set tId = tDoc.id
        Set tScores(tId) = $Get(tScores(tId), 0) + (1 / (pK + tI + 1))
        If '$Data(tDocs(tId)) Set tDocs(tId) = tDoc
    }
    Quit ..FusedResults(.tScores, .tDocs, pTopK)
}

ClassMethod FuseLinear(pVecArr As %DynamicArray, pTxtArr As %DynamicArray, pTopK As %Integer, pAlpha As %Double = 0.7) As %String [ Private ]
{
    Set tSC = $$$OK
    Set tN = pVecArr.%Size()
    For tI = 0:1:(tN-1) {
        Set tDoc = pVecArr.%Get(tI)
        Set tId = tDoc.id
        Set tScores(tId) = $Get(tScores(tId), 0) + (pAlpha * +tDoc.score)
        If '$Data(tDocs(tId)) Set tDocs(tId) = tDoc
    }
    Set tN = pTxtArr.%Size()
    For tI = 0:1:(tN-1) {
        Set tDoc = pTxtArr.%Get(tI)
        Set tId = tDoc.id
        Set tScores(tId) = $Get(tScores(tId), 0) + ((1 - pAlpha) * 1)
        If '$Data(tDocs(tId)) Set tDocs(tId) = tDoc
    }
    Quit ..FusedResults(.tScores, .tDocs, pTopK)
}

ClassMethod SortByScore(pArr As %DynamicArray, pTopK As %Integer) As %DynamicArray [ Private ]
{
    Set tSC = $$$OK
    Set tN = pArr.%Size()
    For tI = 0:1:(tN-2) {
        For tJ = 0:1:(tN-tI-2) {
            Set tA = pArr.%Get(tJ)
            Set tB = pArr.%Get(tJ+1)
            If +tA.score < +tB.score {
                Do pArr.%Set(tJ, tB)
                Do pArr.%Set(tJ+1, tA)
            }
        }
    }
    If tN <= pTopK Return pArr
    Set tResult = []
    For tI = 0:1:(pTopK-1) { Do tResult.%Push(pArr.%Get(tI)) }
    Quit tResult
}

}
