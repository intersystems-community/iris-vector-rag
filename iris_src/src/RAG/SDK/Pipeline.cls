Class RAG.SDK.Pipeline
{

/// The element type to write with when the target column's own type cannot be read. The same
/// canonical type RAG.SDK.Schema creates with, so a table this SDK made is always writable.
Parameter VECTORTYPE = "FLOAT";

/// Insert one document, or update it when its id is already there.
///
/// `table` names the document table, so a caller whose table lives in another schema -- or
/// predates this SDK -- does not need a second copy of this method.
///
/// The vector's element type comes from the column, never from this code: IRIS refuses to
/// compare vectors of different element types (SQLCODE -259), so a FLOAT vector written into
/// a DOUBLE column is a row no search can reach.
ClassMethod AddDocument(pId As %String, pText As %String, pMetadata As %String = "{}", pEmbedding As %String = "", pTable As %String = "RAG.SourceDocuments") As %String
{
    Set tSC = $$$OK
    Set tSC = ##class(RAG.SDK.Identifier).CheckTable(pTable)
    If $$$ISERR(tSC) Return ..ErrorJSON($SYSTEM.Status.GetErrorText(tSC))
    Set tStmt = ##class(%SQL.Statement).%New()
    If pEmbedding '= "" {
        Set tVec = "TO_VECTOR(?,"_..VectorTypeOf(pTable)_","_$Length(pEmbedding, ",")_")"
        Set tSC = tStmt.%Prepare("INSERT OR UPDATE INTO "_pTable_" (doc_id, text_content, metadata, embedding) VALUES (?,?,?,"_tVec_")")
        If $$$ISERR(tSC) {
            Set tSC = tStmt.%Prepare("INSERT INTO "_pTable_" (doc_id, text_content, metadata, embedding) VALUES (?,?,?,"_tVec_")")
            If $$$ISERR(tSC) Return ..ErrorJSON($SYSTEM.Status.GetErrorText(tSC))
        }
        Set tRS = tStmt.%Execute(pId, pText, pMetadata, pEmbedding)
    } Else {
        Set tSC = tStmt.%Prepare("INSERT OR UPDATE INTO "_pTable_" (doc_id, text_content, metadata) VALUES (?,?,?)")
        If $$$ISERR(tSC) {
            Set tSC = tStmt.%Prepare("INSERT INTO "_pTable_" (doc_id, text_content, metadata) VALUES (?,?,?)")
            If $$$ISERR(tSC) Return ..ErrorJSON($SYSTEM.Status.GetErrorText(tSC))
        }
        Set tRS = tStmt.%Execute(pId, pText, pMetadata)
    }
    If tRS.%SQLCODE = 0 Return {"status":"ok","id":(pId)}.%ToJSON()
    // -119 is "this row already exists", which can only be reached when the INSERT OR UPDATE
    // above failed to prepare and the plain INSERT ran in its place.
    If tRS.%SQLCODE = -119 Return ..UpdateDocument(pId, pText, pMetadata, pEmbedding, pTable)
    Quit ..ErrorJSON(tRS.%Message)
}

/// The error shape every method here returns: status "error" and the reason.
ClassMethod ErrorJSON(pMessage As %String) As %String [ Private ]
{
    Set tSC = $$$OK
    Quit {"status":"error","message":(pMessage)}.%ToJSON()
}

/// The element type of a document table's embedding column, or the canonical type when it
/// cannot be resolved -- a table that does not exist yet, or a column that is not a vector.
ClassMethod VectorTypeOf(pTable As %String) As %String [ Private ]
{
    Set tSC = $$$OK
    Set tVecType = ##class(RAG.SDK.Search).GetVectorType(pTable, "embedding")
    If tVecType = "" Return ..#VECTORTYPE
    Quit tVecType
}

/// Update the row an INSERT found already present. The embedding is left as it is when the
/// caller passed none, rather than overwritten with an empty vector.
ClassMethod UpdateDocument(pId As %String, pText As %String, pMetadata As %String, pEmbedding As %String, pTable As %String) As %String [ Private ]
{
    Set tSC = $$$OK
    Set tSql = "UPDATE "_pTable_" SET text_content=?,metadata=?"
    If pEmbedding '= "" {
        Set tSql = tSql_",embedding=TO_VECTOR(?,"_..VectorTypeOf(pTable)_","_$Length(pEmbedding, ",")_")"
    }
    Set tSql = tSql_" WHERE doc_id=?"
    Set tStmt = ##class(%SQL.Statement).%New()
    Set tSC = tStmt.%Prepare(tSql)
    If $$$ISERR(tSC) Return ..ErrorJSON($SYSTEM.Status.GetErrorText(tSC))
    If pEmbedding '= "" {
        Set tRS = tStmt.%Execute(pText, pMetadata, pEmbedding, pId)
    } Else {
        Set tRS = tStmt.%Execute(pText, pMetadata, pId)
    }
    If tRS.%SQLCODE '= 0 Return ..ErrorJSON(tRS.%Message)
    Quit {"status":"ok","id":(pId)}.%ToJSON()
}

ClassMethod AddDocumentBatch(pJsonArray As %String, pTable As %String = "RAG.SourceDocuments") As %String
{
    Set tSC = $$$OK
    Set tArr = [].%FromJSON(pJsonArray)
    Set tInserted = 0
    Set tErrors = 0
    Set tN = tArr.%Size()
    For tI = 0:1:(tN-1) {
        Set tDoc = tArr.%Get(tI)
        Set tResult = ..AddDocument(tDoc.id, tDoc.text, $Select(tDoc.metadata'="":tDoc.metadata, 1:"{}"), $Select(tDoc.embedding'="":tDoc.embedding, 1:""), pTable)
        Set tObj = {}.%FromJSON(tResult)
        If tObj.status = "ok" { Set tInserted = tInserted + 1 }
        Else { Set tErrors = tErrors + 1 }
    }
    Quit {"inserted":(tInserted),"errors":(tErrors)}.%ToJSON()
}

/// Embed the text with a sentence-transformers model, then add the document.
///
/// The embedding step itself is embedded Python and lives in RAG.SDK.Embed. Compiling a class
/// that carries a Python method fails outright on an instance whose embedded Python is not
/// usable -- ERROR #7802 on iris-vector-rag-iris, which takes the compiler's worker jobs down
/// with it -- and that must not stop this class, which holds the write path, from compiling.
ClassMethod AddDocumentWithEmbed(pId As %String, pText As %String, pMetadata As %String = "{}", pModel As %String = "all-MiniLM-L6-v2", pTable As %String = "RAG.SourceDocuments") As %String
{
    Set tSC = $$$OK
    Set tSC = ##class(RAG.SDK.Identifier).CheckTable(pTable)
    If $$$ISERR(tSC) Return ..ErrorJSON($SYSTEM.Status.GetErrorText(tSC))
    If '##class(%Dictionary.CompiledClass).%ExistsId("RAG.SDK.Embed") {
        Return ..ErrorJSON("RAG.SDK.Embed is not compiled on this instance, so no embedding can be computed here")
    }
    Quit $CLASSMETHOD("RAG.SDK.Embed", "AddDocumentWithEmbed", pId, pText, pMetadata, pModel, pTable)
}

}
