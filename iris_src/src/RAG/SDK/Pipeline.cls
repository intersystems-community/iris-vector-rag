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
ClassMethod AddDocument(id As %String, text As %String, metadata As %String = "{}", embedding As %String = "", table As %String = "RAG.SourceDocuments") As %String
{
    Set st = ##class(%SQL.Statement).%New()
    If embedding '= "" {
        Set vec = "TO_VECTOR(?,"_..VectorTypeOf(table)_","_$Length(embedding, ",")_")"
        Set sc = st.%Prepare("INSERT OR UPDATE INTO "_table_" (doc_id, text_content, metadata, embedding) VALUES (?,?,?,"_vec_")")
        If $$$ISERR(sc) {
            Set sc = st.%Prepare("INSERT INTO "_table_" (doc_id, text_content, metadata, embedding) VALUES (?,?,?,"_vec_")")
            If $$$ISERR(sc) Return {"status":"error","message":($SYSTEM.Status.GetErrorText(sc))}.%ToJSON()
        }
        Set rs = st.%Execute(id, text, metadata, embedding)
    } Else {
        Set sc = st.%Prepare("INSERT OR UPDATE INTO "_table_" (doc_id, text_content, metadata) VALUES (?,?,?)")
        If $$$ISERR(sc) {
            Set sc = st.%Prepare("INSERT INTO "_table_" (doc_id, text_content, metadata) VALUES (?,?,?)")
            If $$$ISERR(sc) Return {"status":"error","message":($SYSTEM.Status.GetErrorText(sc))}.%ToJSON()
        }
        Set rs = st.%Execute(id, text, metadata)
    }
    If rs.%SQLCODE = 0 Return {"status":"ok","id":(id)}.%ToJSON()
    // -119 is "this row already exists", which can only be reached when the INSERT OR UPDATE
    // above failed to prepare and the plain INSERT ran in its place.
    If rs.%SQLCODE = -119 Return ..UpdateDocument(id, text, metadata, embedding, table)
    Return {"status":"error","message":(rs.%Message)}.%ToJSON()
}

/// The element type of a document table's embedding column, or the canonical type when it
/// cannot be resolved -- a table that does not exist yet, or a column that is not a vector.
ClassMethod VectorTypeOf(table As %String) As %String [ Private ]
{
    Set vecType = ##class(RAG.SDK.Search).GetVectorType(table, "embedding")
    If vecType = "" Return ..#VECTORTYPE
    Return vecType
}

/// Update the row an INSERT found already present. The embedding is left as it is when the
/// caller passed none, rather than overwritten with an empty vector.
ClassMethod UpdateDocument(id As %String, text As %String, metadata As %String, embedding As %String, table As %String) As %String [ Private ]
{
    Set sql = "UPDATE "_table_" SET text_content=?,metadata=?"
    If embedding '= "" {
        Set sql = sql_",embedding=TO_VECTOR(?,"_..VectorTypeOf(table)_","_$Length(embedding, ",")_")"
    }
    Set sql = sql_" WHERE doc_id=?"
    Set st = ##class(%SQL.Statement).%New()
    Set sc = st.%Prepare(sql)
    If $$$ISERR(sc) Return {"status":"error","message":($SYSTEM.Status.GetErrorText(sc))}.%ToJSON()
    If embedding '= "" {
        Set rs = st.%Execute(text, metadata, embedding, id)
    } Else {
        Set rs = st.%Execute(text, metadata, id)
    }
    If rs.%SQLCODE '= 0 Return {"status":"error","message":(rs.%Message)}.%ToJSON()
    Return {"status":"ok","id":(id)}.%ToJSON()
}

ClassMethod AddDocumentBatch(jsonArray As %String, table As %String = "RAG.SourceDocuments") As %String
{
    Set arr = [].%FromJSON(jsonArray)
    Set inserted = 0
    Set errors = 0
    Set n = arr.%Size()
    For i = 0:1:(n-1) {
        Set doc = arr.%Get(i)
        Set result = ..AddDocument(doc.id, doc.text, $Select(doc.metadata'="":doc.metadata, 1:"{}"), $Select(doc.embedding'="":doc.embedding, 1:""), table)
        Set obj = {}.%FromJSON(result)
        If obj.status = "ok" { Set inserted = inserted + 1 }
        Else { Set errors = errors + 1 }
    }
    Return {"inserted":(inserted),"errors":(errors)}.%ToJSON()
}

/// Embed the text with a sentence-transformers model, then add the document.
///
/// The embedding step itself is embedded Python and lives in RAG.SDK.Embed. Compiling a class
/// that carries a Python method fails outright on an instance whose embedded Python is not
/// usable -- ERROR #7802 on iris-vector-rag-iris, which takes the compiler's worker jobs down
/// with it -- and that must not stop this class, which holds the write path, from compiling.
ClassMethod AddDocumentWithEmbed(id As %String, text As %String, metadata As %String = "{}", model As %String = "all-MiniLM-L6-v2", table As %String = "RAG.SourceDocuments") As %String
{
    If '##class(%Dictionary.CompiledClass).%ExistsId("RAG.SDK.Embed") {
        Return {"status":"error","message":"RAG.SDK.Embed is not compiled on this instance, so no embedding can be computed here"}.%ToJSON()
    }
    Return $CLASSMETHOD("RAG.SDK.Embed", "AddDocumentWithEmbed", id, text, metadata, model, table)
}

}
