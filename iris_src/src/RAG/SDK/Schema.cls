Class RAG.SDK.Schema
{

/// The element type every vector column created here is declared with. IRIS refuses to
/// compare vectors of different element types (SQLCODE -259), so one type has to be
/// canonical: FLOAT, the type this repo's Python paths create. Declared once so a second
/// creation path cannot disagree with the first.
Parameter VECTORTYPE = "FLOAT";

/// Create the document, chunk and entity tables if they are not already there.
///
/// `pSchema` exists so an installation, or a test, can create the tables somewhere other
/// than RAG without a second copy of the DDL to keep in step with this one.
///
/// The first statement that fails stops the run and is named in the error: a half-created
/// schema reported as ok is worse than one reported as broken.
ClassMethod Initialize(pDim As %Integer = 384, pSchema As %String = "RAG") As %String
{
    Set tSC = $$$OK
    If ##class(RAG.SDK.Identifier).TopK(pDim) = 0 {
        Return ..ErrorJSON("embedding dimension must be a positive integer, got '"_pDim_"'")
    }
    Set tSC = ##class(RAG.SDK.Identifier).CheckName(pSchema, "schema")
    If $$$ISERR(tSC) Return ..ErrorJSON($SYSTEM.Status.GetErrorText(tSC))
    Set tStmts = ..GetDefaultDDL(pDim, pSchema)
    Set tPtr = 0
    While $ListNext(tStmts, tPtr, tDDL) {
        Set tStmt = ##class(%SQL.Statement).%New()
        Set tSC = tStmt.%Prepare(tDDL)
        If $$$ISERR(tSC) Return ..ErrorJSON($SYSTEM.Status.GetErrorText(tSC)_" in: "_tDDL)
        Set tRS = tStmt.%Execute()
        If tRS.%SQLCODE < 0 Return ..ErrorJSON(tRS.%Message_" in: "_tDDL)
    }
    Quit {"status":"ok","embeddingDimension":(pDim)}.%ToJSON()
}

/// The error shape every method here returns: status "error" and the reason.
ClassMethod ErrorJSON(pMessage As %String) As %String [ Private ]
{
    Set tSC = $$$OK
    Quit {"status":"error","message":(pMessage)}.%ToJSON()
}

/// The CREATE statements for the three tables, as a list -- one element per statement, so
/// nothing has to be split back out of a string.
ClassMethod GetDefaultDDL(pDim As %Integer, pSchema As %String = "RAG") As %List [ Private ]
{
    Set tSC = $$$OK
    Set tVec = "VECTOR("_..#VECTORTYPE_", "_pDim_")"
    Set tD1 = "CREATE TABLE IF NOT EXISTS "_pSchema_".SourceDocuments (doc_id VARCHAR(64) NOT NULL, title VARCHAR(500), text_content LONGVARCHAR, metadata LONGVARCHAR, embedding "_tVec_", PRIMARY KEY (doc_id))"
    Set tD2 = "CREATE TABLE IF NOT EXISTS "_pSchema_".DocumentChunks (chunk_id VARCHAR(64) NOT NULL, source_doc_id VARCHAR(64), chunk_text LONGVARCHAR, chunk_embedding "_tVec_", PRIMARY KEY (chunk_id))"
    Set tD3 = "CREATE TABLE IF NOT EXISTS "_pSchema_".Entities (entity_id VARCHAR(64) NOT NULL, entity_name VARCHAR(500), entity_type VARCHAR(100) COLLATE EXACT, source_doc_id VARCHAR(64), PRIMARY KEY (entity_id))"
    Quit $ListBuild(tD1, tD2, tD3)
}

/// The shell command Install runs. With no version it installs the latest release: a version
/// written into this class is wrong the day the package moves on, and it did, by
/// several releases. A caller who needs a particular release names it.
ClassMethod InstallCommand(pVersion As %String = "") As %String
{
    Set tSC = $$$OK
    Set tCmd = "pip install iris-vector-rag"
    If pVersion '= "" Set tCmd = tCmd_"=="_pVersion
    Quit tCmd
}

ClassMethod Install(pVersion As %String = "") As %String
{
    Set tSC = $$$OK
    // The version reaches a shell command line, so it has to look like a version.
    If (pVersion '= "") && '(pVersion ? 1.(1AN,1".",1"+",1"-")) {
        Return ..ErrorJSON("'"_pVersion_"' is not a package version")
    }
    Set tCmd = ..InstallCommand(pVersion)
    Set tRC = $ZF(-1, tCmd)
    If tRC = 0 {
        Return {"status":"ok","message":("Ran "_tCmd)}.%ToJSON()
    }
    Quit ..ErrorJSON("'"_tCmd_"' exited with code "_tRC)
}

/// Drop the three tables Initialize creates, in the schema it created them in. A table that
/// is not there is not an error; one that cannot be dropped is.
ClassMethod Drop(pSchema As %String = "RAG") As %String
{
    Set tSC = $$$OK
    Set tSC = ##class(RAG.SDK.Identifier).CheckName(pSchema, "schema")
    If $$$ISERR(tSC) Return ..ErrorJSON($SYSTEM.Status.GetErrorText(tSC))
    Set tTables = $ListBuild(pSchema_".Entities", pSchema_".DocumentChunks", pSchema_".SourceDocuments")
    Set tPtr = 0
    While $ListNext(tTables, tPtr, tTable) {
        Set tStmt = ##class(%SQL.Statement).%New()
        Set tSC = tStmt.%Prepare("DROP TABLE IF EXISTS "_tTable)
        If $$$ISERR(tSC) Return ..ErrorJSON($SYSTEM.Status.GetErrorText(tSC))
        Set tRS = tStmt.%Execute()
        If tRS.%SQLCODE < 0 Return ..ErrorJSON(tRS.%Message)
    }
    Quit {"status":"ok"}.%ToJSON()
}

/// The tables Initialize creates that are present in a schema, the embedding dimension the
/// document table declares (0 when there is no such table) and its row count.
ClassMethod Status(pSchema As %String = "RAG") As %String
{
    Set tSC = $$$OK
    Set tSC = ##class(RAG.SDK.Identifier).CheckName(pSchema, "schema")
    If $$$ISERR(tSC) Return ..ErrorJSON($SYSTEM.Status.GetErrorText(tSC))
    Set tTables = []
    Set tStmt = ##class(%SQL.Statement).%New()
    Set tSC = tStmt.%Prepare("SELECT TABLE_NAME FROM INFORMATION_SCHEMA.TABLES WHERE TABLE_SCHEMA=? AND TABLE_NAME IN ('SourceDocuments','DocumentChunks','Entities') ORDER BY TABLE_NAME")
    If $$$ISERR(tSC) Return ..ErrorJSON($SYSTEM.Status.GetErrorText(tSC))
    Set tRS = tStmt.%Execute(pSchema)
    If tRS.%SQLCODE < 0 Return ..ErrorJSON(tRS.%Message)
    While tRS.%Next() { Do tTables.%Push(pSchema_"."_tRS.%Get("TABLE_NAME")) }
    Set tCount = 0
    Set tDim = 0
    Set tDocs = pSchema_".SourceDocuments"
    If ##class(%SYSTEM.SQL).TableExists(tDocs) {
        Set tStmt2 = ##class(%SQL.Statement).%New()
        Set tSC = tStmt2.%Prepare("SELECT COUNT(*) AS n FROM "_tDocs)
        If $$$ISERR(tSC) Return ..ErrorJSON($SYSTEM.Status.GetErrorText(tSC))
        Set tRS2 = tStmt2.%Execute()
        If tRS2.%SQLCODE < 0 Return ..ErrorJSON(tRS2.%Message)
        If tRS2.%Next() Set tCount = tRS2.%Get("n")
        // From the class dictionary: INFORMATION_SCHEMA reports every VECTOR column as varchar
        // and gives no dimension (see RAG.SDK.Search.GetVectorDim).
        Set tDim = ##class(RAG.SDK.Search).GetVectorDim(tDocs, "embedding")
    }
    Quit {"tables":(tTables),"embeddingDimension":(tDim),"documentCount":(tCount)}.%ToJSON()
}

}
