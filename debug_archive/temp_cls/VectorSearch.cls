Class RAG.VectorSearchUtils
{
ClassMethod SearchSourceDocuments(TopK As %Integer, VectorString As %String, VectorType As %String = "DOUBLE", VectorDimension As %Integer = 768) As %SQL.StatementResult [ SqlProc, SqlName = "VSU_SearchDocsV3" ]
{
//New stmt
//  New sc
Set stmt = ##class(%SQL.Statement).%New()
Set sc = stmt.%Prepare("SELECT ? AS EchoedVectorString")
If $$$ISERR(sc) {
Throw ##class(%Exception.SQL).%New("Diagnostic Prepare failed: "_$SYSTEM.Status.GetErrorText(sc))
}
Quit stmt.%Execute(VectorString)
}

