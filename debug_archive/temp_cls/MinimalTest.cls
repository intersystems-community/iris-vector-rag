Class RAG.MinimalTest
{
ClassMethod Echo() As %SQL.StatementResult [ SqlProc, SqlName="MinimalEcho" ]
{
    New stmt As %SQL.Statement
    Set stmt = ##class(%SQL.Statement).%New()
    // Prepare a SELECT statement that returns a hardcoded value
    Set sc = stmt.%Prepare("SELECT 12345 AS HardcodedNumericValue")
    If $$$ISERR(sc) {
        // Optionally, throw an error or handle it
        Throw ##class(%Exception.SQL).%New("Prepare failed in MinimalTest.Echo: "_$SYSTEM.Status.GetErrorText(sc))
    }
    // Execute the SELECT and Quit the statement result
    Quit stmt.%Execute()
}
}
