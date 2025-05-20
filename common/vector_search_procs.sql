-- Stored Procedure for testing basic SQL SP call via ODBC
-- Using LANGUAGE OBJECTSCRIPT as LANGUAGE SQL was problematic for returning result sets.
CREATE PROCEDURE RAG.SimpleEchoOS(
    InputVal VARCHAR(255) 
)
LANGUAGE OBJECTSCRIPT
-- No RESULT SETS 1, as it forces a complex query procedure structure.
{
    // Use %SQL.Statement for more explicit result set handling for ODBC
    Set tStatement = ##class(%SQL.Statement).%New()
    Set tSQL = "SELECT ? AS EchoedValue"
    
    // %Prepare can take %SQL.Statement.SQLCODE for error checking if needed
    Do tStatement.%Prepare(tSQL) 
    
    // Execute the statement. The result set from this %Execute
    // should be available to the ODBC caller.
    // InputVal is the parameter name from the CREATE PROCEDURE statement.
    Set rset = tStatement.%Execute(InputVal)
    
    // Attempt to explicitly Quit the %SQL.StatementResult object.
    // This is experimental to see if it helps ODBC recognize the result set.
    Quit rset 
}
-- Add other pure SQL or ObjectScript stored procedures below as needed.
